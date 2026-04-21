from typing import List, Optional
import torch

from utils.wan_wrapper import WanDiffusionWrapper, WanTextEncoder, WanVAEWrapper

from utils.memory import gpu, get_cuda_free_memory_gb, move_model_to_device_with_memory_preservation

from functions.conjugate_gradient import CG

from utils.inpaint_util import inpaint_video
from einops import rearrange
import torch.nn.functional as F
import time

class CausalRestorationPipeline(torch.nn.Module):
    def __init__(
            self,
            args,
            device,
            initialization_step,
            sampling_step,
            generator=None,
            text_encoder=None,
            vae=None
    ):
        super().__init__()
        # Step 1: Initialize all models
        self.generator = WanDiffusionWrapper(
            **getattr(args, "model_kwargs", {}), is_causal=True) if generator is None else generator
        self.text_encoder = WanTextEncoder() if text_encoder is None else text_encoder
        self.vae = WanVAEWrapper() if vae is None else vae

        # Step 2: Initialize all causal hyperparmeters
        self.scheduler = self.generator.get_scheduler()

        init_step = initialization_step
        samp_step = sampling_step
        interval = init_step // samp_step
        generated_step_list = list(range(init_step, 0, -interval))
        self.denoising_step_list = torch.tensor(
            generated_step_list, dtype=torch.long)
        
        if args.warp_denoising_step:
            timesteps = torch.cat((self.scheduler.timesteps.cpu(), torch.tensor([0], dtype=torch.float32)))
            self.denoising_step_list = timesteps[1000 - self.denoising_step_list]

        self.num_transformer_blocks = 30
        self.frame_seq_length = 1560

        self.kv_cache1 = None
        self.args = args
        self.num_frame_per_block = getattr(args, "num_frame_per_block", 1)
        self.independent_first_frame = args.independent_first_frame
        self.local_attn_size = self.generator.model.local_attn_size

        print(f"KV inference with {self.num_frame_per_block} frames per block")

        if self.num_frame_per_block > 1:
            self.generator.model.num_frame_per_block = self.num_frame_per_block

    def inference(
        self,
        measurement,
        operator,
        task,
        noise: torch.Tensor,
        text_prompts: List[str],
        initial_latent: Optional[torch.Tensor] = None,
        return_latents: bool = False,
        low_memory: bool = False,
        initialization_index: int = 0,
    ) -> torch.Tensor:
        batch_size, num_frames, num_channels, height, width = noise.shape
        if not self.independent_first_frame or (self.independent_first_frame and initial_latent is not None):
            # If the first frame is independent and the first frame is provided, then the number of frames in the
            # noise should still be a multiple of num_frame_per_block
            assert num_frames % self.num_frame_per_block == 0
            num_blocks = num_frames // self.num_frame_per_block
        else:
            # Using a [1, 4, 4, 4, 4, 4, ...] model to generate a video without image conditioning
            assert (num_frames - 1) % self.num_frame_per_block == 0
            num_blocks = (num_frames - 1) // self.num_frame_per_block
        num_input_frames = initial_latent.shape[1] if initial_latent is not None else 0
        num_output_frames = num_frames + num_input_frames  # add the initial latent frames
        conditional_dict = self.text_encoder(
            text_prompts=text_prompts
        )

        if low_memory:
            gpu_memory_preservation = get_cuda_free_memory_gb(gpu) + 5
            move_model_to_device_with_memory_preservation(self.text_encoder, target_device=gpu, preserved_memory_gb=gpu_memory_preservation)

        output = torch.zeros(
            [batch_size, num_output_frames, num_channels, height, width],
            device=noise.device,
            dtype=noise.dtype
        )

        # Initialize KV cache to all zeros
        if self.kv_cache1 is None:
            self._initialize_kv_cache(
                batch_size=batch_size,
                dtype=noise.dtype,
                device=noise.device
            )
            self._initialize_crossattn_cache(
                batch_size=batch_size,
                dtype=noise.dtype,
                device=noise.device
            )
        else:
            # reset cross attn cache
            for block_index in range(self.num_transformer_blocks):
                self.crossattn_cache[block_index]["is_init"] = False
            # reset kv cache
            for block_index in range(len(self.kv_cache1)):
                self.kv_cache1[block_index]["global_end_index"] = torch.tensor(
                    [0], dtype=torch.long, device=noise.device)
                self.kv_cache1[block_index]["local_end_index"] = torch.tensor(
                    [0], dtype=torch.long, device=noise.device)

        start_time = time.time()
        # Step 1: Measurement-consistent Initialization
        if task == 'random_inpainting':
            measurement_init = inpaint_video(
                video=measurement,
                mask=operator.mask,
                radius=3,
                method="telea"
            )
        elif task in ['super_resolution', 'spatio_temporal_avg']:
            b, t = measurement.shape[0], measurement.shape[1]
            measurement_up = rearrange(measurement, 'b t c h w -> (b t) c h w').float()
            measurement_up = F.interpolate(measurement_up, scale_factor=4, mode='bilinear')
            measurement_up = rearrange(measurement_up, '(b t) c h w -> b t c h w', b=b, t=t).to(measurement.dtype)
            
            if task == 'super_resolution':
                measurement_init = self.CG(
                    x0_hat=measurement_up,
                    y_obs=measurement, 
                    operator=operator,
                    cg_steps=5,
                )
            elif task == 'spatio_temporal_avg':
                measurement_init = self.CG(
                    x0_hat=measurement_up,
                    y_obs=measurement, 
                    operator=operator,
                    cg_steps=100,
                )
        elif task== 'deblur_gauss':
            measurement_init = self.CG(
                x0_hat=measurement,
                y_obs=measurement,
                operator=operator,
                cg_steps=5,
            )
        elif task == 'temporal_avg':
            measurement_init = self.CG(
                x0_hat=measurement,
                y_obs=measurement,
                operator=operator,
                cg_steps=50,
            )

        measurement_latent = self.vae.encode_to_latent(measurement_init.permute(0, 2, 1, 3, 4)).to(device=noise.device, dtype=noise.dtype)
        measurement_latent = measurement_latent.repeat(batch_size, 1, 1, 1, 1)

        # Step 2: Temporal denoising loop
        current_start_frame = 0
        offset = 0
        all_num_frames = [self.num_frame_per_block] * num_blocks
        if self.independent_first_frame and initial_latent is None:
            all_num_frames = [1] + all_num_frames
        initial_token = True
        for current_num_frames in all_num_frames:
            noisy_input = noise[
                :, current_start_frame - num_input_frames:current_start_frame + current_num_frames - num_input_frames]
            measurement_latent_input = measurement_latent[
                :, current_start_frame:current_start_frame + current_num_frames]
            
            if current_start_frame == 0:
                frame_len = 9
            else:
                frame_len = 12
            
            measurement_video_input = measurement[:, offset : offset + frame_len]

            # Step 2.1: Spatial denoising loop
            for index, current_timestep in enumerate(self.denoising_step_list):
                print(f"current_timestep: {current_timestep}")
                timestep = torch.ones(
                    [batch_size, current_num_frames],
                    device=noise.device,
                    dtype=torch.int64) * current_timestep                

                if index < initialization_index:
                    pass

                elif index >= initialization_index:
                    if index == initialization_index:
                        # 2.1.1. Renoise to start time t0
                        denoised_pred = measurement_latent_input
                        noisy_input = self.scheduler.add_noise(
                            denoised_pred.flatten(0, 1),
                            torch.randn_like(denoised_pred.flatten(0, 1)),
                            timestep
                        ).unflatten(0, denoised_pred.shape[:2])

                    # 2.1.2. Denoising step
                    _, denoised_pred = self.generator(
                        noisy_image_or_video=noisy_input,
                        conditional_dict=conditional_dict,
                        timestep=timestep,
                        kv_cache=self.kv_cache1,
                        crossattn_cache=self.crossattn_cache,
                        current_start=current_start_frame * self.frame_seq_length
                    )

                    # 2.1.3. Proximal Update (Data Consistency)
                    if initial_token:
                        x0_hat = self.vae.decode_to_pixel(denoised_pred, use_cache=False).to(denoised_pred.dtype)
                        updated_x0_hat = self.CG(
                            x0_hat=x0_hat,
                            y_obs=measurement_video_input, 
                            operator=operator,
                            proximal=1.0,
                        )
                        denoised_pred = self.vae.encode_to_latent(updated_x0_hat.permute(0, 2, 1, 3, 4)).to(updated_x0_hat.dtype)
                    else:
                        pass
                    
                    # 2.1.4. Re-noise to nest timestep
                    is_last_index = (index == len(self.denoising_step_list) - 1)
                    if not is_last_index:
                        next_timestep = torch.ones(
                            [batch_size, current_num_frames],
                            device=noise.device,
                            dtype=torch.int64) * self.denoising_step_list[index + 1]
                        
                        noisy_input = self.scheduler.add_noise(
                            denoised_pred.flatten(0, 1),
                            torch.randn_like(denoised_pred.flatten(0, 1)),
                            next_timestep
                        ).unflatten(0, denoised_pred.shape[:2])
                    else:
                        pass

            initial_token = False
                    
            # Step 2.2: update the model's output
            output[:, current_start_frame:current_start_frame + current_num_frames] = denoised_pred
            
            # Step 2.3: rerun with timestep zero to update KV cache using clean context
            context_timestep = torch.ones(
                        [batch_size, current_num_frames],
                        device=noise.device,
                        dtype=torch.int64) * self.args.context_noise
            self.generator(
                noisy_image_or_video=denoised_pred,
                conditional_dict=conditional_dict,
                timestep=context_timestep,
                kv_cache=self.kv_cache1,
                crossattn_cache=self.crossattn_cache,
                current_start=current_start_frame * self.frame_seq_length,
            )
            
            # Step 2.4: update the start and end frame indices
            current_start_frame += current_num_frames
            offset += frame_len

        # Step 3: Decode the output
        video = self.vae.decode_to_pixel(output, use_cache=False)
        video = (video * 0.5 + 0.5).clamp(0, 1)

        restoration_time = time.time() - start_time
        print(f"  - Restoration time: {restoration_time:.2f} s")

        if return_latents:
            return video, output
        else:
            return video

    @torch.no_grad()
    def CG(
        self,
        x0_hat: torch.Tensor,
        y_obs: torch.Tensor,
        operator,
        cg_steps: int = 5,
        proximal: float = 0.0
    ) -> torch.Tensor:
        
        orig_dtype = x0_hat.dtype
        
        x0_hat = x0_hat.to(torch.float32)
        y_obs = y_obs.to(torch.float32)
        
        def H_fn(x):
            Ax = operator.At(operator.A(x))
            return Ax + x * proximal
            
        b = operator.At(y_obs) + x0_hat * proximal
        
        x_opt = CG(H_fn, b, x0_hat, m=cg_steps)
        
        x_opt = torch.clamp(x_opt, -1.0, 1.0).to(orig_dtype)
        
        return x_opt
    
    def _initialize_kv_cache(self, batch_size, dtype, device):
        """
        Initialize a Per-GPU KV cache for the Wan model.
        """
        kv_cache1 = []
        if self.local_attn_size != -1:
            # Use the local attention size to compute the KV cache size
            kv_cache_size = self.local_attn_size * self.frame_seq_length
        else:
            # Use the default KV cache size
            kv_cache_size = 32760

        for _ in range(self.num_transformer_blocks):
            kv_cache1.append({
                "k": torch.zeros([batch_size, kv_cache_size, 12, 128], dtype=dtype, device=device),
                "v": torch.zeros([batch_size, kv_cache_size, 12, 128], dtype=dtype, device=device),
                "global_end_index": torch.tensor([0], dtype=torch.long, device=device),
                "local_end_index": torch.tensor([0], dtype=torch.long, device=device)
            })

        self.kv_cache1 = kv_cache1  # always store the clean cache

    def _initialize_crossattn_cache(self, batch_size, dtype, device):
        """
        Initialize a Per-GPU cross-attention cache for the Wan model.
        """
        crossattn_cache = []

        for _ in range(self.num_transformer_blocks):
            crossattn_cache.append({
                "k": torch.zeros([batch_size, 512, 12, 128], dtype=dtype, device=device),
                "v": torch.zeros([batch_size, 512, 12, 128], dtype=dtype, device=device),
                "is_init": False
            })
        self.crossattn_cache = crossattn_cache