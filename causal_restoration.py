import argparse
import torch
import os
from omegaconf import OmegaConf
from tqdm import tqdm
from torchvision import transforms
from torchvision.io import write_video
from einops import rearrange
import torch.distributed as dist
from torch.utils.data import DataLoader, SequentialSampler
from torch.utils.data.distributed import DistributedSampler

from pipeline import (
    CausalRestorationPipeline
)
from utils.dataset import VideoDataset
from utils.misc import set_seed

from utils.memory import gpu, get_cuda_free_memory_gb, DynamicSwapInstaller

from munch import munchify
from functions.degradation import get_degradation, wrap_operator_video

import torch.nn.functional as F

parser = argparse.ArgumentParser()
parser.add_argument('--task_list', type=str, nargs='+', default=['deblur_gauss'],
    choices=[
        'deblur_gauss',
        'super_resolution',
        'random_inpainting',
        'temporal_avg',
        'spatio_temporal_avg',
    ],
    help='List of degradation task types to run')
parser.add_argument('--H', type=int, default=480)
parser.add_argument('--W', type=int, default=832)
parser.add_argument('--initialization_step', type=int, default=100, help='Step for initialization')
parser.add_argument('--sampling_step', type=int, default=2, help='Step for sampling')
parser.add_argument("--config_path", type=str, help="Path to the config file")
parser.add_argument("--checkpoint_path", type=str, help="Path to the checkpoint folder")
parser.add_argument("--data_path", type=str, help="Path to the dataset")
parser.add_argument("--extended_prompt_path", type=str, help="Path to the extended prompt")
parser.add_argument("--output_folder", type=str, help="Output folder")
parser.add_argument("--num_output_frames", type=int, default=21,
                    help="Number of overlap frames between sliding windows")
parser.add_argument("--restoration", action="store_true", help="Whether to perform restoration (or T2V by default)")
parser.add_argument("--use_ema", action="store_true", help="Whether to use EMA parameters")
parser.add_argument("--seed", type=int, default=0, help="Random seed")
parser.add_argument("--num_samples", type=int, default=1, help="Number of samples to generate per prompt")
parser.add_argument("--save_with_index", action="store_true",
                    help="Whether to save the video using the index or prompt as the filename")
args = parser.parse_args()

# # Initialize distributed inference
# if "LOCAL_RANK" in os.environ:
#     dist.init_process_group(backend='nccl')
#     local_rank = int(os.environ["LOCAL_RANK"])
#     torch.cuda.set_device(local_rank)
#     device = torch.device(f"cuda:{local_rank}")
#     world_size = dist.get_world_size()
#     set_seed(args.seed + local_rank)
# else:
device = torch.device("cuda")
local_rank = 0
world_size = 1
set_seed(args.seed)

print(f'Free VRAM {get_cuda_free_memory_gb(gpu)} GB')
low_memory = get_cuda_free_memory_gb(gpu) < 40

torch.set_grad_enabled(False)

config = OmegaConf.load(args.config_path)
default_config = OmegaConf.load("configs/default_config.yaml")
config = OmegaConf.merge(default_config, config)

# load pipeline
pipeline = CausalRestorationPipeline(config, device=device, initialization_step=args.initialization_step, sampling_step=args.sampling_step)

if args.checkpoint_path:
    state_dict = torch.load(args.checkpoint_path, map_location="cpu")
    pipeline.generator.load_state_dict(state_dict['generator' if not args.use_ema else 'generator_ema'])

pipeline = pipeline.to(dtype=torch.bfloat16)
if low_memory:
    DynamicSwapInstaller.install_model(pipeline.text_encoder, device=gpu)
else:
    pipeline.text_encoder.to(device=gpu)
pipeline.generator.to(device=gpu)
pipeline.vae.to(device=gpu)


# Create dataset
transform = transforms.Compose([
    transforms.Resize((args.H, args.W)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])
dataset = VideoDataset(args.data_path, transform=transform)

num_prompts = len(dataset)
print(f"Number of prompts: {num_prompts}")

if dist.is_initialized():
    sampler = DistributedSampler(dataset, shuffle=False, drop_last=True)
else:
    sampler = SequentialSampler(dataset)
dataloader = DataLoader(dataset, batch_size=1, sampler=sampler, num_workers=0, drop_last=False)

# Create output directory (only on main process to avoid race conditions)
if local_rank == 0:
    os.makedirs(args.output_folder, exist_ok=True)

if dist.is_initialized():
    dist.barrier()

for task in args.task_list:
    if task == 'deblur_gauss':
        deg_scale = 61
    elif task in ['super_resolution', 'spatio_temporal_avg']:
        deg_scale = 4
    elif task == 'random_inpainting':
        deg_scale = 0.5
    elif task == 'temporal_avg':
        deg_scale = 7
    else:
        raise NotImplementedError(f'Task {task} not implemented!')

    deg_config = munchify({
        'channels': 3,
        'H': args.H,
        'W': args.W,
        'deg_scale': deg_scale
    })
    
    operator = get_degradation(task, deg_config, device)
    if task not in ['temporal_avg', 'spatio_temporal_avg']:
        operator = wrap_operator_video(operator)

    for i, batch_data in tqdm(enumerate(dataloader), disable=(local_rank != 0)):
        idx = batch_data['idx'].item()

        # For DataLoader batch_size=1, the batch_data is already a single item, but in a batch container
        # Unpack the batch data for convenience
        if isinstance(batch_data, dict):
            batch = batch_data
        elif isinstance(batch_data, list):
            batch = batch_data[0]  # First (and only) item in the batch

        all_video = []
        num_generated_frames = 0  # Number of generated (latent) frames

        if args.restoration:
            prompt = batch['prompts'][0]  # Get caption from batch
            prompts = [prompt] * args.num_samples
            gt = batch['video'].to(device=device, dtype=torch.bfloat16)
            y = operator.A(gt)
            sampled_noise = torch.randn(
                [args.num_samples, args.num_output_frames, 16, 60, 104], device=device, dtype=torch.bfloat16
            )
        
        # Generate frames
        video = pipeline.inference(
            measurement=y,
            operator=operator,
            task=task,
            noise=sampled_noise,
            text_prompts=prompts,
            return_latents=False,
            low_memory=low_memory
        )

        # Resize for visualization
        current_video = rearrange(video, 'b t c h w -> b t h w c').cpu()
        gt = rearrange(gt, 'b t c h w -> b t h w c').cpu()
        if task in ['super_resolution', 'spatio_temporal_avg']:
            b, t = y.shape[0], y.shape[1]
            y = rearrange(y, 'b t c h w -> (b t) c h w').float()
            y = F.interpolate(y, scale_factor=deg_scale, mode='nearest')
            y = rearrange(y, '(b t) c h w -> b t c h w', b=b, t=t).to(y.dtype)
        y = rearrange(y, 'b t c h w -> b t h w c').cpu()
        all_video.append(current_video)

        # Final output video
        video = 255.0 * torch.cat(all_video, dim=1)

        # Clear VAE cache
        pipeline.vae.model.clear_cache()

        # Save the video
        if idx < num_prompts:
            for seed_idx in range(args.num_samples):
                base_name = f'{idx}'
                
                param_folder_name = f'init-{args.initialization_step}_samp-{args.sampling_step}'
                task_folder = os.path.join(args.output_folder, param_folder_name, task)
                os.makedirs(task_folder, exist_ok=True)

                output_path = os.path.join(task_folder, f'{base_name}_recon.mp4')
                gt_path     = os.path.join(task_folder, f'{base_name}_label.mp4')
                meas_path   = os.path.join(task_folder, f'{base_name}_input.mp4')

                write_video(output_path, video[seed_idx], fps=16)
                if gt is not None:
                    write_video(gt_path, (gt[seed_idx] + 1.0) * 127.5, fps=16)
                if y is not None:
                    write_video(meas_path, (y[seed_idx] + 1.0) * 127.5, fps=16)