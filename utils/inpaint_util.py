import numpy as np
import torch
import cv2

def random_sq_bbox(img, mask_shape, image_size=256, margin=(16, 16)):
    """Generate a random sqaure mask for inpainting
    """
    B, C, H, W = img.shape
    h, w = mask_shape
    margin_height, margin_width = margin
    maxt = image_size - margin_height - h
    maxl = image_size - margin_width - w

    # bb
    t = np.random.randint(margin_height, maxt)
    l = np.random.randint(margin_width, maxl)

    # make mask
    mask = torch.ones([B, C, H, W], device=img.device)
    mask[..., t:t+h, l:l+w] = 0

    return mask, t, t+h, l, l+w


class MaskGenerator:
    def __init__(self, mask_type, mask_len_range=None, mask_prob_range=None,
                 image_size=256, margin=(16, 16)):
        """
        (mask_len_range): given in (min, max) tuple.
        Specifies the range of box size in each dimension
        (mask_prob_range): for the case of random masking,
        specify the probability of individual pixels being masked
        """
        assert mask_type in ['box', 'random', 'both', 'extreme']
        self.mask_type = mask_type
        self.mask_len_range = mask_len_range
        self.mask_prob_range = mask_prob_range
        self.image_size = image_size
        self.margin = margin

    def _retrieve_box(self, img):
        l, h = self.mask_len_range
        l, h = int(l), int(h)
        mask_h = np.random.randint(l, h)
        mask_w = np.random.randint(l, h)
        mask, t, tl, w, wh = random_sq_bbox(img,
                              mask_shape=(mask_h, mask_w),
                              image_size=self.image_size,
                              margin=self.margin)
        return mask, t, tl, w, wh

    def _retrieve_random(self, img):
        total = self.image_size ** 2
        # random pixel sampling
        l, h = self.mask_prob_range
        prob = np.random.uniform(l, h)
        mask_vec = torch.ones([1, self.image_size * self.image_size])
        samples = np.random.choice(self.image_size * self.image_size, int(total * prob), replace=False)
        mask_vec[:, samples] = 0
        mask_b = mask_vec.view(1, self.image_size, self.image_size)
        mask_b = mask_b.repeat(3, 1, 1)
        mask = torch.ones_like(img, device=img.device)
        mask[:, ...] = mask_b
        return mask

    def __call__(self, img):
        if self.mask_type == 'random':
            mask = self._retrieve_random(img)
            return mask
        elif self.mask_type == 'box':
            mask, t, th, w, wl = self._retrieve_box(img)
            return mask
        elif self.mask_type == 'extreme':
            mask, t, th, w, wl = self._retrieve_box(img)
            mask = 1. - mask
            return mask

def generate_random_mask(shape, pixel_ratio, device=None):
    B, C, H, W = shape
    if device is None:
        device = torch.device("cpu")

    assert 0.0 <= pixel_ratio <= 1.0, f"pixel_ratio must be in [0,1], got {pixel_ratio}"

    num_pixels = H * W
    num_zeros = int(num_pixels * pixel_ratio)

    # 1D base mask (H*W,)
    flat_mask = torch.ones(num_pixels, device=device, dtype=torch.float32)
    if num_zeros > 0:
        flat_mask[:num_zeros] = 0.0

        # shuffle spatial positions
        perm = torch.randperm(num_pixels, device=device)
        flat_mask = flat_mask[perm]

    # (H,W)
    spatial_mask = flat_mask.view(H, W)

    # (B,C,H,W) - broadcast same mask over batch & channel
    mask = spatial_mask.unsqueeze(0).unsqueeze(0).expand(B, C, H, W)

    return mask

def generate_box_mask(shape, box_size, device=None):
    B, C, H, W = shape
    if device is None:
        device = torch.device("cpu")

    # parse box size
    if isinstance(box_size, int):
        box_h = box_w = box_size
    else:
        assert len(box_size) == 2, "box_size must be int or (box_h, box_w)"
        box_h, box_w = box_size

    # clamp box size to image size
    box_h = min(box_h, H)
    box_w = min(box_w, W)

    # center coordinates
    center_h = H // 2
    center_w = W // 2

    # top-left corner (centered)
    t = center_h - box_h // 2
    l = center_w - box_w // 2

    # bottom-right
    b = t + box_h
    r = l + box_w

    # base spatial mask: 1 = keep, 0 = masked
    spatial_mask = torch.ones((H, W), device=device, dtype=torch.float32)
    spatial_mask[t:b, l:r] = 0.0

    # (B, C, H, W) broadcast
    mask = spatial_mask.unsqueeze(0).unsqueeze(0).expand(B, C, H, W)

    return mask

def inpaint_video(
    video: torch.Tensor,   # (B, T, C, H, W), float, range [-1, 1]
    mask: torch.Tensor,    # (B, C, H, W), {0,1} where 0 = hole, 1 = keep
    radius: int = 3,
    method: str = "telea", # "telea" or "ns"
) -> torch.Tensor:
    """
    Frame-wise OpenCV inpainting on a masked video.

    OpenCV inpaint expects:
      - image: uint8 (H,W,3) BGR
      - mask:  uint8 (H,W) where NON-ZERO pixels are inpaint region

    Here:
      - video is (B,T,C,H,W) in [-1,1]
      - mask is (B,C,H,W) in {0,1}, where 0 indicates region to fill.

    Returns:
      - inpainted video tensor (B,T,C,H,W) in [-1,1], same dtype/device as input.
    """
    assert video.dim() == 5, f"video must be (B,T,C,H,W), got {tuple(video.shape)}"
    assert mask.dim() == 4, f"mask must be (B,C,H,W), got {tuple(mask.shape)}"
    B, T, C, H, W = video.shape
    assert C == 3, "OpenCV inpaint supports 3-channel images best (C=3 expected)."
    assert mask.shape[0] == B and mask.shape[2] == H and mask.shape[3] == W, "mask shape mismatch"

    orig_device = video.device
    orig_dtype = video.dtype

    # Prepare output on CPU float32 (OpenCV runs on CPU)
    out = torch.empty((B, T, C, H, W), dtype=torch.float32, device="cpu")

    # Choose inpaint method
    inpaint_flag = cv2.INPAINT_TELEA if method.lower() == "telea" else cv2.INPAINT_NS

    # Convert mask for OpenCV: non-zero = inpaint region
    # Our mask: 0 = hole -> need to inpaint -> convert to uint8 255 where hole
    # Use only one channel (any channel is fine since mask is duplicated across C usually)
    mask_cpu = mask.detach().to("cpu")
    if mask_cpu.dtype != torch.uint8:
        # mask in {0,1} float/int -> uint8
        mask_cpu_u8 = mask_cpu.to(torch.uint8)
    else:
        mask_cpu_u8 = mask_cpu

    # mask_hole: 255 where mask == 0, else 0
    # shape: (B, H, W)
    mask_hole = ((mask_cpu_u8[:, 0] == 0).to(torch.uint8) * 255).numpy()

    # Video to CPU
    video_cpu = video.detach().to("cpu", dtype=torch.float32)

    for b in range(B):
        m = mask_hole[b]  # (H,W), uint8
        for t in range(T):
            # (C,H,W) in [-1,1] -> (H,W,C) uint8 [0,255]
            frame = video_cpu[b, t]  # (3,H,W)
            frame_u8 = ((frame.permute(1, 2, 0).numpy() + 1.0) * 127.5)
            frame_u8 = np.clip(frame_u8, 0, 255).astype(np.uint8)

            # Inpaint
            filled = cv2.inpaint(frame_u8, m, radius, inpaint_flag)

            # uint8 -> float32 [-1,1], (H,W,C)->(C,H,W)
            filled = (filled.astype(np.float32) / 127.5) - 1.0
            out[b, t] = torch.from_numpy(filled).permute(2, 0, 1)

    # Return to original device/dtype
    return out.to(device=orig_device, dtype=orig_dtype)