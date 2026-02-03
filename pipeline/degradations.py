import torch
import torch.nn.functional as F
from einops import rearrange

def PatchUpsample(x, scale, mode="nearest"):
    """
    x: (B, T, C, H, W)
    """
    assert x.dim() == 5, f"Expected 5D (B,T,C,H,W), got {x.shape}"
    B, T, C, H, W = x.shape

    # (B,T,C,H,W) -> (B*T,C,H,W)
    x = rearrange(x, 'b t c h w -> (b t) c h w')

    # 2D upsample per frame
    x = F.interpolate(
        x,
        scale_factor=scale,
        mode=mode,
        align_corners=False if mode in ["bilinear", "bicubic"] else None
    )

    # (B*T,C,H',W') -> (B,T,C,H',W')
    x = rearrange(x, '(b t) c h w -> b t c h w', b=B, t=T)
    return x


def PatchDownsample(x, scale=4):
    """
    x: (B, T, C, H, W)
    """
    assert x.dim() == 5, f"Expected 5D (B,T,C,H,W), got {x.shape}"
    B, T, C, H, W = x.shape
    out_h, out_w = H // scale, W // scale

    # (B,T,C,H,W) -> (B*T,C,H,W)
    x = rearrange(x, 'b t c h w -> (b t) c h w')

    # 2D pool per frame
    x = F.adaptive_avg_pool2d(x, (out_h, out_w))

    # (B*T,C,h,w) -> (B,T,C,h,w)
    x = rearrange(x, '(b t) c h w -> b t c h w', b=B, t=T)
    return x

def gaussian_kernel_2d(kernel_size: int, sigma: float, dtype=torch.float32):
    """Generate a 2D Gaussian kernel."""
    ax = torch.arange(kernel_size, dtype=dtype) - (kernel_size - 1) / 2
    xx, yy = torch.meshgrid(ax, ax, indexing="ij")
    kernel = torch.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    kernel = kernel / kernel.sum()
    return kernel

def gaussian_blur(tensor: torch.Tensor, kernel_size: int, sigma: float):
    assert tensor.dim() == 5, f"Expected 5D (B,T,C,H,W), got {tensor.shape}"
    device = tensor.device
    dtype = tensor.dtype
    B, T, C, H, W = tensor.shape

    # (B,T,C,H,W) -> (B*T, C, H, W)
    x = rearrange(tensor, 'b t c h w -> (b t) c h w').to(dtype=torch.float32)

    # build kernel
    kernel = gaussian_kernel_2d(kernel_size, sigma).to(device=device, dtype=torch.float32)
    kernel = kernel.view(1, 1, kernel_size, kernel_size).expand(C, 1, kernel_size, kernel_size)

    padding = kernel_size // 2
    x = F.pad(x, (padding, padding, padding, padding), mode='reflect')
    x = F.conv2d(x, kernel, padding=0, groups=C)

    # (B*T, C, H, W) -> (B,T,C,H,W)
    x = rearrange(x, '(b t) c h w -> b t c h w', b=B, t=T).to(dtype=dtype)

    return x

def generate_random_mask(shape, pixel_ratio):
    """
    Generates a random binary mask with the given pixel ratio.

    Args:
        shape (tuple): Shape of the mask (B, C, H, W).
        pixel_ratio (float): Ratio of pixels to be set to 1.

    Returns:
        torch.Tensor: Random binary mask.
    """
    B, C, H, W = shape
    num_pixels = H * W
    num_ones = int(num_pixels * pixel_ratio)
    
    # Generate a flat array with the appropriate ratio of ones and zeros
    flat_mask = torch.zeros(num_pixels, dtype=torch.float32)
    flat_mask[:num_ones] = 1
    
    # Shuffle to randomize the positions of ones and zeros
    flat_mask = flat_mask[torch.randperm(num_pixels)]
    
    # Reshape to the original spatial dimensions and duplicate across channels
    mask = flat_mask.view(1, H, W)
    mask = mask.expand(B, C, H, W)
    
    return mask