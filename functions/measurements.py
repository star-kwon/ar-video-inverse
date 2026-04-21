'''This module handles task-dependent operations (A) and noises (n) to simulate a measurement y=Ax+n.'''

from abc import ABC, abstractmethod
from functools import partial

from torch.nn import functional as F
from torchvision import torch

from utils.blur_util import Blurkernel
from utils.inpaint_util import generate_random_mask

from einops import rearrange

# =================
# Operation classes
# =================

__OPERATOR__ = {}
_GAMMA_FACTOR = 2.2

def register_operator(name: str):
    def wrapper(cls):
        if __OPERATOR__.get(name, None):
            raise NameError(f"Name {name} is already registered!")
        __OPERATOR__[name] = cls
        return cls
    return wrapper


def get_operator(name: str, **kwargs):
    if __OPERATOR__.get(name, None) is None:
        raise NameError(f"Name {name} is not defined.")
    return __OPERATOR__[name](**kwargs)


class LinearOperator(ABC):
    @abstractmethod
    def forward(self, data, **kwargs):
        # calculate A * X
        pass

    @abstractmethod
    def noisy_forward(self, data, **kwargs):
        # calculate A * X + n
        pass

    @abstractmethod
    def transpose(self, data, **kwargs):
        # calculate A^T * X
        pass

    def ortho_project(self, data, **kwargs):
        # calculate (I - A^T * A)X
        return data - self.transpose(self.forward(data, **kwargs), **kwargs)

    def project(self, data, measurement, **kwargs):
        # calculate (I - A^T * A)Y - AX
        return self.ortho_project(measurement, **kwargs) - self.forward(data, **kwargs)


@register_operator(name='noise')
class DenoiseOperator(LinearOperator):
    def __init__(self, device):
        self.device = device

    def forward(self, data):
        return data

    def noisy_forward(self, data):
        return data

    def transpose(self, data):
        return data

    def ortho_project(self, data):
        return data

    def project(self, data):
        return data


@register_operator(name='super_resolution')
class SuperResolutionOperator(LinearOperator):
    def __init__(self,
                 scale_factor,
                 device):
        self.device = device
        self.scale_factor = scale_factor
        self.down_sample = lambda x: F.interpolate(x, scale_factor=1/scale_factor, mode='area')
        self.up_sample = lambda x: F.interpolate(x, scale_factor=scale_factor, mode='bilinear')

    def A(self, data, **kwargs):
        return self.forward(data, **kwargs)

    def forward(self, data, **kwargs):
        return self.down_sample(data)

    def noisy_forward(self, data, **kwargs):
        pass

    def transpose(self, data, **kwargs):
        return self.up_sample(data)

    def project(self, data, measurement, **kwargs):
        return data - self.transpose(self.forward(data)) + self.transpose(measurement)
    
    def A(self, data):
        return self.forward(data)

    def At(self, data):
        return self.transpose(data)

@register_operator(name='deblur_gauss')
class GaussialBlurOperator(LinearOperator):
    def __init__(self,
                 kernel_size,
                 intensity,
                 device):
        self.device = device
        self.kernel_size = kernel_size
        self.conv = Blurkernel(blur_type='gaussian',
                               kernel_size=kernel_size,
                               std=intensity,
                               device=device).to(device)
        self.kernel = self.conv.get_kernel()
        self.conv.update_weights(self.kernel.type(torch.float32))

    def forward(self, data, **kwargs):
        return self.conv(data)

    def noisy_forward(self, data, **kwargs):
        pass

    def transpose(self, data, **kwargs):
        return data

    def get_kernel(self):
        return self.kernel.view(1, 1, self.kernel_size, self.kernel_size)

    def apply_kernel(self, data, kernel):
        self.conv.update_weights(kernel.type(torch.float32))
        return self.conv(data)

    def A(self, data):
        return self.forward(data)

    def At(self, data):
        return self.transpose(data)

@register_operator(name='random_inpainting')
class RandomInpaintingOperator(LinearOperator):
    def __init__(self,
                 C,
                 H,
                 W,
                 ratio,
                 device):
        self.device = device
        # self.mask = generate_box_mask(shape=(1, C, H, W), box_size=int(size)).to(device)
        self.mask = generate_random_mask(shape=(1, C, H, W), pixel_ratio=ratio).to(device)

    def forward(self, data, **kwargs):
        data = data * self.mask
        return data

    def noisy_forward(self, data, **kwargs):
        pass

    def transpose(self, data, **kwargs):
        data = data * self.mask
        return data

    def A(self, data):
        return self.forward(data)

    def At(self, data):
        return self.transpose(data)

# =============
# Noise classes
# =============


__NOISE__ = {}

def register_noise(name: str):
    def wrapper(cls):
        if __NOISE__.get(name, None):
            raise NameError(f"Name {name} is already defined!")
        __NOISE__[name] = cls
        return cls
    return wrapper

def get_noise(name: str, **kwargs):
    if __NOISE__.get(name, None) is None:
        raise NameError(f"Name {name} is not defined.")
    noiser = __NOISE__[name](**kwargs)
    noiser.__name__ = name
    return noiser

class Noise(ABC):
    def __call__(self, data):
        return self.forward(data)

    @abstractmethod
    def forward(self, data):
        pass

@register_noise(name='clean')
class Clean(Noise):
    def __init__(self, **kwargs):
        pass

    def forward(self, data):
        return data

@register_noise(name='gaussian')
class GaussianNoise(Noise):
    def __init__(self, scale):
        self.scale = scale

    def forward(self, data):
        return data + torch.randn_like(data, device=data.device) * self.scale


@register_noise(name='poisson')
class PoissonNoise(Noise):
    def __init__(self, scale):
        self.scale = scale

    def forward(self, data):
        '''
        Follow skimage.util.random_noise.
        '''

        # version 3 (stack-overflow)
        import numpy as np
        data = (data + 1.0) / 2.0
        data = data.clamp(0, 1)
        device = data.device
        data = data.detach().cpu()
        data = torch.from_numpy(np.random.poisson(data * 255.0 * self.scale) / 255.0 / self.scale)
        data = data * 2.0 - 1.0
        data = data.clamp(-1, 1)
        return data.to(device)


@register_operator(name="temporal_avg")
class TemporalAvgOperator(LinearOperator):
    def __init__(self, kernel_size: int, device):
        self.device = device
        self.n = int(kernel_size)

    def forward(self, data, **kwargs):
        return causal_avg_forward(data, self.n)

    def transpose(self, data, **kwargs):
        return causal_avg_transpose(data, self.n)

    def noisy_forward(self, data, **kwargs):
        raise NotImplementedError

    def A(self, data):
        return self.forward(data)

    def At(self, data):
        return self.transpose(data)

@register_operator(name="spatio_temporal_avg")
class SpatioTemporalAvgOperator(LinearOperator):
    def __init__(self, scale_factor: float = 4.0, kernel_size: int = 4, device=None):
        self.device = device
        self.n = int(kernel_size)
        self.scale_factor = float(scale_factor)

    def forward(self, data, **kwargs):
        if self.scale_factor > 1.0:
            b, t = data.shape[0], data.shape[1]
            data_4d = rearrange(data, 'b t c h w -> (b t) c h w').float()
            data_4d_down = F.interpolate(data_4d, scale_factor=1.0/self.scale_factor, mode='area')
            data = rearrange(data_4d_down, '(b t) c h w -> b t c h w', b=b, t=t).to(data.dtype)

        return causal_avg_forward(data, self.n)

    def transpose(self, data, **kwargs):
        data = causal_avg_transpose(data, self.n)

        if self.scale_factor > 1.0:
            b, t = data.shape[0], data.shape[1]
            data_4d = rearrange(data, 'b t c h w -> (b t) c h w').float()
            data_4d_up = F.interpolate(data_4d, scale_factor=self.scale_factor, mode='bilinear')
            data = rearrange(data_4d_up, '(b t) c h w -> b t c h w', b=b, t=t).to(data.dtype)

        return data

    def noisy_forward(self, data, **kwargs):
        raise NotImplementedError

    def A(self, data):
        return self.forward(data)

    def At(self, data):
        return self.transpose(data)

def causal_avg_forward(x: torch.Tensor, n: int) -> torch.Tensor:
    """
    x: (B, T, C, H, W)
    Forward causal average.
    """
    x_fp32 = x.float()
    T = x.shape[1]

    cs = x_fp32.cumsum(dim=1)
    
    cs_shifted = torch.cat([torch.zeros_like(cs[:, :n]), cs[:, :-n]], dim=1)
    window_sum = cs - cs_shifted

    denom = torch.arange(1, T + 1, device=x.device).clamp_max(n).view(1, T, 1, 1, 1)
    
    return (window_sum / denom).to(dtype=x.dtype)

def causal_avg_transpose(y: torch.Tensor, n: int) -> torch.Tensor:
    """
    y: (B, T, C, H, W)
    Adjoint of causal average.
    """
    y_fp32 = y.float()
    T = y.shape[1]

    L = torch.arange(1, T + 1, device=y.device).clamp_max(n).view(1, T, 1, 1, 1)
    z = y_fp32 / L
    
    cs = z.cumsum(dim=1)

    e = (torch.arange(T, device=y.device) + n - 1).clamp_max(T - 1)
    cs_end = cs.index_select(dim=1, index=e)

    cs_before = torch.cat([torch.zeros_like(cs[:, :1]), cs[:, :-1]], dim=1)

    return (cs_end - cs_before).to(dtype=y.dtype)