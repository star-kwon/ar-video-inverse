import torch
from munch import Munch

from functions import measurements
from einops import rearrange

__DEGRADATION__ = {}

def register_degradation(name: str):
    def wrapper(fn):
        if __DEGRADATION__.get(name) is not None:
            raise NameError(f'DEGRADATION {name} is already registered')
        __DEGRADATION__[name]=fn
        return fn
    return wrapper

def get_degradation(name: str,
                    deg_config: Munch,
                    device:torch.device):
    if __DEGRADATION__.get(name) is None:
        raise NameError(f'DEGRADATION {name} does not exist.')
    return __DEGRADATION__[name](deg_config, device)

@register_degradation(name='random_inpainting')
def deg_inpainting(deg_config, device):
    A_funcs = measurements.RandomInpaintingOperator(deg_config.channels,
                                            deg_config.H,
                                            deg_config.W,
                                            ratio=deg_config.deg_scale,
                                            device=device)
    return A_funcs

@register_degradation(name='super_resolution')
def deg_sr_general(deg_config, device):
    blur_by = int(deg_config.deg_scale)
    A_funcs = measurements.SuperResolutionOperator(
                                            blur_by,
                                            device)
    return A_funcs


@register_degradation(name='deblur_gauss')
def deg_deblur_guass_general(deg_config, device):
    A_funcs = measurements.GaussialBlurOperator(
        kernel_size=deg_config.deg_scale,
        intensity=3.0,
        device=device
    )
    return A_funcs

@register_degradation(name='temporal_avg')
def deg_temporal_avg(deg_config, device):
    A_funcs = measurements.TemporalAvgOperator(
        kernel_size=deg_config.deg_scale,
        device=device
    )
    return A_funcs

@register_degradation(name='spatio_temporal_avg')
def deg_spatio_temporal_avg(deg_config, device):
    A_funcs = measurements.SpatioTemporalAvgOperator(
        scale_factor=deg_config.deg_scale,
        kernel_size=deg_config.deg_scale,
        device=device
    )
    return A_funcs

def wrap_operator_video(operator):
    def _wrap_fn(fn, fn_name: str):
        if fn is None:
            return None

        def wrapped(x, *args, **kwargs):
            if x.dim() != 5:
                raise ValueError(f"{fn_name} expects 5D video (B,T,C,H,W), got shape={tuple(x.shape)}")

            orig_dtype = x.dtype
            orig_device = x.device

            b, t, c, h, w = x.shape

            # (B,T,C,H,W) -> (B*T,C,H,W), run in float32 for stability
            x4 = rearrange(x, 'b t c h w -> (b t) c h w').to(dtype=torch.float32)

            y4 = fn(x4, *args, **kwargs)

            # Typical operators preserve 4D shape (B*T, C', H', W')
            if y4.dim() != 4 or y4.shape[0] != b * t:
                raise RuntimeError(
                    f"{fn_name} returned unexpected shape {tuple(y4.shape)} for input {tuple(x.shape)}"
                )

            # back to original dtype/device + unflatten
            y4 = y4.to(device=orig_device, dtype=orig_dtype)
            y5 = rearrange(y4, '(b t) c h w -> b t c h w', b=b, t=t)
            return y5

        return wrapped

    # Save originals
    A_orig = getattr(operator, "A", None)
    At_orig = getattr(operator, "At", None)

    # Wrap & replace
    if callable(A_orig):
        operator.A = _wrap_fn(A_orig, "operator.A")
    else:
        raise AttributeError("operator.A is missing or not callable")

    if callable(At_orig):
        operator.At = _wrap_fn(At_orig, "operator.At")
    else:
        raise AttributeError("operator.At is missing or not callable")

    return operator