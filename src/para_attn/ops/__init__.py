from typing import Optional, Tuple

import torch
import torch.backends.cuda as cuda_backend
from torch.backends.cuda import SDPAParams

aten = torch.ops.aten


def can_use_attention_backend(*args, **kwargs):
    return True


def cannot_use_attention_backend(*args, **kwargs):
    return False


can_use_flash_attention = getattr(cuda_backend, "can_use_flash_attention", cannot_use_attention_backend)
can_use_efficient_attention = getattr(cuda_backend, "can_use_efficient_attention", cannot_use_attention_backend)
can_use_cudnn_attention = getattr(cuda_backend, "can_use_cudnn_attention", cannot_use_attention_backend)

# torch.compile() support is only enabled for pytorch >= 2.4
# The reason for this is that we are using the new custom_op and register_fake
# APIs, which support inplace modification of inputs in the function itself
if torch.__version__ >= "2.4.0":
    _torch_custom_op_wrapper = torch.library.custom_op
    _torch_register_fake_wrapper = torch.library.register_fake
else:
    raise RuntimeError("torch.library.custom_op requires PyTorch version >= 2.4.0")

    # def noop_custom_op_wrapper(name, fn=None, /, *, mutates_args, device_types=None, schema=None):
    #     def wrap(func):
    #         return func
    #     if fn is None:
    #         return wrap
    #     return fn

    # def noop_register_fake_wrapper(op, fn=None, /, *, lib=None, _stacklevel=1):
    #     def wrap(func):
    #         return func
    #     if fn is None:
    #         return wrap
    #     return fn

    # _torch_custom_op_wrapper = noop_custom_op_wrapper
    # _torch_register_fake_wrapper = noop_register_fake_wrapper


def flash_attention_forward_with_lse(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_mask: Optional[torch.Tensor] = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    *,
    scale: Optional[float] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    return aten._scaled_dot_product_flash_attention(
        query, key, value, dropout_p=dropout_p, is_causal=is_causal, scale=scale
    )[:2]


def efficient_attention_forward_with_lse(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_mask: Optional[torch.Tensor] = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    *,
    scale: Optional[float] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    return aten._scaled_dot_product_efficient_attention(
        query,
        key,
        value,
        attn_bias=attn_mask,
        compute_log_sumexp=True,
        dropout_p=dropout_p,
        is_causal=is_causal,
        scale=scale,
    )[:2]


def cudnn_attention_forward_with_lse(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_mask: Optional[torch.Tensor] = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    *,
    scale: Optional[float] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    return aten._scaled_dot_product_cudnn_attention(
        query,
        key,
        value,
        attn_bias=attn_mask,
        compute_log_sumexp=True,
        dropout_p=dropout_p,
        is_causal=is_causal,
        scale=scale,
    )[:2]


@_torch_custom_op_wrapper("para_attn::attention_forward_with_lse", mutates_args=(), device_types="cuda")
def attention_forward_with_lse(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_mask: Optional[torch.Tensor] = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    *,
    scale: Optional[float] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    device = query.device
    assert device.type == "cuda", "dispatch_fastest_attention_impl only supports CUDA tensors"
    assert torch.version.hip is None, "dispatch_fastest_attention_impl does not support HIP devices"
    params_init_args = [
        query,
        key,
        value,
        attn_mask,
        dropout_p,
        is_causal,
    ]
    if hasattr(SDPAParams, "enable_gqa"):
        params_init_args.append(False)
    params = SDPAParams(*params_init_args)
    dispatches = [
        ["flash", can_use_flash_attention, flash_attention_forward_with_lse],
        ["efficient", can_use_efficient_attention, efficient_attention_forward_with_lse],
        ["cudnn", can_use_cudnn_attention, cudnn_attention_forward_with_lse],
    ]
    device_capability = torch.cuda.get_device_capability(device.index)
    if device_capability >= (9, 0):
        priority_order = ["cudnn", "flash", "efficient"]
    else:
        priority_order = ["flash", "cudnn", "efficient"]
    dispatches = sorted(dispatches, key=lambda x: priority_order.index(x[0]))
    for dispatch in dispatches:
        if dispatch[1](params):
            return dispatch[2](
                query, key, value, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal, scale=scale
            )
    raise NotImplementedError("No available implementation for dispatch_fastest_attention_impl")


@_torch_register_fake_wrapper("para_attn::attention_forward_with_lse")
def _(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_mask: Optional[torch.Tensor] = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    *,
    scale: Optional[float] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    return attention_forward_with_lse(
        query,
        key,
        value,
        attn_mask=attn_mask,
        dropout_p=dropout_p,
        is_causal=is_causal,
        scale=scale,
    )