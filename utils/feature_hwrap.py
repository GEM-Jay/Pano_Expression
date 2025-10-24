# utils/feature_hwrap.py
# 作用：仅当 UNet 的 ResBlock 前两层 3×3 卷积计算到“需要水平 padding”的位置时，
#      用对侧像素进行水平环绕补齐；其余位置严格保持与原实现一致（垂直仍为零）。
# 特点：不改权重/形状/分辨率；仅影响边界带；中心区域数值做一次 allclose 自检，异常即回退。

import os
import torch
import torch.nn.functional as F
from torch import nn

try:
    from ldm.modules.diffusionmodules.openaimodel import ResBlock  # LDM 的 ResBlock
except Exception:
    ResBlock = None

_HWRAP_DEBUG = os.getenv("HWRAP_DEBUG", "0") == "1"
_checked_once = False  # 仅首次运行做自检


def _conv_hwrap_edges_generic(conv: nn.Conv2d, x: torch.Tensor) -> torch.Tensor:
    """
    仅覆盖“水平边界带”，带宽 p_w = conv.padding[1]；中心区域严格等价原实现。
    要求：conv 为 3x3, stride=1, dilation=1（由调用方保证）。
    """
    # 原始零填充输出（绕过外层 patch，避免递归）
    out_zero = conv.__class__.forward(conv, x)  # [B, C_out, H, W]
    B, _, H, W = out_zero.shape

    # 读取 padding
    pad = conv.padding if isinstance(conv.padding, tuple) else (conv.padding, conv.padding)
    p_h, p_w = int(pad[0]), int(pad[1])

    # 无水平 padding 或宽度过小，直接返回
    if p_w <= 0 or W <= 2 * p_w:
        return out_zero

    # 构造“水平环绕 + 垂直零”的输入 → 用 padding=0 做等价卷积
    Hpad, Wpad = H + 2 * p_h, W + 2 * p_w
    x_pad = x.new_zeros((x.shape[0], x.shape[1], Hpad, Wpad))
    x_pad[:, :, p_h:p_h + H, p_w:p_w + W] = x
    # 水平环绕（仅中间有效行）
    x_pad[:, :, p_h:p_h + H, :p_w] = x[:, :, :, W - p_w:]   # 左补最右
    x_pad[:, :, p_h:p_h + H, -p_w:] = x[:, :, :, :p_w]      # 右补最左
    # 垂直仍为零填充（极区不环绕）

    out_circ = F.conv2d(
        x_pad, conv.weight, conv.bias,
        stride=conv.stride, padding=0,
        dilation=conv.dilation, groups=conv.groups
    )

    # 中心区域自检：同时裁掉上下和左右 padding，确保中心块与零填充结果完全一致
    global _checked_once
    if not _checked_once:
        same_shape = (out_circ.shape == out_zero.shape)
        if (H - 2 * p_h) > 0 and (W - 2 * p_w) > 0:
            center_equal = torch.allclose(
                out_circ[..., p_h:H - p_h, p_w:W - p_w],
                out_zero[..., p_h:H - p_h, p_w:W - p_w],
                rtol=1e-5, atol=1e-6
            )
        else:
            center_equal = True  # 无中心块可比，视为通过
        if _HWRAP_DEBUG:
            print(f"[feature_hwrap] shape_ok={same_shape}, center_equal={center_equal}, p_w={p_w}, p_h={p_h}")
        if (not same_shape) or (not center_equal):
            return out_zero  # 任何异常直接回退，防止副作用
        _checked_once = True

    # 仅覆盖水平边界带
    out = out_zero.clone()
    out[..., :p_w] = out_circ[..., :p_w]
    out[..., -p_w:] = out_circ[..., -p_w:]
    return out


def _get_resblock_convs(rb: nn.Module):
    """返回 LDM ResBlock 的两层 3×3 Conv2d：in_layers[-1], out_layers[-1]（若不存在返回 None）。"""
    in_conv = out_conv = None
    try:
        if isinstance(getattr(rb, "in_layers", None), nn.Sequential) and isinstance(rb.in_layers[-1], nn.Conv2d):
            in_conv = rb.in_layers[-1]
    except Exception:
        pass
    try:
        if isinstance(getattr(rb, "out_layers", None), nn.Sequential) and isinstance(rb.out_layers[-1], nn.Conv2d):
            out_conv = rb.out_layers[-1]
    except Exception:
        pass
    return in_conv, out_conv


def _wrap_resblock_forward(rb: nn.Module):
    """只给 ResBlock 的两层 3×3/stride=1/dilation=1 卷积加“边界限定水平环绕”补丁。"""
    if getattr(rb, "_hwrap_patched", False):
        return

    orig_forward = rb.forward
    in_conv, out_conv = _get_resblock_convs(rb)

    def _wrap_fwd(conv: nn.Conv2d):
        def _f(xx):
            if (
                isinstance(conv, nn.Conv2d)
                and conv.kernel_size == (3, 3)
                and conv.stride == (1, 1)
                and conv.dilation == (1, 1)
            ):
                return _conv_hwrap_edges_generic(conv, xx)
            # 其他卷积绕过补丁
            return conv.__class__.forward(conv, xx)
        return _f

    def patched_forward(x, emb):
        # 临时替换 forward，调用结束恢复
        in_bak = getattr(in_conv, "forward", None) if in_conv is not None else None
        out_bak = getattr(out_conv, "forward", None) if out_conv is not None else None
        try:
            if in_conv is not None:
                in_conv.forward = _wrap_fwd(in_conv)
            if out_conv is not None:
                out_conv.forward = _wrap_fwd(out_conv)
            return orig_forward(x, emb)
        finally:
            if in_conv is not None and in_bak is not None:
                in_conv.forward = in_bak
            if out_conv is not None and out_bak is not None:
                out_conv.forward = out_bak

    rb._hwrap_forward_orig = orig_forward
    rb.forward = patched_forward
    rb._hwrap_patched = True


def enable_feature_hwrap_for_unet(unet_root: nn.Module, enabled: bool = True):
    """遍历 UNet，启用所有 ResBlock 的边界限定水平环绕（仅前两层 3×3 生效）。"""
    if not enabled or unet_root is None or ResBlock is None:
        return
    for _, m in unet_root.named_modules():
        if isinstance(m, ResBlock):
            _wrap_resblock_forward(m)


def disable_feature_hwrap_for_unet(unet_root: nn.Module):
    """恢复原行为（如需禁用）。"""
    if unet_root is None or ResBlock is None:
        return
    for _, m in unet_root.named_modules():
        if isinstance(m, ResBlock) and getattr(m, "_hwrap_patched", False):
            try:
                m.forward = m._hwrap_forward_orig
            except Exception:
                pass
            m._hwrap_patched = False
            if hasattr(m, "_hwrap_forward_orig"):
                delattr(m, "_hwrap_forward_orig")
