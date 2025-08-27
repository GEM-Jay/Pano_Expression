# utils/feature_hwrap.py
from typing import Tuple
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

def _same_pad_1d(L: int, k: int, s: int, d: int) -> Tuple[int, int]:
    k_eff = d * (k - 1) + 1
    out = math.ceil(L / s)
    total = max((out - 1) * s + k_eff - L, 0)
    left = total // 2
    right = total - left
    return left, right

class HFeaturePad(nn.Module):
    """
    特征图级“前置环绕”：
      - 垂直：reflect / replicate / constant（按原 conv 的 padding 需求或 SAME/VALID 语义）
      - 水平：circular（按原 conv 的 padding 需求）
    注意：这里不做“额外比原需求更多的列”，以保持各层尺寸与原网络一致。
    """
    def __init__(self, conv: nn.Conv2d, v_mode: str = "constant", v_value: float = 0.0):
        super().__init__()
        self.v_mode = v_mode.lower()
        self.v_value = float(v_value)
        self._kernel_size = conv.kernel_size if isinstance(conv.kernel_size, tuple) else (conv.kernel_size, conv.kernel_size)
        self._stride = conv.stride if isinstance(conv.stride, tuple) else (conv.stride, conv.stride)
        self._dilation = conv.dilation if isinstance(conv.dilation, tuple) else (conv.dilation, conv.dilation)
        self._orig_padding = conv.padding  # int/tuple or "same"/"valid"

    def _needed_pad(self, x):
        H = x.size(-2); W = x.size(-1)
        kh, kw = self._kernel_size
        sh, sw = self._stride
        dh, dw = self._dilation
        if isinstance(self._orig_padding, str):
            pad_str = self._orig_padding.lower()
            if pad_str == "same":
                t, b = _same_pad_1d(H, kh, sh, dh)
                l, r = _same_pad_1d(W, kw, sw, dw)
            elif pad_str == "valid":
                t = b = l = r = 0
            else:
                raise ValueError(f"Unsupported padding string: {self._orig_padding}")
        else:
            pad = self._orig_padding if isinstance(self._orig_padding, tuple) else (self._orig_padding, self._orig_padding)
            t = b = int(pad[0])  # 对称
            l = r = int(pad[1])
        return t, b, l, r

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        t, b, l, r = self._needed_pad(x)
        # 垂直 pad
        if t or b:
            if self.v_mode == "constant":
                x = F.pad(x, (0, 0, t, b), mode="constant", value=self.v_value)
            elif self.v_mode in ("reflect", "replicate"):
                x = F.pad(x, (0, 0, t, b), mode=self.v_mode)
            else:
                raise ValueError(f"Unsupported vertical pad mode: {self.v_mode}")
        # 水平 circular
        if l or r:
            x = F.pad(x, (l, r, 0, 0), mode="circular")
        return x

def _clone_conv_no_pad(conv: nn.Conv2d) -> nn.Conv2d:
    k = conv.kernel_size if isinstance(conv.kernel_size, tuple) else (conv.kernel_size, conv.kernel_size)
    new = nn.Conv2d(
        in_channels=conv.in_channels,
        out_channels=conv.out_channels,
        kernel_size=k,
        stride=conv.stride,
        padding=0,  # 关键：移除内置 padding，由 HFeaturePad 负责边界
        dilation=conv.dilation,
        groups=conv.groups,
        bias=(conv.bias is not None),
        padding_mode="zeros",
        device=conv.weight.device,
        dtype=conv.weight.dtype,
    )
    with torch.no_grad():
        new.weight.copy_(conv.weight)
        if conv.bias is not None:
            new.bias.copy_(conv.bias)
    return new

def enable_feature_hwrap(model: nn.Module,
                         only_3x3: bool = True,
                         v_mode: str = "constant",
                         v_value: float = 0.0) -> int:
    """
    遍历 model，将 Conv2d 替换为 (HFeaturePad -> 原 Conv, padding=0) 的顺序层。
    默认仅替换 3x3（与常见 UNet 一致）。返回替换层数。
    """
    replaced = 0
    for name, child in list(model.named_children()):
        replaced += enable_feature_hwrap(child, only_3x3, v_mode, v_value)
        if isinstance(child, nn.Conv2d):
            k = child.kernel_size if isinstance(child.kernel_size, tuple) else (child.kernel_size, child.kernel_size)
            if (not only_3x3) or (k == (3, 3)):
                seq = nn.Sequential(
                    HFeaturePad(child, v_mode=v_mode, v_value=v_value),
                    _clone_conv_no_pad(child)
                )
                setattr(model, name, seq)
                replaced += 1
    return replaced
