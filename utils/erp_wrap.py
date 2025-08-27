# utils/erp_wrap.py
from typing import Tuple
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def _same_pad_1d(L: int, k: int, s: int, d: int) -> Tuple[int, int]:

    """
    计算 SAME padding 的 (left, right)，兼容 stride/dilation。
    P = max((ceil(L/s) - 1)*s + d*(k-1) + 1 - L, 0)
    """
    k_eff = d * (k - 1) + 1
    out = math.ceil(L / s)
    total = max((out - 1) * s + k_eff - L, 0)
    left = total // 2
    right = total - left
    return left, right


def _pad_lr(x: torch.Tensor, left: int, right: int) -> torch.Tensor:
    """Only horizontal circular pad (left/right)."""
    if left == 0 and right == 0:
        return x
    # pad = (left, right, top, bottom)
    return F.pad(x, (left, right, 0, 0), mode="circular")


class HorizontalCircularConv2d(nn.Conv2d):
    """
    Conv2d whose padding is circular ONLY along width (ERP horizontal seam),
    while vertical padding keeps original zero-padding semantics.
    """

    @classmethod
    def from_conv(cls, conv: nn.Conv2d) -> "HorizontalCircularConv2d":
        # 构造一个“零 padding”的同规格卷积，参数复制自原卷积
        k = conv.kernel_size if isinstance(conv.kernel_size, tuple) else (conv.kernel_size, conv.kernel_size)
        new = cls(
            in_channels=conv.in_channels,
            out_channels=conv.out_channels,
            kernel_size=k,
            stride=conv.stride,
            padding=0,  # 我们手动做 padding
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

        # 记录原层的 padding 语义（可能是 int/tuple 或 "same"/"valid"）
        new._orig_padding = conv.padding
        # 记录其余超参
        new._kernel_size = (k[0], k[1])
        new._stride = conv.stride if isinstance(conv.stride, tuple) else (conv.stride, conv.stride)
        new._dilation = conv.dilation if isinstance(conv.dilation, tuple) else (conv.dilation, conv.dilation)

        # 若为数值 padding，预先存静态 pad；字符串模式在 forward 动态计算
        if isinstance(new._orig_padding, str):
            new._static_v_pad = None  # (top, bottom)
            new._static_h_pad = None  # (left, right)
        else:
            pad = new._orig_padding if isinstance(new._orig_padding, tuple) else (new._orig_padding, new._orig_padding)
            # Conv2d 的 tuple padding 语义为 (pad_h, pad_w)（对称），非 (top,bottom,left,right)
            v = int(pad[0])
            h = int(pad[1])
            new._static_v_pad = (v, v)
            new._static_h_pad = (h, h)

        return new

    def _compute_pads(self, x: torch.Tensor) -> Tuple[int, int, int, int]:
        """
        返回 (top, bottom, left, right)，与原层 padding 语义一致；
        仅在后续应用为：垂直方向零填充 / 水平方向 circular。
        """
        H = x.size(-2)
        W = x.size(-1)
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
            # 静态对称 padding：保持与原层一致
            t, b = self._static_v_pad
            l, r = self._static_h_pad

        return t, b, l, r

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 计算与原层一致的 pad（仅水平使用 circular；垂直为零填充）
        t, b, l, r = self._compute_pads(x)

        # 垂直方向：零填充（非环绕）
        if t or b:
            x = F.pad(x, (0, 0, t, b), mode="constant", value=0.0)

        # 水平方向：环绕填充
        if l or r:
            x = F.pad(x, (l, r, 0, 0), mode="circular")

        # conv 内 padding 置 0，其余超参沿用原层
        return F.conv2d(
            x, self.weight, self.bias,
            stride=self._stride, padding=(0, 0),
            dilation=self._dilation, groups=self.groups
        )


def enable_horizontal_circular_padding(module: nn.Module, only_3x3: bool = True) -> int:
    """
    递归替换 module 中的 Conv2d -> HorizontalCircularConv2d。
    only_3x3=True 时仅替换 3x3 卷积；False 则所有卷积。
    返回替换的层数，便于日志确认。
    """
    replaced = 0
    for name, child in list(module.named_children()):
        replaced += enable_horizontal_circular_padding(child, only_3x3)
        if isinstance(child, nn.Conv2d):
            k = child.kernel_size if isinstance(child.kernel_size, tuple) else (child.kernel_size, child.kernel_size)
            if (not only_3x3) or (k == (3, 3)):
                setattr(module, name, HorizontalCircularConv2d.from_conv(child))
                replaced += 1
    return replaced
