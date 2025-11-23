# -*- coding: utf-8 -*-
# dataset/batch_transform.py
import math
import random
from typing import Iterable, Any
import torch

__all__ = [
    "BatchTransform",
    "IdentityBatchTransform",
    "RollHBatchTransform",
    "TailExtendBatchTransform",
    "RollThenExtendBatchTransform",
]

# ====================== 工具函数 ======================

def _find_wdim(t: torch.Tensor) -> int:
    """
    识别宽度维（支持 BCHW / BHWC / CHW / HWC）。
    规则：若通道维在最后（1或3），则宽在倒数第二；否则宽在最后。
    """
    if t.dim() == 4:
        # BCHW / BHWC
        if t.shape[-1] in (1, 3):  # BHWC
            return -2
        elif t.shape[1] in (1, 3):  # BCHW
            return -1
        else:
            return -1
    elif t.dim() == 3:
        # CHW / HWC
        if t.shape[-1] in (1, 3):  # HWC
            return -2
        elif t.shape[0] in (1, 3):  # CHW
            return -1
        else:
            return -1
    else:
        # 不支持更低维度
        return -1


def _find_hdim(t: torch.Tensor) -> int:
    """与 _find_wdim 配套，返回高的维度索引。"""
    if t.dim() == 4:
        if t.shape[-1] in (1, 3):   # BHWC
            return -3
        elif t.shape[1] in (1, 3):  # BCHW
            return -2
        else:
            return -2
    elif t.dim() == 3:
        if t.shape[-1] in (1, 3):   # HWC
            return -3
        elif t.shape[0] in (1, 3):  # CHW
            return -2
        else:
            return -2
    else:
        return -2


def _as_tensor(x: Any) -> torch.Tensor:
    """尽量保持原 tensor，不是 tensor 则转为 tensor（设备保持原样，通常为 CPU）"""
    return x if isinstance(x, torch.Tensor) else torch.as_tensor(x)


def _index_select_w(t: torch.Tensor, idx: torch.Tensor, wdim: int) -> torch.Tensor:
    """
    沿宽度维（wdim）用 idx 索引，兼容负维下标。
    """
    # 将负维转为正维便于 index_select
    wdim_pos = wdim if wdim >= 0 else (t.dim() + wdim)
    return torch.index_select(t, dim=wdim_pos, index=idx)


def _cat_right(t: torch.Tensor, k: int, wdim: int) -> torch.Tensor:
    """
    在宽度维 wdim 上右侧拼接前 k 列（环向/循环取样）。
    非就地，安全可微。
    """
    W = int(t.shape[wdim])
    if k <= 0 or W <= 0:
        return t
    # 取 [:k] 的环向索引，允许 k > W
    idx = (torch.arange(k, device=t.device, dtype=torch.long) % max(1, W))
    tail = _index_select_w(t, idx, wdim)
    return torch.cat([t, tail], dim=wdim)


def _roll_w(t: torch.Tensor, shift: int, wdim: int) -> torch.Tensor:
    """
    沿宽度维 wdim 环向平移 shift（像素）。
    """
    if shift == 0:
        return t
    return torch.roll(t, shifts=shift, dims=wdim)

# ====================== 基类与空操作 ======================

class BatchTransform:
    """批级变换基类：接收/返回 batch(dict)。"""
    def __call__(self, batch: Any) -> Any:
        return batch

class IdentityBatchTransform(BatchTransform):
    """占位：什么都不做。"""
    pass

# ====================== 水平环向 roll ======================

class RollHBatchTransform(BatchTransform):
    """
    沿水平方向做环向平移增强（roll），并把像素位移写入 batch['hshift_pix']（正右负左）。
    - max_frac: 最大位移为 W 的该比例（最终取整）
    - enabled: False 则不 roll，但会保证写出 hshift_pix=0
    - keys: 要 roll 的键（默认只处理 'jpg'）
    """
    def __init__(self, max_frac: float = 0.2, enabled: bool = True, keys: Iterable[str] = ("jpg",)):
        self.max_frac = float(max_frac)
        self.enabled = bool(enabled)
        self.keys = tuple(keys)

    def __call__(self, batch: Any) -> Any:
        if not isinstance(batch, dict):
            return batch

        if "jpg" not in batch:
            # 即使没有图像，也补一个 hshift_pix，便于下游统一
            batch.setdefault("hshift_pix", 0)
            return batch

        ref = _as_tensor(batch["jpg"])
        wdim = _find_wdim(ref)
        W0 = int(ref.shape[wdim])

        # 计算位移：[-max_shift, max_shift]
        if self.enabled and self.max_frac > 0 and W0 > 0:
            max_shift = max(1, int(round(W0 * self.max_frac)))
            shift = random.randint(-max_shift, max_shift)
        else:
            shift = 0

        # 对指定 keys 滚动
        for k in self.keys:
            if k in batch and isinstance(batch[k], torch.Tensor):
                t = batch[k]
                if t.dim() >= 3 and int(t.shape[wdim]) == W0:
                    batch[k] = _roll_w(t, shift, wdim=wdim)

        # 记录像素位移
        batch["hshift_pix"] = int(shift)
        return batch

# ====================== 右侧尾巴拓宽（环向拼接） ======================

class TailExtendBatchTransform(BatchTransform):
    """
    给 batch 中的图像在右侧“拓宽” K 个像素（把左端 [:K] 复制到右端拼接），并记录：
    - batch['orig_size'] = LongTensor([H0, W0])：原图尺寸
    - batch['extend_k_pix'] = LongTensor([K])：此次拓宽的像素带宽
    规则：
    - 若 align <= 0 或 extend_frac <= 0 ⇒ 不拓宽（K=0）
    - 否则 K = ceil( max(1, ceil(W0 * extend_frac)) / align ) * align
    - 拼接来源：原图左端 [:K]（环向/循环，支持 K > W0）
    - 记录原图尺寸到 batch['orig_size'] = LongTensor([H0, W0])
    - 若上游未设置 hshift_pix，则补为 0（便于 val/test 统一）
    """
    def __init__(self, extend_frac: float = 0.1, align: int = 64, keys: Iterable[str] = ("jpg",)):
        self.extend_frac = float(extend_frac)
        self.align = int(align)  # 允许 <=0 表示“不对齐”
        self.keys = tuple(keys)

    def __call__(self, batch: Any) -> Any:
        if not isinstance(batch, dict):
            return batch
        if "jpg" not in batch:
            # 没有参照键：也补个 hshift_pix，避免下游出错
            batch.setdefault("hshift_pix", 0)
            return batch

        ref = _as_tensor(batch["jpg"])
        wdim = _find_wdim(ref)
        hdim = _find_hdim(ref)
        W0 = int(ref.shape[wdim])
        H0 = int(ref.shape[hdim])

        # --- 计算拓宽像素 K（严格遵循：align <= 0 或 extend_frac <= 0 ⇒ 不拓宽） ---
        if self.align is None:
            self.align = 0

        if (self.align <= 0) or (self.extend_frac <= 0) or (W0 <= 0):
            K = 0
        else:
            k0 = int(math.ceil(W0 * float(self.extend_frac)))
            K = int(math.ceil(max(1, k0) / float(self.align)) * self.align)

        # 对指定 keys 依次拓宽（右侧拼接左端 [:K]）
        if K > 0:
            for k in self.keys:
                if k in batch and isinstance(batch[k], torch.Tensor):
                    t = batch[k]
                    # 只对宽度与参照一致的张量做拼接
                    if t.dim() >= 3 and int(t.shape[wdim]) == W0:
                        batch[k] = _cat_right(t, K, wdim=wdim)

        # 记录原图尺寸 & 实际 K
        device = ref.device
        batch["orig_size"] = torch.tensor([H0, W0], dtype=torch.long, device=device)
        batch.setdefault("hshift_pix", 0)
        batch["extend_k_pix"] = torch.tensor(int(K), dtype=torch.long, device=device)
        return batch

# ====================== 先 roll 再拓宽（训练常用） ======================

class RollThenExtendBatchTransform(BatchTransform):
    """
    训练期推荐：先水平 roll（环向随机位移），再右侧拓宽尾巴。
    注意：roll 之后，拓宽仍以 roll 后的“当前左端”作为 [:K] 来源，这与“解码后仅左带写回”的设计相容。
    """
    def __init__(
        self,
        roll_max_frac: float = 0.0,
        extend_frac: float = 0.0,
        align: int = 0,
        keys: Iterable[str] = ("jpg",),
    ):
        self.roller = RollHBatchTransform(max_frac=roll_max_frac, enabled=True, keys=keys)
        self.extender = TailExtendBatchTransform(extend_frac=extend_frac, align=align, keys=keys)

    def __call__(self, batch: Any) -> Any:
        batch = self.roller(batch)     # 写入 hshift_pix
        batch = self.extender(batch)   # 写入 orig_size / extend_k_pix，并把 jpg 等右侧拓宽
        return batch
