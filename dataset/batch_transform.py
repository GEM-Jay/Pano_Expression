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
        raise ValueError("Unsupported tensor rank for horizontal dimension detection")


def _find_hdim(t: torch.Tensor) -> int:
    """识别高度维（配合 _find_wdim 使用）"""
    if t.dim() == 4:
        if t.shape[-1] in (1, 3):  # BHWC
            return -3
        elif t.shape[1] in (1, 3):  # BCHW
            return -2
        else:
            return -2
    elif t.dim() == 3:
        if t.shape[-1] in (1, 3):  # HWC
            return -3
        elif t.shape[0] in (1, 3):  # CHW
            return -2
        else:
            return -2
    else:
        raise ValueError("Unsupported tensor rank for vertical dimension detection")


def _as_tensor(x):
    """尽量保持原 tensor，不是 tensor 则转为 tensor（设备保持原样，通常为 CPU）"""
    return x if isinstance(x, torch.Tensor) else torch.as_tensor(x)


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
    tail = torch.index_select(t, dim=wdim, index=idx)
    return torch.cat([t, tail], dim=wdim)


# ====================== 变换基类 ======================

class BatchTransform:
    """基类：定义统一的 batch 变换接口"""
    def __call__(self, batch: Any) -> Any:
        return batch


class IdentityBatchTransform(BatchTransform):
    """
    什么都不做：验证/测试阶段使用
    - 若 batch 是 dict，则补充 hshift_pix=0（便于下游统一处理）
    """
    def __call__(self, batch: Any) -> Any:
        if isinstance(batch, dict):
            batch.setdefault("hshift_pix", 0)
        return batch


# ====================== 训练用滚动增强 ======================

class RollHBatchTransform(BatchTransform):
    """
    ERP 友好的水平滚动（训练阶段的数据增强）：
    - 同步滚动若干键（默认仅 'jpg'），宽度维自动识别；
    - 把像素域偏移量写入 batch['hshift_pix']（向右为正，整数像素）；
    - 数值/尺寸/dtype 不变（torch.roll 非就地）。
    """
    def __init__(
        self,
        max_frac: float = 0.2,
        enabled: bool = True,
        keys: Iterable[str] = ("jpg",),
    ):
        self.max_frac = float(max_frac)
        self.enabled = bool(enabled)
        self.keys = tuple(keys)

    def __call__(self, batch: Any) -> Any:
        if not isinstance(batch, dict):
            return batch

        # 关闭则补 0 并返回
        if not self.enabled:
            batch.setdefault("hshift_pix", 0)
            return batch

        # 参照张量：决定宽度与 shift 范围
        if "jpg" not in batch:
            batch.setdefault("hshift_pix", 0)
            return batch

        ref = _as_tensor(batch["jpg"])
        try:
            wdim = _find_wdim(ref)
        except Exception:
            batch.setdefault("hshift_pix", 0)
            return batch

        W = int(ref.shape[wdim])
        if W <= 1:
            batch.setdefault("hshift_pix", 0)
            return batch

        max_shift = max(1, int(round(W * self.max_frac)))
        shift = int(random.randint(-max_shift, max_shift))

        if shift != 0:
            for k in self.keys:
                if k not in batch:
                    continue
                t = _as_tensor(batch[k])
                try:
                    _wdim = _find_wdim(t)
                except Exception:
                    continue  # 不支持的形状：跳过
                batch[k] = torch.roll(t, shifts=shift, dims=_wdim)

        batch["hshift_pix"] = shift
        return batch


# ====================== 右侧环向拓宽 ======================

class TailExtendBatchTransform(BatchTransform):
    """
    右侧“环向拓宽”：
    - 拼接像素长度 K = ceil(W0 * extend_frac)，再向上取整到 align 的倍数（默认 64）
    - 拼接来源：原图左端 [:K]（环向/循环，支持 K > W0）
    - 记录原图尺寸到 batch['orig_size'] = LongTensor([H0, W0])
    - 若上游未设置 hshift_pix，则补为 0（便于 val/test 统一）
    """
    def __init__(self, extend_frac: float = 0.1, align: int = 64, keys: Iterable[str] = ("jpg",)):
        self.extend_frac = float(extend_frac)
        self.align = int(max(1, align))
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

        # 计算 K：W0 的 1/10，上取整到 align 的倍数
        k0 = int(math.ceil(W0 * self.extend_frac))
        K = int(math.ceil(max(1, k0) / float(self.align)) * self.align)

        # 对指定 keys 依次拓宽
        for k in self.keys:
            if k not in batch:
                continue
            t = _as_tensor(batch[k])
            _wdim = _find_wdim(t)
            batch[k] = _cat_right(t, K, _wdim)

        # 记录原图尺寸，补齐 hshift_pix（若未给出）
        batch["orig_size"] = torch.tensor([H0, W0], dtype=torch.long)
        batch.setdefault("hshift_pix", 0)

        # 可选：把真实拼接像素 K 暴露出去（下游也可用 W_ext - W0 计算）
        batch["extend_k_pix"] = torch.tensor(int(K), dtype=torch.long)
        if not hasattr(self, "_dbg_once"):
            print(f"[TailExtend] W0={W0}, K={K} -> W_ext={W0 + K}")
            self._dbg_once = True

        return batch


# ====================== 组合：先滚动后拓宽（训练） ======================

class RollThenExtendBatchTransform(BatchTransform):
    """
    训练组合：先水平滚动增强，再做右侧环向拓宽
    - Roller 写入 hshift_pix（可为负/零/正）
    - Extender 追加 jpg 等字段的右侧像素，并写入 orig_size、extend_k_pix
    """
    def __init__(
        self,
        roll_max_frac: float = 0.2,
        extend_frac: float = 0.1,
        align: int = 64,
        keys: Iterable[str] = ("jpg",),
    ):
        self.roller = RollHBatchTransform(max_frac=roll_max_frac, enabled=True, keys=keys)
        self.extender = TailExtendBatchTransform(extend_frac=extend_frac, align=align, keys=keys)

    def __call__(self, batch: Any) -> Any:
        batch = self.roller(batch)     # 写入 hshift_pix
        batch = self.extender(batch)   # 写入 orig_size / extend_k_pix，并把 jpg 等右侧拓宽
        return batch
