# dataset/batch_transform.py
# dataset/batch_transform.py
from typing import Any, Iterable
import random
import torch

class BatchTransform:
    """基类：定义统一的 batch 变换接口"""
    def __call__(self, batch: Any) -> Any:
        return batch

class IdentityBatchTransform(BatchTransform):
    """什么都不做：验证/测试阶段使用"""
    def __call__(self, batch: Any) -> Any:
        # 也给出 hshift_pix=0，便于下游统一处理
        if isinstance(batch, dict):
            batch.setdefault("hshift_pix", 0)
        return batch

class RollHBatchTransform(BatchTransform):
    """
    ERP 友好的水平滚动（用于训练阶段的数据增强）：
    - 同步滚动若干键（默认仅 'jpg'），宽度维自动识别；
    - 把像素域的偏移量写入 batch['hshift_pix']（向右为正，整数像素）；
    - 数值/尺寸/dtype 不变。
    """
    def __init__(
        self,
        max_frac: float = 0.2,
        enabled: bool = True,
        keys: Iterable[str] = ("jpg",),   # 需要同步滚动的键
    ):
        self.max_frac = float(max_frac)
        self.enabled = bool(enabled)
        self.keys = tuple(keys)

    def _find_wdim(self, t: torch.Tensor) -> int:
        # 识别“宽度维”（支持 [B,C,H,W] / [B,H,W,C] / [C,H,W] / [H,W,C]）
        if t.dim() == 4:
            # B C H W  /  B H W C
            if t.shape[-1] in (1, 3):  # BHWC
                return -2
            elif t.shape[1] in (1, 3): # BCHW
                return -1
            else:
                return -1
        elif t.dim() == 3:
            if t.shape[-1] in (1, 3):  # HWC
                return -2
            elif t.shape[0] in (1, 3): # CHW
                return -1
            else:
                return -1
        else:
            raise ValueError("Unsupported tensor rank for horizontal roll")

    def _as_tensor(self, x):
        if isinstance(x, torch.Tensor):
            return x
        return torch.as_tensor(x)

    def __call__(self, batch: Any) -> Any:
        if not isinstance(batch, dict):
            return batch

        if not self.enabled:
            batch.setdefault("hshift_pix", 0)
            return batch

        # 取参照张量决定宽度与 shift
        if "jpg" not in batch:
            batch.setdefault("hshift_pix", 0)
            return batch

        ref = self._as_tensor(batch["jpg"])
        wdim = self._find_wdim(ref)
        W = ref.shape[wdim]
        if W <= 1:
            batch.setdefault("hshift_pix", 0)
            return batch

        max_shift = max(1, int(W * self.max_frac))
        shift = int(random.randint(-max_shift, max_shift))

        # 同步滚动 keys
        if shift != 0:
            for k in self.keys:
                if k not in batch:
                    continue
                t = self._as_tensor(batch[k])
                try:
                    _wdim = self._find_wdim(t)
                except Exception:
                    continue  # 安全回退
                batch[k] = torch.roll(t, shifts=shift, dims=_wdim)

        batch["hshift_pix"] = shift  # 关键：把像素域偏移给到模型
        return batch
