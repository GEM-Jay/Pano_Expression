from typing import Dict, Any, Optional
import os

import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.utilities.types import STEP_OUTPUT
import torch
import torchvision
from PIL import Image
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.utilities.distributed import rank_zero_only

from .mixins import ImageLoggerMixin


__all__ = [
    "ModelCheckpoint",
    "ImageLogger"
]


class ImageLogger(Callback):
    """
    Log images during training (and optionally validating).

    Expect pl_module.log_images(batch, **kwargs) -> (images: Dict[str, Tensor[N,C,H,W]], extra)
    where images are RGB/gray in [0,1].
    """

    def __init__(
        self,
        log_every_n_steps: int = 2000,
        log_start_step: int = 6000,
        max_images_each_step: int = 4,
        log_images_kwargs: Optional[Dict[str, Any]] = None,
        enable_val_logging: bool = False,
    ) -> "ImageLogger":
        super().__init__()
        self.log_every_n_steps = log_every_n_steps
        self.log_start_step = log_start_step
        self.max_images_each_step = max_images_each_step
        self.log_images_kwargs = log_images_kwargs or {}
        self.enable_val_logging = enable_val_logging

    def on_fit_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        assert isinstance(pl_module, ImageLoggerMixin), "pl_module must implement ImageLoggerMixin"

    # ---------- helpers ----------
    @staticmethod
    def _resolve_save_dir(trainer: pl.Trainer, pl_module: pl.LightningModule, split: str) -> str:
        # 优先使用 logger.save_dir，其次 logger.experiment.log_dir，最后回退 default_root_dir
        save_root = getattr(pl_module.logger, "save_dir", None)
        if not save_root and hasattr(pl_module.logger, "experiment"):
            save_root = getattr(pl_module.logger.experiment, "log_dir", None)
        if not save_root:
            save_root = trainer.default_root_dir
        out = os.path.join(save_root, "image_log", split)
        os.makedirs(out, exist_ok=True)
        return out

    @staticmethod
    def _make_grid(images: torch.Tensor, nrow: int = 4) -> np.ndarray:
        """
        images: Tensor[N,C,H,W] in [0,1]
        return: ndarray[H,W,C] (uint8) or [H,W] for gray
        """
        grid = torchvision.utils.make_grid(images, nrow=nrow)  # [C,H,W]
        grid = grid.clamp(0, 1).mul(255).byte().permute(1, 2, 0).cpu().numpy()  # [H,W,C]
        if grid.shape[2] == 1:  # gray
            grid = grid.squeeze(2)  # [H,W]
        return grid

    def _should_log(self, global_step: int) -> bool:
        return (global_step > self.log_start_step) and (global_step % self.log_every_n_steps == 0)

    # ---------- train ----------
    @rank_zero_only
    def on_train_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: STEP_OUTPUT,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
        *args,
        **kwargs,
    ) -> None:
        # 兼容 Lightning：训练阶段通常不传 dataloader_idx，这里给默认值避免缺参
        global_step = trainer.global_step
        if not self._should_log(global_step):
            return

        was_training = pl_module.training
        if was_training and hasattr(pl_module, "freeze"):
            pl_module.freeze()

        with torch.no_grad():
            images_dict, _ = pl_module.log_images(batch, **self.log_images_kwargs)

        save_dir = self._resolve_save_dir(trainer, pl_module, split="train")
        for key, tensor in images_dict.items():
            # tensor: [N,C,H,W]
            if not isinstance(tensor, torch.Tensor):
                continue
            img = tensor.detach().cpu()
            if img.dim() != 4:
                continue
            N = min(self.max_images_each_step, img.size(0))
            grid = self._make_grid(img[:N], nrow=4)
            filename = f"{key}_step-{global_step:06d}_e-{trainer.current_epoch:06d}_b-{batch_idx:06d}.png"
            Image.fromarray(grid).save(os.path.join(save_dir, filename))

        if was_training and hasattr(pl_module, "unfreeze"):
            pl_module.unfreeze()

    # ---------- val (optional) ----------
    @rank_zero_only
    def on_validation_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: STEP_OUTPUT,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
        *args,
        **kwargs,
    ) -> None:
        if not self.enable_val_logging:
            return
        # 可按需要降低频率：例如每个 epoch 的第一个 batch 才存
        if batch_idx != 0:
            return

        with torch.no_grad():
            images_dict, _ = pl_module.log_images(batch, **self.log_images_kwargs)

        save_dir = self._resolve_save_dir(trainer, pl_module, split="val")
        for key, tensor in images_dict.items():
            if not isinstance(tensor, torch.Tensor) or tensor.dim() != 4:
                continue
            N = min(self.max_images_each_step, tensor.size(0))
            grid = self._make_grid(tensor.detach().cpu()[:N], nrow=4)
            filename = f"{key}_val_e-{trainer.current_epoch:06d}_b-{batch_idx:06d}.png"
            Image.fromarray(grid).save(os.path.join(save_dir, filename))
