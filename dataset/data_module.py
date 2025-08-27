# dataset/data_module.py
from typing import Any, Tuple, Mapping, Optional

from pytorch_lightning.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset
from omegaconf import OmegaConf

from utils.common import instantiate_from_config
from dataset.batch_transform import BatchTransform, IdentityBatchTransform


class DataModule(pl.LightningDataModule):

    def __init__(
            self,
            train_config: str,
            val_config: str = None
    ) -> "DataModule":
        super().__init__()
        self.train_config = OmegaConf.load(train_config)
        self.val_config = OmegaConf.load(val_config) if val_config else None

        # 防止属性未定义
        self.train_dataset: Optional[Dataset] = None
        self.val_dataset: Optional[Dataset] = None
        self.train_batch_transform: BatchTransform = IdentityBatchTransform()
        self.val_batch_transform: BatchTransform = IdentityBatchTransform()

    def load_dataset(self, config: Mapping[str, Any]) -> Tuple[Dataset, BatchTransform]:
        dataset = instantiate_from_config(config["dataset"])
        batch_transform = (
            instantiate_from_config(config["batch_transform"])
            if config.get("batch_transform") else IdentityBatchTransform()
        )
        return dataset, batch_transform

    def setup(self, stage: Optional[str] = None) -> None:
        # Lightning 可能传入 None / 'fit' / 'validate'
        if stage in (None, "fit", "validate"):
            self.train_dataset, self.train_batch_transform = self.load_dataset(self.train_config)
            if self.val_config:
                self.val_dataset, self.val_batch_transform = self.load_dataset(self.val_config)
            else:
                self.val_dataset, self.val_batch_transform = None, None

            # 打印放在加载完成之后
            print("[DataModule] train:", type(self.train_dataset).__name__,
                  "val:", type(self.val_dataset).__name__ if self.val_dataset is not None else None)
        else:
            # 其余阶段暂未实现
            raise NotImplementedError(stage)

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(
            dataset=self.train_dataset, **self.train_config["data_loader"]
        )

    def val_dataloader(self) -> EVAL_DATALOADERS:
        if self.val_dataset is None:
            return None
        return DataLoader(
            dataset=self.val_dataset, **self.val_config["data_loader"]
        )

    def on_after_batch_transfer(self, batch: Any, dataloader_idx: int) -> Any:
        self.trainer: pl.Trainer
        if self.trainer.training:
            return self.train_batch_transform(batch)
        elif self.trainer.validating or self.trainer.sanity_checking:
            return self.val_batch_transform(batch)
        else:
            raise RuntimeError(
                "Trainer state: \n"
                f"training: {self.trainer.training}\n"
                f"validating: {self.trainer.validating}\n"
                f"testing: {self.trainer.testing}\n"
                f"predicting: {self.trainer.predicting}\n"
                f"sanity_checking: {self.trainer.sanity_checking}"
            )
