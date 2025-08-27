from typing import Sequence, Dict, Union
import time

import numpy as np
from PIL import Image
import torch.utils.data as data

from utils.file import load_file_list
from utils.image import center_crop_arr, augment, random_crop_arr, random_crop_arr_256


class LICDataset(data.Dataset):

    def __init__(
            self,
            file_list: str,
            out_size: int,
            crop_type: str,
            use_hflip: bool,
            use_rot: bool,
    ) -> "LICDataset":
        super(LICDataset, self).__init__()
        self.file_list = file_list
        self.paths = load_file_list(file_list)
        self.out_size = out_size
        self.crop_type = crop_type
        assert self.crop_type in ["none", "center", "random"]
        self.use_hflip = use_hflip
        self.use_rot = use_rot

    def __getitem__(self, index: int) -> Dict[str, Union[np.ndarray, str]]:
        gt_path = self.paths[index]

        # 1) 读图：PIL(RGB)
        ok = False
        for _ in range(3):
            try:
                img_pil = Image.open(gt_path).convert("RGB")  # 强制三通道
                ok = True
                break
            except Exception:
                time.sleep(1)
        assert ok, f"failed to load image {gt_path}"

        # 2) 裁剪：把 PIL 传给裁剪函数（避免 ndarray.size 被当作整数）
        if self.crop_type == "center":
            cropped = center_crop_arr(img_pil, self.out_size)
        elif self.crop_type == "random":
            cropped = random_crop_arr_256(img_pil, self.out_size) if self.out_size == 256 \
                else random_crop_arr(img_pil, self.out_size)
        else:
            cropped = img_pil

        # 3) 统一到 ndarray(H, W, 3, uint8)
        if isinstance(cropped, Image.Image):
            img = np.array(cropped, dtype=np.uint8)
        else:
            img = cropped
            if img.dtype != np.uint8:
                img = np.clip(img, 0, 255).astype(np.uint8)

        # 4) 再清洗一次通道（双保险）
        if img.ndim == 2:
            img = np.stack([img, img, img], axis=-1)
        elif img.ndim == 3 and img.shape[2] != 3:
            img = img[:, :, :3]

        # 5) 归一化到 [0,1] 并增广（保持 HWC）
        img = img.astype(np.float32) / 255.0
        img = augment(img, hflip=self.use_hflip, rotation=self.use_rot, return_status=False)

        # 6) 增广后最终校验（彻底保证 C==3）
        if img.ndim == 2:
            img = np.stack([img, img, img], axis=-1)
        elif img.ndim == 3 and img.shape[2] != 3:
            img = img[:, :, :3]
        assert img.ndim == 3 and img.shape[2] == 3, f"[LICDataset] got {img.shape} for {gt_path}"

        # 7) 映射到 [-1, 1]
        target = (img * 2 - 1).astype(np.float32)

        return dict(jpg=target, txt="")


    def __len__(self) -> int:
        return len(self.paths)