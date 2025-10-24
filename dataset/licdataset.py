# dataset/licdataset.py
from typing import Sequence, Dict, Union
import time
from utils.resize import ResizeToWidth64x  # ← 新增
import numpy as np
from PIL import Image
import torch  # ← 新增
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
            resize_cfg: dict = None,  # ← 新增：下采样配置
    ) -> "LICDataset":
        super(LICDataset, self).__init__()
        self.file_list = file_list
        self.paths = load_file_list(file_list)
        self.out_size = out_size
        self.crop_type = crop_type
        assert self.crop_type in ["none", "center", "random"]
        self.use_hflip = use_hflip
        self.use_rot = use_rot

        # ===== 新增：可选的宽度对齐下采样（等比例缩放高度） =====
        self.resize = None
        if resize_cfg and resize_cfg.get("enabled", False):
            target_w = int(resize_cfg.get("target_w", 1024))
            assert target_w % 64 == 0, "resize_cfg.target_w 必须是 64 的倍数"
            self.resize = ResizeToWidth64x(target_w)

    def __getitem__(self, index: int) -> Dict[str, Union[np.ndarray, str]]:
        # load gt image
        # Shape: (h, w, c); channel order: BGR; image range: [0, 1], float32.
        gt_path = self.paths[index]
        success = False
        for _ in range(3):
            try:
                pil_img = Image.open(gt_path).convert("RGB")
                success = True
                break
            except:
                time.sleep(1)
        assert success, f"failed to load image {gt_path}"

        # 先按原逻辑裁剪（如需保留整图，请把 crop_type 设为 "none"）
        if self.crop_type == "center":
            pil_img_gt = center_crop_arr(pil_img, self.out_size)
        elif self.crop_type == "random":
            if self.out_size == 256:
                pil_img_gt = random_crop_arr_256(pil_img, self.out_size)
            else:
                pil_img_gt = random_crop_arr(pil_img, self.out_size)
        else:
            pil_img_gt = np.array(pil_img)  # HWC, uint8

        # ===== 新增：可选的按宽度下采样（保持等比例）=====
        if self.resize is not None:
            # 这里用 torch 的 ResizeToWidth64x（输入 [C,H,W]，0..1），再转回 numpy
            img_t = torch.from_numpy(pil_img_gt).permute(2, 0, 1).float() / 255.0  # [C,H,W], 0..1
            img_t = self.resize(img_t)  # 下采样
            pil_img_gt = (img_t.clamp(0, 1).permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)

        # hwc, [0, 255] to [0, 1], float32
        img_gt = (pil_img_gt / 255.0).astype(np.float32)

        # random horizontal flip / rotation
        img_gt = augment(img_gt, hflip=self.use_hflip, rotation=self.use_rot, return_status=False)

        # [-1, 1]
        target = (img_gt * 2 - 1).astype(np.float32)

        return dict(jpg=target, txt="")

    def __len__(self) -> int:
        return len(self.paths)
