# utils/resize.py
import torch
import torch.nn.functional as F

class ResizeToWidth64x:
    """
    将 [C,H,W] 张量按宽度缩放到 target_w（必须是 64 的倍数），高度等比例变化。
    适用于 [-1,1] 或 [0,1] 的图像张量。
    """
    def __init__(self, target_w: int):
        assert target_w % 64 == 0, "target_w 必须为 64 的倍数，才能与 VAE/UNet 尺寸对齐"
        self.target_w = int(target_w)

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        # img: [C,H,W]
        assert img.dim() == 3, f"expected [C,H,W], got {img.shape}"
        C, H, W = img.shape
        if W == self.target_w:
            return img
        new_h = int(round(H * self.target_w / W))
        img = img.unsqueeze(0)  # [1,C,H,W]
        img = F.interpolate(
            img, size=(new_h, self.target_w),
            mode="bilinear", align_corners=False, antialias=True
        )
        return img.squeeze(0)
