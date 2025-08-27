# dataset/lic_full_erp.py
import math
from typing import List, Optional
from PIL import Image
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF

from utils.file import load_file_list  # 作者提供的工具

def _as_rgb(img: Image.Image) -> Image.Image:
    return img if img.mode == "RGB" else img.convert("RGB")

def _to_tensor_norm(img: Image.Image) -> torch.Tensor:
    # [0,1] -> [-1,1]，与 SD/RDEIC 习惯一致
    t = TF.to_tensor(img)
    return t * 2.0 - 1.0

def _resize_keep_ar_to_long(img: Image.Image, long_edge: Optional[int]) -> Image.Image:
    if not long_edge or long_edge <= 0:
        return img
    w, h = img.size
    long_src = max(w, h)
    if long_src == long_edge:
        return img
    scale = long_edge / float(long_src)
    nw, nh = int(round(w * scale)), int(round(h * scale))
    return img.resize((nw, nh), Image.BICUBIC)

def _ceil_mul(x: int, m: int) -> int:
    return int(math.ceil(x / float(m)) * m)

def _pad_to_multiple(x: torch.Tensor, multiple: int,
                     v_mode: str = "reflect", v_value: float = 0.0) -> torch.Tensor:
    """
    把 (N,C,H,W) pad 到 multiple 的倍数：
      - 纵向：reflect/replicate/constant（极区更稳）
      - 横向：circular（经度无缝）
    """
    n, c, h, w = x.shape
    th = _ceil_mul(h, multiple)
    tw = _ceil_mul(w, multiple)
    ph, pw = th - h, tw - w

    if ph > 0:
        top = ph // 2
        bot = ph - top
        if v_mode == "constant":
            x = F.pad(x, (0, 0, top, bot), mode="constant", value=float(v_value))
        elif v_mode in ("reflect", "replicate"):
            x = F.pad(x, (0, 0, top, bot), mode=v_mode)
        else:
            raise ValueError(f"Unsupported vertical pad mode: {v_mode}")

    if pw > 0:
        l = pw // 2
        r = pw - l
        x = F.pad(x, (l, r, 0, 0), mode="circular")

    return x

def _random_hroll(x: torch.Tensor, max_shift: Optional[int] = None) -> torch.Tensor:
    if x.dim() != 4:
        return x
    W = x.size(-1)
    if W <= 1:
        return x
    if not max_shift or max_shift <= 0 or max_shift > W - 1:
        max_shift = W - 1
    shifts = torch.randint(0, max_shift + 1, (x.size(0),), device=x.device)
    return torch.stack([torch.roll(x[i], int(shifts[i]), dims=-1) for i in range(x.size(0))], dim=0)

class LICFullERPDataset(Dataset):
    """
    整幅 ERP，不裁剪：
      1) 可选长边等比缩放（控制显存）
      2) pad 到 multiple（默认 64）的倍数：
         - 纵向 reflect/replicate/constant
         - 横向 circular
      3) 可选训练期随机水平 roll（增强经度平移不变性）
    返回：{'image': tensor[-1,1], 'path': str}
    """
    def __init__(self,
                 file_list: str,
                 long_edge: Optional[int] = None,
                 multiple: int = 64,
                 vertical_pad_mode: str = "reflect",
                 vertical_pad_value: float = 0.0,
                 random_roll: bool = False):
        super().__init__()
        self.paths: List[str] = load_file_list(file_list)
        self.long_edge = long_edge
        self.multiple = multiple
        self.v_mode = vertical_pad_mode
        self.v_value = vertical_pad_value
        self.random_roll = random_roll

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx: int):
        path = self.paths[idx]
        img = _as_rgb(Image.open(path))
        img = _resize_keep_ar_to_long(img, self.long_edge)

        x = _to_tensor_norm(img).unsqueeze(0)              # (1,C,H,W)
        x = _pad_to_multiple(x, self.multiple, self.v_mode, self.v_value)
        if self.random_roll:
            x = _random_hroll(x)                           # 仅训练集建议打开
        x = x.squeeze(0)                                   # (C,H',W')
        return {"image": x, "path": path}
