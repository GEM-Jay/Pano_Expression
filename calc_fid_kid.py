import torch, torchvision
from pathlib import Path
from PIL import Image
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.kid import KernelInceptionDistance

# ---------------- 配置路径 ----------------
real_dir = Path("./dataset/test")        # 真值
fake_dir = Path("./output/rdeic_0.25_step2")     # 重建
device   = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------- 初始化 metric ----------------
n_real = len(list(real_dir.glob('*.png'))) + len(list(real_dir.glob('*.jpg')))
fid = FrechetInceptionDistance(feature=2048, normalize=True,reset_real_features=False).to(device)
kid = KernelInceptionDistance(subset_size=min(50, n_real),  normalize=True,reset_real_features=False).to(device)


# ---------------- 数据加载辅助 ----------------
TFMS = torchvision.transforms.Compose([
    torchvision.transforms.Resize((299, 299)),   # Inception-v3 默认输入
    torchvision.transforms.ToTensor(),           # [0,1]
])

def iter_imgs(folder):
    for p in sorted(folder.glob("*.png")) + sorted(folder.glob("*.jpg")):
        img = TFMS(Image.open(p).convert("RGB"))
        yield img.unsqueeze(0)        # [1,3,299,299]

# ---------------- 先累积 real，再累积 fake ----------------
with torch.no_grad():
    for img in iter_imgs(real_dir):
        fid.update(img.to(device), real=True)
        kid.update(img.to(device), real=True)

    for img in iter_imgs(fake_dir):
        fid.update(img.to(device), real=False)
        kid.update(img.to(device), real=False)

# ---------------- 计算并打印 ----------------
print("FID :", fid.compute().item())                   # 单标量
kid_mean, kid_std = kid.compute()
print(f"KID : {kid_mean.item():.5f} ± {kid_std.item():.5f}")
