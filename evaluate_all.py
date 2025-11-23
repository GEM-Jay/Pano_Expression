#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
python evaluate_all.py orig_dir rec_dir

全景压缩评估脚本：

输入：
    - orig_dir：原始全景图目录（未压缩图像）
    - rec_dir ：重建全景图目录（解码后的图像）

输出：
    - WS-PSNR（逐图像、成对比较）
    - WS-SSIM（逐图像、成对比较）
    - FID / KID（两个图像集合的分布差异）

注意：
    - WS 指标：按「文件名」对齐，例如 00001.png 对应 00001.png。
    - FID / KID：只看集合分布，不要求一一对应。
"""

import argparse
import glob
import math
import os.path as osp
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.kid import KernelInceptionDistance


# ====================== 1. WS-PSNR / WS-SSIM ======================

def gen_erp_weight(row_idx: int, n_rows: int) -> float:
    """ERP 球面加权：按纬度方向 cos 权重。"""
    return math.cos((row_idx - (n_rows / 2) + 0.5) * math.pi / n_rows)


def compute_ws_weight_map(img: np.ndarray) -> np.ndarray:
    """生成整张图的 ERP 权重图，img: HWC, uint8 / float 都可以。"""
    h, w, c = img.shape
    weight = np.zeros((h, w, c), dtype=np.float64)
    for i in range(h):
        weight[i, :, :] = gen_erp_weight(i, h)
    return weight


def calculate_ws_psnr(orig: np.ndarray, rec: np.ndarray) -> float:
    """加权球面 WS-PSNR，输入 BGR uint8。"""
    orig = orig.astype(np.float64)
    rec = rec.astype(np.float64)
    weight = compute_ws_weight_map(orig)

    mse = np.mean((orig - rec) ** 2 * weight) / np.mean(weight)
    if mse == 0:
        return float("inf")
    return 10 * np.log10(255.0 * 255.0 / mse)


def _ws_ssim_single_channel(orig: np.ndarray, rec: np.ndarray) -> float:
    """单通道 WS-SSIM（内部用）。"""
    c1 = (0.01 * 255) ** 2
    c2 = (0.03 * 255) ** 2
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel)

    mu1 = cv2.filter2D(orig, -1, window)[5:-5, 5:-5]
    mu2 = cv2.filter2D(rec, -1, window)[5:-5, 5:-5]
    mu1_sq, mu2_sq = mu1 ** 2, mu2 ** 2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = cv2.filter2D(orig ** 2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(rec ** 2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D((orig * rec), -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / (
        (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2)
    )

    h, w = ssim_map.shape
    weight = np.zeros_like(ssim_map)
    for i in range(h):
        weight[i, :] = gen_erp_weight(i, h)

    return float(np.mean(ssim_map * weight) / np.mean(weight))


def calculate_ws_ssim(orig: np.ndarray, rec: np.ndarray) -> float:
    """多通道 WS-SSIM，输入 BGR uint8。"""
    orig = orig.astype(np.float64)
    rec = rec.astype(np.float64)
    vals = [
        _ws_ssim_single_channel(orig[..., c], rec[..., c])
        for c in range(orig.shape[2])
    ]
    return float(np.mean(vals))


def compute_ws_metrics(orig_dir: str, rec_dir: str):
    """
    根据文件名对齐 orig / rec 图像，计算 WS-PSNR / WS-SSIM。
    会逐张输出每张图片的指标，并最后输出平均值。
    """
    valid_ext = (".png", ".jpg", ".jpeg")

    def build_name_to_path_map(folder: str):
        paths = [
            p for p in glob.glob(osp.join(folder, "*"))
            if p.lower().endswith(valid_ext)
        ]
        mp = {}
        for p in paths:
            name = osp.splitext(osp.basename(p))[0]
            mp[name] = p
        return mp

    orig_map = build_name_to_path_map(orig_dir)
    rec_map = build_name_to_path_map(rec_dir)

    common_names = sorted(set(orig_map.keys()) & set(rec_map.keys()))
    if len(common_names) == 0:
        raise RuntimeError(
            f"没有找到可以对齐的文件名（orig_dir={orig_dir}, rec_dir={rec_dir})"
        )

    if len(orig_map) != len(rec_map):
        print(
            f"[WARN] 原图数量 {len(orig_map)}，重建图数量 {len(rec_map)}；"
            f"实际对齐 {len(common_names)} 张（按文件名交集）"
        )

    ws_psnrs, ws_ssims = [], []

    print("\n===== 每张图像 WS 指标 =====")
    for name in common_names:
        orig_path = orig_map[name]
        rec_path = rec_map[name]

        orig_img = cv2.imread(orig_path)
        rec_img = cv2.imread(rec_path)

        assert orig_img is not None, f"读取失败: {orig_path}"
        assert rec_img is not None, f"读取失败: {rec_path}"
        assert orig_img.shape == rec_img.shape, \
            f"尺寸不一致: {orig_path} vs {rec_path}"

        psnr = calculate_ws_psnr(orig_img, rec_img)
        ssim = calculate_ws_ssim(orig_img, rec_img)

        ws_psnrs.append(psnr)
        ws_ssims.append(ssim)


    avg_psnr = float(sum(ws_psnrs) / len(ws_psnrs))
    avg_ssim = float(sum(ws_ssims) / len(ws_ssims))

    return avg_psnr, avg_ssim, len(common_names)



# ====================== 2. FID / KID ======================

class PanoramaImageDataset(Dataset):
    """
    FID/KID 用的数据集封装：
    直接从目录中读所有 .png/.jpg/.jpeg，resize 到 299×299。
    """

    def __init__(self, root: Path):
        self.root = Path(root)
        exts = (".png", ".jpg", ".jpeg")
        self.paths = sorted(
            [p for p in self.root.iterdir() if p.suffix.lower() in exts]
        )
        if len(self.paths) == 0:
            raise RuntimeError(f"目录中没有图片: {self.root}")

        self.tfms = torchvision.transforms.Compose([
            torchvision.transforms.Resize((299, 299)),  # Inception v3 输入
            torchvision.transforms.ToTensor(),          # [0,1]
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        p = self.paths[idx]
        img = Image.open(p).convert("RGB")
        img = self.tfms(img)
        return img


def compute_fid_kid_for_compression(
    orig_dir: str,
    rec_dir: str,
    batch_size: int = 16,
    max_orig: int | None = None,
    max_rec: int | None = None,
    device: str | None = None,
):
    """
    压缩场景下的 FID/KID：
    - orig_dir：原始图集合
    - rec_dir ：重建图集合
    """
    orig_dir = Path(orig_dir)
    rec_dir = Path(rec_dir)

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    orig_ds = PanoramaImageDataset(orig_dir)
    rec_ds = PanoramaImageDataset(rec_dir)

    if max_orig is not None:
        orig_ds.paths = orig_ds.paths[:max_orig]
    if max_rec is not None:
        rec_ds.paths = rec_ds.paths[:max_rec]

    n_orig = len(orig_ds)
    n_rec = len(rec_ds)

    orig_loader = DataLoader(orig_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    rec_loader = DataLoader(rec_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    fid = FrechetInceptionDistance(
        feature=2048,
        normalize=True,
        reset_real_features=False,
    ).to(device)

    kid_subset_size = min(50, n_orig)
    kid = KernelInceptionDistance(
        subset_size=kid_subset_size,
        normalize=True,
        reset_real_features=False,
    ).to(device)

    with torch.no_grad():
        for imgs in orig_loader:
            imgs = imgs.to(device)
            fid.update(imgs, real=True)
            kid.update(imgs, real=True)

        for imgs in rec_loader:
            imgs = imgs.to(device)
            fid.update(imgs, real=False)
            kid.update(imgs, real=False)

    fid_val = float(fid.compute().item())
    kid_mean, kid_std = kid.compute()

    return {
        "fid": float(fid_val),
        "kid_mean": float(kid_mean.item()),
        "kid_std": float(kid_std.item()),
        "n_orig": n_orig,
        "n_rec": n_rec,
    }


# ====================== 3. 一键评估入口 ======================

def evaluate_panorama_compression(
    orig_dir: str,
    rec_dir: str,
    batch_size: int = 16,
    max_orig_fid: int | None = None,
    max_rec_fid: int | None = None,
    device: str | None = None,
):
    print(f"[INFO] 原始图目录  : {orig_dir}")
    print(f"[INFO] 重建图目录  : {rec_dir}")

    # 1) 成对指标：WS-PSNR / WS-SSIM
    ws_psnr, ws_ssim, n_pairs = compute_ws_metrics(orig_dir, rec_dir)
    print("\n========== WS-PSNR 、 WS-SSIM（原图 vs 重建） ==========")
    print(f"重建图数量 : {n_pairs}")
    print(f"WS-PSNR   : {ws_psnr:.5f}")
    print(f"WS-SSIM   : {ws_ssim:.5f}")
    print("===========================================================")

    # 2) 集合指标：FID / KID
    print("\n[INFO] 计算 FID 、 KID（特征分布差异） ...")
    fk = compute_fid_kid_for_compression(
        orig_dir=orig_dir,
        rec_dir=rec_dir,
        batch_size=batch_size,
        max_orig=max_orig_fid,
        max_rec=max_rec_fid,
        device=device,
    )

    print("\n===================== FID 、 KID =====================")
    print(f"FID       : {fk['fid']:.5f}")
    print(f"KID       : {fk['kid_mean']:.5f} ± {fk['kid_std']:.5f}")
    print("====================================================")

    return {
        "ws_psnr": ws_psnr,
        "ws_ssim": ws_ssim,
        **fk,
        "n_pairs_ws": n_pairs,
    }


# ====================== 4. CLI ======================

def parse_args():
    parser = argparse.ArgumentParser(
        description="全景压缩评估：原图 vs 重建图（WS-PSNR / WS-SSIM / FID / KID）"
    )
    parser.add_argument("orig_dir", type=str, help="原始全景图目录")
    parser.add_argument("rec_dir", type=str, help="重建全景图目录")
    parser.add_argument(
        "--batch_size", type=int, default=16,
        help="FID/KID 时的 batch size（默认 16）"
    )
    parser.add_argument(
        "--max_orig_fid", type=int, default=None,
        help="FID/KID 中最多使用多少张原图（默认全部）"
    )
    parser.add_argument(
        "--max_rec_fid", type=int, default=None,
        help="FID/KID 中最多使用多少张重建图（默认全部）"
    )
    parser.add_argument(
        "--device", type=str, default=None,
        help="cuda / cpu，不填则自动选择"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    evaluate_panorama_compression(
        orig_dir=args.orig_dir,
        rec_dir=args.rec_dir,
        batch_size=args.batch_size,
        max_orig_fid=args.max_orig_fid,
        max_rec_fid=args.max_rec_fid,
        device=args.device,
    )
