#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from argparse import ArgumentParser

from utils.file import list_image_files


parser = ArgumentParser(
    description="Generate a list file of original & reconstructed panorama pairs"
)
parser.add_argument("--orig_folder", type=str, required=True,
                    help="Folder of original images, e.g. ./dataset/test")
parser.add_argument("--rec_folder", type=str, required=True,
                    help="Folder of reconstructed images, e.g. ./output/2_1_12")
parser.add_argument("--save_folder", type=str, required=True,
                    help="Folder to save the .list file")
parser.add_argument("--list_name", type=str, default="pairs.list",
                    help="Name of the output list file (default: pairs.list)")
parser.add_argument("--follow_links", action="store_true")
args = parser.parse_args()

# 1) 读取原图和重建图路径
orig_files = list_image_files(
    args.orig_folder,
    exts=(".jpg", ".png", ".jpeg"),
    follow_links=args.follow_links,
    log_progress=True,
    log_every_n_files=10000,
)

rec_files = list_image_files(
    args.rec_folder,
    exts=(".jpg", ".png", ".jpeg"),
    follow_links=args.follow_links,
    log_progress=True,
    log_every_n_files=10000,
)

print(f"find {len(orig_files)} images in {args.orig_folder} (orig)")
print(f"find {len(rec_files)} images in {args.rec_folder} (recon)")

# 2) 排序，确保按递增顺序一一对应
orig_files = sorted(orig_files)
rec_files = sorted(rec_files)

if len(orig_files) != len(rec_files):
    raise ValueError(
        f"Number of original images ({len(orig_files)}) "
        f"!= number of reconstructed images ({len(rec_files)}). "
        f"Please check {args.orig_folder} and {args.rec_folder}."
    )

# 3) 写 pairs.list：一行 = "orig_path rec_path"
os.makedirs(args.save_folder, exist_ok=True)
out_path = os.path.join(args.save_folder, args.list_name)

with open(out_path, "w", encoding="utf-8") as fp:
    for orig_path, rec_path in zip(orig_files, rec_files):
        fp.write(f"{orig_path} {rec_path}\n")

print(f"Saved pair list to {out_path}")
