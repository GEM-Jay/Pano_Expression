# -*- coding: utf-8 -*-
"""
RDEIC 推理脚本（第二阶段 / refine 阶段）

新管线要点（本脚本逻辑）：

1) 读入全景图，必要时缩放到工作分辨率（例如 train 模式下 1024x512）；
2) 再 pad 到 64 的倍数（只在推理内部使用）；
3) 直接对 pad 后的图做压缩（码流只覆盖当前工作分辨率 H0_in×W0_in）；
4) 从码流解出 c_latent, guide_hint；
5) 计算像素扩展宽度 K_pix，换算成 latent 扩展宽度 K_lat：
      K_lat = K_pix // first_stage_downsample
6) 在 latent 域做“右侧扩宽”：把左侧 K_lat 列复制到最右侧：
      c_ext = [c_latent, c_latent[..., :K_lat]]
7) 在扩宽后的 latent 上跑扩散采样（control / guide 同步扩宽）；
8) VAE 解码一次，内部使用右侧 tail 做左侧融合，然后去 pad、裁回工作分辨率 H0_in×W0_in；
9) 只保存一张最终图（融合 + 裁剪回工作分辨率的 ERP）。

说明：
- 当 resize_mode="train" 时，工作分辨率为 1024x512：
    * 压缩/解码/扩散都在 1024x512 域内进行；
    * bpp 以 1024x512 的像素数为分母；
    * 最终输出图像为 1024x512。
- 当 resize_mode="original" 时，工作分辨率为输入原图分辨率：
    * bpp 以原图像素数为分母；
    * 输出分辨率为原图分辨率。

本版额外修改：
- 强制关闭 LatentDiffusion 的 VAE 分块处理（split_input_params = None），
  让 encode/decode 都走整图路径，方便排查棋盘/点阵伪影。
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "7"

from argparse import ArgumentParser, Namespace
from typing import List, Tuple

import numpy as np
import torch
import einops
import pytorch_lightning as pl
from PIL import Image
from omegaconf import OmegaConf
import time  # 计时

from ldm.xformers_state import disable_xformers
from model.spaced_sampler_relay import SpacedSampler
from model.ddim_sampler_relay import DDIMSampler
from model.rdeic import RDEIC
from utils.common import instantiate_from_config, load_state_dict
from utils.file import list_image_files, get_file_name_parts
from utils.image import pad
from utils.residual_prompt import ResidualPromptDB  # 残差语义支持


def latent_tail_extend(x: torch.Tensor, K_lat: int) -> torch.Tensor:
    """
    在 latent 域做“右侧扩宽”：把左侧 K_lat 列复制到最右侧。
    输入 x: [B, C, H, W]，输出: [B, C, H, W+K_lat]
    """
    if K_lat is None or K_lat <= 0:
        return x
    assert x.dim() == 4
    W = x.shape[-1]
    K_lat = min(K_lat, W)
    head = x[..., :K_lat]
    return torch.cat([x, head], dim=-1)


@torch.no_grad()
def process(
    model: RDEIC,
    imgs: List[np.ndarray],
    sampler: str,
    steps: int,
    stream_path: str,
    guidance_scale: float,
    c_crossattn: List[torch.Tensor],
    extend_px: int = -1,
    resize_mode: str = "original",  # original / train
) -> Tuple[List[np.ndarray], float]:

    assert len(imgs) > 0
    n_samples = len(imgs)

    # 原始图分辨率（仅用于 original 模式；train 模式只是“数据来源”的尺寸）
    H0_orig, W0_orig = imgs[0].shape[:2]

    # 根据模式决定工作分辨率：
    # - original: 工作分辨率 = 原图尺寸
    # - train   : 工作分辨率 = 1024x512（与训练保持一致）
    imgs_proc = imgs
    if resize_mode == "train":
        target_w, target_h = 1024, 512
        imgs_proc = []
        for arr in imgs:
            h, w = arr.shape[:2]
            if h == target_h and w == target_w:
                imgs_proc.append(arr)
            else:
                pil = Image.fromarray(arr)
                pil = pil.resize((target_w, target_h), Image.BICUBIC)
                imgs_proc.append(np.array(pil))

    # 工作分辨率（后续 pad / 解码 / 裁剪都按这个来）
    H0_in, W0_in = imgs_proc[0].shape[:2]

    # pad 到 64 的倍数（只在内部使用）
    imgs_pad = [pad(x, scale=64) for x in imgs_proc]
    x_np = np.stack(imgs_pad).astype(np.float32) / 255.0
    x = torch.from_numpy(x_np).to(model.device)
    x = einops.rearrange(x, "n h w c -> n c h w").contiguous()

    B, C, H_pad, W_pad = x.shape
    assert B == n_samples

    # === bpp 口径：
    # - original: 使用原始分辨率 H0_orig × W0_orig
    # - train   : 使用工作分辨率 H0_in × W0_in
    if resize_mode == "train":
        H_bpp, W_bpp = H0_in, W0_in
    else:
        H_bpp, W_bpp = H0_orig, W0_orig

    bpp = model.apply_condition_compress(
        x,
        stream_path=stream_path,
        H=H_bpp,
        W=W_bpp,
    )

    # 解码出 c_latent, guide_hint（latent+控制信息）
    c_latent, guide_hint = model.apply_condition_decompress(stream_path)
    c_latent = c_latent.to(model.device)
    if guide_hint is not None:
        guide_hint = guide_hint.to(model.device)

    # 像素扩展宽度 -> latent 扩展宽度
    if extend_px is not None and extend_px >= 0:
        K_pix = int(extend_px)
    else:
        K_pix = int(getattr(model, "extend_k_pix_default", 0))
    K_pix = max(0, K_pix)

    ds = int(getattr(model, "first_stage_downsample", 8))
    K_lat = K_pix // max(ds, 1)

    # latent 右侧扩宽（只在 refine 阶段使用）
    if model.is_refine and K_lat > 0:
        c_latent_ext = latent_tail_extend(c_latent, K_lat)
        guide_hint_ext = (
            latent_tail_extend(guide_hint, K_lat)
            if guide_hint is not None else None
        )
    else:
        c_latent_ext = c_latent
        guide_hint_ext = guide_hint
        K_pix = 0  # 没有扩展，方便下游调试

    cond = {
        "c_latent": [c_latent_ext],
        "c_crossattn": c_crossattn,
        "guide_hint": guide_hint_ext,
    }

    b, cch, h_lat, w_lat = c_latent_ext.shape
    shape = (b, model.channels, h_lat, w_lat)

    # 构造起始噪声（从 c_latent_ext 出发加噪，然后再去噪）
    t = torch.ones((b,), device=model.device, dtype=torch.long) * (model.used_timesteps - 1)
    noise = torch.randn_like(c_latent_ext)
    x_T = model.q_sample(x_start=c_latent_ext, t=t, noise=noise)

    sampler_obj = SpacedSampler(model, var_type="fixed_small") if sampler == "ddpm" else DDIMSampler(model)

    if isinstance(sampler_obj, SpacedSampler):
        samples = sampler_obj.sample(
            steps,
            shape,
            cond,
            unconditional_guidance_scale=guidance_scale,
            unconditional_conditioning=None,
            cond_fn=None,
            x_T=x_T,
        )
    else:
        samples, _ = sampler_obj.sample(
            S=steps,
            batch_size=shape[0],
            shape=shape[1:],
            conditioning=cond,
            unconditional_conditioning=None,
            unconditional_guidance_scale=guidance_scale,
            x_T=x_T,
            eta=0.0,
        )

    # 告诉 VAE pad 后的原始宽度（用于内部融合逻辑）
    model._static_W0_pix = int(W_pad)

    # 解码 + 去 pad + 裁剪回工作分辨率（H0_in, W0_in）
    x_rec_full = model.decode_first_stage(samples)
    x_rec_pad = x_rec_full[..., :H_pad, :W_pad]
    x_rec = x_rec_pad[..., :H0_in, :W0_in]

    # 反归一化到 uint8
    x_out = ((x_rec + 1.0) / 2.0).clamp(0.0, 1.0)
    x_out = einops.rearrange(x_out, "b c h w -> b h w c")
    x_out = (x_out * 255.0).round().clamp(0, 255).to(torch.uint8).cpu().numpy()

    # 输出：
    # - original: 工作分辨率=原图分辨率 ⇒ 输出=原图分辨率
    # - train   : 工作分辨率=1024x512 ⇒ 输出=1024x512
    preds_final = [x_out[i] for i in range(n_samples)]

    return preds_final, float(bpp)


def parse_args() -> Namespace:
    p = ArgumentParser()

    p.add_argument("--ckpt", type=str, default="./weight/step2_1_12/last.ckpt")
    p.add_argument("--config", type=str, default="configs/model/rdeic.yaml")
    p.add_argument("--input", type=str, default="./dataset/test")
    p.add_argument("--output", type=str, default="output/with_text")
    p.add_argument("--sampler", type=str, default="ddpm", choices=["ddpm", "ddim"])
    p.add_argument("--steps", type=int, default=5)
    p.add_argument("--guidance_scale", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=231)
    p.add_argument("--device", type=str, default="cuda", choices=["cpu", "cuda"])
    p.add_argument("--extend_px", type=int, default=-1)

    p.add_argument(
        "--resize_mode",
        type=str,
        default="original",
        choices=["original", "train"],
        help="original: 按原图分辨率推理; train: 缩放到 1024x512 域内扩散与评估",
    )

    # 残差语义 JSON（可选）
    p.add_argument(
        "--residual_json",
        type=str,
        default=None,
        help="残差语义 JSON 文件路径；若为 None，则不使用文本残差信息",
    )

    return p.parse_args()


def main() -> None:
    args = parse_args()
    pl.seed_everything(args.seed)

    if args.device == "cpu":
        disable_xformers()

    cfg = OmegaConf.load(args.config)
    model: RDEIC = instantiate_from_config(cfg)

    ckpt = torch.load(args.ckpt, map_location="cpu")
    try:
        load_state_dict(model, ckpt, strict=True)
    except RuntimeError as e:
        print(f"[WARN] strict=True load failed: {e}")
        print("[WARN] retry with strict=False ...")
        load_state_dict(model, ckpt, strict=False)

    if hasattr(model, "preprocess_model"):
        model.preprocess_model.update(force=True)
    model.freeze()
    model.to(args.device).eval()

    # ★ 关键：禁用 VAE 分块处理（encode + decode 都走整图）
    if hasattr(model, "split_input_params"):
        print(f"[INFO] disable VAE split_input_params (was {model.split_input_params})")
        model.split_input_params = None

    # ===== 残差语义数据库（可选） =====
    res_db = None
    if args.residual_json is not None:
        if os.path.isfile(args.residual_json):
            res_db = ResidualPromptDB(args.residual_json)
            try:
                n_entries = len(getattr(res_db, "db", {}))
            except Exception:
                n_entries = -1
            print(f"[INFO] residual_json loaded: {args.residual_json} (entries={n_entries})")
        else:
            print(f"[WARN] residual_json not found: {args.residual_json}, 忽略文本残差")

    src = args.input
    if os.path.isdir(src):
        paths = list_image_files(src, follow_links=True)
    else:
        paths = [src]

    assert len(paths) > 0, f"No images found under {src}"
    os.makedirs(args.output, exist_ok=True)

    print(f"[INFO] device={args.device}")
    print(f"[INFO] steps={args.steps} sampler={args.sampler}")
    print(f"[INFO] extend_px={args.extend_px} (YAML 默认={getattr(model, 'extend_k_pix_default', 'N/A')})")
    print(f"[INFO] resize_mode={args.resize_mode}  (original=原图域, train=1024x512 域)")
    if res_db is not None:
        print("[INFO] text residual conditioning ENABLED")
    else:
        print("[INFO] text residual conditioning DISABLED (no residual_json)")

    bpps = []
    times = []   # 记录每张图耗时

    # 逐张推理
    for file_path in paths:
        img = Image.open(file_path).convert("RGB")
        x = np.array(img)

        rel = (
            os.path.relpath(file_path, start=src)
            if os.path.isdir(src)
            else os.path.basename(file_path)
        )
        save_path = os.path.join(
            args.output,
            os.path.splitext(rel)[0] + ".png",
        )

        parent_path, stem, _ = get_file_name_parts(save_path)
        stream_parent_path = os.path.join(parent_path, "data")
        stream_path = os.path.join(stream_parent_path, stem)

        os.makedirs(parent_path, exist_ok=True)
        os.makedirs(stream_parent_path, exist_ok=True)

        # 文本条件（base + residual，可选）
        if res_db is not None:
            img_abs = os.path.abspath(file_path)
            residual_prompt = res_db.get_full_prompt(img_abs)
            if residual_prompt:
                c_base = model.get_learned_conditioning([""])
                c_res  = model.get_learned_conditioning([residual_prompt])
                c_crossattn = [c_base, c_res]
            else:
                c_crossattn = [model.get_learned_conditioning([""])]
        else:
            c_crossattn = [model.get_learned_conditioning([""])]

        # ====== 开始计时 ======
        t0 = time.time()

        preds_final, bpp = process(
            model=model,
            imgs=[x],
            sampler=args.sampler,
            steps=args.steps,
            stream_path=stream_path,
            guidance_scale=args.guidance_scale,
            c_crossattn=c_crossattn,
            extend_px=args.extend_px,
            resize_mode=args.resize_mode,
        )

        img_final = preds_final[0]
        Image.fromarray(img_final).save(save_path)

        t1 = time.time()
        dt = t1 - t0   # 单张耗时
        times.append(dt)

        bpps.append(bpp)
        print(f"[SAVE] {save_path}  bpp={bpp:.4f}  time={dt:.3f}s")

    if bpps:
        avg_bpp = float(sum(bpps) / len(bpps))
        print(f"[DONE] avg bpp: {avg_bpp:.4f}")

    if times:
        avg_time = float(sum(times) / len(times))
        print(f"[DONE] avg time per image: {avg_time:.3f}s")


if __name__ == "__main__":
    main()
