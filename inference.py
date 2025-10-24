# inference.py
from typing import List, Tuple
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from argparse import ArgumentParser, Namespace
import numpy as np
import torch, einops, pytorch_lightning as pl
from PIL import Image
from omegaconf import OmegaConf

from ldm.xformers_state import disable_xformers
from model.spaced_sampler_relay import SpacedSampler
from model.ddim_sampler_relay import DDIMSampler
from model.rdeic import RDEIC
from utils.common import instantiate_from_config, load_state_dict
from utils.file import list_image_files, get_file_name_parts
from utils.image import pad


# =================== 工具：像素域环绕扩展 & 收尾 ===================

def hwrap_extend_np(img: np.ndarray, k: int) -> np.ndarray:
    """像素域左右各拼接 k 列（HWC, uint8）。"""
    if k <= 0: return img
    H, W, C = img.shape
    k = min(int(k), W // 2)
    left  = img[:, -k:, :]
    right = img[:, :k,  :]
    return np.concatenate([left, img, right], axis=1)

@torch.no_grad()
def _make_window(k: int, device, dtype, mode="cosine"):
    j = torch.arange(k, device=device, dtype=dtype)
    if mode == "linear":
        return (j / (k - 1)).view(1,1,1,k) if k > 1 else torch.ones(1,1,1,1, device=device, dtype=dtype)
    w = (1 - torch.cos(torch.pi * (j / max(1, (k - 1))))) * 0.5
    return w.view(1,1,1,k)

@torch.no_grad()
def seam_close_one_side(img: torch.Tensor, k: int = 12, window: str = "cosine", side: str = "left") -> torch.Tensor:
    """
    单侧融合（推荐用于 360 播放器）：只在 left 或 right 一侧做 cross-fade，另一侧保持原样。
    img: [B,C,H,W] in [0,1] 或 [-1,1]
    """
    B, C, H, W = img.shape
    if W < 2 or k <= 0: return img
    k = max(1, min(int(k), W // 2))
    out = img.clone()
    w = _make_window(k, img.device, img.dtype, window)

    if side == "left":
        L = img[..., :k]      # [B,C,H,k]
        R = img[..., -k:]     # 对侧带
        L_new = (1 - w) * L + w * R
        out[..., :k] = L_new
    else:
        L = img[..., :k]
        R = img[..., -k:]
        w_rev = torch.flip(w, dims=[-1])
        R_new = (1 - w_rev) * R + w_rev * L
        out[..., -k:] = R_new
    return out

@torch.no_grad()
def seam_close_symmetric(img: torch.Tensor, k: int = 16, window: str = "cosine") -> torch.Tensor:
    """
    对称 cross-fade（不推荐给播放器，因为会产生“两条”视觉带）。
    """
    B, C, H, W = img.shape
    if W < 2 or k <= 0: return img
    k = max(1, min(int(k), W // 2))
    L, R = img[..., :k], img[..., -k:]
    w     = _make_window(k, img.device, img.dtype, window)
    w_rev = torch.flip(w, dims=[-1])
    out = img.clone()
    out[..., :k]  = (1 - w)    * L + w     * R
    out[..., -k:] = (1 - w_rev)* R + w_rev * L
    return out

@torch.no_grad()
def average_1px_rgb(img: torch.Tensor) -> torch.Tensor:
    """严格周期闭合：把最左/最右 1 列取均值写回两侧。"""
    if img.ndim != 4 or img.shape[-1] < 2: return img
    c0, c1 = img[..., :1], img[..., -1:]
    avg = 0.5 * (c0 + c1)
    out = img.clone()
    out[..., :1]  = avg
    out[..., -1:] = avg
    return out
# ============================================================


@torch.no_grad()
def process(
    model: RDEIC,
    imgs: List[np.ndarray],
    sampler: str,
    steps: int,
    stream_path: str,
    guidance_scale: float,
    c_crossattn: List[torch.Tensor],
    extend_px: int,
    close_mode: str,
    close_k: int,
    close_window: str,
    do_average_1px: bool,
) -> Tuple[List[np.ndarray], float]:

    n_samples = len(imgs)
    sampler_obj = SpacedSampler(model, var_type="fixed_small") if sampler == "ddpm" else DDIMSampler(model)

    # 1) pad→环绕扩展（像素域）
    imgs_pad = [pad(x, scale=64) for x in imgs]
    k = int(extend_px)
    if k % 64 != 0:  # 为了保证后面 VAE/UNet 步幅对齐
        k = max(0, (k // 64) * 64)
    imgs_ext = [hwrap_extend_np(x, k) for x in imgs_pad]

    control = torch.tensor(np.stack(imgs_ext) / 255.0, dtype=torch.float32, device=model.device).clamp_(0, 1)
    control = einops.rearrange(control, "n h w c -> n c h w").contiguous()

    H, W_ext = control.shape[-2:]
    W_orig   = imgs_pad[0].shape[1]
    assert W_ext == W_orig + 2 * k

    # 2) 条件压缩 + 采样
    bpp = model.apply_condition_compress(control, stream_path, H, W_ext)
    c_latent, guide_hint = model.apply_condition_decompress(stream_path)
    cond = {"c_latent": [c_latent], "c_crossattn": c_crossattn, "guide_hint": guide_hint}

    shape = (n_samples, 4, H // 8, W_ext // 8)
    t = torch.ones((n_samples,), device=model.device, dtype=torch.long) * model.used_timesteps - 1
    x_T = model.q_sample(x_start=c_latent, t=t, noise=torch.randn(shape, device=model.device, dtype=torch.float32))

    if isinstance(sampler_obj, SpacedSampler):
        samples = sampler_obj.sample(
            steps, shape, cond,
            unconditional_guidance_scale=guidance_scale,
            unconditional_conditioning=None, cond_fn=None, x_T=x_T
        )
    else:
        samples, _ = sampler_obj.sample(
            S=steps, batch_size=shape[0], shape=shape[1:],
            conditioning=cond, unconditional_conditioning=None,
            unconditional_guidance_scale=guidance_scale, x_T=x_T, eta=0
        )

    # 3) 解码 & 裁回中心（回到 pad 后原宽度）
    x = model.decode_first_stage(samples)  # [-1,1], [B,3,H,W_ext]
    if k > 0:
        x = x[:, :, :, k:-k]  # 裁回
        if close_mode == "one_side" and close_k > 0:
            x = seam_close_one_side(x, k=int(close_k), window=close_window, side="left")
        elif close_mode == "symmetric" and close_k > 0:
            x = seam_close_symmetric(x, k=int(close_k), window=close_window)
    if do_average_1px:
        x = average_1px_rgb(x)

    # 4) to [0,1] & numpy
    x = ((x + 1) / 2).clamp(0, 1)
    x = (einops.rearrange(x, "b c h w -> b h w c") * 255.0).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
    preds = [x[i] for i in range(n_samples)]
    return preds, float(bpp)


def parse_args() -> Namespace:
    p = ArgumentParser()
    p.add_argument("--ckpt",
        default="/opt/dev/RDEIC-main/logs/independent/2_2/lightning_logs/version_0/checkpoints/last.ckpt",
        type=str)
    p.add_argument("--config", default="configs/model/rdeic.yaml", type=str)
    p.add_argument("--input", type=str, default="/opt/dev/RDEIC-main/assets/demo_images")
    p.add_argument("--sampler", type=str, default="ddpm", choices=["ddpm", "ddim"])
    p.add_argument("--steps", default=5, type=int)
    p.add_argument("--guidance_scale", default=1.0, type=float)
    p.add_argument("--output", type=str, default="results/")
    p.add_argument("--seed", type=int, default=231)
    p.add_argument("--device", type=str, default="cuda", choices=["cpu", "cuda"])

    # 预处理冗余像素 & 收尾
    p.add_argument("--extend_px", type=int, default=128, help="左右冗余带宽（像素），建议 64/128")
    p.add_argument("--close_mode", type=str, default="one_side", choices=["off","one_side","symmetric"])
    p.add_argument("--close_k", type=int, default=12, help="收尾带宽（像素域），8~16 较合适")
    p.add_argument("--close_window", type=str, default="cosine", choices=["linear","cosine"])
    p.add_argument("--average_1px", action="store_true", default=True, help="是否做严格周期 1px 闭合")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    pl.seed_everything(args.seed)
    if args.device == "cpu":
        disable_xformers()

    # 读取配置并在实例化前强制禁用 seam_blend
    cfg = OmegaConf.load(args.config)
    try:
        cfg.params.seam_blend.enabled = False
        cfg.params.seam_blend.runtime_apply = False
        cfg.params.seam_blend_train.enabled = False
        cfg.params.seam_blend_train.online_joint = False
    except Exception:
        pass

    model: RDEIC = instantiate_from_config(cfg)

    # 加载 ckpt：若 seam_blend 缺失则自动降级为 non-strict
    ckpt = torch.load(args.ckpt, map_location="cpu")
    try:
        load_state_dict(model, ckpt, strict=True)
    except RuntimeError as e:
        if "seam_blend" in str(e):
            print("[WARN] seam_blend weights missing; loading with strict=False.")
            load_state_dict(model, ckpt, strict=False)
        else:
            raise

    # 确保推理期不触发小网
    if hasattr(model, "seam_enabled"):        model.seam_enabled = False
    if hasattr(model, "seam_runtime_apply"):  model.seam_runtime_apply = False

    model.preprocess_model.update(force=True)
    model.freeze()
    model.to(args.device).eval()

    c_crossattn = [model.get_learned_conditioning([""])]

    # 收集输入
    src = args.input
    paths = list_image_files(src, follow_links=True) if os.path.isdir(src) else [src]
    assert len(paths) > 0, f"No images found under {src}"
    os.makedirs(args.output, exist_ok=True)

    print(f"[INFO] steps={args.steps} sampler={args.sampler}")
    print(f"[INFO] extend_px={args.extend_px}  close_mode={args.close_mode} k={args.close_k} window={args.close_window} avg1px={args.average_1px}")

    bpps = []
    for file_path in paths:
        img = Image.open(file_path).convert("RGB")
        x = np.array(img)

        rel = os.path.relpath(file_path, start=src) if os.path.isdir(src) else os.path.basename(file_path)
        save_path = os.path.join(args.output, os.path.splitext(rel)[0] + ".png")
        parent_path, stem, _ = get_file_name_parts(save_path)
        stream_parent_path = os.path.join(parent_path, 'data')
        stream_path = os.path.join(stream_parent_path, stem)
        os.makedirs(parent_path, exist_ok=True)
        os.makedirs(stream_parent_path, exist_ok=True)

        preds, bpp = process(
            model, [x], steps=args.steps, sampler=args.sampler,
            stream_path=stream_path, guidance_scale=args.guidance_scale, c_crossattn=c_crossattn,
            extend_px=args.extend_px, close_mode=args.close_mode, close_k=args.close_k,
            close_window=args.close_window, do_average_1px=bool(args.average_1px)
        )
        pred_pad = preds[0]
        pred = pred_pad[:img.height, :img.width, :]  # 去掉最初 pad

        bpps.append(bpp)
        Image.fromarray(pred).save(save_path)
        print(f"[SAVE] {save_path}  bpp={bpp:.4f}")

    print(f"[DONE] avg bpp: {float(sum(bpps)/len(bpps)):.4f}")


if __name__ == "__main__":
    main()
