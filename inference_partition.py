import os
from typing import List, Tuple
from argparse import ArgumentParser, Namespace
import time   # ✅ 新增

import numpy as np
import torch
import einops
import pytorch_lightning as pl
from PIL import Image
from omegaconf import OmegaConf

from ldm.xformers_state import disable_xformers
from model.spaced_sampler_relay import SpacedSampler
from model.ddim_sampler_relay import DDIMSampler
from model.rdeic import RDEIC
from utils.image import pad
from utils.common import instantiate_from_config, load_state_dict
from utils.file import list_image_files, get_file_name_parts

# ⬇️ 引入：ERP 水平环绕卷积替换
# from utils.erp_wrap import enable_horizontal_circular_padding
from utils.feature_hwrap import enable_feature_hwrap
import torch.nn.functional as F


@torch.no_grad()
def process(
        model: RDEIC,
        imgs: List[np.ndarray],
        sampler: str,
        steps: int,
        stream_path: str,
        guidance_scale: float,
        c_crossattn: List[torch.Tensor],
        extra_hwrap_px: int = 1,
) -> Tuple[List[np.ndarray], float]:
    n_samples = len(imgs)
    if sampler == "ddpm":
        sampler = SpacedSampler(model, var_type="fixed_small")
    else:
        sampler = DDIMSampler(model)
    control = torch.tensor(np.stack(imgs) / 255.0, dtype=torch.float32, device=model.device).clamp_(0, 1)
    control = einops.rearrange(control, "n h w c -> n c h w").contiguous()
    k = max(1, int(extra_hwrap_px))
    control = F.pad(control, (k, k, 0, 0), mode="circular")

    height, width = control.size(-2), control.size(-1)
    bpp = model.apply_condition_compress(control, stream_path, height, width)
    c_latent, guide_hint = model.apply_condition_decompress(stream_path)
    cond = {
        "c_latent": [c_latent],
        "c_crossattn": c_crossattn,
        "guide_hint": guide_hint
    }

    shape = (n_samples, 4, height // 8, width // 8)
    x_T = torch.randn(shape, device=model.device, dtype=torch.float32)
    noise = torch.randn(shape, device=model.device, dtype=torch.float32)
    t = torch.ones((n_samples,)).long().to(model.device) * model.used_timesteps - 1
    x_T = model.q_sample(x_start=c_latent, t=t, noise=noise)
    if isinstance(sampler, SpacedSampler):
        samples = sampler.sample(
            steps, shape, cond,
            unconditional_guidance_scale=guidance_scale,
            unconditional_conditioning=None,
            cond_fn=None, x_T=x_T
        )
    else:
        sampler: DDIMSampler
        samples, _ = sampler.sample(
            S=steps, batch_size=shape[0], shape=shape[1:],
            conditioning=cond, unconditional_conditioning=None,
            unconditional_guidance_scale=guidance_scale,
            x_T=x_T, eta=0
        )

    x_samples = model.decode_first_stage(samples)
    x_samples = ((x_samples + 1) / 2).clamp(0, 1)

    x_samples = (einops.rearrange(x_samples, "b c h w -> b h w c") * 255).cpu().numpy().clip(0, 255).astype(np.uint8)

    preds = [x_samples[i] for i in range(n_samples)]

    return preds, bpp


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("--ckpt_sd", default='./weight/v2-1_512-ema-pruned.ckpt', type=str)
    parser.add_argument("--ckpt_cc", default='path to checkpoint file of compression and control module', type=str)
    parser.add_argument("--config", default='configs/model/rdeic.yaml', type=str)
    parser.add_argument("--input", type=str, default='path to input images')
    parser.add_argument("--sampler", type=str, default="ddpm", choices=["ddpm", "ddim"])
    parser.add_argument("--steps", default=2, type=int)
    parser.add_argument("--guidance_scale", default=1.0, type=float)
    parser.add_argument("--output", type=str, default='results/')
    parser.add_argument("--seed", type=int, default=231)
    parser.add_argument("--device", type=str, default="cuda", choices=["cpu", "cuda"])
    parser.add_argument("--erp_wrap", type=int, default=1)
    parser.add_argument("--erp_only_3x3", type=int, default=1)
    parser.add_argument("--extra_hwrap_px", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pl.seed_everything(args.seed)

    if args.device == "cpu":
        disable_xformers()

    model: RDEIC = instantiate_from_config(OmegaConf.load(args.config))

    if bool(args.erp_wrap):
        replaced = enable_feature_hwrap(model, only_3x3=True, v_mode="constant", v_value=0.0)
        print(f"[HFeaturePad] replaced {replaced} Conv2d (feature-level wrap)")

    ckpt_sd = torch.load(args.ckpt_sd, map_location="cpu")['state_dict']
    ckpt_lc = torch.load(args.ckpt_cc, map_location="cpu")['state_dict']
    ckpt_sd.update(ckpt_lc)
    load_state_dict(model, ckpt_sd, strict=False)

    model.preprocess_model.update(force=True)
    model.freeze()
    model.to(args.device)

    bpps = []
    times = []   # ✅ 记录耗时
    assert os.path.isdir(args.input)

    c_crossattn = [model.get_learned_conditioning([""])]

    print(f"sampling {args.steps} steps using {args.sampler} sampler")
    for file_path in list_image_files(args.input, follow_links=True):
        start_time = time.time()

        img = Image.open(file_path).convert("RGB")
        x = pad(np.array(img), scale=64)

        save_path = os.path.join(args.output, os.path.relpath(file_path, args.input))
        parent_path, stem, _ = get_file_name_parts(save_path)
        stream_parent_path = os.path.join(parent_path, 'data')
        save_path = os.path.join(parent_path, f"{stem}.png")
        stream_path = os.path.join(stream_parent_path, f"{stem}")

        os.makedirs(parent_path, exist_ok=True)
        os.makedirs(stream_parent_path, exist_ok=True)

        preds, bpp = process(
            model, [x], steps=args.steps, sampler=args.sampler,
            stream_path=stream_path, guidance_scale=args.guidance_scale,c_crossattn=c_crossattn, extra_hwrap_px=args.extra_hwrap_px
        )
        pred = preds[0]
        bpps.append(bpp)

        pred = pred[:img.height, :img.width, :]
        Image.fromarray(pred).save(save_path)

        elapsed = time.time() - start_time
        times.append(elapsed)   # ✅ 保存每张耗时
        print(f"save to {save_path}, bpp {bpp:.6f}, time {elapsed:.2f}s")

    avg_bpp = sum(bpps) / len(bpps)
    avg_time = sum(times) / len(times)
    total_time = sum(times)
    print(f'avg bpp: {avg_bpp:.6f}')
    print(f'avg time: {avg_time:.2f}s per image, total time: {total_time:.2f}s')


if __name__ == "__main__":
    main()

