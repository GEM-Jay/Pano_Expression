# -*- coding: utf-8 -*-
"""
RDEIC (seam aligned, band-free) — 预处理扩宽版（No latent extend, No tile decode）

要点（与要求严格对齐）：
- 训练/采样阶段：假设输入已在【预处理阶段】完成“先滚动(仅train)再右侧拼接”的扩宽。此文件不在 latent 做任何扩宽。
- VAE 解码：整幅一次 decode（不瓦片），随后在像素域做【单侧融合（只写左带）】：
    * 真实拼接像素 K_true = W_ext - W0     （由预处理决定，建议 ≈ ceil(W0/10) 并按 64 对齐）
    * 融合带宽 K_blend = floor(K_true / 2)  （“最后融合取真正拼接像素的一半”）
    * 左带 [0:K_blend] 与参考带 [W0:W0+K_blend] 按线性窗（可选小网缩放）融合；右端与尾巴不写回。
- latent 阶段不再扩宽、不再 roll。
- 日志/验证流程与原版一致；损失与指标均对齐到裁回 W0 后的成品。
- 重要修复：像素融合避免任何 in-place 赋值，防止 AMP/Autograd 报错（AsStridedBackward + version bump）。
"""

import os
import math
from typing import Mapping, Any, Tuple, Optional

import numpy as np
import pyiqa

import torch
import torch as th
import torch.nn as nn
import torch.nn.functional as F

from pathlib import Path
from PIL import Image

# ---- 可选：开启 UNet 水平环绕卷积（若 utils 支持）----
def enable_feature_hwrap_for_unet(*args, **kwargs):
    from utils.feature_hwrap import enable_feature_hwrap_for_unet as _impl
    return _impl(*args, **kwargs)

from pytorch_lightning.utilities.types import EPOCH_OUTPUT
from utils.utils import *

from ldm.modules.diffusionmodules.util import (
    conv_nd,
    linear,
    zero_module,
    timestep_embedding,
    checkpoint
)
from .spaced_sampler_relay import SpacedSampler
from ldm.modules.attention import SpatialTransformer
from ldm.models.diffusion.ddpm import LatentDiffusion
from ldm.modules.diffusionmodules.openaimodel import (
    UNetModel,
    TimestepEmbedSequential,
    ResBlock as ResBlock_orig,
    Downsample,
    Upsample,
    AttentionBlock,
    TimestepBlock
)
from ldm.util import log_txt_as_img, exists, instantiate_from_config, default
from .lpips import LPIPS


# ------------------------ 像素端融合权重小网（可选） ------------------------
class BlendWeightNet(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        ch = in_channels * 2  # L || R_ref
        self.conv = nn.Conv2d(ch, 1, kernel_size=1, bias=True)
        nn.init.zeros_(self.conv.weight)
        nn.init.constant_(self.conv.bias, 0.0)

    def forward(self, L: torch.Tensor, R_ref: torch.Tensor) -> torch.Tensor:
        # 用带宽方向均值做权重估计，输出在 (0,1)
        Lm = L.mean(dim=2, keepdim=True)       # [B,C,1,K]
        Rm = R_ref.mean(dim=2, keepdim=True)   # [B,C,1,K]
        x = torch.cat([Lm, Rm], dim=1)         # [B,2C,1,K]
        a = torch.sigmoid(self.conv(x))        # [B,1,1,K]
        return a


# ------------------------ 控制分支（与原版一致） ------------------------
class NoiseEstimator(nn.Module):
    def __init__(
        self,
        image_size,
        in_channels,
        model_channels,
        out_channels,
        hint_channels,
        num_res_blocks,
        attention_resolutions,
        dropout=0,
        channel_mult=(1, 2, 4, 8),
        conv_resample=True,
        dims=2,
        use_checkpoint=False,
        use_fp16=False,
        num_heads=-1,
        num_head_channels=-1,
        num_heads_upsample=-1,
        use_scale_shift_norm=False,
        resblock_updown=False,
        use_new_attention_order=False,
        use_spatial_transformer=False,
        transformer_depth=1,
        context_dim=None,
        n_embed=None,
        legacy=False,
        use_linear_in_transformer=False,
        control_model_ratio=1.0,
        learn_embedding=True,
        control_scale=1.0
    ):
        super().__init__()
        self.learn_embedding = learn_embedding
        self.control_model_ratio = control_model_ratio
        self.out_channels = out_channels
        self.dims = 2
        self.model_channels = model_channels
        self.control_scale = control_scale

        base_model = UNetModel(
            image_size=image_size, in_channels=in_channels, model_channels=model_channels,
            out_channels=out_channels, num_res_blocks=num_res_blocks,
            attention_resolutions=attention_resolutions, dropout=dropout, channel_mult=channel_mult,
            conv_resample=conv_resample, dims=dims, use_checkpoint=use_checkpoint,
            use_fp16=use_fp16, num_heads=num_heads, num_head_channels=num_head_channels,
            num_heads_upsample=num_heads_upsample, use_scale_shift_norm=use_scale_shift_norm,
            resblock_updown=resblock_updown, use_new_attention_order=use_new_attention_order,
            use_spatial_transformer=use_spatial_transformer, transformer_depth=transformer_depth,
            context_dim=context_dim, n_embed=n_embed, legacy=legacy,
            use_linear_in_transformer=use_linear_in_transformer,
        )
        self.control_model = ControlModule(
            image_size=image_size, in_channels=in_channels, model_channels=model_channels, hint_channels=hint_channels,
            out_channels=out_channels, num_res_blocks=num_res_blocks,
            attention_resolutions=attention_resolutions, dropout=dropout, channel_mult=channel_mult,
            conv_resample=conv_resample, dims=dims, use_checkpoint=use_checkpoint,
            use_fp16=use_fp16, num_heads=num_heads, num_head_channels=num_head_channels,
            num_heads_upsample=num_heads_upsample,
            use_scale_shift_norm=use_scale_shift_norm,
            resblock_updown=resblock_updown, use_new_attention_order=use_new_attention_order,
            use_spatial_transformer=use_spatial_transformer, transformer_depth=transformer_depth,
            context_dim=context_dim, n_embed=n_embed, legacy=legacy,
            use_linear_in_transformer=use_linear_in_transformer,
            control_model_ratio=control_model_ratio,
        )

        self.enc_zero_convs_out = nn.ModuleList([])
        self.middle_block_out = None
        self.dec_zero_convs_out = nn.ModuleList([])

        ch_inout_ctr = {'enc': [], 'mid': [], 'dec': []}
        ch_inout_base = {'enc': [], 'mid': [], 'dec': []}

        for module in self.control_model.input_blocks:
            if isinstance(module[0], nn.Conv2d):
                ch_inout_ctr['enc'].append((module[0].in_channels, module[0].out_channels))
            elif isinstance(module[0], (ResBlock, ResBlock_orig)):
                ch_inout_ctr['enc'].append((module[0].channels, module[0].out_channels))
            elif isinstance(module[0], Downsample):
                ch_inout_ctr['enc'].append((module[0].channels, module[-1].out_channels))

        for module in base_model.input_blocks:
            if isinstance(module[0], nn.Conv2d):
                ch_inout_base['enc'].append((module[0].in_channels, module[0].out_channels))
            elif isinstance(module[0], (ResBlock, ResBlock_orig)):
                ch_inout_base['enc'].append((module[0].channels, module[0].out_channels))
            elif isinstance(module[0], Downsample):
                ch_inout_base['enc'].append((module[0].channels, module[-1].out_channels))

        ch_inout_ctr['mid'].append(
            (self.control_model.middle_block[0].channels, self.control_model.middle_block[-1].out_channels))
        ch_inout_base['mid'].append((base_model.middle_block[0].channels, base_model.middle_block[-1].out_channels))

        for module in base_model.output_blocks:
            if isinstance(module[0], nn.Conv2d):
                ch_inout_base['dec'].append((module[0].in_channels, module[0].out_channels))
            elif isinstance(module[0], (ResBlock, ResBlock_orig)):
                ch_inout_base['dec'].append((module[0].channels, module[0].out_channels))
            elif isinstance(module[-1], Upsample):
                ch_inout_base['dec'].append((module[0].channels, module[-1].out_channels))

        self.ch_inout_ctr = ch_inout_ctr
        self.ch_inout_base = ch_inout_base

        self.middle_block_out = self.make_zero_conv(ch_inout_ctr['mid'][-1][1], ch_inout_base['mid'][-1][1])

        self.dec_zero_convs_out.append(
            self.make_zero_conv(ch_inout_ctr['enc'][-1][1], ch_inout_base['mid'][-1][1])
        )
        for i in range(1, len(ch_inout_ctr['enc'])):
            self.dec_zero_convs_out.append(
                self.make_zero_conv(ch_inout_ctr['enc'][-(i + 1)][1], ch_inout_base['dec'][i - 1][1])
            )
        for i in range(len(ch_inout_ctr['enc'])):
            self.enc_zero_convs_out.append(self.make_zero_conv(
                in_channels=ch_inout_ctr['enc'][i][1], out_channels=ch_inout_base['enc'][i][1])
            )

        scale_list = [1.] * len(self.enc_zero_convs_out) + [1.] + [1.] * len(self.dec_zero_convs_out)
        self.register_buffer('scale_list', torch.tensor(scale_list) * self.control_scale)

    def make_zero_conv(self, in_channels, out_channels=None):
        self.in_channels = in_channels
        self.out_channels = out_channels or in_channels
        return TimestepEmbedSequential(
            zero_module(conv_nd(self.dims, in_channels, out_channels, 1, padding=0))
        )

    def forward(self, x, guide_hint, timesteps, context, base_model, **kwargs):
        t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False)
        emb = self.control_model.time_embed(t_emb)
        emb_base = base_model.time_embed(t_emb)

        h_base = x.type(base_model.dtype)
        h_ctr = torch.cat((h_base, guide_hint), dim=1)
        hs_base = []
        hs_ctr = []
        it_enc_convs_out = iter(self.enc_zero_convs_out)
        it_dec_convs_out = iter(self.dec_zero_convs_out)
        scales = iter(self.scale_list)

        # encoder
        for module_base, module_ctr in zip(base_model.input_blocks, self.control_model.input_blocks):
            h_base = module_base(h_base, emb_base, context)
            h_ctr = module_ctr(h_ctr, emb, context)
            h_base = h_base + next(it_enc_convs_out)(h_ctr, emb) * next(scales)
            hs_base.append(h_base)
            hs_ctr.append(h_ctr)

        # middle
        h_base = base_model.middle_block(h_base, emb_base, context)
        h_ctr = self.control_model.middle_block(h_ctr, emb, context)
        h_base = h_base + self.middle_block_out(h_ctr, emb) * next(scales)

        # decoder
        for module_base in base_model.output_blocks:
            h_base = h_base + next(it_dec_convs_out)(hs_ctr.pop(), emb) * next(scales)
            h_base = th.cat([h_base, hs_base.pop()], dim=1)
            h_base = module_base(h_base, emb_base, context)

        return base_model.out(h_base)

    def forward_unconditional(self, x, timesteps, context, base_model, **kwargs):
        t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False)
        emb_base = base_model.time_embed(t_emb)

        h_base = x.type(base_model.dtype)
        hs_base = []
        for module_base in base_model.input_blocks:
            h_base = module_base(h_base, emb_base, context)
            hs_base.append(h_base)
        h_base = base_model.middle_block(h_base, emb_base, context)
        for module_base in base_model.output_blocks:
            h_base = th.cat([h_base, hs_base.pop()], dim=1)
            h_base = module_base(h_base, emb_base, context)
        return base_model.out(h_base)


class ControlModule(nn.Module):
    def __init__(
        self,
        image_size,
        in_channels,
        model_channels,
        hint_channels,
        out_channels,
        num_res_blocks,
        attention_resolutions,
        dropout=0,
        channel_mult=(1, 2, 4, 8),
        conv_resample=True,
        dims=2,
        num_classes=None,
        use_checkpoint=False,
        use_fp16=False,
        num_heads=-1,
        num_head_channels=-1,
        num_heads_upsample=-1,
        use_scale_shift_norm=False,
        resblock_updown=False,
        use_new_attention_order=False,
        use_spatial_transformer=False,    # custom transformer support
        transformer_depth=1,              # custom transformer support
        context_dim=None,                 # custom transformer support
        n_embed=None,                     # custom support for prediction of discrete ids into codebook of first stage vq model
        legacy=True,
        disable_self_attentions=None,
        num_attention_blocks=None,
        disable_middle_self_attn=False,
        use_linear_in_transformer=False,
        control_model_ratio=1.0,
    ):
        super().__init__()
        if use_spatial_transformer:
            assert context_dim is not None

        if context_dim is not None:
            assert use_spatial_transformer
            from omegaconf.listconfig import ListConfig
            if type(context_dim) == ListConfig:
                context_dim = list(context_dim)

        if num_heads_upsample == -1:
            num_heads_upsample = num_heads
        if num_heads == -1:
            assert num_head_channels != -1
        if num_head_channels == -1:
            assert num_heads != -1

        self.image_size = image_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        if isinstance(num_res_blocks, int):
            self.num_res_blocks = len(channel_mult) * [num_res_blocks]
        else:
            if len(num_res_blocks) != len(channel_mult):
                raise ValueError("num_res_blocks length mismatch")
            self.num_res_blocks = num_res_blocks
        if disable_self_attentions is not None:
            assert len(disable_self_attentions) == len(self.num_res_blocks)
        if num_attention_blocks is not None:
            assert len(num_attention_blocks) == len(self.num_res_blocks)

        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.num_classes = num_classes
        self.use_checkpoint = use_checkpoint
        self.dtype = th.float16 if use_fp16 else th.float32
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        self.num_heads_upsample = num_heads_upsample
        self.predict_codebook_ids = n_embed is not None

        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            nn.SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )

        model_channels = int(model_channels * control_model_ratio)
        self.model_channels = model_channels
        self.control_model_ratio = control_model_ratio

        if self.num_classes is not None:
            if isinstance(self.num_classes, int):
                self.label_emb = nn.Embedding(num_classes, time_embed_dim)
            elif self.num_classes == "continuous":
                self.label_emb = nn.Linear(1, time_embed_dim)
            else:
                raise ValueError()

        self.input_blocks = nn.ModuleList(
            [
                TimestepEmbedSequential(
                    conv_nd(dims, in_channels+hint_channels, model_channels, 3, padding=1)
                )
            ]
        )
        self._feature_size = model_channels
        input_block_chans = [model_channels]
        ch = model_channels
        ds = 1
        for level, mult in enumerate(channel_mult):
            for nr in range(self.num_res_blocks[level]):
                layers = [
                    ResBlock(
                        ch,
                        time_embed_dim,
                        dropout,
                        out_channels=mult * model_channels,
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = mult * model_channels
                if ds in attention_resolutions:
                    if num_head_channels == -1:
                        dim_head = ch // num_heads
                    else:
                        num_head_channels = find_denominator(ch, self.num_head_channels)
                        num_heads = ch // num_head_channels
                        dim_head = num_head_channels

                    if not exists(num_attention_blocks) or nr < num_attention_blocks[level]:
                        layers.append(
                            AttentionBlock(
                                ch,
                                use_checkpoint=use_checkpoint,
                                num_heads=num_heads,
                                num_head_channels=dim_head,
                                use_new_attention_order=use_new_attention_order,
                            ) if not use_spatial_transformer else SpatialTransformer(
                                ch, num_heads, dim_head, depth=transformer_depth, context_dim=context_dim,
                                disable_self_attn=False, use_linear=use_linear_in_transformer,
                                use_checkpoint=use_checkpoint
                            )
                        )
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch
                input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                out_ch = ch
                self.input_blocks.append(
                    TimestepEmbedSequential(
                        Downsample(ch, conv_resample, dims=dims, out_channels=out_ch)
                    )
                )
                ch = out_ch
                input_block_chans.append(ch)
                ds *= 2
                self._feature_size += ch

        if num_head_channels == -1:
            dim_head = ch // num_heads
        else:
            num_heads = ch // num_head_channels
            dim_head = num_head_channels

        self.middle_block = TimestepEmbedSequential(
            ResBlock(ch, time_embed_dim, dropout, out_channels=ch, dims=dims,
                     use_checkpoint=use_checkpoint, use_scale_shift_norm=use_scale_shift_norm),
            AttentionBlock(ch, use_checkpoint=use_checkpoint, num_heads=num_heads,
                           num_head_channels=dim_head, use_new_attention_order=use_new_attention_order)
            if not use_spatial_transformer else SpatialTransformer(
                ch, num_heads, dim_head, depth=transformer_depth, context_dim=context_dim,
                disable_self_attn=False, use_linear=use_linear_in_transformer, use_checkpoint=use_checkpoint
            ),
            ResBlock(ch, time_embed_dim, dropout, dims=dims,
                     use_checkpoint=use_checkpoint, use_scale_shift_norm=use_scale_shift_norm),
        )
        self._feature_size += ch


def find_denominator(number, start):
    if start >= number: return number
    while start != 0:
        residual = number % start
        if residual == 0: return start
        start -= 1


def normalization(channels):
    return GroupNorm_leq32(find_denominator(channels, 32), channels)


class GroupNorm_leq32(nn.GroupNorm):
    def forward(self, x):
        return super().forward(x.float()).type(x.dtype)


class ResBlock(TimestepBlock):
    def __init__(self, channels, emb_channels, dropout, out_channels=None, use_conv=False,
                 use_scale_shift_norm=False, dims=2, use_checkpoint=False, up=False, down=False):
        super().__init__()
        self.channels = channels
        self.emb_channels = emb_channels
        self.dropout = dropout
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_checkpoint = use_checkpoint
        self.use_scale_shift_norm = use_scale_shift_norm

        self.in_layers = nn.Sequential(
            normalization(channels), nn.SiLU(), conv_nd(dims, channels, self.out_channels, 3, padding=1))
        self.updown = up or down
        if up:
            self.h_upd = Upsample(channels, False, dims); self.x_upd = Upsample(channels, False, dims)
        elif down:
            self.h_upd = Downsample(channels, False, dims); self.x_upd = Downsample(channels, False, dims)
        else:
            self.h_upd = self.x_upd = nn.Identity()
        self.emb_layers = nn.Sequential(
            nn.SiLU(), linear(emb_channels, 2 * self.out_channels if use_scale_shift_norm else self.out_channels))
        self.out_layers = nn.Sequential(
            normalization(self.out_channels), nn.SiLU(), nn.Dropout(p=dropout),
            zero_module(conv_nd(dims, self.out_channels, self.out_channels, 3, padding=1)))
        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        elif use_conv:
            self.skip_connection = conv_nd(dims, channels, self.out_channels, 3, padding=1)
        else:
            self.skip_connection = conv_nd(dims, channels, self.out_channels, 1)

    def forward(self, x, emb):
        return checkpoint(self._forward, (x, emb), self.parameters(), self.use_checkpoint)

    def _forward(self, x, emb):
        if self.updown:
            in_rest, in_conv = self.in_layers[:-1], self.in_layers[-1]
            h = in_rest(x); h = self.h_upd(h); x = self.x_upd(x); h = in_conv(h)
        else:
            h = self.in_layers(x)
        emb_out = self.emb_layers(emb).type(h.dtype)
        while len(emb_out.shape) < len(h.shape): emb_out = emb_out[..., None]
        if self.use_scale_shift_norm:
            out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
            scale, shift = th.chunk(emb_out, 2, dim=1)
            h = out_norm(h) * (1 + scale) + shift; h = out_rest(h)
        else:
            h = h + emb_out; h = self.out_layers(h)
        return self.skip_connection(x) + h


# ============================== RDEIC ==============================
class RDEIC(LatentDiffusion):

    DEFAULT_DOWNSAMPLE = 8  # SD VAE 下采样因子

    def __init__(self,
                 control_stage_config: Mapping[str, Any],
                 sd_locked: bool,
                 is_refine: bool,
                 fixed_step: int,
                 learning_rate: float,
                 l_bpp_weight: float,
                 l_guide_weight: float,
                 used_timesteps: int,
                 sync_path: Optional[str],
                 synch_control: bool,
                 ckpt_path_pre: Optional[str],
                 preprocess_config: Mapping[str, Any],
                 calculate_metrics: Mapping[str, Any],
                 vae_blend_enabled: bool = True,
                 train_blend_net: bool = False,
                 *args, **kwargs) -> "RDEIC":
        super().__init__(*args, **kwargs)

        self.control_model = instantiate_from_config(control_stage_config)
        self.preprocess_model = instantiate_from_config(preprocess_config)
        if sync_path is not None:
            self.sync_control_weights_from_base_checkpoint(sync_path, synch_control=synch_control)
        if ckpt_path_pre is not None:
            self.load_preprocess_ckpt(ckpt_path_pre=ckpt_path_pre)

        self.sd_locked = sd_locked
        self.is_refine = is_refine
        self.fixed_step = fixed_step
        self.learning_rate = learning_rate
        self.l_bpp_weight = l_bpp_weight
        self.l_guide_weight = l_guide_weight

        assert used_timesteps <= self.num_timesteps
        self.used_timesteps = used_timesteps
        self.calculate_metrics = calculate_metrics

        # 指标
        self.metric_funcs = {}
        for _, opt in calculate_metrics.items():
            mopt = opt.copy(); name = mopt.pop('type', None); mopt.pop('better', None)
            self.metric_funcs[name] = pyiqa.create_metric(name, device=self.device, **mopt)

        self.lamba = self.sqrt_recipm1_alphas_cumprod[self.used_timesteps - 1]

        if self.is_refine:
            self.sampler = SpacedSampler(self)
            self.perceptual_loss = LPIPS(pnet_type='alex')

        # 水平环绕卷积开关（若 utils 支持）
        try:
            if self.is_refine:
                enable_feature_hwrap_for_unet(self.model.diffusion_model, enabled=True)
                print("[feature_hwrap] UNet horizontal wrap: ENABLED")
            else:
                print("[feature_hwrap] horizontal wrap: DISABLED (not refine)")
        except Exception as e:
            print("[feature_hwrap] enable failed:", e)

        # 像素端融合设置
        self.vae_blend_enabled = bool(vae_blend_enabled)
        self.train_blend_net = bool(train_blend_net)
        self.blend_net_pixel  = BlendWeightNet(in_channels=3)
        if (not self.is_refine) or (not self.train_blend_net):
            for p in self.blend_net_pixel.parameters():
                p.requires_grad = False

        # VAE 下采样因子（默认 8）
        self.first_stage_downsample = int(getattr(self, 'first_stage_downsample', self.DEFAULT_DOWNSAMPLE))

        # Debug 捕获
        self._dbg_cap = None         # dict: {'capture': True, 'ali': tensor, 'mix': tensor}
        self._dbg_saved_np = None    # dict: {'ali': np.uint8 HxWx3, 'mix': np.uint8 HxWx3}

        # —— 仅记录：原图像素宽 W0，用于融合/裁剪（来自 batch['orig_size']）——
        self._static_W0_pix = 0  # int

    # ---------------- 工具 ----------------
    @staticmethod
    def _to_uint8_image(x: torch.Tensor) -> np.ndarray:
        if x.dim() == 4: x = x[0]
        x = x.detach().float().cpu()
        if x.min() < 0.0 or x.max() > 1.0:
            x = (x + 1) / 2
        x = x.clamp(0, 1).permute(1, 2, 0).numpy()
        return (x * 255.0 + 0.5).astype(np.uint8)

    # =================== 像素域融合（只写回左带 [0:K]；参考带 [W0:W0+K]） ===================
    @staticmethod
    def _pixel_blend_left_only_with_right_tail(img: torch.Tensor, W0: int, K: int, use_net: bool,
                                               blend_net: BlendWeightNet, train_blend_net: bool,
                                               dbg_store: dict = None) -> torch.Tensor:
        """
        img: [B,3,H,W_ext]  VAE 解码后（包含右侧预处理扩宽尾巴）
        W0:  原图像素宽（未扩宽）
        K:   融合带宽（像素）
        逻辑：只回写左带 [0:K]，参考带来自扩充区 [W0:W0+K]；不动右端与尾巴。
        —— 无任何 in-place：通过新张量拼接，避免 Autograd 版本冲突。
        """
        B, C, H, W = img.shape
        if W0 <= 0 or W <= W0 or K <= 0:
            return img
        K = max(1, min(K, W - W0, W0))  # K 不能超过扩充长度与左带宽

        L     = img[..., :K]           # 左带 [0:K]
        R_ref = img[..., W0:W0+K]      # 扩充参考 [W0:W0+K]

        # 线性窗 a: 0..1（缝→内），缝处更贴 R_ref，带内深处更贴 L
        a_lin = torch.linspace(0, 1, K, device=img.device, dtype=img.dtype).view(1,1,1,K)

        if use_net and (blend_net is not None):
            with torch.set_grad_enabled(train_blend_net):
                a_raw = blend_net(L, R_ref)  # [B,1,1,K]
            scale = 0.7 + 0.3 * a_raw
            a = (a_lin * scale).clamp(0.0, 1.0)
        else:
            a = a_lin

        # Debug：热力图（白→红 0→1；只标注左带[0:K]）
        # Debug：热力图（只标注左带[0:K]）
        if isinstance(dbg_store, dict):
            with torch.no_grad():
                # 先画一张完整 W_ext 的热力图（可选，方便排查）
                heat_ext = img.new_zeros((B, 3, H, W))
                v = a.expand(B, 1, H, K)
                r = torch.ones_like(v);
                g = 1.0 - v;
                b = 1.0 - v
                heat_ext[..., :K] = torch.cat([r, g, b], dim=1)
                dbg_store['mix_ext'] = heat_ext.clamp(0, 1)  # 可选：保留完整宽度版本

                # ——按 W0 裁一份（用于 *_mix.png 输出）——
                heat_w0 = heat_ext[..., :W0].contiguous()
                dbg_store['mix'] = heat_w0

        # 新张量拼接，避免 in-place
        L_blended = (1.0 - a) * R_ref + a * L                  # [B,3,H,K]
        rest      = img[..., K:]                               # [B,3,H,W-K]
        out       = torch.cat([L_blended, rest], dim=-1)       # [B,3,H,W]
        return out

    def circular_blend_pixel(self, img: torch.Tensor, W0_pix: int, dbg_store: dict = None) -> torch.Tensor:
        if not (self.is_refine and self.vae_blend_enabled):
            return img
        W_ext  = int(img.shape[-1])
        W0     = int(W0_pix)
        K_true = max(0, W_ext - W0)
        if K_true <= 0:
            return img
        K_blend = max(1, K_true // 2)  # “最后融合取真正拼接像素的一半”
        use_net = bool(self.train_blend_net)
        return self._pixel_blend_left_only_with_right_tail(
            img, W0=W0, K=K_blend, use_net=use_net,
            blend_net=self.blend_net_pixel, train_blend_net=self.train_blend_net,
            dbg_store=dbg_store
        )

    # ---------------- 解码（不做瓦片）：整幅 decode → 单侧融合 ----------------
    def _vae_decode_full(self, z: torch.Tensor, with_grad: bool) -> torch.Tensor:
        """
        直接整幅解码，不做瓦片；随后做像素域单侧融合（仅左带），再返回。
        """
        img_full = self.first_stage_model.decode(z / self.scale_factor) if with_grad else super().decode_first_stage(z)

        # —— 捕获融合前画面（ali）
        cap = getattr(self, "_dbg_cap", None)
        if isinstance(cap, dict) and cap.get('capture', False):
            with torch.no_grad():
                cap['ali'] = img_full.detach().clone()

        # —— 像素域单侧融合（只在 W0 内左带写回）
        W0_pix = int(self._static_W0_pix) if int(self._static_W0_pix) > 0 else int(img_full.shape[-1])
        if self.vae_blend_enabled:
            img_full = self.circular_blend_pixel(img_full, W0_pix, dbg_store=cap if isinstance(cap, dict) else None)
        return img_full

    @torch.no_grad()
    def decode_first_stage(self, z):
        return self._vae_decode_full(z, with_grad=False)

    def decode_first_stage_with_grad(self, z):
        return self._vae_decode_full(z, with_grad=True)

    # ------------------- 公共逻辑 -------------------
    def apply_condition_encoder(self, x):
        c_latent, likelihoods, q_likelihoods, emb_loss, guide_hint = self.preprocess_model(x)
        return c_latent, likelihoods, q_likelihoods, emb_loss, guide_hint

    @torch.no_grad()
    def apply_condition_compress(self, x, stream_path, H, W):
        # 注意：此处的 H、W 应为原图（未扩宽）尺寸，用于 bpp
        _, h = self.encode_first_stage(x * 2 - 1)
        h = h * self.scale_factor
        out = self.preprocess_model.compress(h)
        shape = out["shape"]
        with Path(stream_path).open("wb") as f:
            write_body(f, shape, out["strings"])
        size = filesize(stream_path)
        bpp = float(size) * 8 / (H * W)
        return bpp

    @torch.no_grad()
    def apply_condition_decompress(self, stream_path):
        with Path(stream_path).open("rb") as f:
            strings, shape = read_body(f)
        c_latent, guide_hint = self.preprocess_model.decompress(strings, shape)
        return c_latent, guide_hint

    def get_input(self, batch, k, bs=None, *args, **kwargs):
        """
        关键变化：
        - 假设 batch[first_stage_key] 已为【预处理扩宽后】图像（[-1,1]），宽 W_ext。
        - 从 batch['orig_size'] 读取原始宽 W0（H0,W0），供解码融合与裁剪使用。
        """
        target, x, h, c = super().get_input(batch, self.first_stage_key, bs=bs, *args, **kwargs)
        c_latent, likelihoods, q_likelihoods, emb_loss, guide_hint = self.apply_condition_encoder(h)

        # 从 batch 读取 W0；若缺失则退化为 W0=W_ext（不做融合）
        try:
            if torch.is_tensor(batch["orig_size"]):
                H0, W0 = batch["orig_size"].view(-1).tolist()
            else:
                H0, W0 = batch["orig_size"]
            H0, W0 = int(H0), int(W0)
        except Exception:
            H0, W0 = int(x.shape[-2]), int(x.shape[-1])  # 退化为不融合

        N = x.shape[0]
        num_pixels = N * H0 * W0 * 64   # bpp 用原图面积（未扩宽）
        bpp = sum((torch.log(l).sum() / (-math.log(2) * num_pixels)) for l in likelihoods)
        q_bpp = sum((torch.log(l).sum() / (-math.log(2) * num_pixels)) for l in q_likelihoods)

        # 读取完 H0, W0 后，覆盖 target 为裁回 W0 的成品
        target_W0 = target[..., :H0, :W0]
        self._static_W0_pix = int(W0)  # 继续给像素融合用
        hshift = torch.tensor(0, device=target.device)  # 保留原逻辑

        return x, dict(
            c_crossattn=[c],
            c_latent=[c_latent],
            bpp=bpp, q_bpp=q_bpp, emb_loss=emb_loss,
            guide_hint=guide_hint,
            target=target_W0,  # ← 用裁剪后的 target
            orig_size=torch.tensor([H0, W0], device=target.device),
            hshift_pix=hshift
        )

    # —— 模型前向：不做任何 latent 扩宽/roll —— #
    def apply_model(self, x_noisy, t, cond, *args, **kwargs):
        diffusion_model = self.model.diffusion_model
        cond_txt = torch.cat(cond['c_crossattn'], 1)
        guide_hint = cond['guide_hint']
        eps = self.control_model(
            x=x_noisy, timesteps=t, context=cond_txt,
            guide_hint=guide_hint, base_model=diffusion_model
        )
        return eps

    def apply_model_unconditional(self, x_noisy, t, cond, *args, **kwargs):
        diffusion_model = self.model.diffusion_model
        cond_txt = torch.cat(cond['c_crossattn'], 1)
        return self.control_model.forward_unconditional(
            x=x_noisy, timesteps=t, context=cond_txt, base_model=diffusion_model
        )

    @torch.no_grad()
    def get_unconditional_conditioning(self, N):
        return self.get_learned_conditioning([""] * N)

    # --------- 工具：尺寸对齐 ---------
    @staticmethod
    def _align_to_target(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        H0, W0 = target.shape[-2:]
        return pred[..., :H0, :W0]

    @staticmethod
    def _align_pair(x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        H = min(x.size(-2), y.size(-2)); W = min(x.size(-1), y.size(-1))
        return x[..., :H, :W], y[..., :H, :W]

    # ---------------- 日志可视化 ----------------
    @torch.no_grad()
    def log_images(self, batch, sample_steps=5, bs=2):
        # reset W0 记录
        self._static_W0_pix = 0

        log = dict()
        z, c = self.get_input(batch, self.first_stage_key, bs=bs)
        bpp = c["q_bpp"] + 0.003418
        bpp_val = float(bpp.detach().mean()); bpp_img = [f'{bpp_val:.4f}'] * 4
        target = c["target"]
        log["target"] = (target + 1) / 2

        # —— 直接重建（不采样） —— #
        H0, W0 = map(int, c["orig_size"].tolist())
        vae_rec = self.decode_first_stage(z)
        vae_rec = vae_rec[..., :H0, :W0]  # ← 明确裁到 W0

        log["vae_rec"] = (vae_rec + 1) / 2
        log["text"] = (log_txt_as_img((512, 512), bpp_img, size=16) + 1) / 2

        if not sample_steps or sample_steps <= 0:
            log["samples"] = log["vae_rec"]; return log, bpp

        # —— 采样 —— #
        c_latent = c["c_latent"][0]
        b, cch, h, w_ext = c_latent.shape
        shape = (b, self.channels, h, w_ext)
        t = torch.ones((b,)).long().to(self.device) * self.used_timesteps - 1
        noise = default(None, lambda: torch.randn_like(c_latent))
        x_T = self.q_sample(x_start=c_latent, t=t, noise=noise)

        if self.is_refine:
            samples_ext = self.sampler.sample(self.fixed_step, shape, c, unconditional_guidance_scale=1.0,
                                              unconditional_conditioning=None, x_T=x_T)
        else:
            sampler = SpacedSampler(self)
            samples_ext = sampler.sample(sample_steps, shape, c, unconditional_guidance_scale=1.0,
                                         unconditional_conditioning=None, x_T=x_T)

        # 打开“同管线抓帧”开关（捕获 ali/mix）
        self._dbg_cap = {'capture': True}
        x_samples_full = self.decode_first_stage(samples_ext)  # 内部会融合
        try:
            ali_t = self._dbg_cap.get('ali', None)
            mix_t = self._dbg_cap.get('mix', None)
            dbg_np = {}
            if ali_t is not None:
                dbg_np['ali'] = self._to_uint8_image(ali_t)
            if mix_t is not None:
                dbg_np['mix'] = self._to_uint8_image(mix_t)
            self._dbg_saved_np = dbg_np if dbg_np else None
        except Exception:
            self._dbg_saved_np = None
        finally:
            self._dbg_cap = None

        x_samples = x_samples_full[..., :H0, :W0]

        log["samples"] = (x_samples + 1) / 2
        return log, bpp

    # ---------------- 采样（保留兼容） ----------------
    @torch.no_grad()
    def sample_log(self, cond, steps):
        x_T = cond["c_latent"][0]
        b, c, h, w = x_T.shape
        shape = (b, self.channels, h, w)
        t = torch.ones((b,)).long().to(self.device) * self.used_timesteps - 1
        noise = default(None, lambda: torch.randn_like(x_T))
        x_T = self.q_sample(x_start=x_T, t=t, noise=noise)
        if self.is_refine:
            samples = self.sampler.sample(steps, shape, cond, unconditional_guidance_scale=1.0,
                                          unconditional_conditioning=None, x_T=x_T)
        else:
            sampler = SpacedSampler(self)
            samples = sampler.sample(steps, shape, cond, unconditional_guidance_scale=1.0,
                                     unconditional_conditioning=None, x_T=x_T)
        return samples

    # ================= 训练/验证 =================
    def configure_optimizers(self):
        lr = self.learning_rate
        params = []
        params += list(self.control_model.parameters())
        params += list(self.preprocess_model.parameters())
        if not self.sd_locked:
            params += list(self.model.diffusion_model.output_blocks.parameters())
            params += list(self.model.diffusion_model.out.parameters())
        if self.is_refine and self.train_blend_net:
            params += list(self.blend_net_pixel.parameters())
        params = [p for p in params if p.requires_grad]
        opt = torch.optim.AdamW(params, lr=lr)
        return opt

    def p_losses(self, x_start, cond, t, noise=None):
        loss_dict = {}; prefix = 'T' if self.training else 'V'

        # 这里的 x_start、cond['c_latent'][0] 的宽度都是 W_ext（预处理已扩宽）
        x_start_ext = x_start
        c_latent_ext = cond['c_latent'][0]

        # 标准/精修两路
        if not self.is_refine:
            noise = default(noise, lambda: torch.randn_like(x_start_ext)) + (c_latent_ext - x_start_ext) / self.lamba
            x_noisy_ext = self.q_sample(x_start=x_start_ext, t=t, noise=noise)
            model_output_ext = self.apply_model(x_noisy_ext, t, cond)

            if self.parameterization == "x0":
                target = x_start_ext
            elif self.parameterization == "eps":
                target = x_start_ext
                model_output_ext = self._predict_xstart_from_eps(x_noisy_ext, t, model_output_ext)
            elif self.parameterization == "v":
                target = self.get_v(x_start_ext, noise, t)
            else:
                raise NotImplementedError()

            loss_simple = self.get_loss(model_output_ext, target, mean=False).mean([1,2,3])
            loss_dict.update({f'{prefix}/l_simple': loss_simple.mean()})
            logvar_t = self.logvar[t].to(self.device)
            loss = loss_simple / torch.exp(logvar_t) + logvar_t
            if self.learn_logvar:
                loss_dict.update({f'{prefix}/l_gamma': loss.mean()})
                loss_dict.update({'logvar': self.logvar.data.mean()})
            loss = self.l_guide_weight * loss.mean()

            # bpp/emb
            loss_bpp = cond['bpp']; guide_bpp = cond['q_bpp']
            loss_dict.update({f'{prefix}/l_bpp': loss_bpp.mean()})
            loss_dict.update({f'{prefix}/q_bpp': guide_bpp.mean()})
            loss += self.l_bpp_weight * loss_bpp

            loss_emb = cond['emb_loss']
            loss_dict.update({f'{prefix}/l_emb': loss_emb.mean()})
            loss += self.l_bpp_weight * loss_emb

            # 引导对齐（latent）
            loss_guide = self.get_loss(c_latent_ext, x_start_ext)
            loss_dict.update({f'{prefix}/l_guide': loss_guide.mean()})
            loss += self.l_guide_weight * loss_guide

            loss_dict.update({f'{prefix}/loss': loss})
            return loss, loss_dict

        # refine：sample_grad → 解码（含像素单侧融合）→ 裁回 W0 计算图像损失
        b, cch, h, w_ext = c_latent_ext.shape
        noise = default(noise, lambda: torch.randn_like(c_latent_ext))
        x_T_ext = self.q_sample(x_start=c_latent_ext, t=t, noise=noise)
        steps = self.fixed_step

        samples_ext = self.sampler.sample_grad(steps, (b, self.channels, h, w_ext), cond,
                                               unconditional_guidance_scale=1.0,
                                               unconditional_conditioning=None, x_T=x_T_ext)
        model_output_pix = self.decode_first_stage_with_grad(samples_ext)  # 内含像素端融合
        target = cond['target']
        model_output_pix = self._align_to_target(model_output_pix, target)

        # latent 简单项
        loss_simple = self.get_loss(samples_ext, x_start_ext, mean=False).mean([1,2,3])
        loss_dict.update({f'{prefix}/l_simple': loss_simple.mean()})
        loss = self.l_guide_weight * loss_simple.mean()

        # 像素域损失
        loss_mse = self.get_loss(model_output_pix, target, mean=False).mean([1,2,3])
        loss_dict.update({f'{prefix}/l_mse': loss_mse.mean()})
        loss += self.l_guide_weight * loss_mse.mean()

        with torch.cuda.amp.autocast(enabled=False):
            loss_lpips = self.perceptual_loss(model_output_pix.float(), target.float())
        loss_dict.update({f'{prefix}/l_lpips': loss_lpips.mean()})
        loss += self.l_guide_weight * loss_lpips * 0.5

        # bpp/emb
        loss_bpp = cond['bpp']; guide_bpp = cond['q_bpp']
        loss_dict.update({f'{prefix}/l_bpp': loss_bpp.mean()})
        loss_dict.update({f'{prefix}/q_bpp': guide_bpp.mean()})
        loss += self.l_bpp_weight * loss_bpp

        loss_emb = cond['emb_loss']; loss_dict.update({f'{prefix}/l_emb': loss_emb.mean()})
        loss += self.l_bpp_weight * loss_emb

        # latent 闭环（轻微约束）
        latent_seam = F.l1_loss(samples_ext[..., :1], samples_ext[..., -1:])
        loss += 0.02 * latent_seam
        loss_dict.update({f'{prefix}/l_seam_latent': latent_seam.detach()})

        loss_dict.update({f'{prefix}/loss': loss})
        return loss, loss_dict

    def training_step(self, batch, batch_idx):
        dm = getattr(self.trainer, "datamodule", None)
        bsz = getattr(dm, "train_batch_size", None)
        if bsz is None:
            try: bsz = batch[self.first_stage_key].shape[0]
            except Exception:
                first_val = next(iter(batch.values())); bsz = first_val.shape[0] if hasattr(first_val, "shape") else len(first_val)

        loss, loss_dict = self.shared_step(batch)

        safe_logs = {}
        for k, v in loss_dict.items():
            if torch.is_tensor(v):
                try:
                    safe_v = v.detach()
                    if safe_v.ndim > 0: safe_v = safe_v.mean().detach()
                    safe_logs[k] = float(safe_v.item())
                except Exception:
                    safe_logs[k] = float(v.mean().detach().item())
            else:
                safe_logs[k] = v

        self.log_dict(safe_logs, prog_bar=True, logger=True, on_step=True, on_epoch=True, batch_size=bsz)
        self.log("global_step", float(self.global_step), prog_bar=True, logger=True, on_step=True, on_epoch=False, batch_size=bsz)
        if getattr(self, "use_scheduler", False):
            lr = self.optimizers().param_groups[0]['lr']
            self.log('lr_abs', float(lr), prog_bar=True, logger=True, on_step=True, on_epoch=False, batch_size=bsz)
        return loss

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        sample_steps = int(getattr(self, "val_sample_steps", 5))
        do_metrics = bool(getattr(self, "val_eval_metrics", True)) and bool(getattr(self, "calculate_metrics", None))
        max_per_batch = int(getattr(self, "val_max_per_batch", -1))
        try: B = batch[self.first_stage_key].shape[0]
        except Exception: B = next(iter(batch.values())).shape[0]
        n = B if max_per_batch < 0 else min(B, max_per_batch)
        save_dir = os.path.join(self.logger.save_dir, "validation", f"epoch{self.current_epoch:04d}")
        os.makedirs(save_dir, exist_ok=True)
        bpp_vals = []; psnr_vals, msssim_vals, lpips_vals = [], [], []
        with torch.inference_mode(), torch.cuda.amp.autocast(enabled=True):
            for i in range(n):
                one = {}
                for k, v in batch.items():
                    if torch.is_tensor(v) and v.dim() >= 1 and v.shape[0] == B:
                        one[k] = v[i:i+1]
                    else:
                        one[k] = v
                log, bpp = self.log_images(one, bs=1, sample_steps=sample_steps)
                bpp_vals.append(float(bpp.detach().mean()))

                x_vis = (log["samples"][0].detach().cpu().numpy().transpose(1, 2, 0) * 255).clip(0, 255).astype(np.uint8)
                out_png = os.path.join(save_dir, f"{batch_idx}_{i}.png")
                Image.fromarray(x_vis).save(out_png)

                dbg = getattr(self, "_dbg_saved_np", None)
                if isinstance(dbg, dict):
                    if 'ali' in dbg:
                        Image.fromarray(dbg['ali']).save(os.path.join(save_dir, f"{batch_idx}_{i}_ali.png"))
                    if 'mix' in dbg:
                        Image.fromarray(dbg['mix']).save(os.path.join(save_dir, f"{batch_idx}_{i}_mix.png"))
                self._dbg_saved_np = None

                if do_metrics:
                    x = log["samples"].float(); y = log["target"].float(); x, y = self._align_pair(x, y)
                    if "psnr" in self.metric_funcs:    psnr_vals.append(float(self.metric_funcs["psnr"](x, y)))
                    if "ms_ssim" in self.metric_funcs: msssim_vals.append(float(self.metric_funcs["ms_ssim"](x, y)))
                    if "lpips" in self.metric_funcs:   lpips_vals.append(float(self.metric_funcs["lpips"](x, y)))
                del log, bpp
        out = [sum(bpp_vals) / max(len(bpp_vals), 1)]
        if do_metrics:
            if psnr_vals:   out.append(sum(psnr_vals) / len(psnr_vals))
            if msssim_vals: out.append(sum(msssim_vals) / len(msssim_vals))
            if lpips_vals:  out.append(sum(lpips_vals) / len(lpips_vals))
        return out

    def on_validation_epoch_start(self):
        self.preprocess_model.quantize.reset_usage()
        return super().on_validation_epoch_start()

    def validation_epoch_end(self, outputs: EPOCH_OUTPUT):
        if not outputs: return
        lens = {len(o) for o in outputs}
        if len(lens) != 1:
            L = min(lens); outputs = [o[:L] for o in outputs]
        else: L = lens.pop()
        import torch as _torch
        arr = _torch.tensor(outputs, dtype=_torch.float32); avg_out = arr.mean(dim=0)
        self.log("avg_bpp", float(avg_out[0]), prog_bar=True, logger=True, on_step=False, on_epoch=True, batch_size=1)
        self.log("val/bpp", float(avg_out[0]), prog_bar=False, logger=True, on_step=False, on_epoch=True, batch_size=1)
        self.log("val_bpp", float(avg_out[0]), prog_bar=False, logger=True, on_step=False, on_epoch=True, batch_size=1)
        usage = self.preprocess_model.quantize.get_usage()
        self.log("usage", usage, prog_bar=True, logger=True, on_step=False, on_epoch=True, batch_size=1)
        names = ["bpp"]; metrics_order = ["psnr", "ms_ssim", "lpips"]
        if getattr(self, "calculate_metrics", None): names += [m for m in metrics_order if m in self.calculate_metrics]
        names = names[:len(avg_out)]
        for i, name in enumerate(names[1:], start=1):
            val = float(avg_out[i])
            self.log(f"avg_{name}", val, prog_bar=True, logger=True, on_step=False, on_epoch=True, batch_size=1)
            self.log(f"val/{name}", val, prog_bar=False, logger=True, on_step=False, on_epoch=True, batch_size=1)
            self.log(f"val_{name.replace('/', '_')}", val, prog_bar=False, logger=True, on_step=False, on_epoch=True, batch_size=1)
        try:
            if "psnr" in names and "lpips" in names:
                psnr_idx = names.index("psnr"); lpips_idx = names.index("lpips")
                psnr_val = float(avg_out[psnr_idx]); lpips_val = float(avg_out[lpips_idx])
                score_combo = psnr_val - 5.0 * lpips_val
                self.log("val/score_combo", score_combo, prog_bar=True, logger=True, on_step=False, on_epoch=True, batch_size=1)
                self.log("val_score_combo", score_combo, prog_bar=False, logger=True, on_step=False, on_epoch=True, batch_size=1)
        except Exception as e:
            print(f"[val] score_combo compute failed: {e}")

    # ---------------- 权重加载 ----------------
    def load_preprocess_ckpt(self, ckpt_path_pre):
        ckpt = torch.load(ckpt_path_pre)
        self.preprocess_model.load_state_dict(ckpt)
        print(['CONTROL WEIGHTS LOADED'])

    def sync_control_weights_from_base_checkpoint(self, path, synch_control=True):
        ckpt_base = torch.load(path)
        if synch_control:
            for key in list(ckpt_base['state_dict'].keys()):
                if "diffusion_model." in key:
                    dst_key_old = 'control_model.control' + key[15:]
                    dst_key_new = 'control_model' + key[15:]
                    if dst_key_old in self.state_dict().keys(): dst_key = dst_key_old
                    elif dst_key_new in self.state_dict().keys(): dst_key = dst_key_new
                    else: continue
                    if ckpt_base['state_dict'][key].shape != self.state_dict()[dst_key].shape:
                        if len(ckpt_base['state_dict'][key].shape) == 1:
                            control_dim = self.state_dict()[dst_key].size(0)
                            ckpt_base['state_dict'][dst_key] = torch.cat(
                                [ckpt_base['state_dict'][key], ckpt_base['state_dict'][key]], dim=0)[:control_dim]
                        else:
                            control_dim_0 = self.state_dict()[dst_key].size(0)
                            control_dim_1 = self.state_dict()[dst_key].size(1)
                            ckpt_base['state_dict'][dst_key] = torch.cat(
                                [ckpt_base['state_dict'][key], ckpt_base['state_dict'][key]], dim=1)[:control_dim_0, :control_dim_1, ...]
                    else:
                        ckpt_base['state_dict'][dst_key] = ckpt_base['state_dict'][key]
        res_sync = self.load_state_dict(ckpt_base['state_dict'], strict=False)
        print(f'[{len(res_sync.missing_keys)} keys are missing from the model (hint processing and cross connections included)]')
