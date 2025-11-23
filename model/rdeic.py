# -*- coding: utf-8 -*-
"""
RDEIC (refine stage, latent head-extend + pixel left-only blend)

设计目标：
- 第一阶段（is_refine=False）：完全沿用原作者的编码流程和损失，
  从 image -> first_stage.encode -> preprocess_model -> c_latent -> 码流 zc。
- 第二阶段（is_refine=True）：在此基础上增加：
  1) latent 左侧 head 扩宽；
  2) VAE 解码到扩宽后的像素图；
  3) 像素域只在左带 [0:K_pix] 与右尾巴 [W0:W0+K_pix] 做单侧融合；
  4) 裁回到原宽 W0 做 MSE+LPIPS 等像素域损失。
"""

import os
import math
from typing import Mapping, Any, Tuple, Optional, Dict

import numpy as np
import pyiqa
from pathlib import Path
from PIL import Image

import torch
import torch as th
import torch.nn as nn
import torch.nn.functional as F

from pytorch_lightning.utilities.types import EPOCH_OUTPUT

from utils.utils import *
from ldm.util import log_txt_as_img, exists, instantiate_from_config, default
from ldm.modules.diffusionmodules.util import (
    conv_nd, linear, zero_module, timestep_embedding, checkpoint
)
from ldm.modules.attention import SpatialTransformer
from ldm.models.diffusion.ddpm import LatentDiffusion
from ldm.modules.diffusionmodules.openaimodel import (
    UNetModel, TimestepEmbedSequential, ResBlock as ResBlock_orig,
    Downsample, Upsample, AttentionBlock, TimestepBlock
)
from .spaced_sampler_relay import SpacedSampler
from .lpips import LPIPS


# ===================== 轻量像素融合权重小网（可训练） =====================
class BlendWeightNet(nn.Module):
    """
    输入：左带 L 与右尾巴参考 R_ref（仅 K 带宽内容）
    输出：a_raw \in (0,1)，作为线性窗的“位置自适应”缩放权重
    """
    def __init__(self, in_channels: int):
        super().__init__()
        ch = in_channels * 2  # L || R_ref
        self.conv = nn.Conv2d(ch, 1, kernel_size=1, bias=True)
        nn.init.zeros_(self.conv.weight)
        nn.init.constant_(self.conv.bias, 0.0)

    def forward(self, L: torch.Tensor, R_ref: torch.Tensor) -> torch.Tensor:
        # 沿高方向均值，只随列位置变化
        Lm = L.mean(dim=2, keepdim=True)       # [B,C,1,K]
        Rm = R_ref.mean(dim=2, keepdim=True)   # [B,C,1,K]
        x = torch.cat([Lm, Rm], dim=1)         # [B,2C,1,K]
        a = torch.sigmoid(self.conv(x))        # [B,1,1,K] in (0,1)
        return a


# ===================== 控制分支（和原版保持一致） =====================
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
            num_heads_upsample=num_heads_upsample, use_scale_shift_norm=use_scale_shift_norm,
            resblock_updown=resblock_updown, use_new_attention_order=use_new_attention_order,
            use_spatial_transformer=use_spatial_transformer, transformer_depth=transformer_depth,
            context_dim=context_dim, n_embed=n_embed, legacy=legacy,
            disable_self_attentions=None, num_attention_blocks=None,
            disable_middle_self_attn=False, use_linear_in_transformer=use_linear_in_transformer,
            control_model_ratio=control_model_ratio,
        )

        self.enc_zero_convs_out = nn.ModuleList([])
        self.middle_block_out = None
        self.dec_zero_convs_out = nn.ModuleList([])

        ch_inout_ctr = {'enc': [], 'mid': [], 'dec': []}
        ch_inout_base = {'enc': [], 'mid': [], 'dec': []}

        # --- 收集通道数 ---
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
            (self.control_model.middle_block[0].channels, self.control_model.middle_block[-1].out_channels)
        )
        ch_inout_base['mid'].append(
            (base_model.middle_block[0].channels, base_model.middle_block[-1].out_channels)
        )

        for module in base_model.output_blocks:
            if isinstance(module[0], nn.Conv2d):
                ch_inout_base['dec'].append((module[0].in_channels, module[0].out_channels))
            elif isinstance(module[0], (ResBlock, ResBlock_orig)):
                ch_inout_base['dec'].append((module[0].channels, module[0].out_channels))
            elif isinstance(module[-1], Upsample):
                ch_inout_base['dec'].append((module[0].channels, module[-1].out_channels))

        self.ch_inout_ctr = ch_inout_ctr
        self.ch_inout_base = ch_inout_base

        # --- 零卷积 cross-connection ---
        self.middle_block_out = self.make_zero_conv(ch_inout_ctr['mid'][-1][1], ch_inout_base['mid'][-1][1])

        self.dec_zero_convs_out.append(
            self.make_zero_conv(ch_inout_ctr['enc'][-1][1], ch_inout_base['mid'][-1][1])
        )
        for i in range(1, len(ch_inout_ctr['enc'])):
            self.dec_zero_convs_out.append(
                self.make_zero_conv(ch_inout_ctr['enc'][-(i + 1)][1], ch_inout_base['dec'][i - 1][1])
            )
        for i in range(len(ch_inout_ctr['enc'])):
            self.enc_zero_convs_out.append(
                self.make_zero_conv(ch_inout_ctr['enc'][i][1], ch_inout_base['enc'][i][1])
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
        use_spatial_transformer=False,
        transformer_depth=1,
        context_dim=None,
        n_embed=None,
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
                    conv_nd(dims, in_channels + hint_channels, model_channels, 3, padding=1)
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
                        ch, time_embed_dim, dropout,
                        out_channels=mult * model_channels,
                        dims=dims, use_checkpoint=use_checkpoint,
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
                    if legacy:
                        dim_head = ch // num_heads if use_spatial_transformer else num_head_channels
                    if exists(disable_self_attentions):
                        disabled_sa = disable_self_attentions[level]
                    else:
                        disabled_sa = False

                    if not exists(num_attention_blocks) or nr < num_attention_blocks[level]:
                        layers.append(
                            AttentionBlock(
                                ch, use_checkpoint=use_checkpoint,
                                num_heads=num_heads, num_head_channels=dim_head,
                                use_new_attention_order=use_new_attention_order
                            ) if not use_spatial_transformer else SpatialTransformer(
                                ch, num_heads, dim_head, depth=transformer_depth, context_dim=context_dim,
                                disable_self_attn=disabled_sa, use_linear=use_linear_in_transformer,
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
                        ResBlock(
                            ch, time_embed_dim, dropout,
                            out_channels=out_ch, dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            down=True,
                        ) if resblock_updown else Downsample(
                            ch, conv_resample, dims=dims, out_channels=out_ch
                        )
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
        if legacy:
            dim_head = ch // num_heads if use_spatial_transformer else num_head_channels

        self.middle_block = TimestepEmbedSequential(
            ResBlock(ch, time_embed_dim, dropout, out_channels=ch, dims=dims,
                     use_checkpoint=use_checkpoint, use_scale_shift_norm=use_scale_shift_norm),
            AttentionBlock(ch, use_checkpoint=use_checkpoint,
                           num_heads=num_heads, num_head_channels=dim_head,
                           use_new_attention_order=use_new_attention_order)
            if not use_spatial_transformer else SpatialTransformer(
                ch, num_heads, dim_head, depth=transformer_depth, context_dim=context_dim,
                disable_self_attn=disable_middle_self_attn, use_linear=use_linear_in_transformer,
                use_checkpoint=use_checkpoint
            ),
            ResBlock(ch, time_embed_dim, dropout, dims=dims,
                     use_checkpoint=use_checkpoint, use_scale_shift_norm=use_scale_shift_norm),
        )
        self._feature_size += ch


def find_denominator(number, start):
    if start >= number:
        return number
    while start != 0:
        residual = number % start
        if residual == 0:
            return start
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


# ============================== RDEIC 主体 ==============================
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
                 # === 新增配置项：仅在 refine 阶段生效 ===
                 vae_blend_enabled: bool = True,
                 train_blend_net: bool = True,
                 extend_k_pix: int = 64,
                 *args, **kwargs) -> "RDEIC":
        super().__init__(*args, **kwargs)

        # 控制 & 预处理
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

        # 指标
        self.calculate_metrics = calculate_metrics
        self.metric_funcs: Dict[str, Any] = {}
        for _, opt in calculate_metrics.items():
            mopt = opt.copy(); name = mopt.pop('type', None); mopt.pop('better', None)
            self.metric_funcs[name] = pyiqa.create_metric(name, device=self.device, **mopt)

        # 记录最近 val 指标
        self.last_val_psnr = float('nan')
        self.last_val_ms_ssim = float('nan')
        self.last_val_fid = float('nan')

        self.lamba = self.sqrt_recipm1_alphas_cumprod[self.used_timesteps - 1]

        if self.is_refine:
            self.sampler = SpacedSampler(self)
            self.perceptual_loss = LPIPS(pnet_type='alex')

        # 像素融合相关
        self.vae_blend_enabled = bool(vae_blend_net_bool(vae_blend_enabled))
        self.train_blend_net = bool(train_blend_net)
        self.blend_net_pixel = BlendWeightNet(in_channels=3)
        for p in self.blend_net_pixel.parameters():
            p.requires_grad = bool(self.is_refine and self.train_blend_net)

        # 扩宽相关
        self.extend_k_pix_default = int(max(0, extend_k_pix))  # 默认像素扩宽 K
        self.first_stage_downsample = int(getattr(self, 'first_stage_downsample', self.DEFAULT_DOWNSAMPLE))

        # 临时记录
        self._static_W0_pix = 0   # 原图宽
        self._last_extend_k_pix = 0  # 实际采用的 K_pix

    # ---------------- 工具 ----------------
    @staticmethod
    def _to_uint8_image(x: torch.Tensor) -> np.ndarray:
        if x.dim() == 4: x = x[0]
        x = x.detach().float().cpu()
        if x.min() < 0.0 or x.max() > 1.0:
            x = (x + 1) / 2
        x = x.clamp(0, 1).permute(1, 2, 0).numpy()
        return (x * 255.0 + 0.5).astype(np.uint8)

    @staticmethod
    def _round_up_multiple(x: int, m: int) -> int:
        if m <= 1: return x
        r = x % m
        return x if r == 0 else x + (m - r)

    # ============ latent 右侧扩宽（左侧 head 复制到右侧，工具） ============
    @staticmethod
    def _latent_extend_left_head(x_lat: torch.Tensor, K_lat: int) -> torch.Tensor:
        """在 latent 域做“右侧扩宽”：把左侧 K_lat 列复制到最右侧。
        x_lat: [B,C,H,W_lat]
        返回:  [B,C,H,W_lat + K_lat] = [x_lat, head(K_lat)]
        """
        if K_lat <= 0:
            return x_lat
        head = x_lat[..., :K_lat]
        x_ext = torch.cat([x_lat, head], dim=-1)
        return x_ext

    # ============ 像素域单侧融合（左带写回；右尾巴为参照） ============
    @staticmethod
    def _pixel_blend_left_only_with_right_tail(img: torch.Tensor, W0: int, K: int,
                                               use_net: bool, blend_net: BlendWeightNet,
                                               train_blend_net: bool) -> torch.Tensor:
        """
        img: [B,3,H,W_ext]  VAE 解码后（包含右侧“尾巴”）
        W0:  原图宽
        K:   融合带宽（像素）
        逻辑：只回写左带 [0:K]，参考带来自右尾巴 [W0:W0+K]；右端不动。
        """
        B, C, H, W = img.shape
        if W0 <= 0 or W <= W0 or K <= 0:
            return img
        K = max(1, min(K, W - W0, W0))

        L     = img[..., :K]
        R_ref = img[..., W0:W0+K]

        a_lin = torch.linspace(0, 1, K, device=img.device, dtype=img.dtype).view(1,1,1,K)

        if use_net and (blend_net is not None):
            with torch.set_grad_enabled(train_blend_net):
                a_raw = blend_net(L, R_ref)  # [B,1,1,K]
            scale = 0.7 + 0.3 * a_raw
            a = (a_lin * scale).clamp(0.0, 1.0)
        else:
            a = a_lin

        L_blended = (1.0 - a) * R_ref + a * L
        out = torch.cat([L_blended, img[..., K:]], dim=-1)
        return out

    # ============ VAE 解码（含像素端融合，仅 refine 时用） ============
    def _vae_decode_full(self, z: torch.Tensor, with_grad: bool) -> torch.Tensor:
        if with_grad:
            img_full = self.first_stage_model.decode(z / self.scale_factor)
        else:
            img_full = super().decode_first_stage(z)

        W0_pix = int(self._static_W0_pix) if int(self._static_W0_pix) > 0 else int(img_full.shape[-1])
        if self.is_refine and self.vae_blend_enabled:
            W_ext  = int(img_full.shape[-1])
            K_true = max(0, W_ext - W0_pix)
            K_blend = max(1, K_true // 2)
            if K_blend > 0:
                img_full = self._pixel_blend_left_only_with_right_tail(
                    img_full, W0=W0_pix, K=K_blend,
                    use_net=bool(self.train_blend_net),
                    blend_net=self.blend_net_pixel,
                    train_blend_net=bool(self.train_blend_net)
                )
        return img_full

    @torch.no_grad()
    def decode_first_stage(self, z):
        return self._vae_decode_full(z, with_grad=False)

    def decode_first_stage_with_grad(self, z):
        return self._vae_decode_full(z, with_grad=True)

    # ---------------- 公共逻辑 ----------------
    def apply_condition_encoder(self, x):
        c_latent, likelihoods, q_likelihoods, emb_loss, guide_hint = self.preprocess_model(x)
        return c_latent, likelihoods, q_likelihoods, emb_loss, guide_hint

    @torch.no_grad()
    def apply_condition_compress(self, x, stream_path, H, W):
        # H,W 是原图尺寸，用于 bpp
        _, h = self.encode_first_stage(x * 2 - 1)
        h = h * self.scale_factor
        out = self.preprocess_model.compress(h)
        shape = out["shape"]
        with Path(stream_path).open("wb") as f:
            write_body(f, shape, out["strings"])
        size = filesize(stream_path)
        num_pixels = max(H * W, 1)
        bpp = float(size) * 8.0 / float(num_pixels)
        return bpp

    @torch.no_grad()
    def apply_condition_decompress(self, stream_path):
        with Path(stream_path).open("rb") as f:
            strings, shape = read_body(f)
        c_latent, guide_hint = self.preprocess_model.decompress(strings, shape)
        return c_latent, guide_hint

    def get_input(self, batch, k, bs=None, *args, **kwargs):
        """
        - is_refine=False：完全沿用原作者的第一阶段写法。
        - is_refine=True：额外读取 orig_size / extend_k_pix，并在 latent 域做扩宽（训练/采样阶段再执行）。
        """
        target, x, h, c = super().get_input(batch, self.first_stage_key, bs=bs, *args, **kwargs)
        c_latent, likelihoods, q_likelihoods, emb_loss, guide_hint = self.apply_condition_encoder(h)

        # ---------- 第一阶段：原作者路径 ----------
        if not self.is_refine:
            N, _, H, W = x.shape
            num_pixels = N * H * W * 64  # 和原 baseline 保持一致
            bpp = sum((torch.log(l).sum() / (-math.log(2.0) * num_pixels)) for l in likelihoods)
            q_bpp = sum((torch.log(l).sum() / (-math.log(2.0) * num_pixels)) for l in q_likelihoods)
            return x, dict(
                c_crossattn=[c],
                c_latent=[c_latent],
                bpp=bpp,
                q_bpp=q_bpp,
                emb_loss=emb_loss,
                guide_hint=guide_hint,
                target=target,
            )

        # ---------- 第二阶段：refine 路径 ----------
        # 读取 H0,W0（原图尺寸），用于裁剪和 bpp 口径
        try:
            if torch.is_tensor(batch.get("orig_size", None)):
                H0, W0 = batch["orig_size"].view(-1).tolist()
            else:
                H0, W0 = batch.get("orig_size", (int(x.shape[-2]), int(x.shape[-1])))
            H0, W0 = int(H0), int(W0)
        except Exception:
            H0, W0 = int(x.shape[-2]), int(x.shape[-1])

        H0 = max(H0, 1); W0 = max(W0, 1)
        N = x.shape[0]
        num_pixels = max(N * H0 * W0, 1)
        bpp = sum((torch.log(l).sum() / (-math.log(2.0) * num_pixels)) for l in likelihoods)
        q_bpp = sum((torch.log(l).sum() / (-math.log(2.0) * num_pixels)) for l in q_likelihoods)

        target_crop = target[..., :H0, :W0]
        self._static_W0_pix = int(W0)

        # 读取/决定扩宽 K_pix（只记录，不在这里对 latent 做扩展）
        extend_k_pix = int(self.extend_k_pix_default)
        try:
            if "extend_k_pix" in batch:
                extend_k_pix = int(batch["extend_k_pix"])
        except Exception:
            pass
        extend_k_pix = max(0, extend_k_pix)
        self._last_extend_k_pix = extend_k_pix

        # 再稳妥记录一次 orig_size
        try:
            if torch.is_tensor(batch.get("orig_size", None)):
                H0, W0 = map(int, batch["orig_size"].view(-1).tolist())
            else:
                H0, W0 = int(x.shape[-2]), int(x.shape[-1])
        except Exception:
            H0, W0 = int(x.shape[-2]), int(x.shape[-1])
        self._static_W0_pix = int(W0)

        # 真正的 latent 扩展在训练 / 采样阶段执行，这里只传回原始 c_latent 和像素扩展量
        K_true_pix = int(self._last_extend_k_pix)

        return x, dict(
            c_crossattn=[c],
            c_latent=[c_latent],
            bpp=bpp,
            q_bpp=q_bpp,
            emb_loss=emb_loss,
            guide_hint=guide_hint,
            target=target_crop,
            orig_size=torch.tensor([H0, W0], device=target.device),
            extend_k_pix=torch.tensor([K_true_pix], device=target.device),
        )

    # --------- 扩宽后前向（diffusion） ---------
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
        self._static_W0_pix = 0
        log = dict()

        z, c = self.get_input(batch, self.first_stage_key, bs=bs)
        target = c["target"]
        log["target"] = (target + 1) / 2
        log["vae_rec"] = (self.decode_first_stage(z) + 1) / 2

        # 这里沿用原作者的 “估算 bpp = q_bpp + 常数偏置”
        bpp = c["q_bpp"] + 0.003418
        if torch.is_tensor(bpp):
            bpp_val = float(bpp.detach().mean())
        else:
            bpp_val = float(bpp)
        bpp_img = [f'{bpp_val:.4f}'] * 4
        log["text"] = (log_txt_as_img((512, 512), bpp_img, size=16) + 1) / 2

        # -------- 第一阶段：不做扩宽，直接采样 --------
        if not self.is_refine:
            c_latent = c["c_latent"][0]
            guide_hint = c["guide_hint"]
            c_txt = c["c_crossattn"][0]
            samples = self.sample_log(
                cond={"c_crossattn": [c_txt], "c_latent": [c_latent], "guide_hint": guide_hint},
                steps=sample_steps,
            )
            x_samples = self.decode_first_stage(samples)
            log["samples"] = (x_samples + 1) / 2
            return log, bpp

        # -------- 第二阶段：在 latent 上右侧扩宽 + 像素融合 --------
        H0, W0 = map(int, c["orig_size"].tolist())
        K_pix = int(c["extend_k_pix"].item())
        K_lat = int(math.ceil(K_pix / float(self.first_stage_downsample))) if K_pix > 0 else 0

        c_latent = c["c_latent"][0]
        guide_hint = c["guide_hint"]
        c_txt = c["c_crossattn"][0]

        b, cch, h, w0_lat = c_latent.shape
        w_ext_lat = w0_lat + max(K_lat, 0)
        shape_ext = (b, self.channels, h, w_ext_lat)

        # 只在这里对 latent 扩展一次（右侧扩宽）
        x_start_ext = self._latent_extend_left_head(c_latent, K_lat)
        t = torch.ones((b,), device=self.device).long() * self.used_timesteps - 1
        noise = torch.randn_like(x_start_ext)
        x_T = self.q_sample(x_start=x_start_ext, t=t, noise=noise)

        steps = self.fixed_step
        cond_full = {
            "c_crossattn": [c_txt],
            "c_latent": [c_latent],
            "guide_hint": guide_hint,
            "orig_size": c["orig_size"],
            "extend_k_pix": c["extend_k_pix"],
        }
        samples_ext = self.sampler.sample(
            steps, shape_ext, cond_full,
            unconditional_guidance_scale=1.0,
            unconditional_conditioning=None, x_T=x_T
        )

        self._static_W0_pix = int(W0)
        x_samples_full = self.decode_first_stage(samples_ext)
        x_samples = x_samples_full[..., :H0, :W0]
        log["samples"] = (x_samples + 1) / 2

        return log, bpp

    # ---------------- 采样（兼容原接口） ----------------
    @torch.no_grad()
    def sample_log(self, cond, steps):
        x_T = cond["c_latent"][0]
        b, c, h, w = x_T.shape
        shape = (b, self.channels, h, w)
        t = torch.ones((b,)).long().to(self.device) * self.used_timesteps - 1
        noise = default(None, lambda: torch.randn_like(x_T))
        x_T = self.q_sample(x_start=x_T, t=t, noise=noise)
        if self.is_refine:
            samples = self.sampler.sample(
                steps, shape, cond, unconditional_guidance_scale=1.0,
                unconditional_conditioning=None, x_T=x_T
            )
        else:
            sampler = SpacedSampler(self)
            samples = sampler.sample(
                steps, shape, cond, unconditional_guidance_scale=1.0,
                unconditional_conditioning=None, x_T=x_T
            )
        return samples

    # ================= 训练 =================
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
        c_latent = cond['c_latent'][0]

        # ================= 第一阶段：完全沿用原作者写法 =================
        if not self.is_refine:
            noise = default(noise, lambda: torch.randn_like(x_start)) + (c_latent - x_start) / self.lamba
            x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
            model_output = self.apply_model(x_noisy, t, cond)

            if self.parameterization == "x0":
                target = x_start
            elif self.parameterization == "eps":
                target = x_start
                model_output = self._predict_xstart_from_eps(x_noisy, t, model_output)
            elif self.parameterization == "v":
                target = self.get_v(x_start, noise, t)
            else:
                raise NotImplementedError()

            loss_simple = self.get_loss(model_output, target, mean=False).mean([1, 2, 3])
            loss_dict.update({f'{prefix}/l_simple': loss_simple.mean()})

            logvar_t = self.logvar[t].to(self.device)
            loss = loss_simple / torch.exp(logvar_t) + logvar_t
            if self.learn_logvar:
                loss_dict.update({f'{prefix}/l_gamma': loss.mean()})
                loss_dict.update({'logvar': self.logvar.data.mean()})

            loss = self.l_guide_weight * loss.mean()

            # bpp / emb
            loss_bpp = cond['bpp']; guide_bpp = cond['q_bpp']
            loss_dict.update({f'{prefix}/l_bpp': loss_bpp.mean()})
            loss_dict.update({f'{prefix}/q_bpp': guide_bpp.mean()})
            loss += self.l_bpp_weight * loss_bpp

            loss_emb = cond['emb_loss']
            loss_dict.update({f'{prefix}/l_emb': loss_emb.mean()})
            loss += self.l_bpp_weight * loss_emb

            # latent 引导
            loss_guide = self.get_loss(c_latent, x_start)
            loss_dict.update({f'{prefix}/l_guide': loss_guide.mean()})
            loss += self.l_guide_weight * loss_guide

            loss_dict.update({f'{prefix}/loss': loss})
            return loss, loss_dict

        # ================= 第二阶段：refine + latent 扩宽 =================
        H0, W0 = map(int, cond["orig_size"].tolist())
        K_pix = int(cond["extend_k_pix"].item())
        K_lat = int(math.ceil(K_pix / float(self.first_stage_downsample))) if K_pix > 0 else 0

        # 扩宽 latent（右侧扩宽： [x, head]）
        x_start_lat = c_latent                      # 原始 latent
        x_start_ext = self._latent_head_extend(x_start_lat, K_lat)
        b, cch, h, w_ext = x_start_ext.shape
        W_lat = x_start_lat.shape[-1]

        # 采样（在扩展后的 latent 上做扩散）
        noise = default(noise, lambda: torch.randn_like(x_start_ext))
        x_T_ext = self.q_sample(x_start=x_start_ext, t=t, noise=noise)
        steps = self.fixed_step
        samples_ext = self.sampler.sample_grad(
            steps, (b, self.channels, h, w_ext), cond,
            unconditional_guidance_scale=1.0,
            unconditional_conditioning=None, x_T=x_T_ext
        )

        # 解码（像素端单侧融合）并裁回 W0
        self._static_W0_pix = int(W0)
        model_output_pix = self.decode_first_stage_with_grad(samples_ext)
        model_output_pix = self._align_to_target(model_output_pix, cond['target'])

        # ---------- latent 简单项：主体 + 扩展 ----------
        # 主体区域：对齐原始 latent
        samples_core = samples_ext[..., :W_lat]
        loss_lat_main = self.get_loss(samples_core, x_start_lat, mean=False).mean([1, 2, 3])

        # 扩宽区域：对齐复制的 head（参与，但权重略低，避免“过度拉扯”）
        if K_lat > 0:
            samples_tail = samples_ext[..., W_lat:]
            target_tail = x_start_lat[..., :K_lat]
            loss_lat_ext = self.get_loss(samples_tail, target_tail, mean=False).mean([1, 2, 3])
        else:
            loss_lat_ext = torch.zeros_like(loss_lat_main)

        # 将扩展部分以 0.5 的系数合入，既保证约束，又不过分影响主体
        loss_simple = loss_lat_main + 0.5 * loss_lat_ext
        loss_dict.update({f'{prefix}/l_simple': loss_simple.mean()})
        loss = self.l_guide_weight * loss_simple.mean()

        # ---------- 像素域损失 ----------
        loss_mse = self.get_loss(model_output_pix, cond['target'], mean=False).mean([1, 2, 3])
        loss_dict.update({f'{prefix}/l_mse': loss_mse.mean()})
        loss += self.l_guide_weight * loss_mse.mean()

        with torch.cuda.amp.autocast(enabled=False):
            loss_lpips = self.perceptual_loss(model_output_pix.float(), cond['target'].float())
        loss_dict.update({f'{prefix}/l_lpips': loss_lpips.mean()})
        loss += self.l_guide_weight * loss_lpips * 0.5

        # ---------- bpp / emb（沿用第一阶段逻辑） ----------
        loss_bpp = cond['bpp']; guide_bpp = cond['q_bpp']
        loss_dict.update({f'{prefix}/l_bpp': loss_bpp.mean()})
        loss_dict.update({f'{prefix}/q_bpp': guide_bpp.mean()})
        loss += self.l_bpp_weight * loss_bpp

        loss_emb = cond['emb_loss']
        loss_dict.update({f'{prefix}/l_emb': loss_emb.mean()})
        loss += self.l_bpp_weight * loss_emb

        loss_dict.update({f'{prefix}/loss': loss})
        return loss, loss_dict

    def training_step(self, batch, batch_idx):
        dm = getattr(self.trainer, "datamodule", None)
        bsz = getattr(dm, "train_batch_size", None)
        if bsz is None:
            try:
                bsz = batch[self.first_stage_key].shape[0]
            except Exception:
                first_val = next(iter(batch.values()))
                bsz = first_val.shape[0] if hasattr(first_val, "shape") else len(first_val)

        loss, loss_dict = self.shared_step(batch)

        # 安全转成标量
        safe_logs = {}
        for k, v in loss_dict.items():
            if torch.is_tensor(v):
                try:
                    safe_v = v.detach()
                    if safe_v.ndim > 0:
                        safe_v = safe_v.mean().detach()
                    safe_logs[k] = float(safe_v.item())
                except Exception:
                    safe_logs[k] = float(v.mean().detach().item())
            else:
                safe_logs[k] = v

        # 1) 所有细节指标只写 logger，不上进度条
        self.log_dict(
            safe_logs,
            prog_bar=False,
            logger=True,
            on_step=True,
            on_epoch=True,
            batch_size=bsz,
        )

        # 2) 关心的少数指标挂到进度条
        self.log(
            "train/loss",
            loss,
            prog_bar=True,
            logger=True,
            on_step=True,
            on_epoch=True,
            batch_size=bsz,
        )

        prefix = 'T' if self.training else 'V'
        bpp_key = f"{prefix}/l_bpp"
        if bpp_key in loss_dict:
            self.log(
                "train/bpp",
                loss_dict[bpp_key],
                prog_bar=True,
                logger=True,
                on_step=True,
                on_epoch=True,
                batch_size=bsz,
            )

        # 最近一次 val 的指标也挂一挂
        if not math.isnan(self.last_val_psnr):
            self.log(
                "train/psnr",
                torch.tensor(self.last_val_psnr, device=self.device),
                prog_bar=True,
                logger=False,
                on_step=True,
                on_epoch=False,
                batch_size=bsz,
            )
        if not math.isnan(self.last_val_ms_ssim):
            self.log(
                "train/ms_ssim",
                torch.tensor(self.last_val_ms_ssim, device=self.device),
                prog_bar=True,
                logger=False,
                on_step=True,
                on_epoch=False,
                batch_size=bsz,
            )
        if not math.isnan(self.last_val_fid):
            self.log(
                "train/fid",
                torch.tensor(self.last_val_fid, device=self.device),
                prog_bar=True,
                logger=False,
                on_step=True,
                on_epoch=False,
                batch_size=bsz,
            )

        self.log(
            "global_step",
            float(self.global_step),
            prog_bar=False,
            logger=True,
            on_step=True,
            on_epoch=False,
            batch_size=bsz,
        )
        if getattr(self, "use_scheduler", False):
            lr = self.optimizers().param_groups[0]['lr']
            self.log(
                "lr_abs",
                float(lr),
                prog_bar=False,
                logger=True,
                on_step=True,
                on_epoch=False,
                batch_size=bsz,
            )

        return loss

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        sample_steps = int(getattr(self, "val_sample_steps", 5))
        do_metrics = bool(getattr(self, "val_eval_metrics", True)) and bool(getattr(self, "calculate_metrics", None))
        max_per_batch = int(getattr(self, "val_max_per_batch", -1))
        try:
            B = batch[self.first_stage_key].shape[0]
        except Exception:
            B = next(iter(batch.values())).shape[0]
        n = B if max_per_batch < 0 else min(B, max_per_batch)
        save_dir = os.path.join(self.logger.save_dir, "validation", f"epoch{self.current_epoch:04d}")
        os.makedirs(save_dir, exist_ok=True)
        bpp_vals = []; psnr_vals, msssim_vals, lpips_vals, fid_vals = [], [], [], []
        with torch.inference_mode(), torch.cuda.amp.autocast(enabled=True):
            for i in range(n):
                one = {}
                for k, v in batch.items():
                    if torch.is_tensor(v) and v.dim() >= 1 and v.shape[0] == B:
                        one[k] = v[i:i+1]
                    else:
                        one[k] = v
                log, bpp = self.log_images(one, bs=1, sample_steps=sample_steps)
                if torch.is_tensor(bpp):
                    bpp_vals.append(float(bpp.detach().mean()))
                else:
                    bpp_vals.append(float(bpp))

                x_vis = (log["samples"][0].detach().cpu().numpy().transpose(1, 2, 0) * 255).clip(0, 255).astype(np.uint8)
                out_png = os.path.join(save_dir, f"{batch_idx*10+i}.png")
                Image.fromarray(x_vis).save(out_png)

                if do_metrics:
                    x = log["samples"].float(); y = log["target"].float(); x, y = self._align_pair(x, y)
                    if "psnr" in self.metric_funcs:    psnr_vals.append(float(self.metric_funcs["psnr"](x, y)))
                    if "ms_ssim" in self.metric_funcs: msssim_vals.append(float(self.metric_funcs["ms_ssim"](x, y)))
                    if "lpips" in self.metric_funcs:   lpips_vals.append(float(self.metric_funcs["lpips"](x, y)))
                    if "fid" in self.metric_funcs:     fid_vals.append(float(self.metric_funcs["fid"](x, y)))
                del log, bpp
        out = [sum(bpp_vals) / max(len(bpp_vals), 1)]
        if do_metrics:
            if psnr_vals:   out.append(sum(psnr_vals) / len(psnr_vals))
            if msssim_vals: out.append(sum(msssim_vals) / len(msssim_vals))
            if lpips_vals:  out.append(sum(lpips_vals) / len(lpips_vals))
            if fid_vals:    out.append(sum(fid_vals) / len(fid_vals))
        return out

    def _latent_head_extend(self, x: torch.Tensor, K_lat: int) -> torch.Tensor:
        """在 latent 域做“右侧扩宽”：把前 K_lat 列（左侧 head）复制到最右侧。
        x: [B,C,H,W],  K_lat >= 0
        out: [B,C,H,W + K_lat] = [x, head(K_lat)]
        """
        if K_lat is None or K_lat <= 0:
            return x
        head = x[..., :K_lat]
        return torch.cat([x, head], dim=-1)

    def on_validation_epoch_start(self):
        self.preprocess_model.quantize.reset_usage()
        return super().on_validation_epoch_start()

    def validation_epoch_end(self, outputs: EPOCH_OUTPUT):
        if not outputs:
            return
        lens = {len(o) for o in outputs}
        if len(lens) != 1:
            L = min(lens); outputs = [o[:L] for o in outputs]
        else:
            L = lens.pop()

        arr = torch.tensor(outputs, dtype=torch.float32)
        avg_out = arr.mean(dim=0)

        self.log("avg_bpp", float(avg_out[0]), prog_bar=True, logger=True, on_step=False, on_epoch=True, batch_size=1)
        self.log("val/bpp", float(avg_out[0]), prog_bar=False, logger=True, on_step=False, on_epoch=True, batch_size=1)
        self.log("val_bpp", float(avg_out[0]), prog_bar=False, logger=True, on_step=False, on_epoch=True, batch_size=1)
        usage = self.preprocess_model.quantize.get_usage()
        self.log("usage", usage, prog_bar=True, logger=True, on_step=False, on_epoch=True, batch_size=1)

        names = ["bpp"]; metrics_order = ["psnr", "ms_ssim", "lpips", "fid"]
        if getattr(self, "calculate_metrics", None):
            names += [m for m in metrics_order if m in self.calculate_metrics]
        names = names[:len(avg_out)]

        metric_map = {}
        for i, name in enumerate(names[1:], start=1):
            val = float(avg_out[i])
            metric_map[name] = val
            self.log(f"avg_{name}", val, prog_bar=True, logger=True, on_step=False, on_epoch=True, batch_size=1)
            self.log(f"val/{name}", val, prog_bar=False, logger=True, on_step=False, on_epoch=True, batch_size=1)
            self.log(f"val_{name.replace('/', '_')}", val, prog_bar=False, logger=True, on_step=False, on_epoch=True, batch_size=1)

        self.last_val_psnr = metric_map.get("psnr", float('nan'))
        self.last_val_ms_ssim = metric_map.get("ms_ssim", float('nan'))
        self.last_val_fid = metric_map.get("fid", float('nan'))

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
                    if dst_key_old in self.state_dict().keys():
                        dst_key = dst_key_old
                    elif dst_key_new in self.state_dict().keys():
                        dst_key = dst_key_new
                    else:
                        continue
                    if ckpt_base['state_dict'][key].shape != self.state_dict()[dst_key].shape:
                        if len(ckpt_base['state_dict'][key].shape) == 1:
                            control_dim = self.state_dict()[dst_key].size(0)
                            ckpt_base['state_dict'][dst_key] = torch.cat(
                                [ckpt_base['state_dict'][key], ckpt_base['state_dict'][key]], dim=0
                            )[:control_dim]
                        else:
                            control_dim_0 = self.state_dict()[dst_key].size(0)
                            control_dim_1 = self.state_dict()[dst_key].size(1)
                            ckpt_base['state_dict'][dst_key] = torch.cat(
                                [ckpt_base['state_dict'][key], ckpt_base['state_dict'][key]], dim=1
                            )[:control_dim_0, :control_dim_1, ...]
                    else:
                        ckpt_base['state_dict'][dst_key] = ckpt_base['state_dict'][key]
        res_sync = self.load_state_dict(ckpt_base['state_dict'], strict=False)
        print(f'[{len(res_sync.missing_keys)} keys are missing from the model (hint processing and cross connections included)]')


def vae_blend_net_bool(x) -> bool:
    # 兼容 YAML 中的各种真假写法
    if isinstance(x, bool): return x
    if isinstance(x, (int, float)): return bool(x)
    if isinstance(x, str): return x.lower() not in ("0", "false", "off", "no", "")
    return True
