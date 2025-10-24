# -*- coding: utf-8 -*-
"""
RDEIC (seam aligned, band-free) — 静态扩展版

要点：
- 静态扩展：在去噪开始前，对 c_latent 与 guide_hint 做一次性右侧拼尾（W→W+K），
  之后整条去噪轨迹都在 W+K 上进行；损失目标也统一扩到 W+K；最终像素域裁回 W0。
- 去噪阶段不再每步 roll/临时拼尾；若静态扩展激活，apply_model 直接在扩展宽上前向。
- VAE 解码：latent 右侧再扩边(≈Wz/4)→瓦片解码→横向 feather；像素域环向融合时
  只写回左带[0:K]，参考带来自扩充区[W0:W0+K]；回卷相位后由上层裁回 W0。
- Debug：ali 仍为融合前拼 tile 图；mix 改为融合权值热力图（白→红 0→1）。

其它接口/名字保持不变。
"""

import os
import math
from typing import Mapping, Any, Tuple

import numpy as np
import pyiqa

import torch
import torch as th
import torch.nn as nn
import torch.nn.functional as F

from pathlib import Path
from PIL import Image

# 启用 UNet 水平环绕卷积
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


# ------------------------ 融合权重小网（像素端可选，不默认启用） ------------------------
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
            assert context_dim is not None, 'Fool!! You forgot to include the dimension of your cross-attention conditioning...'

        if context_dim is not None:
            assert use_spatial_transformer, 'Fool!! You forgot to use the spatial transformer for your cross-attention conditioning...'
            from omegaconf.listconfig import ListConfig
            if type(context_dim) == ListConfig:
                context_dim = list(context_dim)

        if num_heads_upsample == -1:
            num_heads_upsample = num_heads

        if num_heads == -1:
            assert num_head_channels != -1, 'Either num_heads or num_head_channels has to be set'

        if num_head_channels == -1:
            assert num_heads != -1, 'Either num_heads or num_head_channels has to be set'

        self.image_size = image_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        if isinstance(num_res_blocks, int):
            self.num_res_blocks = len(channel_mult) * [num_res_blocks]
        else:
            if len(num_res_blocks) != len(channel_mult):
                raise ValueError("provide num_res_blocks either as an int (globally constant) or "
                                 "as a list/tuple (per-level) with the same length as channel_mult")
            self.num_res_blocks = num_res_blocks
        if disable_self_attentions is not None:
            # should be a list of booleans, indicating whether to disable self-attention in TransformerBlocks or not
            assert len(disable_self_attentions) == len(channel_mult)
        if num_attention_blocks is not None:
            assert len(num_attention_blocks) == len(self.num_res_blocks)
            assert all(map(lambda i: self.num_res_blocks[i] >= num_attention_blocks[i], range(len(num_attention_blocks))))
            print(f"Constructor of UNetModel received num_attention_blocks={num_attention_blocks}. "
                  f"This option has LESS priority than attention_resolutions {attention_resolutions}, "
                  f"i.e., in cases where num_attention_blocks[i] > 0 but 2**i not in attention_resolutions, "
                  f"attention will still not be set.")

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
                print("setting up linear c_adm embedding layer")
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
                    if legacy:
                        dim_head = ch // num_heads if use_spatial_transformer else num_head_channels
                    if exists(disable_self_attentions):
                        disabled_sa = disable_self_attentions[level]
                    else:
                        disabled_sa = False

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
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            down=True,
                        )
                        if resblock_updown
                        else Downsample(
                            ch,
                            conv_resample, dims=dims, out_channels=out_ch
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
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                out_channels=ch,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
            AttentionBlock(
                ch,
                use_checkpoint=use_checkpoint,
                num_heads=num_heads,
                num_head_channels=dim_head,
                use_new_attention_order=use_new_attention_order,
            ) if not use_spatial_transformer else SpatialTransformer(  # always uses a self-attn
                ch, num_heads, dim_head, depth=transformer_depth, context_dim=context_dim,
                disable_self_attn=disable_middle_self_attn, use_linear=use_linear_in_transformer,
                use_checkpoint=use_checkpoint
            ),
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
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
        self.emb_layers = nn.Sequential(nn.SiLU(),
                                        linear(emb_channels, 2 * self.out_channels if use_scale_shift_norm else self.out_channels))
        self.out_layers = nn.Sequential(normalization(self.out_channels), nn.SiLU(), nn.Dropout(p=dropout),
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
        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None]
        if self.use_scale_shift_norm:
            out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
            scale, shift = th.chunk(emb_out, 2, dim=1)
            h = out_norm(h) * (1 + scale) + shift; h = out_rest(h)
        else:
            h = h + emb_out; h = self.out_layers(h)
        return self.skip_connection(x) + h


# ============================== RDEIC ==============================
class RDEIC(LatentDiffusion):

    # 去噪阶段 latent 拼尾尺度：W/TAIL_DEN（作为默认 K 的基准）
    TAIL_DEN = 12  # 改为 16 即采用 1/16

    def __init__(self,
                 control_stage_config: Mapping[str, Any],
                 sd_locked: bool,
                 is_refine: bool,
                 fixed_step: int,
                 learning_rate: float,
                 l_bpp_weight: float,
                 l_guide_weight: float,
                 used_timesteps: int,
                 sync_path: str,
                 synch_control: bool,
                 ckpt_path_pre: str,
                 preprocess_config: Mapping[str, Any],
                 calculate_metrics: Mapping[str, Any],
                 vae_blend_enabled: bool = True,
                 vae_blend_k: int = 64,
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

        if self.is_refine:
            try:
                enable_feature_hwrap_for_unet(self.model.diffusion_model, enabled=True)
                print("[feature_hwrap] UNet horizontal wrap: ENABLED (refine)")
            except Exception as e:
                print("[feature_hwrap] enable failed:", e)
        else:
            print("[feature_hwrap] horizontal wrap: DISABLED (not refine)")

        # —— 默认关闭 latent 侧融合，仅像素端可选 —— #
        self.vae_blend_enabled = bool(vae_blend_enabled)
        self.train_blend_net = bool(train_blend_net)

        self.blend_net_pixel  = BlendWeightNet(in_channels=3)

        if (not self.is_refine) or (not self.train_blend_net):
            for p in self.blend_net_pixel.parameters():  p.requires_grad = False

        self.vae_tile_lat = 128
        self.vae_tile_overlap = 32

        if vae_blend_k not in (None, 0):
            print("[blend] NOTE: vae_blend_k ignored. Using Auto-K.]")

        # 估计像素/latent 宽度比例（默认 SD VAE 为 8）
        self.first_stage_downsample = getattr(self, 'first_stage_downsample', 8)

        # ---- Debug 捕获（x_samples 解码同管线抓帧）----
        self._dbg_cap = None         # dict: {'capture': True, 'ali': tensor, 'mix': tensor}
        self._dbg_saved_np = None    # dict: {'ali': np.uint8 HxWx3, 'mix': np.uint8 HxWx3}

        # ---- 静态扩展状态 ----
        self._static_ext_active = False
        self._static_W0_latent = 0
        self._static_K_lat = 0
        self._static_W0_pix = 0

    # ---------------- 工具：K 自适应 + 权值 ----------------
    @staticmethod
    def _auto_k_latent(W_lat: int) -> int:
        # 保留接口但不再使用（VAE 解码扩边时使用较大K）
        return max(8, int(W_lat // 32))

    @staticmethod
    def _auto_k_pixel(W_pix: int) -> int:
        # 经验：像素//32（或//64）
        return max(1, int(W_pix // 32))

    @staticmethod
    def _linear_alpha(k: int, device, dtype) -> torch.Tensor:
        if k <= 1:
            return torch.zeros((1,1,1,max(1,k)), device=device, dtype=dtype)
        j = torch.arange(k, device=device, dtype=dtype)  # 0..k-1
        a = j / float(k - 1)  # 0..1，缝→内
        return a.view(1, 1, 1, k)

    @staticmethod
    def _to_uint8_image(x: torch.Tensor) -> np.ndarray:
        """
        x: [B,3,H,W] in [-1,1] 或 [0,1] 都可。输出 uint8 HxWx3。
        """
        if x.dim() == 4:
            x = x[0]
        x = x.detach().float().cpu()
        if x.min() < 0.0 or x.max() > 1.0:
            x = (x + 1) / 2
        x = x.clamp(0, 1).permute(1, 2, 0).numpy()
        return (x * 255.0 + 0.5).astype(np.uint8)

    @staticmethod
    def _wrap_tail_lat(x: torch.Tensor, k_lat: int) -> torch.Tensor:
        # x: [B,C,H,W] ; 右侧拼接 k_lat 列
        if k_lat <= 0:
            return x
        return torch.cat([x, x[..., :k_lat]], dim=-1)

    # =================== 像素域融合（只写回左带 [0:K]；参考带取扩充区 [W0:W0+K]） ===================
    @staticmethod
    def _pixel_blend_left_only_with_right_tail(img: torch.Tensor, W0: int, K: int, use_net: bool,
                                               blend_net: BlendWeightNet, train_blend_net: bool,
                                               dbg_store: dict = None) -> torch.Tensor:
        """
        img: [B,3,H,W_wrap]  VAE 解码后（包含扩边尾巴）
        W0:  原图像素宽（未扩边）
        K:   融合带宽（像素）
        逻辑：只回写左带 [0:K]，参考带来自扩充区 [W0:W0+K]；不动右带与尾巴。
        dbg_store: 若非空，用于保存权值热力图（白→红 0→1）
        """
        B, C, H, W = img.shape
        if W0 <= 0 or W <= W0 or K <= 0:
            return img
        K = max(1, min(K, W0, W - W0))  # 不能超过扩充长度与 W0

        out = img  # 原地写回
        L = img[..., :K]           # 左带 [0:K]
        R_ref = img[..., W0:W0+K]  # 扩充区参考带 [W0:W0+K]

        # 线性窗 a: 0..1（缝→内），缝处更贴 R_ref，带内深处更贴 L
        a_lin = torch.linspace(0, 1, K, device=img.device, dtype=img.dtype).view(1,1,1,K)

        if use_net and (blend_net is not None):
            with torch.set_grad_enabled(train_blend_net):
                a_raw = blend_net(L, R_ref)  # [B,1,1,K] in (0,1)
            # 稳定缩放：避免过弱融合（下限 0.7）
            scale = 0.7 + 0.3 * a_raw
            a = (a_lin * scale).clamp(0.0, 1.0)
        else:
            a = a_lin

        # Debug：保存热力图（白→红 0→1；只标注左带[0:K]）
        if isinstance(dbg_store, dict):
            B_, _, _, W_ = img.shape
            heat = img.new_zeros((B_, 3, H, W_))
            # 颜色映射：v∈[0,1] -> (1, 1-v, 1-v)
            v = a.expand(B_, 1, H, K)                 # [B,1,H,K]
            r = torch.ones_like(v)                    # 1
            g = 1.0 - v
            b = 1.0 - v
            heat[..., :K] = torch.cat([r, g, b], dim=1)  # 左带上着色
            dbg_store['mix'] = heat.clamp(0,1).detach().clone()

        # 只写回左侧 [0:K]
        out[..., :K] = (1.0 - a) * R_ref + a * L
        return out

    # ------- 像素域融合入口：只在 W0 内写回左带，参考带来自 W0:W0+K -------
    def circular_blend_pixel(self, img: torch.Tensor, W0_pix: int, dbg_store: dict = None) -> torch.Tensor:
        if not (self.is_refine and self.vae_blend_enabled):
            return img
        W_wrap = int(img.shape[-1])
        W0 = int(W0_pix)

        # 扩充长度
        ext = max(0, W_wrap - W0)
        if ext <= 0:
            return img  # 没有扩边则无需融合

        # 选 K：不超过 auto_k_pixel(W0) 且不超过扩充长度 ext
        K_auto = self._auto_k_pixel(W0)
        K = max(1, min(K_auto, ext, W0 // 2))

        use_net = bool(self.train_blend_net)
        return self._pixel_blend_left_only_with_right_tail(
            img, W0=W0, K=K, use_net=use_net,
            blend_net=self.blend_net_pixel, train_blend_net=self.train_blend_net,
            dbg_store=dbg_store
        )

    # ---------------- 解码：seam 对齐 + wrap/tile + 像素域(左写回)融合 + **回卷到原相位** ----------------
    def _vae_decode_wrap_tiled(self, z: torch.Tensor, with_grad: bool, hshift_pix: int = 0) -> torch.Tensor:
        """
        z: [B,Cz,Hz,Wz]（Wz 为当前 latent 宽，静态扩展后即 W+K）
        解码/拼 tile 用扩边后的 z_wrap；像素域融合只在原图宽 W0 内进行，且仅写回左带 [0:K]。
        """
        B, Cz, Hz, Wz = z.shape

        # 1) 把 seam 对齐到列0（latent 域临时相位）
        scale = int(getattr(self, 'first_stage_downsample', 8))  # 典型 8
        r_lat = int(round(hshift_pix / float(scale)))
        if (r_lat % Wz) != 0:
            z = torch.roll(z, shifts=(-r_lat), dims=-1)

        # 2) latent 右侧扩边（供 VAE 解码/拼 tile 使用）
        K_lat = max(int(Wz // 4), self._auto_k_latent(Wz))  # 保持较大感受野
        z_wrap = torch.cat([z, z[..., :K_lat]], dim=-1).contiguous()
        Wz_wrap = z_wrap.shape[-1]

        # 3) 瓦片解码
        tile = int(getattr(self, 'vae_tile_lat', 128))
        ov   = int(getattr(self, 'vae_tile_overlap', 32))
        tile = max(1, min(tile, Wz_wrap))
        ov   = max(0, min(ov, tile - 1))
        step = tile - ov

        def decode_chunk(zc: torch.Tensor) -> torch.Tensor:
            return self.first_stage_model.decode(zc / self.scale_factor)

        chunks = []
        x0 = 0
        while x0 < Wz_wrap:
            x1 = min(x0 + tile, Wz_wrap)
            if (x1 - x0) < tile and x0 != 0:
                x0 = max(0, Wz_wrap - tile); x1 = Wz_wrap
            z_chunk = z_wrap[..., x0:x1]
            img_chunk = decode_chunk(z_chunk)  # [B,3,Hpix,Wpix_chunk]
            chunks.append(img_chunk)
            if x1 == Wz_wrap:
                break
            x0 += step

        # 4) 横向 feather 合成
        out = chunks[0]
        for i in range(1, len(chunks)):
            left = out; right = chunks[i]
            ov_pix = int(round(right.shape[-1] * (ov / float(tile))))
            if ov_pix <= 0:
                out = torch.cat([left, right], dim=-1); continue
            left_keep = left[..., :-ov_pix]; left_ov = left[..., -ov_pix:]
            right_ov = right[..., :ov_pix]; right_keep = right[..., ov_pix:]
            ramp = torch.linspace(0, 1, ov_pix, device=right.device, dtype=right.dtype)[None, None, None, :]
            wl = 1.0 - ramp; wr = ramp
            fuse = left_ov * wl + right_ov * wr
            out = torch.cat([left_keep, fuse, right_keep], dim=-1)
        img_wrap = out  # 宽度 ≈ (Wz + K_lat) * scale

        # —— 捕获“同管线”的 ali：瓦片拼接后、像素融合/回卷前（仅当 _dbg_cap 打开时）
        cap = getattr(self, "_dbg_cap", None)
        if isinstance(cap, dict) and cap.get('capture', False):
            try:
                cap['ali'] = img_wrap.detach().clone()
            except Exception:
                pass

        # 5) 像素域环向融合 —— 仅在原图宽 W0_pix 内 **左写回**；不动右侧与尾巴
        # 若静态扩展激活，则 W0_pix 以记录的原宽为准；否则退回当前 Wz 推断
        if self._static_ext_active and self._static_W0_pix > 0:
            W0_pix = int(self._static_W0_pix)
        else:
            W0_pix = int((Wz - self._static_K_lat + self._static_K_lat) * scale)  # 等价于 Wz*scale，保底
        if self.vae_blend_enabled:
            img_wrap = self.circular_blend_pixel(img_wrap, W0_pix, dbg_store=cap if isinstance(cap, dict) else None)

        # 6) **必做**：回卷到原始相位（撤销第 1 步的临时相位）
        if (hshift_pix % img_wrap.shape[-1]) != 0:
            img_wrap = torch.roll(img_wrap, shifts=(+int(hshift_pix)), dims=-1)

        return img_wrap  # 不在此处裁剪，由上层对齐 target

    # —— 仅内部调用：带 seam 对齐的解码 ——
    def _decode_with_shift(self, z, with_grad: bool, hshift_pix: int):
        if self.is_refine:
            return self._vae_decode_wrap_tiled(z, with_grad=with_grad, hshift_pix=hshift_pix)
        else:
            return self.first_stage_model.decode(z / self.scale_factor) if with_grad else super().decode_first_stage(z)

    @torch.no_grad()
    def decode_first_stage(self, z):
        if self.is_refine:
            return self._vae_decode_wrap_tiled(z, with_grad=False, hshift_pix=0)
        else:
            return super().decode_first_stage(z)

    def decode_first_stage_with_grad(self, z):
        if self.is_refine:
            return self._vae_decode_wrap_tiled(z, with_grad=True, hshift_pix=0)
        else:
            return self.first_stage_model.decode(z / self.scale_factor)

    # ------------------- 公共逻辑 -------------------
    def apply_condition_encoder(self, x):
        c_latent, likelihoods, q_likelihoods, emb_loss, guide_hint = self.preprocess_model(x)
        return c_latent, likelihoods, q_likelihoods, emb_loss, guide_hint

    @torch.no_grad()
    def apply_condition_compress(self, x, stream_path, H, W):
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
        target, x, h, c = super().get_input(batch, self.first_stage_key, bs=bs, *args, **kwargs)
        c_latent, likelihoods, q_likelihoods, emb_loss, guide_hint = self.apply_condition_encoder(h)
        N, _, H, W = x.shape
        num_pixels = N * H * W * 64
        bpp = sum((torch.log(likelihood).sum() / (-math.log(2) * num_pixels)) for likelihood in likelihoods)
        q_bpp = sum((torch.log(likelihood).sum() / (-math.log(2) * num_pixels)) for likelihood in q_likelihoods)
        hshift_pix = batch.get('hshift_pix', None)
        if hshift_pix is None:
            hshift = torch.tensor(0, device=target.device)
        else:
            hshift = hshift_pix if torch.is_tensor(hshift_pix) else torch.tensor(int(hshift_pix), device=target.device)
        return x, dict(c_crossattn=[c], c_latent=[c_latent], bpp=bpp, q_bpp=q_bpp, emb_loss=emb_loss,
                       guide_hint=guide_hint, target=target, orig_size=torch.tensor([H, W], device=target.device),
                       hshift_pix=hshift)

    # ---------- 静态扩展构建：对 c_latent 与 guide_hint 一次性右侧拼尾 ----------
    def _build_static_ext_cond(self, cond: dict) -> Tuple[int, int]:
        """
        返回：(K_lat, W0_latent)
        """
        # 已扩展则直接返回
        c_lat = cond['c_latent'][0]
        guide = cond['guide_hint']
        W0 = c_lat.shape[-1]
        L = 8  # 总下采样因子
        raw_k = max(1, W0 // int(self.TAIL_DEN))
        K_lat = raw_k + ((L - ((W0 + raw_k) % L)) % L)  # 让 W0+K_lat ≡ 0 (mod 8)
        K_lat = min(K_lat, max(1, W0 // 2))
        c_lat_ext = self._wrap_tail_lat(c_lat, K_lat)
        guide_ext = self._wrap_tail_lat(guide, K_lat)

        cond['c_latent'] = [c_lat_ext]
        cond['guide_hint'] = guide_ext

        # 标记静态扩展状态
        self._static_ext_active = True
        self._static_W0_latent = int(W0)
        self._static_K_lat = int(K_lat)
        scale = int(getattr(self, 'first_stage_downsample', 8))
        self._static_W0_pix = int(W0 * scale)
        return K_lat, W0

    # —— 每个去噪步：静态扩展开启时不再临时拼尾；否则保持原策略（兼容）
    def apply_model(self, x_noisy, t, cond, *args, **kwargs):
        diffusion_model = self.model.diffusion_model
        cond_txt = torch.cat(cond['c_crossattn'], 1)
        guide_hint = cond['guide_hint']

        if self._static_ext_active:
            # 已静态扩展：直接前向
            eps = self.control_model(
                x=x_noisy, timesteps=t, context=cond_txt,
                guide_hint=guide_hint, base_model=diffusion_model
            )
            return eps
        else:
            # 兼容路径：临时拼尾
            W = x_noisy.shape[-1]
            L = 8
            raw_k = max(1, W // 12)
            K_lat = raw_k + ((L - ((W + raw_k) % L)) % L)
            K_lat = min(K_lat, max(1, W // 2))
            x_wrap = self._wrap_tail_lat(x_noisy, K_lat)
            hint_wrap = self._wrap_tail_lat(guide_hint, K_lat)
            eps_wrap = self.control_model(
                x=x_wrap, timesteps=t, context=cond_txt,
                guide_hint=hint_wrap, base_model=diffusion_model
            )
            eps = eps_wrap[..., :W]
            return eps

    def apply_model_unconditional(self, x_noisy, t, cond, *args, **kwargs):
        diffusion_model = self.model.diffusion_model
        cond_txt = torch.cat(cond['c_crossattn'], 1)

        if self._static_ext_active:
            eps = self.control_model.forward_unconditional(
                x=x_noisy, timesteps=t, context=cond_txt, base_model=diffusion_model
            )
            return eps
        else:
            W = x_noisy.shape[-1]
            K_lat = max(1, W // int(self.TAIL_DEN))
            x_wrap = self._wrap_tail_lat(x_noisy, K_lat)
            eps_wrap = self.control_model.forward_unconditional(
                x=x_wrap, timesteps=t, context=cond_txt, base_model=diffusion_model
            )
            eps = eps_wrap[..., :W]
            return eps

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

    @torch.no_grad()
    def log_images(self, batch, sample_steps=5, bs=2):
        # 清理/初始化静态扩展状态
        self._static_ext_active = False
        self._static_W0_latent = 0
        self._static_K_lat = 0
        self._static_W0_pix = 0

        log = dict()
        z, c = self.get_input(batch, self.first_stage_key, bs=bs)
        bpp = c["q_bpp"] + 0.003418
        bpp_val = float(bpp.detach().mean()); bpp_img = [f'{bpp_val:.4f}'] * 4
        c_latent = c["c_latent"][0]; guide_hint = c['guide_hint']; target = c["target"]; c_txt = c["c_crossattn"][0]
        hshift_pix = int(c.get('hshift_pix', torch.tensor(0)).item())
        log["target"] = (target + 1) / 2

        # —— 静态扩展：进轨迹前做一次性扩展
        K_lat, W0_lat = self._build_static_ext_cond(c)
        c_latent_ext = c["c_latent"][0]

        # 直解码（展示基础重建）——使用未采样的 z 做解码（但这里 z 不是 c_latent；保持和原逻辑一致）
        vae_rec = self._decode_with_shift(z, with_grad=False, hshift_pix=hshift_pix)
        vae_rec = self._align_to_target(vae_rec, target)
        log["vae_rec"] = (vae_rec + 1) / 2
        log["text"] = (log_txt_as_img((512, 512), bpp_img, size=16) + 1) / 2

        if not sample_steps or sample_steps <= 0:
            log["samples"] = log["vae_rec"]; return log, bpp

        # —— 采样一路（fixed_step）；形状改为扩展后的宽
        b, cch, h, w_ext = c_latent_ext.shape
        shape = (b, self.channels, h, w_ext)
        t = torch.ones((b,)).long().to(self.device) * self.used_timesteps - 1
        noise = default(None, lambda: torch.randn_like(c_latent_ext))
        x_T = self.q_sample(x_start=c_latent_ext, t=t, noise=noise)

        if self.is_refine:
            samples_ext = self.sampler.sample(self.fixed_step, shape, c, unconditional_guidance_scale=1.0,
                                              unconditional_conditioning=None, x_T=x_T)
        else:
            sampler = SpacedSampler(self)
            samples_ext = sampler.sample(sample_steps, shape, c, unconditional_guidance_scale=1.0,
                                         unconditional_conditioning=None, x_T=x_T)

        # 打开“同管线抓帧”开关，供 _vae_decode_wrap_tiled 捕获 ali + 权值热力图
        self._dbg_cap = {'capture': True}
        x_samples_full = self._decode_with_shift(samples_ext, with_grad=False, hshift_pix=hshift_pix)

        # 捕获 ali（融合前）与 mix（热力图）
        try:
            ali_t = self._dbg_cap.get('ali', None)
            mix_t = self._dbg_cap.get('mix', None)
            dbg_np = {}
            if ali_t is not None:
                dbg_np['ali'] = self._to_uint8_image(ali_t)
            if mix_t is not None:
                dbg_np['mix'] = self._to_uint8_image(mix_t)  # 已在 0..1
            self._dbg_saved_np = dbg_np if dbg_np else None
        except Exception:
            self._dbg_saved_np = None
        finally:
            self._dbg_cap = None

        # 成品对齐到 target（裁回 W0×H0）
        x_samples = self._align_to_target(x_samples_full, target)
        log["samples"] = (x_samples + 1) / 2
        return log, bpp

    @torch.no_grad()
    def sample_log(self, cond, steps):
        # 注意：log_images 内部已经直接采样；此函数保留兼容
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

    # ---------------- 接缝正则（左带） ----------------
    def _seam_band_mask_left(self, x: torch.Tensor, K: int) -> torch.Tensor:
        B, C, H, W = x.shape; K = max(1, min(int(K), W // 2))
        m = x.new_zeros((1, 1, 1, W)); m[..., :K] = 1.0
        return m.expand(B, 1, H, W)

    def _seam_losses(self, img: torch.Tensor, K: int) -> torch.Tensor:
        B, C, H, W = img.shape; loss = 0.0 * img.sum()
        loss += F.l1_loss(img[..., :1], img[..., -1:])  # 闭环约束
        m = self._seam_band_mask_left(img, K=K)
        dx = (img[..., 1:] - img[..., :-1]); tv = (dx.abs() * m[..., 1:]).mean()
        loss = loss + 0.05 * tv
        if K >= 1 and (K < W): loss += 0.05 * F.l1_loss(img[..., K-1:K], img[..., K:K+1])
        return loss

    def p_losses(self, x_start, cond, t, noise=None):
        # 清理/初始化静态扩展状态
        self._static_ext_active = False
        self._static_W0_latent = 0
        self._static_K_lat = 0
        self._static_W0_pix = 0

        loss_dict = {}; prefix = 'T' if self.training else 'V'

        # —— 静态扩展：统一把 c_latent/guide_hint 扩到 W+K
        K_lat, W0_lat = self._build_static_ext_cond(cond)
        c_latent_ext = cond['c_latent'][0]        # [B,C,H,W+K]
        guide_hint_ext = cond['guide_hint']

        # 把 x_start 也扩到 W+K 以对齐损失/噪声空间
        x_start_ext = self._wrap_tail_lat(x_start, K_lat)

        if not self.is_refine:
            # 标准分支
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
            loss_simple = self.get_loss(model_output_ext, target, mean=False).mean([1, 2, 3])
            loss_dict.update({f'{prefix}/l_simple': loss_simple.mean()})
            logvar_t = self.logvar[t].to(self.device)
            loss = loss_simple / torch.exp(logvar_t) + logvar_t
            if self.learn_logvar:
                loss_dict.update({f'{prefix}/l_gamma': loss.mean()}); loss_dict.update({'logvar': self.logvar.data.mean()})
            loss = self.l_guide_weight * loss.mean()

            loss_bpp = cond['bpp']; guide_bpp = cond['q_bpp']
            loss_dict.update({f'{prefix}/l_bpp': loss_bpp.mean()}); loss_dict.update({f'{prefix}/q_bpp': guide_bpp.mean()})
            loss += self.l_bpp_weight * loss_bpp

            loss_emb = cond['emb_loss']; loss_dict.update({f'{prefix}/l_emb': loss_emb.mean()}); loss += self.l_bpp_weight * loss_emb

            # guide 对齐到 W+K
            loss_guide = self.get_loss(c_latent_ext, x_start_ext)
            loss_dict.update({f'{prefix}/l_guide': loss_guide.mean()}); loss += self.l_guide_weight * loss_guide

            loss_dict.update({f'{prefix}/loss': loss}); return loss, loss_dict

        # refine：在 W+K 上去噪，再解码到像素域、对齐 target 计算图像损失
        b, cch, h, w_ext = c_latent_ext.shape; shape = (b, self.channels, h, w_ext)
        noise = default(noise, lambda: torch.randn_like(c_latent_ext))
        x_T_ext = self.q_sample(x_start=c_latent_ext, t=t, noise=noise); steps = self.fixed_step

        samples_ext = self.sampler.sample_grad(steps, shape, cond, unconditional_guidance_scale=1.0,
                                               unconditional_conditioning=None, x_T=x_T_ext)
        hshift_pix = int(cond.get('hshift_pix', torch.tensor(0)).item())
        model_output_pix = self._decode_with_shift(samples_ext, with_grad=True, hshift_pix=hshift_pix)
        target = cond['target']; model_output_pix = self._align_to_target(model_output_pix, target)

        # 损失（注意：samples_ext 与 x_start_ext 同宽）
        loss_simple = self.get_loss(samples_ext, x_start_ext, mean=False).mean([1, 2, 3])
        loss_dict.update({f'{prefix}/l_simple': loss_simple.mean()})
        loss  = self.l_guide_weight * loss_simple.mean()

        loss_mse    = self.get_loss(model_output_pix, target, mean=False).mean([1, 2, 3])
        loss_dict.update({f'{prefix}/l_mse': loss_mse.mean()}); loss += self.l_guide_weight * loss_mse.mean()

        loss_lpips  = self.perceptual_loss(model_output_pix, target)
        loss_dict.update({f'{prefix}/l_lpips': loss_lpips.mean()}); loss += self.l_guide_weight * loss_lpips * 0.5

        loss_bpp = cond['bpp']; guide_bpp = cond['q_bpp']
        loss_dict.update({f'{prefix}/l_bpp': loss_bpp.mean()}); loss_dict.update({f'{prefix}/q_bpp': guide_bpp.mean()})
        loss += self.l_bpp_weight * loss_bpp

        loss_emb = cond['emb_loss']; loss_dict.update({f'{prefix}/l_emb': loss_emb.mean()}); loss += self.l_bpp_weight * loss_emb

        latent_seam = F.l1_loss(samples_ext[..., :1], samples_ext[..., -1:])
        loss += 0.05 * latent_seam; loss_dict.update({f'{prefix}/l_seam_latent': latent_seam.detach()})

        if self.is_refine and self.train_blend_net and self.vae_blend_enabled:
            K_pix = self._auto_k_pixel(model_output_pix.shape[-1]); seam_loss = self._seam_losses(model_output_pix, K=K_pix)
            loss = loss + 0.1 * seam_loss; loss_dict.update({f'{prefix}/l_seam': seam_loss.detach()})

        loss_dict.update({f'{prefix}/loss': loss}); return loss, loss_dict

    # ===== 训练/验证 =====
    def training_step(self, batch, batch_idx):
        dm = getattr(self.trainer, "datamodule", None)
        bsz = getattr(dm, "train_batch_size", None)
        if bsz is None:
            try:
                bsz = batch[self.first_stage_key].shape[0]
            except Exception:
                first_val = next(iter(batch.values()))
                bsz = first_val.shape[0] if hasattr(first_val, "shape") else len(first_val)

        # === 关键：所有日志值都要脱离计算图 ===
        loss, loss_dict = self.shared_step(batch)

        safe_logs = {}
        for k, v in loss_dict.items():
            if torch.is_tensor(v):
                # 对步进日志用标量；如果想保留均值曲线，也建议 on_epoch=True + item()
                # 避免在 on_step 聚合时保留大图
                try:
                    safe_logs[k] = v.detach()
                    # 如果某些值很大（带 HxW），千万别直接 log；先 .mean() 再 .detach()
                    if safe_logs[k].ndim > 0:
                        safe_logs[k] = safe_logs[k].mean().detach()
                    # 再变成标量，彻底切断图
                    safe_logs[k] = safe_logs[k].item()
                except Exception:
                    # 兜底：直接 item 失败就取 mean().item()
                    safe_logs[k] = v.mean().detach().item()
            else:
                safe_logs[k] = v

        # 建议：on_step 少一些、on_epoch 多一些，进一步降低显存抖动
        self.log_dict(
            safe_logs,
            prog_bar=True,
            logger=True,
            on_step=True,
            on_epoch=True,
            batch_size=bsz,
        )

        # 这个 log 不要传入带图张量
        self.log("global_step", float(self.global_step), prog_bar=True, logger=True, on_step=True, on_epoch=False,
                 batch_size=bsz)

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
        save_dir = os.path.join(self.logger.save_dir, "validation", f"epoch{self.current_epoch:04d}"); os.makedirs(save_dir, exist_ok=True)
        bpp_vals = []; psnr_vals, msssim_vals, lpips_vals = [], [], []
        with torch.inference_mode(), torch.cuda.amp.autocast(enabled=True):
            for i in range(n):
                # —— 不再设置 _dump_ali_prefix，避免直解码路径写 ali；改为在 x_samples 同管线抓帧
                one = {}
                for k, v in batch.items():
                    if torch.is_tensor(v) and v.dim() >= 1 and v.shape[0] == B: one[k] = v[i:i + 1]
                    else: one[k] = v
                log, bpp = self.log_images(one, bs=1, sample_steps=sample_steps)
                bpp_vals.append(float(bpp.detach().mean()))

                # 保存最终成品
                x_vis = (log["samples"][0].detach().cpu().numpy().transpose(1, 2, 0) * 255).clip(0, 255).astype(np.uint8)
                out_png = os.path.join(save_dir, f"{batch_idx}_{i}.png")
                Image.fromarray(x_vis).save(out_png)

                # 若抓到了同管线的 ali/热力图，则与成品同目录落盘
                dbg = getattr(self, "_dbg_saved_np", None)
                if isinstance(dbg, dict):
                    if 'ali' in dbg:
                        Image.fromarray(dbg['ali']).save(os.path.join(save_dir, f"{batch_idx}_{i}_ali.png"))
                    if 'mix' in dbg:
                        Image.fromarray(dbg['mix']).save(os.path.join(save_dir, f"{batch_idx}_{i}_mix.png"))
                # 用后即弃，避免串批
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
        self.preprocess_model.quantize.reset_usage(); return super().on_validation_epoch_start()

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
        usage = self.preprocess_model.quantize.get_usage(); self.log("usage", usage, prog_bar=True, logger=True, on_step=False, on_epoch=True, batch_size=1)
        names = ["bpp"]; metrics_order = ["psnr", "ms_ssim", "lpips"]
        if getattr(self, "calculate_metrics", None): names += [m for m in metrics_order if m in self.calculate_metrics]
        names = names[:len(avg_out)]
        for i, name in enumerate(names[1:], start=1):
            val = float(avg_out[i]); self.log(f"avg_{name}", val, prog_bar=True, logger=True, on_step=False, on_epoch=True, batch_size=1)
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

    def load_preprocess_ckpt(self, ckpt_path_pre):
        ckpt = torch.load(ckpt_path_pre); self.preprocess_model.load_state_dict(ckpt); print(['CONTROL WEIGHTS LOADED'])

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
                            ckpt_base['state_dict'][dst_key] = torch.cat([ckpt_base['state_dict'][key], ckpt_base['state_dict'][key]], dim=0)[:control_dim]
                        else:
                            control_dim_0 = self.state_dict()[dst_key].size(0); control_dim_1 = self.state_dict()[dst_key].size(1)
                            ckpt_base['state_dict'][dst_key] = torch.cat([ckpt_base['state_dict'][key], ckpt_base['state_dict'][key]], dim=1)[:control_dim_0, :control_dim_1, ...]
                    else:
                        ckpt_base['state_dict'][dst_key] = ckpt_base['state_dict'][key]
        res_sync = self.load_state_dict(ckpt_base['state_dict'], strict=False)
        print(f'[{len(res_sync.missing_keys)} keys are missing from the model (hint processing and cross connections included)]')
