import torch
import torch.nn as nn
from compressai.entropy_models import GaussianConditional, EntropyModel
import torchac

def ckbd_split(y):
    """
    Split y to anchor and non-anchor
    anchor :
        0 1 0 1 0
        1 0 1 0 1
        0 1 0 1 0
        1 0 1 0 1
        0 1 0 1 0
    non-anchor:
        1 0 1 0 1
        0 1 0 1 0
        1 0 1 0 1
        0 1 0 1 0
        1 0 1 0 1
    """
    anchor = ckbd_anchor(y)
    nonanchor = ckbd_nonanchor(y)
    return anchor, nonanchor

def ckbd_merge(anchor, nonanchor):
    # out = torch.zeros_like(anchor).to(anchor.device)
    # out[:, :, 0::2, 0::2] = non_anchor[:, :, 0::2, 0::2]
    # out[:, :, 1::2, 1::2] = non_anchor[:, :, 1::2, 1::2]
    # out[:, :, 0::2, 1::2] = anchor[:, :, 0::2, 1::2]
    # out[:, :, 1::2, 0::2] = anchor[:, :, 1::2, 0::2]

    return anchor + nonanchor

def ckbd_anchor(y):
    anchor = torch.zeros_like(y).to(y.device)
    anchor[:, :, 0::2, 1::2] = y[:, :, 0::2, 1::2]
    anchor[:, :, 1::2, 0::2] = y[:, :, 1::2, 0::2]
    return anchor

def ckbd_nonanchor(y):
    nonanchor = torch.zeros_like(y).to(y.device)
    nonanchor[:, :, 0::2, 0::2] = y[:, :, 0::2, 0::2]
    nonanchor[:, :, 1::2, 1::2] = y[:, :, 1::2, 1::2]
    return nonanchor

def ckbd_anchor_sequeeze(y):
    B, C, H, W = y.shape
    anchor = torch.zeros([B, C, H, W // 2]).to(y.device)
    anchor[:, :, 0::2, :] = y[:, :, 0::2, 1::2]
    anchor[:, :, 1::2, :] = y[:, :, 1::2, 0::2]
    return anchor

def ckbd_nonanchor_sequeeze(y):
    B, C, H, W = y.shape
    nonanchor = torch.zeros([B, C, H, W // 2]).to(y.device)
    nonanchor[:, :, 0::2, :] = y[:, :, 0::2, 0::2]
    nonanchor[:, :, 1::2, :] = y[:, :, 1::2, 1::2]
    return nonanchor

def ckbd_anchor_unsequeeze(anchor):
    B, C, H, W = anchor.shape
    y_anchor = torch.zeros([B, C, H, W * 2]).to(anchor.device)
    y_anchor[:, :, 0::2, 1::2] = anchor[:, :, 0::2, :]
    y_anchor[:, :, 1::2, 0::2] = anchor[:, :, 1::2, :]
    return y_anchor

def ckbd_nonanchor_unsequeeze(nonanchor):
    B, C, H, W = nonanchor.shape
    y_nonanchor = torch.zeros([B, C, H, W * 2]).to(nonanchor.device)
    y_nonanchor[:, :, 0::2, 0::2] = nonanchor[:, :, 0::2, :]
    y_nonanchor[:, :, 1::2, 1::2] = nonanchor[:, :, 1::2, :]
    return y_nonanchor


def compress_anchor(gaussian_conditional:EntropyModel, anchor, scales_anchor, means_anchor, symbols_list, indexes_list):
    # squeeze anchor to avoid non-anchor symbols
    anchor_squeeze = ckbd_anchor_sequeeze(anchor)
    scales_anchor_squeeze = ckbd_anchor_sequeeze(scales_anchor)
    means_anchor_squeeze = ckbd_anchor_sequeeze(means_anchor)
    indexes = gaussian_conditional.build_indexes(scales_anchor_squeeze)
    anchor_hat = gaussian_conditional.quantize(anchor_squeeze, "symbols", means_anchor_squeeze)
    symbols_list.extend(anchor_hat.reshape(-1).tolist())
    indexes_list.extend(indexes.reshape(-1).tolist())
    anchor_hat = ckbd_anchor_unsequeeze(anchor_hat + means_anchor_squeeze)
    return anchor_hat

def compress_nonanchor(gaussian_conditional:EntropyModel, nonanchor, scales_nonanchor, means_nonanchor, symbols_list, indexes_list):
    nonanchor_squeeze = ckbd_nonanchor_sequeeze(nonanchor)
    scales_nonanchor_squeeze = ckbd_nonanchor_sequeeze(scales_nonanchor)
    means_nonanchor_squeeze = ckbd_nonanchor_sequeeze(means_nonanchor)
    indexes = gaussian_conditional.build_indexes(scales_nonanchor_squeeze)
    nonanchor_hat = gaussian_conditional.quantize(nonanchor_squeeze, "symbols", means_nonanchor_squeeze)
    symbols_list.extend(nonanchor_hat.reshape(-1).tolist())
    indexes_list.extend(indexes.reshape(-1).tolist())
    nonanchor_hat = ckbd_nonanchor_unsequeeze(nonanchor_hat + means_nonanchor_squeeze)
    return nonanchor_hat

def decompress_anchor(gaussian_conditional:EntropyModel, scales_anchor, means_anchor, decoder, cdf, cdf_lengths, offsets):
    scales_anchor_squeeze = ckbd_anchor_sequeeze(scales_anchor)
    means_anchor_squeeze = ckbd_anchor_sequeeze(means_anchor)
    indexes = gaussian_conditional.build_indexes(scales_anchor_squeeze)
    anchor_hat = decoder.decode_stream(indexes.reshape(-1).tolist(), cdf, cdf_lengths, offsets)
    anchor_hat = torch.Tensor(anchor_hat).reshape(scales_anchor_squeeze.shape).to(scales_anchor.device) + means_anchor_squeeze
    anchor_hat = ckbd_anchor_unsequeeze(anchor_hat)
    return anchor_hat

def decompress_nonanchor(gaussian_conditional:EntropyModel, scales_nonanchor, means_nonanchor, decoder, cdf, cdf_lengths, offsets):
    scales_nonanchor_squeeze = ckbd_nonanchor_sequeeze(scales_nonanchor)
    means_nonanchor_squeeze = ckbd_nonanchor_sequeeze(means_nonanchor)
    indexes = gaussian_conditional.build_indexes(scales_nonanchor_squeeze)
    nonanchor_hat = decoder.decode_stream(indexes.reshape(-1).tolist(), cdf, cdf_lengths, offsets)
    nonanchor_hat = torch.Tensor(nonanchor_hat).reshape(scales_nonanchor_squeeze.shape).to(scales_nonanchor.device) + means_nonanchor_squeeze
    nonanchor_hat = ckbd_nonanchor_unsequeeze(nonanchor_hat)
    return nonanchor_hat

def compute_cdf_uniform_prob(codebook_size, target_shape):
    """Obtain CDF from uniform distribution, cast to target_shape"""
    b, h, w = target_shape
    prob_per_entry = 1.0 / codebook_size

    # Compute the cumulative sum starting from 0
    cdf = torch.cumsum(torch.full((codebook_size,), prob_per_entry), dim=0)
    cdf = torch.cat([torch.zeros(1), cdf])
    cdf = cdf.view(1, 1, 1, -1).expand(b, h, w, -1)
    cdf = cdf.clone()
    cdf[..., -1] = 1.0
    return cdf

def compress_hyper_latent(z_hat_indices, codebook_size):
    """Compress hyper-latent to bytes using torchac."""
    cdf = compute_cdf_uniform_prob(codebook_size, z_hat_indices.shape)
    z_hat_indices = z_hat_indices.to(torch.int16).to('cpu')
    return torchac.encode_float_cdf(cdf, z_hat_indices, check_input_bounds=True)


def decompress_hyper_latent(compressed_hyper_latent, shape, codebook_size):
    """Decompress hyper-latent using torchac."""
    h, w = shape
    cdf = compute_cdf_uniform_prob(codebook_size, (1, int(h), int(w)))
    return torchac.decode_float_cdf(cdf, compressed_hyper_latent)

def calculate_bpp(compressed_data, num_pixels, bytes=True, num_bytes=None):
    """Calculate bpp given the compressed text and number of pixels."""
    scaling_factor = 8 if bytes else 1
    if num_bytes:
        return num_bytes * scaling_factor / num_pixels
    return len(compressed_data) * scaling_factor / num_pixels

