import itertools
import torch
import torch.nn as nn
import numpy as np
import math

from abc import ABC, abstractmethod
from torch.nn.utils.rnn import pad_sequence
from typing import List, Callable, Optional, Tuple, Union, Sequence


from .multimodal_encoder.builder import build_vision_tower
from .multimodal_projector.builder import (
    build_vision_projector,
    build_subobject_vision_projector,
)
from .subobject_tokenization_utils import VisualTokenEmbedding
from .masking_utils import (
    MaskEmbedder,
    downsample_mask_to_1d_counts,
    check_resised_image_masks,
)

from llava.constants import (
    IGNORE_INDEX,
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_PATCH_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
)

from llava.mm_utils import get_anyres_image_grid_shape


def build_sinusoidal(max_pos: int, dim: int, zero_pad: bool = True):
    """
    Returns a [max_pos+(1 if zero_pad else 0), dim] tensor:
     - row 0 is all zeros (if zero_pad=True)
     - rows 1..max_pos are sin/cos encodings for positions 0..max_pos-1
    """
    # base table for positions 0..max_pos-1
    position = torch.arange(max_pos, dtype=torch.float32).unsqueeze(1)  # [max_pos,1]
    div_term = torch.exp(
        torch.arange(0, dim, 2, dtype=torch.float32) * -(math.log(10000.0) / dim)
    )  # [dim/2]
    pe = torch.zeros(max_pos, dim, dtype=torch.float32)  # [max_pos,dim]
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)

    if zero_pad:
        # prepend a zero‐vector so index=0 → no encoding
        return torch.cat([torch.zeros(1, dim), pe], dim=0)  # [max_pos+1,dim]
    else:
        return pe


class SegmentSinusoidalPE(nn.Module):
    def __init__(self, max_segs: int, d_model: int):
        super().__init__()
        sinus = build_sinusoidal(max_segs, d_model, zero_pad=True)
        self.register_buffer("sinus", sinus)  # shape [max_segs+1, d_model]

    def forward(self, x: torch.Tensor, seg_idx: torch.LongTensor):
        """
        x:       [B, L, D]
        seg_idx: [B, L] with values in 0..max_segs
                 (0 = no‐segment, 1..k real segment IDs)
        """
        # gather and add in one go
        return x + self.sinus[seg_idx]  # advanced‐index on GPU → [B,L,D]


def make_2d_sincos_pos_encoding(
    H: int = 24,
    W: int = 24,
    D: int = 768,  # any even number
    base: float = 10_000.0,
) -> torch.Tensor:
    """
    Returns a tensor of shape (H, W, D) with fixed 2-D sinusoidal positional encoding.
    Half the channels encode rows, half encode columns.
    """
    assert D % 2 == 0, "D must be even."
    d_half = D // 2  # channels per axis
    # ----- frequencies (like in the paper) -----
    idx = torch.arange(d_half // 2, dtype=torch.float32)  # 0 … d_half/2-1
    freq = base ** (2 * idx / d_half)  # (d_half/2,)

    # ----- raw positions -----
    y = torch.arange(H, dtype=torch.float32)[:, None]  # (H, 1)
    x = torch.arange(W, dtype=torch.float32)[:, None]  # (W, 1)

    # ----- apply sin / cos -----
    y_enc = y / freq  # broadcasting (H, d_half/2)
    x_enc = x / freq  # (W, d_half/2)

    y_sin, y_cos = torch.sin(y_enc), torch.cos(y_enc)  # each (H, d_half/2)
    x_sin, x_cos = torch.sin(x_enc), torch.cos(x_enc)  # each (W, d_half/2)

    # ----- interleave sin & cos so they alternate -----
    y_emb = torch.stack((y_sin, y_cos), dim=-1).flatten(1)  # (H, d_half)
    x_emb = torch.stack((x_sin, x_cos), dim=-1).flatten(1)  # (W, d_half)

    # ----- combine into grid -----
    pos_y = y_emb[:, None, :].repeat(1, W, 1)  # (H, W, d_half)
    pos_x = x_emb[None, :, :].repeat(H, 1, 1)  # (H, W, d_half)
    pos = torch.cat((pos_y, pos_x), dim=-1)  # (H, W, D)
    return pos


def fixed_sinusoidal_encoding(image_features, group_ranges):
    B, L, D = (
        image_features.shape[0],
        image_features.shape[1],
        image_features.shape[-1],
    )
    img_device = image_features.device
    img_typ = image_features.dtype

    max_segs_gr_segs = max(len(spans) for spans in group_ranges)
    MAX_SEGS = 300
    if max_segs_gr_segs > MAX_SEGS:
        return image_features

    # 1) build seg_idx tensor
    seg_idx = torch.zeros(B, L, dtype=torch.long)
    for b, spans in enumerate(group_ranges):
        for s_id, (start, end) in enumerate(spans):
            seg_idx[b, start:end] = s_id

    pe_module = SegmentSinusoidalPE(max_segs=MAX_SEGS, d_model=D).to(img_device)
    # print("pe_module.sinus.shape", pe_module.sinus.shape)
    image_features = pe_module(image_features, seg_idx).to(img_typ)

    return image_features


def fixed_sinusoidal_encoding_2d(image_features, group_ranges, flat_indices):
    grid_H, grid_W = 24, 24  # to change with different vision encoder
    B, N, D = image_features.shape
    device = image_features.device

    pe_grid = make_2d_sincos_pos_encoding(grid_H, grid_W, D).to(device)  # (24,24,D)
    pe_flat = pe_grid.reshape(grid_H * grid_W, D).to(device)  # (576,D)

    for b in range(B):
        for m, (start, end) in enumerate(group_ranges[b]):
            idxs = flat_indices[b][m]  # exact cells

            seg_len = end - start

            if idxs.numel() == (seg_len - 1):
                seg_len -= 1
                start += 1

            elif seg_len == (idxs.numel() - 1):
                seg_len += 1
                start -= 1

            assert idxs.numel() == seg_len  # now always true
            idxs = idxs.to(torch.long).to(device)
            image_features[b, start:end] += pe_flat[idxs]

    return image_features


def learnable_positional_encoding_1d(model, image_features, group_ranges):
    B, L, D = (
        image_features.shape[0],
        image_features.shape[1],
        image_features.shape[-1],
    )
    MAX_SEGS = 300
    img_device = image_features.device
    img_typ = image_features.dtype

    # 1) build seg_idx tensor
    seg_idx = torch.zeros(B, L, dtype=torch.long)
    for b, spans in enumerate(group_ranges):
        for s_id, (start, end) in enumerate(spans):
            seg_idx[b, start:end] = s_id

    seg_idx = seg_idx.to(img_device, dtype=torch.long)
    image_features = image_features.to(img_device)
    ##clamping to 300 masks
    seg_idx = seg_idx.clamp(max=MAX_SEGS)

    image_features = model.mm_masks_pos_encoding(image_features, seg_idx).to(
        dtype=img_typ
    )

    return image_features
