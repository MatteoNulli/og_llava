#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


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
from .positional_encoding_utils import (
    fixed_sinusoidal_encoding,
    fixed_sinusoidal_encoding_2d,
    learnable_positional_encoding_1d,
)

from llava.constants import (
    IGNORE_INDEX,
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_PATCH_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
)

from llava.mm_utils import get_anyres_image_grid_shape


class BOMMaskToken(nn.Module):
    def __init__(self, input_features, dtype=torch.float32, device="cuda"):
        super().__init__()
        # Create the token parameter as part of this module
        self.mm_bom_mask_token = nn.Parameter(
            torch.randn((1, input_features), dtype=dtype, device=device)
        )

    def forward(self, x=None):
        return self.mm_bom_mask_token


def build_24x24_sinusoidal(d_model: int) -> torch.FloatTensor:
    """
    Returns a Tensor of shape [576, d_model], where index f∈[0..575] ↔ (row=f//24, col=f%24)
    and we do a standard Transformer‐style 2D sinusoidal:

        • First half of the features come from row ∈ [0..23].
        • Second half come from col ∈ [0..23].

    Concretely, if d_model=64, then d_row=32, d_col=32.  If d_model is odd,
    we let d_row = ceil(d_model/2), d_col = floor(d_model/2).  In all cases,
    d_row + d_col = d_model.

    This tensor lives on CPU by default; you’ll do .to(device) in forward.
    """
    H, W = 24, 24
    Npos = H * W  # 576

    # split d_model into (d_row + d_col) so that d_row = ceil(d_model/2), d_col = floor(d_model/2)
    d_row = (d_model + 1) // 2
    d_col = d_model - d_row

    # build a 1D sinusoidal table for rows of length = H, dimension = d_row
    def build_1d_sinusoid(pos_max: int, dim: int) -> torch.FloatTensor:
        # returns [pos_max, dim] where each row p∈[0..pos_max-1] is the standard "positional sin/cos(dim)".
        pe = torch.zeros(pos_max, dim)
        half_dim = (
            dim // 2
        )  # if dim is odd, the last channel (index = 2*half_dim) stays 0
        inv_freq = 1.0 / (
            10000 ** (torch.arange(0, half_dim, dtype=torch.float32) * 2.0 / dim)
        )
        for p in range(pos_max):
            v = p * inv_freq  # [half_dim]
            pe[p, 0 : 2 * half_dim : 2] = torch.sin(v)
            pe[p, 1 : 2 * half_dim : 2] = torch.cos(v)
            # if dim is odd, pe[p, 2*half_dim] remains 0
        return pe  # [pos_max, dim]

    row_sinusoid = build_1d_sinusoid(H, d_row)  # [24, d_row]
    col_sinusoid = build_1d_sinusoid(W, d_col)  # [24, d_col]

    # now build a [24,24,d_model] by concatenating row/col
    full = torch.zeros(H, W, d_model)
    for r in range(H):
        for c in range(W):
            full[r, c, :d_row] = row_sinusoid[r]
            full[r, c, d_row:] = col_sinusoid[c]

    # flatten to [576, d_model], row-major:
    return full.view(Npos, d_model)  # [576, d_model]


def flat_indices_for_box(
    box: Tuple[int, int, int, int], grid_w: int = 24, device="cpu"
) -> torch.Tensor:
    """
    Returns a 1-D LongTensor with the flat (row*grid_w + col) indices that lie
    inside the bounding box, **row-major order**.
    """
    y0, x0, h, w = box
    ys = torch.arange(y0, y0 + h, device=device)
    xs = torch.arange(x0, x0 + w, device=device)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")  # (h,w)
    return (grid_y * grid_w + grid_x).reshape(-1)


class MasksPositionalEncoding(nn.Module):
    def __init__(self, max_segs: int, d_model: int):
        """
        max_segs: the maximum number of *real* segments you ever expect (e.g. 40)
        we add +1 for the padding index 0
        """
        super().__init__()
        self.seg_embed = nn.Embedding(
            num_embeddings=max_segs + 1,  # 0 ... max_segs
            embedding_dim=d_model,
            padding_idx=0,  # embedding[0] is always zero
        )

    def forward(self, x: torch.Tensor, seg_idx: torch.LongTensor):
        # x: [B, L, D], seg_idx: [B, L] in 0..max_segs
        return x + self.seg_embed(seg_idx)


class LlavaMetaModel:

    def __init__(self, config):
        super(LlavaMetaModel, self).__init__(config)

        if hasattr(config, "mm_vision_tower"):
            self.vision_tower = build_vision_tower(config, delay_load=True)
            self.mm_projector = build_vision_projector(config)
            # self.config.mm_projector_type = "mlp2x_gelu,subobject_tokenization"
            # if self.config.mm_projector_type == "mlp2x_gelu,subobject_tokenization":
            #     self.mm_subobject_projector = build_subobject_vision_projector(config)

            if "unpad" in getattr(config, "mm_patch_merge_type", ""):
                self.image_newline = nn.Parameter(
                    torch.empty(config.hidden_size, dtype=self.dtype)
                )

    def get_vision_tower(self):
        vision_tower = getattr(self, "vision_tower", None)
        if type(vision_tower) is list:
            vision_tower = vision_tower[0]
        return vision_tower

    def initialize_vision_modules(self, model_args, fsdp=None):
        vision_tower = model_args.vision_tower
        mm_vision_select_layer = model_args.mm_vision_select_layer
        mm_vision_select_feature = model_args.mm_vision_select_feature
        pretrain_mm_mlp_adapter = model_args.pretrain_mm_mlp_adapter
        pretrain_mm_bom_mask_token = model_args.pretrain_mm_bom_mask_token
        pretrain_mm_masks_pos_encoding = model_args.pretrain_mm_masks_pos_encoding
        mm_patch_merge_type = model_args.mm_patch_merge_type

        self.sam2_masking_token = model_args.sam2_masking_token
        self.custom_rotary_embedding = model_args.custom_rotary_embedding

        self.config.mm_vision_tower = vision_tower

        if self.get_vision_tower() is None:
            vision_tower = build_vision_tower(model_args)

            if fsdp is not None and len(fsdp) > 0:
                self.vision_tower = [vision_tower]
            else:
                self.vision_tower = vision_tower
        else:
            if fsdp is not None and len(fsdp) > 0:
                vision_tower = self.vision_tower[0]
            else:
                vision_tower = self.vision_tower
            vision_tower.load_model()

        self.config.use_mm_proj = True
        self.config.mm_projector_type = getattr(
            model_args, "mm_projector_type", "linear"
        )
        self.config.mm_hidden_size = vision_tower.hidden_size
        self.config.mm_vision_select_layer = mm_vision_select_layer
        self.config.mm_vision_select_feature = mm_vision_select_feature
        self.config.mm_patch_merge_type = mm_patch_merge_type

        if getattr(self, "mm_projector", None) is None:
            self.mm_projector = build_vision_projector(self.config)
            if "unpad" in mm_patch_merge_type:
                embed_std = 1 / torch.sqrt(
                    torch.tensor(self.config.hidden_size, dtype=self.dtype)
                )
                self.image_newline = nn.Parameter(
                    torch.randn(self.config.hidden_size, dtype=self.dtype) * embed_std
                )
        else:
            # In case it is frozen by LoRA
            for p in self.mm_projector.parameters():
                p.requires_grad = True

        if pretrain_mm_mlp_adapter is not None:
            mm_projector_weights = torch.load(
                pretrain_mm_mlp_adapter, map_location="cpu"
            )

            def get_w(weights, keyword):
                return {
                    k.split(keyword + ".")[1]: v
                    for k, v in weights.items()
                    if keyword in k
                }

            self.mm_projector.load_state_dict(
                get_w(mm_projector_weights, "mm_projector")
            )
        if pretrain_mm_bom_mask_token is not None:
            mm_bom_mask_token_weights = torch.load(
                pretrain_mm_bom_mask_token, map_location="cpu"
            )
            projector_input_features = self.mm_projector[0].in_features

            # Initialize the helper module
            self.mm_bom_mask_token = BOMMaskToken(
                projector_input_features, self.dtype, device="cuda"
            )

            def get_w(weights, keyword):
                return {
                    k.split("model.mm_bom_mask_token.")[1]: v
                    for k, v in weights.items()
                }

            self.mm_bom_mask_token.load_state_dict(
                get_w(mm_bom_mask_token_weights, "mm_bom_mask_token")
            )

        else:
            print("Radomly Initializing mm_bom_mask_token")
            if self.config.mm_projector_type == "mlp2x_gelu,subobject_tokenization":
                projector_input_features = self.mm_projector[0][0].in_features
                self.mm_bom_mask_token = BOMMaskToken(
                    projector_input_features, self.dtype, device="cuda"
                )
            else:
                projector_input_features = self.mm_projector[0].in_features
                ve_output_features = self.vision_tower.vision_tower.vision_model.post_layernorm.normalized_shape[
                    0
                ]

                self.mm_bom_mask_token = BOMMaskToken(
                    projector_input_features, self.dtype, device="cuda"
                )
        if pretrain_mm_masks_pos_encoding:
            mm_masks_pos_encoding_weights = torch.load(
                pretrain_mm_masks_pos_encoding, map_location="cpu"
            )
            projector_output_features = self.mm_projector[0].out_features

            # Initialize the helper module
            self.mm_masks_pos_encoding = MasksPositionalEncoding(
                max_segs=300, d_model=projector_output_features
            )

            def get_w(weights, keyword):
                return {
                    k.split("model.mm_masks_pos_encoding.")[1]: v
                    for k, v in weights.items()
                }

            self.mm_masks_pos_encoding.load_state_dict(
                get_w(mm_masks_pos_encoding_weights, "mm_masks_pos_encoding")
            )
            print(
                f"Loaded mm_masks_pos_encoding_weights from {pretrain_mm_masks_pos_encoding}"
            )
        else:
            print("Radomly Initializing pretrain_mm_masks_pos_encoding")
            projector_output_features = self.mm_projector[0].out_features
            ve_output_features = self.vision_tower.vision_tower.vision_model.post_layernorm.normalized_shape[
                0
            ]
            self.mm_masks_pos_encoding = MasksPositionalEncoding(
                max_segs=300, d_model=projector_output_features
            )


def unpad_image(tensor, original_size):
    """
    Unpads a PyTorch tensor of a padded and resized image.

    Args:
    tensor (torch.Tensor): The image tensor, assumed to be in CxHxW format.
    original_size (tuple): The original size of PIL image (width, height).

    Returns:
    torch.Tensor: The unpadded image tensor.
    """
    original_width, original_height = original_size
    current_height, current_width = tensor.shape[1:]

    original_aspect_ratio = original_width / original_height
    current_aspect_ratio = current_width / current_height

    if original_aspect_ratio > current_aspect_ratio:
        scale_factor = current_width / original_width
        new_height = int(original_height * scale_factor)
        padding = (current_height - new_height) // 2
        unpadded_tensor = tensor[:, padding : current_height - padding, :]
    else:
        scale_factor = current_height / original_height
        new_width = int(original_width * scale_factor)
        padding = (current_width - new_width) // 2
        unpadded_tensor = tensor[:, :, padding : current_width - padding]

    return unpadded_tensor


class LlavaMetaForCausalLM(ABC):

    @abstractmethod
    def get_model(self):
        pass

    def get_vision_tower(self):
        return self.get_model().get_vision_tower()

    def encode_images(self, images, masks, masking=False):
        """
        Encodes images in the vision encoder and then in the multimodal projector

        Parameters:
            images (torch.Tensor): Tensor of input images to the vision encoder shape (batch_size, vision_encoder_input, feature_dim).
            masks (List[torch.Tensor]): List of Tensor. The Outer list if of batch_size length, the inner list is of lenght number of masks in each image.

        Returns:
            torch.Tensor: Image features outputted from multimodal projector.
        """

        self.global_view = True
        self.averaging = False
        self.mask_removing = False
        self.mask_limiting = False
        self.mask_limit = 20
        self.averaging_global_view = True
        self.no_masktoken = False
        self.use_sliding_window = False
        self.number_of_masks = 5
        self.use_dummy_masks = False
        self.zero_indices = False
        self.image_filling = False
        self.absolute_positional_encoding = False
        self.learnable_positional_encoding = False

        if (
            masking
            and masks is not None
            and any(torch.any(mask != 0).item() for mask in masks)
            and self.config.mm_projector_type == "mlp2x_gelu,subobject_tokenization"
        ):

            image_features = self.get_model().get_vision_tower()(images)

            ## B.O. MASKING PART
            mask_embedder = MaskEmbedder(
                model=self,  ## LlavaLlamaForCausalLM
                config=self.config,
                vision_tower=self.get_model().get_vision_tower(),
            )
            image_features, group_ranges = mask_embedder(image_features, masks)
            image_features = image_features.to(images.dtype).to(images.device)
            ## E.O. MASKING PART

            image_features = self.get_model().mm_projector(image_features)

            ## B.O. Mixing with subojbect tokenization
            if self.zero_indices:
                raise NotImplementedError(
                    "Re-Implement Zero_indices check within mask embedder. We did not previously find any instances of zero indices."
                )
                zero_indices = []
                ## old implementation after getting zero indices from mask embedder
                if len(zero_indices) > 0:
                    visual_token_embedding = VisualTokenEmbedding(
                        model=self,
                        get_model=self.get_model(),
                        config=self.config,
                        vision_tower=self.get_model().get_vision_tower(),
                    )

                    visual_token_embeds, _, _ = (
                        visual_token_embedding.prepare_visual_embeds(images, masks)
                    )
                    tokens_to_add = visual_token_embeds[
                        :, zero_indices, :
                    ]  # shape [B, x, 4096], where x is the number of masks which, when downsampled have zero True values (too small object).
                    # concatenate along the “token” dimension (dim=1)
                    new_image_features = torch.cat(
                        [image_features, tokens_to_add], dim=1
                    )
                    image_features = new_image_features
            ## E.O. Mixing with subojbect tokenization

            return image_features

        elif (
            masking
            and "subobject_tokenization" not in self.config.mm_projector_type
            and masks is not None
            and any(torch.any(mask != 0).item() for mask in masks)
        ):
            image_features = self.get_model().get_vision_tower()(images)

            ## B.O. MASKING PART
            image_features = image_features.to(images.dtype).to(images.device)
            mask_embedder = MaskEmbedder(
                model=self,  ## LlavaLlamaForCausalLM
                config=self.config,
                vision_tower=self.get_model().get_vision_tower(),
            )
            image_features, group_ranges, flat_indices = mask_embedder(
                image_features, masks
            )

            if getattr(self.model, "custom_rotary_embedding", False):
                self.model.rotary_emb.group_ranges = group_ranges  # Expected [[(0, 100), (100, 200), (200, 576)], [(0, 200), (200, 600)], ... ] of length batch_size.
            image_features = image_features.to(images.dtype).to(images.device)
            ## E.O. MASKING PART
            image_features = self.get_model().mm_projector(image_features)

            ## Pos encoding
            self.sinusoidal_encoding_fixed = False
            self.sinusoidal_encoding_2d = False
            self.learnable_encoding = False

            if self.sinusoidal_encoding_fixed:
                image_features = fixed_sinusoidal_encoding(image_features, group_ranges)

            elif self.sinusoidal_encoding_2d:
                image_features = fixed_sinusoidal_encoding_2d(
                    image_features, group_ranges, flat_indices
                )

            elif self.learnable_encoding:
                image_features = learnable_positional_encoding_1d(
                    self.model, image_features, group_ranges
                )
            ##

            return image_features

        elif masking and self.config.mm_projector_type == "subobject_tokenization":

            visual_token_embedding = VisualTokenEmbedding(
                model=self,
                get_model=self.get_model(),
                config=self.config,
                vision_tower=self.get_model().get_vision_tower(),
                no_masks=False,
            )

            visual_token_embeds, _, _ = visual_token_embedding.prepare_visual_embeds(
                images, masks
            )
            return visual_token_embeds
        elif not masking and self.config.mm_projector_type == "subobject_tokenization":
            raise ValueError(
                f"Cannot have masking={masking} and mm_projector_type={self.config.mm_projector_type} with positive number of masks"
            )
        else:
            image_features = self.get_model().get_vision_tower()(images)
            # image_features = self.get_model().vision_resampler(image_features, images=images)

            image_features = self.get_model().mm_projector(image_features)

            return image_features

    def prepare_inputs_labels_for_multimodal(
        self,
        input_ids,
        position_ids,
        attention_mask,
        past_key_values,
        labels,
        images,
        masks,
        image_sizes=None,
    ):
        vision_tower = self.get_vision_tower()
        if vision_tower is None or images is None or input_ids.shape[1] == 1:
            return (
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                None,
                labels,
            )

        if type(images) is list or images.ndim == 5:
            if type(images) is list:
                images = [x.unsqueeze(0) if x.ndim == 3 else x for x in images]
            concat_images = torch.cat([image for image in images], dim=0)

            if hasattr(self.model, "sam2_masking_token"):
                image_features = self.encode_images(
                    concat_images, masks, masking=self.model.sam2_masking_token
                )
            else:
                image_features = self.encode_images(concat_images, masks)

            split_sizes = [image.shape[0] for image in images]
            image_features = torch.split(image_features, split_sizes, dim=0)
            mm_patch_merge_type = getattr(self.config, "mm_patch_merge_type", "flat")
            image_aspect_ratio = getattr(self.config, "image_aspect_ratio", "square")
            if mm_patch_merge_type == "flat":
                image_features = [x.flatten(0, 1) for x in image_features]
            elif mm_patch_merge_type.startswith("spatial"):
                new_image_features = []
                for image_idx, image_feature in enumerate(image_features):
                    if image_feature.shape[0] > 1:
                        base_image_feature = image_feature[0]
                        image_feature = image_feature[1:]
                        height = width = self.get_vision_tower().num_patches_per_side
                        assert height * width == base_image_feature.shape[0]
                        if image_aspect_ratio == "anyres":
                            num_patch_width, num_patch_height = (
                                get_anyres_image_grid_shape(
                                    image_sizes[image_idx],
                                    self.config.image_grid_pinpoints,
                                    self.get_vision_tower().config.image_size,
                                )
                            )
                            image_feature = image_feature.view(
                                num_patch_height, num_patch_width, height, width, -1
                            )
                        else:
                            raise NotImplementedError
                        if "unpad" in mm_patch_merge_type:
                            image_feature = image_feature.permute(
                                4, 0, 2, 1, 3
                            ).contiguous()
                            image_feature = image_feature.flatten(1, 2).flatten(2, 3)
                            image_feature = unpad_image(
                                image_feature, image_sizes[image_idx]
                            )
                            image_feature = torch.cat(
                                (
                                    image_feature,
                                    self.model.image_newline[:, None, None]
                                    .expand(*image_feature.shape[:-1], 1)
                                    .to(image_feature.device),
                                ),
                                dim=-1,
                            )
                            image_feature = image_feature.flatten(1, 2).transpose(0, 1)
                        else:
                            image_feature = image_feature.permute(
                                0, 2, 1, 3, 4
                            ).contiguous()
                            image_feature = image_feature.flatten(0, 3)
                        image_feature = torch.cat(
                            (base_image_feature, image_feature), dim=0
                        )
                    else:
                        image_feature = image_feature[0]
                        if "unpad" in mm_patch_merge_type:
                            image_feature = torch.cat(
                                (
                                    image_feature,
                                    self.model.image_newline[None].to(
                                        image_feature.device
                                    ),
                                ),
                                dim=0,
                            )
                    new_image_features.append(image_feature)
                image_features = new_image_features
            else:
                raise ValueError(
                    f"Unexpected mm_patch_merge_type: {self.config.mm_patch_merge_type}"
                )
        else:
            if hasattr(self.model, "sam2_masking_token"):
                image_features = self.encode_images(
                    images, masks, masking=self.model.sam2_masking_token
                )
            else:
                image_features = self.encode_images(images, masks)

        # print("image_features.shape", image_features.shape)

        # TODO: image start / end is not implemented here to support pretraining.
        if getattr(self.config, "tune_mm_mlp_adapter", False) and getattr(
            self.config, "mm_use_im_start_end", False
        ):
            raise NotImplementedError

        # Let's just add dummy tensors if they do not exist,
        # it is a headache to deal with None all the time.
        # But it is not ideal, and if you have a better idea,
        # please open an issue / submit a PR, thanks.
        _labels = labels
        _position_ids = position_ids
        _attention_mask = attention_mask
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        else:
            attention_mask = attention_mask.bool()
        if position_ids is None:
            position_ids = torch.arange(
                0, input_ids.shape[1], dtype=torch.long, device=input_ids.device
            )
        if labels is None:
            labels = torch.full_like(input_ids, IGNORE_INDEX)

        # remove the padding using attention_mask -- FIXME
        _input_ids = input_ids
        input_ids = [
            cur_input_ids[cur_attention_mask]
            for cur_input_ids, cur_attention_mask in zip(input_ids, attention_mask)
        ]
        labels = [
            cur_labels[cur_attention_mask]
            for cur_labels, cur_attention_mask in zip(labels, attention_mask)
        ]

        new_input_embeds = []
        new_labels = []
        cur_image_idx = 0
        for batch_idx, cur_input_ids in enumerate(input_ids):
            num_images = (cur_input_ids == IMAGE_TOKEN_INDEX).sum()
            if num_images == 0:
                cur_image_features = image_features[cur_image_idx]
                cur_input_embeds_1 = self.get_model().embed_tokens(cur_input_ids)
                cur_input_embeds = torch.cat(
                    [cur_input_embeds_1, cur_image_features[0:0]], dim=0
                )
                new_input_embeds.append(cur_input_embeds)
                new_labels.append(labels[batch_idx])
                cur_image_idx += 1
                continue

            ##splitting input_ids based on where the IMAGE_TOKEN_INDEX is
            image_token_indices = (
                [-1]
                + torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0].tolist()
                + [cur_input_ids.shape[0]]
            )
            cur_input_ids_noim = []
            cur_labels = labels[batch_idx]
            cur_labels_noim = []
            for i in range(len(image_token_indices) - 1):
                cur_input_ids_noim.append(
                    cur_input_ids[
                        image_token_indices[i] + 1 : image_token_indices[i + 1]
                    ]
                )
                cur_labels_noim.append(
                    cur_labels[image_token_indices[i] + 1 : image_token_indices[i + 1]]
                )

            ## keep acting on split version of token
            ## creating two split types of embeddings based on before/after IMAGE_TOKEN_INDEX
            split_sizes = [x.shape[0] for x in cur_labels_noim]
            cur_input_embeds = self.get_model().embed_tokens(
                torch.cat(cur_input_ids_noim)
            )
            cur_input_embeds_no_im = torch.split(cur_input_embeds, split_sizes, dim=0)
            # print('cur_input_embeds', cur_input_embeds)
            # print('cur_input_embeds_no_im', cur_input_embeds_no_im)
            cur_new_input_embeds = []
            cur_new_labels = []

            ## adding for each image (in our case 1) the image-embedding between the text embedding from before the IMAGE_TOKEN_INDEX and the text embedding after.
            for i in range(num_images + 1):
                cur_new_input_embeds.append(cur_input_embeds_no_im[i])
                cur_new_labels.append(cur_labels_noim[i])
                if i < num_images:
                    cur_image_features = image_features[cur_image_idx]
                    cur_image_idx += 1
                    cur_new_input_embeds.append(cur_image_features)
                    cur_new_labels.append(
                        torch.full(
                            (cur_image_features.shape[0],),
                            IGNORE_INDEX,
                            device=cur_labels.device,
                            dtype=cur_labels.dtype,
                        )
                    )

            cur_new_input_embeds = [x.to(self.device) for x in cur_new_input_embeds]

            cur_new_input_embeds = torch.cat(cur_new_input_embeds)
            cur_new_labels = torch.cat(cur_new_labels)

            new_input_embeds.append(cur_new_input_embeds)
            new_labels.append(cur_new_labels)
            ## new_input_embeds and new_labels are the input to the model which include both image and text embeddings. 'text..text <image> text..text' --> [text_embedding] + [image_embedding] + [text_embedding]

        # Truncate sequences to max length as image embeddings can make the sequence longer
        tokenizer_model_max_length = getattr(
            self.config, "tokenizer_model_max_length", None
        )
        if tokenizer_model_max_length is not None:
            new_input_embeds = [
                x[:tokenizer_model_max_length] for x in new_input_embeds
            ]
            new_labels = [x[:tokenizer_model_max_length] for x in new_labels]

        # Combine them
        max_len = max(x.shape[0] for x in new_input_embeds)
        batch_size = len(new_input_embeds)

        new_input_embeds_padded = []
        new_labels_padded = torch.full(
            (batch_size, max_len),
            IGNORE_INDEX,
            dtype=new_labels[0].dtype,
            device=new_labels[0].device,
        )
        attention_mask = torch.zeros(
            (batch_size, max_len),
            dtype=attention_mask.dtype,
            device=attention_mask.device,
        )
        position_ids = torch.zeros(
            (batch_size, max_len), dtype=position_ids.dtype, device=position_ids.device
        )

        for i, (cur_new_embed, cur_new_labels) in enumerate(
            zip(new_input_embeds, new_labels)
        ):
            cur_len = cur_new_embed.shape[0]
            if getattr(self.config, "tokenizer_padding_side", "right") == "left":
                new_input_embeds_padded.append(
                    torch.cat(
                        (
                            torch.zeros(
                                (max_len - cur_len, cur_new_embed.shape[1]),
                                dtype=cur_new_embed.dtype,
                                device=cur_new_embed.device,
                            ),
                            cur_new_embed,
                        ),
                        dim=0,
                    )
                )
                if cur_len > 0:
                    new_labels_padded[i, -cur_len:] = cur_new_labels
                    attention_mask[i, -cur_len:] = True
                    position_ids[i, -cur_len:] = torch.arange(
                        0, cur_len, dtype=position_ids.dtype, device=position_ids.device
                    )
            else:
                new_input_embeds_padded.append(
                    torch.cat(
                        (
                            cur_new_embed,
                            torch.zeros(
                                (max_len - cur_len, cur_new_embed.shape[1]),
                                dtype=cur_new_embed.dtype,
                                device=cur_new_embed.device,
                            ),
                        ),
                        dim=0,
                    )
                )
                if cur_len > 0:
                    new_labels_padded[i, :cur_len] = cur_new_labels
                    attention_mask[i, :cur_len] = True
                    position_ids[i, :cur_len] = torch.arange(
                        0, cur_len, dtype=position_ids.dtype, device=position_ids.device
                    )

        new_input_embeds = torch.stack(new_input_embeds_padded, dim=0)

        if _labels is None:
            new_labels = None
        else:
            new_labels = new_labels_padded

        if _attention_mask is None:
            attention_mask = None
        else:
            attention_mask = attention_mask.to(dtype=_attention_mask.dtype)

        if _position_ids is None:
            position_ids = None

        return (
            None,
            position_ids,
            attention_mask,
            past_key_values,
            new_input_embeds,
            new_labels,
        )

    def initialize_vision_tokenizer(self, model_args, tokenizer):
        if model_args.mm_use_im_patch_token:
            ## dynamically adding the DEFAULT_IMAGE_PATCH_TOKEN to the special tokens
            tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))

        if model_args.mm_use_im_start_end:
            ## dynamically adding the DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN to the special tokens
            num_new_tokens = tokenizer.add_tokens(
                [DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True
            )
            self.resize_token_embeddings(len(tokenizer))

            if num_new_tokens > 0:
                input_embeddings = self.get_input_embeddings().weight.data
                output_embeddings = self.get_output_embeddings().weight.data

                input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
                    dim=0, keepdim=True
                )
                output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
                    dim=0, keepdim=True
                )

                input_embeddings[-num_new_tokens:] = input_embeddings_avg
                output_embeddings[-num_new_tokens:] = output_embeddings_avg

                # assert num_new_tokens == 2
                # output_embeddings[-num_new_tokens:] = mm_bom_mask_token_weight

            if model_args.tune_mm_mlp_adapter:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = True
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False

            if model_args.pretrain_mm_mlp_adapter:
                mm_projector_weights = torch.load(
                    model_args.pretrain_mm_mlp_adapter, map_location="cpu"
                )
                embed_tokens_weight = mm_projector_weights["model.embed_tokens.weight"]
                assert num_new_tokens == 2
                if input_embeddings.shape == embed_tokens_weight.shape:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight[
                        -num_new_tokens:
                    ]
                elif embed_tokens_weight.shape[0] == num_new_tokens:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight
                else:
                    raise ValueError(
                        f"Unexpected embed_tokens_weight shape. Pretrained: {embed_tokens_weight.shape}. Current: {input_embeddings.shape}. Numer of new tokens: {num_new_tokens}."
                    )

        elif model_args.mm_use_im_patch_token:
            if model_args.tune_mm_mlp_adapter:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = False
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False
