# Copyright (c) 2021 Mobvoi Inc (Binbin Zhang, Di Wu)
#               2022 Xingchen Song (sxc19@mails.tsinghua.edu.cn)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# Modified from ESPnet(https://github.com/espnet/espnet)
"""Encoder definition."""
from typing import Optional

import torch

from wenet.transformer.encoder_layer import TransformerEncoderLayer
from wenet.utils.class_utils import (
    WENET_EMB_CLASSES,
    WENET_MLP_CLASSES,
    WENET_NORM_CLASSES,
    WENET_SUBSAMPLE_CLASSES,
    WENET_ATTENTION_CLASSES,
    WENET_ACTIVATION_CLASSES,
)
from wenet.utils.mask import make_pad_mask


class ContextEncoder(torch.nn.Module):

    def __init__(
            self,
            input_size: int,
            global_cmvn: torch.nn.Module = None,  # 占位符
            output_size: int = 256,
            attention_heads: int = 4,
            linear_units: int = 2048,
            num_blocks: int = 6,
            dropout_rate: float = 0.1,
            positional_dropout_rate: float = 0.1,
            attention_dropout_rate: float = 0.0,
            input_layer: str = "embed",
            pos_enc_layer_type: str = "abs_pos",
            normalize_before: bool = True,
            use_sdpa: bool = False,
            layer_norm_type: str = 'layer_norm',
            norm_eps: float = 1e-5,
            selfattention_layer_type: str = "selfattn",
            mlp_type: str = 'position_wise_feed_forward',
            mlp_bias: bool = True,
            n_expert: int = 8,
            n_expert_activated: int = 2,
            query_bias: bool = True,
            key_bias: bool = True,
            value_bias: bool = True,
            activation_type: str = "relu",
            n_kv_head: Optional[int] = None,
            head_dim: Optional[int] = None,
    ):

        super().__init__()

        self._output_size = output_size

        pos_emb_class = WENET_EMB_CLASSES[pos_enc_layer_type]

        self.embed = WENET_SUBSAMPLE_CLASSES[input_layer](
            input_size,
            output_size,
            dropout_rate,
            pos_emb_class(output_size, positional_dropout_rate)
            if pos_enc_layer_type != 'rope_pos' else pos_emb_class(output_size, output_size // attention_heads, positional_dropout_rate))

        pos_projs_class = WENET_EMB_CLASSES["no_pos"]
        proj = WENET_SUBSAMPLE_CLASSES[input_layer](
            input_size,
            output_size,
            0.0,
            pos_projs_class(output_size, positional_dropout_rate)
           )

        self.projs = torch.nn.ModuleList([proj for _ in range(num_blocks)])

        assert layer_norm_type in ['layer_norm', 'rms_norm']
        self.normalize_before = normalize_before
        self.after_norm = WENET_NORM_CLASSES[layer_norm_type](output_size, eps=norm_eps)

        assert selfattention_layer_type in ['selfattn', 'rope_abs_selfattn']
        activation = WENET_ACTIVATION_CLASSES[activation_type]()
        mlp_class = WENET_MLP_CLASSES[mlp_type]

        self.encoders = torch.nn.ModuleList([
            TransformerEncoderLayer(output_size,
                                    WENET_ATTENTION_CLASSES[selfattention_layer_type](
                                        attention_heads,
                                        output_size,
                                        attention_dropout_rate,
                                        query_bias,
                                        key_bias,
                                        value_bias,
                                        use_sdpa,
                                        n_kv_head,
                                        head_dim),
                                    mlp_class(output_size,
                                              linear_units,
                                              dropout_rate,
                                              activation,
                                              mlp_bias,
                                              n_expert=n_expert,
                                              n_expert_activated=n_expert_activated),
                                    dropout_rate,
                                    normalize_before,
                                    layer_norm_type=layer_norm_type,
                                    norm_eps=norm_eps,
                                    ) for _ in range(num_blocks)
        ])

    def output_size(self) -> int:
        return self._output_size

    def forward(
            self,
            xs: torch.Tensor,
            xs_lens: torch.Tensor,
            combine_context: dict,
            return_layers_output=False
    ):
        T = xs.size(1)
        masks = ~make_pad_mask(xs_lens, T).unsqueeze(1)  # (B, 1, T)
        cs_emb = combine_context["emb"]
        xs = torch.cat([xs, cs_emb], dim=-1)
        xs, pos_emb, masks = self.embed(xs, masks)
        mask_pad = masks  # (B, 1, T/subsample_rate)
        xs, xs_dict = self.forward_layers(xs, mask_pad, pos_emb, mask_pad, combine_context, return_layers_output)
        if self.normalize_before:
            xs = self.after_norm(xs)
        if return_layers_output:
            return xs, masks, xs_dict
        else:
            return xs, masks

    def forward_layers(
            self,
            xs: torch.Tensor,
            chunk_masks: torch.Tensor,
            pos_emb: torch.Tensor,
            mask_pad: torch.Tensor,
            combine_context: dict,
            return_layers_output=False):
        xs_dict = {}
        for i, layer in enumerate(self.encoders):
            xs, chunk_masks, _, _ = layer(xs, chunk_masks, pos_emb, mask_pad)
            cs = combine_context.get(f'layer_{i}', None)
            if cs is None:
                raise RuntimeError("No context for layer {}".format(i))
            xs = torch.cat([xs, cs], dim=-1)
            xs, *_ = self.projs[i](xs, mask_pad)
            if return_layers_output:
                xs_dict[f'layer_{i}'] = xs
        return xs, xs_dict
