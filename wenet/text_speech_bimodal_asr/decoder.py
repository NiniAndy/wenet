import logging
import os
from typing import Dict, Tuple, List, Optional

import torch
import torch.utils.checkpoint as ckpt

from wenet.transformer.attention import T_CACHE
from wenet.transformer.decoder_layer import DecoderLayer
from wenet.utils.class_utils import (
    WENET_EMB_CLASSES,
    WENET_ATTENTION_CLASSES,
    WENET_ACTIVATION_CLASSES,
    WENET_MLP_CLASSES,
    WENET_NORM_CLASSES,
    WENET_SUBSAMPLE_CLASSES,
)
from wenet.utils.common import mask_to_bias
from wenet.utils.mask import (subsequent_mask, make_pad_mask)


class ContextDecoder(torch.nn.Module):
    """Base class of Transfomer context_decoder module.
    Args:
        vocab_size: output dim
        encoder_output_size: dimension of attention
        attention_heads: the number of heads of multi head attention
        linear_units: the hidden units number of position-wise feedforward
        num_blocks: the number of context_decoder blocks
        dropout_rate: dropout rate
        self_attention_dropout_rate: dropout rate for attention
        input_layer: input layer type
        use_output_layer: whether to use output layer
        pos_enc_class: PositionalEncoding or ScaledPositionalEncoding
        normalize_before:
            True: use layer_norm before each sub-block of a layer.
            False: use layer_norm after each sub-block of a layer.
        src_attention: if false, audio_encoder-context_decoder cross attention is not
                       applied, such as CIF model
        query_bias: whether use bias in attention.linear_q
        key_bias: whether use bias in attention.linear_k, False for whisper models.
        value_bias: whether use bias in attention.linear_v
        gradient_checkpointing: rerunning a forward-pass segment for each
            checkpointed segment during backward.
        tie_word_embedding: Tie or clone module weights depending of whether we are
            using TorchScript or not
    """

    def __init__(
            self,
            vocab_size: int,
            encoder_output_size: int,
            combine_method: str = 'add',
            attention_heads: int = 4,
            linear_units: int = 2048,
            num_blocks: int = 6,
            dropout_rate: float = 0.1,
            positional_dropout_rate: float = 0.1,
            self_attention_dropout_rate: float = 0.0,
            src_attention_dropout_rate: float = 0.0,
            input_layer: str = "embed",
            use_output_layer: bool = True,
            normalize_before: bool = True,
            src_attention: bool = True,
            query_bias: bool = True,
            key_bias: bool = True,
            value_bias: bool = True,
            activation_type: str = "relu",
            gradient_checkpointing: bool = False,
            tie_word_embedding: bool = False,
            use_sdpa: bool = False,
            layer_norm_type: str = 'layer_norm',
            norm_eps: float = 1e-5,
            n_kv_head: Optional[int] = None,
            head_dim: Optional[int] = None,
            mlp_type: str = 'position_wise_feed_forward',
            mlp_bias: bool = True,
            n_expert: int = 8,
            n_expert_activated: int = 2,
    ):
        super().__init__()
        attention_dim = encoder_output_size
        activation = WENET_ACTIVATION_CLASSES[activation_type]()

        self.combine_method = combine_method




        self.embed = torch.nn.Sequential(
            torch.nn.Identity() if input_layer == "no_pos" else
            torch.nn.Embedding(vocab_size, attention_dim),
            WENET_EMB_CLASSES[input_layer](attention_dim, positional_dropout_rate),
        )

        if combine_method == 'concat':
            proj_input_size = 2 * attention_dim
            proj_output_size = attention_dim

            pos_projs_class = WENET_EMB_CLASSES["no_pos"]
            proj = WENET_SUBSAMPLE_CLASSES[input_layer](
                proj_input_size,
                proj_output_size,
                0.0,
                pos_projs_class(proj_output_size, positional_dropout_rate)
            )
            self.emb_proj = proj
            self.projs = torch.nn.ModuleList([proj for _ in range(num_blocks)])

        assert layer_norm_type in ['layer_norm', 'rms_norm']
        self.normalize_before = normalize_before
        self.after_norm = WENET_NORM_CLASSES[layer_norm_type](attention_dim,
                                                              eps=norm_eps)
        self.use_output_layer = use_output_layer
        if use_output_layer:
            self.output_layer = torch.nn.Linear(attention_dim, vocab_size)
        else:
            self.output_layer = torch.nn.Identity()
        self.num_blocks = num_blocks

        mlp_class = WENET_MLP_CLASSES[mlp_type]
        self.decoders = torch.nn.ModuleList([
            DecoderLayer(
                attention_dim,
                WENET_ATTENTION_CLASSES["selfattn"](
                    attention_heads, attention_dim,
                    self_attention_dropout_rate, query_bias, key_bias,
                    value_bias, use_sdpa, n_kv_head, head_dim),
                WENET_ATTENTION_CLASSES["crossattn"](
                    attention_heads, attention_dim, src_attention_dropout_rate,
                    query_bias, key_bias, value_bias, use_sdpa, n_kv_head,
                    head_dim) if src_attention else None,
                mlp_class(attention_dim,
                          linear_units,
                          dropout_rate,
                          activation,
                          mlp_bias,
                          n_expert=n_expert,
                          n_expert_activated=n_expert_activated),
                dropout_rate,
                normalize_before,
                layer_norm_type,
                norm_eps,
            ) for _ in range(self.num_blocks)
        ])

        self.gradient_checkpointing = gradient_checkpointing
        self.tie_word_embedding = tie_word_embedding
        self.use_sdpa = use_sdpa

    def forward(
            self,
            memory: torch.Tensor,
            memory_mask: torch.Tensor,
            ys_in_pad: torch.Tensor,
            ys_in_lens: torch.Tensor,
            r_ys_in_pad: torch.Tensor = torch.empty(0),
            combine_dict: Dict[str, torch.Tensor] = None,
            reverse_weight: float = 0.0,
            return_layers_output: bool = False,
    ):
        """Forward context_decoder.
        Args:
            memory: encoded memory, float32  (batch, maxlen_in, feat)
            memory_mask: audio_encoder memory mask, (batch, 1, maxlen_in)
            ys_in_pad: padded input token ids, int64 (batch, maxlen_out)
            ys_in_lens: input lengths of this batch (batch)
            r_ys_in_pad: not used in transformer context_decoder, in order to unify api
                with bidirectional context_decoder
            reverse_weight: not used in transformer context_decoder, in order to unify
                api with bidirectional decode
        Returns:
            (tuple): tuple containing:
                x: decoded token score before softmax (batch, maxlen_out,
                    vocab_size) if use_output_layer is True,
                torch.tensor(0.0), in order to unify api with bidirectional context_decoder
                olens: (batch, )
        NOTE(xcsong):
            We pass the `__call__` method of the modules instead of `forward` to the
            checkpointing API because `__call__` attaches all the hooks of the module.
            https://discuss.pytorch.org/t/any-different-between-model-input-and-model-forward-input/3690/2
        """
        tgt = ys_in_pad
        maxlen = tgt.size(1)
        # tgt_mask: (B, 1, L)
        tgt_mask = ~make_pad_mask(ys_in_lens, maxlen).unsqueeze(1)
        tgt_mask = tgt_mask.to(tgt.device)
        # m: (1, L, L)
        m = subsequent_mask(tgt_mask.size(-1), device=tgt_mask.device).unsqueeze(0)
        # tgt_mask: (B, L, L)
        tgt_mask = tgt_mask & m
        if self.use_sdpa:
            tgt_mask = mask_to_bias(tgt_mask, memory.dtype)
            memory_mask = mask_to_bias(memory_mask, memory.dtype)

        x, _ = self.embed(tgt)

        if self.combine_method == 'add':
            combine_emb = combine_dict['emb']
            x = x + combine_emb
        elif self.combine_method == 'concat':
            combine_emb = combine_dict['emb']
            x = self.emb_proj(torch.cat([x, combine_emb], dim=-1))
        else:
            raise ValueError(f"Unknown combine_method: {self.combine_method}")

        emb = x
        if self.gradient_checkpointing and self.training:
            x, x_dict = self.forward_layers_checkpointed(x, tgt_mask, memory, memory_mask, return_layers_output)
        else:
            x, x_dict = self.forward_layers(x, tgt_mask, memory, memory_mask, combine_dict, return_layers_output)

        if self.normalize_before:
            x = self.after_norm(x)

        if self.use_output_layer:
            x = self.output_layer(x)
        olens = tgt_mask.sum(1)
        if return_layers_output:
            x_dict['emb'] = emb
            return x, torch.tensor(0.0), olens, x_dict
        else:
            return x, torch.tensor(0.0), olens

    def forward_layers(
            self,
            x: torch.Tensor,
            tgt_mask: torch.Tensor,
            memory: torch.Tensor,
            memory_mask: torch.Tensor,
            combine_dict: Dict[str, torch.Tensor] = None,
            return_layers_output=False) -> torch.Tensor:

        x_dict = {}
        for i, layer in enumerate(self.decoders):
            x, tgt_mask, memory, memory_mask = layer(x, tgt_mask, memory, memory_mask)
            corresponding_layer_x = combine_dict[f'layer_{i}']
            if self.combine_method == 'add':
                x = x + corresponding_layer_x
            elif self.combine_method == 'concat':
                xs, *_ = self.projs[i](torch.cat([x, corresponding_layer_x], dim=-1), tgt_mask)
            else:
                raise ValueError(f"Unknown combine_method: {self.combine_method}")
            if return_layers_output:
                x_dict[f'layer_{i}'] = self.after_norm(x) if self.normalize_before else x
        return x, x_dict

    @torch.jit.unused
    def forward_layers_checkpointed(
            self,
            x: torch.Tensor,
            tgt_mask: torch.Tensor,
            memory: torch.Tensor,
            memory_mask: torch.Tensor,
            return_layers_output=False  ) -> torch.Tensor:
        x_dict = {}
        for i, layer in enumerate(self.decoders):
            x, tgt_mask, memory, memory_mask = ckpt.checkpoint(
                layer.__call__,
                x,
                tgt_mask,
                memory,
                memory_mask,
                use_reentrant=False)
            if return_layers_output:
                x_dict[f'layer_{i}'] = self.after_norm(x) if self.normalize_before else x
        return x, x_dict

    def forward_one_step(
            self,
            memory: torch.Tensor,
            memory_mask: torch.Tensor,
            tgt: torch.Tensor,
            tgt_mask: torch.Tensor,
            cache: Dict[str, Dict[str, T_CACHE]],
    ) -> torch.Tensor:
        """Forward one step.
            This is only used for decoding.
        Args:
            memory: encoded memory, float32  (batch, maxlen_in, feat)
            memory_mask: encoded memory mask, (batch, 1, maxlen_in)
            tgt: input token ids, int64 (batch, maxlen_out)
            tgt_mask: input token mask,  (batch, maxlen_out)
                      dtype=torch.uint8 in PyTorch 1.2-
                      dtype=torch.bool in PyTorch 1.2+ (include 1.2)
            cache: cached output list of (batch, max_time_out-1, size)
        Returns:
            y, cache: NN output value and cache per `self.decoders`.
            y.shape` is (batch, maxlen_out, token)
        """
        x, _ = self.embed(tgt)
        update_cross_att_cache = True
        if len(cache['cross_att_cache']) != 0:
            assert len(cache['cross_att_cache']) == self.num_blocks
            update_cross_att_cache = False
        for i, decoder in enumerate(self.decoders):
            layer_i = 'layer_{}'.format(i)
            self_att_cache = cache['self_att_cache'].get(layer_i, None)
            cross_att_cache = cache['cross_att_cache'].get(layer_i, None)
            c = {
                'self_att_cache': self_att_cache,
                'cross_att_cache': cross_att_cache,
            }

            x, tgt_mask, memory, memory_mask = decoder(x, tgt_mask, memory, memory_mask, cache=c)

            # update cache dict
            assert c['self_att_cache'] is not None
            assert c['cross_att_cache'] is not None
            cache['self_att_cache'][layer_i] = c['self_att_cache']
            if update_cross_att_cache:
                cache['cross_att_cache'][layer_i] = c['cross_att_cache']

        if self.normalize_before:
            y = self.after_norm(x[:, -1])
        else:
            y = x[:, -1]
        if self.use_output_layer:
            y = torch.log_softmax(self.output_layer(y), dim=-1)
        return y

    def tie_or_clone_weights(self, jit_mode: bool = True):
        """Tie or clone module weights (between word_emb and output_layer)
            depending of whether we are using TorchScript or not"""
        rank = int(os.environ.get('RANK', 0))
        if not self.use_output_layer:
            return
        if not self.tie_word_embedding:
            return
        if jit_mode:
            if rank == 0:
                logging.info("clone emb.weight to output.weight")
            self.output_layer.weight = torch.nn.Parameter(
                self.embed[0].weight.clone())
        else:
            if rank == 0:
                logging.info("tie emb.weight with output.weight")
            self.output_layer.weight = self.embed[0].weight

        if getattr(self.output_layer, "bias", None) is not None:
            self.output_layer.bias.data = torch.nn.functional.pad(
                self.output_layer.bias.data,
                (
                    0,
                    self.output_layer.weight.shape[0] -
                    self.output_layer.bias.shape[0],
                ),
                "constant",
                0,
            )

