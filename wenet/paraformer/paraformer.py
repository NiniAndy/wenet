# Copyright (c) 2020 Mobvoi Inc. (authors: Binbin Zhang, Di Wu)
#               2023 ASLP@NWPU (authors: He Wang, Fan Yu)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# Modified from ESPnet(https://github.com/espnet/espnet) and
# FunASR(https://github.com/alibaba-damo-academy/FunASR)

from typing import Dict, List, Optional, Tuple

import torch
from wenet.paraformer.cif import Cif, cif_without_hidden
from wenet.paraformer.layers import LFR, SanmDecoder, SanmEncoder
from wenet.paraformer.search import (paraformer_beam_search,
                                     paraformer_greedy_search)
from wenet.transformer.asr_model import ASRModel
from wenet.transformer.ctc import CTC
from wenet.transformer.decoder import TransformerDecoder
from wenet.transformer.encoder import BaseEncoder
from wenet.transformer.search import (DecodeResult, ctc_greedy_search,
                                      ctc_prefix_beam_search)
from wenet.utils.common import IGNORE_ID, add_sos_eos, th_accuracy
from wenet.utils.mask import make_non_pad_mask
from wenet.utils.class_module import WENET_PREDICTOR_CLASSES
from wenet.transducer.predictor import PredictorBase




# class Paraformer(ASRModel):
""" Paraformer: Fast and Accurate Parallel Transformer for
    Non-autoregressive End-to-End Speech Recognition
    see https://arxiv.org/pdf/2206.08317.pdf

"""

    # def __init__(self,
    #              vocab_size: int,
    #              encoder: BaseEncoder,
    #              decoder: TransformerDecoder,
    #              predictor: Predictor,
    #              ctc: CTC,
    #              ctc_weight: float = 0.5,
    #              ignore_id: int = -1,
    #              lsm_weight: float = 0,
    #              length_normalized_loss: bool = False,
    #              sampler: bool = True,
    #              sampling_ratio: float = 0.75,
    #              add_eos: bool = False,
    #              special_tokens: Optional[Dict] = None,
    #              apply_non_blank_embedding: bool = False):
    #     # assert isinstance(encoder, SanmEncoder), isinstance(decoder, SanmDecoder)
    #     super().__init__(
    #         vocab_size,
    #         encoder,
    #         decoder,
    #         ctc,
    #         ctc_weight,
    #         IGNORE_ID,
    #         0.0,
    #         lsm_weight,
    #         length_normalized_loss,
    #         None,
    #         apply_non_blank_embedding)

class Paraformer(ASRModel):
    def __init__(
            self,
            input_dim: int,
            output_dim: int,
            encoder: BaseEncoder,
            encoder_conf: dict,
            predictor: PredictorBase,
            predictor_conf: dict,
            decoder: TransformerDecoder,
            decoder_conf: dict,
            ctc: CTC,
            ctc_conf: dict,
            cmvn: Optional[str] = None,
            cmvn_conf: Optional[dict] = None,
            **kwargs: dict,
    ):
        super().__init__(
            input_dim,
            output_dim,
            encoder,
            encoder_conf,
            decoder,
            decoder_conf,
            ctc,
            ctc_conf,
            cmvn,
            cmvn_conf,
            **kwargs,
            )

        if self.ctc_weight == 0.0:
            del ctc

        predictor = WENET_PREDICTOR_CLASSES[predictor](**predictor_conf)
        self.predictor = predictor

        assert self.special_tokens is not None
        self.sos = self.special_tokens['<sos>']
        self.eos = self.special_tokens['<eos>']

        self.sampling_ratio = kwargs['model_conf'].get('sampling_ratio', 0.0)
        self.sampler = self.sampling_ratio > 0.0
        self.add_eos = kwargs['model_conf'].get('add_eos', False)

    @torch.jit.unused
    def forward(
        self,
        batch: Dict,
        device: torch.device,
    ) -> Dict[str, Optional[torch.Tensor]]:
        """Frontend + Encoder + Predictor + Decoder + Calc loss
        """
        speech = batch['feats'].to(device)
        speech_lengths = batch['feats_lengths'].to(device)
        text = batch['target'].to(device)
        text_lengths = batch['target_lengths'].to(device)

        # 0 encoder
        encoder_out, encoder_mask = self.encoder(speech, speech_lengths)
        encoder_out_lens = encoder_mask.squeeze(1).sum(1)

        # 1 predictor
        ys_pad, ys_pad_lens = text, text_lengths
        if self.add_eos:
            _, ys_pad = add_sos_eos(text, self.sos, self.eos, self.ignore_id)
            ys_pad_lens = text_lengths + 1
        acoustic_embd, token_num, _, _, _, tp_token_num, _ = self.predictor(encoder_out, ys_pad, encoder_mask, self.ignore_id)

        # 2 decoder with sampler
        # TODO(Mddct): support mwer here
        decoder_out_1st = None
        if self.sampler:
            acoustic_embd, decoder_out_1st = self._sampler(encoder_out, encoder_mask, ys_pad, ys_pad_lens, acoustic_embd)

        # 3 loss
        # 3.1 ctc branhch
        loss_ctc: Optional[torch.Tensor] = None
        if self.ctc_weight != 0.0:
            loss_ctc, _ = self._forward_ctc(encoder_out, encoder_mask, text, text_lengths)
        # 3.2 quantity loss for cif
        loss_quantity = torch.nn.functional.l1_loss(token_num, ys_pad_lens.to(token_num.dtype), reduction='sum',)
        loss_quantity = loss_quantity / ys_pad_lens.sum().to(token_num.dtype)
        loss_quantity_tp = torch.nn.functional.l1_loss(tp_token_num, ys_pad_lens.to(token_num.dtype),reduction='sum') / ys_pad_lens.sum().to(token_num.dtype)
        infos = {"acoustic_embd": acoustic_embd, "decoder_out_1st": decoder_out_1st}
        loss_decoder, acc_att = self._calc_att_loss(encoder_out, encoder_mask, ys_pad, ys_pad_lens, infos)

        loss = loss_decoder
        if loss_ctc is not None:
            loss = loss + self.ctc_weight * loss_ctc
        loss = loss + loss_quantity + loss_quantity_tp
        return {
            "loss": loss,
            "loss_ctc": loss_ctc,
            "loss_decoder": loss_decoder,
            "loss_quantity": loss_quantity,
            "loss_quantity_tp": loss_quantity_tp,
            "th_accuracy": acc_att,
        }


    def _calc_att_loss(
        self,
        encoder_out: torch.Tensor,
        encoder_mask: torch.Tensor,
        ys_pad: torch.Tensor,
        ys_pad_lens: torch.Tensor,
        infos: Dict[str, List[str]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        ys_pad_emb = infos['acoustic_embd']
        decoder_out_1st = infos['decoder_out_1st']
        decoder_out, _ = self.decoder(encoder_out, encoder_mask, ys_pad_emb, ys_pad_lens)
        if decoder_out_1st is None:
            decoder_out_1st = decoder_out
        loss_att = self.criterion_att(decoder_out, ys_pad)
        acc_att = th_accuracy(decoder_out_1st.view(-1, self.vocab_size), ys_pad, ignore_label=self.ignore_id)
        return loss_att, acc_att


    @torch.jit.unused
    def _sampler(self, encoder_out, encoder_out_mask, ys_pad, ys_pad_lens, pre_acoustic_embeds):
        device = encoder_out.device
        B, _ = ys_pad.size()

        tgt_mask = make_non_pad_mask(ys_pad_lens)
        # zero the ignore id
        ys_pad = ys_pad * tgt_mask
        ys_pad_embed, _ = self.decoder.embed(ys_pad)
        with torch.no_grad():
            decoder_out, _ = self.decoder(encoder_out, encoder_out_mask, pre_acoustic_embeds, ys_pad_lens)
            pred_tokens = decoder_out.argmax(-1)
            nonpad_positions = tgt_mask
            same_num = ((pred_tokens == ys_pad) * nonpad_positions).sum(1)
            input_mask = torch.ones_like(nonpad_positions, device=device, dtype=tgt_mask.dtype,)
            for li in range(B):
                target_num = (ys_pad_lens[li] - same_num[li].sum()).float() * self.sampling_ratio
                target_num = target_num.long()
                if target_num > 0:
                    input_mask[li].scatter_(dim=0, index=torch.randperm(ys_pad_lens[li], device=device)[:target_num], value=0,)
            input_mask = torch.where(input_mask > 0, 1, 0)
            input_mask = input_mask * tgt_mask
            input_mask_expand = input_mask.unsqueeze(2)  # [B, T, 1]

        sematic_embeds = torch.where(input_mask_expand == 1, pre_acoustic_embeds, ys_pad_embed)
        # zero out the paddings
        return sematic_embeds * tgt_mask.unsqueeze(2), decoder_out * tgt_mask.unsqueeze(2)

    def _forward_encoder(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        decoding_chunk_size: int = -1,
        num_decoding_left_chunks: int = -1,
        simulate_streaming: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Let's assume B = batch_size
        # 1. Encoder
        if simulate_streaming and decoding_chunk_size > 0:
            encoder_out, encoder_mask = self.encoder.forward_chunk_by_chunk(
                speech,
                decoding_chunk_size=decoding_chunk_size,
                num_decoding_left_chunks=num_decoding_left_chunks
            )  # (B, maxlen, encoder_dim)
        else:
            encoder_out, encoder_mask = self.encoder(
                speech,
                speech_lengths,
                decoding_chunk_size=decoding_chunk_size,
                num_decoding_left_chunks=num_decoding_left_chunks
            )  # (B, maxlen, encoder_dim)
        return encoder_out, encoder_mask

    @torch.jit.export
    def forward_paraformer(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        res = self._forward_paraformer(speech, speech_lengths)
        return res['decoder_out'], res['decoder_out_lens'], res['tp_alphas'], res['tp_mask'].sum(1).squeeze(-1)

    @torch.jit.export
    def forward_encoder_chunk(
        self,
        xs: torch.Tensor,
        offset: int,
        required_cache_size: int,
        att_cache: torch.Tensor = torch.zeros(0, 0, 0, 0),
        cnn_cache: torch.Tensor = torch.zeros(0, 0, 0, 0),
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # TODO(Mddct): fix
        xs_lens = torch.tensor(xs.size(1), dtype=torch.int)
        encoder_out, _ = self._forward_encoder(xs, xs_lens)
        return encoder_out, att_cache, cnn_cache

    @torch.jit.export
    def forward_cif_peaks(self, alphas: torch.Tensor, token_nums: torch.Tensor) -> torch.Tensor:
        cif2_token_nums = alphas.sum(-1)
        scale_alphas = alphas / (cif2_token_nums / token_nums).unsqueeze(1)
        peaks = cif_without_hidden(scale_alphas, self.predictor.predictor.threshold - 1e-4)

        return peaks

    def _forward_paraformer(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        decoding_chunk_size: int = -1,
        num_decoding_left_chunks: int = -1,
    ) -> Dict[str, torch.Tensor]:
        # encoder
        encoder_out, encoder_out_mask = self._forward_encoder(
            speech, speech_lengths, decoding_chunk_size,
            num_decoding_left_chunks)

        # cif predictor
        acoustic_embed, token_num, _, _, tp_alphas, _, tp_mask = self.predictor(encoder_out,mask=encoder_out_mask,)
        token_num = token_num.floor().to(speech_lengths.dtype)

        # decoder
        decoder_out, _ = self.decoder(encoder_out, encoder_out_mask, acoustic_embed, token_num)
        decoder_out = decoder_out.log_softmax(dim=-1)

        return {
            "encoder_out": encoder_out,
            "encoder_out_mask": encoder_out_mask,
            "decoder_out": decoder_out,
            "tp_alphas": tp_alphas,
            "decoder_out_lens": token_num,
            "tp_mask": tp_mask
        }

    def decode(
        self,
        methods: List[str],
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        beam_size: int,
        decoding_chunk_size: int = -1,
        num_decoding_left_chunks: int = -1,
        ctc_weight: float = 0,
        simulate_streaming: bool = False,
        reverse_weight: float = 0,
        context_graph=None,
        blank_id: int = 0,
        blank_penalty: float = 0.0,
        length_penalty: float = 0.0,
        infos: Dict[str, List[str]] = None,
    ) -> Dict[str, List[DecodeResult]]:
        res = self._forward_paraformer(speech, speech_lengths,
                                       decoding_chunk_size,
                                       num_decoding_left_chunks)
        encoder_out, encoder_mask, decoder_out, decoder_out_lens, tp_alphas = res[
            'encoder_out'], res['encoder_out_mask'], res['decoder_out'], res[
                'decoder_out_lens'], res['tp_alphas']
        peaks = self.forward_cif_peaks(tp_alphas, decoder_out_lens)
        results = {}
        if 'paraformer_greedy_search' in methods:
            assert decoder_out is not None
            assert decoder_out_lens is not None
            paraformer_greedy_result = paraformer_greedy_search(
                decoder_out, decoder_out_lens, peaks)
            results['paraformer_greedy_search'] = paraformer_greedy_result
        if 'paraformer_beam_search' in methods:
            assert decoder_out is not None
            assert decoder_out_lens is not None
            paraformer_beam_result = paraformer_beam_search(
                decoder_out,
                decoder_out_lens,
                beam_size=beam_size,
                eos=self.eos)
            results['paraformer_beam_search'] = paraformer_beam_result
        if 'ctc_greedy_search' in methods or 'ctc_prefix_beam_search' in methods:
            ctc_probs = self.ctc_logprobs(encoder_out, blank_penalty, blank_id)
            encoder_lens = encoder_mask.squeeze(1).sum(1)
            if 'ctc_greedy_search' in methods:
                results['ctc_greedy_search'] = ctc_greedy_search(
                    ctc_probs, encoder_lens, blank_id)
            if 'ctc_prefix_beam_search' in methods:
                ctc_prefix_result = ctc_prefix_beam_search(
                    ctc_probs, encoder_lens, beam_size, context_graph,
                    blank_id)
                results['ctc_prefix_beam_search'] = ctc_prefix_result
        return results
