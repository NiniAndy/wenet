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
from torch.nn.utils.rnn import pad_sequence

from wenet.paraformer.cif import Cif, cif_without_hidden
from wenet.paraformer.layers import LFR, SanmDecoder, SanmEncoder
from wenet.paraformer.search import (paraformer_beam_search,
                                     paraformer_greedy_search)
from wenet.transformer.asr_model import ASRModel
from wenet.transformer.ctc import CTC
from wenet.transformer.decoder import TransformerDecoder
from wenet.transformer.encoder import BaseEncoder
from wenet.transformer.search import (DecodeResult, ctc_greedy_search, ctc_prefix_beam_search)
from wenet.utils.common import IGNORE_ID, add_sos_eos, th_accuracy
from wenet.utils.mask import make_non_pad_mask



class ParaformerV2(ASRModel):
    def __init__(
            self,
            input_dim: int,
            output_dim: int,
            encoder: BaseEncoder,
            encoder_conf: dict,
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
            raise ValueError("ParaformerV2 requires ctc_weight > 0.0")

        assert self.special_tokens is not None
        self.sos = self.special_tokens['<sos>']
        self.eos = self.special_tokens['<eos>']
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

        # 0 audio_encoder
        encoder_out, encoder_mask = self.encoder(speech, speech_lengths)
        encoder_out_lens = encoder_mask.squeeze(1).sum(1)

        loss_ctc, _ = self._forward_ctc(encoder_out, encoder_mask, text, text_lengths)

        loss_decoder, acc_att = self._calc_att_loss(encoder_out, encoder_mask, text, text_lengths,)

        loss = loss_decoder
        if loss_ctc is not None:
            loss = loss + self.ctc_weight * loss_ctc

        return {
            "loss": loss,
            "loss_ctc": loss_ctc,
            "loss_decoder": loss_decoder,
            "th_accuracy": acc_att,
        }


    def _calc_att_loss(
        self,
        encoder_out: torch.Tensor,
        encoder_mask: torch.Tensor,
        ys_pad: torch.Tensor,
        ys_pad_lens: torch.Tensor,
        infos: Dict[str, List[str]] = None,
    ):
        encoder_out_lens = encoder_mask.squeeze(1).sum(1)
        batch_size = encoder_out.size(0)
        with torch.no_grad():
            compressed_ctc_batch  = []
            ctc_probs = self.ctc.log_softmax(encoder_out).detach()

            for b in range(batch_size):
                ctc_prob = ctc_probs[b][: encoder_out_lens[b]].cpu() # [T, N]
                text_b = ys_pad[b][: ys_pad_lens[b]].cpu() # [1, U]
                text_audio_alignment = self.ctc.force_align(ctc_prob, text_b)
                # text_audio_alignment =
                audio_text = self.ctc.remove_duplicates_and_blank(text_audio_alignment, self.blank_id)
                if len(audio_text) != ys_pad_lens[b]:
                    print (f"ctc alignment error: {audio_text}, {text_b}")
                # 把相同的不为0的帧的概率平均
                ctc_comp = self.average_repeats(ctc_prob, text_audio_alignment)
                if ctc_comp.size(0) != ys_pad_lens[b]:
                    print (f"ctc_comp error: {ctc_comp.size(0)}, {text_b}")
                compressed_ctc_batch.append(ctc_comp)

            padded_ctc_batch = pad_sequence(compressed_ctc_batch, batch_first=True).to(encoder_out.device)

        decoder_out, _ = self.decoder(encoder_out, encoder_mask, padded_ctc_batch, ys_pad_lens)

        loss_att = self.criterion_att(decoder_out, ys_pad)
        acc_att = th_accuracy(decoder_out.view(-1, self.vocab_size), ys_pad, ignore_label=self.ignore_id)

        return loss_att, acc_att



    def average_repeats(self, ctc_prob, alignment):
        """
        Averages the repeated frames based on alignment.

        Args:
            ctc_prob (torch.Tensor): Tensor of shape [T, VocabSize + 1] representing frame-wise CTC posteriors.
            alignment (torch.Tensor): Tensor of shape [T,] representing the target alignment from Viterbi algorithm.

        Returns:
            torch.Tensor: Compressed CTC posterior with repeated frames averaged and blanks removed.
        """
        unique_tokens = []
        unique_probs = []
        current_sum = None
        current_count = 0

        for t in range(alignment.size(0)):
            token = alignment[t].item()
            prob = ctc_prob[t]

            if len(unique_tokens) == 0 or token != unique_tokens[-1]:
                if current_count > 0:
                    unique_probs.append(current_sum / current_count)
                unique_tokens.append(token)
                current_sum = prob
                current_count = 1
            else:
                current_sum += prob
                current_count += 1

        # Append the last averaged probability
        if current_count > 0:
            unique_probs.append(current_sum / current_count)

        non_blank_ctc_prob = []
        responded_id = self.ctc.remove_duplicates(alignment)
        for i in range(len(responded_id)):
            id  = responded_id[i]
            if id == self.blank_id:
                continue
            else:
                non_blank_ctc_prob.append(unique_probs[i])

        compressed_ctc_prob = torch.stack(non_blank_ctc_prob, dim=0)
        return compressed_ctc_prob


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


    def forward_paraformer_v2(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        res = self._forward_paraformer(speech, speech_lengths)
        return res['decoder_out'], res['decoder_out_lens'], res['tp_alphas'], res['tp_mask'].sum(1).squeeze(-1)


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


    def _forward_paraformer_v2(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        decoding_chunk_size: int = -1,
        num_decoding_left_chunks: int = -1,
    ) -> Dict[str, torch.Tensor]:
        # audio_encoder
        encoder_out, encoder_out_mask = self._forward_encoder(speech, speech_lengths, decoding_chunk_size, num_decoding_left_chunks)

        batch_size = encoder_out.size(0)
        compressed_ctc_batch = []
        token_num = []
        ctc_probs = self.ctc.log_softmax(encoder_out).detach()
        ctc_outputs = self.ctc.argmax(encoder_out).detach()
        for b in range(batch_size):
            encoder_out_len_b = encoder_out_mask[b].squeeze(0).sum().item()
            ctc_prob = ctc_probs[b][: encoder_out_len_b] # [T, N]
            text_audio_alignment = ctc_outputs[b][: encoder_out_len_b]
            # 把相同的不为0的帧的概率平均
            ctc_comp = self.average_repeats(ctc_prob, text_audio_alignment)
            num = len(ctc_comp)
            token_num.append(num)
            compressed_ctc_batch.append(ctc_comp)

        padded_ctc_batch = pad_sequence(compressed_ctc_batch, batch_first=True).to(encoder_out.device)
        token_num = torch.tensor(token_num, dtype=torch.int, device=encoder_out.device)

        decoder_out, _ = self.decoder(encoder_out, encoder_out_mask, padded_ctc_batch, token_num)
        decoder_out = decoder_out.log_softmax(dim=-1)

        return {
            "encoder_out": encoder_out,
            "encoder_out_mask": encoder_out_mask,
            "decoder_out": decoder_out,
            "decoder_out_lens": token_num,
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
        res = self._forward_paraformer_v2(speech, speech_lengths, decoding_chunk_size, num_decoding_left_chunks)
        encoder_out, encoder_mask, decoder_out, decoder_out_lens = (
            res['encoder_out'], res['encoder_out_mask'], res['decoder_out'], res['decoder_out_lens'])

        results = {}
        if 'paraformer_greedy_search' in methods:
            assert decoder_out is not None
            assert decoder_out_lens is not None
            paraformer_greedy_result = paraformer_greedy_search(decoder_out, decoder_out_lens, None)
            results['paraformer_greedy_search'] = paraformer_greedy_result
        if 'paraformer_beam_search' in methods:
            assert decoder_out is not None
            assert decoder_out_lens is not None
            paraformer_beam_result = paraformer_beam_search(decoder_out, decoder_out_lens, beam_size=beam_size, eos=self.eos)
            results['paraformer_beam_search'] = paraformer_beam_result
        if 'ctc_greedy_search' in methods or 'ctc_prefix_beam_search' in methods:
            ctc_probs = self.ctc_logprobs(encoder_out, blank_penalty, blank_id)
            encoder_lens = encoder_mask.squeeze(1).sum(1)
            if 'ctc_greedy_search' in methods:
                results['ctc_greedy_search'] = ctc_greedy_search(ctc_probs, encoder_lens, blank_id)
            if 'ctc_prefix_beam_search' in methods:
                ctc_prefix_result = ctc_prefix_beam_search(ctc_probs, encoder_lens, beam_size, context_graph, blank_id)
                results['ctc_prefix_beam_search'] = ctc_prefix_result
        return results
