
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence

from wenet.transformer.ctc import CTC
from wenet.transformer.decoder import TransformerDecoder
from wenet.transformer.encoder import BaseEncoder
from wenet.transformer.label_smoothing_loss import LabelSmoothingLoss
from wenet.transformer.search import (ctc_greedy_search,
                                      ctc_prefix_beam_search,
                                      attention_beam_search,
                                      attention_rescoring, DecodeResult)
from wenet.utils.mask import make_pad_mask
from wenet.utils.common import (IGNORE_ID, add_sos_eos, th_accuracy,
                                reverse_pad_list)
from wenet.utils.context_graph import ContextGraph
from wenet.transformer.cmvn import GlobalCMVN
from wenet.utils.cmvn import load_cmvn
from wenet.utils.class_module import WENET_ENCODER_CLASSES, WENET_DECODER_CLASSES, WENET_CTC_CLASSES


class TextSpeechASR(torch.nn.Module):

    def __init__(
            self,
            input_dim: int,
            vocab_size: int,
            pny_vocab_size: int,
            audio_encoder: BaseEncoder,
            audio_encoder_conf: dict,
            pny2han_encoder: BaseEncoder,
            pny2han_encoder_conf: dict,
            context_encoder: BaseEncoder,
            context_encoder_conf: dict,
            align_decoder: TransformerDecoder,
            align_decoder_conf: dict,
            pny2han_decoder: TransformerDecoder,
            pny2han_decoder_conf: dict,
            context_decoder: TransformerDecoder,
            context_decoder_conf: dict,
            ctc: CTC,
            ctc_conf: dict,
            cmvn: Optional[str] = None,
            cmvn_conf: Optional[dict] = None,
            **kwargs,
    ):

        super().__init__()

        special_tokens = kwargs['tokenizer_conf'].get('special_tokens', None)

        # TODO(xcsong): Forcefully read the 'cmvn' attribute.
        if cmvn == 'global_cmvn':
            mean, istd = load_cmvn(cmvn_conf['cmvn_file'], cmvn_conf['is_json_cmvn'])
            global_cmvn = GlobalCMVN(torch.from_numpy(mean).float(), torch.from_numpy(istd).float())
        else:
            global_cmvn = None

        audio_encoder = WENET_ENCODER_CLASSES[audio_encoder](
            input_dim,
            global_cmvn=global_cmvn,
            **audio_encoder_conf,
            **audio_encoder_conf['efficient_conf'] if 'efficient_conf' in audio_encoder_conf else {})

        pny2han_encoder = WENET_ENCODER_CLASSES[pny2han_encoder](
            pny_vocab_size,
            global_cmvn=None,
            **pny2han_encoder_conf,
            **pny2han_encoder_conf['efficient_conf'] if 'efficient_conf' in pny2han_encoder_conf else {})

        context_encoder = WENET_ENCODER_CLASSES[context_encoder](
            audio_encoder.output_size() + pny2han_encoder.output_size(),
            global_cmvn=None,
            **context_encoder_conf,
            **context_encoder_conf['efficient_conf'] if 'efficient_conf' in context_encoder_conf else {})

        align_decoder = WENET_DECODER_CLASSES[align_decoder](pny_vocab_size, audio_encoder.output_size(), **align_decoder_conf)
        pny2han_decoder = WENET_DECODER_CLASSES[pny2han_decoder](vocab_size, audio_encoder.output_size(), **pny2han_decoder_conf)
        context_decoder = WENET_DECODER_CLASSES[context_decoder](vocab_size, audio_encoder.output_size(), **context_decoder_conf)

        context_ctc = WENET_CTC_CLASSES[ctc](vocab_size, context_encoder.output_size(), blank_id=ctc_conf['ctc_blank_id'])
        pny2han_ctc = WENET_CTC_CLASSES[ctc](vocab_size, pny2han_encoder.output_size(), blank_id=ctc_conf['ctc_blank_id'])
        audio_ctc = WENET_CTC_CLASSES[ctc](pny_vocab_size, audio_encoder.output_size(), blank_id=ctc_conf['ctc_blank_id'])

        self.ctc_weight = kwargs.get('ctc_conf', {}).get('ctc_weight', 0.5)
        self.ignore_id = kwargs['model_conf'].get('ignore_id', IGNORE_ID)
        self.blank_id = kwargs.get('ctc_conf', {}).get('ctc_blank_id', 0)
        self.reverse_weight = kwargs['model_conf'].get('reverse_weight', 0.0)
        self.special_tokens = special_tokens
        self.apply_non_blank_embedding = kwargs['model_conf'].get('apply_non_blank_embedding', False)
        self.lsm_weight = kwargs['model_conf'].get('lsm_weight', 0.0)
        self.length_normalized_loss = kwargs['model_conf'].get('length_normalized_loss', False)
        self.sampling_ratio = kwargs['model_conf'].get('sampling_ratio', 0.75)

        # note that eos is the same as sos (equivalent ID)
        special_tokens = self.special_tokens
        self.sos = (vocab_size - 1 if special_tokens is None else  special_tokens.get("<sos>", vocab_size - 1))
        self.eos = (vocab_size - 1 if special_tokens is None else  special_tokens.get("<eos>", vocab_size - 1))
        self.vocab_size = vocab_size
        self.pny_vocab_size = pny_vocab_size

        self.audio_encoder = audio_encoder
        self.pny2han_encoder = pny2han_encoder
        self.context_encoder = context_encoder

        self.align_decoder = align_decoder
        self.pny2han_decoder = pny2han_decoder
        self.context_decoder = context_decoder

        self.context_ctc = context_ctc
        self.pny2han_ctc = pny2han_ctc
        self.audio_ctc = audio_ctc

        self.pny_criterion_att = LabelSmoothingLoss(
            size=pny_vocab_size,
            padding_idx=self.ignore_id,
            smoothing=self.lsm_weight,
            normalize_length=self.length_normalized_loss,
            )

        self.han_criterion_att = LabelSmoothingLoss(
            size=vocab_size,
            padding_idx=self.ignore_id,
            smoothing=self.lsm_weight,
            normalize_length=self.length_normalized_loss,
            )




    @torch.jit.unused
    def forward(
        self,
        batch: dict,
        device: torch.device,
    ) -> Dict[str, Optional[torch.Tensor]]:
        """Frontend + Encoder + Decoder + Calc loss"""
        speech = batch['feats'].to(device)
        speech_lengths = batch['feats_lengths'].to(device)
        text = batch['target'].to(device)
        text_lengths = batch['target_lengths'].to(device)
        pny = batch['pny_target'].to(device)
        pny_lengths = batch['pny_target_lengths'].to(device)

        assert text_lengths.dim() == 1, text_lengths.shape
        # Check that batch_size is unified
        assert (speech.shape[0] == speech_lengths.shape[0] == text.shape[0] ==text_lengths.shape[0]), \
            (speech.shape, speech_lengths.shape, text.shape, text_lengths.shape)

        # 1. Audio Encoder
        audio_encoder_out, audio_encoder_mask = self.audio_encoder(speech, speech_lengths)
        audio_encoder_out_lens = audio_encoder_mask.squeeze(1).sum(1)
        loss_audio_ctc, _ = self.audio_ctc(audio_encoder_out, audio_encoder_out_lens, pny, pny_lengths)

        # 2. Align the audio to pny
        align_decoder_input, pny2han_encoder_input = self._align_audio2pny(audio_encoder_out, audio_encoder_out_lens, pny, pny_lengths)
        loss_align_att, acc_align_att = self._calc_align_att_loss(audio_encoder_out, audio_encoder_mask, align_decoder_input, pny, pny_lengths)

        # pny2han Encoder + Decoder
        pny2han_encoder_out, pny2han_encoder_mask, pny2han_encoder_out_dict = self.pny2han_encoder(
            pny2han_encoder_input, audio_encoder_out_lens, return_layers_output=True)
        loss_pny2han_ctc, _ = self.pny2han_ctc(pny2han_encoder_out, audio_encoder_out_lens, text, text_lengths)
        loss_pny2han_att, acc_pny2han_att, pny2han_decoder_out_dict = self._calc_pny2han_att_loss(
            pny2han_encoder_out, pny2han_encoder_mask, text,  text_lengths)

        # 3. Context Encoder + Decoder
        context_encoder_out, context_encoder_mask = self.context_encoder(
            audio_encoder_out, audio_encoder_out_lens, pny2han_encoder_out_dict)
        loss_context_ctc, _ = self.context_ctc(context_encoder_out, audio_encoder_out_lens, text, text_lengths)
        loss_context_att, acc_context_att = self._calc_context_att_loss(
            context_encoder_out, context_encoder_mask, pny2han_decoder_out_dict, text, text_lengths)

        loss_ctc =  loss_audio_ctc + loss_pny2han_ctc + loss_context_ctc
        loss_att = loss_align_att + loss_pny2han_att + loss_context_att

        loss = self.ctc_weight * loss_ctc + (1 - self.ctc_weight) * loss_att
        acc_att = acc_context_att


        return {
            "loss": loss,
            "loss_att": loss_att,
            "loss_ctc": loss_ctc,
            "th_accuracy": acc_att,
            "loss_align_att": loss_align_att,"loss_pny2han_att": loss_pny2han_att, "loss_context_att": loss_context_att,
            "loss_audio_ctc": loss_audio_ctc, "loss_pny2han_ctc": loss_pny2han_ctc, "loss_context_ctc": loss_context_ctc,
            "acc_align_att": acc_align_att, "acc_pny2han_att": acc_pny2han_att, "acc_context_att": acc_context_att
        }

    def _align_audio2pny(self, audio_encoder_output, audio_encoder_output_lens, pny, pny_lens):
        batch_size = audio_encoder_output.size(0)
        device = audio_encoder_output.device
        with torch.no_grad():
            compressed_ctc_batch = []
            sample_ctc_batch = []
            ctc_probs = self.audio_ctc.log_softmax(audio_encoder_output).detach()
            pred_tokens = ctc_probs.argmax(-1)
            input_mask = torch.ones_like(pred_tokens, device=device )

            for b in range(batch_size):
                audio_encoder_output_len = audio_encoder_output_lens[b]
                pny_len = pny_lens[b]
                ctc_prob = ctc_probs[b][: audio_encoder_output_len]  # [T, N]
                text_b = pny[b][: pny_len] # [1, U]
                text_audio_alignment = self.audio_ctc.force_align(ctc_prob, text_b).to(device)
                exist_non_blank_mask = ~(text_audio_alignment==0).to(device)
                pred_token = pred_tokens[b][: audio_encoder_output_len] * exist_non_blank_mask
                same_num = ((pred_token == text_audio_alignment) * exist_non_blank_mask).sum(0)
                target_num = (exist_non_blank_mask.sum() - same_num).float() * self.sampling_ratio
                target_num = target_num.long()
                if target_num > 0:
                    non_blank_indices = torch.nonzero(exist_non_blank_mask).squeeze()
                    if len(non_blank_indices.size()) == 0:
                        non_blank_indices = non_blank_indices.unsqueeze(0)
                        random_indices = non_blank_indices
                    else:
                        random_indices = non_blank_indices[torch.randperm(len(non_blank_indices))[:target_num]]
                    pred_token[random_indices] = text_audio_alignment[random_indices]
                # 把相同的不为0的帧的概率平均
                ctc_comp = self._average_repeats(ctc_prob, text_audio_alignment)
                if ctc_comp.size(0) != pny_len:
                    print(f"ctc_comp error: {ctc_comp.size(0)}, {text_b}")
                compressed_ctc_batch.append(ctc_comp)
                sample_ctc_batch.append(pred_token)

            padded_ctc_prob = pad_sequence(compressed_ctc_batch, batch_first=True).to(audio_encoder_output.device)
            padded_sample_ctc = pad_sequence(sample_ctc_batch, batch_first=True, padding_value=0).to(audio_encoder_output.device)

        return padded_ctc_prob, padded_sample_ctc



    def _average_repeats(self, ctc_prob, alignment):
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
        responded_id = self.audio_ctc.remove_duplicates(alignment)
        for i in range(len(responded_id)):
            id  = responded_id[i]
            if id == self.blank_id:
                continue
            else:
                non_blank_ctc_prob.append(unique_probs[i])

        compressed_ctc_prob = torch.stack(non_blank_ctc_prob, dim=0)
        return compressed_ctc_prob


    def _calc_align_att_loss(
        self,
        audio_encoder_out: torch.Tensor,
        audio_encoder_mask: torch.Tensor,
        align_decoder_input: torch.Tensor,
        pny: torch.Tensor,
        pny_lens: torch.Tensor,
        infos: Dict[str, List[str]] = None,
    ):
        align_decoder_out, _ = self.align_decoder(audio_encoder_out, audio_encoder_mask, align_decoder_input, pny_lens)
        loss_align_att = self.pny_criterion_att(align_decoder_out, pny)
        acc_align_att = th_accuracy(align_decoder_out.view(-1, self.pny_vocab_size), pny, ignore_label=self.ignore_id)
        return loss_align_att, acc_align_att


    def _calc_pny2han_att_loss(
        self,
        pny2han_encoder_out: torch.Tensor,
        pny2han_encoder_mask: torch.Tensor,
        ys_pad: torch.Tensor,
        ys_pad_lens: torch.Tensor,
        infos: Dict[str, List[str]] = None,
    ):
        ys_in_pad, ys_out_pad = add_sos_eos(ys_pad, self.sos, self.eos, self.ignore_id)
        ys_in_lens = ys_pad_lens + 1

        # reverse the seq, used for right to left context_decoder
        r_ys_pad = reverse_pad_list(ys_pad, ys_pad_lens, float(self.ignore_id))
        r_ys_in_pad, r_ys_out_pad = add_sos_eos(r_ys_pad, self.sos, self.eos, self.ignore_id)
        # 1. Forward context_decoder
        decoder_out, r_decoder_out, _,  decoder_out_dict = self.pny2han_decoder(
            pny2han_encoder_out, pny2han_encoder_mask, ys_in_pad, ys_in_lens, r_ys_in_pad, self.reverse_weight, return_layers_output=True)
        # 2. Compute attention loss
        loss_att = self.han_criterion_att(decoder_out, ys_out_pad)
        r_loss_att = torch.tensor(0.0)
        if self.reverse_weight > 0.0:
            r_loss_att = self.criterion_att(r_decoder_out, r_ys_out_pad)
        loss_pny2han_att = loss_att * (1 - self.reverse_weight) + r_loss_att * self.reverse_weight
        acc_pny2han_att = th_accuracy(decoder_out.view(-1, self.vocab_size), ys_out_pad, ignore_label=self.ignore_id, )
        return loss_pny2han_att, acc_pny2han_att, decoder_out_dict



    def tie_or_clone_weights(self, jit_mode: bool = True):
        self.context_decoder.tie_or_clone_weights(jit_mode)

    @torch.jit.unused
    def _forward_ctc(
            self,
            encoder_out: torch.Tensor,
            encoder_mask: torch.Tensor,
            text: torch.Tensor,
            text_lengths: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        encoder_out_lens = encoder_mask.squeeze(1).sum(1)
        loss_ctc, ctc_probs = self.ctc(encoder_out, encoder_out_lens, text, text_lengths)
        return loss_ctc, ctc_probs

    def filter_blank_embedding(
            self, ctc_probs: torch.Tensor,
            encoder_out: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = encoder_out.size(0)
        maxlen = encoder_out.size(1)
        top1_index = torch.argmax(ctc_probs, dim=2)
        indices = []
        for j in range(batch_size):
            indices.append(torch.tensor( [i for i in range(maxlen) if top1_index[j][i] != 0]))

        select_encoder_out = [
            torch.index_select(encoder_out[i, :, :], 0, indices[i].to(encoder_out.device))
            for i in range(batch_size) ]
        select_encoder_out = pad_sequence(select_encoder_out, batch_first=True, padding_value=0).to(encoder_out.device)
        xs_lens = torch.tensor([len(indices[i]) for i in range(batch_size) ]).to(encoder_out.device)
        T = select_encoder_out.size(1)
        encoder_mask = ~make_pad_mask(xs_lens, T).unsqueeze(1)  # (B, 1, T)
        encoder_out = select_encoder_out
        return encoder_out, encoder_mask

    def _calc_context_att_loss(
        self,
        encoder_out: torch.Tensor,
        encoder_mask: torch.Tensor,
        pny2han_decoder_out_dict: dict,
        ys_pad: torch.Tensor,
        ys_pad_lens: torch.Tensor,
        infos: Dict[str, List[str]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        ys_in_pad, ys_out_pad = add_sos_eos(ys_pad, self.sos, self.eos, self.ignore_id)
        ys_in_lens = ys_pad_lens + 1

        # reverse the seq, used for right to left context_decoder
        r_ys_pad = reverse_pad_list(ys_pad, ys_pad_lens, float(self.ignore_id))
        r_ys_in_pad, r_ys_out_pad = add_sos_eos(r_ys_pad, self.sos, self.eos, self.ignore_id)
        # 1. Forward context_decoder
        decoder_out, r_decoder_out, _ = self.context_decoder(
            encoder_out, encoder_mask, ys_in_pad, ys_in_lens, r_ys_in_pad, pny2han_decoder_out_dict, self.reverse_weight)
        # 2. Compute attention loss
        loss_att = self.han_criterion_att(decoder_out, ys_out_pad)
        r_loss_att = torch.tensor(0.0)
        if self.reverse_weight > 0.0:
            r_loss_att = self.han_criterion_att(r_decoder_out, r_ys_out_pad)
        loss_att = loss_att * (1 - self.reverse_weight) + r_loss_att * self.reverse_weight
        acc_att = th_accuracy(decoder_out.view(-1, self.vocab_size), ys_out_pad, ignore_label=self.ignore_id,)
        return loss_att, acc_att

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
            encoder_out, encoder_mask = self.audio_encoder.forward_chunk_by_chunk(
                speech,
                decoding_chunk_size=decoding_chunk_size,
                num_decoding_left_chunks=num_decoding_left_chunks
            )  # (B, maxlen, encoder_dim)
        else:
            encoder_out, encoder_mask = self.audio_encoder(
                speech,
                speech_lengths,
                decoding_chunk_size=decoding_chunk_size,
                num_decoding_left_chunks=num_decoding_left_chunks
            )  # (B, maxlen, encoder_dim)
        return encoder_out, encoder_mask

    @torch.jit.unused
    def ctc_logprobs(self,
                     encoder_out: torch.Tensor,
                     blank_penalty: float = 0.0,
                     blank_id: int = 0):
        if blank_penalty > 0.0:
            logits = self.ctc.ctc_lo(encoder_out)
            logits[:, :, blank_id] -= blank_penalty
            ctc_probs = logits.log_softmax(dim=2)
        else:
            ctc_probs = self.ctc.log_softmax(encoder_out)

        return ctc_probs

    def decode(
        self,
        methods: List[str],
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        beam_size: int,
        decoding_chunk_size: int = -1,
        num_decoding_left_chunks: int = -1,
        ctc_weight: float = 0.0,
        simulate_streaming: bool = False,
        reverse_weight: float = 0.0,
        context_graph: ContextGraph = None,
        blank_id: int = 0,
        blank_penalty: float = 0.0,
        length_penalty: float = 0.0,
        infos: Dict[str, List[str]] = None,
    ) -> Dict[str, List[DecodeResult]]:
        """ Decode input speech

        Args:
            methods:(List[str]): list of decoding methods to use, which could
                could contain the following decoding methods, please refer paper:
                https://arxiv.org/pdf/2102.01547.pdf
                   * ctc_greedy_search
                   * ctc_prefix_beam_search
                   * atttention
                   * attention_rescoring
            speech (torch.Tensor): (batch, max_len, feat_dim)
            speech_length (torch.Tensor): (batch, )
            beam_size (int): beam size for beam search
            decoding_chunk_size (int): decoding chunk for dynamic chunk
                trained model.
                <0: for decoding, use full chunk.
                >0: for decoding, use fixed chunk size as set.
                0: used for training, it's prohibited here
            simulate_streaming (bool): whether do audio_encoder forward in a
                streaming fashion
            reverse_weight (float): right to left context_decoder weight
            ctc_weight (float): ctc score weight

        Returns: dict results of all decoding methods
        """
        assert speech.shape[0] == speech_lengths.shape[0]
        assert decoding_chunk_size != 0
        encoder_out, encoder_mask = self._forward_encoder(
            speech, speech_lengths, decoding_chunk_size,
            num_decoding_left_chunks, simulate_streaming)
        encoder_lens = encoder_mask.squeeze(1).sum(1)
        ctc_probs = self.ctc_logprobs(encoder_out, blank_penalty, blank_id)
        results = {}
        if 'attention' in methods:
            results['attention'] = attention_beam_search(
                self, encoder_out, encoder_mask, beam_size, length_penalty,
                infos)
        if 'ctc_greedy_search' in methods:
            results['ctc_greedy_search'] = ctc_greedy_search(
                ctc_probs, encoder_lens, blank_id)
        if 'ctc_prefix_beam_search' in methods:
            ctc_prefix_result = ctc_prefix_beam_search(ctc_probs, encoder_lens,
                                                       beam_size,
                                                       context_graph, blank_id)
            results['ctc_prefix_beam_search'] = ctc_prefix_result
        if 'attention_rescoring' in methods:
            # attention_rescoring depends on ctc_prefix_beam_search nbest
            if 'ctc_prefix_beam_search' in results:
                ctc_prefix_result = results['ctc_prefix_beam_search']
            else:
                ctc_prefix_result = ctc_prefix_beam_search(
                    ctc_probs, encoder_lens, beam_size, context_graph,
                    blank_id)
            if self.apply_non_blank_embedding:
                encoder_out, _ = self.filter_blank_embedding(
                    ctc_probs, encoder_out)
            results['attention_rescoring'] = attention_rescoring(
                self, ctc_prefix_result, encoder_out, encoder_lens, ctc_weight,
                reverse_weight, infos)
        return results

    @torch.jit.export
    def subsampling_rate(self) -> int:
        """ Export interface for c++ call, return subsampling_rate of the
            model
        """
        return self.audio_encoder.embed.subsampling_rate

    @torch.jit.export
    def right_context(self) -> int:
        """ Export interface for c++ call, return right_context of the model
        """
        return self.audio_encoder.embed.right_context

    @torch.jit.export
    def sos_symbol(self) -> int:
        """ Export interface for c++ call, return sos symbol id of the model
        """
        return self.sos

    @torch.jit.export
    def eos_symbol(self) -> int:
        """ Export interface for c++ call, return eos symbol id of the model
        """
        return self.eos

    @torch.jit.export
    def forward_encoder_chunk(
        self,
        xs: torch.Tensor,
        offset: int,
        required_cache_size: int,
        att_cache: torch.Tensor = torch.zeros(0, 0, 0, 0),
        cnn_cache: torch.Tensor = torch.zeros(0, 0, 0, 0),
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """ Export interface for c++ call, give input chunk xs, and return
            output from time 0 to current chunk.

        Args:
            xs (torch.Tensor): chunk input, with shape (b=1, time, mel-dim),
                where `time == (chunk_size - 1) * subsample_rate + \
                        subsample.right_context + 1`
            offset (int): current offset in audio_encoder output time stamp
            required_cache_size (int): cache size required for next chunk
                compuation
                >=0: actual cache size
                <0: means all history cache is required
            att_cache (torch.Tensor): cache tensor for KEY & VALUE in
                transformer/conformer attention, with shape
                (elayers, head, cache_t1, d_k * 2), where
                `head * d_k == hidden-dim` and
                `cache_t1 == chunk_size * num_decoding_left_chunks`.
            cnn_cache (torch.Tensor): cache tensor for cnn_module in conformer,
                (elayers, b=1, hidden-dim, cache_t2), where
                `cache_t2 == cnn.lorder - 1`

        Returns:
            torch.Tensor: output of current input xs,
                with shape (b=1, chunk_size, hidden-dim).
            torch.Tensor: new attention cache required for next chunk, with
                dynamic shape (elayers, head, ?, d_k * 2)
                depending on required_cache_size.
            torch.Tensor: new conformer cnn cache required for next chunk, with
                same shape as the original cnn_cache.

        """
        return self.audio_encoder.forward_chunk(xs, offset, required_cache_size,
                                                att_cache, cnn_cache)

    @torch.jit.export
    def ctc_activation(self, xs: torch.Tensor) -> torch.Tensor:
        """ Export interface for c++ call, apply linear transform and log
            softmax before ctc
        Args:
            xs (torch.Tensor): audio_encoder output

        Returns:
            torch.Tensor: activation before ctc

        """
        return self.ctc.log_softmax(xs)

    @torch.jit.export
    def is_bidirectional_decoder(self) -> bool:
        """
        Returns:
            torch.Tensor: context_decoder output
        """
        if hasattr(self.context_decoder, 'right_decoder'):
            return True
        else:
            return False

    @torch.jit.export
    def forward_attention_decoder(
        self,
        hyps: torch.Tensor,
        hyps_lens: torch.Tensor,
        encoder_out: torch.Tensor,
        reverse_weight: float = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """ Export interface for c++ call, forward context_decoder with multiple
            hypothesis from ctc prefix beam search and one audio_encoder output
        Args:
            hyps (torch.Tensor): hyps from ctc prefix beam search, already
                pad sos at the begining
            hyps_lens (torch.Tensor): length of each hyp in hyps
            encoder_out (torch.Tensor): corresponding audio_encoder output
            r_hyps (torch.Tensor): hyps from ctc prefix beam search, already
                pad eos at the begining which is used fo right to left context_decoder
            reverse_weight: used for verfing whether used right to left context_decoder,
            > 0 will use.

        Returns:
            torch.Tensor: context_decoder output
        """
        assert encoder_out.size(0) == 1
        num_hyps = hyps.size(0)
        assert hyps_lens.size(0) == num_hyps
        encoder_out = encoder_out.repeat(num_hyps, 1, 1)
        encoder_mask = torch.ones(num_hyps,
                                  1,
                                  encoder_out.size(1),
                                  dtype=torch.bool,
                                  device=encoder_out.device)

        # input for right to left context_decoder
        # this hyps_lens has count <sos> token, we need minus it.
        r_hyps_lens = hyps_lens - 1
        # this hyps has included <sos> token, so it should be
        # convert the original hyps.
        r_hyps = hyps[:, 1:]
        #   >>> r_hyps
        #   >>> tensor([[ 1,  2,  3],
        #   >>>         [ 9,  8,  4],
        #   >>>         [ 2, -1, -1]])
        #   >>> r_hyps_lens
        #   >>> tensor([3, 3, 1])

        # NOTE(Mddct): `pad_sequence` is not supported by ONNX, it is used
        #   in `reverse_pad_list` thus we have to refine the below code.
        #   Issue: https://github.com/wenet-e2e/wenet/issues/1113
        # Equal to:
        #   >>> r_hyps = reverse_pad_list(r_hyps, r_hyps_lens, float(self.ignore_id))
        #   >>> r_hyps, _ = add_sos_eos(r_hyps, self.sos, self.eos, self.ignore_id)
        max_len = torch.max(r_hyps_lens)
        index_range = torch.arange(0, max_len, 1).to(encoder_out.device)
        seq_len_expand = r_hyps_lens.unsqueeze(1)
        seq_mask = seq_len_expand > index_range  # (beam, max_len)
        #   >>> seq_mask
        #   >>> tensor([[ True,  True,  True],
        #   >>>         [ True,  True,  True],
        #   >>>         [ True, False, False]])
        index = (seq_len_expand - 1) - index_range  # (beam, max_len)
        #   >>> index
        #   >>> tensor([[ 2,  1,  0],
        #   >>>         [ 2,  1,  0],
        #   >>>         [ 0, -1, -2]])
        index = index * seq_mask
        #   >>> index
        #   >>> tensor([[2, 1, 0],
        #   >>>         [2, 1, 0],
        #   >>>         [0, 0, 0]])
        r_hyps = torch.gather(r_hyps, 1, index)
        #   >>> r_hyps
        #   >>> tensor([[3, 2, 1],
        #   >>>         [4, 8, 9],
        #   >>>         [2, 2, 2]])
        r_hyps = torch.where(seq_mask, r_hyps, self.eos)
        #   >>> r_hyps
        #   >>> tensor([[3, 2, 1],
        #   >>>         [4, 8, 9],
        #   >>>         [2, eos, eos]])
        r_hyps = torch.cat([hyps[:, 0:1], r_hyps], dim=1)
        #   >>> r_hyps
        #   >>> tensor([[sos, 3, 2, 1],
        #   >>>         [sos, 4, 8, 9],
        #   >>>         [sos, 2, eos, eos]])

        decoder_out, r_decoder_out, _ = self.context_decoder(
            encoder_out, encoder_mask, hyps, hyps_lens, r_hyps,
            reverse_weight)  # (num_hyps, max_hyps_len, vocab_size)
        decoder_out = torch.nn.functional.log_softmax(decoder_out, dim=-1)

        # right to left context_decoder may be not used during decoding process,
        # which depends on reverse_weight param.
        # r_dccoder_out will be 0.0, if reverse_weight is 0.0
        r_decoder_out = torch.nn.functional.log_softmax(r_decoder_out, dim=-1)
        return decoder_out, r_decoder_out
