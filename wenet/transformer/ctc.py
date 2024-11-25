# Copyright (c) 2020 Mobvoi Inc. (authors: Binbin Zhang, Di Wu)
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
# Modified from ESPnet(https://github.com/espnet/espnet)

from typing import Tuple
import numpy as np
import torch
import torchaudio
import torch.nn.functional as F


class CTC(torch.nn.Module):
    """CTC module"""

    def __init__(
        self,
        odim: int,
        encoder_output_size: int,
        dropout_rate: float = 0.0,
        reduce: bool = True,
        blank_id: int = 0,
    ):
        """ Construct CTC module
        Args:
            odim: dimension of outputs
            encoder_output_size: number of encoder projection units
            dropout_rate: dropout rate (0.0 ~ 1.0)
            reduce: reduce the CTC loss into a scalar
            blank_id: blank label.
        """
        super().__init__()
        eprojs = encoder_output_size
        self.dropout_rate = dropout_rate
        self.ctc_lo = torch.nn.Linear(eprojs, odim)

        reduction_type = "sum" if reduce else "none"
        self.ctc_loss = torch.nn.CTCLoss(blank=blank_id,
                                         reduction=reduction_type,
                                         zero_infinity=True)

    def forward(self, hs_pad: torch.Tensor, hlens: torch.Tensor,
                ys_pad: torch.Tensor,
                ys_lens: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Calculate CTC loss.

        Args:
            hs_pad: batch of padded hidden state sequences (B, Tmax, D)
            hlens: batch of lengths of hidden state sequences (B)
            ys_pad: batch of padded character id sequence tensor (B, Lmax)
            ys_lens: batch of lengths of character sequence (B)
        """
        # hs_pad: (B, L, NProj) -> ys_hat: (B, L, Nvocab)
        ys_hat = self.ctc_lo(F.dropout(hs_pad, p=self.dropout_rate))
        # ys_hat: (B, L, D) -> (L, B, D)
        ys_hat = ys_hat.transpose(0, 1)
        ys_hat = ys_hat.log_softmax(2)
        loss = self.ctc_loss(ys_hat, ys_pad, hlens, ys_lens)
        # Batch-size average
        loss = loss / ys_hat.size(1)
        ys_hat = ys_hat.transpose(0, 1)
        return loss, ys_hat

    def log_softmax(self, hs_pad: torch.Tensor) -> torch.Tensor:
        """log_softmax of frame activations

        Args:
            Tensor hs_pad: 3d tensor (B, Tmax, eprojs)
        Returns:
            torch.Tensor: log softmax applied 3d tensor (B, Tmax, odim)
        """
        return F.log_softmax(self.ctc_lo(hs_pad), dim=2)

    def argmax(self, hs_pad: torch.Tensor) -> torch.Tensor:
        """argmax of frame activations

        Args:
            torch.Tensor hs_pad: 3d tensor (B, Tmax, eprojs)
        Returns:
            torch.Tensor: argmax applied 2d tensor (B, Tmax)
        """
        return torch.argmax(self.ctc_lo(hs_pad), dim=2)


    def ctc_logprobs(self, hs_pad, blank_id=0, blank_penalty: float = 0.0,):
        """log softmax of frame activations

        Args:
            torch.Tensor hs_pad: 3d tensor (B, Tmax, eprojs)
        Returns:
            torch.Tensor: log softmax applied 3d tensor (B, Tmax, odim)
        """
        if blank_penalty > 0.0:
            logits = self.ctc_lo(hs_pad)
            logits[:, :, blank_id] -= blank_penalty
            ctc_probs = logits.log_softmax(dim=2)
        else:
            ctc_probs = self.log_softmax(hs_pad)
        return ctc_probs

    def insert_blank(self, label, blank_id=0):
        """Insert blank token between every two label token."""
        label = np.expand_dims(label, 1)
        blanks = np.zeros((label.shape[0], 1), dtype=np.int64) + blank_id
        label = np.concatenate([blanks, label], axis=1)
        label = label.reshape(-1)
        label = np.append(label, label[0])
        return label

    def force_align(self, ctc_probs: torch.Tensor, y: torch.Tensor, blank_id=0) -> list:
        """ctc forced alignment.

        Args:
            torch.Tensor ctc_probs: hidden state sequence, 2d tensor (T, D)
            torch.Tensor y: id sequence tensor 1d tensor (L)
            int blank_id: blank symbol index
        Returns:
            torch.Tensor: alignment result
        """
        ctc_probs = ctc_probs[None].cpu()
        y = y[None].cpu()
        alignments, _ = torchaudio.functional.forced_align(ctc_probs, y, blank=blank_id)
        return alignments[0]

    def remove_duplicates_and_blank(self, alignment, blank_id=0):
        """
        去除对齐路径中的空白标签和重复标签。

        alignment: 对齐路径，可能包含空白标签和重复标签。
        blank_id: 空白标签的 ID。

        返回：
        filtered_alignment: 去除空白和重复标签后的对齐路径。
        """
        filtered_alignment = []
        prev_token = None
        for token in alignment:
            if token != blank_id and token != prev_token:
                filtered_alignment.append(token)
            prev_token = token
        return filtered_alignment

    def remove_duplicates(self, alignment):
        """
        去除对齐路径中的空白标签和重复标签。

        alignment: 对齐路径，可能包含空白标签和重复标签。
        blank_id: 空白标签的 ID。

        返回：
        filtered_alignment: 去除空白和重复标签后的对齐路径。
        """
        filtered_alignment = []
        prev_token = None
        for token in alignment:
            if  token != prev_token:
                filtered_alignment.append(token)
            prev_token = token
        return filtered_alignment

