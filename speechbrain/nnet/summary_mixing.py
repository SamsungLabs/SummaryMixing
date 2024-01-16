""" SummaryMixing © 2023 by Samsung Electronics is licensed under CC BY-NC 4.0.

This library provides the basic building blocks for SummaryMixing.

Usage: Install SpeechBrain and copy this file under speechbrain/nnet/
Source: https://arxiv.org/abs/2307.07421

Authors
 * Titouan Parcollet 2023
 * Shucong Zhang 2023
 * Rogier van Dalen 2023
 * Sourav Bhattacharya 2023
"""

import math
import torch
import logging
import torch.nn as nn
from typing import Optional
from speechbrain.lobes.models.VanillaNN import VanillaNN


logger = logging.getLogger(__name__)


class SummaryMixing(nn.Module):
    """ This class implements SummaryMixing as defined
    in https://arxiv.org/abs/2307.07421

    Arguments
    ---------
    enc_dim: int
        Feature dimension of the input tensor.
    nhead : int
        Number of mixing heads.
    local_proj_hid_dim: list [int], optional
        A list of dimension specifying both the number of hidden layers
        as well as the size of them in the local projection branch
        (default: [512]).
    local_proj_out_dim: int, optional
        The dimension of the output of the local projection branch. This
        will be concatenated with the output of the summary branch
        (default: 512).
    summary_hid_dim: list [int], optional
        A list of dimension specifying both the number of hidden layers
        as well as the size of them in the summary projection branch
        (default: [512]).
    summary_out_dim: int, optional
        The dimension of the output of the summary projection branch. This
        will be concatenated with the output of the local branch
        (default: 512).
    activation: torch.nn.Module, optional
        Torch module specifying the activation function used in both the local
        and summary branches.
        (default: torch.nn.GELU)
    mode: string, optional
        One of "SummaryMixing" or "SummaryMixing-lite". Changes the SummaryMixing cell
        according to the definition of the article. "SummaryMixing-lite" removes the
        local project branch.


    Example
    -------
    >>> x = torch.rand(2,4,8)
    >>> sum = SummaryMixing(8)
    >>> out = sum(x)
    >>> print(out)
    torch.Size([2, 4, 8])
    """

    def __init__(
        self,
        enc_dim,
        nhead,
        local_proj_hid_dim: Optional[list] = [512],
        local_proj_out_dim: Optional[int] = 512,
        summary_hid_dim: Optional[list] = [512],
        summary_out_dim: Optional[int] = 512,
        activation: Optional[nn.Module] = nn.GELU,
        mode: Optional[str] = "SummaryMixing",
    ):
        super(SummaryMixing, self).__init__()

        if mode not in ["SummaryMixing", "SummaryMixing-lite"]:
            raise ValueError(
                "The SummaryMixing mode should either be 'SummaryMixing' or 'SummaryMixing-lite'"
            )

        self.local_proj_hid_dim = local_proj_hid_dim
        self.local_proj_out_dim = local_proj_out_dim
        self.summary_hid_dim = summary_hid_dim
        self.summary_out_dim = summary_out_dim
        self.enc_dim = enc_dim
        self.activation = activation()
        self.local_dnn_blocks = local_proj_hid_dim + [local_proj_out_dim]
        self.summary_dnn_blocks = summary_hid_dim + [summary_out_dim]
        self.mode = mode

        if self.mode == "SummaryMixing":

            self.local_proj = VanillaNN(
                input_shape=[None, None, enc_dim],
                dnn_blocks=len(self.local_dnn_blocks),
                dnn_neurons=self.local_dnn_blocks,
                activation=activation,
                n_split=nhead,
            )

            self.summary_local_merging = VanillaNN(
                input_shape=[None, None, local_proj_out_dim + summary_out_dim],
                dnn_blocks=1,
                dnn_neurons=[summary_out_dim],
                activation=activation,
            )

            self.local_norm = nn.LayerNorm(local_proj_out_dim)
            self.summary_norm = nn.LayerNorm(summary_out_dim)

        self.summary_proj = VanillaNN(
            input_shape=[None, None, enc_dim],
            dnn_blocks=len(self.summary_dnn_blocks),
            dnn_neurons=self.summary_dnn_blocks,
            activation=activation,
            n_split=nhead,
        )

        self.apply(self._init_parameters)

    def forward(self, x, attention_mask=None):
        """ This function simply goes forward!

        Arguments
        ---------
        x: torch.Tensor
            The expected shape is the standard SpeechBrain one - [Batch, Time, Features]
        attention_mask: torch.Tensor
            (B, S) to pad before summarizing in time.
        """

        if attention_mask is not None:
            attention_mask = torch.logical_not(attention_mask).unsqueeze(-1).float()
        else:
            attention_mask = torch.ones((x.shape[0], x.shape[1])).unsqueeze(-1).float()

        if self.mode == "SummaryMixing":
            return self._forward_mixing(x, attention_mask)
        elif self.mode == "SummaryMixing-lite":
            return self._forward_avgonly(x, attention_mask)

    def _forward_mixing(self, x, attention_mask):
        """ Perform full SummaryMixing.

        Arguments
        ---------
        x: torch.Tensor
            The expected shape is the standard SpeechBrain one - [Batch, Time, Features]
        attention_mask: torch.Tensor
            (B, S) to pad before summarizing in time.
        """

        B, T, F = x.shape

        # f() (Eq. 1b)
        local_summary = self.local_norm(self.local_proj(x) * attention_mask)

        # s() (Eq. 2 and 1c)
        time_summary = self.summary_proj(x) * attention_mask

        # We normalise by real length by counting masking
        time_summary = self.summary_norm(
            torch.sum(time_summary, dim=1) / torch.sum(attention_mask, dim=1)
        )
        time_summary = time_summary.unsqueeze(1).repeat(1, T, 1)

        return self.summary_local_merging(
            torch.cat([local_summary, time_summary], dim=-1)
        )

    def _forward_avgonly(self, x, attention_mask):
        """ Perform SummaryMixing-lite.

        Arguments
        ---------
        x: torch.Tensor
            The expected shape is the standard SpeechBrain one - [Batch, Time, Features]
        attention_mask: torch.Tensor
            (B, S) to pad before summarizing in time.
        """

        B, T, F = x.shape

        # s() We just do the mean over time
        # Then we repeat the output matrix T times along the time axis
        time_summary = self.summary_proj(x) * attention_mask
        time_summary = torch.sum(time_summary, dim=1) / torch.sum(attention_mask, dim=1)
        time_summary = time_summary.unsqueeze(1).expand(-1, T, -1)

        return time_summary

    def _init_parameters(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.zeros_(module.bias)

    def _reset_parameters(self):
        # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        # uniform(-1/sqrt(in_features), 1/sqrt(in_features)). For details, see
        # https://github.com/pytorch/pytorch/issues/57109
        nn.init.kaiming_uniform_(self.A_weights, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.B_weights, a=math.sqrt(5))
