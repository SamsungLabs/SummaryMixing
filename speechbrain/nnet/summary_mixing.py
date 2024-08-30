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

import logging
import math
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

from speechbrain.lobes.models.VanillaNN import VanillaNN

logger = logging.getLogger(__name__)


class SummaryMixing(nn.Module):
    """This class implements SummaryMixing as defined
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
    global_dropout: float, optional
        Amount of dropout applied when concatenating  the local and summary.
    mode: string, optional
        One of "SummaryMixing", "SummaryMixing-lite" or "SummaryMixing-fast". Changes the SummaryMixing cell
        according to the definition of the article. "SummaryMixing-lite" removes the
        local project branch. "SummaryMixing-expdecay" is another alternative using
        an exponential decay for the window, it's slower.
    use_layernorm: bool, optional
        Using layernorm for the local and the global branch in SummaryMixing or not.


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
        global_dropout: Optional[float] = 0.1,
        mode: Optional[str] = "SummaryMixing",
        use_layernorm: Optional[bool] = True,
    ):
        super(SummaryMixing, self).__init__()

        if mode not in [
            "SummaryMixing",
            "SummaryMixing-lite",
            "SummaryMixing-expdecay",
            "SummaryMixing-fast",
        ]:
            raise ValueError(
                "The SummaryMixing mode should either be 'SummaryMixing', 'SummaryMixing-lite', 'SummaryMixing-fast' or 'SummaryMixing-expdecay'"
            )

        self.local_proj_hid_dim = local_proj_hid_dim
        self.local_proj_out_dim = local_proj_out_dim
        self.summary_hid_dim = summary_hid_dim
        self.summary_out_dim = summary_out_dim
        self.summary_reshaped_dim = int(np.sqrt(summary_out_dim))
        self.enc_dim = enc_dim
        self.activation = activation()
        self.local_dnn_blocks = local_proj_hid_dim + [local_proj_out_dim]
        self.summary_dnn_blocks = summary_hid_dim + [summary_out_dim]
        self.mode = mode
        self.use_layernorm = use_layernorm
        self.dropout = nn.Dropout(global_dropout)

        if self.mode == "SummaryMixing" or self.mode == "SummaryMixing-expdecay":

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

        if self.mode == "SummaryMixing-fast":
            self.global_proj = VanillaNN(
                input_shape=[None, None, enc_dim],
                dnn_blocks=1,
                dnn_neurons=self.local_proj_out_dim * 2,
                activation=activation,
                n_split=1,
            )

            self.summary_local_merging = VanillaNN(
                input_shape=[None, None, self.local_proj_out_dim * 2],
                dnn_blocks=1,
                dnn_neurons=[summary_out_dim],
                activation=activation,
            )

        else:
            self.summary_proj = VanillaNN(
                input_shape=[None, None, enc_dim],
                dnn_blocks=len(self.summary_dnn_blocks),
                dnn_neurons=self.summary_dnn_blocks,
                activation=activation,
                n_split=nhead,
            )

        if self.mode == "SummaryMixing-expdecay":
            self.decay_constant = nn.Parameter(
                data=torch.tensor(0.995), requires_grad=False
            )

        if self.use_layernorm:
            self.local_norm = nn.LayerNorm(local_proj_out_dim)
            self.summary_norm = nn.LayerNorm(summary_out_dim)

        self.apply(self._init_parameters)

    def forward(self, x, sum_mask=None, src_padding_mask=None):
        """This function simply goes forward!

        Arguments
        ---------
        x: torch.Tensor
            The expected shape is the standard SpeechBrain one - [Batch, Time, Features]
        sum_mask: torch.Tensor
            (Time, Time) per time step mask that can be used to compute different sum between time-step.
            this can be useful for streaming, for instance, where each time step has a limited context.
        src_padding_mask: torch.Tensor
            (Batch, Time) corresponding to padding. We avoid padding when summarizing in time.
        """

        if src_padding_mask is not None:
            src_padding_mask = src_padding_mask.unsqueeze(-1)
        else:
            src_padding_mask = torch.ones((x.shape[0], x.shape[1])).unsqueeze(-1)

        if sum_mask is not None:
            sum_mask = sum_mask.float()

        if self.mode == "SummaryMixing" or self.mode == "SummaryMixing-expdecay":
            return self._forward_mixing(x, sum_mask, src_padding_mask)
        elif self.mode == "SummaryMixing-fast":
            return self._forward_mixing_fast(x, sum_mask, src_padding_mask)
        elif self.mode == "SummaryMixing-lite":
            return self._forward_avgonly(x, sum_mask, src_padding_mask)

    def _forward_mixing(self, x, sum_mask, src_padding_mask):
        """Perform full SummaryMixing.

        Arguments
        ---------
        x: torch.Tensor
            The expected shape is the standard SpeechBrain one - [Batch, Time, Features]
        sum_mask: torch.Tensor
            (Time, Time) per time step mask that can be used to compute different sum between time-step.
            this can be useful for streaming, for instance, where each time step has a limited context.
        src_padding_mask: torch.Tensor
            (Batch, Time) corresponding to padding. We avoid padding when summarizing in time.
        """

        B, T, F = x.shape

        # f() (Eq. 1b)
        local_summary = self.local_proj(x) * src_padding_mask

        if self.use_layernorm:
            local_summary = self.local_norm(local_summary)

        # s() (Eq. 2 and 1c)
        time_summary = self.summary_proj(x) * src_padding_mask

        if self.mode == "SummaryMixing-expdecay":
            sum_mask = self._laplace_weights(T, self.decay_constant, sum_mask, x.device)

        if sum_mask is None:

            # We normalise by real length by counting masking
            time_summary = torch.sum(time_summary, dim=1) / torch.sum(
                src_padding_mask, dim=1
            )

            time_summary = time_summary.unsqueeze(1).repeat(1, T, 1)

        else:

            # We must do a masked sum. The mask is [Time, Time] and the features are [B,T,F]
            # We therefore can do a matmul between [B,F,T] and [Time,Time].T to obtain [B,F,T] that we can re-transpose.
            # We need to be careful when dividing as padding is not included in sum_mask. We need to build the intersection
            # of both mask to know the actual real number of elements excluding padding.

            # full_mask_with_pad = torch.matmul(sum_mask, src_padding_mask)

            time_summary = torch.matmul(sum_mask, time_summary) / torch.sum(
                sum_mask, dim=1
            ).unsqueeze(-1)

        if self.use_layernorm:
            time_summary = self.summary_norm(time_summary)

        return self.summary_local_merging(
            self.dropout(torch.cat([local_summary, time_summary], dim=-1))
        )

    def _forward_mixing_fast(self, x, sum_mask, src_padding_mask):
        """Perform full SummaryMixing.

        Arguments
        ---------
        x: torch.Tensor
            The expected shape is the standard SpeechBrain one - [Batch, Time, Features]
        sum_mask: torch.Tensor
            (Time, Time) per time step mask that can be used to compute different sum between time-step.
            this can be useful for streaming, for instance, where each time step has a limited context.
        src_padding_mask: torch.Tensor
            (Batch, Time) corresponding to padding. We avoid padding when summarizing in time.
        """

        B, T, F = x.shape

        global_proj = self.global_proj(x) * src_padding_mask
        split_global_proj = torch.split(global_proj, self.local_proj_out_dim, dim=-1)

        # split_global_proj[0] = local projection
        # split_global_proj[1] = summary projection
        if sum_mask is None:
            # We normalise by real length by counting masking
            time_summary = torch.sum(split_global_proj[1], dim=1) / torch.sum(
                src_padding_mask, dim=1
            )
            time_summary = time_summary.unsqueeze(1).repeat(1, T, 1)

        else:

            # We must do a masked sum. The mask is [Time, Time] and the features are [B,T,F]
            # We therefore can do a matmul between [B,F,T] and [Time,Time].T to obtain [B,F,T] that we can re-transpose.
            # We need to be careful when dividing as padding is not included in sum_mask. We need to build the intersection
            # of both mask to know the actual real number of elements excluding padding.

            # full_mask_with_pad = torch.matmul(sum_mask, src_padding_mask)

            time_summary = torch.matmul(sum_mask, split_global_proj[1]) / torch.sum(
                sum_mask, dim=1
            ).unsqueeze(-1)

        return self.summary_local_merging(
            self.dropout(torch.cat([split_global_proj[0], time_summary], dim=-1))
        )

    def _forward_avgonly(self, x, sum_mask, src_padding_mask):
        """Perform SummaryMixing-lite.

        Arguments
        ---------
        x: torch.Tensor
            The expected shape is the standard SpeechBrain one - [Batch, Time, Features]
        sum_mask: torch.Tensor
            (Time, Time) per time step mask that can be used to compute different sum between time-step.
            this can be useful for streaming, for instance, where each time step has a limited context.
        src_padding_mask: torch.Tensor
            (Batch, Time) corresponding to padding. We avoid padding when summarizing in time.
        """

        B, T, F = x.shape

        # s() We just do the mean over time
        # Then we repeat the output matrix T times along the time axis
        time_summary = self.summary_proj(x) * src_padding_mask
        time_summary = torch.sum(time_summary, dim=1) / torch.sum(
            src_padding_mask, dim=1
        )
        time_summary = time_summary.unsqueeze(1).expand(-1, T, -1)

        return time_summary

    def _init_parameters(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.zeros_(module.bias)

    def _laplace_weights(
        self,
        size: int,
        decay_constant,
        binary_mask: Optional[torch.Tensor] = None,
        device="cpu",
        normalise=False,
    ):
        """
        Return a square matrix with the diagonal entries the maximum one in each row
        and the entries left and right decaying exponentially.
        This is like a discrete Laplacian distribution.
        If normalise is set to True, in each row, the entries add up to 1.

        Arguments
        ---------
        size: int
            The height and width of the returned matrix.
        decay_constant: float
            The exponential decay per position.
            This must be a positive value, and will normally be less than 1.
        binary_mask: torch.Tensor
            A binary mask applied before the rows are normalised.
        device: str
            Torch device to copy the generated masks to.
        """

        # Fill a matrix with integers indicating how far away each element is from
        # the diagonal.
        horizontal_distance_to_diagonal = torch.abs(
            torch.arange(size) - torch.arange(size).unsqueeze(-1)
        ).to(device)

        # A Laplacian-like shape with the correct decay, but where the diagonal
        # elements are all 1.
        absolute_laplacian = torch.exp(
            horizontal_distance_to_diagonal * torch.log(decay_constant)
        )

        if binary_mask is not None:
            absolute_laplacian = absolute_laplacian * binary_mask

        if normalise:
            # Normalise each row.
            normalised = absolute_laplacian / torch.sum(
                absolute_laplacian, dim=1, keepdim=True
            )
            return normalised

        return absolute_laplacian

    def _reset_parameters(self):
        # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        # uniform(-1/sqrt(in_features), 1/sqrt(in_features)). For details, see
        # https://github.com/pytorch/pytorch/issues/57109
        nn.init.kaiming_uniform_(self.A_weights, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.B_weights, a=math.sqrt(5))
