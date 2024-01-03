""" SummaryMixing © 2023 by Samsung Electronics is licensed under CC BY-NC 4.0.

This extend the head concept of SummaryMixing to the standard SpeechBrain lobes for VanillaNN.
Large parts of the code come from the SpeechBrain repository.

Usage: Install SpeechBrain and copy this file under speechbrain/lobes/models/
Source: https://arxiv.org/abs/2307.07421

Authors
 * Titouan Parcollet 2023
 * Shucong Zhang 2023
 * Rogier van Dalen 2023
 * Sourav Bhattacharya 2023
"""
import torch
import math
from torch import nn
import speechbrain as sb
from typing import Optional
from speechbrain.nnet.containers import Sequential


class ParallelLinear(torch.nn.Module):
    """Computes a parallel linear transformation y = wx + b.
    In practice the input and the output are split n_split times.
    Hence we create n_split parallel linear op that will operate on
    each splited dimension. E.g. if x = [B,T,F] and n_split = 4
    then x = [B,T,4,F/4] and W = [4,F/4,out_dim/4].

    Arguments
    ---------
    n_neurons : int
        It is the number of output neurons (i.e, the dimensionality of the
        output).
    input_shape: tuple, optional
        It is the shape of the input tensor.
    input_size: int, optional
        Size of the input tensor.
    n_split: int, optional
        The number of split to create n_split linear transformations.
    bias : bool, optional
        If True, the additive bias b is adopted.
    combiner_out_dims : bool, optional
        If True, the output vector is reshaped to be [B, T, S].

    Example
    -------
    >>> x = torch.rand([64, 50, 512])
    >>> lin_t = ParallelLinear(n_neurons=64, input_size=512, n_split=4)
    >>> output = lin_t(x)
    >>> output.shape
    torch.Size([64, 50, 64])
    """

    def __init__(
        self,
        n_neurons,
        input_shape: Optional[list] = None,
        input_size: Optional[int] = None,
        n_split: Optional[int] = 1,
        bias: Optional[bool] = True,
        combine_out_dims: Optional[bool] = True,
    ):
        super().__init__()
        self.n_split = n_split
        self.combine_out_dims = combine_out_dims

        if input_shape is None and input_size is None:
            raise ValueError("Expected one of input_shape or input_size")

        if input_size is None:
            input_size = input_shape[-1]
            if len(input_shape) == 4:
                input_size = input_shape[-1] * input_shape[-2]

        if input_size % n_split != 0 or n_neurons % n_split != 0:
            raise ValueError("input_size and n_neurons must be dividible by n_split!")

        self.split_inp_dim = input_size // n_split
        self.split_out_dim = n_neurons // n_split

        self.weights = nn.Parameter(
            torch.empty(self.n_split, self.split_inp_dim, self.split_out_dim)
        )
        self.biases = nn.Parameter(torch.zeros(self.n_split, self.split_out_dim))

        self._reset_parameters()

    def _reset_parameters(self):
        # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        # uniform(-1/sqrt(in_features), 1/sqrt(in_features)). For details, see
        # https://github.com/pytorch/pytorch/issues/57109
        nn.init.kaiming_uniform_(self.weights, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.biases, a=math.sqrt(5))

    def forward(self, x):
        """Returns the linear transformation of input tensor.

        Arguments
        ---------
        x : torch.Tensor
            Input to transform linearly, may be 3 or four dimensional.
            [B,T,F] or [B,T,n_split,F//n_split]
        """
        if x.ndim == 3:
            B, T, F = x.shape
            x = x.view(B, T, self.n_split, self.split_inp_dim)

        x = torch.einsum("btmf,mfh->btmh", x, self.weights) + self.biases

        if self.combine_out_dims:
            x = x.reshape(x.shape[0], x.shape[1], -1)

        return x


class VanillaNN(Sequential):
    """A simple vanilla Deep Neural Network.

    Arguments
    ---------
    activation : torch class
        A class used for constructing the activation layers.
    dnn_blocks : int
        The number of linear neural blocks to include.
    dnn_neurons : int or list[int]
        The number of neurons in the different linear layers.
        If a list is given, the length must correspond to the
        number of layers. If a int is given, all layers will
        have the same size.
    n_split: int
        The number of split to create n_split linear transformations.
        In practice the input and the output are split n_split times.
        Hence we create n_split parallel linear op that will operate on
        each splited dimension. E.g. if x = [B,T,F] and n_split = 4
        then x = [B,T,4,F/4] and W = [4,F/4,out_dim/4]. This will happen
        in each layer of the VanillaNN.

    Example
    -------
    >>> inputs = torch.rand([10, 120, 60])
    >>> model = VanillaNN(input_shape=inputs.shape)
    >>> outputs = model(inputs)
    >>> outputs.shape
    torch.Size([10, 120, 512])
    """

    def __init__(
        self,
        input_shape,
        activation: Optional[nn.Module] = torch.nn.LeakyReLU,
        dnn_blocks: Optional[int] = 2,
        dnn_neurons: Optional[int] = 512,
        n_split: Optional[int] = 1,
    ):
        super().__init__(input_shape=input_shape)

        if isinstance(dnn_neurons, list):
            if len(dnn_neurons) != dnn_blocks:
                msg = "The length of the dnn_neurons list must match dnn_blocks..."
                raise ValueError(msg)

        for block_index in range(dnn_blocks):
            if isinstance(dnn_neurons, list):
                current_nb_neurons = dnn_neurons[block_index]
            else:
                current_nb_neurons = dnn_neurons

            if n_split > 1:
                # ParrallelLinear does a costly reshape operation, hence we minimise this
                # cost by only doing this reshape for the last layer of the MLP.
                if block_index < (dnn_blocks - 1):
                    combine_out_dims = False
                else:
                    combine_out_dims = True
                self.append(
                    ParallelLinear,
                    n_neurons=current_nb_neurons,
                    bias=True,
                    n_split=n_split,
                    layer_name="linear",
                    combine_out_dims=combine_out_dims,
                )
            else:
                self.append(
                    sb.nnet.linear.Linear,
                    n_neurons=current_nb_neurons,
                    bias=True,
                    layer_name="linear",
                )
            self.append(activation(), layer_name="act")
