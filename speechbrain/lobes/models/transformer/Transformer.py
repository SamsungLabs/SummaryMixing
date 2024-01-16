""" SummaryMixing © 2023 by Samsung Electronics is licensed under CC BY-NC 4.0.

This library connects SummaryMixing to the standard SpeechBrain lobes for Transformer ASR.
Large parts of the code come from the SpeechBrain repository.

Usage: Install SpeechBrain and copy this file under speechbrain/lobes/models/transformer/
Source: https://arxiv.org/abs/2307.07421

Authors
 * Titouan Parcollet 2023
 * Shucong Zhang 2023
 * Rogier van Dalen 2023
 * Sourav Bhattacharya 2023
"""

import math
import torch
import torch.nn as nn
import speechbrain as sb
from typing import Optional
import numpy as np

from .Conformer import ConformerEncoder
from .Branchformer import BranchformerEncoder
from speechbrain.nnet.activations import Swish
from speechbrain.nnet.attention import RelPosEncXL
from speechbrain.nnet.summary_mixing import SummaryMixing


class TransformerInterface(nn.Module):
    """This is an interface for transformer model.

    Users can modify the attributes and define the forward function as
    needed according to their own tasks.

    The architecture is based on the paper "Attention Is All You Need":
    https://arxiv.org/pdf/1706.03762.pdf

    Arguments
    ----------
    d_model: int
        The number of expected features in the encoder/decoder inputs (default=512).
    nhead: int
        The number of heads in the multi-head attention models (default=8).
    num_encoder_layers: int, optional
        The number of encoder layers in1ì the encoder.
    num_decoder_layers: int, optional
        The number of decoder layers in the decoder.
    dim_ffn: int, optional
        The dimension of the feedforward network model hidden layer.
    dropout: int, optional
        The dropout value.
    activation: torch.nn.Module, optional
        The activation function for Feed-Forward Network layer,
        e.g., relu or gelu or swish.
    custom_src_module: torch.nn.Module, optional
        Module that processes the src features to expected feature dim.
    custom_tgt_module: torch.nn.Module, optional
        Module that processes the src features to expected feature dim.
    positional_encoding: str, optional
        Type of positional encoding used. e.g. 'fixed_abs_sine' for fixed absolute positional encodings.
    normalize_before: bool, optional
        Whether normalization should be applied before or after MHA or FFN in Transformer layers.
        Defaults to True as this was shown to lead to better performance and training stability.
    kernel_size: int, optional
        Kernel size in convolutional layers when Conformer is used.
    bias: bool, optional
        Whether to use bias in Conformer convolutional layers.
    encoder_module: str, optional
        Choose between Branchformer, Conformer and Transformer for the encoder. The decoder is fixed to be a Transformer.
    conformer_activation: torch.nn.Module, optional
        Activation module used after Conformer convolutional layers. E.g. Swish, ReLU etc. it has to be a torch Module.
    branchformer_activation: torch.nn.Module, optional
        Activation module used within the Branchformer Encoder. E.g. Swish, ReLU etc. it has to be a torch Module.
    attention_type: str, optional
        Type of attention layer used in all Transformer or Conformer layers.
        e.g. SummaryMixing, regularMHA or RelPosMHA.
    max_length: int, optional
        Max length for the target and source sequence in input.
        Used for positional encodings.
    causal: bool, optional
        Whether the encoder should be causal or not (the decoder is always causal).
        If causal the Conformer convolutional layer is causal.
    encoder_kdim: int, optional
        Dimension of the key for the encoder.
    encoder_vdim: int, optional
        Dimension of the value for the encoder.
    decoder_kdim: int, optional
        Dimension of the key for the decoder.
    decoder_vdim: int, optional
        Dimension of the value for the decoder.
    csgu_linear_units: int, optional
        Number of neurons in the hidden linear units of the CSGU Module.
        -> Branchformer
    gate_activation: torch.nn.Module, optional
        Activation function used at the gate of the CSGU module.
        -> Branchformer
    use_linear_after_conv: bool, optional
        If True, will apply a linear transformation of size input_size//2.
        -> Branchformer
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
        (default: [1024]).
    summary_out_dim: int, optional
        The dimension of the output of the summary projection branch. This
        will be concatenated with the output of the local branch
        (default: 1024).
    activation: torch.nn.Module, optional
        Torch module specifying the activation function used in both the local
        and summary branches.
        (default: torch.nn.GELU)
    mode: string, optional
        One of "SummaryMixing" or "SummaryMixing-lite". Changes the SummaryMixing cell
        according to the definition of the article. "SummaryMixing-lite" removes the
        local project branch.
    """

    def __init__(
        self,
        d_model=512,
        nhead=8,
        num_encoder_layers=6,
        num_decoder_layers=6,
        d_ffn=2048,
        dropout=0.1,
        activation=nn.ReLU,
        custom_src_module=None,
        custom_tgt_module=None,
        positional_encoding="fixed_abs_sine",
        normalize_before=True,
        kernel_size: Optional[int] = 31,
        bias: Optional[bool] = True,
        encoder_module: Optional[str] = "transformer",
        conformer_activation: Optional[nn.Module] = Swish,
        branchformer_activation: Optional[nn.Module] = nn.GELU,
        attention_type: Optional[str] = "SummaryMixing",
        max_length: Optional[int] = 2500,
        causal: Optional[bool] = False,
        encoder_kdim: Optional[int] = None,
        encoder_vdim: Optional[int] = None,
        decoder_kdim: Optional[int] = None,
        decoder_vdim: Optional[int] = None,
        csgu_linear_units: Optional[int] = 3072,
        gate_activation: Optional[nn.Module] = nn.Identity,
        use_linear_after_conv: Optional[bool] = False,
        local_proj_hid_dim: Optional[list] = [512],
        local_proj_out_dim: Optional[int] = 512,
        summary_hid_dim: Optional[list] = [1024],
        summary_out_dim: Optional[int] = 1024,
        mode: Optional[str] = "SummaryMixing",
    ):
        super().__init__()
        self.causal = causal
        self.attention_type = attention_type
        self.positional_encoding_type = positional_encoding
        self.encoder_kdim = encoder_kdim
        self.encoder_vdim = encoder_vdim
        self.decoder_kdim = decoder_kdim
        self.decoder_vdim = decoder_vdim

        assert attention_type in [
            "regularMHA",
            "RelPosMHAXL",
            "hypermixing",
            "SummaryMixing",
        ]
        assert positional_encoding in ["fixed_abs_sine", None]

        assert (
            num_encoder_layers + num_decoder_layers > 0
        ), "number of encoder layers and number of decoder layers cannot both be 0!"

        if positional_encoding == "fixed_abs_sine":
            self.positional_encoding = PositionalEncoding(d_model, max_length)
        elif positional_encoding is None:
            pass
            # no positional encodings

        # overrides any other pos_embedding
        if attention_type == "RelPosMHAXL":
            self.positional_encoding = RelPosEncXL(d_model)
            self.positional_encoding_decoder = PositionalEncoding(d_model, max_length)

        # initialize the encoder
        if num_encoder_layers > 0:
            if custom_src_module is not None:
                self.custom_src_module = custom_src_module(d_model)
            if encoder_module == "transformer":
                self.encoder = TransformerEncoder(
                    nhead=nhead,
                    num_layers=num_encoder_layers,
                    d_ffn=d_ffn,
                    d_model=d_model,
                    dropout=dropout,
                    activation=activation,
                    normalize_before=normalize_before,
                    causal=self.causal,
                    attention_type=self.attention_type,
                    kdim=self.encoder_kdim,
                    vdim=self.encoder_vdim,
                    local_proj_hid_dim=local_proj_hid_dim,
                    local_proj_out_dim=local_proj_out_dim,
                    summary_hid_dim=summary_hid_dim,
                    summary_out_dim=summary_out_dim,
                    mode=mode,
                )
            elif encoder_module == "conformer":
                self.encoder = ConformerEncoder(
                    nhead=nhead,
                    num_layers=num_encoder_layers,
                    d_ffn=d_ffn,
                    d_model=d_model,
                    dropout=dropout,
                    activation=conformer_activation,
                    kernel_size=kernel_size,
                    bias=bias,
                    causal=self.causal,
                    attention_type=self.attention_type,
                )
                assert normalize_before, "normalize_before must be True for Conformer"

                assert (
                    conformer_activation is not None
                ), "conformer_activation must not be None"
            elif encoder_module == "branchformer":
                self.encoder = BranchformerEncoder(
                    nhead=nhead,
                    num_layers=num_encoder_layers,
                    d_model=d_model,
                    dropout=dropout,
                    activation=branchformer_activation,
                    kernel_size=kernel_size,
                    attention_type=self.attention_type,
                    csgu_linear_units=csgu_linear_units,
                    gate_activation=gate_activation,
                    use_linear_after_conv=use_linear_after_conv,
                    local_proj_hid_dim=local_proj_hid_dim,
                    local_proj_out_dim=local_proj_out_dim,
                    summary_hid_dim=summary_hid_dim,
                    summary_out_dim=summary_out_dim,
                    mode=mode,
                )

        # initialize the decoder
        if num_decoder_layers > 0:
            if custom_tgt_module is not None:
                self.custom_tgt_module = custom_tgt_module(d_model)
            self.decoder = TransformerDecoder(
                num_layers=num_decoder_layers,
                nhead=nhead,
                d_ffn=d_ffn,
                d_model=d_model,
                dropout=dropout,
                activation=activation,
                normalize_before=normalize_before,
                causal=True,
                attention_type="regularMHA",  # always use regular attention in decoder
                kdim=self.decoder_kdim,
                vdim=self.decoder_vdim,
            )

    def forward(self, **kwags):
        """Users should modify this function according to their own tasks."""
        raise NotImplementedError


class PositionalEncoding(nn.Module):
    """This class implements the absolute sinusoidal positional encoding function.

    PE(pos, 2i)   = sin(pos/(10000^(2i/dmodel)))
    PE(pos, 2i+1) = cos(pos/(10000^(2i/dmodel)))

    Arguments
    ---------
    input_size: int
        Embedding dimension.
    max_len : int, optional
        Max length of the input sequences (default 2500).

    Example
    -------
    >>> a = torch.rand((8, 120, 512))
    >>> enc = PositionalEncoding(input_size=a.shape[-1])
    >>> b = enc(a)
    >>> b.shape
    torch.Size([1, 120, 512])
    """

    def __init__(self, input_size, max_len=2500):
        super().__init__()
        self.max_len = max_len
        pe = torch.zeros(self.max_len, input_size, requires_grad=False)
        positions = torch.arange(0, self.max_len).unsqueeze(1).float()
        denominator = torch.exp(
            torch.arange(0, input_size, 2).float() * -(math.log(10000.0) / input_size)
        )

        pe[:, 0::2] = torch.sin(positions * denominator)
        pe[:, 1::2] = torch.cos(positions * denominator)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        """
        Arguments
        ---------
        x : tensor
            Input feature shape (batch, time, fea)
        """
        return self.pe[:, : x.size(1)].clone().detach()


class TransformerEncoderLayer(nn.Module):
    """This is an implementation of self-attention encoder layer.

    Arguments
    ----------
    d_ffn: int
        The dimension of the feedforward network model hidden layer.
    nhead: int
        The number of heads in the multi-head attention models (default=8).
    d_model: int
        The number of expected features in the encoder/decoder inputs (default=512).
    kdim: int, optional
        Dimension of the key.
    vdim: int, optional
        Dimension of the value.
    dropout: int, optional
        The dropout value.
    activation: torch.nn.Module, optional
        The activation function for Feed-Forward Netowrk layer,
        e.g., relu or gelu or swish.
    normalize_before: bool, optional
        Whether normalization should be applied before or after MHA or FFN in Transformer layers.
        Defaults to True as this was shown to lead to better performance and training stability.
    attention_type: str, optional
        Type of attention layer used in all Transformer or Conformer layers.
        e.g. SummaryMixing, regularMHA or RelPosMHA.

    Example
    -------
    >>> import torch
    >>> x = torch.rand((8, 60, 512))
    >>> net = TransformerEncoderLayer(512, 8, d_model=512)
    >>> output = net(x)
    >>> output[0].shape
    torch.Size([8, 60, 512])
    """

    def __init__(
        self,
        d_ffn,
        nhead,
        d_model,
        kdim=None,
        vdim=None,
        dropout=0.0,
        activation=nn.ReLU,
        normalize_before=False,
        attention_type="SummaryMixing",
        causal=False,
        local_proj_hid_dim=[512],
        local_proj_out_dim=512,
        summary_hid_dim=[1024],
        summary_out_dim=1024,
        mode="SummaryMixing",
    ):
        super().__init__()

        self.attention_type = attention_type

        if attention_type == "regularMHA":
            self.self_att = sb.nnet.attention.MultiheadAttention(
                nhead=nhead, d_model=d_model, dropout=dropout, kdim=kdim, vdim=vdim,
            )

        elif attention_type == "RelPosMHAXL":
            self.self_att = sb.nnet.attention.RelPosMHAXL(
                d_model, nhead, dropout, mask_pos_future=causal
            )
        elif attention_type == "hypermixing":
            self.self_att = sb.nnet.hypermixing.HyperMixing(
                input_output_dim=d_model,
                hypernet_size=d_ffn,
                tied=False,
                num_heads=nhead,
                fix_tm_hidden_size=False,
            )
        elif attention_type == "SummaryMixing":
            self.self_att = SummaryMixing(
                enc_dim=d_model,
                local_proj_hid_dim=local_proj_hid_dim,
                local_proj_out_dim=local_proj_out_dim,
                summary_hid_dim=summary_hid_dim,
                summary_out_dim=summary_out_dim,
                activation=activation,
                mode=mode,
            )

        self.pos_ffn = sb.nnet.attention.PositionalwiseFeedForward(
            d_ffn=d_ffn, input_size=d_model, dropout=dropout, activation=activation,
        )

        self.norm1 = sb.nnet.normalization.LayerNorm(d_model, eps=1e-6)
        self.norm2 = sb.nnet.normalization.LayerNorm(d_model, eps=1e-6)
        self.dropout1 = torch.nn.Dropout(dropout)
        self.dropout2 = torch.nn.Dropout(dropout)

        self.normalize_before = normalize_before

    def forward(
        self,
        src,
        src_mask: Optional[torch.Tensor] = None,
        src_key_padding_mask: Optional[torch.Tensor] = None,
        pos_embs: Optional[torch.Tensor] = None,
    ):
        """
        Arguments
        ----------
        src : torch.Tensor
            The sequence to the encoder layer.
        src_mask : torch.Tensor
            The mask for the src query for each example in the batch.
        src_key_padding_mask : torch.Tensor, optional
            The mask for the src keys for each example in the batch.
        """

        if self.normalize_before:
            src1 = self.norm1(src)
        else:
            src1 = src

        if self.attention_type == "SummaryMixing":
            output = self.self_att(src1, attention_mask=src_key_padding_mask)
            self_attn = None
        else:
            output, self_attn = self.self_att(
                src1,
                src1,
                src1,
                attn_mask=src_mask,
                key_padding_mask=src_key_padding_mask,
                pos_embs=pos_embs,
            )

        # add & norm
        src = src + self.dropout1(output)
        if not self.normalize_before:
            src = self.norm1(src)

        if self.normalize_before:
            src1 = self.norm2(src)
        else:
            src1 = src
        output = self.pos_ffn(src1)

        # add & norm
        output = src + self.dropout2(output)
        if not self.normalize_before:
            output = self.norm2(output)
        return output, self_attn


class TransformerEncoder(nn.Module):
    """This class implements the transformer encoder.

    Arguments
    ---------
    num_layers : int
        Number of transformer layers to include.
    nhead : int
        Number of attention heads.
    d_ffn : int
        Hidden size of self-attention Feed Forward layer.
    d_model : int
        The dimension of the input embedding.
    kdim : int
        Dimension for key (Optional).
    vdim : int
        Dimension for value (Optional).
    dropout : float
        Dropout for the encoder (Optional).
    input_module: torch class
        The module to process the source input feature to expected
        feature dimension (Optional).

    Example
    -------
    >>> import torch
    >>> x = torch.rand((8, 60, 512))
    >>> net = TransformerEncoder(1, 8, 512, d_model=512)
    >>> output, _ = net(x)
    >>> output.shape
    torch.Size([8, 60, 512])
    """

    def __init__(
        self,
        num_layers,
        nhead,
        d_ffn,
        input_shape=None,
        d_model=None,
        kdim=None,
        vdim=None,
        dropout=0.0,
        activation=nn.ReLU,
        normalize_before=False,
        causal=False,
        layerdrop_prob=0.0,
        attention_type="SummaryMixing",
        local_proj_hid_dim=[512],
        local_proj_out_dim=512,
        summary_hid_dim=[1024],
        summary_out_dim=1024,
        mode="SummaryMixing",
    ):
        super().__init__()

        self.layers = torch.nn.ModuleList(
            [
                TransformerEncoderLayer(
                    d_ffn=d_ffn,
                    nhead=nhead,
                    d_model=d_model,
                    kdim=kdim,
                    vdim=vdim,
                    dropout=dropout,
                    activation=activation,
                    normalize_before=normalize_before,
                    causal=causal,
                    attention_type=attention_type,
                    local_proj_hid_dim=local_proj_hid_dim,
                    local_proj_out_dim=local_proj_out_dim,
                    summary_hid_dim=summary_hid_dim,
                    summary_out_dim=summary_out_dim,
                    mode=mode,
                )
                for i in range(num_layers)
            ]
        )
        self.norm = sb.nnet.normalization.LayerNorm(d_model, eps=1e-6)
        self.layerdrop_prob = layerdrop_prob
        self.rng = np.random.default_rng()

    def forward(
        self,
        src,
        src_mask: Optional[torch.Tensor] = None,
        src_key_padding_mask: Optional[torch.Tensor] = None,
        pos_embs: Optional[torch.Tensor] = None,
    ):
        """
        Arguments
        ----------
        src : tensor
            The sequence to the encoder layer (required).
        src_mask : tensor
            The mask for the src sequence (optional).
        src_key_padding_mask : tensor
            The mask for the src keys per batch (optional).
        """
        output = src
        if self.layerdrop_prob > 0.0:
            keep_probs = self.rng.random(len(self.layers))
        else:
            keep_probs = None
        attention_lst = []
        for i, enc_layer in enumerate(self.layers):
            if (
                not self.training
                or self.layerdrop_prob == 0.0
                or keep_probs[i] > self.layerdrop_prob
            ):
                output, attention = enc_layer(
                    output,
                    src_mask=src_mask,
                    src_key_padding_mask=src_key_padding_mask,
                    pos_embs=pos_embs,
                )

                attention_lst.append(attention)
        output = self.norm(output)
        return output, attention_lst


class TransformerDecoderLayer(nn.Module):
    """This class implements the self-attention decoder layer.

    Arguments
    ----------
    d_ffn : int
        Hidden size of self-attention Feed Forward layer.
    nhead : int
        Number of attention heads.
    d_model : int
        Dimension of the model.
    kdim : int
        Dimension for key (optional).
    vdim : int
        Dimension for value (optional).
    dropout : float
        Dropout for the decoder (optional).

    Example
    -------
    >>> src = torch.rand((8, 60, 512))
    >>> tgt = torch.rand((8, 60, 512))
    >>> net = TransformerDecoderLayer(1024, 8, d_model=512)
    >>> output, self_attn, multihead_attn = net(src, tgt)
    >>> output.shape
    torch.Size([8, 60, 512])
    """

    def __init__(
        self,
        d_ffn,
        nhead,
        d_model,
        kdim=None,
        vdim=None,
        dropout=0.0,
        activation=nn.ReLU,
        normalize_before=False,
        attention_type="regularMHA",
        causal=None,
    ):
        super().__init__()
        self.nhead = nhead

        if attention_type == "regularMHA":
            self.self_attn = sb.nnet.attention.MultiheadAttention(
                nhead=nhead, d_model=d_model, kdim=kdim, vdim=vdim, dropout=dropout,
            )
            self.mutihead_attn = sb.nnet.attention.MultiheadAttention(
                nhead=nhead, d_model=d_model, kdim=kdim, vdim=vdim, dropout=dropout,
            )

        elif attention_type == "RelPosMHAXL":
            self.self_attn = sb.nnet.attention.RelPosMHAXL(
                d_model, nhead, dropout, mask_pos_future=causal
            )
            self.mutihead_attn = sb.nnet.attention.RelPosMHAXL(
                d_model, nhead, dropout, mask_pos_future=causal
            )

        self.pos_ffn = sb.nnet.attention.PositionalwiseFeedForward(
            d_ffn=d_ffn, input_size=d_model, dropout=dropout, activation=activation,
        )

        # normalization layers
        self.norm1 = sb.nnet.normalization.LayerNorm(d_model, eps=1e-6)
        self.norm2 = sb.nnet.normalization.LayerNorm(d_model, eps=1e-6)
        self.norm3 = sb.nnet.normalization.LayerNorm(d_model, eps=1e-6)
        self.dropout1 = torch.nn.Dropout(dropout)
        self.dropout2 = torch.nn.Dropout(dropout)
        self.dropout3 = torch.nn.Dropout(dropout)

        self.normalize_before = normalize_before

    def forward(
        self,
        tgt,
        memory,
        tgt_mask=None,
        memory_mask=None,
        tgt_key_padding_mask=None,
        memory_key_padding_mask=None,
        pos_embs_tgt=None,
        pos_embs_src=None,
    ):
        """
        Arguments
        ----------
        tgt: tensor
            The sequence to the decoder layer (required).
        memory: tensor
            The sequence from the last layer of the encoder (required).
        tgt_mask: tensor
            The mask for the tgt sequence (optional).
        memory_mask: tensor
            The mask for the memory sequence (optional).
        tgt_key_padding_mask: tensor
            The mask for the tgt keys per batch (optional).
        memory_key_padding_mask: tensor
            The mask for the memory keys per batch (optional).
        """
        if self.normalize_before:
            tgt1 = self.norm1(tgt)
        else:
            tgt1 = tgt

        # self-attention over the target sequence
        tgt2, self_attn = self.self_attn(
            query=tgt1,
            key=tgt1,
            value=tgt1,
            attn_mask=tgt_mask,
            key_padding_mask=tgt_key_padding_mask,
            pos_embs=pos_embs_tgt,
        )

        # add & norm
        tgt = tgt + self.dropout1(tgt2)
        if not self.normalize_before:
            tgt = self.norm1(tgt)

        if self.normalize_before:
            tgt1 = self.norm2(tgt)
        else:
            tgt1 = tgt

        # multi-head attention over the target sequence and encoder states

        tgt2, multihead_attention = self.mutihead_attn(
            query=tgt1,
            key=memory,
            value=memory,
            attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask,
            pos_embs=pos_embs_src,
        )

        # add & norm
        tgt = tgt + self.dropout2(tgt2)
        if not self.normalize_before:
            tgt = self.norm2(tgt)

        if self.normalize_before:
            tgt1 = self.norm3(tgt)
        else:
            tgt1 = tgt

        tgt2 = self.pos_ffn(tgt1)

        # add & norm
        tgt = tgt + self.dropout3(tgt2)
        if not self.normalize_before:
            tgt = self.norm3(tgt)

        return tgt, self_attn, multihead_attention


class TransformerDecoder(nn.Module):
    """This class implements the Transformer decoder.

    Arguments
    ----------
    nhead : int
        Number of attention heads.
    d_ffn : int
        Hidden size of self-attention Feed Forward layer.
    d_model : int
        Dimension of the model.
    kdim : int, optional
        Dimension for key (Optional).
    vdim : int, optional
        Dimension for value (Optional).
    dropout : float, optional
        Dropout for the decoder (Optional).

    Example
    -------
    >>> src = torch.rand((8, 60, 512))
    >>> tgt = torch.rand((8, 60, 512))
    >>> net = TransformerDecoder(1, 8, 1024, d_model=512)
    >>> output, _, _ = net(src, tgt)
    >>> output.shape
    torch.Size([8, 60, 512])
    """

    def __init__(
        self,
        num_layers,
        nhead,
        d_ffn,
        d_model,
        kdim=None,
        vdim=None,
        dropout=0.0,
        activation=nn.ReLU,
        normalize_before=False,
        causal=False,
        attention_type="regularMHA",
    ):
        super().__init__()
        self.layers = torch.nn.ModuleList(
            [
                TransformerDecoderLayer(
                    d_ffn=d_ffn,
                    nhead=nhead,
                    d_model=d_model,
                    kdim=kdim,
                    vdim=vdim,
                    dropout=dropout,
                    activation=activation,
                    normalize_before=normalize_before,
                    causal=causal,
                    attention_type=attention_type,
                )
                for _ in range(num_layers)
            ]
        )
        self.norm = sb.nnet.normalization.LayerNorm(d_model, eps=1e-6)

    def forward(
        self,
        tgt,
        memory,
        tgt_mask=None,
        memory_mask=None,
        tgt_key_padding_mask=None,
        memory_key_padding_mask=None,
        pos_embs_tgt=None,
        pos_embs_src=None,
    ):
        """
        Arguments
        ----------
        tgt : tensor
            The sequence to the decoder layer (required).
        memory : tensor
            The sequence from the last layer of the encoder (required).
        tgt_mask : tensor
            The mask for the tgt sequence (optional).
        memory_mask : tensor
            The mask for the memory sequence (optional).
        tgt_key_padding_mask : tensor
            The mask for the tgt keys per batch (optional).
        memory_key_padding_mask : tensor
            The mask for the memory keys per batch (optional).
        """
        output = tgt
        self_attns, multihead_attns = [], []
        for dec_layer in self.layers:
            output, self_attn, multihead_attn = dec_layer(
                output,
                memory,
                tgt_mask=tgt_mask,
                memory_mask=memory_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask,
                pos_embs_tgt=pos_embs_tgt,
                pos_embs_src=pos_embs_src,
            )
            self_attns.append(self_attn)
            multihead_attns.append(multihead_attn)
        output = self.norm(output)

        return output, self_attns, multihead_attns


class NormalizedEmbedding(nn.Module):
    """This class implements the normalized embedding layer for the transformer.

    Since the dot product of the self-attention is always normalized by sqrt(d_model)
    and the final linear projection for prediction shares weight with the embedding layer,
    we multiply the output of the embedding by sqrt(d_model).

    Arguments
    ---------
    d_model: int
        The number of expected features in the encoder/decoder inputs (default=512).
    vocab: int
        The vocab size.

    Example
    -------
    >>> emb = NormalizedEmbedding(512, 1000)
    >>> trg = torch.randint(0, 999, (8, 50))
    >>> emb_fea = emb(trg)
    """

    def __init__(self, d_model, vocab):
        super().__init__()
        self.emb = sb.nnet.embedding.Embedding(
            num_embeddings=vocab, embedding_dim=d_model, blank_id=0
        )
        self.d_model = d_model

    def forward(self, x):
        """ Processes the input tensor x and returns an output tensor."""
        return self.emb(x) * math.sqrt(self.d_model)


def get_key_padding_mask(padded_input, pad_idx):
    """Creates a binary mask to prevent attention to padded locations.

    Arguments
    ----------
    padded_input: int
        Padded input.
    pad_idx:
        idx for padding element.

    Example
    -------
    >>> a = torch.LongTensor([[1,1,0], [2,3,0], [4,5,0]])
    >>> get_key_padding_mask(a, pad_idx=0)
    tensor([[False, False,  True],
            [False, False,  True],
            [False, False,  True]])
    """
    if len(padded_input.shape) == 4:
        bz, time, ch1, ch2 = padded_input.shape
        padded_input = padded_input.reshape(bz, time, ch1 * ch2)

    key_padded_mask = padded_input.eq(pad_idx).to(padded_input.device)

    # if the input is more than 2d, mask the locations where they are silence
    # across all channels
    if len(padded_input.shape) > 2:
        key_padded_mask = key_padded_mask.float().prod(dim=-1).bool()
        return key_padded_mask.detach()

    return key_padded_mask.detach()


def get_lookahead_mask(padded_input):
    """Creates a binary mask for each sequence which maskes future frames.

    Arguments
    ---------
    padded_input: torch.Tensor
        Padded input tensor.

    Example
    -------
    >>> a = torch.LongTensor([[1,1,0], [2,3,0], [4,5,0]])
    >>> get_lookahead_mask(a)
    tensor([[0., -inf, -inf],
            [0., 0., -inf],
            [0., 0., 0.]])
    """
    seq_len = padded_input.shape[1]
    mask = (
        torch.triu(torch.ones((seq_len, seq_len), device=padded_input.device)) == 1
    ).transpose(0, 1)
    mask = (
        mask.float()
        .masked_fill(mask == 0, float("-inf"))
        .masked_fill(mask == 1, float(0.0))
    )
    return mask.detach().to(padded_input.device)
