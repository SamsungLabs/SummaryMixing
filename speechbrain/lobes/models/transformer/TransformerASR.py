""" SummaryMixing © 2023 by Samsung Electronics is licensed under CC BY-NC 4.0.

This library connects SummaryMixing to the standard SpeechBrain lobes for Transformer-based ASR.
Large parts of the code come from the SpeechBrain repository.

Usage: Install SpeechBrain and copy this file under speechbrain/lobes/models/transformer/
Source: https://arxiv.org/abs/2307.07421

Authors
 * Titouan Parcollet 2023
 * Shucong Zhang 2023
 * Rogier van Dalen 2023
 * Sourav Bhattacharya 2023
"""
from dataclasses import dataclass
import torch  # noqa 42
from torch import nn
from typing import Any, Optional
from speechbrain.nnet.linear import Linear
from speechbrain.nnet.containers import ModuleList
from speechbrain.lobes.models.transformer.Transformer import (
    TransformerInterface,
    get_lookahead_mask,
    get_key_padding_mask,
    NormalizedEmbedding,
)
from speechbrain.nnet.activations import Swish
from speechbrain.dataio.dataio import length_to_mask
from speechbrain.utils.dynamic_chunk_training import DynChunkTrainConfig


@dataclass
class TransformerASRStreamingContext:
    """Streaming metadata and state for a `TransformerASR` instance."""

    dynchunktrain_config: DynChunkTrainConfig
    """Dynamic Chunk Training configuration holding chunk size and context size
    information."""

    encoder_context: Any
    """Opaque encoder context information. It is constructed by the encoder's
    `make_streaming_context` method and is passed to the encoder when using
    `encode_streaming`.
    """


def make_transformer_src_mask(
    src: torch.Tensor,
    causal: bool = False,
    dynchunktrain_config: Optional[DynChunkTrainConfig] = None,
) -> Optional[torch.Tensor]:
    """Prepare the source transformer mask that restricts which frames can
    attend to which frames depending on causal or other simple restricted
    attention methods.

    Arguments
    ---------
    src: torch.Tensor
        The source tensor to build a mask from. The contents of the tensor are
        not actually used currently; only its shape and other metadata (e.g.
        device).

    causal: bool
        Whether strict causality shall be used. Frames will not be able to
        attend to any future frame.

    dynchunktrain_config: DynChunkTrainConfig, optional
        Dynamic Chunk Training configuration. This implements a simple form of
        chunkwise attention. Incompatible with `causal`."""

    if causal:
        assert dynchunktrain_config is None
        return get_lookahead_mask(src)

    if dynchunktrain_config is not None:
        # init a mask that masks nothing by default
        # 0 == no mask, 1 == mask
        src_mask = torch.zeros(
            (src.shape[1], src.shape[1]), device=src.device, dtype=torch.bool,
        )

        # The following is not really the sole source used to implement this,
        # but it helps introduce the concept.
        # ref: Unified Streaming and Non-streaming Two-pass End-to-end Model
        # for Speech Recognition
        # https://arxiv.org/pdf/2012.05481.pdf

        timesteps = src.size(1)

        # mask the future at the right of each chunk
        for t in range(timesteps):
            # if we have a chunk size of 8 then:
            # for 0..7  -> mask 8..
            # for 8..15 -> mask 16..
            # etc.
            next_chunk_index = (t // dynchunktrain_config.chunk_size) + 1
            visible_range = next_chunk_index * dynchunktrain_config.chunk_size
            src_mask[t, visible_range:] = True

        # mask the past at the left of each chunk (accounting for left context)
        # only relevant if using left context
        if not dynchunktrain_config.is_infinite_left_context():
            for t in range(timesteps):
                chunk_index = t // dynchunktrain_config.chunk_size
                chunk_first_t = chunk_index * dynchunktrain_config.chunk_size

                left_context_frames = (
                    dynchunktrain_config.left_context_size
                    * dynchunktrain_config.chunk_size
                )

                frame_remaining_context = max(0, chunk_first_t - left_context_frames,)

                # end range is exclusive, so there is no off-by-one here
                src_mask[t, :frame_remaining_context] = True

        return src_mask

    return None


def make_transformer_src_tgt_masks(
    src,
    tgt=None,
    wav_len=None,
    pad_idx=0,
    causal: bool = False,
    dynchunktrain_config: Optional[DynChunkTrainConfig] = None,
):
    """This function generates masks for training the transformer model,
    opiniated for an ASR context with encoding masks and, optionally, decoding
    masks (if specifying `tgt`).

    Arguments
    ---------
    src : tensor
        The sequence to the encoder (required).
    tgt : tensor
        The sequence to the decoder.
    pad_idx : int
        The index for <pad> token (default=0).
    causal: bool
        Whether strict causality shall be used. See `make_asr_src_mask`
    dynchunktrain_config: DynChunkTrainConfig, optional
        Dynamic Chunk Training configuration. See `make_asr_src_mask`
    """
    src_key_padding_mask = None

    # mask out audio beyond the length of audio for each batch
    if wav_len is not None:
        abs_len = torch.round(wav_len * src.shape[1])
        src_key_padding_mask = ~length_to_mask(abs_len).bool()

    # mask out the source
    src_mask = make_transformer_src_mask(
        src, causal=causal, dynchunktrain_config=dynchunktrain_config
    )

    # If no decoder in the transformer...
    if tgt is not None:
        tgt_key_padding_mask = get_key_padding_mask(tgt, pad_idx=pad_idx)
        tgt_mask = get_lookahead_mask(tgt)
    else:
        tgt_key_padding_mask = None
        tgt_mask = None

    return src_key_padding_mask, tgt_key_padding_mask, src_mask, tgt_mask


class TransformerASR(TransformerInterface):
    """This is an implementation of transformer model for ASR.

    The architecture is based on the paper "Attention Is All You Need":
    https://arxiv.org/pdf/1706.03762.pdf

    Arguments
    ----------
    tgt_vocab: int
        Size of vocabulary.
    input_size: int
        Input feature size.
    d_model : int, optional
        Embedding dimension size.
        (default=512).
    nhead : int, optional
        The number of heads in the multi-head attention models (default=8).
    num_encoder_layers : int, optional
        The number of sub-encoder-layers in the encoder (default=6).
    num_decoder_layers : int, optional
        The number of sub-decoder-layers in the decoder (default=6).
    dim_ffn : int, optional
        The dimension of the feedforward network model (default=2048).
    dropout : int, optional
        The dropout value (default=0.1).
    activation : torch.nn.Module, optional
        The activation function of FFN layers.
        Recommended: relu or gelu (default=relu).
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
        Choose between Conformer and Transformer for the encoder. The decoder is fixed to be a Transformer.
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
    csgu_linear_units: int, optional
        Number of neurons in the hidden linear units of the CSGU Module.
        -> Branchformer
    gate_activation: torch.nn.Module, optional
        Activation function used at the gate of the CSGU module.
        -> Branchformer
    use_linear_after_conv: bool, optional
        If True, will apply a linear transformation of size input_size//2.
        -> Branchformer
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

    Example
    -------
    >>> src = torch.rand([8, 120, 512])
    >>> tgt = torch.randint(0, 720, [8, 120])
    >>> net = TransformerASR(
    ...     720, 512, 512, 1, 1, 1, 1024, activation=torch.nn.GELU
    ... )
    >>> enc_out, dec_out = net.forward(src, tgt)
    >>> enc_out.shape
    torch.Size([8, 120, 512])
    >>> dec_out.shape
    torch.Size([8, 120, 512])
    """

    def __init__(
        self,
        tgt_vocab,
        input_size,
        d_model=512,
        nhead=8,
        num_encoder_layers=6,
        num_decoder_layers=6,
        d_ffn=2048,
        dropout=0.1,
        activation=nn.ReLU,
        positional_encoding="fixed_abs_sine",
        normalize_before=False,
        kernel_size: Optional[int] = 31,
        bias: Optional[bool] = True,
        encoder_module: Optional[str] = "transformer",
        conformer_activation: Optional[nn.Module] = Swish,
        branchformer_activation: Optional[nn.Module] = nn.GELU,
        attention_type: Optional[str] = "SummaryMixing",
        max_length: Optional[int] = 2500,
        causal: Optional[bool] = True,
        csgu_linear_units: Optional[int] = 3072,
        gate_activation: Optional[nn.Module] = nn.Identity,
        use_linear_after_conv: Optional[bool] = False,
        local_proj_hid_dim: Optional[list] = [512],
        local_proj_out_dim: Optional[int] = 512,
        summary_hid_dim: Optional[list] = [1024],
        summary_out_dim: Optional[int] = 1024,
        mode: Optional[str] = "SummaryMixing",
    ):
        super().__init__(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            d_ffn=d_ffn,
            dropout=dropout,
            activation=activation,
            positional_encoding=positional_encoding,
            normalize_before=normalize_before,
            kernel_size=kernel_size,
            bias=bias,
            encoder_module=encoder_module,
            conformer_activation=conformer_activation,
            branchformer_activation=branchformer_activation,
            attention_type=attention_type,
            max_length=max_length,
            causal=causal,
            csgu_linear_units=csgu_linear_units,
            gate_activation=gate_activation,
            use_linear_after_conv=use_linear_after_conv,
            local_proj_hid_dim=local_proj_hid_dim,
            local_proj_out_dim=local_proj_out_dim,
            summary_hid_dim=summary_hid_dim,
            summary_out_dim=summary_out_dim,
            mode=mode,
        )

        self.num_decoder_layers = num_decoder_layers
        self.num_encoder_layers = num_encoder_layers

        self.custom_src_module = ModuleList(
            Linear(
                input_size=input_size, n_neurons=d_model, bias=True, combine_dims=False,
            ),
            torch.nn.Dropout(dropout),
        )

        if num_decoder_layers > 0:
            self.custom_tgt_module = ModuleList(NormalizedEmbedding(d_model, tgt_vocab))

        # reset parameters using xavier_normal_
        self._init_params()

    def forward(self, src, tgt=None, wav_len=None, pad_idx=0):
        """
        Arguments
        ----------
        src : torch.Tensor
            The sequence to the encoder.
        tgt : torch.Tensor
            The sequence to the decoder. If None, only the encoder is run.
        wav_len: torch.Tensor, optional
            Torch Tensor of shape (batch, ) containing the relative length to padded length for each example.
        pad_idx : int, optional
            The index for <pad> token (default=0).
        """

        # reshpae the src vector to [Batch, Time, Fea] is a 4d vector is given
        if src.ndim == 4:
            bz, t, ch1, ch2 = src.shape
            src = src.reshape(bz, t, ch1 * ch2)

        (
            src_key_padding_mask,
            tgt_key_padding_mask,
            src_mask,
            tgt_mask,
        ) = make_transformer_src_tgt_masks(
            src, tgt, wav_len, causal=self.causal, pad_idx=pad_idx
        )

        src = self.custom_src_module(src)
        # add pos encoding to queries if are sinusoidal ones else
        if (
            self.attention_type == "hypermixing"
            or self.attention_type == "SummaryMixing"
        ):
            pos_embs_encoder = None
        elif self.attention_type == "RelPosMHAXL":
            pos_embs_encoder = self.positional_encoding(src)
        elif self.positional_encoding_type == "fixed_abs_sine":
            src = src + self.positional_encoding(src)  # add the encodings here
            pos_embs_encoder = None

        encoder_out, _ = self.encoder(
            src=src,
            src_mask=src_mask,
            src_key_padding_mask=src_key_padding_mask,
            pos_embs=pos_embs_encoder,
        )

        tgt = self.custom_tgt_module(tgt)

        if self.attention_type == "RelPosMHAXL":
            # use standard sinusoidal pos encoding in decoder
            tgt = tgt + self.positional_encoding_decoder(tgt)
            pos_embs_encoder = None  # self.positional_encoding(src)
            pos_embs_target = None
        elif (
            self.positional_encoding_type == "fixed_abs_sine"
            or self.attention_type == "hypermixing"
        ):
            tgt = tgt + self.positional_encoding(tgt)
            pos_embs_target = None
            pos_embs_encoder = None

        decoder_out, _, _ = self.decoder(
            tgt=tgt,
            memory=encoder_out,
            memory_mask=None,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=src_key_padding_mask,
            pos_embs_tgt=pos_embs_target,
            pos_embs_src=pos_embs_encoder,
        )

        return encoder_out, decoder_out

    @torch.no_grad()
    def decode(self, tgt, encoder_out, enc_len=None):
        """This method implements a decoding step for the transformer model.

        Arguments
        ---------
        tgt : torch.Tensor
            The sequence to the decoder.
        encoder_out : torch.Tensor
            Hidden output of the encoder.
        enc_len : torch.LongTensor
            The actual length of encoder states.
        """
        tgt_mask = get_lookahead_mask(tgt)
        src_key_padding_mask = None
        if enc_len is not None:
            src_key_padding_mask = (1 - length_to_mask(enc_len)).bool()

        tgt = self.custom_tgt_module(tgt)
        if self.attention_type == "RelPosMHAXL":
            # we use fixed positional encodings in the decoder
            tgt = tgt + self.positional_encoding_decoder(tgt)
            encoder_out = encoder_out + self.positional_encoding_decoder(encoder_out)
            # pos_embs_target = self.positional_encoding(tgt)
            pos_embs_encoder = None  # self.positional_encoding(src)
            pos_embs_target = None
        elif (
            self.positional_encoding_type == "fixed_abs_sine"
            or self.attention_type == "hypermixing"
        ):
            tgt = tgt + self.positional_encoding(tgt)  # add the encodings here
            pos_embs_target = None
            pos_embs_encoder = None

        prediction, self_attns, multihead_attns = self.decoder(
            tgt,
            encoder_out,
            tgt_mask=tgt_mask,
            memory_key_padding_mask=src_key_padding_mask,
            pos_embs_tgt=pos_embs_target,
            pos_embs_src=pos_embs_encoder,
        )
        return prediction, multihead_attns[-1]

    def encode(
        self,
        src,
        wav_len=None,
        pad_idx=0,
        dynchunktrain_config: Optional[DynChunkTrainConfig] = None,
    ):
        """
        Encoder forward pass

        Arguments
        ----------
        src : torch.Tensor
            The sequence to the encoder.
        wav_len: torch.Tensor, optional
            Torch Tensor of shape (batch, ) containing the relative length to padded length for each example.
        """
        # reshape the src vector to [Batch, Time, Fea] if a 4d vector is given
        if src.dim() == 4:
            bz, t, ch1, ch2 = src.shape
            src = src.reshape(bz, t, ch1 * ch2)

        (src_key_padding_mask, _, src_mask, _,) = make_transformer_src_tgt_masks(
            src,
            None,
            wav_len,
            pad_idx=pad_idx,
            causal=self.causal,
            dynchunktrain_config=dynchunktrain_config,
        )

        src = self.custom_src_module(src)
        if self.attention_type == "hypermixing":
            pos_embs_source = None
        elif self.attention_type == "RelPosMHAXL":
            pos_embs_source = self.positional_encoding(src)
        elif self.positional_encoding_type == "fixed_abs_sine":
            src = src + self.positional_encoding(src)
            pos_embs_source = None

        encoder_out, _ = self.encoder(
            src=src,
            src_mask=src_mask,
            src_key_padding_mask=src_key_padding_mask,
            pos_embs=pos_embs_source,
            dynchunktrain_config=dynchunktrain_config,
        )
        return encoder_out

    def encode_streaming(self, src, context: TransformerASRStreamingContext):
        """
        Streaming encoder forward pass

        Arguments
        ---------
        src : torch.Tensor
            The sequence (chunk) to the encoder.

        context : TransformerASRStreamingContext
            Mutable reference to the streaming context. This holds the state
            needed to persist across chunk inferences and can be built using
            `make_streaming_context`. This will get mutated by this function.

        Returns
        -------
        Encoder output for this chunk.

        Example
        -------
        >>> import torch
        >>> from speechbrain.lobes.models.transformer.TransformerASR import TransformerASR
        >>> from speechbrain.utils.dynamic_chunk_training import DynChunkTrainConfig
        >>> net = TransformerASR(
        ...     tgt_vocab=100,
        ...     input_size=64,
        ...     d_model=64,
        ...     nhead=8,
        ...     num_encoder_layers=1,
        ...     num_decoder_layers=0,
        ...     d_ffn=128,
        ...     attention_type="RelPosMHAXL",
        ...     positional_encoding=None,
        ...     encoder_module="conformer",
        ...     normalize_before=True,
        ...     causal=False,
        ... )
        >>> ctx = net.make_streaming_context(
        ...     DynChunkTrainConfig(16, 24),
        ...     encoder_kwargs={"mha_left_context_size": 24},
        ... )
        >>> src1 = torch.rand([8, 16, 64])
        >>> src2 = torch.rand([8, 16, 64])
        >>> out1 = net.encode_streaming(src1, ctx)
        >>> out1.shape
        torch.Size([8, 16, 64])
        >>> ctx.encoder_context.layers[0].mha_left_context.shape
        torch.Size([8, 16, 64])
        >>> out2 = net.encode_streaming(src2, ctx)
        >>> out2.shape
        torch.Size([8, 16, 64])
        >>> ctx.encoder_context.layers[0].mha_left_context.shape
        torch.Size([8, 24, 64])
        >>> combined_out = torch.concat((out1, out2), dim=1)
        >>> combined_out.shape
        torch.Size([8, 32, 64])
        """

        if src.dim() == 4:
            bz, t, ch1, ch2 = src.shape
            src = src.reshape(bz, t, ch1 * ch2)

        # HACK: our problem here is that the positional_encoding is computed
        # against the size of our source tensor, but we only know how many left
        # context frames we're injecting to the encoder within the encoder
        # context.
        # so this workaround does just that.
        #
        # i'm not sure how this would be best refactored, but an option would be
        # to let the encoder get the pos embedding itself and have a way to
        # cache it.
        #
        # additionally, positional encoding functions take in a whole source
        # tensor just to get its attributes (size, device, type) but this is
        # sort of silly for the embeddings that don't need one.
        # so we craft a dummy empty (uninitialized) tensor to help...
        known_left_context = context.encoder_context.layers[0].mha_left_context
        if known_left_context is None:
            pos_encoding_dummy = src
        else:
            target_shape = list(src.shape)
            target_shape[-2] += known_left_context.shape[-2]
            pos_encoding_dummy = torch.empty(size=target_shape).to(src)

        src = self.custom_src_module(src)
        if self.attention_type == "RelPosMHAXL":
            pos_embs_source = self.positional_encoding(pos_encoding_dummy)

        elif self.positional_encoding_type == "fixed_abs_sine":
            src = src + self.positional_encoding(pos_encoding_dummy)
            pos_embs_source = None

        encoder_out, _ = self.encoder.forward_streaming(
            src=src, pos_embs=pos_embs_source, context=context.encoder_context
        )
        return encoder_out

    def make_streaming_context(
        self, dynchunktrain_config: DynChunkTrainConfig, encoder_kwargs={}
    ):
        """Creates a blank streaming context for this transformer and its
        encoder.

        Arguments
        ---------
        dynchunktrain_config : DynChunkTrainConfig
            Runtime chunkwise attention configuration.

        encoder_kwargs : dict
            Parameters to be forward to the encoder's `make_streaming_context`.
            Metadata required for the encoder could differ depending on the
            encoder.
        """
        return TransformerASRStreamingContext(
            dynchunktrain_config=dynchunktrain_config,
            encoder_context=self.encoder.make_streaming_context(**encoder_kwargs,),
        )

    def _init_params(self):
        for p in self.parameters():
            if p.dim() > 1:
                torch.nn.init.xavier_normal_(p)


class EncoderWrapper(nn.Module):
    """This is a wrapper of any ASR transformer encoder. By default, the
    TransformerASR .forward() function encodes and decodes. With this wrapper
    the .forward() function becomes .encode() only.

    Important: The TransformerASR class must contain a .encode() function.

    Arguments
    ----------
    transformer : sb.lobes.models.TransformerInterface
        A Transformer instance that contains a .encode() function.

    Example
    -------
    >>> src = torch.rand([8, 120, 512])
    >>> tgt = torch.randint(0, 720, [8, 120])
    >>> net = TransformerASR(
    ...     720, 512, 512, 8, 1, 1, 1024, activation=torch.nn.GELU
    ... )
    >>> encoder = EncoderWrapper(net)
    >>> enc_out = encoder(src)
    >>> enc_out.shape
    torch.Size([8, 120, 512])
    """

    def __init__(self, transformer, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.transformer = transformer

    def forward(self, x, wav_lens=None, pad_idx=0, **kwargs):
        """ Processes the input tensor x and returns an output tensor."""
        x = self.transformer.encode(x, wav_lens, pad_idx, **kwargs,)
        return x
