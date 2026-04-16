# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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
"""PyTorch Parakeet model."""

import math
from collections.abc import Callable
from dataclasses import dataclass

import torch
from torch import nn

from ... import initialization as init
from ...activations import ACT2FN
from ...modeling_layers import GradientCheckpointingLayer
from ...modeling_outputs import BaseModelOutput, CausalLMOutput
from ...modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from ...processing_utils import Unpack
from ...utils import ModelOutput, TransformersKwargs, auto_docstring, can_return_tuple
from ...utils.generic import maybe_autocast, merge_with_config_defaults
from ...utils.output_capturing import capture_outputs
from ..fastspeech2_conformer.modeling_fastspeech2_conformer import FastSpeech2ConformerConvolutionModule
from ..llama.modeling_llama import LlamaAttention, eager_attention_forward
from .configuration_parakeet import (
    ParakeetCTCConfig,
    ParakeetEncoderConfig,
    ParakeetJointNetworkConfig,
    ParakeetPredictionNetworkConfig,
    ParakeetRNNTConfig,
    ParakeetTDTConfig,
)


@dataclass
@auto_docstring(
    custom_intro="""
    Extends [~modeling_outputs.BaseModelOutput] to include the output attention mask since sequence length is not preserved in the model's forward.
    """
)
class ParakeetEncoderModelOutput(BaseModelOutput):
    attention_mask: torch.Tensor | None = None


class ParakeetEncoderRelPositionalEncoding(nn.Module):
    """Relative positional encoding for Parakeet."""

    inv_freq: torch.Tensor  # fix linting for `register_buffer`

    def __init__(self, config: ParakeetEncoderConfig, device=None):
        super().__init__()
        self.max_position_embeddings = config.max_position_embeddings
        base = 10000.0
        inv_freq = 1.0 / (
            base
            ** (
                torch.arange(0, config.hidden_size, 2, dtype=torch.int64).to(device=device, dtype=torch.float)
                / config.hidden_size
            )
        )

        self.register_buffer("inv_freq", inv_freq, persistent=False)

    @torch.no_grad()
    def forward(self, hidden_states: torch.Tensor):
        seq_length = hidden_states.shape[1]
        if seq_length > self.max_position_embeddings:
            raise ValueError(
                f"Sequence Length: {seq_length} has to be less or equal than "
                f"config.max_position_embeddings {self.max_position_embeddings}."
            )

        position_ids = torch.arange(seq_length - 1, -seq_length, -1, device=hidden_states.device)
        inv_freq_expanded = (
            self.inv_freq[None, :, None].float().expand(hidden_states.shape[0], -1, 1).to(hidden_states.device)
        )
        position_ids_expanded = position_ids[None, None, :].float()

        device_type = (
            hidden_states.device.type
            if isinstance(hidden_states.device.type, str) and hidden_states.device.type != "mps"
            else "cpu"
        )
        with maybe_autocast(device_type=device_type, enabled=False):  # Force float32
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            sin = freqs.sin()
            cos = freqs.cos()
            # interleave sin and cos
            pos_embed = torch.stack([sin, cos], dim=-1)
            pos_embed = pos_embed.reshape(*pos_embed.shape[:-2], -1)

        return pos_embed.to(dtype=hidden_states.dtype)


class ParakeetEncoderFeedForward(nn.Module):
    def __init__(self, config: ParakeetEncoderConfig):
        super().__init__()
        self.linear1 = nn.Linear(config.hidden_size, config.intermediate_size, bias=config.attention_bias)
        self.activation = ACT2FN[config.hidden_act]
        self.linear2 = nn.Linear(config.intermediate_size, config.hidden_size, bias=config.attention_bias)
        self.activation_dropout = config.activation_dropout

    def forward(self, hidden_states):
        hidden_states = self.activation(self.linear1(hidden_states))
        hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)
        hidden_states = self.linear2(hidden_states)
        return hidden_states


class ParakeetEncoderConvolutionModule(FastSpeech2ConformerConvolutionModule):
    def __init__(self, config: ParakeetEncoderConfig, module_config=None):
        super().__init__(config, module_config)


class ParakeetEncoderAttention(LlamaAttention):
    """Multi-head attention with relative positional encoding. See section 3.3 of https://huggingface.co/papers/1901.02860."""

    def __init__(self, config: ParakeetEncoderConfig, layer_idx: int):
        super().__init__(config, layer_idx=layer_idx)
        self.is_causal = False
        # W_{k,R} projection
        self.relative_k_proj = nn.Linear(config.hidden_size, config.num_attention_heads * self.head_dim, bias=False)
        # global content bias
        self.bias_u = nn.Parameter(torch.zeros(config.num_attention_heads, self.head_dim))
        # global positional bias
        self.bias_v = nn.Parameter(torch.zeros(config.num_attention_heads, self.head_dim))

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: torch.Tensor | None,
        attention_mask: torch.Tensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        input_shape = hidden_states.shape[:-1]
        batch_size, seq_length = input_shape
        hidden_shape = (batch_size, seq_length, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        attention_interface: Callable = ALL_ATTENTION_FUNCTIONS.get_interface(
            self.config._attn_implementation, eager_attention_forward
        )

        query_states_with_bias_u = query_states + self.bias_u.view(
            1, self.config.num_attention_heads, 1, self.head_dim
        )
        query_states_with_bias_v = query_states + self.bias_v.view(
            1, self.config.num_attention_heads, 1, self.head_dim
        )

        relative_key_states = self.relative_k_proj(position_embeddings)
        relative_key_states = relative_key_states.view(batch_size, -1, self.config.num_attention_heads, self.head_dim)

        # terms (b) and (d)
        matrix_bd = query_states_with_bias_v @ relative_key_states.permute(0, 2, 3, 1)
        matrix_bd = self._rel_shift(matrix_bd)
        matrix_bd = matrix_bd[..., :seq_length]
        matrix_bd = matrix_bd * self.scaling

        if attention_mask is not None:
            # here the original codebase uses -10000.0 rather than float("-inf") and then manual masked fill with 0.0s
            # see: https://github.com/NVIDIA-NeMo/NeMo/blob/8cfedd7203462cb251a914e700e5605444277561/nemo/collections/asr/parts/submodules/multi_head_attention.py#L320-L340
            # we rather went for a straight-forward approach with float("-inf")
            matrix_bd = matrix_bd.masked_fill_(attention_mask.logical_not(), float("-inf"))

        # will compute matrix_ac - terms (a) and (c) - and add matrix_bd
        attn_output, attn_weights = attention_interface(
            self,
            query=query_states_with_bias_u,
            key=key_states,
            value=value_states,
            attention_mask=matrix_bd,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights

    def _rel_shift(self, attention_scores):
        """Relative position shift for Shaw et al. style attention. See appendix B of https://huggingface.co/papers/1901.02860."""
        batch_size, num_heads, query_length, position_length = attention_scores.shape
        attention_scores = nn.functional.pad(attention_scores, pad=(1, 0))
        attention_scores = attention_scores.view(batch_size, num_heads, -1, query_length)
        attention_scores = attention_scores[:, :, 1:].view(batch_size, num_heads, query_length, position_length)
        return attention_scores


class ParakeetEncoderSubsamplingConv2D(nn.Module):
    def __init__(self, config: ParakeetEncoderConfig):
        super().__init__()

        self.kernel_size = config.subsampling_conv_kernel_size
        self.stride = config.subsampling_conv_stride
        self.channels = config.subsampling_conv_channels
        self.padding = (self.kernel_size - 1) // 2
        self.num_layers = int(math.log2(config.subsampling_factor))

        # define layers
        self.layers = nn.ModuleList()
        self.layers.append(
            nn.Conv2d(1, self.channels, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding)
        )
        self.layers.append(nn.ReLU())
        for i in range(self.num_layers - 1):
            # depthwise conv
            self.layers.append(
                nn.Conv2d(
                    self.channels,
                    self.channels,
                    kernel_size=self.kernel_size,
                    stride=self.stride,
                    padding=self.padding,
                    groups=self.channels,
                )
            )
            # pointwise conv
            self.layers.append(nn.Conv2d(self.channels, self.channels, kernel_size=1))
            # activation
            self.layers.append(nn.ReLU())

        out_length = config.num_mel_bins // (self.stride**self.num_layers)
        self.linear = nn.Linear(config.subsampling_conv_channels * out_length, config.hidden_size, bias=True)

    def _get_output_length(self, input_lengths: torch.Tensor, conv_layer: nn.Conv2d):
        if hasattr(conv_layer, "stride") and conv_layer.stride != (1, 1):
            padding = conv_layer.padding
            kernel_size = conv_layer.kernel_size[0]
            stride = conv_layer.stride[0]

            output_lengths = (input_lengths + padding[0] + padding[1] - kernel_size) // stride + 1
            return output_lengths

        return input_lengths

    def forward(self, input_features: torch.Tensor, attention_mask: torch.Tensor = None):
        hidden_states = input_features.unsqueeze(1)
        current_lengths = attention_mask.sum(-1) if attention_mask is not None else None

        for layer in self.layers:
            hidden_states = layer(hidden_states)

            # mask the hidden states
            if isinstance(layer, nn.Conv2d) and attention_mask is not None:
                current_lengths = self._get_output_length(current_lengths, layer)
                current_seq_length = hidden_states.shape[2]
                channel_mask = (
                    torch.arange(current_seq_length, device=attention_mask.device) < current_lengths[:, None]
                )
                hidden_states *= channel_mask[:, None, :, None]

        hidden_states = hidden_states.transpose(1, 2).reshape(hidden_states.shape[0], hidden_states.shape[2], -1)
        hidden_states = self.linear(hidden_states)

        return hidden_states


class ParakeetEncoderBlock(GradientCheckpointingLayer):
    def __init__(self, config: ParakeetEncoderConfig, layer_idx: int | None = None):
        super().__init__()
        self.gradient_checkpointing = False

        self.feed_forward1 = ParakeetEncoderFeedForward(config)
        self.self_attn = ParakeetEncoderAttention(config, layer_idx)
        self.conv = ParakeetEncoderConvolutionModule(config)
        self.feed_forward2 = ParakeetEncoderFeedForward(config)

        self.norm_feed_forward1 = nn.LayerNorm(config.hidden_size)
        self.norm_self_att = nn.LayerNorm(config.hidden_size)
        self.norm_conv = nn.LayerNorm(config.hidden_size)
        self.norm_feed_forward2 = nn.LayerNorm(config.hidden_size)
        self.norm_out = nn.LayerNorm(config.hidden_size)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_embeddings: torch.Tensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.feed_forward1(self.norm_feed_forward1(hidden_states))
        hidden_states = residual + 0.5 * hidden_states  # the conformer architecture uses a factor of 0.5

        normalized_hidden_states = self.norm_self_att(hidden_states)
        attn_output, _ = self.self_attn(
            hidden_states=normalized_hidden_states,
            attention_mask=attention_mask,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states = hidden_states + attn_output

        conv_output = self.conv(self.norm_conv(hidden_states), attention_mask=attention_mask)
        hidden_states = hidden_states + conv_output

        ff2_output = self.feed_forward2(self.norm_feed_forward2(hidden_states))
        hidden_states = hidden_states + 0.5 * ff2_output  # the conformer architecture uses a factor of 0.5

        hidden_states = self.norm_out(hidden_states)

        return hidden_states


@auto_docstring
class ParakeetPreTrainedModel(PreTrainedModel):
    config: ParakeetCTCConfig
    base_model_prefix = "model"
    main_input_name = "input_features"
    input_modalities = "audio"
    supports_gradient_checkpointing = True
    _no_split_modules = ["ParakeetEncoderBlock"]
    _supports_flat_attention_mask = True
    _supports_sdpa = True
    _supports_flex_attn = True

    # TODO: @eustlb, add support when flash attention supports custom attention bias
    _supports_flash_attn = False

    _can_compile_fullgraph = True
    _supports_attention_backend = True
    _can_record_outputs = {
        "hidden_states": ParakeetEncoderBlock,
        "attentions": ParakeetEncoderAttention,
    }

    @torch.no_grad()
    def _init_weights(self, module):
        super()._init_weights(module)

        if hasattr(self.config, "initializer_range"):
            std = self.config.initializer_range
        else:
            # 0.02 is the standard default value across the library
            std = getattr(self.config.get_text_config(), "initializer_range", 0.02)

        if isinstance(module, ParakeetEncoderAttention):
            # Initialize positional bias parameters
            init.normal_(module.bias_u, mean=0.0, std=std)
            init.normal_(module.bias_v, mean=0.0, std=std)
        elif isinstance(module, ParakeetEncoderRelPositionalEncoding):
            encoder_config = self.config.encoder_config if hasattr(self.config, "encoder_config") else self.config
            inv_freq = 1.0 / (
                10000.0
                ** (
                    torch.arange(0, encoder_config.hidden_size, 2, dtype=torch.int64)
                    / encoder_config.hidden_size
                )
            )
            init.copy_(module.inv_freq, inv_freq)

    def _get_subsampling_output_length(self, input_lengths: torch.Tensor):
        encoder_config = self.config.encoder_config if hasattr(self.config, "encoder_config") else self.config

        kernel_size = encoder_config.subsampling_conv_kernel_size
        stride = encoder_config.subsampling_conv_stride
        num_layers = int(math.log2(encoder_config.subsampling_factor))

        all_paddings = (kernel_size - 1) // 2 * 2
        add_pad = all_paddings - kernel_size
        lengths = input_lengths

        for _ in range(num_layers):
            lengths = torch.div(lengths.to(dtype=torch.float) + add_pad, stride) + 1.0
            lengths = torch.floor(lengths)

        return lengths.to(dtype=torch.int)

    def _get_output_attention_mask(self, attention_mask: torch.Tensor, target_length: int | None = None):
        """
        Convert the input attention mask to its subsampled form. `target_length` sets the desired output length, useful
        when the attention mask length differs from `sum(-1).max()` (i.e., when the longest sequence in the batch is padded)
        """
        output_lengths = self._get_subsampling_output_length(attention_mask.sum(-1))
        # Use target_length if provided, otherwise use max length in batch
        max_length = target_length if target_length is not None else output_lengths.max()
        attention_mask = torch.arange(max_length, device=attention_mask.device) < output_lengths[:, None]
        return attention_mask


@auto_docstring(
    custom_intro="""
    The Parakeet Encoder model, based on the [Fast Conformer architecture](https://huggingface.co/papers/2305.05084).
    """
)
class ParakeetEncoder(ParakeetPreTrainedModel):
    config: ParakeetEncoderConfig
    base_model_prefix = "encoder"

    def __init__(self, config: ParakeetEncoderConfig):
        super().__init__(config)
        self.config = config
        self.gradient_checkpointing = False

        self.dropout = config.dropout
        self.dropout_positions = config.dropout_positions
        self.layerdrop = config.layerdrop

        self.input_scale = math.sqrt(config.hidden_size) if config.scale_input else 1.0
        self.subsampling = ParakeetEncoderSubsamplingConv2D(config)
        self.encode_positions = ParakeetEncoderRelPositionalEncoding(config)

        self.layers = nn.ModuleList(
            [ParakeetEncoderBlock(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )

        self.post_init()

    @auto_docstring
    @merge_with_config_defaults
    @capture_outputs
    @can_return_tuple
    def forward(
        self,
        input_features: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        output_attention_mask: bool = True,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutput:
        r"""
        output_attention_mask (`bool`, *optional*, defaults to `True`):
            Whether to return the output attention mask. Only effective when `attention_mask` is provided.

        Example:

        ```python
        >>> from transformers import AutoProcessor, ParakeetEncoder
        >>> from datasets import load_dataset, Audio

        >>> model_id = "nvidia/parakeet-ctc-1.1b"
        >>> processor = AutoProcessor.from_pretrained(model_id)
        >>> encoder = ParakeetEncoder.from_pretrained(model_id)

        >>> ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
        >>> ds = ds.cast_column("audio", Audio(sampling_rate=processor.feature_extractor.sampling_rate))

        >>> inputs = processor(ds[0]["audio"]["array"])
        >>> encoder_outputs = encoder(**inputs)

        >>> print(encoder_outputs.last_hidden_state.shape)
        ```
        """

        hidden_states = self.subsampling(input_features, attention_mask)
        hidden_states = hidden_states * self.input_scale
        position_embeddings = self.encode_positions(hidden_states)

        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        position_embeddings = nn.functional.dropout(
            position_embeddings, p=self.dropout_positions, training=self.training
        )

        if attention_mask is not None:
            output_mask = self._get_output_attention_mask(attention_mask, target_length=hidden_states.shape[1])
            attention_mask = output_mask.unsqueeze(1).expand(-1, hidden_states.shape[1], -1)
            attention_mask = attention_mask & attention_mask.transpose(1, 2)
            attention_mask = attention_mask.unsqueeze(1)

        for encoder_layer in self.layers:
            # add LayerDrop (see https://huggingface.co/papers/1909.11556 for description)
            to_drop = False
            if self.training:
                dropout_probability = torch.rand([])
                if dropout_probability < self.layerdrop:  # skip the layer
                    to_drop = True

            if not to_drop:
                hidden_states = encoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_embeddings=position_embeddings,
                    **kwargs,
                )

        return ParakeetEncoderModelOutput(
            last_hidden_state=hidden_states,
            attention_mask=output_mask.int() if attention_mask is not None and output_attention_mask else None,
        )


@dataclass
class ParakeetGenerateOutput(ModelOutput):
    """
    Outputs of Parakeet models.

    Args:
        sequences (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            The generated sequences. The second dimension (sequence_length) is either equal to `max_length` or shorter
            if all batches finished early due to the `eos_token_id`.
        logits (`tuple(torch.FloatTensor)` *optional*, returned when `output_logits=True`):
            Unprocessed prediction scores of the language modeling head (scores for each vocabulary token before SoftMax)
            at each generation step. Tuple of `torch.FloatTensor` with up to `max_new_tokens` elements (one element for
            each generated token), with each tensor of shape `(batch_size, config.vocab_size)`.
        attentions (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `output_attentions=True`):
            Tuple (one element for each generated token) of tuples (one element for each layer of the decoder) of
            `torch.FloatTensor` of shape `(batch_size, num_heads, generated_length, sequence_length)`.
        hidden_states (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `output_hidden_states=True`):
            Tuple (one element for each generated token) of tuples (one element for each layer of the decoder) of
            `torch.FloatTensor` of shape `(batch_size, generated_length, hidden_size)`.
    """

    sequences: torch.LongTensor
    logits: tuple[torch.FloatTensor] | None = None
    attentions: tuple[tuple[torch.FloatTensor]] | None = None
    hidden_states: tuple[tuple[torch.FloatTensor]] | None = None


@auto_docstring(
    custom_intro="""
    Parakeet Encoder with a Connectionist Temporal Classification (CTC) head.
    """
)
class ParakeetForCTC(ParakeetPreTrainedModel):
    config: ParakeetCTCConfig

    def __init__(self, config: ParakeetCTCConfig):
        super().__init__(config)
        self.encoder = ParakeetEncoder(config.encoder_config)
        # Conv rather than linear to be consistent with NeMO decoding layer
        self.ctc_head = nn.Conv1d(config.encoder_config.hidden_size, config.vocab_size, kernel_size=1)

        self.post_init()

    @auto_docstring
    @can_return_tuple
    def forward(
        self,
        input_features: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> CausalLMOutput:
        r"""
        Example:

        ```python
        >>> from transformers import AutoProcessor, ParakeetForCTC
        >>> from datasets import load_dataset, Audio

        >>> model_id = "nvidia/parakeet-ctc-1.1b"
        >>> processor = AutoProcessor.from_pretrained(model_id)
        >>> model = ParakeetForCTC.from_pretrained(model_id)

        >>> ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
        >>> ds = ds.cast_column("audio", Audio(sampling_rate=processor.feature_extractor.sampling_rate))

        >>> inputs = processor(ds[0]["audio"]["array"], text=ds[0]["text"])
        >>> outputs = model(**inputs)

        >>> print(outputs.loss)
        ```"""

        encoder_outputs = self.encoder(
            input_features=input_features,
            attention_mask=attention_mask,
            **kwargs,
        )

        hidden_states = encoder_outputs.last_hidden_state
        logits = self.ctc_head(hidden_states.transpose(1, 2)).transpose(1, 2)

        loss = None
        if labels is not None:
            # retrieve loss input_lengths from attention_mask
            attention_mask = (
                attention_mask if attention_mask is not None else torch.ones_like(input_features, dtype=torch.long)
            )
            input_lengths = self._get_subsampling_output_length(attention_mask.sum(-1))

            # assuming that padded tokens are filled with -100
            # when not being attended to
            labels_mask = labels != self.config.pad_token_id
            target_lengths = labels_mask.sum(-1)
            flattened_targets = labels.masked_select(labels_mask)

            # ctc_loss doesn't support fp16
            log_probs = nn.functional.log_softmax(logits, dim=-1, dtype=torch.float32).transpose(0, 1)

            with torch.backends.cudnn.flags(enabled=False):
                loss = nn.functional.ctc_loss(
                    log_probs,
                    flattened_targets,
                    input_lengths,
                    target_lengths,
                    blank=self.config.pad_token_id,
                    reduction=self.config.ctc_loss_reduction,
                    zero_infinity=self.config.ctc_zero_infinity,
                )

        return CausalLMOutput(
            loss=loss,
            logits=logits,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )

    @torch.no_grad()
    def generate(
        self,
        input_features: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        return_dict_in_generate: bool = False,
        **kwargs: Unpack[TransformersKwargs],
    ) -> ParakeetGenerateOutput | torch.LongTensor:
        r"""
        Example:

        ```python
        >>> from transformers import AutoProcessor, ParakeetForCTC
        >>> from datasets import load_dataset, Audio

        >>> model_id = "nvidia/parakeet-ctc-1.1b"
        >>> processor = AutoProcessor.from_pretrained(model_id)
        >>> model = ParakeetForCTC.from_pretrained(model_id)

        >>> ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
        >>> ds = ds.cast_column("audio", Audio(sampling_rate=processor.feature_extractor.sampling_rate))

        >>> inputs = processor(ds[0]["audio"]["array"], text=ds[0]["text"])
        >>> predicted_ids = model.generate(**inputs)
        >>> transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)

        >>> print(transcription)
        ```
        """
        kwargs["return_dict"] = True
        outputs: CausalLMOutput = self.forward(
            input_features=input_features,
            attention_mask=attention_mask,
            **kwargs,
        )

        # greedy decoding
        sequences = outputs.logits.argmax(dim=-1)

        # mask out padded tokens
        if attention_mask is not None:
            attention_mask = self._get_output_attention_mask(attention_mask, target_length=sequences.shape[1])
            sequences[~attention_mask] = self.config.pad_token_id

        if return_dict_in_generate:
            return ParakeetGenerateOutput(
                sequences=sequences,
                logits=outputs.logits,
                attentions=outputs.attentions,
                hidden_states=outputs.hidden_states,
            )

        return sequences


class ParakeetRNNTLoss(nn.Module):
    """
    Pure-PyTorch RNNT loss via the forward-backward algorithm on a T×U lattice.

    Ports NeMo's ``RNNTLossPytorch`` with no additional dependencies.  For
    production use a warp-RNNT kernel backend can be substituted here without
    changing any calling code.

    Args:
        blank: Index of the blank token (must equal ``vocab_size - 1``).
        reduction: Per-sample reduction.  One of ``"mean_batch"``,
            ``"mean"``, ``"sum"``, ``"mean_volume"``.
    """

    def __init__(self, blank: int, reduction: str = "mean_batch"):
        super().__init__()
        self.blank = blank
        self.reduction = reduction

    def forward(
        self,
        acts: torch.Tensor,
        labels: torch.Tensor,
        act_lens: torch.Tensor,
        label_lens: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            acts: Joint logits of shape ``[B, T, U, V + 1]``.  *Not*
                log-softmax'd — this method applies log-softmax internally.
            labels: Target token ids of shape ``[B, U - 1]``.
            act_lens: Valid encoder-output lengths, shape ``[B]``.
            label_lens: Valid label lengths, shape ``[B]``.

        Returns:
            Scalar loss tensor.
        """
        # CPU fp16 → fp32 for numerical stability
        if not acts.is_cuda and acts.dtype == torch.float16:
            acts = acts.float()

        log_probs = torch.log_softmax(acts, dim=-1)
        forward_log_probs = self._compute_forward_prob(log_probs, labels, act_lens, label_lens)
        losses = -forward_log_probs
        return self._reduce(losses, label_lens)

    def _compute_forward_prob(
        self,
        log_probs: torch.Tensor,
        labels: torch.Tensor,
        act_lens: torch.Tensor,
        label_lens: torch.Tensor,
    ) -> torch.Tensor:
        B, T, U, _ = log_probs.shape

        # log_alpha[b, t, u] = log P(alignment prefix uses frames 0..t and emits labels 0..u-1)
        log_alpha = torch.zeros(B, T, U, device=log_probs.device, dtype=log_probs.dtype)

        for t in range(T):
            for u in range(U):
                if u == 0:
                    if t == 0:
                        log_alpha[:, t, u] = 0.0
                    else:
                        # (t-1, 0) blank transition
                        log_alpha[:, t, u] = log_alpha[:, t - 1, u] + log_probs[:, t - 1, 0, self.blank]
                else:
                    if t == 0:
                        # (0, u-1) label transition
                        label_scores = torch.gather(
                            log_probs[:, t, u - 1],
                            dim=1,
                            index=labels[:, u - 1].unsqueeze(1).long(),
                        ).squeeze(1)
                        log_alpha[:, t, u] = log_alpha[:, t, u - 1] + label_scores
                    else:
                        # blank: (t-1, u) → (t, u)
                        blank_path = log_alpha[:, t - 1, u] + log_probs[:, t - 1, u, self.blank]
                        # label: (t, u-1) → (t, u)
                        label_scores = torch.gather(
                            log_probs[:, t, u - 1],
                            dim=1,
                            index=labels[:, u - 1].unsqueeze(1).long(),
                        ).squeeze(1)
                        label_path = log_alpha[:, t, u - 1] + label_scores
                        log_alpha[:, t, u] = torch.logaddexp(blank_path, label_path)

        # Terminal: alpha[T_b-1, U_b] + blank(T_b-1, U_b)
        terminal_log_probs = torch.stack(
            [
                log_alpha[b, act_lens[b] - 1, label_lens[b]]
                + log_probs[b, act_lens[b] - 1, label_lens[b], self.blank]
                for b in range(B)
            ]
        )
        return terminal_log_probs

    def _reduce(self, losses: torch.Tensor, label_lens: torch.Tensor) -> torch.Tensor:
        if self.reduction == "mean_batch":
            return losses.mean()
        elif self.reduction == "mean":
            return torch.div(losses, label_lens).mean()
        elif self.reduction == "sum":
            return losses.sum()
        elif self.reduction == "mean_volume":
            return losses.sum() / label_lens.sum()
        return losses


class ParakeetTDTLoss(nn.Module):
    """
    Pure-PyTorch TDT (Token-and-Duration Transducer) loss.

    Ports NeMo's ``TDTLossPytorch`` and adds the RNNT regularisation term
    weighted by ``omega``.  The combined objective is::

        loss = omega * rnnt_loss + (1 - omega) * tdt_loss

    where both components use sigma-underhnormalised label logits as described
    in `Efficient Sequence Transduction by Jointly Predicting Tokens and
    Durations <https://arxiv.org/abs/2304.06795>`__.

    Args:
        blank: Index of the blank token.
        durations: Ordered candidate frame-skip values (must include 0 and at
            least one positive integer), e.g. ``[0, 1, 2, 3, 4]``.
        reduction: Per-sample reduction.  One of ``"mean_batch"``,
            ``"mean"``, ``"sum"``, ``"mean_volume"``.
        sigma: Log-domain under-normalisation coefficient applied to label
            logits before computing the forward probability.
        omega: Weight of the RNNT regularisation term.  Set to 0 to use
            the pure TDT objective.
    """

    def __init__(
        self,
        blank: int,
        durations: list[int],
        reduction: str = "mean_batch",
        sigma: float = 0.05,
        omega: float = 0.1,
    ):
        super().__init__()
        self.blank = blank
        self.durations = durations
        self.n_durations = len(durations)
        self.reduction = reduction
        self.sigma = sigma
        self.omega = omega
        self._rnnt_loss = ParakeetRNNTLoss(blank=blank, reduction=reduction)

    def forward(
        self,
        acts: torch.Tensor,
        labels: torch.Tensor,
        act_lens: torch.Tensor,
        label_lens: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            acts: Joint logits of shape ``[B, T, U, V + 1 + num_durations]``.
                *Not* log-softmax'd.
            labels: Target token ids of shape ``[B, U - 1]``.
            act_lens: Valid encoder-output lengths, shape ``[B]``.
            label_lens: Valid label lengths, shape ``[B]``.

        Returns:
            Scalar combined TDT loss.
        """
        label_acts = acts[:, :, :, : -self.n_durations]
        duration_acts = acts[:, :, :, -self.n_durations :]

        # sigma under-normalisation on label logits; duration logits are normalised normally
        log_label_acts = torch.log_softmax(label_acts, dim=-1) - self.sigma
        log_duration_acts = torch.log_softmax(duration_acts, dim=-1)

        tdt_forward_log_probs = self._compute_tdt_forward_prob(
            log_label_acts, log_duration_acts, labels, act_lens, label_lens
        )
        tdt_losses = -tdt_forward_log_probs
        tdt_loss = self._reduce(tdt_losses, label_lens)

        if self.omega == 0.0:
            return tdt_loss

        # RNNT regularisation term: standard RNNT loss on the sigma-normalised label logits.
        # We pass pre-softmax'd label_acts to ParakeetRNNTLoss, which re-applies log_softmax.
        # To avoid double-softmax we build a thin wrapper that takes already-log-softmax'd input.
        rnnt_forward_log_probs = self._rnnt_loss._compute_forward_prob(
            log_label_acts, labels, act_lens, label_lens
        )
        rnnt_losses = -rnnt_forward_log_probs
        rnnt_loss = self._rnnt_loss._reduce(rnnt_losses, label_lens)

        return self.omega * rnnt_loss + (1.0 - self.omega) * tdt_loss

    def _compute_tdt_forward_prob(
        self,
        log_label_acts: torch.Tensor,
        log_duration_acts: torch.Tensor,
        labels: torch.Tensor,
        act_lens: torch.Tensor,
        label_lens: torch.Tensor,
    ) -> torch.Tensor:
        B, T, U, _ = log_label_acts.shape

        NEG_INF = -1000.0
        log_alpha = torch.full((B, T, U), NEG_INF, device=log_label_acts.device, dtype=log_label_acts.dtype)

        for b in range(B):
            for t in range(T):
                for u in range(U):
                    if u == 0:
                        if t == 0:
                            log_alpha[b, t, u] = 0.0
                        else:
                            # only blank transitions reach (t, 0) for t > 0
                            for n, dur in enumerate(self.durations):
                                if dur > 0 and t - dur >= 0:
                                    tmp = (
                                        log_alpha[b, t - dur, u]
                                        + log_label_acts[b, t - dur, u, self.blank]
                                        + log_duration_acts[b, t - dur, u, n]
                                    )
                                    log_alpha[b, t, u] = torch.logaddexp(
                                        log_alpha[b, t, u],
                                        tmp,
                                    )
                    else:
                        for n, dur in enumerate(self.durations):
                            if t - dur >= 0:
                                if dur > 0:
                                    # blank transition from (t-dur, u)
                                    tmp = (
                                        log_alpha[b, t - dur, u]
                                        + log_label_acts[b, t - dur, u, self.blank]
                                        + log_duration_acts[b, t - dur, u, n]
                                    )
                                    log_alpha[b, t, u] = torch.logaddexp(log_alpha[b, t, u], tmp)
                                # label transition from (t-dur, u-1)
                                tmp = (
                                    log_alpha[b, t - dur, u - 1]
                                    + log_label_acts[b, t - dur, u - 1, labels[b, u - 1]]
                                    + log_duration_acts[b, t - dur, u - 1, n]
                                )
                                log_alpha[b, t, u] = torch.logaddexp(log_alpha[b, t, u], tmp)

        # Collect terminal probabilities: sum over all valid ending blank durations
        terminal_log_probs = []
        for b in range(B):
            t_b = act_lens[b].item()
            u_b = label_lens[b].item()
            log_prob = torch.tensor(NEG_INF, device=log_label_acts.device, dtype=log_label_acts.dtype)
            for n, dur in enumerate(self.durations):
                if dur > 0 and t_b - dur >= 0:
                    tmp = (
                        log_alpha[b, t_b - dur, u_b]
                        + log_label_acts[b, t_b - dur, u_b, self.blank]
                        + log_duration_acts[b, t_b - dur, u_b, n]
                    )
                    log_prob = torch.logaddexp(log_prob, tmp)
            terminal_log_probs.append(log_prob)
        return torch.stack(terminal_log_probs)

    def _reduce(self, losses: torch.Tensor, label_lens: torch.Tensor) -> torch.Tensor:
        return self._rnnt_loss._reduce(losses, label_lens)


class ParakeetPredictionNetwork(nn.Module):
    """
    LSTM-based prediction network (decoder) for RNNT and TDT models.

    Args:
        config: Prediction network configuration.
        vocab_size: Full vocabulary size *including* the blank token.
        blank_id: Index of the blank token.  When ``config.blank_as_pad`` is
            ``True`` the blank embedding is tied to the zero vector via
            ``padding_idx``, eliminating a special-case branch in the decode
            loop and enabling CUDA-graph capture.
    """

    def __init__(
        self,
        config: ParakeetPredictionNetworkConfig,
        vocab_size: int,
        blank_id: int,
    ):
        super().__init__()
        self.pred_hidden = config.pred_hidden
        self.pred_rnn_layers = config.pred_rnn_layers
        self.blank_as_pad = config.blank_as_pad
        self.blank_id = blank_id

        padding_idx = blank_id if config.blank_as_pad else None
        self.embed = nn.Embedding(vocab_size, config.pred_hidden, padding_idx=padding_idx)

        # nn.LSTM dropout is only applied between layers, so it is 0 for single-layer models
        lstm_dropout = config.dropout if config.pred_rnn_layers > 1 else 0.0
        self.lstm = nn.LSTM(
            input_size=config.pred_hidden,
            hidden_size=config.pred_hidden,
            num_layers=config.pred_rnn_layers,
            batch_first=True,
            dropout=lstm_dropout,
        )

    def forward(
        self,
        labels: torch.Tensor,
        label_lengths: torch.Tensor,
        state: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """
        Training forward pass — processes the full label sequence in one call.

        Args:
            labels: Token ids of shape ``[B, U]``.
            label_lengths: Valid label counts of shape ``[B]`` (unused for
                padding but kept for API consistency).
            state: Optional initial LSTM state ``(h, c)``, each of shape
                ``[num_layers, B, pred_hidden]``.

        Returns:
            Tuple of ``(output, (h_n, c_n))`` where *output* has shape
            ``[B, U + 1, pred_hidden]`` (SOS prepended as a zero vector) and
            *(h_n, c_n)* are the final LSTM hidden and cell states.
        """
        B = labels.size(0)
        embedded = self.embed(labels)  # [B, U, H]

        # Prepend blank "start-of-sequence" as a zero vector
        sos = torch.zeros(B, 1, self.pred_hidden, device=labels.device, dtype=embedded.dtype)
        embedded = torch.cat([sos, embedded], dim=1)  # [B, U+1, H]

        output, (h_n, c_n) = self.lstm(embedded, state)  # [B, U+1, H]
        return output, (h_n, c_n)

    def predict(
        self,
        y: torch.Tensor | None,
        state: tuple[torch.Tensor, torch.Tensor] | None,
        batch_size: int | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """
        Single-step prediction for greedy / beam-search decoding.

        Args:
            y: Last emitted token ids of shape ``[B, 1]``, or ``None`` to
               emit a zero vector (used for the very first SOS step).
            state: Current LSTM state ``(h, c)`` or ``None`` for the initial
                zero state.
            batch_size: Required when both ``y`` and ``state`` are ``None``.

        Returns:
            Tuple of ``(output, (h_n, c_n))`` where *output* has shape
            ``[B, 1, pred_hidden]``.
        """
        device = self.embed.weight.device
        dtype = self.embed.weight.dtype

        if y is not None:
            embedded = self.embed(y)  # [B, 1, H]
        else:
            if batch_size is None:
                batch_size = state[0].size(1) if state is not None else 1
            embedded = torch.zeros(batch_size, 1, self.pred_hidden, device=device, dtype=dtype)

        output, (h_n, c_n) = self.lstm(embedded, state)
        return output, (h_n, c_n)

    def initialize_state(
        self,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return zero-initialised ``(h, c)`` LSTM state for ``batch_size`` sequences."""
        h = torch.zeros(self.pred_rnn_layers, batch_size, self.pred_hidden, device=device, dtype=dtype)
        c = torch.zeros(self.pred_rnn_layers, batch_size, self.pred_hidden, device=device, dtype=dtype)
        return h, c


class ParakeetJointNetwork(nn.Module):
    """
    Joint network for RNNT and TDT models.

    Projects encoder and prediction-network outputs to a shared hidden space,
    sums them element-wise, applies an activation, and maps to vocabulary
    (plus optional extra) logits.

    Args:
        config: Joint network configuration.
        encoder_hidden: Hidden size of the encoder output.
        pred_hidden: Hidden size of the prediction network output.
        vocab_size: Vocabulary size *including* the blank token.
    """

    def __init__(
        self,
        config: ParakeetJointNetworkConfig,
        encoder_hidden: int,
        pred_hidden: int,
        vocab_size: int,
    ):
        super().__init__()
        self.enc_proj = nn.Linear(encoder_hidden, config.joint_hidden)
        self.pred_proj = nn.Linear(pred_hidden, config.joint_hidden)
        self.activation = ACT2FN[config.activation]
        self.dropout = nn.Dropout(p=config.dropout)
        # vocab_size already includes blank; num_extra_outputs adds duration heads for TDT
        self.out_proj = nn.Linear(config.joint_hidden, vocab_size + config.num_extra_outputs)

    def project_encoder(self, encoder_output: torch.Tensor) -> torch.Tensor:
        """Project encoder output ``[B, T, enc_hidden]`` → ``[B, T, joint_hidden]``."""
        return self.enc_proj(encoder_output)

    def project_prednet(self, prednet_output: torch.Tensor) -> torch.Tensor:
        """Project prediction-net output ``[B, U, pred_hidden]`` → ``[B, U, joint_hidden]``."""
        return self.pred_proj(prednet_output)

    def joint_after_projection(
        self,
        f: torch.Tensor,
        g: torch.Tensor,
    ) -> torch.Tensor:
        """
        Combine already-projected encoder and prediction-net outputs.

        Args:
            f: Projected encoder output, shape ``[B, T, joint_hidden]`` or
               ``[B, 1, joint_hidden]`` for a single time-step.
            g: Projected prediction-net output, shape ``[B, U, joint_hidden]``
               or ``[B, 1, joint_hidden]`` for a single label step.

        Returns:
            Logits of shape ``[B, T, U, vocab_size + num_extra_outputs]``.
        """
        f = f.unsqueeze(2)  # [B, T, 1, H]
        g = g.unsqueeze(1)  # [B, 1, U, H]
        out = self.activation(f + g)  # [B, T, U, H]
        out = self.dropout(out)
        return self.out_proj(out)  # [B, T, U, V + extras]

    def forward(
        self,
        encoder_output: torch.Tensor,
        prednet_output: torch.Tensor,
    ) -> torch.Tensor:
        """
        Full joint forward (projects + combines).

        Args:
            encoder_output: ``[B, T, enc_hidden]``
            prednet_output: ``[B, U, pred_hidden]``

        Returns:
            Logits ``[B, T, U, vocab_size + num_extra_outputs]``.
        """
        return self.joint_after_projection(
            self.project_encoder(encoder_output),
            self.project_prednet(prednet_output),
        )


@dataclass
class ParakeetTransducerModelOutput(ModelOutput):
    """
    Outputs for ``ParakeetForRNNT`` and ``ParakeetForTDT``.

    Args:
        loss (`torch.FloatTensor`, *optional*):
            Transducer loss, present when ``labels`` are supplied.
        logits (`torch.FloatTensor` of shape ``[B, T, U, V]``, *optional*):
            Raw joint-network logits.  Only returned when ``return_logits=True``
            to avoid materialising the large ``T × U`` tensor during inference.
        encoder_last_hidden_state (`torch.FloatTensor` of shape ``[B, T, H]``, *optional*):
            Encoder output hidden states.
        encoder_attention_mask (`torch.LongTensor` of shape ``[B, T]``, *optional*):
            Encoder output attention mask after subsampling.
    """

    loss: torch.FloatTensor | None = None
    logits: torch.FloatTensor | None = None
    encoder_last_hidden_state: torch.FloatTensor | None = None
    encoder_attention_mask: torch.LongTensor | None = None


@auto_docstring(
    custom_intro="""
    Parakeet model with an RNNT (RNN-Transducer) head for automatic speech recognition.

    The model is composed of a FastConformer encoder, an LSTM prediction network, and
    a joint network. At inference time, ``generate()`` runs batched frame-looping greedy
    decoding.
    """
)
class ParakeetForRNNT(ParakeetPreTrainedModel):
    config: ParakeetRNNTConfig

    def __init__(self, config: ParakeetRNNTConfig):
        super().__init__(config)
        self.encoder = ParakeetEncoder(config.encoder_config)
        self.prediction_network = ParakeetPredictionNetwork(
            config.prediction_network_config,
            vocab_size=config.vocab_size,
            blank_id=config.blank_id,
        )
        self.joint_network = ParakeetJointNetwork(
            config.joint_network_config,
            encoder_hidden=config.encoder_config.hidden_size,
            pred_hidden=config.prediction_network_config.pred_hidden,
            vocab_size=config.vocab_size,
        )
        self.post_init()

    @auto_docstring
    @can_return_tuple
    def forward(
        self,
        input_features: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        label_lengths: torch.Tensor | None = None,
        return_logits: bool = False,
        **kwargs: Unpack[TransformersKwargs],
    ) -> ParakeetTransducerModelOutput:
        r"""
        labels (`torch.LongTensor` of shape ``[B, U]``, *optional*):
            Target token ids for computing the RNNT loss.  Padded positions
            should be filled with ``config.blank_id``.
        label_lengths (`torch.LongTensor` of shape ``[B]``, *optional*):
            Number of valid tokens in each row of ``labels``.  Required when
            ``labels`` is provided.
        return_logits (`bool`, *optional*, defaults to ``False``):
            Whether to return the joint-network logits tensor ``[B, T, U, V]``.
            Materialising this tensor is memory-intensive at training time.

        Example:

        ```python
        >>> from transformers import AutoProcessor, ParakeetForRNNT
        >>> from datasets import load_dataset, Audio

        >>> model_id = "nvidia/parakeet-rnnt-1.1b"
        >>> processor = AutoProcessor.from_pretrained(model_id)
        >>> model = ParakeetForRNNT.from_pretrained(model_id)

        >>> ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
        >>> ds = ds.cast_column("audio", Audio(sampling_rate=processor.feature_extractor.sampling_rate))

        >>> inputs = processor(ds[0]["audio"]["array"])
        >>> transcription = model.generate(**inputs)
        >>> print(processor.batch_decode(transcription))
        ```
        """
        encoder_outputs = self.encoder(
            input_features=input_features,
            attention_mask=attention_mask,
            **kwargs,
        )
        encoder_hidden = encoder_outputs.last_hidden_state  # [B, T, enc_hidden]

        loss = None
        logits = None

        if labels is not None:
            if label_lengths is None:
                raise ValueError("`label_lengths` must be provided when `labels` is supplied.")

            pred_output, _ = self.prediction_network(labels, label_lengths)  # [B, U+1, pred_hidden]
            logits = self.joint_network(encoder_hidden, pred_output)  # [B, T, U+1, V]

            input_lengths = self._get_subsampling_output_length(
                attention_mask.sum(-1) if attention_mask is not None else torch.full(
                    (input_features.size(0),), input_features.size(1),
                    dtype=torch.long, device=input_features.device,
                )
            )
            loss_fn = ParakeetRNNTLoss(blank=self.config.blank_id, reduction=self.config.rnnt_loss_reduction)
            loss = loss_fn(logits, labels, input_lengths, label_lengths)

        if not return_logits:
            logits = None

        return ParakeetTransducerModelOutput(
            loss=loss,
            logits=logits,
            encoder_last_hidden_state=encoder_hidden,
            encoder_attention_mask=encoder_outputs.attention_mask,
        )

    def _pred_step(
        self,
        last_label: torch.Tensor | int,
        state: tuple[torch.Tensor, torch.Tensor] | None,
        batch_size: int,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """One prediction-network step for the greedy decoder.

        Args:
            last_label: Either a scalar SOS sentinel (``self.config.blank_id``)
                for the very first step, or a ``[B, 1]`` long tensor of the
                most recently emitted label for each item in the batch.
            state: Current LSTM state or ``None``.
            batch_size: Batch size (needed when ``last_label`` is a scalar).

        Returns:
            Tuple of projected prediction-network output ``[B, 1, pred_hidden]``
            and updated LSTM state.
        """
        if isinstance(last_label, int):
            # SOS: emit zero vector
            y = None
        else:
            y = last_label  # [B, 1]
        return self.prediction_network.predict(y, state, batch_size=batch_size)

    def _joint_step(
        self,
        f: torch.Tensor,
        g: torch.Tensor,
        log_normalize: bool | None = None,
    ) -> torch.Tensor:
        """Single joint-network step.

        Args:
            f: Encoder frame, shape ``[B, 1, enc_hidden]``.
            g: Prediction-network output, shape ``[B, 1, pred_hidden]``.
            log_normalize: Whether to apply log-softmax.  ``None`` means apply
                only on CPU (matching NeMo behaviour).

        Returns:
            Logits or log-probs of shape ``[B, 1, 1, V]``.
        """
        out = self.joint_network(f, g)  # [B, 1, 1, V]
        should_normalize = (log_normalize is True) or (log_normalize is None and not out.is_cuda)
        if should_normalize:
            out = torch.log_softmax(out, dim=-1)
        return out

    @torch.no_grad()
    def generate(
        self,
        input_features: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        return_dict_in_generate: bool = False,
        **kwargs: Unpack[TransformersKwargs],
    ) -> ParakeetGenerateOutput | torch.LongTensor:
        r"""
        Greedy frame-looping RNNT decoding.

        Example:

        ```python
        >>> from transformers import AutoProcessor, ParakeetForRNNT
        >>> from datasets import load_dataset, Audio

        >>> model_id = "nvidia/parakeet-rnnt-1.1b"
        >>> processor = AutoProcessor.from_pretrained(model_id)
        >>> model = ParakeetForRNNT.from_pretrained(model_id)

        >>> ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
        >>> ds = ds.cast_column("audio", Audio(sampling_rate=processor.feature_extractor.sampling_rate))

        >>> inputs = processor(ds[0]["audio"]["array"])
        >>> transcription = model.generate(**inputs)
        >>> print(processor.batch_decode(transcription))
        ```
        """
        kwargs["return_dict"] = True
        outputs: ParakeetTransducerModelOutput = self.forward(
            input_features=input_features,
            attention_mask=attention_mask,
            **kwargs,
        )
        encoder_hidden = outputs.encoder_last_hidden_state  # [B, T, H]

        if attention_mask is not None:
            encoder_lengths = self._get_subsampling_output_length(attention_mask.sum(-1))
        else:
            encoder_lengths = torch.full(
                (encoder_hidden.size(0),),
                encoder_hidden.size(1),
                dtype=torch.long,
                device=encoder_hidden.device,
            )

        hypotheses = self._greedy_decode(encoder_hidden, encoder_lengths)
        sequences = self._hypotheses_to_tensor(hypotheses, encoder_hidden.device)

        if return_dict_in_generate:
            return ParakeetGenerateOutput(sequences=sequences)
        return sequences

    def _greedy_decode(
        self,
        encoder_output: torch.Tensor,
        encoder_output_length: torch.Tensor,
    ) -> list[list[int]]:
        """Batched frame-looping greedy RNNT decoding.

        Args:
            encoder_output: ``[B, T, enc_hidden]``
            encoder_output_length: ``[B]`` valid frame counts.

        Returns:
            List of B token-id lists (blank tokens excluded).
        """
        B, _T, _H = encoder_output.shape
        device = encoder_output.device
        dtype = encoder_output.dtype

        # Per-sample token sequences and scores
        y_sequences: list[list[int]] = [[] for _ in range(B)]

        state = self.prediction_network.initialize_state(B, device, dtype)
        # last_label[b] = most recently emitted non-blank token id for sample b
        last_label = torch.full((B, 1), fill_value=self.config.blank_id, dtype=torch.long, device=device)
        blank_mask = torch.zeros(B, dtype=torch.bool, device=device)

        max_out_len = int(encoder_output_length.max().item())

        # --- outer loop: one pass per encoder frame ---
        for time_idx in range(max_out_len):
            f = encoder_output[:, time_idx : time_idx + 1, :]  # [B, 1, H]

            symbols_added = 0
            blank_mask.fill_(False)
            # samples whose valid length is <= time_idx are already done
            blank_mask |= time_idx >= encoder_output_length

            # --- inner loop: emit non-blank tokens until blank or max_symbols ---
            not_blank = True
            while not_blank and symbols_added < self.config.max_symbols_per_step:
                if time_idx == 0 and symbols_added == 0 and all(
                    state[0].abs().sum() == 0 for _ in [None]
                ):
                    g, state_prime = self._pred_step(self.config.blank_id, None, batch_size=B)
                else:
                    g, state_prime = self._pred_step(last_label, state, batch_size=B)

                # logp: [B, V]  (squeeze T and U dimensions)
                logp = self._joint_step(f, g, log_normalize=None)[:, 0, 0, :]
                if logp.dtype != torch.float32:
                    logp = logp.float()

                # greedy pick
                scores, k = logp.max(dim=1)  # [B]

                k_is_blank = k == self.config.blank_id
                blank_mask |= k_is_blank

                if blank_mask.all():
                    not_blank = False
                else:
                    # Update LSTM state only for samples that emitted a non-blank token
                    not_blank_mask = ~blank_mask  # [B]
                    # batch_replace_states_mask: in-place masked update (CUDA-graph friendly)
                    torch.where(
                        not_blank_mask.unsqueeze(0).unsqueeze(-1),  # [1, B, 1]
                        state_prime[0],
                        state[0],
                        out=state[0],
                    )
                    torch.where(
                        not_blank_mask.unsqueeze(0).unsqueeze(-1),
                        state_prime[1],
                        state[1],
                        out=state[1],
                    )

                    # Update last_label for non-blank samples; keep previous for blank samples
                    k_masked = torch.where(blank_mask, last_label.squeeze(1), k)
                    last_label = k_masked.unsqueeze(1)

                    # Record emitted non-blank tokens
                    for b in range(B):
                        if not blank_mask[b]:
                            y_sequences[b].append(int(k[b].item()))

                    symbols_added += 1

        return y_sequences

    def _hypotheses_to_tensor(
        self,
        hypotheses: list[list[int]],
        device: torch.device,
    ) -> torch.LongTensor:
        """Pack variable-length hypotheses into a right-padded tensor."""
        max_len = max((len(h) for h in hypotheses), default=0)
        if max_len == 0:
            return torch.zeros(len(hypotheses), 1, dtype=torch.long, device=device)
        out = torch.full(
            (len(hypotheses), max_len),
            fill_value=self.config.blank_id,
            dtype=torch.long,
            device=device,
        )
        for b, hyp in enumerate(hypotheses):
            if hyp:
                out[b, : len(hyp)] = torch.tensor(hyp, dtype=torch.long, device=device)
        return out


@auto_docstring(
    custom_intro="""
    Parakeet model with a TDT (Token-and-Duration Transducer) head for automatic
    speech recognition.

    TDT extends RNNT by predicting both the emitted token *and* the number of
    encoder frames to skip (the *duration*) at each decoding step.  The joint
    network output is split into ``vocab_size`` label logits and
    ``len(durations)`` duration logits.
    """
)
class ParakeetForTDT(ParakeetForRNNT):
    config: ParakeetTDTConfig

    def __init__(self, config: ParakeetTDTConfig):
        super().__init__(config)
        # Joint network is already constructed by the parent with the correct
        # num_extra_outputs (set to len(durations) in ParakeetTDTConfig.__post_init__)

    @auto_docstring
    @can_return_tuple
    def forward(
        self,
        input_features: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        label_lengths: torch.Tensor | None = None,
        return_logits: bool = False,
        **kwargs: Unpack[TransformersKwargs],
    ) -> ParakeetTransducerModelOutput:
        r"""
        labels (`torch.LongTensor` of shape ``[B, U]``, *optional*):
            Target token ids for computing the TDT loss.
        label_lengths (`torch.LongTensor` of shape ``[B]``, *optional*):
            Number of valid tokens in each row of ``labels``.
        return_logits (`bool`, *optional*, defaults to ``False``):
            Whether to return the joint-network logits tensor.

        Example:

        ```python
        >>> from transformers import AutoProcessor, ParakeetForTDT
        >>> from datasets import load_dataset, Audio

        >>> model_id = "nvidia/parakeet-tdt-1.1b"
        >>> processor = AutoProcessor.from_pretrained(model_id)
        >>> model = ParakeetForTDT.from_pretrained(model_id)

        >>> ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
        >>> ds = ds.cast_column("audio", Audio(sampling_rate=processor.feature_extractor.sampling_rate))

        >>> inputs = processor(ds[0]["audio"]["array"])
        >>> transcription = model.generate(**inputs)
        >>> print(processor.batch_decode(transcription))
        ```
        """
        encoder_outputs = self.encoder(
            input_features=input_features,
            attention_mask=attention_mask,
            **kwargs,
        )
        encoder_hidden = encoder_outputs.last_hidden_state

        loss = None
        logits = None

        if labels is not None:
            if label_lengths is None:
                raise ValueError("`label_lengths` must be provided when `labels` is supplied.")

            pred_output, _ = self.prediction_network(labels, label_lengths)
            logits = self.joint_network(encoder_hidden, pred_output)

            input_lengths = self._get_subsampling_output_length(
                attention_mask.sum(-1) if attention_mask is not None else torch.full(
                    (input_features.size(0),), input_features.size(1),
                    dtype=torch.long, device=input_features.device,
                )
            )
            loss_fn = ParakeetTDTLoss(
                blank=self.config.blank_id,
                durations=self.config.durations,
                reduction=self.config.rnnt_loss_reduction,
                sigma=self.config.sigma,
                omega=self.config.omega,
            )
            loss = loss_fn(logits, labels, input_lengths, label_lengths)

        if not return_logits:
            logits = None

        return ParakeetTransducerModelOutput(
            loss=loss,
            logits=logits,
            encoder_last_hidden_state=encoder_hidden,
            encoder_attention_mask=encoder_outputs.attention_mask,
        )

    def _greedy_decode(
        self,
        encoder_output: torch.Tensor,
        encoder_output_length: torch.Tensor,
    ) -> list[list[int]]:
        """Batched frame-looping greedy TDT decoding with duration prediction.

        For non-blank emissions the duration is always 0 (stay at the same
        frame and continue the inner loop).  For blank emissions the duration
        gives the number of frames to skip before the next outer-loop step.
        """
        B, _T, _H = encoder_output.shape
        device = encoder_output.device
        dtype = encoder_output.dtype
        n_dur = len(self.config.durations)
        durations_tensor = torch.tensor(self.config.durations, dtype=torch.long, device=device)

        y_sequences: list[list[int]] = [[] for _ in range(B)]

        state = self.prediction_network.initialize_state(B, device, dtype)
        last_label = torch.full((B, 1), fill_value=self.config.blank_id, dtype=torch.long, device=device)

        # time_indices[b] = current frame position for each sample in the batch
        time_indices = torch.zeros(B, dtype=torch.long, device=device)
        last_valid = torch.clamp(encoder_output_length - 1, min=0)  # [B]

        while (time_indices < encoder_output_length).any():
            # Clamp to avoid out-of-bounds gather
            safe_t = torch.minimum(time_indices, last_valid)
            f = encoder_output[torch.arange(B, device=device), safe_t, :].unsqueeze(1)  # [B, 1, H]

            symbols_added = 0
            blank_mask = time_indices >= encoder_output_length  # [B]
            frame_done = blank_mask.clone()

            while not frame_done.all() and symbols_added < self.config.max_symbols_per_step:
                g, state_prime = self._pred_step(last_label, state, batch_size=B)

                joint_out = self._joint_step(f, g, log_normalize=None)[:, 0, 0, :]  # [B, V+D]
                if joint_out.dtype != torch.float32:
                    joint_out = joint_out.float()

                label_logits = joint_out[:, :-n_dur]  # [B, V]
                dur_logits = joint_out[:, -n_dur:]    # [B, D]

                _, k = label_logits.max(dim=1)        # [B]
                dur_idx = dur_logits.argmax(dim=1)    # [B]
                predicted_dur = durations_tensor[dur_idx]  # [B]

                k_is_blank = k == self.config.blank_id
                blank_mask |= k_is_blank

                # Blanks with duration 0 must advance by at least 1 to avoid infinite loop
                safe_dur = torch.where(k_is_blank & (predicted_dur == 0), torch.ones_like(predicted_dur), predicted_dur)

                not_blank_mask = ~blank_mask
                # Update LSTM state for non-blank samples only (CUDA-graph–friendly masked write)
                torch.where(
                    not_blank_mask.unsqueeze(0).unsqueeze(-1),
                    state_prime[0], state[0], out=state[0],
                )
                torch.where(
                    not_blank_mask.unsqueeze(0).unsqueeze(-1),
                    state_prime[1], state[1], out=state[1],
                )

                k_masked = torch.where(blank_mask, last_label.squeeze(1), k)
                last_label = k_masked.unsqueeze(1)

                for b in range(B):
                    if not blank_mask[b]:
                        y_sequences[b].append(int(k[b].item()))

                # Advance time indices for samples that emitted blank
                advance = k_is_blank & ~frame_done
                time_indices = torch.where(advance, time_indices + safe_dur, time_indices)
                frame_done |= blank_mask

                symbols_added += 1

            # After inner loop: samples that never went blank still advance by 1 via duration
            # (handled above via safe_dur; but if loop exited due to max_symbols, force advance)
            still_active = ~frame_done & (time_indices < encoder_output_length)
            time_indices = torch.where(still_active, time_indices + 1, time_indices)

        return y_sequences


__all__ = [
    "ParakeetForCTC",
    "ParakeetForRNNT",
    "ParakeetForTDT",
    "ParakeetEncoder",
    "ParakeetJointNetwork",
    "ParakeetPredictionNetwork",
    "ParakeetPreTrainedModel",
    "ParakeetRNNTLoss",
    "ParakeetTDTLoss",
]
