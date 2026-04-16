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
"""Parakeet model configuration."""

from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring


@auto_docstring(checkpoint="nvidia/parakeet-ctc-1.1b")
@strict
class ParakeetEncoderConfig(PreTrainedConfig):
    r"""
    convolution_bias (`bool`, *optional*, defaults to `True`):
        Whether to use bias in convolutions of the conformer's convolution module.
    conv_kernel_size (`int`, *optional*, defaults to 9):
        The kernel size of the convolution layers in the Conformer block.
    subsampling_factor (`int`, *optional*, defaults to 8):
        The factor by which the input sequence is subsampled.
    subsampling_conv_channels (`int`, *optional*, defaults to 256):
        The number of channels in the subsampling convolution layers.
    num_mel_bins (`int`, *optional*, defaults to 80):
        Number of mel features.
    subsampling_conv_kernel_size (`int`, *optional*, defaults to 3):
        The kernel size of the subsampling convolution layers.
    subsampling_conv_stride (`int`, *optional*, defaults to 2):
        The stride of the subsampling convolution layers.
    dropout_positions (`float`, *optional*, defaults to 0.0):
        The dropout ratio for the positions in the input sequence.
    scale_input (`bool`, *optional*, defaults to `True`):
        Whether to scale the input embeddings.

    Example:
        ```python
        >>> from transformers import ParakeetEncoderModel, ParakeetEncoderConfig

        >>> # Initializing a `ParakeetEncoder` configuration
        >>> configuration = ParakeetEncoderConfig()

        >>> # Initializing a model from the configuration
        >>> model = ParakeetEncoderModel(configuration)

        >>> # Accessing the model configuration
        >>> configuration = model.config
        ```

    This configuration class is based on the ParakeetEncoder architecture from NVIDIA NeMo. You can find more details
    and pre-trained models at [nvidia/parakeet-ctc-1.1b](https://huggingface.co/nvidia/parakeet-ctc-1.1b).
    """

    model_type = "parakeet_encoder"
    keys_to_ignore_at_inference = ["past_key_values"]

    hidden_size: int = 1024
    num_hidden_layers: int = 24
    num_attention_heads: int = 8
    intermediate_size: int = 4096
    hidden_act: str = "silu"
    attention_bias: bool = True
    convolution_bias: bool = True
    conv_kernel_size: int = 9
    subsampling_factor: int = 8
    subsampling_conv_channels: int = 256
    num_mel_bins: int = 80
    subsampling_conv_kernel_size: int = 3
    subsampling_conv_stride: int = 2
    dropout: float | int = 0.1
    dropout_positions: float | int = 0.0
    layerdrop: float | int = 0.1
    activation_dropout: float | int = 0.1
    attention_dropout: float | int = 0.1
    max_position_embeddings: int = 5000
    scale_input: bool = True
    initializer_range: float = 0.02

    def __post_init__(self, **kwargs):
        self.num_key_value_heads = self.num_attention_heads
        super().__post_init__(**kwargs)


@auto_docstring(checkpoint="nvidia/parakeet-ctc-1.1b")
@strict
class ParakeetCTCConfig(PreTrainedConfig):
    r"""
    ctc_loss_reduction (`str`, *optional*, defaults to `"mean"`):
        Specifies the reduction to apply to the output of `torch.nn.CTCLoss`. Only relevant when training an
        instance of [`ParakeetForCTC`].
    ctc_zero_infinity (`bool`, *optional*, defaults to `True`):
        Whether to zero infinite losses and the associated gradients of `torch.nn.CTCLoss`. Infinite losses mainly
        occur when the inputs are too short to be aligned to the targets. Only relevant when training an instance
        of [`ParakeetForCTC`].
    encoder_config (`Union[dict, ParakeetEncoderConfig]`, *optional*):
        The config object or dictionary of the encoder.

    Example:

    ```python
    >>> from transformers import ParakeetForCTC, ParakeetCTCConfig
    >>> # Initializing a Parakeet configuration
    >>> configuration = ParakeetCTCConfig()
    >>> # Initializing a model from the configuration
    >>> model = ParakeetForCTC(configuration)
    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```
    """

    model_type = "parakeet_ctc"
    sub_configs = {"encoder_config": ParakeetEncoderConfig}

    vocab_size: int = 1025
    ctc_loss_reduction: str = "mean"
    ctc_zero_infinity: bool = True
    encoder_config: dict | PreTrainedConfig | None = None
    pad_token_id: int | None = 1024

    def __post_init__(self, **kwargs):
        if isinstance(self.encoder_config, dict):
            self.encoder_config = ParakeetEncoderConfig(**self.encoder_config)
        elif self.encoder_config is None:
            self.encoder_config = ParakeetEncoderConfig()
        self.initializer_range = self.encoder_config.initializer_range
        super().__post_init__(**kwargs)


@auto_docstring
@strict
class ParakeetPredictionNetworkConfig(PreTrainedConfig):
    r"""
    Configuration class for the LSTM-based prediction network used in
    [`ParakeetForRNNT`] and [`ParakeetForTDT`].

    pred_hidden (`int`, *optional*, defaults to 640):
        Hidden dimension of the LSTM layers and the output projection.
    pred_rnn_layers (`int`, *optional*, defaults to 1):
        Number of stacked LSTM layers.
    dropout (`float`, *optional*, defaults to 0.0):
        Dropout probability applied to the LSTM output.
    blank_as_pad (`bool`, *optional*, defaults to `True`):
        When `True` the blank token embedding is tied to the zero vector via
        ``padding_idx``.  This eliminates a special-case branch in the decode
        loop and is a prerequisite for CUDA-graph capture in future releases.
    """

    model_type = "parakeet_prediction_network"

    pred_hidden: int = 640
    pred_rnn_layers: int = 1
    dropout: float = 0.0
    blank_as_pad: bool = True

    def __post_init__(self, **kwargs):
        super().__post_init__(**kwargs)


@auto_docstring
@strict
class ParakeetJointNetworkConfig(PreTrainedConfig):
    r"""
    Configuration class for the joint network used in [`ParakeetForRNNT`] and
    [`ParakeetForTDT`].

    The joint network projects the encoder and prediction-network outputs into
    a shared hidden space, combines them, and produces per-frame per-label
    logits.

    joint_hidden (`int`, *optional*, defaults to 640):
        Width of the shared hidden layer that combines encoder and prediction
        network projections.
    activation (`str`, *optional*, defaults to `"relu"`):
        Activation function applied after the element-wise sum of projections.
        Must be a key accepted by ``transformers.activations.ACT2FN``.
    dropout (`float`, *optional*, defaults to 0.0):
        Dropout probability applied inside the joint network before the output
        projection.
    num_extra_outputs (`int`, *optional*, defaults to 0):
        Number of additional output heads appended after the vocabulary logits.
        Set to 0 for standard RNNT.  For TDT this is set automatically to
        ``len(durations)`` by [`ParakeetTDTConfig`] — do not set it manually
        when using TDT.
    """

    model_type = "parakeet_joint_network"

    joint_hidden: int = 640
    activation: str = "relu"
    dropout: float = 0.0
    num_extra_outputs: int = 0

    def __post_init__(self, **kwargs):
        super().__post_init__(**kwargs)


@auto_docstring
@strict
class ParakeetRNNTConfig(PreTrainedConfig):
    r"""
    Configuration class for a Parakeet RNNT (RNN-Transducer) model composed of
    a FastConformer encoder, an LSTM prediction network, and a joint network.

    encoder_config (`dict` or [`ParakeetEncoderConfig`], *optional*):
        Configuration for the FastConformer encoder.  When ``None`` a default
        [`ParakeetEncoderConfig`] is created.
    prediction_network_config (`dict` or [`ParakeetPredictionNetworkConfig`], *optional*):
        Configuration for the LSTM prediction network.  When ``None`` a default
        [`ParakeetPredictionNetworkConfig`] is created.
    joint_network_config (`dict` or [`ParakeetJointNetworkConfig`], *optional*):
        Configuration for the joint network.  When ``None`` a default
        [`ParakeetJointNetworkConfig`] is created.
    vocab_size (`int`, *optional*, defaults to 1025):
        Size of the vocabulary, including the blank token.
    blank_id (`int`, *optional*, defaults to 1024):
        Index of the CTC/RNNT blank token.  Must satisfy
        ``blank_id == vocab_size - 1``.
    rnnt_loss_reduction (`str`, *optional*, defaults to ``"mean_batch"``):
        Reduction applied to the per-sample RNNT loss values.  Accepted values:
        ``"mean_batch"`` (mean over samples), ``"mean"`` (mean over symbols),
        ``"sum"``, ``"mean_volume"`` (mean over total symbols in batch).
    max_symbols_per_step (`int`, *optional*, defaults to 10):
        Maximum number of non-blank symbols the decoder may emit at a single
        encoder time-step during greedy decoding.  This bounds the inner
        while-loop and is also used as the fixed unroll depth for CUDA-graph
        capture.

    Example:

    ```python
    >>> from transformers import ParakeetForRNNT, ParakeetRNNTConfig

    >>> configuration = ParakeetRNNTConfig()
    >>> model = ParakeetForRNNT(configuration)
    >>> configuration = model.config
    ```
    """

    model_type = "parakeet_rnnt"
    sub_configs = {
        "encoder_config": ParakeetEncoderConfig,
        "prediction_network_config": ParakeetPredictionNetworkConfig,
        "joint_network_config": ParakeetJointNetworkConfig,
    }

    vocab_size: int = 1025
    blank_id: int = 1024
    rnnt_loss_reduction: str = "mean_batch"
    max_symbols_per_step: int = 10
    encoder_config: dict | PreTrainedConfig | None = None
    prediction_network_config: dict | PreTrainedConfig | None = None
    joint_network_config: dict | PreTrainedConfig | None = None

    def __post_init__(self, **kwargs):
        if isinstance(self.encoder_config, dict):
            self.encoder_config = ParakeetEncoderConfig(**self.encoder_config)
        elif self.encoder_config is None:
            self.encoder_config = ParakeetEncoderConfig()

        if isinstance(self.prediction_network_config, dict):
            self.prediction_network_config = ParakeetPredictionNetworkConfig(
                **self.prediction_network_config
            )
        elif self.prediction_network_config is None:
            self.prediction_network_config = ParakeetPredictionNetworkConfig()

        if isinstance(self.joint_network_config, dict):
            self.joint_network_config = ParakeetJointNetworkConfig(**self.joint_network_config)
        elif self.joint_network_config is None:
            self.joint_network_config = ParakeetJointNetworkConfig()

        self.initializer_range = self.encoder_config.initializer_range
        super().__post_init__(**kwargs)


@auto_docstring
@strict
class ParakeetTDTConfig(ParakeetRNNTConfig):
    r"""
    Configuration class for a Parakeet TDT (Token-and-Duration Transducer) model.

    TDT extends RNNT by predicting both the emitted token *and* the number of
    encoder frames to skip (the *duration*) at each decoding step.  The joint
    network output is split into ``vocab_size`` label logits followed by
    ``len(durations)`` duration logits; ``joint_network_config.num_extra_outputs``
    is set automatically by this config and must not be overridden manually.

    encoder_config (`dict` or [`ParakeetEncoderConfig`], *optional*):
        Configuration for the FastConformer encoder.  Inherited from
        [`ParakeetRNNTConfig`].
    prediction_network_config (`dict` or [`ParakeetPredictionNetworkConfig`], *optional*):
        Configuration for the LSTM prediction network.  Inherited from
        [`ParakeetRNNTConfig`].
    joint_network_config (`dict` or [`ParakeetJointNetworkConfig`], *optional*):
        Configuration for the joint network.  Inherited from
        [`ParakeetRNNTConfig`].  ``num_extra_outputs`` is set automatically
        to ``len(durations)`` during ``__post_init__``.
    blank_id (`int`, *optional*, defaults to 1024):
        Index of the RNNT blank token.  Inherited from [`ParakeetRNNTConfig`].
    rnnt_loss_reduction (`str`, *optional*, defaults to ``"mean_batch"``):
        Reduction applied to per-sample RNNT loss values.  Inherited from
        [`ParakeetRNNTConfig`].
    max_symbols_per_step (`int`, *optional*, defaults to 10):
        Maximum non-blank symbols emitted at a single encoder frame during
        greedy decoding.  Inherited from [`ParakeetRNNTConfig`].
    durations (`list[int]`, *optional*, defaults to ``[0, 1, 2, 3, 4]``):
        Ordered list of candidate frame-skip values.  ``0`` corresponds to
        emitting a token *without* advancing the encoder time index (used for
        non-blank emissions); positive values advance by that many frames on a
        blank emission.  The list must contain ``0`` and at least one positive
        integer.
    sigma (`float`, *optional*, defaults to 0.05):
        Log-domain under-normalisation coefficient applied to label logits
        before computing the TDT loss, as described in
        `Efficient Sequence Transduction by Jointly Predicting Tokens and
        Durations <https://arxiv.org/abs/2304.06795>`__.
    omega (`float`, *optional*, defaults to 0.1):
        Weight of the standard RNNT loss term in the combined TDT loss:
        ``loss = omega * rnnt_loss + (1 - omega) * duration_loss``.

    Example:

    ```python
    >>> from transformers import ParakeetForTDT, ParakeetTDTConfig

    >>> configuration = ParakeetTDTConfig()
    >>> model = ParakeetForTDT(configuration)
    >>> configuration = model.config
    ```
    """

    model_type = "parakeet_tdt"

    durations: list[int] | None = None
    sigma: float = 0.05
    omega: float = 0.1

    def __post_init__(self, **kwargs):
        super().__post_init__(**kwargs)
        if self.durations is None:
            self.durations = [0, 1, 2, 3, 4]
        # Keep the joint network output width in sync with the duration count.
        # This must happen after super().__post_init__() has instantiated the
        # joint_network_config sub-config object.
        self.joint_network_config.num_extra_outputs = len(self.durations)


__all__ = [
    "ParakeetCTCConfig",
    "ParakeetEncoderConfig",
    "ParakeetJointNetworkConfig",
    "ParakeetPredictionNetworkConfig",
    "ParakeetRNNTConfig",
    "ParakeetTDTConfig",
]
