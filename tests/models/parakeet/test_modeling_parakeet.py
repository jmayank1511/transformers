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
"""Testing suite for the PyTorch Parakeet model."""

import json
import tempfile
import unittest
from pathlib import Path

from transformers import is_datasets_available, is_torch_available
from transformers.testing_utils import cleanup, require_torch, slow, torch_device

from ...test_configuration_common import ConfigTester
from ...test_modeling_common import ModelTesterMixin, floats_tensor, ids_tensor, random_attention_mask


if is_datasets_available():
    from datasets import Audio, load_dataset

if is_torch_available():
    import torch

    from transformers import (
        AutoProcessor,
        ParakeetCTCConfig,
        ParakeetEncoder,
        ParakeetEncoderConfig,
        ParakeetForCTC,
    )
    from transformers.models.parakeet.modeling_parakeet import ParakeetCTCModelOutput


class ParakeetEncoderModelTester:
    def __init__(
        self,
        parent,
        batch_size=13,
        seq_length=1024,
        is_training=True,
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=256,
        hidden_act="silu",
        dropout=0.0,  # so gradient checkpointing doesn't fail
        conv_kernel_size=9,
        subsampling_factor=8,
        subsampling_conv_channels=32,
        use_bias=True,
        num_mel_bins=80,
        scale_input=True,
    ):
        # testing suite parameters
        self.parent = parent
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.num_mel_bins = num_mel_bins
        self.is_training = is_training

        # config parameters
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.dropout = dropout
        self.conv_kernel_size = conv_kernel_size
        self.subsampling_factor = subsampling_factor
        self.subsampling_conv_channels = subsampling_conv_channels
        self.use_bias = use_bias
        self.num_mel_bins = num_mel_bins
        self.scale_input = scale_input

        # Calculate output sequence length after subsampling
        self.output_seq_length = seq_length // subsampling_factor
        self.encoder_seq_length = self.output_seq_length
        self.key_length = self.output_seq_length

    def prepare_config_and_inputs(self):
        input_features = floats_tensor([self.batch_size, self.seq_length, self.num_mel_bins])
        attention_mask = random_attention_mask([self.batch_size, self.seq_length])
        config = self.get_config()

        return config, input_features, attention_mask

    def get_config(self):
        return ParakeetEncoderConfig(
            hidden_size=self.hidden_size,
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=self.num_attention_heads,
            intermediate_size=self.intermediate_size,
            hidden_act=self.hidden_act,
            dropout=self.dropout,
            dropout_positions=self.dropout,
            layerdrop=self.dropout,
            activation_dropout=self.dropout,
            attention_dropout=self.dropout,
            conv_kernel_size=self.conv_kernel_size,
            subsampling_factor=self.subsampling_factor,
            subsampling_conv_channels=self.subsampling_conv_channels,
            use_bias=self.use_bias,
            num_mel_bins=self.num_mel_bins,
            scale_input=self.scale_input,
        )

    def create_and_check_model(self, config, input_features, attention_mask):
        model = ParakeetEncoder(config=config)
        model.to(torch_device)
        model.eval()
        with torch.no_grad():
            result = model(input_features, attention_mask=attention_mask)

        self.parent.assertEqual(
            result.last_hidden_state.shape, (self.batch_size, self.output_seq_length, config.hidden_size)
        )

    def prepare_config_and_inputs_for_common(self):
        config, input_features, attention_mask = self.prepare_config_and_inputs()
        inputs_dict = {
            "input_features": input_features,
            "attention_mask": attention_mask,
        }
        return config, inputs_dict

    def check_ctc_loss(self, config, input_values, *args):
        model = ParakeetForCTC(config=config)
        model.to(torch_device)

        # make sure that dropout is disabled
        model.eval()

        input_values = input_values[:3]
        attention_mask = torch.ones(input_values.shape, device=torch_device, dtype=torch.long)

        input_lengths = [input_values.shape[-1] // i for i in [4, 2, 1]]
        max_length_labels = model._get_feat_extract_output_lengths(torch.tensor(input_lengths))
        labels = ids_tensor((input_values.shape[0], min(max_length_labels) - 1), model.config.vocab_size)

        # pad input
        for i in range(len(input_lengths)):
            input_values[i, input_lengths[i] :] = 0.0
            attention_mask[i, input_lengths[i] :] = 0

        model.config.ctc_loss_reduction = "sum"
        sum_loss = model(input_values, attention_mask=attention_mask, labels=labels).loss.item()

        model.config.ctc_loss_reduction = "mean"
        mean_loss = model(input_values, attention_mask=attention_mask, labels=labels).loss.item()

        self.parent.assertTrue(isinstance(sum_loss, float))
        self.parent.assertTrue(isinstance(mean_loss, float))


@require_torch
class ParakeetEncoderModelTest(ModelTesterMixin, unittest.TestCase):
    all_model_classes = (ParakeetEncoder,) if is_torch_available() else ()

    test_resize_embeddings = False

    def setUp(self):
        self.model_tester = ParakeetEncoderModelTester(self)
        self.config_tester = ConfigTester(self, config_class=ParakeetEncoderConfig, has_text_modality=False)

    def test_config(self):
        self.config_tester.run_common_tests()

    def test_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model(*config_and_inputs)

    @unittest.skip(reason="ParakeetEncoder does not use inputs_embeds")
    def test_model_get_set_embeddings(self):
        pass


class ParakeetForCTCModelTester:
    def __init__(self, parent, encoder_kwargs=None, is_training=True, vocab_size=128, pad_token_id=0):
        if encoder_kwargs is None:
            encoder_kwargs = {}

        self.parent = parent
        self.encoder_model_tester = ParakeetEncoderModelTester(parent, **encoder_kwargs)
        self.is_training = is_training

        self.batch_size = self.encoder_model_tester.batch_size
        self.output_seq_length = self.encoder_model_tester.output_seq_length
        self.num_hidden_layers = self.encoder_model_tester.num_hidden_layers
        self.seq_length = vocab_size
        self.hidden_size = self.encoder_model_tester.hidden_size

        self.vocab_size = vocab_size
        self.pad_token_id = pad_token_id

    def prepare_config_and_inputs(self):
        _, input_features, attention_mask = self.encoder_model_tester.prepare_config_and_inputs()
        config = self.get_config()
        return config, input_features, attention_mask

    def get_config(self):
        return ParakeetCTCConfig(
            encoder_config=self.encoder_model_tester.get_config(),
            vocab_size=self.vocab_size,
            pad_token_id=self.pad_token_id,
        )

    def create_and_check_model(self, config, input_features, attention_mask):
        model = ParakeetForCTC(config=config)
        model.to(torch_device)
        model.eval()
        with torch.no_grad():
            result = model(input_features, attention_mask=attention_mask)
        self.parent.assertEqual(result.logits.shape, (self.batch_size, self.output_seq_length, self.vocab_size))

    def prepare_config_and_inputs_for_common(self):
        config, input_features, attention_mask = self.prepare_config_and_inputs()
        inputs_dict = {
            "input_features": input_features,
            "attention_mask": attention_mask,
        }
        return config, inputs_dict

    def test_ctc_loss_inference(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.encoder_model_tester.check_ctc_loss(*config_and_inputs)


@require_torch
class ParakeetForCTCModelTest(ModelTesterMixin, unittest.TestCase):
    all_model_classes = (ParakeetForCTC,) if is_torch_available() else ()
    pipeline_model_mapping = (
        {
            "feature-extraction": ParakeetEncoder,
            "automatic-speech-recognition": ParakeetForCTC,
        }
        if is_torch_available()
        else {}
    )

    test_attention_outputs = False

    test_resize_embeddings = False

    _is_composite = True

    def setUp(self):
        self.model_tester = ParakeetForCTCModelTester(self)
        self.config_tester = ConfigTester(self, config_class=ParakeetCTCConfig)

    def test_config(self):
        self.config_tester.run_common_tests()

    def test_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model(*config_and_inputs)

    @unittest.skip(reason="ParakeetEncoder does not use inputs_embeds")
    def test_model_get_set_embeddings(self):
        pass

    # Original function assumes vision+text model, so overwrite since Parakeet is audio+text
    # Below is modified from `tests/models/granite_speech/test_modeling_granite_speech.py`
    def test_sdpa_can_dispatch_composite_models(self):
        if not self.has_attentions:
            self.skipTest(reason="Model architecture does not support attentions")

        if not self._is_composite:
            self.skipTest(f"{self.all_model_classes[0].__name__} does not support SDPA")

        for model_class in self.all_model_classes:
            config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
            model = model_class(config)

            with tempfile.TemporaryDirectory() as tmpdirname:
                model.save_pretrained(tmpdirname)
                model_sdpa = model_class.from_pretrained(tmpdirname)
                model_sdpa = model_sdpa.eval().to(torch_device)

                model_eager = model_class.from_pretrained(tmpdirname, attn_implementation="eager")
                model_eager = model_eager.eval().to(torch_device)
                self.assertTrue(model_eager.config._attn_implementation == "eager")

                for name, submodule in model_eager.named_modules():
                    class_name = submodule.__class__.__name__
                    if "SdpaAttention" in class_name or "SdpaSelfAttention" in class_name:
                        raise ValueError("The eager model should not have SDPA attention layers")


@require_torch
class ParakeetForCTCIntegrationTest(unittest.TestCase):
    _dataset = None

    @classmethod
    def setUp(cls):
        cls.checkpoint_name = "nvidia/parakeet-ctc-1.1b"
        cls.dtype = torch.bfloat16
        cls.processor = AutoProcessor.from_pretrained("nvidia/parakeet-ctc-1.1b")

    def tearDown(self):
        cleanup(torch_device, gc_collect=True)

    @classmethod
    def _load_dataset(cls):
        # Lazy loading of the dataset. Because it is a class method, it will only be loaded once per pytest process.
        if cls._dataset is None:
            cls._dataset = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
            cls._dataset = cls._dataset.cast_column(
                "audio", Audio(sampling_rate=cls.processor.feature_extractor.sampling_rate)
            )

    def _load_datasamples(self, num_samples):
        self._load_dataset()
        ds = self._dataset
        speech_samples = ds.sort("id")[:num_samples]["audio"]
        return [x["array"] for x in speech_samples]

    @slow
    def test_1b_model_integration(self):
        """
        bezzam reproducer (creates JSON directly in repo): https://gist.github.com/ebezzam/6382bdabfc64bb2541ca9f77deb7678d#file-reproducer_single-py
        eustlb reproducer: https://gist.github.com/eustlb/6e9e3aa85de3f7c340ec3c36e65f2fe6
        """
        RESULTS_PATH = Path(__file__).parent.parent.parent / "fixtures/parakeet/expected_results_single.json"
        with open(RESULTS_PATH, "r") as f:
            raw_data = json.load(f)
        EXPECTED_TOKEN_IDS = torch.tensor(raw_data["token_ids"])
        EXPECTED_TRANSCRIPTIONS = raw_data["transcriptions"]

        samples = self._load_datasamples(1)
        model = ParakeetForCTC.from_pretrained(self.checkpoint_name, torch_dtype=self.dtype, device_map=torch_device)
        model.eval()
        model.to(torch_device)

        # -- apply
        inputs = self.processor(samples)
        inputs.to(torch_device, dtype=self.dtype)
        predicted_ids = model.generate(**inputs)
        torch.testing.assert_close(predicted_ids.cpu(), EXPECTED_TOKEN_IDS)
        predicted_transcripts = self.processor.batch_decode(predicted_ids, skip_special_tokens=True)
        self.assertListEqual(predicted_transcripts, EXPECTED_TRANSCRIPTIONS)

    @slow
    def test_1b_model_integration_batched(self):
        """
        bezzam reproducer (creates JSON directly in repo): https://gist.github.com/ebezzam/6382bdabfc64bb2541ca9f77deb7678d#file-reproducer_batched-py
        eustlb reproducer: https://gist.github.com/eustlb/575b5da58de34a70116a1955b1183596
        """

        RESULTS_PATH = Path(__file__).parent.parent.parent / "fixtures/parakeet/expected_results_batch.json"
        with open(RESULTS_PATH, "r") as f:
            raw_data = json.load(f)
        EXPECTED_TOKEN_IDS = torch.tensor(raw_data["token_ids"])
        EXPECTED_TRANSCRIPTIONS = raw_data["transcriptions"]

        samples = self._load_datasamples(5)
        model = ParakeetForCTC.from_pretrained(self.checkpoint_name, torch_dtype=self.dtype, device_map=torch_device)
        model.eval()
        model.to(torch_device)

        # -- apply
        inputs = self.processor(samples)
        inputs.to(torch_device, dtype=self.dtype)
        predicted_ids = model.generate(**inputs)
        torch.testing.assert_close(predicted_ids.cpu(), EXPECTED_TOKEN_IDS)
        predicted_transcripts = self.processor.batch_decode(predicted_ids, skip_special_tokens=True)
        self.assertListEqual(predicted_transcripts, EXPECTED_TRANSCRIPTIONS)


# ──────────────────────────────────────────────────────────────────────────────
# Streaming / cache-aware tests
# ──────────────────────────────────────────────────────────────────────────────


def _make_streaming_encoder_config(
    att_context_size=None,
    att_context_probs=None,
    conv_context_size="causal",
    causal_downsampling=True,
    num_hidden_layers=2,
    hidden_size=64,
    num_attention_heads=4,
    intermediate_size=256,
    conv_kernel_size=9,
    subsampling_factor=8,
    subsampling_conv_channels=32,
    num_mel_bins=80,
):
    """Return a small ParakeetEncoderConfig with streaming parameters."""
    if att_context_size is None:
        att_context_size = [8, 4]
    return ParakeetEncoderConfig(
        hidden_size=hidden_size,
        num_hidden_layers=num_hidden_layers,
        num_attention_heads=num_attention_heads,
        intermediate_size=intermediate_size,
        hidden_act="silu",
        dropout=0.0,
        dropout_positions=0.0,
        layerdrop=0.0,
        activation_dropout=0.0,
        attention_dropout=0.0,
        conv_kernel_size=conv_kernel_size,
        subsampling_factor=subsampling_factor,
        subsampling_conv_channels=subsampling_conv_channels,
        num_mel_bins=num_mel_bins,
        att_context_size=att_context_size,
        att_context_probs=att_context_probs,
        conv_context_size=conv_context_size,
        causal_downsampling=causal_downsampling,
    )


@require_torch
class ParakeetStreamingTest(unittest.TestCase):
    """Tests for the cache-aware (streaming) forward path of ParakeetEncoder."""

    BATCH = 2
    SEQ = 128  # input frames (before subsampling)
    MEL = 80
    N_LAYERS = 2
    HIDDEN = 64
    LEFT_CTX = 8  # att_context_size[0]
    CONV_LEFT = 8  # conv_kernel_size - 1 (causal: 9-1=8)
    # Causal conv (k=3, s=2) applies floor(L/2)+1 per layer for 3 layers:
    # 128 → 65 → 33 → 17
    OUTPUT_T = 17

    def _make_model(self, config=None):
        if config is None:
            config = _make_streaming_encoder_config()
        model = ParakeetEncoder(config=config)
        model.to(torch_device)
        model.eval()
        return model

    def _make_input(self, seq=None, batch=None):
        B = batch or self.BATCH
        T = seq or self.SEQ
        feats = floats_tensor([B, T, self.MEL]).to(torch_device)
        mask = torch.ones(B, T, dtype=torch.long, device=torch_device)
        return feats, mask

    # ── get_initial_cache_state ────────────────────────────────────────────────

    def test_get_initial_cache_state_shapes(self):
        model = self._make_model()
        cache = model.get_initial_cache_state(batch_size=self.BATCH, device=torch_device)

        self.assertEqual(
            tuple(cache["cache_last_channel"].shape),
            (self.N_LAYERS, self.BATCH, self.LEFT_CTX, self.HIDDEN),
        )
        self.assertEqual(
            tuple(cache["cache_last_time"].shape),
            (self.N_LAYERS, self.BATCH, self.HIDDEN, self.CONV_LEFT),
        )
        self.assertEqual(tuple(cache["cache_last_channel_len"].shape), (self.BATCH,))
        self.assertTrue(cache["cache_last_channel"].eq(0).all())
        self.assertTrue(cache["cache_last_time"].eq(0).all())

    def test_get_initial_cache_state_raises_on_offline_model(self):
        offline_config = ParakeetEncoderConfig(
            hidden_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            intermediate_size=256,
            num_mel_bins=self.MEL,
        )
        model = ParakeetEncoder(config=offline_config).eval()
        with self.assertRaises(ValueError):
            model.get_initial_cache_state(batch_size=1)

    # ── use_cache flag ─────────────────────────────────────────────────────────

    def test_use_cache_returns_updated_caches(self):
        model = self._make_model()
        feats, mask = self._make_input()
        cache = model.get_initial_cache_state(batch_size=self.BATCH, device=torch_device)

        with torch.no_grad():
            out = model(feats, attention_mask=mask, use_cache=True, **cache)

        self.assertIsNotNone(out.cache_last_channel)
        self.assertIsNotNone(out.cache_last_time)

    def test_use_cache_false_returns_no_caches(self):
        model = self._make_model()
        feats, mask = self._make_input()
        cache = model.get_initial_cache_state(batch_size=self.BATCH, device=torch_device)

        with torch.no_grad():
            out = model(feats, attention_mask=mask, use_cache=False, **cache)

        self.assertIsNone(out.cache_last_channel)
        self.assertIsNone(out.cache_last_time)

    # ── cache shape invariance across two consecutive chunks ──────────────────

    def test_cache_shapes_preserved_across_chunks(self):
        model = self._make_model()
        cache = model.get_initial_cache_state(batch_size=self.BATCH, device=torch_device)
        expected_ch_shape = tuple(cache["cache_last_channel"].shape)
        expected_t_shape = tuple(cache["cache_last_time"].shape)

        with torch.no_grad():
            out1 = model(*self._make_input(), use_cache=True, **cache)
            cache2 = {
                "cache_last_channel": out1.cache_last_channel,
                "cache_last_time": out1.cache_last_time,
                "cache_last_channel_len": out1.cache_last_channel_len,
            }
            out2 = model(*self._make_input(), use_cache=True, **cache2)

        self.assertEqual(tuple(out1.cache_last_channel.shape), expected_ch_shape)
        self.assertEqual(tuple(out1.cache_last_time.shape), expected_t_shape)
        self.assertEqual(tuple(out2.cache_last_channel.shape), expected_ch_shape)
        self.assertEqual(tuple(out2.cache_last_time.shape), expected_t_shape)

    def test_cache_channel_len_increments_and_clamps(self):
        """cache_last_channel_len grows per chunk and is capped at left_ctx."""
        model = self._make_model()
        cache = model.get_initial_cache_state(batch_size=self.BATCH, device=torch_device)

        with torch.no_grad():
            out1 = model(*self._make_input(), use_cache=True, **cache)

        expected_len = min(self.OUTPUT_T, self.LEFT_CTX)
        self.assertTrue((out1.cache_last_channel_len == expected_len).all())

        cache2 = {
            "cache_last_channel": out1.cache_last_channel,
            "cache_last_time": out1.cache_last_time,
            "cache_last_channel_len": out1.cache_last_channel_len,
        }
        with torch.no_grad():
            out2 = model(*self._make_input(), use_cache=True, **cache2)
        self.assertTrue((out2.cache_last_channel_len <= self.LEFT_CTX).all())

    def test_streaming_output_shape_matches_offline(self):
        """Hidden state shape is identical between streaming and offline paths."""
        model = self._make_model()
        feats, mask = self._make_input()
        cache = model.get_initial_cache_state(batch_size=self.BATCH, device=torch_device)

        with torch.no_grad():
            offline_out = model(feats, attention_mask=mask)
            streaming_out = model(feats, attention_mask=mask, use_cache=True, **cache)

        self.assertEqual(offline_out.last_hidden_state.shape, streaming_out.last_hidden_state.shape)

    # ── attention window mask ──────────────────────────────────────────────────

    def test_build_att_window_mask_shape(self):
        model = self._make_model()
        chunk_len, cache_len = 16, 8
        mask = model._build_att_window_mask(
            seq_len=chunk_len,
            total_key_len=chunk_len + cache_len,
            left_ctx=self.LEFT_CTX,
            right_ctx=4,
            device=torch_device,
        )
        self.assertEqual(tuple(mask.shape), (1, 1, chunk_len, chunk_len + cache_len))
        self.assertEqual(mask.dtype, torch.bool)

    def test_build_att_window_mask_correctness(self):
        """Every (q, k) entry must match the window formula dist ∈ [-right, left]."""
        model = self._make_model()
        left_ctx, right_ctx = 2, 1
        seq_len, cache_len = 4, 2
        total = seq_len + cache_len
        mask = model._build_att_window_mask(seq_len, total, left_ctx, right_ctx, device=torch_device)

        for q in range(seq_len):
            for k in range(total):
                k_pos = k - cache_len
                dist = q - k_pos
                expected = (-right_ctx <= dist) and (dist <= left_ctx)
                self.assertEqual(
                    mask[0, 0, q, k].item(),
                    expected,
                    msg=f"mask[{q},{k}] wrong (dist={dist}, expected={expected})",
                )

    # ── multi-lookahead context selection ─────────────────────────────────────

    def test_multi_lookahead_uses_first_context_by_default(self):
        ctx_list = [[8, 4], [8, 0]]
        config = _make_streaming_encoder_config(att_context_size=ctx_list, att_context_probs=[0.5, 0.5])
        model = self._make_model(config)
        self.assertEqual(model._resolve_att_context_size(None), ctx_list[0])

    def test_multi_lookahead_valid_override(self):
        ctx_list = [[8, 4], [8, 0]]
        config = _make_streaming_encoder_config(att_context_size=ctx_list, att_context_probs=[0.5, 0.5])
        model = self._make_model(config)
        self.assertEqual(model._resolve_att_context_size([8, 0]), [8, 0])

    def test_multi_lookahead_invalid_override_raises(self):
        ctx_list = [[8, 4], [8, 0]]
        config = _make_streaming_encoder_config(att_context_size=ctx_list, att_context_probs=[0.5, 0.5])
        model = self._make_model(config)
        with self.assertRaises(ValueError):
            model._resolve_att_context_size([99, 99])

    def test_offline_model_returns_none_context(self):
        offline_config = ParakeetEncoderConfig(
            hidden_size=64, num_hidden_layers=2, num_attention_heads=4, intermediate_size=256, num_mel_bins=self.MEL
        )
        model = ParakeetEncoder(config=offline_config).eval()
        self.assertIsNone(model._resolve_att_context_size(None))

    def test_single_pair_context_resolved(self):
        config = _make_streaming_encoder_config(att_context_size=[8, 4])
        model = self._make_model(config)
        self.assertEqual(model._resolve_att_context_size(None), [8, 4])

    # ── ParakeetForCTC streaming ───────────────────────────────────────────────

    def test_ctc_use_cache_returns_caches(self):
        enc_config = _make_streaming_encoder_config()
        ctc_config = ParakeetCTCConfig(encoder_config=enc_config, vocab_size=64)
        model = ParakeetForCTC(config=ctc_config).to(torch_device).eval()

        feats, mask = self._make_input()
        cache = model.encoder.get_initial_cache_state(batch_size=self.BATCH, device=torch_device)

        with torch.no_grad():
            out = model(feats, attention_mask=mask, use_cache=True, **cache)

        self.assertIsInstance(out, ParakeetCTCModelOutput)
        self.assertIsNotNone(out.cache_last_channel)
        self.assertIsNotNone(out.cache_last_time)
        self.assertEqual(out.logits.shape, (self.BATCH, self.OUTPUT_T, 64))

    def test_ctc_cache_shapes_preserved(self):
        enc_config = _make_streaming_encoder_config()
        ctc_config = ParakeetCTCConfig(encoder_config=enc_config, vocab_size=64)
        model = ParakeetForCTC(config=ctc_config).to(torch_device).eval()

        feats, mask = self._make_input()
        cache = model.encoder.get_initial_cache_state(batch_size=self.BATCH, device=torch_device)
        expected_ch_shape = tuple(cache["cache_last_channel"].shape)
        expected_t_shape = tuple(cache["cache_last_time"].shape)

        with torch.no_grad():
            out = model(feats, attention_mask=mask, use_cache=True, **cache)

        self.assertEqual(tuple(out.cache_last_channel.shape), expected_ch_shape)
        self.assertEqual(tuple(out.cache_last_time.shape), expected_t_shape)
