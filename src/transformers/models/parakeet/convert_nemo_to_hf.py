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

import argparse
import gc
import os
import re
import tarfile

import torch
import yaml
from tokenizers import AddedToken

from transformers import (
    ParakeetCTCConfig,
    ParakeetEncoder,
    ParakeetEncoderConfig,
    ParakeetFeatureExtractor,
    ParakeetForCTC,
    ParakeetProcessor,
    ParakeetTokenizer,
)
from transformers.convert_slow_tokenizer import ParakeetConverter
from transformers.utils.hub import cached_file


NEMO_TO_HF_ENCODER_MAPPING = {
    r"encoder\.pre_encode\.conv\.": r"encoder.subsampling.layers.",
    r"encoder\.pre_encode\.out\.": r"encoder.subsampling.linear.",
    r"encoder\.pos_enc\.": r"encoder.encode_positions.",
    r"encoder\.layers\.(\d+)\.conv\.batch_norm\.": r"encoder.layers.\1.conv.norm.",
    r"encoder\.layers\.(\d+)\.conv\.layer_norm\.": r"encoder.layers.\1.conv.norm.",
    r"linear_([kv])": r"\1_proj",
    r"linear_out": r"o_proj",
    r"linear_q": r"q_proj",
    r"pos_bias_([uv])": r"bias_\1",
    r"linear_pos": r"relative_k_proj",
}

# CTC head key prefix differs between pure-CTC and hybrid models
NEMO_CTC_KEY_PREFIX = {
    "ctc": r"decoder\.decoder_layers\.0\.(weight|bias)",
    "hybrid_rnnt_ctc": r"ctc_decoder\.decoder_layers\.0\.(weight|bias)",
}

# Keys to skip when converting to a given HF model type
NEMO_SKIP_PREFIXES = {
    "ctc": [],
    "encoder": [],
    "rnnt": ["decoder.", "joint.", "preprocessor."],
    "hybrid_rnnt_ctc": ["decoder.", "joint.", "preprocessor."],  # skip RNNT decoder/joint
}


def convert_key(key, mapping):
    for pattern, replacement in mapping.items():
        key = re.sub(pattern, replacement, key)
    return key


def extract_nemo_archive(nemo_file_path: str, extract_dir: str) -> dict[str, str]:
    """
    Extract .nemo file (tar archive) and return paths to important files.

    Args:
        nemo_file_path: Path to .nemo file
        extract_dir: Directory to extract to

    Returns:
        Dictionary with paths to model.pt, model_config.yaml, etc.
    """
    print(f"Extracting NeMo archive: {nemo_file_path}")

    with tarfile.open(nemo_file_path, "r", encoding="utf-8") as tar:
        tar.extractall(extract_dir)

    # Log all extracted files for debugging
    all_files = []
    for root, dirs, files in os.walk(extract_dir):
        for file in files:
            file_path = os.path.join(root, file)
            all_files.append(file_path)

    print(f"All extracted files: {[os.path.basename(f) for f in all_files]}")

    # Find important files with more robust detection
    model_files = {}
    for root, dirs, files in os.walk(extract_dir):
        for file in files:
            file_path = os.path.join(root, file)
            file_lower = file.lower()

            # Look for model weights with various common names
            if (
                file.endswith(".pt")
                or file.endswith(".pth")
                or file.endswith(".ckpt")
                or file.endswith(".bin")
                or "model" in file_lower
                and ("weight" in file_lower or "state" in file_lower)
                or file_lower == "model.pt"
                or file_lower == "pytorch_model.bin"
                or file_lower == "model_weights.ckpt"
            ):
                model_files["model_weights"] = file_path
                print(f"Found model weights: {file}")

            # Look for config files
            elif (
                file == "model_config.yaml"
                or file == "config.yaml"
                or (file.endswith(".yaml") and "config" in file_lower)
            ):
                if "model_config" not in model_files:  # Prefer model_config.yaml
                    model_files["model_config"] = file_path
                    print(f"Found config file: {file}")
                if file == "model_config.yaml":
                    model_files["model_config"] = file_path  # Override with preferred name

            # Look for vocabulary files
            elif (
                file.endswith(".vocab")
                or file.endswith(".model")
                or file.endswith(".txt")
                or ("tokenizer" in file_lower and (file.endswith(".vocab") or file.endswith(".model")))
            ):
                # Prefer .vocab files over others
                if "tokenizer_model_file" not in model_files or file.endswith(".model"):
                    model_files["tokenizer_model_file"] = file_path
                    print(f"Found tokenizer model file: {file}")
                else:
                    print(f"Found additional vocabulary file (using existing): {file}")

    print(f"Found model files: {list(model_files.keys())}")

    # Validate that we found the required files
    if "model_weights" not in model_files:
        raise FileNotFoundError(
            f"Could not find model weights file in {nemo_file_path}. "
            f"Expected files with extensions: .pt, .pth, .ckpt, .bin. "
            f"Found files: {[os.path.basename(f) for f in all_files]}"
        )

    if "model_config" not in model_files:
        raise FileNotFoundError(
            f"Could not find model config file in {nemo_file_path}. "
            f"Expected: model_config.yaml or config.yaml. "
            f"Found files: {[os.path.basename(f) for f in all_files]}"
        )

    return model_files


def write_processor(nemo_config: dict, model_files, output_dir, push_to_repo_id=None):
    tokenizer_converted = ParakeetConverter(model_files["tokenizer_model_file"]).converted()
    tokenizer_converted_fast = ParakeetTokenizer(
        tokenizer_object=tokenizer_converted,
        clean_up_tokenization_spaces=False,
    )
    tokenizer_converted_fast.add_tokens(
        [AddedToken("<unk>", normalized=False, special=True), AddedToken("<pad>", normalized=False, special=True)]
    )
    tokenizer_converted_fast.add_special_tokens(
        {
            "pad_token": AddedToken("<pad>", normalized=False, special=True),
            "unk_token": AddedToken("<unk>", normalized=False, special=True),
        }
    )

    feature_extractor_keys_to_ignore = [
        "_target_", "pad_to", "frame_splicing", "dither", "window", "log",
        "nb_augmentation_prob",  # training-only augmentation flag
    ]
    feature_extractor_config_keys_mapping = {
        "sample_rate": "sampling_rate",
        "window_size": "win_length",
        "window_stride": "hop_length",
        "window": "window",
        "n_fft": "n_fft",
        "log": "log",
        "features": "feature_size",
        "dither": "dither",
        "pad_to": "pad_to",
        "pad_value": "padding_value",
        "frame_splicing": "frame_splicing",
        "preemphasis": "preemphasis",
        "hop_length": "hop_length",
        "normalize": "do_normalize",
    }
    converted_feature_extractor_config = {}

    for key, value in nemo_config["preprocessor"].items():
        if key in feature_extractor_keys_to_ignore:
            continue
        if key in feature_extractor_config_keys_mapping:
            if key in ["window_size", "window_stride"]:
                value = int(value * nemo_config["preprocessor"]["sample_rate"])
            elif key == "normalize":
                # NeMo "NA" means no normalization; anything else (e.g. "per_feature") means normalize
                value = value != "NA"
            converted_feature_extractor_config[feature_extractor_config_keys_mapping[key]] = value
        else:
            raise ValueError(f"Key {key} not found in feature_extractor_keys_mapping")

    feature_extractor = ParakeetFeatureExtractor(**converted_feature_extractor_config)

    processor = ParakeetProcessor(
        feature_extractor=feature_extractor,
        tokenizer=tokenizer_converted_fast,
    )
    processor.save_pretrained(output_dir)

    if push_to_repo_id:
        processor.push_to_hub(push_to_repo_id)


def detect_nemo_model_type(nemo_config: dict) -> str:
    """Detect NeMo model class from _target_ and return one of: 'ctc', 'rnnt', 'hybrid_rnnt_ctc'."""
    target = nemo_config.get("_target_", nemo_config.get("target", ""))
    if "EncDecHybridRNNTCTC" in target:
        return "hybrid_rnnt_ctc"
    if "RNNT" in target or "Transducer" in target:
        return "rnnt"
    if "CTC" in target:
        return "ctc"
    raise ValueError(f"Cannot determine NeMo model type from _target_: '{target}'")


def convert_encoder_config(nemo_config):
    """Convert NeMo encoder config to HF encoder config."""
    encoder_keys_to_ignore = [
        "stochastic_depth_start_layer",
        "feat_out",
        "stochastic_depth_drop_prob",
        "_target_",
        "ff_expansion_factor",
        "untie_biases",
        "self_attention_model",
        "subsampling",
        "stochastic_depth_mode",
        "dropout_pre_encoder",
        "reduction",
        "reduction_factor",
        "reduction_position",
        "att_context_probs",  # training-only; stored as-is if present
    ]
    encoder_config_keys_mapping = {
        "d_model": "hidden_size",
        "n_heads": "num_attention_heads",
        "n_layers": "num_hidden_layers",
        "feat_in": "num_mel_bins",
        "conv_kernel_size": "conv_kernel_size",
        "subsampling_factor": "subsampling_factor",
        "subsampling_conv_channels": "subsampling_conv_channels",
        "pos_emb_max_len": "max_position_embeddings",
        "dropout": "dropout",
        "dropout_emb": "dropout_positions",
        "dropout_att": "attention_dropout",
        "xscaling": "scale_input",
        "use_bias": "attention_bias",
        # streaming fields
        "att_context_size": "att_context_size",
        "att_context_style": "att_context_style",
        "conv_context_size": "conv_context_size",
        "causal_downsampling": "causal_downsampling",
        "conv_norm_type": "conv_norm_type",
    }
    converted_encoder_config = {}

    for key, value in nemo_config["encoder"].items():
        if key in encoder_keys_to_ignore:
            continue
        if key in encoder_config_keys_mapping:
            converted_encoder_config[encoder_config_keys_mapping[key]] = value
            # NeMo uses 'use_bias' for both attention and convolution bias, but HF separates them
            if key == "use_bias":
                converted_encoder_config["convolution_bias"] = value
        else:
            raise ValueError(f"Key {key} not found in encoder_config_keys_mapping")

    # Compute intermediate_size from ff_expansion_factor × d_model
    nemo_enc = nemo_config["encoder"]
    if "ff_expansion_factor" in nemo_enc:
        converted_encoder_config["intermediate_size"] = (
            nemo_enc["ff_expansion_factor"] * nemo_enc["d_model"]
        )

    # Normalise offline att_context_size: [-1, -1] → None
    ctx = converted_encoder_config.get("att_context_size")
    if ctx is not None:
        is_offline = ctx == [-1, -1] or (isinstance(ctx[0], list) and all(c == [-1, -1] for c in ctx))
        if is_offline:
            converted_encoder_config.pop("att_context_size")

    return ParakeetEncoderConfig(**converted_encoder_config)


def load_and_convert_state_dict(model_files, nemo_model_type: str, hf_model_type: str):
    """Load NeMo state dict and convert keys to HF format.

    Args:
        nemo_model_type: detected NeMo architecture ('ctc', 'rnnt', 'hybrid_rnnt_ctc').
        hf_model_type: target HF model type ('ctc', 'encoder').
    """
    state_dict = torch.load(model_files["model_weights"], map_location="cpu", weights_only=True)

    # Build the CTC head mapping based on where the CTC head lives in this checkpoint
    ctc_pattern = NEMO_CTC_KEY_PREFIX.get(nemo_model_type, NEMO_CTC_KEY_PREFIX["ctc"])
    weight_mapping = dict(NEMO_TO_HF_ENCODER_MAPPING)
    weight_mapping[ctc_pattern] = r"ctc_head.\1"

    # Prefixes to skip for the requested hf_model_type
    skip_prefixes = tuple(NEMO_SKIP_PREFIXES.get(hf_model_type, []))
    # Always skip featurizer
    skip_suffixes = ("featurizer.window", "featurizer.fb")

    converted_state_dict = {}
    for key, value in state_dict.items():
        if any(key.endswith(s) for s in skip_suffixes):
            print(f"Skipping preprocessing weight: {key}")
            continue
        if skip_prefixes and key.startswith(skip_prefixes):
            print(f"Skipping decoder weight (not needed for {hf_model_type}): {key}")
            continue
        converted_key = convert_key(key, weight_mapping)
        converted_state_dict[converted_key] = value

    return converted_state_dict


def write_ctc_model(encoder_config, converted_state_dict, output_dir, push_to_repo_id=None):
    """Write CTC model using encoder config and converted state dict."""
    # Infer vocab_size from the CTC head weight shape
    ctc_weight_key = next(k for k in converted_state_dict if "ctc_head" in k and "weight" in k)
    vocab_size = converted_state_dict[ctc_weight_key].shape[0]
    model_config = ParakeetCTCConfig(encoder_config=encoder_config, vocab_size=vocab_size)

    print("Loading the checkpoint in a Parakeet CTC model.")
    with torch.device("meta"):
        model = ParakeetForCTC(model_config)
    model.load_state_dict(converted_state_dict, strict=True, assign=True)
    print("Checkpoint loaded successfully.")
    del model.config._name_or_path

    print("Saving the model.")
    model.save_pretrained(output_dir)

    if push_to_repo_id:
        model.push_to_hub(push_to_repo_id)

    del model

    # Safety check: reload the converted model
    gc.collect()
    print("Reloading the model to check if it's saved correctly.")
    try:
        ParakeetForCTC.from_pretrained(output_dir, dtype=torch.bfloat16, device_map="auto")
    except ValueError:
        ParakeetForCTC.from_pretrained(output_dir, dtype=torch.bfloat16)
    print("Model reloaded successfully.")


def write_encoder_model(encoder_config, converted_state_dict, output_dir, push_to_repo_id=None):
    """Write encoder model using encoder config and converted state dict."""
    # Filter to only encoder weights (exclude CTC head if present)
    encoder_state_dict = {
        k.replace("encoder.", "", 1) if k.startswith("encoder.") else k: v
        for k, v in converted_state_dict.items()
        if k.startswith("encoder.")
    }

    print("Loading the checkpoint in a Parakeet Encoder model (for TDT).")
    with torch.device("meta"):
        model = ParakeetEncoder(encoder_config)

    model.load_state_dict(encoder_state_dict, strict=True, assign=True)
    print("Checkpoint loaded successfully.")
    del model.config._name_or_path

    print("Saving the model.")
    model.save_pretrained(output_dir)

    if push_to_repo_id:
        model.push_to_hub(push_to_repo_id)
    del model

    # Safety check: reload the converted model
    gc.collect()
    print("Reloading the model to check if it's saved correctly.")
    try:
        ParakeetEncoder.from_pretrained(output_dir, dtype=torch.bfloat16, device_map="auto")
    except ValueError:
        ParakeetEncoder.from_pretrained(output_dir, dtype=torch.bfloat16)
    print("Model reloaded successfully.")


def write_model(nemo_config, model_files, model_type, output_dir, push_to_repo_id=None):
    """Main model conversion function."""
    nemo_model_type = detect_nemo_model_type(nemo_config)
    print(f"Detected NeMo model type: {nemo_model_type}")

    encoder_config = convert_encoder_config(nemo_config)
    print(f"Converted encoder config: {encoder_config}")

    converted_state_dict = load_and_convert_state_dict(model_files, nemo_model_type, model_type)

    if model_type == "encoder":
        write_encoder_model(encoder_config, converted_state_dict, output_dir, push_to_repo_id)
    elif model_type == "ctc":
        if nemo_model_type == "rnnt":
            raise ValueError("The checkpoint has no CTC decoder (pure RNNT). Use --model_type encoder.")
        write_ctc_model(encoder_config, converted_state_dict, output_dir, push_to_repo_id)
    else:
        raise ValueError(f"Model type '{model_type}' not supported. Choose from: encoder, ctc.")


def main(
    hf_repo_id,
    output_dir,
    model_type,
    push_to_repo_id=None,
):
    nemo_filename = f"{hf_repo_id.split('/')[-1]}.nemo"
    filepath = cached_file(hf_repo_id, nemo_filename)

    model_files = extract_nemo_archive(filepath, os.path.dirname(filepath))
    nemo_config = yaml.load(open(model_files["model_config"], "r"), Loader=yaml.FullLoader)

    write_processor(nemo_config, model_files, output_dir, push_to_repo_id)
    write_model(nemo_config, model_files, model_type, output_dir, push_to_repo_id)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--hf_repo_id", required=True, help="Model repo on huggingface.co")
    parser.add_argument(
        "--model_type", required=True, choices=["encoder", "ctc"], help="Model type (`encoder`, `ctc`)"
    )
    parser.add_argument("--output_dir", required=True, help="Output directory for HuggingFace model")
    parser.add_argument("--push_to_repo_id", help="Repository ID to push the model to on the Hub")
    args = parser.parse_args()
    main(
        args.hf_repo_id,
        args.output_dir,
        args.model_type,
        args.push_to_repo_id,
    )
