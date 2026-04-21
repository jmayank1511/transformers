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
    ParakeetForRNNT,
    ParakeetForTDT,
    ParakeetProcessor,
    ParakeetRNNTConfig,
    ParakeetTDTConfig,
    ParakeetTokenizer,
)
from transformers.convert_slow_tokenizer import ParakeetConverter
from transformers.utils.hub import cached_file


NEMO_TO_HF_WEIGHT_MAPPING = {
    r"encoder\.pre_encode\.conv\.": r"encoder.subsampling.layers.",
    r"encoder\.pre_encode\.out\.": r"encoder.subsampling.linear.",
    r"encoder\.pos_enc\.": r"encoder.encode_positions.",
    r"encoder\.layers\.(\d+)\.conv\.batch_norm\.": r"encoder.layers.\1.conv.norm.",
    r"decoder\.decoder_layers\.0\.(weight|bias)": r"ctc_head.\1",
    r"linear_([kv])": r"\1_proj",
    r"linear_out": r"o_proj",
    r"linear_q": r"q_proj",
    r"pos_bias_([uv])": r"bias_\1",
    r"linear_pos": r"relative_k_proj",
}

# Additional key mappings for transducer (RNNT / TDT) models.
# Applied on top of NEMO_TO_HF_WEIGHT_MAPPING when converting RNNT/TDT checkpoints.
NEMO_TO_HF_RNNT_EXTRA_MAPPING = {
    r"decoder\.prediction\.embed\.": r"prediction_network.embed.",
    r"decoder\.prediction\.dec_rnn\.lstm\.": r"prediction_network.lstm.",
    r"joint\.enc\.": r"joint_network.enc_proj.",
    r"joint\.pred\.": r"joint_network.pred_proj.",
    # joint_net may be Sequential([Linear]) or Sequential([Dropout, Linear]);
    # only the Linear has parameters so any index works.
    r"joint\.joint_net\.\d+\.": r"joint_network.out_proj.",
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

    feature_extractor_keys_to_ignore = ["_target_", "pad_to", "frame_splicing", "dither", "normalize", "window", "log"]
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
    }
    converted_feature_extractor_config = {}

    for key, value in nemo_config["preprocessor"].items():
        if key in feature_extractor_keys_to_ignore:
            continue
        if key in feature_extractor_config_keys_mapping:
            if key in ["window_size", "window_stride"]:
                value = int(value * nemo_config["preprocessor"]["sample_rate"])
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


def convert_encoder_config(nemo_config):
    """Convert NeMo encoder config to HF encoder config."""
    encoder_keys_to_ignore = [
        "att_context_size",
        "causal_downsampling",
        "stochastic_depth_start_layer",
        "feat_out",
        "stochastic_depth_drop_prob",
        "_target_",
        "ff_expansion_factor",
        "untie_biases",
        "att_context_style",
        "self_attention_model",
        "conv_norm_type",
        "subsampling",
        "stochastic_depth_mode",
        "conv_context_size",
        "dropout_pre_encoder",
        "reduction",
        "reduction_factor",
        "reduction_position",
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

    return ParakeetEncoderConfig(**converted_encoder_config)


def load_and_convert_state_dict(model_files, extra_mapping=None):
    """Load NeMo state dict and convert keys to HF format.

    Args:
        model_files: Dict with path to model weights.
        extra_mapping: Optional additional regex key-mapping dict merged on top of
            NEMO_TO_HF_WEIGHT_MAPPING (used for RNNT / TDT models).
    """
    mapping = {**NEMO_TO_HF_WEIGHT_MAPPING, **(extra_mapping or {})}
    state_dict = torch.load(model_files["model_weights"], map_location="cpu", weights_only=True)
    converted_state_dict = {}
    for key, value in state_dict.items():
        # Skip preprocessing weights (featurizer components)
        if key.endswith("featurizer.window") or key.endswith("featurizer.fb"):
            print(f"Skipping preprocessing weight: {key}")
            continue
        converted_key = convert_key(key, mapping)
        converted_state_dict[converted_key] = value

    return converted_state_dict


def write_ctc_model(encoder_config, converted_state_dict, output_dir, push_to_repo_id=None):
    """Write CTC model using encoder config and converted state dict."""
    model_config = ParakeetCTCConfig.from_encoder_config(encoder_config)

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
    ParakeetForCTC.from_pretrained(output_dir, dtype=torch.bfloat16, device_map="auto")
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
    ParakeetEncoder.from_pretrained(output_dir, dtype=torch.bfloat16, device_map="auto")
    print("Model reloaded successfully.")


def _get_prednet(nemo_config):
    """Return the prediction-network sub-dict, handling flat vs. nested NeMo layouts."""
    decoder = nemo_config["decoder"]
    return decoder.get("prednet", decoder)


def _get_jointnet(nemo_config):
    """Return the joint-network sub-dict, handling flat vs. nested NeMo layouts."""
    joint = nemo_config["joint"]
    return joint.get("jointnet", joint)


def _get_nemo_vocab_size(nemo_config):
    """Return the NeMo vocab_size (number of non-blank tokens).

    NeMo stores 'vocab_size' as the count of regular tokens, with the blank
    token appended afterwards.  The total HF vocab_size = nemo_vocab_size + 1
    and blank_id = nemo_vocab_size.
    """
    # NeMo RNNTJoint.num_classes = number of non-blank tokens (blank appended after).
    joint = nemo_config.get("joint", {})
    if "num_classes" in joint:
        return joint["num_classes"]
    # Fall back to the top-level 'vocab_size' key (excludes blank).
    if "vocab_size" in nemo_config:
        return nemo_config["vocab_size"]
    raise ValueError(
        "Cannot determine vocab_size from NeMo config. "
        "Expected 'joint.num_classes' or top-level 'vocab_size'."
    )


def convert_prediction_network_config(nemo_config):
    """Convert NeMo RNNT decoder config to a dict suitable for ParakeetPredictionNetworkConfig."""
    decoder = nemo_config["decoder"]
    prednet = _get_prednet(nemo_config)
    return {
        "pred_hidden": prednet.get("pred_hidden", 640),
        "pred_rnn_layers": prednet.get("pred_rnn_layers", 1),
        "dropout": prednet.get("dropout", 0.0),
        "blank_as_pad": decoder.get("blank_as_pad", True),
    }


def convert_joint_network_config(nemo_config):
    """Convert NeMo RNNT joint config to a dict suitable for ParakeetJointNetworkConfig."""
    jointnet = _get_jointnet(nemo_config)
    return {
        "joint_hidden": jointnet.get("joint_hidden", 640),
        "activation": jointnet.get("activation", "relu"),
        "dropout": jointnet.get("dropout", 0.0),
    }


def convert_rnnt_config(nemo_config, encoder_config):
    """Build a ParakeetRNNTConfig from NeMo model config + already-converted encoder config."""
    nemo_vocab = _get_nemo_vocab_size(nemo_config)
    # NeMo appends blank after all regular tokens: blank_id = nemo_vocab_size.
    blank_id = nemo_vocab
    vocab_size = nemo_vocab + 1  # total including blank
    pred_cfg = convert_prediction_network_config(nemo_config)
    joint_cfg = convert_joint_network_config(nemo_config)
    return ParakeetRNNTConfig(
        vocab_size=vocab_size,
        blank_id=blank_id,
        encoder_config=encoder_config,
        prediction_network_config=pred_cfg,
        joint_network_config=joint_cfg,
    )


def convert_tdt_config(nemo_config, encoder_config):
    """Build a ParakeetTDTConfig from NeMo model config + already-converted encoder config."""
    nemo_vocab = _get_nemo_vocab_size(nemo_config)
    blank_id = nemo_vocab
    vocab_size = nemo_vocab + 1
    pred_cfg = convert_prediction_network_config(nemo_config)
    joint_cfg = convert_joint_network_config(nemo_config)

    # TDT-specific parameters live under the loss config in NeMo.
    loss_cfg = nemo_config.get("loss", {})
    durations = loss_cfg.get("durations", [0, 1, 2, 3, 4])
    sigma = loss_cfg.get("sigma", 0.05)
    omega = loss_cfg.get("omega", 0.1)

    return ParakeetTDTConfig(
        vocab_size=vocab_size,
        blank_id=blank_id,
        encoder_config=encoder_config,
        prediction_network_config=pred_cfg,
        joint_network_config=joint_cfg,
        durations=durations,
        sigma=sigma,
        omega=omega,
    )


def write_rnnt_model(rnnt_config, converted_state_dict, output_dir, push_to_repo_id=None):
    """Write RNNT model using already-converted config and state dict."""
    print("Loading the checkpoint in a Parakeet RNNT model.")
    with torch.device("meta"):
        model = ParakeetForRNNT(rnnt_config)
    model.load_state_dict(converted_state_dict, strict=True, assign=True)
    print("Checkpoint loaded successfully.")
    del model.config._name_or_path

    print("Saving the model.")
    model.save_pretrained(output_dir)

    if push_to_repo_id:
        model.push_to_hub(push_to_repo_id)
    del model

    gc.collect()
    print("Reloading the model to check if it's saved correctly.")
    ParakeetForRNNT.from_pretrained(output_dir, dtype=torch.bfloat16, device_map="auto")
    print("Model reloaded successfully.")


def write_tdt_model(tdt_config, converted_state_dict, output_dir, push_to_repo_id=None):
    """Write TDT model using already-converted config and state dict."""
    print("Loading the checkpoint in a Parakeet TDT model.")
    with torch.device("meta"):
        model = ParakeetForTDT(tdt_config)
    model.load_state_dict(converted_state_dict, strict=True, assign=True)
    print("Checkpoint loaded successfully.")
    del model.config._name_or_path

    print("Saving the model.")
    model.save_pretrained(output_dir)

    if push_to_repo_id:
        model.push_to_hub(push_to_repo_id)
    del model

    gc.collect()
    print("Reloading the model to check if it's saved correctly.")
    ParakeetForTDT.from_pretrained(output_dir, dtype=torch.bfloat16, device_map="auto")
    print("Model reloaded successfully.")


def write_model(nemo_config, model_files, model_type, output_dir, push_to_repo_id=None):
    """Main model conversion function."""
    # Step 1: Convert encoder config (shared across all model types)
    encoder_config = convert_encoder_config(nemo_config)
    print(f"Converted encoder config: {encoder_config}")

    # Step 2: Load and convert state dict (shared across all model types;
    # RNNT/TDT require the extra transducer key mappings).
    is_transducer = model_type in ("rnnt", "tdt")
    extra_mapping = NEMO_TO_HF_RNNT_EXTRA_MAPPING if is_transducer else None
    converted_state_dict = load_and_convert_state_dict(model_files, extra_mapping=extra_mapping)

    # Step 3: Write model based on type
    if model_type == "encoder":
        write_encoder_model(encoder_config, converted_state_dict, output_dir, push_to_repo_id)
    elif model_type == "ctc":
        write_ctc_model(encoder_config, converted_state_dict, output_dir, push_to_repo_id)
    elif model_type == "rnnt":
        rnnt_config = convert_rnnt_config(nemo_config, encoder_config)
        print(f"Converted RNNT config: {rnnt_config}")
        write_rnnt_model(rnnt_config, converted_state_dict, output_dir, push_to_repo_id)
    elif model_type == "tdt":
        tdt_config = convert_tdt_config(nemo_config, encoder_config)
        print(f"Converted TDT config: {tdt_config}")
        write_tdt_model(tdt_config, converted_state_dict, output_dir, push_to_repo_id)
    else:
        raise ValueError(f"Model type {model_type} not supported.")


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
        "--model_type",
        required=True,
        choices=["encoder", "ctc", "rnnt", "tdt"],
        help="Model type (`encoder`, `ctc`, `rnnt`, `tdt`)",
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
