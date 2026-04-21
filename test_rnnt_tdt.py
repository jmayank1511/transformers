"""
Quick end-to-end test: convert RNNT/TDT .nemo weights -> HF model, then run inference.
Bypasses the tokenizer converter (uses pre-existing HF processor instead).
"""
import gc
import os
import tarfile
import tempfile

import soundfile as sf
import torch
import yaml

from transformers import (
    AutoProcessor,
    ParakeetForRNNT,
    ParakeetForTDT,
    ParakeetProcessor,
)
from transformers.models.parakeet.convert_nemo_to_hf import (
    NEMO_TO_HF_RNNT_EXTRA_MAPPING,
    convert_encoder_config,
    convert_rnnt_config,
    convert_tdt_config,
    load_and_convert_state_dict,
)
from transformers.utils.hub import cached_file

AUDIO_FILE = "/media/mayjain/Seagate/tmp/tmp/riva-speech/test_files/asr/public/en-US_sample.wav"


def extract_nemo(nemo_path, extract_dir):
    print(f"Extracting {os.path.basename(nemo_path)} ...")
    with tarfile.open(nemo_path, "r", encoding="utf-8") as tar:
        tar.extractall(extract_dir)
    files = {}
    for root, _, fnames in os.walk(extract_dir):
        for fname in fnames:
            p = os.path.join(root, fname)
            if fname.endswith(".ckpt") or fname.endswith(".pt"):
                files["model_weights"] = p
            elif fname == "model_config.yaml":
                files["model_config"] = p
    return files


def load_audio(path):
    audio, sr = sf.read(path, dtype="float32")
    if audio.ndim > 1:
        audio = audio[:, 0]
    return audio, sr


def test_rnnt(output_dir):
    print("\n" + "=" * 60)
    print("RNNT conversion + inference test")
    print("=" * 60)

    nemo_path = cached_file("nvidia/parakeet-rnnt-1.1b", "parakeet-rnnt-1.1b.nemo")

    with tempfile.TemporaryDirectory() as tmpdir:
        model_files = extract_nemo(nemo_path, tmpdir)
        nemo_config = yaml.safe_load(open(model_files["model_config"]))

        print("Converting config ...")
        encoder_config = convert_encoder_config(nemo_config)
        rnnt_config = convert_rnnt_config(nemo_config, encoder_config)
        print(f"  vocab_size={rnnt_config.vocab_size}, blank_id={rnnt_config.blank_id}")

        print("Converting weights ...")
        state_dict = load_and_convert_state_dict(model_files, extra_mapping=NEMO_TO_HF_RNNT_EXTRA_MAPPING)
        print(f"  {len(state_dict)} tensors converted")

        print("Loading model (strict=False to see gaps) ...")
        with torch.device("meta"):
            model = ParakeetForRNNT(rnnt_config)
        missing, unexpected = model.load_state_dict(state_dict, strict=False, assign=True)
        if missing:
            print(f"  MISSING ({len(missing)}): {missing[:10]}")
        if unexpected:
            print(f"  UNEXPECTED ({len(unexpected)}): {unexpected[:10]}")
        if not missing and not unexpected:
            print("  load_state_dict strict=True OK")

        model.save_pretrained(output_dir)
        del model
        gc.collect()

    print("Copying processor from nvidia/parakeet-ctc-1.1b ...")
    proc = ParakeetProcessor.from_pretrained("nvidia/parakeet-ctc-1.1b")
    proc.save_pretrained(output_dir)

    print("Running inference ...")
    model = ParakeetForRNNT.from_pretrained(output_dir)
    model.eval()
    audio, sr = load_audio(AUDIO_FILE)
    inputs = proc(audio, sampling_rate=sr, return_tensors="pt")
    with torch.no_grad():
        tokens = model.generate(**inputs)
    print(f"  RNNT transcript: {proc.batch_decode(tokens)[0]!r}")


def test_tdt(output_dir):
    print("\n" + "=" * 60)
    print("TDT conversion + inference test")
    print("=" * 60)

    nemo_path = cached_file("nvidia/parakeet-tdt-0.6b-v3", "parakeet-tdt-0.6b-v3.nemo")

    with tempfile.TemporaryDirectory() as tmpdir:
        model_files = extract_nemo(nemo_path, tmpdir)
        nemo_config = yaml.safe_load(open(model_files["model_config"]))

        print("Converting config ...")
        encoder_config = convert_encoder_config(nemo_config)
        tdt_config = convert_tdt_config(nemo_config, encoder_config)
        print(f"  vocab_size={tdt_config.vocab_size}, blank_id={tdt_config.blank_id}, durations={tdt_config.durations}")

        print("Converting weights ...")
        state_dict = load_and_convert_state_dict(model_files, extra_mapping=NEMO_TO_HF_RNNT_EXTRA_MAPPING)
        print(f"  {len(state_dict)} tensors converted")

        print("Loading model (strict=False to see gaps) ...")
        with torch.device("meta"):
            model = ParakeetForTDT(tdt_config)
        missing, unexpected = model.load_state_dict(state_dict, strict=False, assign=True)
        if missing:
            print(f"  MISSING ({len(missing)}): {missing[:10]}")
        if unexpected:
            print(f"  UNEXPECTED ({len(unexpected)}): {unexpected[:10]}")
        if not missing and not unexpected:
            print("  load_state_dict strict=True OK")

        model.save_pretrained(output_dir)
        del model
        gc.collect()

    print("Copying processor from nvidia/parakeet-tdt-0.6b-v3 ...")
    proc = AutoProcessor.from_pretrained("nvidia/parakeet-tdt-0.6b-v3")
    proc.save_pretrained(output_dir)

    print("Running inference ...")
    model = ParakeetForTDT.from_pretrained(output_dir)
    model.eval()
    audio, sr = load_audio(AUDIO_FILE)
    inputs = proc(audio, sampling_rate=sr, return_tensors="pt")
    with torch.no_grad():
        tokens = model.generate(**inputs)
    print(f"  TDT transcript: {proc.batch_decode(tokens)[0]!r}")


if __name__ == "__main__":
    os.makedirs("/tmp/parakeet_hf/rnnt", exist_ok=True)
    os.makedirs("/tmp/parakeet_hf/tdt", exist_ok=True)

    try:
        test_rnnt("/tmp/parakeet_hf/rnnt")
    except Exception as e:
        print(f"RNNT FAILED: {e}")
        import traceback
        traceback.print_exc()

    try:
        test_tdt("/tmp/parakeet_hf/tdt")
    except Exception as e:
        print(f"TDT FAILED: {e}")
        import traceback
        traceback.print_exc()
