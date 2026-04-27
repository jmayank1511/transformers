<!--Copyright 2025 The NVIDIA NeMo Team and The HuggingFace Inc. team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->
*This model was released on {release_date} and added to Hugging Face Transformers on 2025-09-25.*

<div class="flex flex-wrap space-x-1">
<img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-DE3412?style=flat&logo=pytorch&logoColor=white">
<img alt="SDPA" src="https://img.shields.io/badge/SDPA-DE3412?style=flat&logo=pytorch&logoColor=white">
</div>

# Parakeet

## Overview

Parakeet models, [introduced by NVIDIA NeMo](https://developer.nvidia.com/blog/pushing-the-boundaries-of-speech-recognition-with-nemo-parakeet-asr-models/), are models that combine a [Fast Conformer](https://docs.nvidia.com/nemo-framework/user-guide/latest/nemotoolkit/asr/models.html#fast-conformer) encoder with connectionist temporal classification (CTC), recurrent neural network transducer (RNNT) or token and duration transducer (TDT) decoder for automatic speech recognition.

**Model Architecture**

- **Fast Conformer Encoder**: A linearly scalable Conformer architecture that processes mel-spectrogram features and reduces sequence length through subsampling. This is more efficient version of the Conformer Encoder found in [FastSpeech2Conformer](./fastspeech2_conformer.md) (see [`ParakeetEncoder`] for the encoder implementation and details).
- [**ParakeetForCTC**](#parakeetforctc): a Fast Conformer Encoder + a CTC decoder
  - **CTC Decoder**: Simple but effective decoder consisting of:
    - 1D convolution projection from encoder hidden size to vocabulary size (for optimal NeMo compatibility).
    - CTC loss computation for training.
    - Greedy CTC decoding for inference.

The original implementation can be found in [NVIDIA NeMo](https://github.com/NVIDIA/NeMo).
Model checkpoints are to be found under [the NVIDIA organization](https://huggingface.co/nvidia/models?search=parakeet).

This model was contributed by [Nithin Rao Koluguri](https://huggingface.co/nithinraok), [Eustache Le Bihan](https://huggingface.co/eustlb) and [Eric Bezzam](https://huggingface.co/bezzam).

## Usage

### Basic usage

<hfoptions id="usage">
<hfoption id="Pipeline">

```py
from transformers import pipeline

pipe = pipeline("automatic-speech-recognition", model="nvidia/parakeet-ctc-1.1b")
out = pipe("https://huggingface.co/datasets/hf-internal-testing/dummy-audio-samples/resolve/main/bcn_weather.mp3")
print(out)
```

</hfoption>
<hfoption id="AutoModel">

```py
from transformers import AutoModelForCTC, AutoProcessor
from datasets import load_dataset, Audio
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

processor = AutoProcessor.from_pretrained("nvidia/parakeet-ctc-1.1b")
model = AutoModelForCTC.from_pretrained("nvidia/parakeet-ctc-1.1b", dtype="auto", device_map=device)

ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
ds = ds.cast_column("audio", Audio(sampling_rate=processor.feature_extractor.sampling_rate))
speech_samples = [el['array'] for el in ds["audio"][:5]]

inputs = processor(speech_samples, sampling_rate=processor.feature_extractor.sampling_rate)
inputs.to(model.device, dtype=model.dtype)
outputs = model.generate(**inputs)
print(processor.batch_decode(outputs))
```

</hfoption>
</hfoptions>

### Cache-aware streaming

Cache-aware Parakeet models are purpose-built for real-time streaming ASR. They are trained with a sliding-window attention context (`att_context_size`), which limits how many past and future encoder frames each frame can attend to. A typical multi-lookahead config looks like `[[70, 13], [70, 6], [70, 1], [70, 0]]` — each pair is `[left_context, right_context]`. Lower right context means lower latency; `[70, 0]` is fully causal.

#### Chunk-by-chunk streaming inference

The encoder processes audio one chunk at a time. A KV cache carries state between chunks so the model never re-processes past audio. CTC token IDs are accumulated across chunks and decoded at the end (or at any step for partial results).

```python
import numpy as np
import torch
from transformers import AutoProcessor, ParakeetForCTC

processor = AutoProcessor.from_pretrained("nvidia/parakeet-ctc-streaming")
model = ParakeetForCTC.from_pretrained("nvidia/parakeet-ctc-streaming")
model.eval()
encoder = model.encoder

# --- Derive chunk parameters from model config ---
hop = processor.feature_extractor.hop_length          # samples per mel frame (160)
n_fft_half = processor.feature_extractor.n_fft // 2  # lookahead samples for STFT edge (256)
S = encoder.config.subsampling_factor                 # 8

# Choose a context size; default is the first (largest lookahead) entry
ctx = encoder.config.att_context_size
if isinstance(ctx[0], list):
    ctx = ctx[0]   # e.g. [70, 6]
R = ctx[1]         # right context in encoder frames

# Pre-encode cache: a few mel frames prepended to each subsequent chunk so
# the subsampling Conv2d has proper left context.
pre_encode_cache_mel  = S + 1                              # 9 mel frames
pre_cache_samples     = pre_encode_cache_mel * hop         # 1440 audio samples
drop_extra_pre_encoded = 1 + (pre_encode_cache_mel - 1) // S  # encoder frames to drop after subsampling

# Chunk sizes in mel frames / audio samples
first_chunk_mel     = 1 + S * R                            # e.g. 49 for R=6
first_chunk_samples = first_chunk_mel * hop
chunk_mel           = S * (R + 1)                          # e.g. 56 for R=6
chunk_samples       = chunk_mel * hop
target_mel_len      = pre_encode_cache_mel + chunk_mel     # e.g. 65

# --- Split audio into chunks ---
# `audio` is a float32 numpy array at processor.feature_extractor.sampling_rate
audio = ...   # load your audio here

chunks = [audio[:first_chunk_samples]]
remaining = audio[first_chunk_samples:]
i = 0
while i < len(remaining):
    chunks.append(remaining[i : i + chunk_samples])
    i += chunk_samples

# --- Stream chunk by chunk ---
cache      = encoder.get_initial_cache_state(batch_size=1)
past_audio = np.zeros(pre_cache_samples, dtype=np.float32)
accumulated_ids = None

for chunk_idx, chunk in enumerate(chunks):
    is_first = chunk_idx == 0

    if is_first:
        # No pre-encode cache on the first chunk. STFT center=True can add one
        # extra mel frame — trim to exactly first_chunk_mel.
        inputs = processor([chunk], return_tensors="pt")
        if inputs["input_features"].shape[1] > first_chunk_mel:
            inputs["input_features"] = inputs["input_features"][:, :first_chunk_mel, :]
            inputs["attention_mask"] = inputs["attention_mask"][:, :first_chunk_mel]
        drop = 0
    else:
        # Prepend past audio for Conv2d left context. Append a few samples from
        # the next chunk so the last mel frame is computed from real audio rather
        # than zero-padding, then trim the result to exactly target_mel_len.
        next_start = first_chunk_samples + chunk_idx * chunk_samples
        lookahead  = audio[next_start : next_start + n_fft_half]
        extended   = np.concatenate([past_audio, chunk, lookahead])
        inputs = processor([extended], return_tensors="pt")
        if inputs["input_features"].shape[1] > target_mel_len:
            inputs["input_features"] = inputs["input_features"][:, :target_mel_len, :]
            inputs["attention_mask"] = inputs["attention_mask"][:, :target_mel_len]
        drop = drop_extra_pre_encoded

    with torch.no_grad():
        enc_out = encoder(
            **inputs,
            use_cache=True,
            att_context_size=ctx,
            cache_last_channel=cache["cache_last_channel"],
            cache_last_time=cache["cache_last_time"],
            cache_last_channel_len=cache["cache_last_channel_len"],
            drop_extra_pre_encoded=drop,
        )
        logits = model.ctc_head(
            enc_out.last_hidden_state.transpose(1, 2)
        ).transpose(1, 2)

    past_audio = chunk[-pre_cache_samples:]
    cache = {
        "cache_last_channel":     enc_out.cache_last_channel,
        "cache_last_time":        enc_out.cache_last_time,
        "cache_last_channel_len": enc_out.cache_last_channel_len,
    }

    # Accumulate raw argmax IDs; CTC-decode the full sequence at each step.
    # Concatenating across chunks lets CTC naturally collapse tokens that
    # span a chunk boundary without duplicates.
    chunk_ids = logits.argmax(-1).squeeze(0)
    accumulated_ids = (
        chunk_ids if accumulated_ids is None
        else torch.cat([accumulated_ids, chunk_ids])
    )

    # Partial result after this chunk (optional):
    partial = processor.batch_decode(
        accumulated_ids.unsqueeze(0), skip_special_tokens=True
    )[0].strip()
    print(f"[chunk {chunk_idx + 1}] {partial!r}")

# Final transcription
transcription = processor.batch_decode(
    accumulated_ids.unsqueeze(0), skip_special_tokens=True
)[0].strip()
print(transcription)
```

#### Choosing a context size

Smaller right context reduces latency at the cost of accuracy. Pass `att_context_size` to `encoder()` to select any trained context:

```python
# Zero lookahead — lowest latency, fully causal
enc_out = encoder(**inputs, use_cache=True, att_context_size=[70, 0], ...)
```

The chunk size parameters (`first_chunk_mel`, `chunk_mel`, `target_mel_len`) must be recomputed for each context size since they depend on `R = att_context_size[1]`.

#### One-shot inference on full audio

Streaming models also work offline — `pipeline` and `generate` use the full audio at once. This gives the best accuracy since every frame has unrestricted right context:

```python
from transformers import pipeline

pipe = pipeline("automatic-speech-recognition", model="nvidia/parakeet-ctc-streaming")
print(pipe("audio.wav"))
```

### Making The Model Go Brrr

Parakeet supports full-graph compilation with CUDA graphs! This optimization is most effective when you know the maximum audio length you want to transcribe. The key idea is using static input shapes to avoid recompilation. For example, if you know your audio will be under 30 seconds, you can use the processor to pad all inputs to 30 seconds, preparing consistent input features and attention masks. See the example below!

```python
from transformers import AutoModelForCTC, AutoProcessor
from datasets import load_dataset, Audio
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

processor = AutoProcessor.from_pretrained("nvidia/parakeet-ctc-1.1b")
model = AutoModelForCTC.from_pretrained("nvidia/parakeet-ctc-1.1b", dtype="auto", device_map=device)

ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
ds = ds.cast_column("audio", Audio(sampling_rate=processor.feature_extractor.sampling_rate))
speech_samples = [el['array'] for el in ds["audio"][:5]]

# Compile the generate method with fullgraph and CUDA graphs
model.generate = torch.compile(model.generate, fullgraph=True, mode="reduce-overhead")

# let's define processor kwargs to pad to 30 seconds
processor_kwargs = {
    "padding": "max_length",
    "max_length": 30 * processor.feature_extractor.sampling_rate,
}

# Define a timing context using CUDA events
class TimerContext:
    def __init__(self, name="Execution"):
        self.name = name
        self.start_event = None
        self.end_event = None
        
    def __enter__(self):
        # Use CUDA events for more accurate GPU timing
        self.start_event = torch.cuda.Event(enable_timing=True)
        self.end_event = torch.cuda.Event(enable_timing=True)
        self.start_event.record()
        return self

    def __exit__(self, *args):
        self.end_event.record()
        torch.cuda.synchronize()
        elapsed_time = self.start_event.elapsed_time(self.end_event) / 1000.0
        print(f"{self.name} time: {elapsed_time:.4f} seconds")


inputs = processor(speech_samples[0], **processor_kwargs)
inputs.to(device, dtype=model.dtype)
print("\n" + "="*50)
print("First generation - compiling...")
# Generate with the compiled model
with TimerContext("First generation"):
    outputs = model.generate(**inputs)
print(processor.batch_decode(outputs))

inputs = processor(speech_samples[1], **processor_kwargs)
inputs.to(device, dtype=model.dtype)
print("\n" + "="*50)
print("Second generation - recording CUDA graphs...")
with TimerContext("Second generation"):
    outputs = model.generate(**inputs)
print(processor.batch_decode(outputs))

inputs = processor(speech_samples[2], **processor_kwargs)
inputs.to(device, dtype=model.dtype)
print("\n" + "="*50)
print("Third generation - fast !!!")
with TimerContext("Third generation"):
    outputs = model.generate(**inputs)
print(processor.batch_decode(outputs))

inputs = processor(speech_samples[3], **processor_kwargs)
inputs.to(device, dtype=model.dtype)
print("\n" + "="*50)
print("Fourth generation - still fast !!!")
with TimerContext("Fourth generation"):
    outputs = model.generate(**inputs)
print(processor.batch_decode(outputs))
```

### Training

```python
from transformers import AutoModelForCTC, AutoProcessor
from datasets import load_dataset, Audio
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

processor = AutoProcessor.from_pretrained("nvidia/parakeet-ctc-1.1b")
model = AutoModelForCTC.from_pretrained("nvidia/parakeet-ctc-1.1b", dtype="auto", device_map=device)

ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
ds = ds.cast_column("audio", Audio(sampling_rate=processor.feature_extractor.sampling_rate))
speech_samples = [el['array'] for el in ds["audio"][:5]]
text_samples = [el for el in ds["text"][:5]]

# passing `text` to the processor will prepare inputs' `labels` key
inputs = processor(audio=speech_samples, text=text_samples, sampling_rate=processor.feature_extractor.sampling_rate)
inputs.to(device, dtype=model.dtype)

outputs = model(**inputs)
outputs.loss.backward()
```

## ParakeetTokenizer

[[autodoc]] ParakeetTokenizer

## ParakeetFeatureExtractor

[[autodoc]] ParakeetFeatureExtractor
    - __call__

## ParakeetProcessor

[[autodoc]] ParakeetProcessor
    - __call__
    - batch_decode
    - decode

## ParakeetEncoderConfig

[[autodoc]] ParakeetEncoderConfig

## ParakeetCTCConfig

[[autodoc]] ParakeetCTCConfig

## ParakeetEncoderModelOutput

[[autodoc]] models.parakeet.modeling_parakeet.ParakeetEncoderModelOutput

## ParakeetCTCModelOutput

[[autodoc]] models.parakeet.modeling_parakeet.ParakeetCTCModelOutput

## ParakeetEncoder

[[autodoc]] ParakeetEncoder

## ParakeetForCTC

[[autodoc]] ParakeetForCTC
