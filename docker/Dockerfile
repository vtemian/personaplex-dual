FROM python:3.12-slim

WORKDIR /app

# cu130 to match the box's driver (CUDA 13.0 on GB10). A cu128 build has no
# kernels for it and falls back to the CPU silently, which looks like a very
# slow but healthy run. This layer is 4.4 GB with its nvidia-* deps and never
# changes; everything cheap goes below it.
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cu130 \
    torch==2.13.0

# sphn 0.2.1 needed CMAKE_POLICY_VERSION_MINIMUM=3.5 on the box (cmake 3.28 vs
# opus cmake < 3.5, docs/results/moshi-probe.md, 2026-08-13). The same source
# build also needs g++, make, and git: 2026-08-19 smoke died with ToolNotFound
# c++ and CMAKE_MAKE_PROGRAM unset. gcc stays for Triton's first backward
# pass, same reason as docker/train.
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      gcc g++ make git libc6-dev cmake pkg-config \
 && rm -rf /var/lib/apt/lists/*

# The box moshi venv, not the helium image's peft 0.20.0. 0.17.1 is the pin
# out-moshi1/model.pt was trained under; moshi4's intended variable is the data.
ENV CMAKE_POLICY_VERSION_MINIMUM=3.5
# moshi's metadata caps torch <2.10, which the cu130 pin above can never satisfy:
# with a torch constraint the resolver replaces the pin (2.9.1, no GB10 kernels,
# silent CPU fallback), and with -c it is ResolutionImpossible (both 2026-08-19
# smokes). So moshi installs --no-deps and its real imports are pinned by hand,
# measured on the box venv the same day: tts.py needs sphn, torch, safetensors,
# sentencepiece, tqdm; loaders.py adds huggingface-hub (which brings tqdm); the
# heavy extras (bitsandbytes, sounddevice, aiohttp) are imported by neither the
# load path nor TTS and are deliberately absent.
RUN pip install --no-cache-dir --no-deps moshi==0.2.13 \
 && pip install --no-cache-dir \
      sphn==0.2.1 \
      "peft==0.17.1" \
      "numpy>=1.26,<2.3" \
      "safetensors>=0.4.0,<0.8.0" \
      "huggingface-hub>=0.24,<1.0" \
      "einops>=0.7,<0.9" \
      "sentencepiece>=0.2.0,<0.3"

# moshiko 15.8 GB (15.4 GB bf16 + 385 MB Mimi sidecar) and TTS 4.07 GB, baked
# rather than fetched per run. The one expresso voice tts_corpus.VOICE names,
# because get_voice_path hits the hub and HF_HUB_OFFLINE=1 would fail it.
ENV HF_HOME=/opt/hf
RUN python -c "\
from huggingface_hub import snapshot_download; \
from moshi.models.loaders import CheckpointInfo; \
from moshi.models.tts import DEFAULT_DSM_TTS_REPO, DEFAULT_DSM_TTS_VOICE_REPO; \
CheckpointInfo.from_hf_repo('kyutai/moshiko-pytorch-bf16'); \
CheckpointInfo.from_hf_repo(DEFAULT_DSM_TTS_REPO); \
snapshot_download(DEFAULT_DSM_TTS_VOICE_REPO, allow_patterns=['expresso/ex03-ex01_happy_001_channel1_334s.wav*'])" \
 && chmod -R a+rX /opt/hf

# The job runs as the submitting uid, which has no /etc/passwd entry, so HOME
# is unset and torch's Triton cache lands on /.triton - killing the run after
# the first backward pass, which reads as a mid-training crash. PYTHONPATH
# because the container's working directory is the box's shared dir, so a
# relative one cannot work.
ENV HOME=/tmp \
    TRITON_CACHE_DIR=/tmp/triton \
    HF_HUB_OFFLINE=1 \
    PYTHONPATH=/app/src:/app \
    PYTHONUNBUFFERED=1

# The distribution is `sparks-dgx`; the import package is `sparks`. Pinned to an
# exact version, not a range: Docker keys the layer on the instruction text, so
# a range never changes and the cache would keep serving whatever it first
# resolved. 0.1.1 is the floor for two fixes this image depends on - the
# staleness grace, without which a run's last epoch reaches Prometheus and no
# query can reach it, and the run-id cleanup, without which a second phase in
# one process attaches to the first phase's dead run.
RUN pip install --no-cache-dir "sparks-dgx==0.1.1"

# Last, so a code change re-pushes ~1 MB. vendor/ is chalkboard's src at CHALKBOARD_SHA.
COPY vendor /app/vendor
COPY src /app/src
ENV PYTHONPATH=/app/src:/app/vendor:/app
