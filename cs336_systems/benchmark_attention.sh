#!/usr/bin/env bash
set -euo pipefail

batch_size=8
warmup_steps=10
forward_steps=100
backward_steps=100

d_models=(16 32 64 128)
context_lengths=(256 1024 4096 8192 16384)

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
RUN_PY="${SCRIPT_DIR}/run2.py"
UV_BIN="${UV_BIN:-uv}"

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

for d_model in "${d_models[@]}"; do
  for context_length in "${context_lengths[@]}"; do
    echo "[run2.sh] Start benchmark: batch_size=${batch_size}, d_model=${d_model}, context_length=${context_length}"
    "${UV_BIN}" run --project "${PROJECT_ROOT}" python "${RUN_PY}" \
      --batch_size "${batch_size}" \
      --d_model "${d_model}" \
      --context_length "${context_length}" \
      --warmup_steps "${warmup_steps}" \
      --forward_steps "${forward_steps}" \
      --backward_steps "${backward_steps}"
  done
done
