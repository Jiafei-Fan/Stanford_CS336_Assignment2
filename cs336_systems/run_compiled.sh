#!/usr/bin/env bash
set -euo pipefail

# Toggle to 1 to run that model size.
run_small=0
run_medium=0
run_large=0
run_xl=0
run_2_7B=1  # 2.7B

context_lengths=(128 256 512)

# Benchmark mode toggles.
run_only_partial=1
run_with_warmup=1
run_feedfoward=1
run_both_feedfoward_backward=0
mixed_precision=1
write_benchmark_csv=0
dump_memory_snapshot=0

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
RUN_PY="${SCRIPT_DIR}/run_compiled.py"
UV_BIN="${UV_BIN:-uv}"
export UV_MANAGED_PYTHON="${UV_MANAGED_PYTHON:-1}"
export UV_PYTHON="${UV_PYTHON:-3.12}"

run_config() {
  local model_size="$1"
  local d_model="$2"
  local d_ff="$3"
  local num_layers="$4"
  local num_heads="$5"
  local context_length="$6"
  local extra_args=()

  if [[ "${run_only_partial}" -eq 1 ]]; then
    extra_args+=(--run_only_partial)
  fi
  if [[ "${run_with_warmup}" -eq 1 ]]; then
    extra_args+=(--run_with_warmup)
  fi
  if [[ "${run_feedfoward}" -eq 1 ]]; then
    extra_args+=(--run_feedfoward)
  fi
  if [[ "${run_both_feedfoward_backward}" -eq 1 ]]; then
    extra_args+=(--run_both_feedfoward_backward)
  fi
  if [[ "${mixed_precision}" -eq 1 ]]; then
    extra_args+=(--mixed_precision)
  fi
  if [[ "${write_benchmark_csv}" -eq 1 ]]; then
    extra_args+=(--write_benchmark_csv)
  fi
  if [[ "${dump_memory_snapshot}" -eq 1 ]]; then
    extra_args+=(--dump_memory_snapshot)
  fi

  echo "[run_compiled.sh] Start ${model_size}: d_model=${d_model}, d_ff=${d_ff}, num_layers=${num_layers}, num_heads=${num_heads}, context_length=${context_length}, mixed_precision=${mixed_precision}, write_benchmark_csv=${write_benchmark_csv}, dump_memory_snapshot=${dump_memory_snapshot}"
  "${UV_BIN}" run --project "${PROJECT_ROOT}" python "${RUN_PY}" \
    --model_size "${model_size}" \
    --d_model "${d_model}" \
    --d_ff "${d_ff}" \
    --num_layers "${num_layers}" \
    --num_heads "${num_heads}" \
    --context_length "${context_length}" \
    "${extra_args[@]}"
}

any_run=0

if [[ "${run_feedfoward}" -eq 1 && "${run_both_feedfoward_backward}" -eq 1 ]]; then
  echo "[run_compiled.sh] Choose only one: run_feedfoward=1 or run_both_feedfoward_backward=1."
  exit 1
fi

if [[ "${run_small}" -eq 1 ]]; then
  for context_length in "${context_lengths[@]}"; do
    run_config "small" 768 3072 12 12 "${context_length}"
    any_run=1
  done
fi

if [[ "${run_medium}" -eq 1 ]]; then
  for context_length in "${context_lengths[@]}"; do
    run_config "medium" 1024 4096 24 16 "${context_length}"
    any_run=1
  done
fi

if [[ "${run_large}" -eq 1 ]]; then
  for context_length in "${context_lengths[@]}"; do
    run_config "large" 1280 5120 36 20 "${context_length}"
    any_run=1
  done
fi

if [[ "${run_xl}" -eq 1 ]]; then
  for context_length in "${context_lengths[@]}"; do
    run_config "xl" 1600 6400 48 25 "${context_length}"
    any_run=1
  done
fi

if [[ "${run_2_7B}" -eq 1 ]]; then
  for context_length in "${context_lengths[@]}"; do
    run_config "2.7B" 2560 10240 32 32 "${context_length}"
    any_run=1
  done
fi

if [[ "${any_run}" -eq 0 ]]; then
  echo "[run_compiled.sh] No model is selected. Set one or more run_* flags to 1 and rerun."
fi
