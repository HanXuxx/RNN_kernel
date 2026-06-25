#!/usr/bin/env bash

set -euo pipefail

source scripts/env.sh
mkdir -p results

COMMON_SHORT=(
  --device cuda
  --cell-types GRU
  --hidden-sizes 128,130
  --batch-size 16
  --seq-len 256
  --num-layers 4
  --dataset-batches 4
  --warmup-steps 1
  --timed-steps 2
)

.venv/bin/python rnn_benchmark.py \
  --implementation torch \
  "${COMMON_SHORT[@]}" \
  --output-csv results/a100_custom_study_cudnn_seq256.csv

.venv/bin/python rnn_benchmark.py \
  --implementation torch \
  "${COMMON_SHORT[@]}" \
  --disable-cudnn \
  --output-csv results/a100_custom_study_torch_native_seq256.csv

.venv/bin/python rnn_benchmark.py \
  --implementation custom_gru \
  "${COMMON_SHORT[@]}" \
  --output-csv results/a100_custom_study_custom_torch_seq256.csv

.venv/bin/python rnn_benchmark.py \
  --implementation custom_gru_triton \
  "${COMMON_SHORT[@]}" \
  --output-csv results/a100_custom_study_custom_triton_seq256.csv

# 长序列单层估算：量化自定义 pointwise kernel 的 launch 和小 GEMM 成本。
COMMON_LONG_SINGLE_LAYER=(
  --device cuda
  --cell-types GRU
  --hidden-sizes 130
  --batch-size 16
  --seq-len 8000
  --num-layers 1
  --dataset-batches 2
  --warmup-steps 1
  --timed-steps 1
)

.venv/bin/python rnn_benchmark.py \
  --implementation torch \
  "${COMMON_LONG_SINGLE_LAYER[@]}" \
  --output-csv results/a100_custom_study_cudnn_seq8000_layer1.csv

.venv/bin/python rnn_benchmark.py \
  --implementation custom_gru_triton \
  "${COMMON_LONG_SINGLE_LAYER[@]}" \
  --output-csv results/a100_custom_study_custom_triton_seq8000_layer1.csv

