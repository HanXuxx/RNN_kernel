#!/usr/bin/env bash

set -euo pipefail

source scripts/env.sh
mkdir -p results

for chunk_len in 0 4000 2000 1000 500; do
  .venv/bin/python rnn_benchmark.py \
    --device cuda \
    --cell-types GRU \
    --hidden-sizes 128,130 \
    --batch-size 16 \
    --seq-len 8000 \
    --num-layers 4 \
    --sequence-chunk-len "${chunk_len}" \
    --dataset-batches 4 \
    --warmup-steps 2 \
    --timed-steps 3 \
    --output-csv "results/a100_fp32_chunk_${chunk_len}.csv"
done

