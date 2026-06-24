#!/usr/bin/env bash

set -euo pipefail

source scripts/env.sh
mkdir -p results profiles

# 总 step 计时尽量少插入同步，用于判断真实训练吞吐。
.venv/bin/python rnn_benchmark.py \
  --device cuda \
  --cell-types GRU,LSTM \
  --hidden-sizes 96,112,120,128,129,130,144,160,192,256 \
  --dataset-batches 32 \
  --warmup-steps 3 \
  --timed-steps 10 \
  --output-csv results/a100_baseline_total.csv

# 分段计时会在每个 step 后同步，只用于定位瓶颈来源，不直接替代总吞吐结果。
.venv/bin/python rnn_benchmark.py \
  --device cuda \
  --cell-types GRU \
  --hidden-sizes 128,129,130,160,256 \
  --dataset-batches 8 \
  --warmup-steps 2 \
  --timed-steps 5 \
  --breakdown-timing \
  --output-csv results/a100_gru_breakdown.csv

