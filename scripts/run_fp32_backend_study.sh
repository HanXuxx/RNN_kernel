#!/usr/bin/env bash

set -euo pipefail

source scripts/env.sh
mkdir -p results

COMMON_ARGS=(
  --device cuda
  --cell-types GRU
  --hidden-sizes 128,129,130,160
  --dataset-batches 8
  --warmup-steps 2
  --timed-steps 5
)

# 默认 fp32 路径：TF32 关闭，cuDNN benchmark 按 deterministic 自动控制。
.venv/bin/python rnn_benchmark.py \
  "${COMMON_ARGS[@]}" \
  --output-csv results/a100_fp32_default.csv

# 关闭 cuDNN benchmark，验证断崖是否由算法搜索缓存造成。
.venv/bin/python rnn_benchmark.py \
  "${COMMON_ARGS[@]}" \
  --cudnn-benchmark off \
  --output-csv results/a100_fp32_cudnn_benchmark_off.csv

# deterministic 会限制 cuDNN 算法选择，但仍保持 fp32。
.venv/bin/python rnn_benchmark.py \
  "${COMMON_ARGS[@]}" \
  --deterministic \
  --output-csv results/a100_fp32_deterministic.csv

# torch.compile 保持 fp32；cuDNN RNN 通常不透明，但需要实测确认。
.venv/bin/python rnn_benchmark.py \
  --device cuda \
  --cell-types GRU \
  --hidden-sizes 128,130 \
  --dataset-batches 4 \
  --warmup-steps 2 \
  --timed-steps 3 \
  --compile-model \
  --output-csv results/a100_fp32_compile.csv

# 禁用 cuDNN 后会走 PyTorch 原生路径；完整 seq_len=8000 预期不可用。
# 这里只用短序列判断其是否有成为优化路线的可能。
.venv/bin/python rnn_benchmark.py \
  --device cuda \
  --cell-types GRU \
  --hidden-sizes 128,130 \
  --batch-size 16 \
  --seq-len 256 \
  --num-layers 4 \
  --dataset-batches 4 \
  --warmup-steps 1 \
  --timed-steps 2 \
  --disable-cudnn \
  --output-csv results/a100_fp32_cudnn_disabled_short_seq.csv

# 相同 tokens/step 的形状敏感性：不改变浮点精度，但会改变 BPTT 形状。
.venv/bin/python rnn_benchmark.py \
  --device cuda \
  --cell-types GRU \
  --hidden-sizes 128,130 \
  --batch-size 32 \
  --seq-len 4000 \
  --num-layers 4 \
  --dataset-batches 8 \
  --warmup-steps 2 \
  --timed-steps 5 \
  --output-csv results/a100_fp32_shape_bs32_seq4000.csv

.venv/bin/python rnn_benchmark.py \
  --device cuda \
  --cell-types GRU \
  --hidden-sizes 128,130 \
  --batch-size 8 \
  --seq-len 16000 \
  --num-layers 4 \
  --dataset-batches 6 \
  --warmup-steps 2 \
  --timed-steps 3 \
  --output-csv results/a100_fp32_shape_bs8_seq16000.csv

