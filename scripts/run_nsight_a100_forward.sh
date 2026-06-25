#!/usr/bin/env bash
# 使用项目本地 Nsight Systems 采集 A100 GRU forward 调度信息。

set -euo pipefail

source scripts/env.sh
source scripts/env_nsight.sh

mkdir -p profiles

run_case() {
  local name="$1"
  local variant="$2"
  local hidden_size="$3"
  local seq_len="${4:-256}"
  local output="profiles/${name}"

  nsys profile \
    --force-overwrite=true \
    --trace=cuda,nvtx,osrt \
    --sample=none \
    --cpuctxsw=none \
    --output="${output}" \
    .venv/bin/python scripts/profile_a100_forward.py \
      --variant "${variant}" \
      --hidden-size "${hidden_size}" \
      --seq-len "${seq_len}" \
      --warmup-steps 1 \
      --profile-steps 1

  nsys stats \
    --force-export=true \
    --report cuda_gpu_kern_sum \
    "${output}.nsys-rep"

  .venv/bin/python scripts/extract_nsys_kernel_table.py "${output}.sqlite" --limit 8
}

run_case nsys2024_torch_h256_s256 torch 256 256
run_case nsys2024_a100_fused_h256_s256 a100_fused 256 256
run_case nsys2024_a100_cooperative4_h256_s256 a100_cooperative4 256 256
run_case nsys2024_a100_cooperative_h256_s256 a100_cooperative_h256 256 256
run_case nsys2024_a100_cooperative_h256_cached_shmem_s256 a100_cooperative_h256_cached_shmem 256 256
run_case nsys2024_a100_cooperative_h256_parallel_update_s256 a100_cooperative_h256_parallel_update 256 256
run_case nsys2024_a100_cooperative_h256_shmem_s256 a100_cooperative_h256_shmem 256 256
run_case nsys2024_a100_cooperative_h256_qwarp_shmem_s256 a100_cooperative_h256_qwarp_shmem 256 256
