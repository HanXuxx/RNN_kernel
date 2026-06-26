from __future__ import annotations

import argparse
import time

import torch

from rnn_benchmark import RNNBenchmarkModel, resolve_a100_block_threads


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--implementation",
        choices=[
            "a100_gru_h256",
            "a100_gru_h256_recurrent_kernel",
            "a100_gru_h256_recompute",
            "a100_gru_h256_tiled_recurrent",
            "a100_gru_h256_split_recurrent",
            "a100_gru_h256_split4_recurrent",
            "a100_gru_h256_coop_split4",
            "a100_gru_h256_coop_split2",
            "a100_gru_h256_coop_split2_cached",
            "a100_gru_h256_coop_split2_specialized",
            "a100_gru_h256_coop_split2_cached_local",
            "a100_gru_h256_coop_split2_gate_cache",
            "a100_gru_h256_coop_split2_persistent",
            "a100_gru_h256_coop_split2_persistent_state",
            "a100_gru_h256_coop_split2_persistent_state_local",
            "a100_gru_h256_coop_split4_persistent_state",
            "a100_gru_h256_coop_split8_persistent_state",
            "a100_gru_h256_coop_split16_persistent_state",
            "a100_gru_h256_coop_split16_persistent_state_gate_cache",
            "a100_gru_h256_coop_split16_persistent_state_gate_cache_tiled",
            "a100_gru_h256_coop_split16_persistent_state_gate_cache_tiled_cta6",
            "a100_gru_h256_coop_split16_persistent_state_gate_cache_tiled_htile2",
            "a100_gru_h256_coop_split16_persistent_state_gate_cache_tiled_htile4",
            "a100_gru_h256_coop_split16_persistent_state_gate_cache_tiled_htile4_compact",
            "a100_gru_h256_coop_split16_persistent_state_gate_cache_tiled_htile4_compact_hoist",
            "a100_gru_h256_coop_split16_persistent_state_gate_cache_tiled_htile4_compact_hoist_row4",
            "a100_gru_h256_coop_split16_persistent_state_gate_cache_tiled_weight_shmem_htile4_compact_hoist_row4",
            "a100_gru_h256_coop_split16_persistent_state_gate_cache_tiled_weight_shmem_split0_keep_htile4_compact_hoist_row4",
            "a100_gru_h256_coop_split6_persistent_state_gate_cache_tiled_weight_shmem_split0_keep_htile4_compact_hoist_row4",
            "a100_gru_h256_coop_split6_persistent_state_gate_cache_tiled_weight_shmem_split0_keep_htile4_compact_hoist_row4_ldg",
            "a100_gru_h256_coop_split6_persistent_state_gate_cache_tiled_weight_shmem_split0_keep_htile4_compact_hoist_row4_parallel_update",
            "a100_gru_h256_coop_split6_persistent_state_gate_cache_tiled_weight_shmem_split0_keep_htile4_compact_hoist_row4_prev_cache",
            "a100_gru_h256_coop_split6_persistent_state_gate_cache_tiled_weight_shmem_split0_keep_htile4_compact_hoist_row4_pack_prev",
            "a100_gru_h256_coop_split5_persistent_state_gate_cache_tiled_weight_shmem_split0_keep_htile4_compact_hoist_row4",
            "a100_gru_h256_coop_split12_persistent_state_gate_cache_tiled_weight_shmem_split0_keep_htile4_compact_hoist_row4",
            "a100_gru_h256_coop_split12_persistent_state_gate_cache_tiled_weight_shmem_split0_keep_unroll8_htile4_compact_hoist_row4",
            "a100_gru_h256_coop_split24_persistent_state_gate_cache_tiled_weight_shmem_split0_keep_htile4_compact_hoist_row4",
            "a100_gru_h256_coop_split16_persistent_state_gate_cache_tiled_weight_shmem_split0_keep_own_shmem_htile4_compact_hoist_row4",
            "a100_gru_h256_coop_split16_persistent_state_gate_cache_tiled_weight_shmem_htile4_compact_hoist_qwarp",
            "a100_gru_h256_coop_split16_persistent_state_gate_cache_tiled_weight_shmem_htile4_compact_hoist_row3",
            "a100_gru_h256_coop_split16_persistent_state_gate_cache_tiled_weight_shmem_htile4_compact_hoist_row4_forward_hidden_shmem",
            "a100_gru_h256_coop_split16_persistent_state_gate_cache_tiled_weight_shmem_htile4_compact_hoist_row4_forward_weight_shmem",
            "a100_gru_h256_coop_split8_persistent_state_gate_cache_tiled_htile4_compact_hoist_row4",
            "a100_gru_h256_coop_split16_persistent_state_gate_cache_tiled_htile8_compact",
            "a100_gru_h256_coop_split16_persistent_state_grad_coeff_cache_tiled",
            "a100_gru_h256_coop_split32_persistent_state_gate_cache_tiled",
            "a100_gru_h256_coop_split16_persistent_state_gate_cache_parallel_update",
            "a100_gru_h256_coop_split16_persistent_state_gate_cache_cta8",
            "a100_gru_h256_coop_split16_persistent_state_global_gates",
            "a100_gru_h256_coop_split32_persistent_state",
        ],
        default="a100_gru_h256_recurrent_kernel",
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--input-dim", type=int, default=9)
    parser.add_argument("--a100-block-threads", type=int, default=0)
    parser.add_argument("--warmup-steps", type=int, default=2)
    return parser.parse_args()


def run_step(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    x: torch.Tensor,
    y: torch.Tensor,
) -> float:
    optimizer.zero_grad(set_to_none=True)
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    pred = model(x)
    loss = torch.nn.functional.mse_loss(pred, y)
    loss.backward()
    optimizer.step()
    end.record()
    torch.cuda.synchronize()
    return float(start.elapsed_time(end))


def main() -> None:
    args = parse_args()
    args.a100_block_threads = resolve_a100_block_threads(
        args.implementation,
        args.a100_block_threads,
    )
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(2071)

    device = torch.device("cuda")
    model = RNNBenchmarkModel(
        cell_type="GRU",
        input_dim=args.input_dim,
        hidden_size=256,
        num_layers=1,
        implementation=args.implementation,
        a100_block_threads=args.a100_block_threads,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    x = torch.randn(args.batch_size, args.seq_len, args.input_dim, device=device)
    y = torch.randn(args.batch_size, args.seq_len, device=device)

    # 先触发 NVRTC 编译和缓存分配，避免 profiler 把初始化计入训练 step。
    for _ in range(args.warmup_steps):
        run_step(model, optimizer, x, y)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    torch.cuda.cudart().cudaProfilerStart()
    wall_start = time.perf_counter()
    step_ms = run_step(model, optimizer, x, y)
    torch.cuda.synchronize()
    wall_ms = (time.perf_counter() - wall_start) * 1000.0
    torch.cuda.cudart().cudaProfilerStop()

    peak_gb = torch.cuda.max_memory_allocated() / 1024**3
    print(
        f"implementation={args.implementation} "
        f"seq_len={args.seq_len} step_ms={step_ms:.3f} "
        f"wall_ms={wall_ms:.3f} peak_mem={peak_gb:.2f}GB"
    )


if __name__ == "__main__":
    main()
