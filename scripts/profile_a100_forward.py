#!/usr/bin/env python3
"""给 Nsight CLI 使用的 A100 GRU forward 单目标 profiling 驱动。"""

import argparse
import time

import torch

from rnn_kernel.a100 import (
    a100_gru_forward_layer,
    a100_gru_forward_layer_precompute_input_cooperative,
    a100_gru_forward_layer_precompute_input_cooperative_h256,
    a100_gru_forward_layer_precompute_input_cooperative_h256_cached_shmem,
    a100_gru_forward_layer_precompute_input_cooperative_h256_parallel_update,
    a100_gru_forward_layer_precompute_input_cooperative_h256_qwarp_shmem,
    a100_gru_forward_layer_precompute_input_cooperative_h256_shmem,
    a100_gru_forward_layer_precompute_input_fused,
    a100_gru_forward_layer_precompute_input_fused_pingpong,
    a100_gru_forward_layer_precompute_input_fused_specialized,
    a100_gru_forward_layer_precompute_input,
    a100_gru_forward_layer_precompute_input_subwarp,
)
from rnn_kernel.cuda_gru_forward import cuda_gru_forward_layer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Profile one GRU forward variant.")
    parser.add_argument(
        "--variant",
        choices=[
            "torch",
            "generic_cuda",
            "a100",
            "a100_precompute",
            "a100_fused",
            "a100_fused_pingpong",
            "a100_fused_specialized",
            "a100_cooperative2",
            "a100_cooperative4",
            "a100_cooperative_h256",
            "a100_cooperative_h256_cached_shmem",
            "a100_cooperative_h256_parallel_update",
            "a100_cooperative_h256_qwarp_shmem",
            "a100_cooperative_h256_shmem",
            "a100_subwarp8",
            "a100_subwarp16",
        ],
        default="a100_precompute",
    )
    parser.add_argument("--hidden-size", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--input-dim", type=int, default=9)
    parser.add_argument("--warmup-steps", type=int, default=2)
    parser.add_argument("--profile-steps", type=int, default=1)
    parser.add_argument("--block-threads", type=int, default=1024)
    parser.add_argument("--cooperative-block-threads", type=int, default=1024)
    parser.add_argument("--h256-block-threads", type=int, default=704)
    parser.add_argument("--seed", type=int, default=2029)
    return parser.parse_args()


def make_runner(args: argparse.Namespace):
    device = torch.device("cuda")
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = True

    gru = torch.nn.GRU(
        input_size=args.input_dim,
        hidden_size=args.hidden_size,
        num_layers=1,
        batch_first=True,
    ).to(device)
    x = torch.randn(args.batch_size, args.seq_len, args.input_dim, device=device)
    h0 = torch.randn(1, args.batch_size, args.hidden_size, device=device)

    if args.variant == "torch":
        return lambda: gru(x, h0)[0]
    if args.variant == "generic_cuda":
        return lambda: cuda_gru_forward_layer(
            x,
            h0[0],
            gru.weight_ih_l0,
            gru.weight_hh_l0,
            gru.bias_ih_l0,
            gru.bias_hh_l0,
        )
    if args.variant == "a100":
        return lambda: a100_gru_forward_layer(
            x,
            h0[0],
            gru.weight_ih_l0,
            gru.weight_hh_l0,
            gru.bias_ih_l0,
            gru.bias_hh_l0,
            block_threads=args.block_threads,
        )
    if args.variant == "a100_precompute":
        return lambda: a100_gru_forward_layer_precompute_input(
            x,
            h0[0],
            gru.weight_ih_l0,
            gru.weight_hh_l0,
            gru.bias_ih_l0,
            gru.bias_hh_l0,
            block_threads=args.block_threads,
        )
    if args.variant == "a100_fused":
        return lambda: a100_gru_forward_layer_precompute_input_fused(
            x,
            h0[0],
            gru.weight_ih_l0,
            gru.weight_hh_l0,
            gru.bias_ih_l0,
            gru.bias_hh_l0,
            block_threads=args.block_threads,
        )
    if args.variant == "a100_fused_pingpong":
        return lambda: a100_gru_forward_layer_precompute_input_fused_pingpong(
            x,
            h0[0],
            gru.weight_ih_l0,
            gru.weight_hh_l0,
            gru.bias_ih_l0,
            gru.bias_hh_l0,
            block_threads=args.block_threads,
        )
    if args.variant == "a100_fused_specialized":
        return lambda: a100_gru_forward_layer_precompute_input_fused_specialized(
            x,
            h0[0],
            gru.weight_ih_l0,
            gru.weight_hh_l0,
            gru.bias_ih_l0,
            gru.bias_hh_l0,
            block_threads=args.block_threads,
        )
    if args.variant in {"a100_cooperative2", "a100_cooperative4"}:
        ctas_per_batch = 2 if args.variant == "a100_cooperative2" else 4
        return lambda: a100_gru_forward_layer_precompute_input_cooperative(
            x,
            h0[0],
            gru.weight_ih_l0,
            gru.weight_hh_l0,
            gru.bias_ih_l0,
            gru.bias_hh_l0,
            block_threads=args.cooperative_block_threads,
            ctas_per_batch=ctas_per_batch,
        )
    if args.variant == "a100_cooperative_h256":
        return lambda: a100_gru_forward_layer_precompute_input_cooperative_h256(
            x,
            h0[0],
            gru.weight_ih_l0,
            gru.weight_hh_l0,
            gru.bias_ih_l0,
            gru.bias_hh_l0,
            block_threads=args.h256_block_threads,
        )
    if args.variant == "a100_cooperative_h256_cached_shmem":
        return lambda: a100_gru_forward_layer_precompute_input_cooperative_h256_cached_shmem(
            x,
            h0[0],
            gru.weight_ih_l0,
            gru.weight_hh_l0,
            gru.bias_ih_l0,
            gru.bias_hh_l0,
            block_threads=args.h256_block_threads,
        )
    if args.variant == "a100_cooperative_h256_parallel_update":
        return lambda: a100_gru_forward_layer_precompute_input_cooperative_h256_parallel_update(
            x,
            h0[0],
            gru.weight_ih_l0,
            gru.weight_hh_l0,
            gru.bias_ih_l0,
            gru.bias_hh_l0,
            block_threads=args.h256_block_threads,
        )
    if args.variant == "a100_cooperative_h256_qwarp_shmem":
        return lambda: a100_gru_forward_layer_precompute_input_cooperative_h256_qwarp_shmem(
            x,
            h0[0],
            gru.weight_ih_l0,
            gru.weight_hh_l0,
            gru.bias_ih_l0,
            gru.bias_hh_l0,
            block_threads=args.h256_block_threads,
        )
    if args.variant == "a100_cooperative_h256_shmem":
        return lambda: a100_gru_forward_layer_precompute_input_cooperative_h256_shmem(
            x,
            h0[0],
            gru.weight_ih_l0,
            gru.weight_hh_l0,
            gru.bias_ih_l0,
            gru.bias_hh_l0,
            block_threads=args.h256_block_threads,
        )
    subwarp_size = 8 if args.variant == "a100_subwarp8" else 16
    return lambda: a100_gru_forward_layer_precompute_input_subwarp(
        x,
        h0[0],
        gru.weight_ih_l0,
        gru.weight_hh_l0,
        gru.bias_ih_l0,
        gru.bias_hh_l0,
        block_threads=args.block_threads,
        subwarp_size=subwarp_size,
    )


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required.")
    if args.variant.startswith("a100") and torch.cuda.get_device_capability() != (8, 0):
        raise RuntimeError("A100/SM80 is required for A100 variants.")

    runner = make_runner(args)
    with torch.no_grad():
        torch.cuda.nvtx.range_push("warmup")
        for _ in range(args.warmup_steps):
            runner()
        torch.cuda.synchronize()
        torch.cuda.nvtx.range_pop()

        torch.cuda.nvtx.range_push(f"profile:{args.variant}")
        start = time.perf_counter()
        for _ in range(args.profile_steps):
            runner()
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        torch.cuda.nvtx.range_pop()

    print(
        f"variant={args.variant} hidden={args.hidden_size} "
        f"seq_len={args.seq_len} steps={args.profile_steps} "
        f"ms_per_step={elapsed * 1000.0 / max(args.profile_steps, 1):.3f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
