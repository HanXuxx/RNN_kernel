#!/usr/bin/env python3
"""对 A100/SM80 专用 forward-only GRU 原型做正确性和性能测试。"""

import argparse
import csv
import time
from pathlib import Path

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
    parser = argparse.ArgumentParser(description="Benchmark A100 GRU forward prototype.")
    parser.add_argument("--hidden-sizes", type=str, default="256")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--input-dim", type=int, default=9)
    parser.add_argument("--warmup-steps", type=int, default=3)
    parser.add_argument("--timed-steps", type=int, default=10)
    parser.add_argument("--block-threads", type=int, default=1024)
    parser.add_argument("--subwarp-sizes", type=str, default="16")
    parser.add_argument("--cooperative-ctas", type=str, default="4")
    parser.add_argument("--cooperative-block-threads", type=int, default=1024)
    parser.add_argument("--h256-block-threads", type=int, default=704)
    parser.add_argument("--output-csv", type=Path, default=None)
    parser.add_argument(
        "--skip-generic",
        action="store_true",
        help="Skip the previous generic CUDA/NVRTC prototype.",
    )
    return parser.parse_args()


def parse_hidden_sizes(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def parse_subwarp_sizes(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def parse_cooperative_ctas(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def time_call(fn, warmup_steps: int, timed_steps: int) -> float:
    for _ in range(warmup_steps):
        fn()
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(timed_steps):
        fn()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    return elapsed * 1000.0 / max(timed_steps, 1)


def run_case(args: argparse.Namespace, hidden_size: int) -> dict[str, float | int | str]:
    device = torch.device("cuda")
    torch.manual_seed(2027)
    torch.cuda.manual_seed_all(2027)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = True

    gru = torch.nn.GRU(
        input_size=args.input_dim,
        hidden_size=hidden_size,
        num_layers=1,
        batch_first=True,
    ).to(device)
    x = torch.randn(args.batch_size, args.seq_len, args.input_dim, device=device)
    h0 = torch.randn(1, args.batch_size, hidden_size, device=device)

    def torch_forward() -> tuple[torch.Tensor, torch.Tensor]:
        return gru(x, h0)

    def a100_forward() -> torch.Tensor:
        return a100_gru_forward_layer(
            x,
            h0[0],
            gru.weight_ih_l0,
            gru.weight_hh_l0,
            gru.bias_ih_l0,
            gru.bias_hh_l0,
            block_threads=args.block_threads,
        )

    def a100_precompute_forward() -> torch.Tensor:
        return a100_gru_forward_layer_precompute_input(
            x,
            h0[0],
            gru.weight_ih_l0,
            gru.weight_hh_l0,
            gru.bias_ih_l0,
            gru.bias_hh_l0,
            block_threads=args.block_threads,
        )

    def a100_fused_forward() -> torch.Tensor:
        return a100_gru_forward_layer_precompute_input_fused(
            x,
            h0[0],
            gru.weight_ih_l0,
            gru.weight_hh_l0,
            gru.bias_ih_l0,
            gru.bias_hh_l0,
            block_threads=args.block_threads,
        )

    def a100_fused_pingpong_forward() -> torch.Tensor:
        return a100_gru_forward_layer_precompute_input_fused_pingpong(
            x,
            h0[0],
            gru.weight_ih_l0,
            gru.weight_hh_l0,
            gru.bias_ih_l0,
            gru.bias_hh_l0,
            block_threads=args.block_threads,
        )

    def a100_fused_specialized_forward() -> torch.Tensor:
        return a100_gru_forward_layer_precompute_input_fused_specialized(
            x,
            h0[0],
            gru.weight_ih_l0,
            gru.weight_hh_l0,
            gru.bias_ih_l0,
            gru.bias_hh_l0,
            block_threads=args.block_threads,
        )

    def a100_cooperative_h256_forward() -> torch.Tensor:
        return a100_gru_forward_layer_precompute_input_cooperative_h256(
            x,
            h0[0],
            gru.weight_ih_l0,
            gru.weight_hh_l0,
            gru.bias_ih_l0,
            gru.bias_hh_l0,
            block_threads=args.h256_block_threads,
        )

    def a100_cooperative_h256_parallel_update_forward() -> torch.Tensor:
        return a100_gru_forward_layer_precompute_input_cooperative_h256_parallel_update(
            x,
            h0[0],
            gru.weight_ih_l0,
            gru.weight_hh_l0,
            gru.bias_ih_l0,
            gru.bias_hh_l0,
            block_threads=args.h256_block_threads,
        )

    def a100_cooperative_h256_shmem_forward() -> torch.Tensor:
        return a100_gru_forward_layer_precompute_input_cooperative_h256_shmem(
            x,
            h0[0],
            gru.weight_ih_l0,
            gru.weight_hh_l0,
            gru.bias_ih_l0,
            gru.bias_hh_l0,
            block_threads=args.h256_block_threads,
        )

    def a100_cooperative_h256_qwarp_shmem_forward() -> torch.Tensor:
        return a100_gru_forward_layer_precompute_input_cooperative_h256_qwarp_shmem(
            x,
            h0[0],
            gru.weight_ih_l0,
            gru.weight_hh_l0,
            gru.bias_ih_l0,
            gru.bias_hh_l0,
            block_threads=args.h256_block_threads,
        )

    def a100_cooperative_h256_cached_shmem_forward() -> torch.Tensor:
        return a100_gru_forward_layer_precompute_input_cooperative_h256_cached_shmem(
            x,
            h0[0],
            gru.weight_ih_l0,
            gru.weight_hh_l0,
            gru.bias_ih_l0,
            gru.bias_hh_l0,
            block_threads=args.h256_block_threads,
        )

    subwarp_sizes = parse_subwarp_sizes(args.subwarp_sizes)
    subwarp_fns = {
        size: (
            lambda subwarp_size=size: a100_gru_forward_layer_precompute_input_subwarp(
                x,
                h0[0],
                gru.weight_ih_l0,
                gru.weight_hh_l0,
                gru.bias_ih_l0,
                gru.bias_hh_l0,
                block_threads=args.block_threads,
                subwarp_size=subwarp_size,
            )
        )
        for size in subwarp_sizes
    }
    cooperative_ctas = parse_cooperative_ctas(args.cooperative_ctas)
    cooperative_fns = {
        ctas_per_batch: (
            lambda ctas=ctas_per_batch: a100_gru_forward_layer_precompute_input_cooperative(
                x,
                h0[0],
                gru.weight_ih_l0,
                gru.weight_hh_l0,
                gru.bias_ih_l0,
                gru.bias_hh_l0,
                block_threads=args.cooperative_block_threads,
                ctas_per_batch=ctas,
            )
        )
        for ctas_per_batch in cooperative_ctas
    }

    def generic_forward() -> torch.Tensor:
        return cuda_gru_forward_layer(
            x,
            h0[0],
            gru.weight_ih_l0,
            gru.weight_hh_l0,
            gru.bias_ih_l0,
            gru.bias_hh_l0,
        )

    with torch.no_grad():
        torch_out, _ = torch_forward()
        a100_out = a100_forward()
        a100_precompute_out = a100_precompute_forward()
        a100_fused_out = a100_fused_forward()
        a100_fused_pingpong_out = a100_fused_pingpong_forward()
        a100_fused_specialized_out = a100_fused_specialized_forward()
        if hidden_size == 256:
            a100_cooperative_h256_out = a100_cooperative_h256_forward()
            a100_cooperative_h256_parallel_update_out = (
                a100_cooperative_h256_parallel_update_forward()
            )
            a100_cooperative_h256_shmem_out = a100_cooperative_h256_shmem_forward()
            a100_cooperative_h256_qwarp_shmem_out = (
                a100_cooperative_h256_qwarp_shmem_forward()
            )
            a100_cooperative_h256_cached_shmem_out = (
                a100_cooperative_h256_cached_shmem_forward()
            )
            cooperative_h256_max_abs_diff = (
                torch_out - a100_cooperative_h256_out
            ).abs().max().item()
            cooperative_h256_parallel_update_max_abs_diff = (
                torch_out - a100_cooperative_h256_parallel_update_out
            ).abs().max().item()
            cooperative_h256_shmem_max_abs_diff = (
                torch_out - a100_cooperative_h256_shmem_out
            ).abs().max().item()
            cooperative_h256_qwarp_shmem_max_abs_diff = (
                torch_out - a100_cooperative_h256_qwarp_shmem_out
            ).abs().max().item()
            cooperative_h256_cached_shmem_max_abs_diff = (
                torch_out - a100_cooperative_h256_cached_shmem_out
            ).abs().max().item()
        else:
            cooperative_h256_max_abs_diff = 0.0
            cooperative_h256_parallel_update_max_abs_diff = 0.0
            cooperative_h256_shmem_max_abs_diff = 0.0
            cooperative_h256_qwarp_shmem_max_abs_diff = 0.0
            cooperative_h256_cached_shmem_max_abs_diff = 0.0
        max_abs_diff = (torch_out - a100_out).abs().max().item()
        precompute_max_abs_diff = (torch_out - a100_precompute_out).abs().max().item()
        fused_max_abs_diff = (torch_out - a100_fused_out).abs().max().item()
        fused_pingpong_max_abs_diff = (torch_out - a100_fused_pingpong_out).abs().max().item()
        fused_specialized_max_abs_diff = (
            torch_out - a100_fused_specialized_out
        ).abs().max().item()
        subwarp_outputs = {size: fn() for size, fn in subwarp_fns.items()}
        subwarp_diffs = {
            size: (torch_out - output).abs().max().item()
            for size, output in subwarp_outputs.items()
        }
        cooperative_outputs = {ctas: fn() for ctas, fn in cooperative_fns.items()}
        cooperative_diffs = {
            ctas: (torch_out - output).abs().max().item()
            for ctas, output in cooperative_outputs.items()
        }

        torch_ms = time_call(torch_forward, args.warmup_steps, args.timed_steps)
        a100_ms = time_call(a100_forward, args.warmup_steps, args.timed_steps)
        a100_precompute_ms = time_call(a100_precompute_forward, args.warmup_steps, args.timed_steps)
        a100_fused_ms = time_call(a100_fused_forward, args.warmup_steps, args.timed_steps)
        a100_fused_pingpong_ms = time_call(
            a100_fused_pingpong_forward,
            args.warmup_steps,
            args.timed_steps,
        )
        a100_fused_specialized_ms = time_call(
            a100_fused_specialized_forward,
            args.warmup_steps,
            args.timed_steps,
        )
        a100_cooperative_h256_ms = 0.0
        if hidden_size == 256:
            a100_cooperative_h256_ms = time_call(
                a100_cooperative_h256_forward,
                args.warmup_steps,
                args.timed_steps,
            )
        a100_cooperative_h256_parallel_update_ms = 0.0
        if hidden_size == 256:
            a100_cooperative_h256_parallel_update_ms = time_call(
                a100_cooperative_h256_parallel_update_forward,
                args.warmup_steps,
                args.timed_steps,
            )
        a100_cooperative_h256_shmem_ms = 0.0
        if hidden_size == 256:
            a100_cooperative_h256_shmem_ms = time_call(
                a100_cooperative_h256_shmem_forward,
                args.warmup_steps,
                args.timed_steps,
            )
        a100_cooperative_h256_qwarp_shmem_ms = 0.0
        if hidden_size == 256:
            a100_cooperative_h256_qwarp_shmem_ms = time_call(
                a100_cooperative_h256_qwarp_shmem_forward,
                args.warmup_steps,
                args.timed_steps,
            )
        a100_cooperative_h256_cached_shmem_ms = 0.0
        if hidden_size == 256:
            a100_cooperative_h256_cached_shmem_ms = time_call(
                a100_cooperative_h256_cached_shmem_forward,
                args.warmup_steps,
                args.timed_steps,
            )
        subwarp_ms = {
            size: time_call(fn, args.warmup_steps, args.timed_steps)
            for size, fn in subwarp_fns.items()
        }
        cooperative_ms = {
            ctas: time_call(fn, args.warmup_steps, args.timed_steps)
            for ctas, fn in cooperative_fns.items()
        }
        generic_ms = 0.0
        if not args.skip_generic:
            generic_ms = time_call(generic_forward, args.warmup_steps, args.timed_steps)

    generic_speedup = generic_ms / a100_ms if generic_ms else 0.0
    generic_precompute_speedup = generic_ms / a100_precompute_ms if generic_ms else 0.0
    result = {
        "hidden_size": hidden_size,
        "batch_size": args.batch_size,
        "seq_len": args.seq_len,
        "input_dim": args.input_dim,
        "block_threads": args.block_threads,
        "cooperative_block_threads": args.cooperative_block_threads,
        "h256_block_threads": args.h256_block_threads,
        "torch_ms": torch_ms,
        "generic_cuda_ms": generic_ms,
        "a100_ms": a100_ms,
        "a100_precompute_ms": a100_precompute_ms,
        "a100_fused_precompute_ms": a100_fused_ms,
        "a100_fused_pingpong_precompute_ms": a100_fused_pingpong_ms,
        "a100_fused_specialized_precompute_ms": a100_fused_specialized_ms,
        "a100_cooperative_h256_precompute_ms": a100_cooperative_h256_ms,
        "a100_cooperative_h256_parallel_update_precompute_ms": (
            a100_cooperative_h256_parallel_update_ms
        ),
        "a100_cooperative_h256_shmem_precompute_ms": a100_cooperative_h256_shmem_ms,
        "a100_cooperative_h256_qwarp_shmem_precompute_ms": (
            a100_cooperative_h256_qwarp_shmem_ms
        ),
        "a100_cooperative_h256_cached_shmem_precompute_ms": (
            a100_cooperative_h256_cached_shmem_ms
        ),
        "speedup_vs_torch": torch_ms / a100_ms,
        "precompute_speedup_vs_torch": torch_ms / a100_precompute_ms,
        "fused_speedup_vs_torch": torch_ms / a100_fused_ms,
        "fused_pingpong_speedup_vs_torch": torch_ms / a100_fused_pingpong_ms,
        "fused_specialized_speedup_vs_torch": torch_ms / a100_fused_specialized_ms,
        "cooperative_h256_speedup_vs_torch": (
            torch_ms / a100_cooperative_h256_ms if a100_cooperative_h256_ms else 0.0
        ),
        "cooperative_h256_parallel_update_speedup_vs_torch": (
            torch_ms / a100_cooperative_h256_parallel_update_ms
            if a100_cooperative_h256_parallel_update_ms
            else 0.0
        ),
        "cooperative_h256_shmem_speedup_vs_torch": (
            torch_ms / a100_cooperative_h256_shmem_ms
            if a100_cooperative_h256_shmem_ms
            else 0.0
        ),
        "cooperative_h256_qwarp_shmem_speedup_vs_torch": (
            torch_ms / a100_cooperative_h256_qwarp_shmem_ms
            if a100_cooperative_h256_qwarp_shmem_ms
            else 0.0
        ),
        "cooperative_h256_cached_shmem_speedup_vs_torch": (
            torch_ms / a100_cooperative_h256_cached_shmem_ms
            if a100_cooperative_h256_cached_shmem_ms
            else 0.0
        ),
        "speedup_vs_generic_cuda": generic_speedup,
        "precompute_speedup_vs_generic_cuda": generic_precompute_speedup,
        "fused_speedup_vs_generic_cuda": generic_ms / a100_fused_ms if generic_ms else 0.0,
        "fused_pingpong_speedup_vs_generic_cuda": (
            generic_ms / a100_fused_pingpong_ms if generic_ms else 0.0
        ),
        "fused_specialized_speedup_vs_generic_cuda": (
            generic_ms / a100_fused_specialized_ms if generic_ms else 0.0
        ),
        "cooperative_h256_speedup_vs_generic_cuda": (
            generic_ms / a100_cooperative_h256_ms
            if generic_ms and a100_cooperative_h256_ms
            else 0.0
        ),
        "cooperative_h256_parallel_update_speedup_vs_generic_cuda": (
            generic_ms / a100_cooperative_h256_parallel_update_ms
            if generic_ms and a100_cooperative_h256_parallel_update_ms
            else 0.0
        ),
        "cooperative_h256_shmem_speedup_vs_generic_cuda": (
            generic_ms / a100_cooperative_h256_shmem_ms
            if generic_ms and a100_cooperative_h256_shmem_ms
            else 0.0
        ),
        "cooperative_h256_qwarp_shmem_speedup_vs_generic_cuda": (
            generic_ms / a100_cooperative_h256_qwarp_shmem_ms
            if generic_ms and a100_cooperative_h256_qwarp_shmem_ms
            else 0.0
        ),
        "cooperative_h256_cached_shmem_speedup_vs_generic_cuda": (
            generic_ms / a100_cooperative_h256_cached_shmem_ms
            if generic_ms and a100_cooperative_h256_cached_shmem_ms
            else 0.0
        ),
        "max_abs_diff": max_abs_diff,
        "precompute_max_abs_diff": precompute_max_abs_diff,
        "fused_max_abs_diff": fused_max_abs_diff,
        "fused_pingpong_max_abs_diff": fused_pingpong_max_abs_diff,
        "fused_specialized_max_abs_diff": fused_specialized_max_abs_diff,
        "cooperative_h256_max_abs_diff": cooperative_h256_max_abs_diff,
        "cooperative_h256_parallel_update_max_abs_diff": (
            cooperative_h256_parallel_update_max_abs_diff
        ),
        "cooperative_h256_shmem_max_abs_diff": cooperative_h256_shmem_max_abs_diff,
        "cooperative_h256_qwarp_shmem_max_abs_diff": (
            cooperative_h256_qwarp_shmem_max_abs_diff
        ),
        "cooperative_h256_cached_shmem_max_abs_diff": (
            cooperative_h256_cached_shmem_max_abs_diff
        ),
    }
    for size in subwarp_sizes:
        ms = subwarp_ms[size]
        result[f"subwarp{size}_precompute_ms"] = ms
        result[f"subwarp{size}_speedup_vs_torch"] = torch_ms / ms
        result[f"subwarp{size}_speedup_vs_generic_cuda"] = generic_ms / ms if generic_ms else 0.0
        result[f"subwarp{size}_max_abs_diff"] = subwarp_diffs[size]
    for ctas in cooperative_ctas:
        ms = cooperative_ms[ctas]
        result[f"cooperative{ctas}_precompute_ms"] = ms
        result[f"cooperative{ctas}_speedup_vs_torch"] = torch_ms / ms
        result[f"cooperative{ctas}_speedup_vs_generic_cuda"] = generic_ms / ms if generic_ms else 0.0
        result[f"cooperative{ctas}_max_abs_diff"] = cooperative_diffs[ctas]
    return result


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required.")
    if torch.cuda.get_device_capability() != (8, 0):
        raise RuntimeError("A100/SM80 is required.")

    results = [run_case(args, hidden_size) for hidden_size in parse_hidden_sizes(args.hidden_sizes)]
    for result in results:
        print(
            f"hidden={result['hidden_size']} "
            f"torch_ms={result['torch_ms']:.3f} "
            f"generic_cuda_ms={result['generic_cuda_ms']:.3f} "
            f"a100_ms={result['a100_ms']:.3f} "
            f"a100_precompute_ms={result['a100_precompute_ms']:.3f} "
            f"a100_fused_precompute_ms={result['a100_fused_precompute_ms']:.3f} "
            f"a100_fused_pingpong_precompute_ms={result['a100_fused_pingpong_precompute_ms']:.3f} "
            f"a100_fused_specialized_precompute_ms={result['a100_fused_specialized_precompute_ms']:.3f} "
            f"a100_cooperative_h256_precompute_ms={result['a100_cooperative_h256_precompute_ms']:.3f} "
            f"a100_cooperative_h256_parallel_update_precompute_ms="
            f"{result['a100_cooperative_h256_parallel_update_precompute_ms']:.3f} "
            f"a100_cooperative_h256_shmem_precompute_ms={result['a100_cooperative_h256_shmem_precompute_ms']:.3f} "
            f"a100_cooperative_h256_qwarp_shmem_precompute_ms="
            f"{result['a100_cooperative_h256_qwarp_shmem_precompute_ms']:.3f} "
            f"a100_cooperative_h256_cached_shmem_precompute_ms="
            f"{result['a100_cooperative_h256_cached_shmem_precompute_ms']:.3f} "
            + "".join(
                f" subwarp{size}_precompute_ms={result[f'subwarp{size}_precompute_ms']:.3f}"
                for size in parse_subwarp_sizes(args.subwarp_sizes)
            )
            + "".join(
                f" cooperative{ctas}_precompute_ms={result[f'cooperative{ctas}_precompute_ms']:.3f}"
                for ctas in parse_cooperative_ctas(args.cooperative_ctas)
            )
            + " "
            f"speedup_vs_torch={result['speedup_vs_torch']:.3f} "
            f"precompute_speedup_vs_torch={result['precompute_speedup_vs_torch']:.3f} "
            f"speedup_vs_generic_cuda={result['speedup_vs_generic_cuda']:.3f} "
            f"precompute_speedup_vs_generic_cuda={result['precompute_speedup_vs_generic_cuda']:.3f} "
            f"max_abs_diff={result['max_abs_diff']:.6f} "
            f"precompute_max_abs_diff={result['precompute_max_abs_diff']:.6f} "
            f"fused_max_abs_diff={result['fused_max_abs_diff']:.6f} "
            f"fused_pingpong_max_abs_diff={result['fused_pingpong_max_abs_diff']:.6f} "
            f"fused_specialized_max_abs_diff={result['fused_specialized_max_abs_diff']:.6f} "
            f"cooperative_h256_max_abs_diff={result['cooperative_h256_max_abs_diff']:.6f} "
            f"cooperative_h256_parallel_update_max_abs_diff="
            f"{result['cooperative_h256_parallel_update_max_abs_diff']:.6f} "
            f"cooperative_h256_shmem_max_abs_diff={result['cooperative_h256_shmem_max_abs_diff']:.6f} "
            f"cooperative_h256_qwarp_shmem_max_abs_diff="
            f"{result['cooperative_h256_qwarp_shmem_max_abs_diff']:.6f} "
            f"cooperative_h256_cached_shmem_max_abs_diff="
            f"{result['cooperative_h256_cached_shmem_max_abs_diff']:.6f}",
            flush=True,
        )

    if args.output_csv is not None:
        args.output_csv.parent.mkdir(parents=True, exist_ok=True)
        with args.output_csv.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(results[0].keys()))
            writer.writeheader()
            writer.writerows(results)
        print(f"wrote {args.output_csv}")


if __name__ == "__main__":
    main()
