#!/usr/bin/env python3
"""对 forward-only CUDA C/NVRTC GRU time-loop 原型做正确性和性能测试。"""

import argparse
import csv
import time
from pathlib import Path

import torch

from rnn_kernel.cuda_gru_forward import cuda_gru_forward_layer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark CUDA C/NVRTC GRU forward prototype.")
    parser.add_argument("--hidden-sizes", type=str, default="128,130,160")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--input-dim", type=int, default=9)
    parser.add_argument("--warmup-steps", type=int, default=3)
    parser.add_argument("--timed-steps", type=int, default=10)
    parser.add_argument("--output-csv", type=Path, default=Path(""))
    return parser.parse_args()


def parse_hidden_sizes(value: str) -> list[int]:
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
    torch.manual_seed(2026)
    torch.cuda.manual_seed_all(2026)
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

    def cuda_forward() -> torch.Tensor:
        return cuda_gru_forward_layer(
            x,
            h0[0],
            gru.weight_ih_l0,
            gru.weight_hh_l0,
            gru.bias_ih_l0,
            gru.bias_hh_l0,
        )

    with torch.no_grad():
        torch_out, _ = gru(x, h0)
        cuda_out = cuda_forward()
        max_abs_diff = (torch_out - cuda_out).abs().max().item()

        torch_ms = time_call(lambda: gru(x, h0), args.warmup_steps, args.timed_steps)
        cuda_ms = time_call(cuda_forward, args.warmup_steps, args.timed_steps)

    return {
        "hidden_size": hidden_size,
        "batch_size": args.batch_size,
        "seq_len": args.seq_len,
        "input_dim": args.input_dim,
        "torch_ms": torch_ms,
        "cuda_ms": cuda_ms,
        "speedup_vs_torch": torch_ms / cuda_ms,
        "max_abs_diff": max_abs_diff,
    }


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required.")

    results = [run_case(args, hidden_size) for hidden_size in parse_hidden_sizes(args.hidden_sizes)]
    for result in results:
        print(
            f"hidden={result['hidden_size']} "
            f"torch_ms={result['torch_ms']:.3f} "
            f"cuda_ms={result['cuda_ms']:.3f} "
            f"speedup={result['speedup_vs_torch']:.3f} "
            f"max_abs_diff={result['max_abs_diff']:.6f}",
            flush=True,
        )

    if args.output_csv:
        args.output_csv.parent.mkdir(parents=True, exist_ok=True)
        with args.output_csv.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(results[0].keys()))
            writer.writeheader()
            writer.writerows(results)
        print(f"wrote {args.output_csv}")


if __name__ == "__main__":
    main()
