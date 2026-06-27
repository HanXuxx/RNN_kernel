#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _ensure_source_import() -> None:
    root = Path(__file__).resolve().parents[2]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seq-len", type=int, default=8000)
    parser.add_argument("--input-size", type=int, default=16)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--warmup-steps", type=int, default=3)
    parser.add_argument("--timed-steps", type=int, default=10)
    parser.add_argument("--include-torch", action="store_true")
    parser.add_argument("--include-inference", action="store_true")
    parser.add_argument("--seed", type=int, default=2132)
    return parser.parse_args()


def _time_step(torch, model, optimizer, x, target) -> float:
    optimizer.zero_grad(set_to_none=True)
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    output, _ = model(x)
    loss = torch.nn.functional.mse_loss(output[..., 0], target)
    loss.backward()
    optimizer.step()
    end.record()
    torch.cuda.synchronize()
    return float(start.elapsed_time(end))


def _benchmark(torch, name: str, model, x, target, warmup_steps: int, timed_steps: int) -> None:
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    for _ in range(warmup_steps):
        _time_step(torch, model, optimizer, x, target)
    times = [_time_step(torch, model, optimizer, x, target) for _ in range(timed_steps)]
    mean_ms = sum(times) / len(times)
    tokens_per_second = x.size(0) * x.size(1) / (mean_ms / 1000.0)
    print(f"{name}: ms_per_step={mean_ms:.3f}, tokens_per_second={tokens_per_second:.0f}")


def _time_inference(torch, model, x) -> float:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    with torch.no_grad():
        model(x)
    end.record()
    torch.cuda.synchronize()
    return float(start.elapsed_time(end))


def _benchmark_inference(torch, name: str, model, x, warmup_steps: int, timed_steps: int) -> None:
    if hasattr(model, "eval"):
        model.eval()
    for _ in range(warmup_steps):
        _time_inference(torch, model, x)
    times = [_time_inference(torch, model, x) for _ in range(timed_steps)]
    mean_ms = sum(times) / len(times)
    tokens_per_second = x.size(0) * x.size(1) / (mean_ms / 1000.0)
    print(f"{name}: inference_ms={mean_ms:.3f}, tokens_per_second={tokens_per_second:.0f}")


def main() -> None:
    _ensure_source_import()
    import torch
    from a100_gru_h256 import from_torch_gru, is_a100_available

    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA 不可用。")
    if not is_a100_available():
        raise SystemExit("当前 GPU 不是 A100/SM80。")

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(args.seed)
    device = torch.device("cuda")

    torch_gru = torch.nn.GRU(args.input_size, 256, num_layers=args.num_layers, batch_first=True).to(device)
    fast_gru = from_torch_gru(torch_gru)
    x = torch.randn(args.batch_size, args.seq_len, args.input_size, device=device)
    target = torch.randn(args.batch_size, args.seq_len, device=device)

    print(f"gpu={torch.cuda.get_device_name()}")
    print(
        f"batch_size={args.batch_size}, seq_len={args.seq_len}, "
        f"input_size={args.input_size}, num_layers={args.num_layers}"
    )
    _benchmark(torch, "A100GRUH256", fast_gru, x, target, args.warmup_steps, args.timed_steps)
    if args.include_torch:
        _benchmark(torch, "torch.nn.GRU", torch_gru, x, target, args.warmup_steps, args.timed_steps)
    if args.include_inference:
        _benchmark_inference(
            torch,
            "A100GRUH256.forward_inference",
            fast_gru.forward_inference,
            x,
            args.warmup_steps,
            args.timed_steps,
        )
        if args.include_torch:
            _benchmark_inference(
                torch,
                "torch.nn.GRU.forward",
                torch_gru,
                x,
                args.warmup_steps,
                args.timed_steps,
            )


if __name__ == "__main__":
    main()
