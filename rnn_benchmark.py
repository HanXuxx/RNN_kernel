#!/usr/bin/env python3
"""
Minimal single-GPU benchmark for GRU/LSTM training speed vs hidden_size.

This intentionally avoids DataLoader, distributed training, checkpointing, AMP,
and the production loss. Random batches are pre-generated on the GPU so the
timing mostly reflects the RNN forward/backward kernels.
"""

import argparse
import csv
import gc
import time
from dataclasses import dataclass
from typing import Iterable, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class BenchmarkResult:
    cell_type: str
    hidden_size: int
    params: int
    ms_per_step: Optional[float]
    steps_per_second: Optional[float]
    tokens_per_second: Optional[float]
    peak_memory_gb: Optional[float]
    last_loss: Optional[float]
    status: str


class RNNBenchmarkModel(nn.Module):
    def __init__(
        self,
        cell_type: str,
        input_dim: int,
        hidden_size: int,
        num_layers: int,
    ):
        super().__init__()
        rnn_cls = {"GRU": nn.GRU, "LSTM": nn.LSTM}[cell_type]
        self.rnn = rnn_cls(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=0.0,
            bidirectional=False,
            batch_first=True,
        )
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.rnn(x)
        return self.head(out).squeeze(-1)


def parse_int_list(value: str) -> List[int]:
    return [int(x.strip()) for x in value.split(",") if x.strip()]


def parse_cell_types(value: str) -> List[str]:
    cell_types = [x.strip().upper() for x in value.split(",") if x.strip()]
    invalid = sorted(set(cell_types) - {"GRU", "LSTM"})
    if invalid:
        raise argparse.ArgumentTypeError(f"Unsupported cell type(s): {invalid}")
    return cell_types


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def make_random_batches(
    num_batches: int,
    batch_size: int,
    seq_len: int,
    input_dim: int,
    device: torch.device,
) -> List[tuple[torch.Tensor, torch.Tensor]]:
    batches = []
    for _ in range(num_batches):
        x = torch.randn(batch_size, seq_len, input_dim, device=device)
        y = torch.randn(batch_size, seq_len, device=device)
        batches.append((x, y))
    return batches


def train_step(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    x: torch.Tensor,
    y: torch.Tensor,
) -> torch.Tensor:
    optimizer.zero_grad(set_to_none=True)
    pred = model(x)
    loss = F.mse_loss(pred, y)
    loss.backward()
    optimizer.step()
    return loss.detach()


def run_case(
    cell_type: str,
    hidden_size: int,
    batches: List[tuple[torch.Tensor, torch.Tensor]],
    args: argparse.Namespace,
    device: torch.device,
) -> BenchmarkResult:
    model = RNNBenchmarkModel(
        cell_type=cell_type,
        input_dim=args.input_dim,
        hidden_size=hidden_size,
        num_layers=args.num_layers,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    params = count_parameters(model)

    try:
        if device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)

        last_loss = None
        for step in range(args.warmup_steps):
            x, y = batches[step % len(batches)]
            last_loss = train_step(model, optimizer, x, y)

        if device.type == "cuda":
            torch.cuda.synchronize(device)

        start = time.perf_counter()
        for step in range(args.timed_steps):
            x, y = batches[step % len(batches)]
            last_loss = train_step(model, optimizer, x, y)

        if device.type == "cuda":
            torch.cuda.synchronize(device)

        elapsed = time.perf_counter() - start
        ms_per_step = elapsed * 1000.0 / max(args.timed_steps, 1)
        steps_per_second = args.timed_steps / elapsed
        tokens_per_second = steps_per_second * args.batch_size * args.seq_len
        peak_memory_gb = (
            torch.cuda.max_memory_allocated(device) / (1024.0 ** 3)
            if device.type == "cuda"
            else None
        )
        return BenchmarkResult(
            cell_type=cell_type,
            hidden_size=hidden_size,
            params=params,
            ms_per_step=ms_per_step,
            steps_per_second=steps_per_second,
            tokens_per_second=tokens_per_second,
            peak_memory_gb=peak_memory_gb,
            last_loss=float(last_loss.item()) if last_loss is not None else None,
            status="ok",
        )
    except RuntimeError as e:
        if "out of memory" not in str(e).lower():
            raise
        return BenchmarkResult(
            cell_type=cell_type,
            hidden_size=hidden_size,
            params=params,
            ms_per_step=None,
            steps_per_second=None,
            tokens_per_second=None,
            peak_memory_gb=None,
            last_loss=None,
            status=f"oom: {str(e).splitlines()[0]}",
        )
    finally:
        del optimizer
        del model
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()


def print_result(result: BenchmarkResult) -> None:
    if result.status == "ok":
        peak_memory = (
            f"{result.peak_memory_gb:6.2f}GB"
            if result.peak_memory_gb is not None
            else "   n/a"
        )
        print(
            f"{result.cell_type:4s} hidden={result.hidden_size:3d} "
            f"params={result.params:10d} "
            f"ms/step={result.ms_per_step:9.3f} "
            f"steps/s={result.steps_per_second:8.3f} "
            f"tokens/s={result.tokens_per_second:12.0f} "
            f"peak_mem={peak_memory} "
            f"loss={result.last_loss:.6f}",
            flush=True,
        )
    else:
        print(
            f"{result.cell_type:4s} hidden={result.hidden_size:3d} "
            f"params={result.params:10d} status={result.status}",
            flush=True,
        )


def write_csv(path: str, results: Iterable[BenchmarkResult]) -> None:
    fields = [
        "cell_type",
        "hidden_size",
        "params",
        "ms_per_step",
        "steps_per_second",
        "tokens_per_second",
        "peak_memory_gb",
        "last_loss",
        "status",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for result in results:
            writer.writerow({field: getattr(result, field) for field in fields})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Minimal GRU/LSTM hidden_size training-speed benchmark."
    )
    parser.add_argument("--hidden-sizes", type=parse_int_list, default=parse_int_list("64,96,128,130,160,192,256"))
    parser.add_argument("--cell-types", type=parse_cell_types, default=parse_cell_types("GRU,LSTM"))
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seq-len", type=int, default=8000)
    parser.add_argument("--input-dim", type=int, default=9)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--dataset-batches", type=int, default=32)
    parser.add_argument("--warmup-steps", type=int, default=3)
    parser.add_argument("--timed-steps", type=int, default=10)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--output-csv", type=str, default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")

    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = args.deterministic
        torch.backends.cudnn.benchmark = not args.deterministic

    print("RNN hidden_size benchmark")
    print(f"torch={torch.__version__}")
    if device.type == "cuda":
        print(f"cuda={torch.version.cuda}")
        print(f"cudnn={torch.backends.cudnn.version()}")
        print(f"gpu={torch.cuda.get_device_name(device)}")
    print(
        f"batch_size={args.batch_size}, seq_len={args.seq_len}, input_dim={args.input_dim}, "
        f"num_layers={args.num_layers}, dataset_batches={args.dataset_batches}, "
        f"warmup_steps={args.warmup_steps}, timed_steps={args.timed_steps}, "
        f"deterministic={args.deterministic}"
    )
    print()

    batches = make_random_batches(
        num_batches=args.dataset_batches,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        input_dim=args.input_dim,
        device=device,
    )

    results = []
    for cell_type in args.cell_types:
        for hidden_size in args.hidden_sizes:
            result = run_case(cell_type, hidden_size, batches, args, device)
            results.append(result)
            print_result(result)

    if args.output_csv:
        write_csv(args.output_csv, results)
        print(f"\nwrote {args.output_csv}")


if __name__ == "__main__":
    main()
