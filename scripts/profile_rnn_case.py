#!/usr/bin/env python3
"""对单个 RNN hidden size 运行 PyTorch profiler。"""

import argparse
from pathlib import Path

import torch
from torch.profiler import ProfilerActivity, profile, record_function

from rnn_benchmark import RNNBenchmarkModel, make_random_batches, train_step


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Profile one GRU/LSTM benchmark case.")
    parser.add_argument("--cell-type", choices=["GRU", "LSTM"], default="GRU")
    parser.add_argument("--hidden-size", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seq-len", type=int, default=8000)
    parser.add_argument("--input-dim", type=int, default=9)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--warmup-steps", type=int, default=2)
    parser.add_argument("--profile-steps", type=int, default=2)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=Path("profiles"))
    parser.add_argument("--row-limit", type=int, default=30)
    parser.add_argument("--allow-tf32", action="store_true")
    parser.add_argument("--export-trace", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for profiler runs.")

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = args.allow_tf32
    torch.backends.cudnn.allow_tf32 = args.allow_tf32

    device = torch.device("cuda")
    model = RNNBenchmarkModel(
        cell_type=args.cell_type,
        input_dim=args.input_dim,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    batches = make_random_batches(
        num_batches=max(args.warmup_steps, args.profile_steps, 1),
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        input_dim=args.input_dim,
        device=device,
    )

    for step in range(args.warmup_steps):
        x, y = batches[step % len(batches)]
        train_step(model, optimizer, x, y)
    torch.cuda.synchronize(device)

    activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA]
    with profile(
        activities=activities,
        record_shapes=True,
        profile_memory=True,
        with_stack=False,
    ) as prof:
        for step in range(args.profile_steps):
            x, y = batches[step % len(batches)]
            with record_function(f"{args.cell_type}_hidden_{args.hidden_size}_step"):
                train_step(model, optimizer, x, y)
            prof.step()
    torch.cuda.synchronize(device)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"{args.cell_type.lower()}_hidden_{args.hidden_size}"
    table_path = args.output_dir / f"{prefix}_profiler_table.txt"
    table = prof.key_averages().table(
        sort_by="self_cuda_time_total",
        row_limit=args.row_limit,
    )
    table_path.write_text(table, encoding="utf-8")

    if args.export_trace:
        trace_path = args.output_dir / f"{prefix}_trace.json"
        prof.export_chrome_trace(str(trace_path))

    print(f"wrote {table_path}")


if __name__ == "__main__":
    main()
