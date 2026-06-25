#!/usr/bin/env python3
"""
GRU/LSTM 训练速度基准测试。

脚本刻意避开 DataLoader、分布式训练、checkpoint、AMP 和生产损失函数。
随机 batch 会预先生成到目标设备上，让计时尽量反映 RNN forward/backward kernel。
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

try:
    from rnn_kernel.custom_gru import CustomGRU
except ImportError:  # pragma: no cover - 允许不 source scripts/env.sh 时直接运行。
    from src.rnn_kernel.custom_gru import CustomGRU


@dataclass
class BenchmarkResult:
    implementation: str
    cell_type: str
    hidden_size: int
    sequence_chunk_len: int
    deterministic: bool
    cudnn_enabled: bool
    cudnn_benchmark: bool
    cuda_matmul_allow_tf32: bool
    cudnn_allow_tf32: bool
    compile_model: bool
    params: int
    ms_per_step: Optional[float]
    zero_grad_ms_per_step: Optional[float]
    forward_ms_per_step: Optional[float]
    loss_ms_per_step: Optional[float]
    backward_ms_per_step: Optional[float]
    optimizer_ms_per_step: Optional[float]
    measured_ms_per_step: Optional[float]
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
        sequence_chunk_len: int = 0,
        implementation: str = "torch",
    ):
        super().__init__()
        self.sequence_chunk_len = sequence_chunk_len
        self.implementation = implementation
        if implementation == "torch":
            rnn_cls = {"GRU": nn.GRU, "LSTM": nn.LSTM}[cell_type]
            self.rnn = rnn_cls(
                input_size=input_dim,
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout=0.0,
                bidirectional=False,
                batch_first=True,
            )
        elif implementation in {"custom_gru", "custom_gru_triton"}:
            if cell_type != "GRU":
                raise ValueError("custom_gru implementations only support GRU.")
            pointwise_backend = "triton" if implementation == "custom_gru_triton" else "torch"
            self.rnn = CustomGRU(
                input_size=input_dim,
                hidden_size=hidden_size,
                num_layers=num_layers,
                pointwise_backend=pointwise_backend,
                batch_first=True,
            )
        else:
            raise ValueError(f"Unsupported implementation: {implementation}")
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.sequence_chunk_len > 0 and x.size(1) > self.sequence_chunk_len:
            outputs = []
            hidden = None
            for start in range(0, x.size(1), self.sequence_chunk_len):
                chunk = x[:, start : start + self.sequence_chunk_len, :]
                out, hidden = self.rnn(chunk, hidden)
                outputs.append(out)
            out = torch.cat(outputs, dim=1)
        else:
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


def resolve_cudnn_benchmark(args: argparse.Namespace) -> bool:
    if args.cudnn_benchmark == "on":
        return True
    if args.cudnn_benchmark == "off":
        return False
    return not args.deterministic


def configure_backends(args: argparse.Namespace, device: torch.device) -> None:
    if device.type != "cuda":
        return

    torch.backends.cudnn.enabled = not args.disable_cudnn
    torch.backends.cudnn.deterministic = args.deterministic
    torch.backends.cudnn.benchmark = resolve_cudnn_benchmark(args)

    # 默认关闭 TF32，保证本轮实验不通过降低 fp32 计算精度换速度。
    torch.backends.cuda.matmul.allow_tf32 = args.allow_tf32
    torch.backends.cudnn.allow_tf32 = args.allow_tf32


@dataclass
class StepTiming:
    zero_grad_ms: float
    forward_ms: float
    loss_ms: float
    backward_ms: float
    optimizer_ms: float

    @property
    def measured_ms(self) -> float:
        return (
            self.zero_grad_ms
            + self.forward_ms
            + self.loss_ms
            + self.backward_ms
            + self.optimizer_ms
        )


@dataclass
class TimingAccumulator:
    zero_grad_ms: float = 0.0
    forward_ms: float = 0.0
    loss_ms: float = 0.0
    backward_ms: float = 0.0
    optimizer_ms: float = 0.0
    count: int = 0

    def add(self, timing: StepTiming) -> None:
        self.zero_grad_ms += timing.zero_grad_ms
        self.forward_ms += timing.forward_ms
        self.loss_ms += timing.loss_ms
        self.backward_ms += timing.backward_ms
        self.optimizer_ms += timing.optimizer_ms
        self.count += 1

    def average(self) -> Optional[StepTiming]:
        if self.count == 0:
            return None
        return StepTiming(
            zero_grad_ms=self.zero_grad_ms / self.count,
            forward_ms=self.forward_ms / self.count,
            loss_ms=self.loss_ms / self.count,
            backward_ms=self.backward_ms / self.count,
            optimizer_ms=self.optimizer_ms / self.count,
        )


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


def timed_train_step(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    x: torch.Tensor,
    y: torch.Tensor,
    device: torch.device,
) -> tuple[torch.Tensor, StepTiming]:
    if device.type == "cuda":
        events = [torch.cuda.Event(enable_timing=True) for _ in range(6)]

        events[0].record()
        optimizer.zero_grad(set_to_none=True)
        events[1].record()
        pred = model(x)
        events[2].record()
        loss = F.mse_loss(pred, y)
        events[3].record()
        loss.backward()
        events[4].record()
        optimizer.step()
        events[5].record()
        torch.cuda.synchronize(device)

        timing = StepTiming(
            zero_grad_ms=events[0].elapsed_time(events[1]),
            forward_ms=events[1].elapsed_time(events[2]),
            loss_ms=events[2].elapsed_time(events[3]),
            backward_ms=events[3].elapsed_time(events[4]),
            optimizer_ms=events[4].elapsed_time(events[5]),
        )
        return loss.detach(), timing

    starts = []
    starts.append(time.perf_counter())
    optimizer.zero_grad(set_to_none=True)
    starts.append(time.perf_counter())
    pred = model(x)
    starts.append(time.perf_counter())
    loss = F.mse_loss(pred, y)
    starts.append(time.perf_counter())
    loss.backward()
    starts.append(time.perf_counter())
    optimizer.step()
    starts.append(time.perf_counter())

    timing = StepTiming(
        zero_grad_ms=(starts[1] - starts[0]) * 1000.0,
        forward_ms=(starts[2] - starts[1]) * 1000.0,
        loss_ms=(starts[3] - starts[2]) * 1000.0,
        backward_ms=(starts[4] - starts[3]) * 1000.0,
        optimizer_ms=(starts[5] - starts[4]) * 1000.0,
    )
    return loss.detach(), timing


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
        sequence_chunk_len=args.sequence_chunk_len,
        implementation=args.implementation,
    ).to(device)
    if (
        args.implementation == "torch"
        and device.type == "cuda"
        and torch.backends.cudnn.enabled
    ):
        model.rnn.flatten_parameters()
    params = count_parameters(model)
    if args.compile_model:
        model = torch.compile(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

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

        timed_accumulator = TimingAccumulator()
        start = time.perf_counter()
        for step in range(args.timed_steps):
            x, y = batches[step % len(batches)]
            if args.breakdown_timing:
                last_loss, step_timing = timed_train_step(model, optimizer, x, y, device)
                timed_accumulator.add(step_timing)
            else:
                last_loss = train_step(model, optimizer, x, y)

        if device.type == "cuda":
            torch.cuda.synchronize(device)

        elapsed = time.perf_counter() - start
        ms_per_step = elapsed * 1000.0 / max(args.timed_steps, 1)
        average_timing = timed_accumulator.average()
        steps_per_second = args.timed_steps / elapsed
        tokens_per_second = steps_per_second * args.batch_size * args.seq_len
        peak_memory_gb = (
            torch.cuda.max_memory_allocated(device) / (1024.0 ** 3)
            if device.type == "cuda"
            else None
        )
        return BenchmarkResult(
            implementation=args.implementation,
            cell_type=cell_type,
            hidden_size=hidden_size,
            sequence_chunk_len=args.sequence_chunk_len,
            deterministic=args.deterministic,
            cudnn_enabled=torch.backends.cudnn.enabled,
            cudnn_benchmark=torch.backends.cudnn.benchmark,
            cuda_matmul_allow_tf32=(
                torch.backends.cuda.matmul.allow_tf32 if device.type == "cuda" else False
            ),
            cudnn_allow_tf32=(
                torch.backends.cudnn.allow_tf32 if device.type == "cuda" else False
            ),
            compile_model=args.compile_model,
            params=params,
            ms_per_step=ms_per_step,
            zero_grad_ms_per_step=(
                average_timing.zero_grad_ms if average_timing is not None else None
            ),
            forward_ms_per_step=(
                average_timing.forward_ms if average_timing is not None else None
            ),
            loss_ms_per_step=average_timing.loss_ms if average_timing is not None else None,
            backward_ms_per_step=(
                average_timing.backward_ms if average_timing is not None else None
            ),
            optimizer_ms_per_step=(
                average_timing.optimizer_ms if average_timing is not None else None
            ),
            measured_ms_per_step=(
                average_timing.measured_ms if average_timing is not None else None
            ),
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
            implementation=args.implementation,
            cell_type=cell_type,
            hidden_size=hidden_size,
            sequence_chunk_len=args.sequence_chunk_len,
            deterministic=args.deterministic,
            cudnn_enabled=torch.backends.cudnn.enabled,
            cudnn_benchmark=torch.backends.cudnn.benchmark,
            cuda_matmul_allow_tf32=(
                torch.backends.cuda.matmul.allow_tf32 if device.type == "cuda" else False
            ),
            cudnn_allow_tf32=(
                torch.backends.cudnn.allow_tf32 if device.type == "cuda" else False
            ),
            compile_model=args.compile_model,
            params=params,
            ms_per_step=None,
            zero_grad_ms_per_step=None,
            forward_ms_per_step=None,
            loss_ms_per_step=None,
            backward_ms_per_step=None,
            optimizer_ms_per_step=None,
            measured_ms_per_step=None,
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
        message = (
            f"{result.cell_type:4s} hidden={result.hidden_size:3d} "
            f"params={result.params:10d} "
            f"ms/step={result.ms_per_step:9.3f} "
            f"steps/s={result.steps_per_second:8.3f} "
            f"tokens/s={result.tokens_per_second:12.0f} "
            f"peak_mem={peak_memory} "
            f"loss={result.last_loss:.6f}"
        )
        if result.measured_ms_per_step is not None:
            message += (
                f" breakdown_ms="
                f"fwd:{result.forward_ms_per_step:.3f},"
                f"loss:{result.loss_ms_per_step:.3f},"
                f"bwd:{result.backward_ms_per_step:.3f},"
                f"opt:{result.optimizer_ms_per_step:.3f}"
            )
        print(message, flush=True)
    else:
        print(
            f"{result.cell_type:4s} hidden={result.hidden_size:3d} "
            f"params={result.params:10d} status={result.status}",
            flush=True,
        )


def write_csv(path: str, results: Iterable[BenchmarkResult]) -> None:
    fields = [
        "cell_type",
        "implementation",
        "hidden_size",
        "sequence_chunk_len",
        "deterministic",
        "cudnn_enabled",
        "cudnn_benchmark",
        "cuda_matmul_allow_tf32",
        "cudnn_allow_tf32",
        "compile_model",
        "params",
        "ms_per_step",
        "zero_grad_ms_per_step",
        "forward_ms_per_step",
        "loss_ms_per_step",
        "backward_ms_per_step",
        "optimizer_ms_per_step",
        "measured_ms_per_step",
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
    parser.add_argument(
        "--implementation",
        choices=["torch", "custom_gru", "custom_gru_triton"],
        default="torch",
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seq-len", type=int, default=8000)
    parser.add_argument("--input-dim", type=int, default=9)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument(
        "--sequence-chunk-len",
        type=int,
        default=0,
        help="Split sequence into chunks while carrying hidden state. 0 disables chunking.",
    )
    parser.add_argument("--dataset-batches", type=int, default=32)
    parser.add_argument("--warmup-steps", type=int, default=3)
    parser.add_argument("--timed-steps", type=int, default=10)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument(
        "--cudnn-benchmark",
        choices=["auto", "on", "off"],
        default="auto",
        help="Control torch.backends.cudnn.benchmark independently from deterministic.",
    )
    parser.add_argument("--disable-cudnn", action="store_true")
    parser.add_argument(
        "--allow-tf32",
        action="store_true",
        help="Allow TF32. Default is false to keep full fp32 precision.",
    )
    parser.add_argument("--compile-model", action="store_true")
    parser.add_argument(
        "--breakdown-timing",
        action="store_true",
        help="Measure per-step zero_grad/forward/loss/backward/optimizer timing.",
    )
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
    configure_backends(args, device)

    print("RNN hidden_size benchmark")
    print(f"implementation={args.implementation}")
    print(f"torch={torch.__version__}")
    if device.type == "cuda":
        print(f"cuda={torch.version.cuda}")
        print(f"cudnn={torch.backends.cudnn.version()}")
        print(f"gpu={torch.cuda.get_device_name(device)}")
        print(
            f"cudnn_enabled={torch.backends.cudnn.enabled}, "
            f"cudnn_benchmark={torch.backends.cudnn.benchmark}, "
            f"cudnn_deterministic={torch.backends.cudnn.deterministic}, "
            f"cuda_matmul_allow_tf32={torch.backends.cuda.matmul.allow_tf32}, "
            f"cudnn_allow_tf32={torch.backends.cudnn.allow_tf32}"
        )
    print(
        f"batch_size={args.batch_size}, seq_len={args.seq_len}, input_dim={args.input_dim}, "
        f"num_layers={args.num_layers}, sequence_chunk_len={args.sequence_chunk_len}, "
        f"dataset_batches={args.dataset_batches}, "
        f"warmup_steps={args.warmup_steps}, timed_steps={args.timed_steps}, "
        f"deterministic={args.deterministic}, breakdown_timing={args.breakdown_timing}, "
        f"compile_model={args.compile_model}"
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
