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
    parser.add_argument("--run-smoke", action="store_true", help="运行一次小型 forward/backward。")
    return parser.parse_args()


def main() -> None:
    _ensure_source_import()
    import torch
    from a100_gru_h256 import A100GRUH256, is_a100_available

    print(f"python={sys.version.split()[0]}")
    print(f"torch={torch.__version__}")
    print(f"cuda_available={torch.cuda.is_available()}")
    if torch.cuda.is_available():
        device = torch.cuda.current_device()
        print(f"device_index={device}")
        print(f"device_name={torch.cuda.get_device_name(device)}")
        print(f"capability={torch.cuda.get_device_capability(device)}")
        print(f"is_a100_available={is_a100_available(device)}")

    cubin_path = Path(__file__).resolve().parents[1] / "kernels" / "a100_gru_h256_sm80.cubin"
    print(f"cubin_exists={cubin_path.exists()}")
    print(f"cubin_path={cubin_path}")

    if not cubin_path.exists():
        raise SystemExit("缺少预编译 cubin，请在构建环境运行 a100_gru_h256/scripts/build_cubin.py。")
    if not torch.cuda.is_available():
        raise SystemExit("CUDA 不可用。")
    if not is_a100_available():
        raise SystemExit("当前 GPU 不是 A100/SM80。")

    args = parse_args()
    if args.run_smoke:
        model = A100GRUH256(input_size=5).cuda()
        x = torch.randn(2, 5, 5, device="cuda", requires_grad=True)
        output, h_n = model(x)
        (output.sum() + h_n.sum()).backward()
        torch.cuda.synchronize()
        print("smoke_test=ok")


if __name__ == "__main__":
    main()
