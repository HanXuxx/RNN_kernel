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
    parser.add_argument("--num-layers", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    _ensure_source_import()
    import torch
    from a100_gru_h256 import from_torch_gru

    args = parse_args()
    torch.manual_seed(2133)
    torch_gru = torch.nn.GRU(16, 256, num_layers=args.num_layers, batch_first=True).cuda()
    fast_gru = from_torch_gru(torch_gru)
    x = torch.randn(16, 128, 16, device="cuda", requires_grad=True)
    output, h_n = fast_gru(x)
    loss = output.square().mean() + h_n.square().mean()
    loss.backward()
    torch.cuda.synchronize()
    print(f"output_shape={tuple(output.shape)}")
    print(f"h_n_shape={tuple(h_n.shape)}")
    print(f"loss={float(loss.detach().cpu()):.6f}")


if __name__ == "__main__":
    main()
