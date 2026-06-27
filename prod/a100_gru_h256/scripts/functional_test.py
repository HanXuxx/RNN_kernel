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
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seq-len", type=int, default=5)
    parser.add_argument("--input-size", type=int, default=16)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--seed", type=int, default=2130)
    return parser.parse_args()


def main() -> None:
    _ensure_source_import()
    import torch
    from a100_gru_h256 import from_torch_gru, is_a100_available

    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA 不可用。")
    if not is_a100_available():
        raise SystemExit("当前 GPU 不是 A100/SM80。")

    torch.manual_seed(args.seed)
    device = torch.device("cuda")
    hidden_size = 256

    torch_gru = torch.nn.GRU(
        input_size=args.input_size,
        hidden_size=hidden_size,
        num_layers=args.num_layers,
        batch_first=True,
    ).to(device)
    fast_gru = from_torch_gru(torch_gru)

    x_torch = torch.randn(args.batch_size, args.seq_len, args.input_size, device=device, requires_grad=True)
    x_fast = x_torch.detach().clone().requires_grad_(True)
    h0_torch = torch.randn(args.num_layers, args.batch_size, hidden_size, device=device, requires_grad=True)
    h0_fast = h0_torch.detach().clone().requires_grad_(True)

    torch_out, torch_h = torch_gru(x_torch, h0_torch)
    fast_out, fast_h = fast_gru(x_fast, h0_fast)
    grad_out = torch.randn_like(torch_out)
    grad_h = torch.randn_like(torch_h)

    torch_out.backward(grad_out, retain_graph=True)
    torch_h.backward(grad_h)
    fast_out.backward(grad_out, retain_graph=True)
    fast_h.backward(grad_h)
    torch.cuda.synchronize()

    checks = {
        "output": torch.allclose(torch_out, fast_out, atol=4e-4, rtol=1e-4),
        "h_n": torch.allclose(torch_h, fast_h, atol=4e-4, rtol=1e-4),
        "grad_x": torch.allclose(x_torch.grad, x_fast.grad, atol=1e-3, rtol=3e-4),
        "grad_h0": torch.allclose(h0_torch.grad, h0_fast.grad, atol=1e-3, rtol=3e-4),
    }
    for name, torch_param in torch_gru.named_parameters():
        fast_param = getattr(fast_gru, name)
        checks[f"grad_{name}"] = torch.allclose(torch_param.grad, fast_param.grad, atol=5e-3, rtol=1e-3)

    with torch.no_grad():
        torch_eval_out, torch_eval_h = torch_gru(x_torch.detach(), h0_torch.detach())
        fast_eval_out, fast_eval_h = fast_gru(x_fast.detach(), h0_fast.detach())
    explicit_eval_out, explicit_eval_h = fast_gru.forward_inference(x_fast.detach(), h0_fast.detach())
    checks["inference_output"] = torch.allclose(torch_eval_out, fast_eval_out, atol=4e-4, rtol=1e-4)
    checks["inference_h_n"] = torch.allclose(torch_eval_h, fast_eval_h, atol=4e-4, rtol=1e-4)
    checks["forward_inference_output"] = torch.allclose(
        torch_eval_out,
        explicit_eval_out,
        atol=4e-4,
        rtol=1e-4,
    )
    checks["forward_inference_h_n"] = torch.allclose(
        torch_eval_h,
        explicit_eval_h,
        atol=4e-4,
        rtol=1e-4,
    )

    for name, passed in checks.items():
        print(f"{name}={passed}")
    if not all(checks.values()):
        raise SystemExit("functional_test=failed")
    print("functional_test=ok")


if __name__ == "__main__":
    main()
