import json
import os
import subprocess
import sys

import pytest
import torch

from rnn_kernel.a100.prod import A100GRU, from_torch_gru, is_a100_available, is_supported_gru


def _requires_a100() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    if torch.cuda.get_device_capability() != (8, 0):
        pytest.skip("A100/SM80 is required")


def test_prod_a100_gru_rejects_unsupported_torch_gru() -> None:
    gru = torch.nn.GRU(5, 128, num_layers=1, batch_first=True)

    assert not is_supported_gru(gru)
    with pytest.raises(ValueError, match="hidden_size=256"):
        A100GRU(input_size=5, hidden_size=128)
    with pytest.raises(ValueError, match="Only single-layer"):
        from_torch_gru(gru)


def test_prod_is_a100_available_accepts_cpu_device() -> None:
    assert not is_a100_available(torch.device("cpu"))


def test_prod_import_does_not_load_experimental_modules() -> None:
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{os.getcwd()}/src:{env.get('PYTHONPATH', '')}"
    script = """
import json
import sys
import torch
from rnn_kernel.a100.prod import A100GRU, from_torch_gru
print(json.dumps({
    "cuda_nvrtc": "cuda.nvrtc" in sys.modules,
    "nvidia_cuda_nvcc": "nvidia.cuda_nvcc" in sys.modules,
    "experimental_autograd": "rnn_kernel.a100.gru_autograd" in sys.modules,
    "experimental_forward": "rnn_kernel.a100.gru_forward" in sys.modules,
}))
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        check=True,
        env=env,
        text=True,
        capture_output=True,
    )
    loaded = json.loads(result.stdout)

    assert loaded == {
        "cuda_nvrtc": False,
        "nvidia_cuda_nvcc": False,
        "experimental_autograd": False,
        "experimental_forward": False,
    }


def test_prod_a100_gru_from_torch_gru_matches_torch_gru() -> None:
    _requires_a100()
    torch.manual_seed(2130)
    device = torch.device("cuda")
    input_size = 5
    hidden_size = 256
    batch_size = 2
    seq_len = 5

    torch_gru = torch.nn.GRU(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=1,
        batch_first=True,
    ).to(device)
    a100_gru = from_torch_gru(torch_gru)

    x_torch = torch.randn(batch_size, seq_len, input_size, device=device, requires_grad=True)
    x_a100 = x_torch.detach().clone().requires_grad_(True)
    h0_torch = torch.randn(1, batch_size, hidden_size, device=device, requires_grad=True)
    h0_a100 = h0_torch.detach().clone().requires_grad_(True)

    torch_out, torch_h = torch_gru(x_torch, h0_torch)
    a100_out, a100_h = a100_gru(x_a100, h0_a100)
    grad_out = torch.randn_like(torch_out)
    grad_h = torch.randn_like(torch_h)
    torch_out.backward(grad_out, retain_graph=True)
    torch_h.backward(grad_h)
    a100_out.backward(grad_out, retain_graph=True)
    a100_h.backward(grad_h)

    assert torch.allclose(torch_out, a100_out, atol=4e-4, rtol=1e-4)
    assert torch.allclose(torch_h, a100_h, atol=4e-4, rtol=1e-4)
    assert torch.allclose(x_torch.grad, x_a100.grad, atol=1e-3, rtol=3e-4)
    assert torch.allclose(h0_torch.grad, h0_a100.grad, atol=1e-3, rtol=3e-4)

    for name, torch_param in torch_gru.named_parameters():
        a100_param = getattr(a100_gru, name)
        assert torch.allclose(torch_param.grad, a100_param.grad, atol=5e-3, rtol=1e-3)


def test_prod_a100_gru_forward_inference_matches_torch_gru() -> None:
    _requires_a100()
    torch.manual_seed(2133)
    device = torch.device("cuda")
    input_size = 5
    hidden_size = 256
    batch_size = 2
    seq_len = 5

    torch_gru = torch.nn.GRU(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=1,
        batch_first=True,
    ).to(device)
    a100_gru = from_torch_gru(torch_gru)
    x = torch.randn(batch_size, seq_len, input_size, device=device)
    h0 = torch.randn(1, batch_size, hidden_size, device=device)

    with torch.no_grad():
        torch_out, torch_h = torch_gru(x, h0)
        no_grad_out, no_grad_h = a100_gru(x, h0)
    explicit_out, explicit_h = a100_gru.forward_inference(x, h0)

    assert not no_grad_out.requires_grad
    assert not explicit_out.requires_grad
    assert torch.allclose(torch_out, no_grad_out, atol=4e-4, rtol=1e-4)
    assert torch.allclose(torch_h, no_grad_h, atol=4e-4, rtol=1e-4)
    assert torch.allclose(torch_out, explicit_out, atol=4e-4, rtol=1e-4)
    assert torch.allclose(torch_h, explicit_h, atol=4e-4, rtol=1e-4)
