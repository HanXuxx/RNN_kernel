import pytest
import torch

from rnn_kernel.custom_gru import CustomGRU, copy_from_torch_gru


def _compare_with_torch_gru(
    pointwise_backend: str,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
    atol: float = 1e-6,
    rtol: float = 1e-6,
) -> None:
    torch.manual_seed(321)
    input_size = 4
    hidden_size = 7
    num_layers = 2
    batch_size = 3
    seq_len = 5

    torch_gru = torch.nn.GRU(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        batch_first=True,
    ).to(device=device, dtype=dtype)
    custom_gru = CustomGRU(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        pointwise_backend=pointwise_backend,
        batch_first=True,
    ).to(device=device, dtype=dtype)
    copy_from_torch_gru(custom_gru, torch_gru)

    x_torch = torch.randn(
        batch_size,
        seq_len,
        input_size,
        device=device,
        dtype=dtype,
        requires_grad=True,
    )
    x_custom = x_torch.detach().clone().requires_grad_(True)
    h0 = torch.randn(
        num_layers,
        batch_size,
        hidden_size,
        device=device,
        dtype=dtype,
        requires_grad=True,
    )
    h0_custom = h0.detach().clone().requires_grad_(True)

    torch_out, torch_h = torch_gru(x_torch, h0)
    custom_out, custom_h = custom_gru(x_custom, h0_custom)
    grad_out = torch.randn_like(torch_out)
    grad_h = torch.randn_like(torch_h)
    torch_out.backward(grad_out, retain_graph=True)
    torch_h.backward(grad_h)
    custom_out.backward(grad_out, retain_graph=True)
    custom_h.backward(grad_h)

    assert torch.allclose(torch_out, custom_out, atol=atol, rtol=rtol)
    assert torch.allclose(torch_h, custom_h, atol=atol, rtol=rtol)
    assert torch.allclose(x_torch.grad, x_custom.grad, atol=atol, rtol=rtol)
    assert torch.allclose(h0.grad, h0_custom.grad, atol=atol, rtol=rtol)

    for name, torch_param in torch_gru.named_parameters():
        custom_param = getattr(custom_gru, name)
        assert torch.allclose(torch_param.grad, custom_param.grad, atol=atol, rtol=rtol)


def _compare_custom_backends(device: torch.device) -> None:
    torch.manual_seed(654)
    input_size = 4
    hidden_size = 7
    num_layers = 2
    batch_size = 3
    seq_len = 5

    torch_backend = CustomGRU(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        pointwise_backend="torch",
        batch_first=True,
    ).to(device)
    triton_backend = CustomGRU(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        pointwise_backend="triton",
        batch_first=True,
    ).to(device)
    triton_backend.load_state_dict(torch_backend.state_dict())

    x_torch = torch.randn(batch_size, seq_len, input_size, device=device, requires_grad=True)
    x_triton = x_torch.detach().clone().requires_grad_(True)
    h_torch = torch.randn(
        num_layers,
        batch_size,
        hidden_size,
        device=device,
        requires_grad=True,
    )
    h_triton = h_torch.detach().clone().requires_grad_(True)

    torch_out, torch_h = torch_backend(x_torch, h_torch)
    triton_out, triton_h = triton_backend(x_triton, h_triton)
    grad_out = torch.randn_like(torch_out)
    grad_h = torch.randn_like(torch_h)
    torch_out.backward(grad_out, retain_graph=True)
    torch_h.backward(grad_h)
    triton_out.backward(grad_out, retain_graph=True)
    triton_h.backward(grad_h)

    assert torch.allclose(torch_out, triton_out, atol=1e-6, rtol=1e-6)
    assert torch.allclose(torch_h, triton_h, atol=1e-6, rtol=1e-6)
    assert torch.allclose(x_torch.grad, x_triton.grad, atol=1e-6, rtol=1e-6)
    assert torch.allclose(h_torch.grad, h_triton.grad, atol=1e-6, rtol=1e-6)
    for torch_param, triton_param in zip(torch_backend.parameters(), triton_backend.parameters()):
        assert torch.allclose(torch_param.grad, triton_param.grad, atol=1e-6, rtol=1e-6)


def test_custom_gru_torch_backend_matches_torch_gru_cpu() -> None:
    _compare_with_torch_gru("torch", torch.device("cpu"))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_custom_gru_triton_backend_matches_torch_gru_cuda() -> None:
    _compare_with_torch_gru("triton", torch.device("cuda"), atol=1e-3, rtol=1e-4)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_custom_gru_triton_backend_matches_torch_backend_cuda() -> None:
    _compare_custom_backends(torch.device("cuda"))
