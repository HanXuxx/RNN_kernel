import pytest
import torch

from rnn_kernel.cuda_gru_forward import cuda_gru_forward_layer


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
@pytest.mark.parametrize("hidden_size", [16, 33, 130])
def test_cuda_gru_forward_matches_torch_gru(hidden_size: int) -> None:
    torch.manual_seed(2026)
    device = torch.device("cuda")
    batch_size = 3
    seq_len = 11
    input_size = 5

    gru = torch.nn.GRU(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=1,
        batch_first=True,
    ).to(device)
    x = torch.randn(batch_size, seq_len, input_size, device=device)
    h0 = torch.randn(1, batch_size, hidden_size, device=device)

    with torch.no_grad():
        torch_out, _ = gru(x, h0)
        cuda_out = cuda_gru_forward_layer(
            x,
            h0[0],
            gru.weight_ih_l0,
            gru.weight_hh_l0,
            gru.bias_ih_l0,
            gru.bias_hh_l0,
        )

    assert torch.allclose(torch_out, cuda_out, atol=2e-4, rtol=1e-4)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_cuda_gru_forward_rejects_large_hidden_size() -> None:
    device = torch.device("cuda")
    hidden_size = 257
    batch_size = 2
    seq_len = 4
    input_size = 3

    with pytest.raises(ValueError, match="hidden_size <= 256"):
        cuda_gru_forward_layer(
            torch.randn(batch_size, seq_len, input_size, device=device),
            torch.randn(batch_size, hidden_size, device=device),
            torch.randn(3 * hidden_size, input_size, device=device),
            torch.randn(3 * hidden_size, hidden_size, device=device),
            torch.randn(3 * hidden_size, device=device),
            torch.randn(3 * hidden_size, device=device),
        )
