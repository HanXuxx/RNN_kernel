import pytest
import torch

from rnn_kernel.triton_gru_forward import triton_gru_forward_layer


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
@pytest.mark.parametrize("hidden_size", [16, 33, 130])
def test_triton_gru_forward_matches_torch_gru(hidden_size: int) -> None:
    torch.manual_seed(2025)
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
        triton_out = triton_gru_forward_layer(
            x,
            h0[0],
            gru.weight_ih_l0,
            gru.weight_hh_l0,
            gru.bias_ih_l0,
            gru.bias_hh_l0,
        )

    assert torch.allclose(torch_out, triton_out, atol=2e-4, rtol=1e-4)

