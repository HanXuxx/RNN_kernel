import pytest
import torch

from rnn_kernel.a100 import (
    a100_gru_forward_layer,
    a100_gru_forward_layer_precompute_input_cooperative,
    a100_gru_forward_layer_precompute_input_cooperative_h256,
    a100_gru_forward_layer_precompute_input_cooperative_h256_cached_shmem,
    a100_gru_forward_layer_precompute_input_cooperative_h256_parallel_update,
    a100_gru_forward_layer_precompute_input_cooperative_h256_qwarp_shmem,
    a100_gru_forward_layer_precompute_input_cooperative_h256_shmem,
    a100_gru_forward_layer_precompute_input_fused,
    a100_gru_forward_layer_precompute_input_fused_pingpong,
    a100_gru_forward_layer_precompute_input_fused_specialized,
    a100_gru_forward_layer_precompute_input,
    a100_gru_forward_layer_precompute_input_subwarp,
)


def _requires_a100() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    if torch.cuda.get_device_capability() != (8, 0):
        pytest.skip("A100/SM80 is required")


@pytest.mark.parametrize("hidden_size", [16, 33, 130])
def test_a100_gru_forward_matches_torch_gru(hidden_size: int) -> None:
    _requires_a100()
    torch.manual_seed(2027)
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
        a100_out = a100_gru_forward_layer(
            x,
            h0[0],
            gru.weight_ih_l0,
            gru.weight_hh_l0,
            gru.bias_ih_l0,
            gru.bias_hh_l0,
        )

    assert torch.allclose(torch_out, a100_out, atol=2e-4, rtol=1e-4)


@pytest.mark.parametrize("hidden_size", [16, 33, 130])
def test_a100_gru_forward_precompute_input_matches_torch_gru(hidden_size: int) -> None:
    _requires_a100()
    torch.manual_seed(2028)
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
        a100_out = a100_gru_forward_layer_precompute_input(
            x,
            h0[0],
            gru.weight_ih_l0,
            gru.weight_hh_l0,
            gru.bias_ih_l0,
            gru.bias_hh_l0,
        )

    assert torch.allclose(torch_out, a100_out, atol=2e-4, rtol=1e-4)


@pytest.mark.parametrize("hidden_size", [16, 33, 130])
@pytest.mark.parametrize("subwarp_size", [8, 16])
def test_a100_gru_forward_subwarp_matches_torch_gru(
    hidden_size: int,
    subwarp_size: int,
) -> None:
    _requires_a100()
    torch.manual_seed(2030 + subwarp_size)
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
        a100_out = a100_gru_forward_layer_precompute_input_subwarp(
            x,
            h0[0],
            gru.weight_ih_l0,
            gru.weight_hh_l0,
            gru.bias_ih_l0,
            gru.bias_hh_l0,
            subwarp_size=subwarp_size,
        )

    assert torch.allclose(torch_out, a100_out, atol=2e-4, rtol=1e-4)


@pytest.mark.parametrize("hidden_size", [16, 33, 130])
def test_a100_gru_forward_fused_matches_torch_gru(hidden_size: int) -> None:
    _requires_a100()
    torch.manual_seed(2040)
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
        a100_out = a100_gru_forward_layer_precompute_input_fused(
            x,
            h0[0],
            gru.weight_ih_l0,
            gru.weight_hh_l0,
            gru.bias_ih_l0,
            gru.bias_hh_l0,
        )

    assert torch.allclose(torch_out, a100_out, atol=2e-4, rtol=1e-4)


@pytest.mark.parametrize("hidden_size", [16, 33, 130])
def test_a100_gru_forward_fused_pingpong_matches_torch_gru(hidden_size: int) -> None:
    _requires_a100()
    torch.manual_seed(2041)
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
        a100_out = a100_gru_forward_layer_precompute_input_fused_pingpong(
            x,
            h0[0],
            gru.weight_ih_l0,
            gru.weight_hh_l0,
            gru.bias_ih_l0,
            gru.bias_hh_l0,
        )

    assert torch.allclose(torch_out, a100_out, atol=2e-4, rtol=1e-4)


@pytest.mark.parametrize("hidden_size", [16, 33, 128, 130, 160])
def test_a100_gru_forward_fused_specialized_matches_torch_gru(hidden_size: int) -> None:
    _requires_a100()
    torch.manual_seed(2042)
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
        a100_out = a100_gru_forward_layer_precompute_input_fused_specialized(
            x,
            h0[0],
            gru.weight_ih_l0,
            gru.weight_hh_l0,
            gru.bias_ih_l0,
            gru.bias_hh_l0,
        )

    assert torch.allclose(torch_out, a100_out, atol=2e-4, rtol=1e-4)


@pytest.mark.parametrize("hidden_size", [16, 33, 130, 160, 256])
@pytest.mark.parametrize("ctas_per_batch", [2, 4])
def test_a100_gru_forward_cooperative_matches_torch_gru(
    hidden_size: int,
    ctas_per_batch: int,
) -> None:
    _requires_a100()
    torch.manual_seed(2043 + ctas_per_batch)
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
        a100_out = a100_gru_forward_layer_precompute_input_cooperative(
            x,
            h0[0],
            gru.weight_ih_l0,
            gru.weight_hh_l0,
            gru.bias_ih_l0,
            gru.bias_hh_l0,
            block_threads=256,
            ctas_per_batch=ctas_per_batch,
        )

    assert torch.allclose(torch_out, a100_out, atol=3e-4, rtol=1e-4)


def test_a100_gru_forward_cooperative_h256_matches_torch_gru() -> None:
    _requires_a100()
    torch.manual_seed(2051)
    device = torch.device("cuda")
    hidden_size = 256
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
        a100_out = a100_gru_forward_layer_precompute_input_cooperative_h256(
            x,
            h0[0],
            gru.weight_ih_l0,
            gru.weight_hh_l0,
            gru.bias_ih_l0,
            gru.bias_hh_l0,
        )

    assert torch.allclose(torch_out, a100_out, atol=3e-4, rtol=1e-4)


def test_a100_gru_forward_cooperative_h256_shmem_matches_torch_gru() -> None:
    _requires_a100()
    torch.manual_seed(2052)
    device = torch.device("cuda")
    hidden_size = 256
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
        a100_out = a100_gru_forward_layer_precompute_input_cooperative_h256_shmem(
            x,
            h0[0],
            gru.weight_ih_l0,
            gru.weight_hh_l0,
            gru.bias_ih_l0,
            gru.bias_hh_l0,
        )

    assert torch.allclose(torch_out, a100_out, atol=3e-4, rtol=1e-4)


def test_a100_gru_forward_cooperative_h256_parallel_update_matches_torch_gru() -> None:
    _requires_a100()
    torch.manual_seed(2053)
    device = torch.device("cuda")
    hidden_size = 256
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
        a100_out = a100_gru_forward_layer_precompute_input_cooperative_h256_parallel_update(
            x,
            h0[0],
            gru.weight_ih_l0,
            gru.weight_hh_l0,
            gru.bias_ih_l0,
            gru.bias_hh_l0,
        )

    assert torch.allclose(torch_out, a100_out, atol=3e-4, rtol=1e-4)


def test_a100_gru_forward_cooperative_h256_qwarp_shmem_matches_torch_gru() -> None:
    _requires_a100()
    torch.manual_seed(2054)
    device = torch.device("cuda")
    hidden_size = 256
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
        a100_out = a100_gru_forward_layer_precompute_input_cooperative_h256_qwarp_shmem(
            x,
            h0[0],
            gru.weight_ih_l0,
            gru.weight_hh_l0,
            gru.bias_ih_l0,
            gru.bias_hh_l0,
        )

    assert torch.allclose(torch_out, a100_out, atol=3e-4, rtol=1e-4)


def test_a100_gru_forward_cooperative_h256_cached_shmem_matches_torch_gru() -> None:
    _requires_a100()
    torch.manual_seed(2055)
    device = torch.device("cuda")
    hidden_size = 256
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
        a100_out = a100_gru_forward_layer_precompute_input_cooperative_h256_cached_shmem(
            x,
            h0[0],
            gru.weight_ih_l0,
            gru.weight_hh_l0,
            gru.bias_ih_l0,
            gru.bias_hh_l0,
        )

    assert torch.allclose(torch_out, a100_out, atol=3e-4, rtol=1e-4)


def test_a100_gru_forward_rejects_large_hidden_size() -> None:
    _requires_a100()
    device = torch.device("cuda")
    hidden_size = 257
    batch_size = 2
    seq_len = 4
    input_size = 3

    with pytest.raises(ValueError, match="hidden_size <= 256"):
        a100_gru_forward_layer(
            torch.randn(batch_size, seq_len, input_size, device=device),
            torch.randn(batch_size, hidden_size, device=device),
            torch.randn(3 * hidden_size, input_size, device=device),
            torch.randn(3 * hidden_size, hidden_size, device=device),
            torch.randn(3 * hidden_size, device=device),
            torch.randn(3 * hidden_size, device=device),
        )
