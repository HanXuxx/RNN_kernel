import pytest
import torch

from rnn_kernel.a100 import A100GRUH256, copy_from_torch_gru
from rnn_kernel.a100.gru_autograd import (
    _a100_gru_h256_backward_step,
    _a100_gru_h256_backward_step_cooperative_split,
    _a100_gru_h256_backward_step_cooperative_split2,
    _a100_gru_h256_backward_step_cooperative_split2_cached_local,
    _a100_gru_h256_backward_step_cooperative_split2_gate_cache,
    _a100_gru_h256_backward_step_cooperative_split_cached,
    _a100_gru_h256_backward_step_recompute,
    _a100_gru_h256_pack_hidden_prev_time_major,
    _a100_gru_h256_pointwise_backward,
    _a100_gru_h256_recurrent_backward,
    _a100_gru_h256_recurrent_backward_split,
    _a100_gru_h256_recurrent_backward_tiled,
)


def _requires_a100() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    if torch.cuda.get_device_capability() != (8, 0):
        pytest.skip("A100/SM80 is required")


def test_a100_gru_h256_autograd_matches_torch_gru() -> None:
    _requires_a100()
    torch.manual_seed(2061)
    device = torch.device("cuda")
    input_size = 5
    hidden_size = 256
    batch_size = 2
    seq_len = 6

    torch_gru = torch.nn.GRU(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=1,
        batch_first=True,
    ).to(device)
    a100_gru = A100GRUH256(input_size=input_size).to(device)
    copy_from_torch_gru(a100_gru, torch_gru)

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
    assert torch.allclose(x_torch.grad, x_a100.grad, atol=8e-4, rtol=2e-4)
    assert torch.allclose(h0_torch.grad, h0_a100.grad, atol=8e-4, rtol=2e-4)

    for name, torch_param in torch_gru.named_parameters():
        a100_param = getattr(a100_gru, name)
        assert torch.allclose(torch_param.grad, a100_param.grad, atol=3e-3, rtol=5e-4)


def test_a100_gru_h256_rejects_non_h256() -> None:
    with pytest.raises(ValueError, match="hidden_size=256"):
        A100GRUH256(input_size=5, hidden_size=128)


def test_a100_gru_h256_pack_hidden_prev_time_major_matches_torch_layout() -> None:
    _requires_a100()
    torch.manual_seed(2122)
    device = torch.device("cuda")
    batch_size = 3
    seq_len = 7
    hidden_size = 256

    h0 = torch.randn(batch_size, hidden_size, device=device)
    output = torch.randn(batch_size, seq_len, hidden_size, device=device)

    packed = _a100_gru_h256_pack_hidden_prev_time_major(h0, output)
    reference = torch.cat((h0.unsqueeze(1), output[:, :-1, :]), dim=1).transpose(0, 1).contiguous()

    assert torch.equal(packed, reference)


def test_a100_gru_h256_autograd_supports_final_hidden_only_loss() -> None:
    _requires_a100()
    torch.manual_seed(2062)
    device = torch.device("cuda")
    input_size = 5
    hidden_size = 256
    batch_size = 2
    seq_len = 6

    torch_gru = torch.nn.GRU(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=1,
        batch_first=True,
    ).to(device)
    a100_gru = A100GRUH256(input_size=input_size).to(device)
    copy_from_torch_gru(a100_gru, torch_gru)

    x_torch = torch.randn(batch_size, seq_len, input_size, device=device, requires_grad=True)
    x_a100 = x_torch.detach().clone().requires_grad_(True)
    h0_torch = torch.randn(1, batch_size, hidden_size, device=device, requires_grad=True)
    h0_a100 = h0_torch.detach().clone().requires_grad_(True)

    _, torch_h = torch_gru(x_torch, h0_torch)
    _, a100_h = a100_gru(x_a100, h0_a100)
    grad_h = torch.randn_like(torch_h)
    torch_h.backward(grad_h)
    a100_h.backward(grad_h)

    assert torch.allclose(h0_torch.grad, h0_a100.grad, atol=8e-4, rtol=2e-4)
    assert torch.allclose(x_torch.grad, x_a100.grad, atol=8e-4, rtol=2e-4)


def test_a100_gru_h256_recurrent_backward_matches_torch_matmul() -> None:
    _requires_a100()
    torch.manual_seed(2063)
    device = torch.device("cuda")
    batch_size = 4
    hidden_size = 256

    grad_hidden_gates = torch.randn(batch_size, 3 * hidden_size, device=device)
    weight_hh = torch.randn(3 * hidden_size, hidden_size, device=device)
    grad_hidden_prev_direct = torch.randn(batch_size, hidden_size, device=device)

    torch_out = grad_hidden_prev_direct + grad_hidden_gates.matmul(weight_hh)
    a100_out = _a100_gru_h256_recurrent_backward(
        grad_hidden_gates,
        weight_hh,
        grad_hidden_prev_direct,
    )

    assert torch.allclose(torch_out, a100_out, atol=2e-4, rtol=2e-5)


def test_a100_gru_h256_tiled_recurrent_backward_matches_torch_matmul() -> None:
    _requires_a100()
    torch.manual_seed(2068)
    device = torch.device("cuda")
    batch_size = 16
    hidden_size = 256

    grad_hidden_gates = torch.randn(batch_size, 3 * hidden_size, device=device)
    weight_hh = torch.randn(3 * hidden_size, hidden_size, device=device)
    grad_hidden_prev_direct = torch.randn(batch_size, hidden_size, device=device)

    torch_out = grad_hidden_prev_direct + grad_hidden_gates.matmul(weight_hh)
    a100_out = _a100_gru_h256_recurrent_backward_tiled(
        grad_hidden_gates,
        weight_hh,
        grad_hidden_prev_direct,
    )

    assert torch.allclose(torch_out, a100_out, atol=2e-4, rtol=2e-5)


def test_a100_gru_h256_split_recurrent_backward_matches_torch_matmul() -> None:
    _requires_a100()
    torch.manual_seed(2070)
    device = torch.device("cuda")
    batch_size = 16
    hidden_size = 256
    split_count = 8

    grad_hidden_gates = torch.randn(batch_size, 3 * hidden_size, device=device)
    weight_hh = torch.randn(3 * hidden_size, hidden_size, device=device)
    grad_hidden_prev_direct = torch.randn(batch_size, hidden_size, device=device)
    partial_sums = torch.empty(split_count, batch_size, hidden_size, device=device)

    torch_out = grad_hidden_prev_direct + grad_hidden_gates.matmul(weight_hh)
    a100_out = _a100_gru_h256_recurrent_backward_split(
        grad_hidden_gates,
        weight_hh,
        grad_hidden_prev_direct,
        partial_sums,
        split_count=split_count,
    )

    assert torch.allclose(torch_out, a100_out, atol=2e-4, rtol=2e-5)

    partial_sums4 = torch.empty(4, batch_size, hidden_size, device=device)
    a100_out4 = _a100_gru_h256_recurrent_backward_split(
        grad_hidden_gates,
        weight_hh,
        grad_hidden_prev_direct,
        partial_sums4,
        split_count=4,
    )
    assert torch.allclose(torch_out, a100_out4, atol=2e-4, rtol=2e-5)


def test_a100_gru_h256_backward_step_matches_pointwise_and_matmul() -> None:
    _requires_a100()
    torch.manual_seed(2065)
    device = torch.device("cuda")
    batch_size = 3
    seq_len = 4
    hidden_size = 256
    step = 2

    grad_hidden_next = torch.randn(batch_size, hidden_size, device=device)
    input_gates = torch.randn(batch_size, seq_len, 3 * hidden_size, device=device)
    hidden_gates = torch.randn(batch_size, 3 * hidden_size, device=device)
    h0 = torch.randn(batch_size, hidden_size, device=device)
    output = torch.randn(batch_size, seq_len, hidden_size, device=device)
    weight_hh = torch.randn(3 * hidden_size, hidden_size, device=device)
    grad_input_gates_ref = torch.empty_like(input_gates)
    grad_input_gates_fused = torch.empty_like(input_gates)
    grad_hidden_gates_fused = torch.empty(batch_size, 3 * hidden_size, device=device)

    grad_hidden_gates_ref, grad_hidden_prev_direct = _a100_gru_h256_pointwise_backward(
        grad_hidden_next,
        input_gates,
        hidden_gates,
        h0,
        output,
        grad_input_gates_ref,
        step,
    )
    grad_hidden_prev_ref = grad_hidden_prev_direct + grad_hidden_gates_ref.matmul(weight_hh)
    grad_hidden_prev_fused = _a100_gru_h256_backward_step(
        grad_hidden_next,
        input_gates,
        hidden_gates,
        h0,
        output,
        weight_hh,
        grad_input_gates_fused,
        grad_hidden_gates_fused,
        step,
    )

    assert torch.allclose(grad_hidden_prev_ref, grad_hidden_prev_fused, atol=2e-4, rtol=2e-5)
    assert torch.allclose(grad_hidden_gates_ref, grad_hidden_gates_fused, atol=2e-5, rtol=2e-5)
    assert torch.allclose(
        grad_input_gates_ref[:, step, :],
        grad_input_gates_fused[:, step, :],
        atol=2e-5,
        rtol=2e-5,
    )


def test_a100_gru_h256_backward_step_recompute_matches_fused_step() -> None:
    _requires_a100()
    torch.manual_seed(2066)
    device = torch.device("cuda")
    batch_size = 3
    seq_len = 4
    hidden_size = 256
    step = 2

    grad_hidden_next = torch.randn(batch_size, hidden_size, device=device)
    input_gates = torch.randn(batch_size, seq_len, 3 * hidden_size, device=device)
    h0 = torch.randn(batch_size, hidden_size, device=device)
    output = torch.randn(batch_size, seq_len, hidden_size, device=device)
    weight_hh = torch.randn(3 * hidden_size, hidden_size, device=device)
    bias_hh = torch.randn(3 * hidden_size, device=device)
    h_prev = output[:, step - 1, :]
    hidden_gates = torch.nn.functional.linear(h_prev, weight_hh, bias_hh)
    grad_input_gates_fused = torch.empty_like(input_gates)
    grad_input_gates_recompute = torch.empty_like(input_gates)
    grad_hidden_gates_fused = torch.empty(batch_size, 3 * hidden_size, device=device)
    grad_hidden_gates_recompute = torch.empty(batch_size, 3 * hidden_size, device=device)

    grad_hidden_prev_fused = _a100_gru_h256_backward_step(
        grad_hidden_next,
        input_gates,
        hidden_gates,
        h0,
        output,
        weight_hh,
        grad_input_gates_fused,
        grad_hidden_gates_fused,
        step,
    )
    grad_hidden_prev_recompute = _a100_gru_h256_backward_step_recompute(
        grad_hidden_next,
        input_gates,
        h0,
        output,
        weight_hh,
        bias_hh,
        grad_input_gates_recompute,
        grad_hidden_gates_recompute,
        step,
    )

    assert torch.allclose(
        grad_hidden_prev_fused,
        grad_hidden_prev_recompute,
        atol=5e-4,
        rtol=8e-5,
    )
    assert torch.allclose(
        grad_hidden_gates_fused,
        grad_hidden_gates_recompute,
        atol=5e-4,
        rtol=8e-5,
    )
    assert torch.allclose(
        grad_input_gates_fused[:, step, :],
        grad_input_gates_recompute[:, step, :],
        atol=5e-4,
        rtol=8e-5,
    )


def test_a100_gru_h256_backward_step_cooperative_split_matches_fused_step() -> None:
    _requires_a100()
    torch.manual_seed(2073)
    device = torch.device("cuda")
    batch_size = 4
    seq_len = 5
    hidden_size = 256
    step = 3
    split_count = 4

    grad_hidden_next = torch.randn(batch_size, hidden_size, device=device)
    input_gates = torch.randn(batch_size, seq_len, 3 * hidden_size, device=device)
    hidden_gates = torch.randn(batch_size, 3 * hidden_size, device=device)
    h0 = torch.randn(batch_size, hidden_size, device=device)
    output = torch.randn(batch_size, seq_len, hidden_size, device=device)
    weight_hh = torch.randn(3 * hidden_size, hidden_size, device=device)
    grad_input_gates_fused = torch.empty_like(input_gates)
    grad_input_gates_coop = torch.empty_like(input_gates)
    grad_hidden_gates_fused = torch.empty(batch_size, 3 * hidden_size, device=device)
    grad_hidden_gates_coop = torch.empty(batch_size, 3 * hidden_size, device=device)
    partial_sums = torch.empty(batch_size, split_count, hidden_size, device=device)

    grad_hidden_prev_fused = _a100_gru_h256_backward_step(
        grad_hidden_next,
        input_gates,
        hidden_gates,
        h0,
        output,
        weight_hh,
        grad_input_gates_fused,
        grad_hidden_gates_fused,
        step,
    )
    grad_hidden_prev_coop = _a100_gru_h256_backward_step_cooperative_split(
        grad_hidden_next,
        input_gates,
        hidden_gates,
        h0,
        output,
        weight_hh,
        grad_input_gates_coop,
        grad_hidden_gates_coop,
        partial_sums,
        step,
        split_count=split_count,
    )

    assert torch.allclose(grad_hidden_prev_fused, grad_hidden_prev_coop, atol=5e-4, rtol=8e-5)
    assert torch.allclose(grad_hidden_gates_fused, grad_hidden_gates_coop, atol=2e-5, rtol=2e-5)
    assert torch.allclose(
        grad_input_gates_fused[:, step, :],
        grad_input_gates_coop[:, step, :],
        atol=2e-5,
        rtol=2e-5,
    )

    split_count2 = 2
    grad_input_gates_coop2 = torch.empty_like(input_gates)
    grad_hidden_gates_coop2 = torch.empty(batch_size, 3 * hidden_size, device=device)
    partial_sums2 = torch.empty(batch_size, split_count2, hidden_size, device=device)
    grad_hidden_prev_coop2 = _a100_gru_h256_backward_step_cooperative_split(
        grad_hidden_next,
        input_gates,
        hidden_gates,
        h0,
        output,
        weight_hh,
        grad_input_gates_coop2,
        grad_hidden_gates_coop2,
        partial_sums2,
        step,
        split_count=split_count2,
    )
    assert torch.allclose(grad_hidden_prev_fused, grad_hidden_prev_coop2, atol=5e-4, rtol=8e-5)
    assert torch.allclose(grad_hidden_gates_fused, grad_hidden_gates_coop2, atol=2e-5, rtol=2e-5)

    grad_input_gates_cached = torch.empty_like(input_gates)
    grad_hidden_gates_cached = torch.empty(batch_size, 3 * hidden_size, device=device)
    partial_sums_cached = torch.empty(batch_size, split_count2, hidden_size, device=device)
    grad_hidden_prev_cached = _a100_gru_h256_backward_step_cooperative_split_cached(
        grad_hidden_next,
        input_gates,
        hidden_gates,
        h0,
        output,
        weight_hh,
        grad_input_gates_cached,
        grad_hidden_gates_cached,
        partial_sums_cached,
        step,
        split_count=split_count2,
    )
    assert torch.allclose(grad_hidden_prev_fused, grad_hidden_prev_cached, atol=5e-4, rtol=8e-5)
    assert torch.allclose(grad_hidden_gates_fused, grad_hidden_gates_cached, atol=2e-5, rtol=2e-5)

    grad_input_gates_split2 = torch.empty_like(input_gates)
    grad_hidden_gates_split2 = torch.empty(batch_size, 3 * hidden_size, device=device)
    partial_sums_split2 = torch.empty(batch_size, 2, hidden_size, device=device)
    grad_hidden_prev_split2 = _a100_gru_h256_backward_step_cooperative_split2(
        grad_hidden_next,
        input_gates,
        hidden_gates,
        h0,
        output,
        weight_hh,
        grad_input_gates_split2,
        grad_hidden_gates_split2,
        partial_sums_split2,
        step,
    )
    assert torch.allclose(grad_hidden_prev_fused, grad_hidden_prev_split2, atol=5e-4, rtol=8e-5)
    assert torch.allclose(grad_hidden_gates_fused, grad_hidden_gates_split2, atol=2e-5, rtol=2e-5)

    grad_input_gates_cached_local = torch.empty_like(input_gates)
    grad_hidden_gates_cached_local = torch.empty(batch_size, 3 * hidden_size, device=device)
    partial_sums_cached_local = torch.empty(batch_size, 2, hidden_size, device=device)
    grad_hidden_prev_cached_local = _a100_gru_h256_backward_step_cooperative_split2_cached_local(
        grad_hidden_next,
        input_gates,
        hidden_gates,
        h0,
        output,
        weight_hh,
        grad_input_gates_cached_local,
        grad_hidden_gates_cached_local,
        partial_sums_cached_local,
        step,
    )
    assert torch.allclose(
        grad_hidden_prev_fused,
        grad_hidden_prev_cached_local,
        atol=5e-4,
        rtol=8e-5,
    )
    assert torch.allclose(
        grad_hidden_gates_fused,
        grad_hidden_gates_cached_local,
        atol=2e-5,
        rtol=2e-5,
    )

    gate_cache = torch.zeros(batch_size, seq_len, 4 * hidden_size, device=device)
    input_step = input_gates[:, step, :]
    reset_gate = torch.sigmoid(input_step[:, :hidden_size] + hidden_gates[:, :hidden_size])
    update_gate = torch.sigmoid(
        input_step[:, hidden_size : 2 * hidden_size]
        + hidden_gates[:, hidden_size : 2 * hidden_size]
    )
    new_gate = torch.tanh(input_step[:, 2 * hidden_size :] + reset_gate * hidden_gates[:, 2 * hidden_size :])
    gate_cache[:, step, :hidden_size] = reset_gate
    gate_cache[:, step, hidden_size : 2 * hidden_size] = update_gate
    gate_cache[:, step, 2 * hidden_size : 3 * hidden_size] = new_gate
    gate_cache[:, step, 3 * hidden_size :] = hidden_gates[:, 2 * hidden_size :]
    grad_input_gates_gate_cache = torch.empty_like(input_gates)
    grad_hidden_gates_gate_cache = torch.empty(batch_size, 3 * hidden_size, device=device)
    partial_sums_gate_cache = torch.empty(batch_size, 2, hidden_size, device=device)
    grad_hidden_prev_gate_cache = _a100_gru_h256_backward_step_cooperative_split2_gate_cache(
        grad_hidden_next,
        gate_cache,
        h0,
        output,
        weight_hh,
        grad_input_gates_gate_cache,
        grad_hidden_gates_gate_cache,
        partial_sums_gate_cache,
        step,
    )
    assert torch.allclose(
        grad_hidden_prev_fused,
        grad_hidden_prev_gate_cache,
        atol=5e-4,
        rtol=8e-5,
    )
    assert torch.allclose(
        grad_hidden_gates_fused,
        grad_hidden_gates_gate_cache,
        atol=2e-5,
        rtol=2e-5,
    )
    assert torch.allclose(
        grad_input_gates_fused[:, step, :],
        grad_input_gates_gate_cache[:, step, :],
        atol=2e-5,
        rtol=2e-5,
    )


def test_a100_gru_h256_recurrent_backward_autograd_matches_torch_gru() -> None:
    _requires_a100()
    torch.manual_seed(2064)
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
    a100_gru = A100GRUH256(
        input_size=input_size,
        use_recurrent_backward_kernel=True,
    ).to(device)
    copy_from_torch_gru(a100_gru, torch_gru)

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
        assert torch.allclose(torch_param.grad, a100_param.grad, atol=4e-3, rtol=8e-4)


def test_a100_gru_h256_recompute_autograd_matches_torch_gru() -> None:
    _requires_a100()
    torch.manual_seed(2067)
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
    a100_gru = A100GRUH256(
        input_size=input_size,
        recompute_hidden_gates=True,
    ).to(device)
    copy_from_torch_gru(a100_gru, torch_gru)

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


def test_a100_gru_h256_tiled_recurrent_autograd_matches_torch_gru() -> None:
    _requires_a100()
    torch.manual_seed(2069)
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
    a100_gru = A100GRUH256(
        input_size=input_size,
        use_tiled_recurrent_backward_kernel=True,
    ).to(device)
    copy_from_torch_gru(a100_gru, torch_gru)

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


def test_a100_gru_h256_split_recurrent_autograd_matches_torch_gru() -> None:
    _requires_a100()
    torch.manual_seed(2072)
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
    a100_gru = A100GRUH256(
        input_size=input_size,
        use_split_recurrent_backward_kernel=True,
    ).to(device)
    copy_from_torch_gru(a100_gru, torch_gru)

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


def test_a100_gru_h256_cooperative_split_autograd_matches_torch_gru() -> None:
    _requires_a100()
    torch.manual_seed(2074)
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
    a100_gru = A100GRUH256(
        input_size=input_size,
        use_cooperative_split_backward_kernel=True,
        cooperative_split_count=4,
    ).to(device)
    copy_from_torch_gru(a100_gru, torch_gru)

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


def test_a100_gru_h256_cooperative_split_cached_autograd_matches_torch_gru() -> None:
    _requires_a100()
    torch.manual_seed(2075)
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
    a100_gru = A100GRUH256(
        input_size=input_size,
        use_cooperative_split_cached_backward_kernel=True,
        cooperative_split_count=2,
    ).to(device)
    copy_from_torch_gru(a100_gru, torch_gru)

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


def test_a100_gru_h256_cooperative_split2_autograd_matches_torch_gru() -> None:
    _requires_a100()
    torch.manual_seed(2076)
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
    a100_gru = A100GRUH256(
        input_size=input_size,
        use_cooperative_split2_backward_kernel=True,
    ).to(device)
    copy_from_torch_gru(a100_gru, torch_gru)

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


def test_a100_gru_h256_cooperative_split2_cached_local_autograd_matches_torch_gru() -> None:
    _requires_a100()
    torch.manual_seed(2077)
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
    a100_gru = A100GRUH256(
        input_size=input_size,
        use_cooperative_split2_cached_local_backward_kernel=True,
    ).to(device)
    copy_from_torch_gru(a100_gru, torch_gru)

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


def test_a100_gru_h256_gate_cache_autograd_matches_torch_gru() -> None:
    _requires_a100()
    torch.manual_seed(2078)
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
    a100_gru = A100GRUH256(
        input_size=input_size,
        use_gate_cache_backward_kernel=True,
    ).to(device)
    copy_from_torch_gru(a100_gru, torch_gru)

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


def test_a100_gru_h256_persistent_autograd_matches_torch_gru() -> None:
    _requires_a100()
    torch.manual_seed(2079)
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
    a100_gru = A100GRUH256(
        input_size=input_size,
        use_persistent_backward_kernel=True,
    ).to(device)
    copy_from_torch_gru(a100_gru, torch_gru)

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


def test_a100_gru_h256_persistent_state_autograd_matches_torch_gru() -> None:
    _requires_a100()
    torch.manual_seed(2080)
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
    a100_gru = A100GRUH256(
        input_size=input_size,
        use_persistent_state_backward_kernel=True,
    ).to(device)
    copy_from_torch_gru(a100_gru, torch_gru)

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


def test_a100_gru_h256_persistent_state_local_autograd_matches_torch_gru() -> None:
    _requires_a100()
    torch.manual_seed(2081)
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
    a100_gru = A100GRUH256(
        input_size=input_size,
        use_persistent_state_local_backward_kernel=True,
    ).to(device)
    copy_from_torch_gru(a100_gru, torch_gru)

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


def test_a100_gru_h256_persistent_state4_autograd_matches_torch_gru() -> None:
    _requires_a100()
    torch.manual_seed(2082)
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
    a100_gru = A100GRUH256(
        input_size=input_size,
        use_persistent_state4_backward_kernel=True,
    ).to(device)
    copy_from_torch_gru(a100_gru, torch_gru)

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


def test_a100_gru_h256_persistent_state8_autograd_matches_torch_gru() -> None:
    _requires_a100()
    torch.manual_seed(2083)
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
    a100_gru = A100GRUH256(
        input_size=input_size,
        use_persistent_state8_backward_kernel=True,
    ).to(device)
    copy_from_torch_gru(a100_gru, torch_gru)

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


def test_a100_gru_h256_persistent_state16_autograd_matches_torch_gru() -> None:
    _requires_a100()
    torch.manual_seed(2084)
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
    a100_gru = A100GRUH256(
        input_size=input_size,
        use_persistent_state16_backward_kernel=True,
    ).to(device)
    copy_from_torch_gru(a100_gru, torch_gru)

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


def test_a100_gru_h256_persistent_state32_autograd_matches_torch_gru() -> None:
    _requires_a100()
    torch.manual_seed(2085)
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
    a100_gru = A100GRUH256(
        input_size=input_size,
        use_persistent_state32_backward_kernel=True,
    ).to(device)
    copy_from_torch_gru(a100_gru, torch_gru)

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


def test_a100_gru_h256_persistent_state16_gate_cache_autograd_matches_torch_gru() -> None:
    _requires_a100()
    torch.manual_seed(2087)
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
    a100_gru = A100GRUH256(
        input_size=input_size,
        use_persistent_state16_gate_cache_backward_kernel=True,
    ).to(device)
    copy_from_torch_gru(a100_gru, torch_gru)

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


def test_a100_gru_h256_persistent_state16_gate_cache_tiled_autograd_matches_torch_gru() -> None:
    _requires_a100()
    torch.manual_seed(2090)
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
    a100_gru = A100GRUH256(
        input_size=input_size,
        use_persistent_state16_gate_cache_tiled_backward_kernel=True,
    ).to(device)
    copy_from_torch_gru(a100_gru, torch_gru)

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


def test_a100_gru_h256_persistent_state32_gate_cache_tiled_autograd_matches_torch_gru() -> None:
    _requires_a100()
    torch.manual_seed(2091)
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
    a100_gru = A100GRUH256(
        input_size=input_size,
        use_persistent_state32_gate_cache_tiled_backward_kernel=True,
    ).to(device)
    copy_from_torch_gru(a100_gru, torch_gru)

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


def test_a100_gru_h256_persistent_state16_grad_coeff_cache_tiled_autograd_matches_torch_gru() -> None:
    _requires_a100()
    torch.manual_seed(2092)
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
    a100_gru = A100GRUH256(
        input_size=input_size,
        use_persistent_state16_grad_coeff_cache_tiled_backward_kernel=True,
    ).to(device)
    copy_from_torch_gru(a100_gru, torch_gru)

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


def test_a100_gru_h256_persistent_state16_gate_cache_tiled_cta6_autograd_matches_torch_gru() -> None:
    _requires_a100()
    torch.manual_seed(2093)
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
    a100_gru = A100GRUH256(
        input_size=input_size,
        use_persistent_state16_gate_cache_tiled_backward_kernel=True,
        use_gate_cache_cta6_forward_kernel=True,
    ).to(device)
    copy_from_torch_gru(a100_gru, torch_gru)

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


def test_a100_gru_h256_persistent_state16_gate_cache_tiled_htile2_autograd_matches_torch_gru() -> None:
    _requires_a100()
    torch.manual_seed(2094)
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
    a100_gru = A100GRUH256(
        input_size=input_size,
        use_persistent_state16_gate_cache_tiled_backward_kernel=True,
        use_gate_cache_htile2_forward_kernel=True,
    ).to(device)
    copy_from_torch_gru(a100_gru, torch_gru)

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


def test_a100_gru_h256_persistent_state16_gate_cache_tiled_htile4_autograd_matches_torch_gru() -> None:
    _requires_a100()
    torch.manual_seed(2095)
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
    a100_gru = A100GRUH256(
        input_size=input_size,
        block_threads=256,
        use_persistent_state16_gate_cache_tiled_backward_kernel=True,
        use_gate_cache_htile4_forward_kernel=True,
    ).to(device)
    copy_from_torch_gru(a100_gru, torch_gru)

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


def test_a100_gru_h256_persistent_state16_gate_cache_tiled_htile4_compact_autograd_matches_torch_gru() -> None:
    _requires_a100()
    torch.manual_seed(2096)
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
    a100_gru = A100GRUH256(
        input_size=input_size,
        block_threads=256,
        use_persistent_state16_gate_cache_tiled_backward_kernel=True,
        use_gate_cache_htile4_compact_forward_kernel=True,
    ).to(device)
    copy_from_torch_gru(a100_gru, torch_gru)

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


def test_a100_gru_h256_persistent_state16_gate_cache_tiled_htile4_compact_hoist_autograd_matches_torch_gru() -> None:
    _requires_a100()
    torch.manual_seed(2098)
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
    a100_gru = A100GRUH256(
        input_size=input_size,
        block_threads=256,
        use_persistent_state16_gate_cache_tiled_backward_kernel=True,
        use_gate_cache_htile4_compact_hoist_forward_kernel=True,
    ).to(device)
    copy_from_torch_gru(a100_gru, torch_gru)

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


def test_a100_gru_h256_persistent_state16_gate_cache_tiled_htile4_compact_hoist_row4_autograd_matches_torch_gru() -> None:
    _requires_a100()
    torch.manual_seed(2099)
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
    a100_gru = A100GRUH256(
        input_size=input_size,
        block_threads=256,
        use_persistent_state16_gate_cache_tiled_backward_kernel=True,
        use_gate_cache_htile4_compact_hoist_row4_forward_kernel=True,
    ).to(device)
    copy_from_torch_gru(a100_gru, torch_gru)

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


def test_a100_gru_h256_persistent_state8_gate_cache_tiled_htile4_compact_hoist_row4_autograd_matches_torch_gru() -> None:
    _requires_a100()
    torch.manual_seed(2101)
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
    a100_gru = A100GRUH256(
        input_size=input_size,
        block_threads=256,
        use_persistent_state8_gate_cache_tiled_backward_kernel=True,
        use_gate_cache_htile4_compact_hoist_row4_forward_kernel=True,
    ).to(device)
    copy_from_torch_gru(a100_gru, torch_gru)

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


def test_a100_gru_h256_persistent_state16_gate_cache_tiled_weight_shmem_htile4_compact_hoist_row4_autograd_matches_torch_gru() -> None:
    _requires_a100()
    torch.manual_seed(2102)
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
    a100_gru = A100GRUH256(
        input_size=input_size,
        block_threads=256,
        use_persistent_state16_gate_cache_tiled_weight_shmem_backward_kernel=True,
        use_gate_cache_htile4_compact_hoist_row4_forward_kernel=True,
    ).to(device)
    copy_from_torch_gru(a100_gru, torch_gru)

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


def test_a100_gru_h256_persistent_state16_gate_cache_tiled_weight_shmem_split0_keep_htile4_compact_hoist_row4_autograd_matches_torch_gru() -> None:
    _requires_a100()
    torch.manual_seed(2107)
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
    a100_gru = A100GRUH256(
        input_size=input_size,
        block_threads=256,
        use_persistent_state16_gate_cache_tiled_weight_shmem_split0_keep_backward_kernel=True,
        use_gate_cache_htile4_compact_hoist_row4_forward_kernel=True,
    ).to(device)
    copy_from_torch_gru(a100_gru, torch_gru)

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


def test_a100_gru_h256_persistent_state12_gate_cache_tiled_weight_shmem_split0_keep_htile4_compact_hoist_row4_autograd_matches_torch_gru() -> None:
    _requires_a100()
    torch.manual_seed(2109)
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
    a100_gru = A100GRUH256(
        input_size=input_size,
        block_threads=256,
        use_persistent_state12_gate_cache_tiled_weight_shmem_split0_keep_backward_kernel=True,
        use_gate_cache_htile4_compact_hoist_row4_forward_kernel=True,
    ).to(device)
    copy_from_torch_gru(a100_gru, torch_gru)

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


def test_a100_gru_h256_persistent_state6_gate_cache_tiled_weight_shmem_split0_keep_htile4_compact_hoist_row4_autograd_matches_torch_gru() -> None:
    _requires_a100()
    torch.manual_seed(2112)
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
    a100_gru = A100GRUH256(
        input_size=input_size,
        block_threads=256,
        use_persistent_state6_gate_cache_tiled_weight_shmem_split0_keep_backward_kernel=True,
        use_gate_cache_htile4_compact_hoist_row4_forward_kernel=True,
    ).to(device)
    copy_from_torch_gru(a100_gru, torch_gru)

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


def test_a100_gru_h256_persistent_state6_gate_cache_tiled_weight_shmem_split0_keep_htile4_compact_hoist_row4_pack_prev_autograd_matches_torch_gru() -> None:
    _requires_a100()
    torch.manual_seed(2123)
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
    a100_gru = A100GRUH256(
        input_size=input_size,
        block_threads=256,
        use_persistent_state6_gate_cache_tiled_weight_shmem_split0_keep_backward_kernel=True,
        use_gate_cache_htile4_compact_hoist_row4_forward_kernel=True,
        use_pack_hidden_prev_time_major_kernel=True,
    ).to(device)
    copy_from_torch_gru(a100_gru, torch_gru)

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


def test_a100_gru_h256_persistent_state5_gate_cache_tiled_weight_shmem_split0_keep_htile4_compact_hoist_row4_autograd_matches_torch_gru() -> None:
    _requires_a100()
    torch.manual_seed(2119)
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
    a100_gru = A100GRUH256(
        input_size=input_size,
        block_threads=256,
        use_persistent_state5_gate_cache_tiled_weight_shmem_split0_keep_backward_kernel=True,
        use_gate_cache_htile4_compact_hoist_row4_forward_kernel=True,
    ).to(device)
    copy_from_torch_gru(a100_gru, torch_gru)

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


def test_a100_gru_h256_persistent_state6_gate_cache_tiled_weight_shmem_split0_keep_htile4_compact_hoist_row4_ldg_autograd_matches_torch_gru() -> None:
    _requires_a100()
    torch.manual_seed(2118)
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
    a100_gru = A100GRUH256(
        input_size=input_size,
        block_threads=256,
        use_persistent_state6_gate_cache_tiled_weight_shmem_split0_keep_backward_kernel=True,
        use_gate_cache_htile4_compact_hoist_row4_ldg_forward_kernel=True,
    ).to(device)
    copy_from_torch_gru(a100_gru, torch_gru)

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


def test_a100_gru_h256_persistent_state6_gate_cache_tiled_weight_shmem_split0_keep_htile4_compact_hoist_row4_parallel_update_autograd_matches_torch_gru() -> None:
    _requires_a100()
    torch.manual_seed(2120)
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
    a100_gru = A100GRUH256(
        input_size=input_size,
        block_threads=256,
        use_persistent_state6_gate_cache_tiled_weight_shmem_split0_keep_backward_kernel=True,
        use_gate_cache_htile4_compact_hoist_row4_parallel_update_forward_kernel=True,
    ).to(device)
    copy_from_torch_gru(a100_gru, torch_gru)

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


def test_a100_gru_h256_persistent_state6_gate_cache_tiled_weight_shmem_split0_keep_htile4_compact_hoist_row4_prev_cache_autograd_matches_torch_gru() -> None:
    _requires_a100()
    torch.manual_seed(2121)
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
    a100_gru = A100GRUH256(
        input_size=input_size,
        block_threads=256,
        use_persistent_state6_gate_cache_tiled_weight_shmem_split0_keep_backward_kernel=True,
        use_gate_cache_htile4_compact_hoist_row4_prev_cache_forward_kernel=True,
    ).to(device)
    copy_from_torch_gru(a100_gru, torch_gru)

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


def test_a100_gru_h256_persistent_state12_gate_cache_tiled_weight_shmem_split0_keep_unroll8_htile4_compact_hoist_row4_autograd_matches_torch_gru() -> None:
    _requires_a100()
    torch.manual_seed(2111)
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
    a100_gru = A100GRUH256(
        input_size=input_size,
        block_threads=256,
        use_persistent_state12_gate_cache_tiled_weight_shmem_split0_keep_unroll8_backward_kernel=True,
        use_gate_cache_htile4_compact_hoist_row4_forward_kernel=True,
    ).to(device)
    copy_from_torch_gru(a100_gru, torch_gru)

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


def test_a100_gru_h256_persistent_state24_gate_cache_tiled_weight_shmem_split0_keep_htile4_compact_hoist_row4_autograd_matches_torch_gru() -> None:
    _requires_a100()
    torch.manual_seed(2110)
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
    a100_gru = A100GRUH256(
        input_size=input_size,
        block_threads=256,
        use_persistent_state24_gate_cache_tiled_weight_shmem_split0_keep_backward_kernel=True,
        use_gate_cache_htile4_compact_hoist_row4_forward_kernel=True,
    ).to(device)
    copy_from_torch_gru(a100_gru, torch_gru)

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


def test_a100_gru_h256_persistent_state16_gate_cache_tiled_weight_shmem_split0_keep_own_shmem_htile4_compact_hoist_row4_autograd_matches_torch_gru() -> None:
    _requires_a100()
    torch.manual_seed(2108)
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
    a100_gru = A100GRUH256(
        input_size=input_size,
        block_threads=256,
        use_persistent_state16_gate_cache_tiled_weight_shmem_split0_keep_own_shmem_backward_kernel=True,
        use_gate_cache_htile4_compact_hoist_row4_forward_kernel=True,
    ).to(device)
    copy_from_torch_gru(a100_gru, torch_gru)

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


def test_a100_gru_h256_persistent_state16_gate_cache_tiled_weight_shmem_htile4_compact_hoist_row4_forward_weight_shmem_autograd_matches_torch_gru() -> None:
    _requires_a100()
    torch.manual_seed(2103)
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
    a100_gru = A100GRUH256(
        input_size=input_size,
        block_threads=256,
        use_persistent_state16_gate_cache_tiled_weight_shmem_backward_kernel=True,
        use_gate_cache_htile4_compact_hoist_row4_weight_shmem_forward_kernel=True,
    ).to(device)
    copy_from_torch_gru(a100_gru, torch_gru)

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


def test_a100_gru_h256_persistent_state16_gate_cache_tiled_weight_shmem_htile4_compact_hoist_row4_forward_hidden_shmem_autograd_matches_torch_gru() -> None:
    _requires_a100()
    torch.manual_seed(2104)
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
    a100_gru = A100GRUH256(
        input_size=input_size,
        block_threads=256,
        use_persistent_state16_gate_cache_tiled_weight_shmem_backward_kernel=True,
        use_gate_cache_htile4_compact_hoist_row4_hidden_shmem_forward_kernel=True,
    ).to(device)
    copy_from_torch_gru(a100_gru, torch_gru)

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


def test_a100_gru_h256_persistent_state16_gate_cache_tiled_weight_shmem_htile4_compact_hoist_qwarp_autograd_matches_torch_gru() -> None:
    _requires_a100()
    torch.manual_seed(2105)
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
    a100_gru = A100GRUH256(
        input_size=input_size,
        block_threads=256,
        use_persistent_state16_gate_cache_tiled_weight_shmem_backward_kernel=True,
        use_gate_cache_htile4_compact_hoist_qwarp_forward_kernel=True,
    ).to(device)
    copy_from_torch_gru(a100_gru, torch_gru)

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


def test_a100_gru_h256_persistent_state16_gate_cache_tiled_weight_shmem_htile4_compact_hoist_row3_autograd_matches_torch_gru() -> None:
    _requires_a100()
    torch.manual_seed(2106)
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
    a100_gru = A100GRUH256(
        input_size=input_size,
        block_threads=256,
        use_persistent_state16_gate_cache_tiled_weight_shmem_backward_kernel=True,
        use_gate_cache_htile4_compact_hoist_row3_forward_kernel=True,
    ).to(device)
    copy_from_torch_gru(a100_gru, torch_gru)

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


def test_a100_gru_h256_persistent_state16_gate_cache_tiled_htile8_compact_autograd_matches_torch_gru() -> None:
    _requires_a100()
    torch.manual_seed(2097)
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
    a100_gru = A100GRUH256(
        input_size=input_size,
        block_threads=128,
        use_persistent_state16_gate_cache_tiled_backward_kernel=True,
        use_gate_cache_htile8_compact_forward_kernel=True,
    ).to(device)
    copy_from_torch_gru(a100_gru, torch_gru)

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


def test_a100_gru_h256_persistent_state16_gate_cache_parallel_update_autograd_matches_torch_gru() -> None:
    _requires_a100()
    torch.manual_seed(2088)
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
    a100_gru = A100GRUH256(
        input_size=input_size,
        use_persistent_state16_gate_cache_backward_kernel=True,
        use_gate_cache_parallel_update_forward_kernel=True,
    ).to(device)
    copy_from_torch_gru(a100_gru, torch_gru)

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


def test_a100_gru_h256_persistent_state16_gate_cache_cta8_autograd_matches_torch_gru() -> None:
    _requires_a100()
    torch.manual_seed(2089)
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
    a100_gru = A100GRUH256(
        input_size=input_size,
        use_persistent_state16_gate_cache_backward_kernel=True,
        use_gate_cache_cta8_forward_kernel=True,
    ).to(device)
    copy_from_torch_gru(a100_gru, torch_gru)

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


def test_a100_gru_h256_persistent_state16_global_gates_autograd_matches_torch_gru() -> None:
    _requires_a100()
    torch.manual_seed(2086)
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
    a100_gru = A100GRUH256(
        input_size=input_size,
        use_persistent_state16_global_gates_backward_kernel=True,
    ).to(device)
    copy_from_torch_gru(a100_gru, torch_gru)

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
