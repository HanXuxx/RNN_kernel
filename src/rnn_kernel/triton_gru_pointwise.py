from __future__ import annotations

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice


@triton.jit
def _tanh(x):
    return libdevice.tanh(x)


@triton.jit
def _gru_pointwise_forward_kernel(
    input_gates,
    hidden_gates,
    hidden_prev,
    hidden_next,
    total: tl.constexpr,
    hidden_size: tl.constexpr,
    block_size: tl.constexpr,
):
    offsets = tl.program_id(0) * block_size + tl.arange(0, block_size)
    mask = offsets < total
    hidden_idx = offsets % hidden_size
    batch_idx = offsets // hidden_size
    gate_base = batch_idx * (3 * hidden_size) + hidden_idx

    i_r = tl.load(input_gates + gate_base, mask=mask, other=0.0)
    i_z = tl.load(input_gates + gate_base + hidden_size, mask=mask, other=0.0)
    i_n = tl.load(input_gates + gate_base + 2 * hidden_size, mask=mask, other=0.0)
    h_r = tl.load(hidden_gates + gate_base, mask=mask, other=0.0)
    h_z = tl.load(hidden_gates + gate_base + hidden_size, mask=mask, other=0.0)
    h_n = tl.load(hidden_gates + gate_base + 2 * hidden_size, mask=mask, other=0.0)
    h_prev = tl.load(hidden_prev + offsets, mask=mask, other=0.0)

    reset_gate = tl.sigmoid(i_r + h_r)
    update_gate = tl.sigmoid(i_z + h_z)
    new_gate = _tanh(i_n + reset_gate * h_n)
    h_next = new_gate + update_gate * (h_prev - new_gate)

    tl.store(hidden_next + offsets, h_next, mask=mask)


@triton.jit
def _gru_pointwise_backward_kernel(
    grad_hidden_next,
    input_gates,
    hidden_gates,
    hidden_prev,
    grad_input_gates,
    grad_hidden_gates,
    grad_hidden_prev,
    total: tl.constexpr,
    hidden_size: tl.constexpr,
    block_size: tl.constexpr,
):
    offsets = tl.program_id(0) * block_size + tl.arange(0, block_size)
    mask = offsets < total
    hidden_idx = offsets % hidden_size
    batch_idx = offsets // hidden_size
    gate_base = batch_idx * (3 * hidden_size) + hidden_idx

    i_r = tl.load(input_gates + gate_base, mask=mask, other=0.0)
    i_z = tl.load(input_gates + gate_base + hidden_size, mask=mask, other=0.0)
    i_n = tl.load(input_gates + gate_base + 2 * hidden_size, mask=mask, other=0.0)
    h_r = tl.load(hidden_gates + gate_base, mask=mask, other=0.0)
    h_z = tl.load(hidden_gates + gate_base + hidden_size, mask=mask, other=0.0)
    h_n = tl.load(hidden_gates + gate_base + 2 * hidden_size, mask=mask, other=0.0)
    h_prev = tl.load(hidden_prev + offsets, mask=mask, other=0.0)
    grad_out = tl.load(grad_hidden_next + offsets, mask=mask, other=0.0)

    reset_gate = tl.sigmoid(i_r + h_r)
    update_gate = tl.sigmoid(i_z + h_z)
    new_gate = _tanh(i_n + reset_gate * h_n)

    grad_update = grad_out * (h_prev - new_gate)
    grad_new = grad_out * (1.0 - update_gate)
    grad_hidden_prev_direct = grad_out * update_gate

    grad_new_pre = grad_new * (1.0 - new_gate * new_gate)
    grad_reset = grad_new_pre * h_n
    grad_h_n = grad_new_pre * reset_gate
    grad_reset_pre = grad_reset * reset_gate * (1.0 - reset_gate)
    grad_update_pre = grad_update * update_gate * (1.0 - update_gate)

    tl.store(grad_input_gates + gate_base, grad_reset_pre, mask=mask)
    tl.store(grad_input_gates + gate_base + hidden_size, grad_update_pre, mask=mask)
    tl.store(grad_input_gates + gate_base + 2 * hidden_size, grad_new_pre, mask=mask)
    tl.store(grad_hidden_gates + gate_base, grad_reset_pre, mask=mask)
    tl.store(grad_hidden_gates + gate_base + hidden_size, grad_update_pre, mask=mask)
    tl.store(grad_hidden_gates + gate_base + 2 * hidden_size, grad_h_n, mask=mask)
    tl.store(grad_hidden_prev + offsets, grad_hidden_prev_direct, mask=mask)


class TritonGRUPointwise(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        input_gates: torch.Tensor,
        hidden_gates: torch.Tensor,
        hidden_prev: torch.Tensor,
    ) -> torch.Tensor:
        if not input_gates.is_cuda:
            raise RuntimeError("TritonGRUPointwise requires CUDA tensors.")
        if input_gates.dtype != torch.float32:
            raise RuntimeError("TritonGRUPointwise currently supports fp32 only.")

        input_gates = input_gates.contiguous()
        hidden_gates = hidden_gates.contiguous()
        hidden_prev = hidden_prev.contiguous()
        batch_size = hidden_prev.size(0)
        hidden_size = hidden_prev.size(1)
        total = batch_size * hidden_size
        hidden_next = torch.empty_like(hidden_prev)
        block_size = 256
        grid = (triton.cdiv(total, block_size),)
        _gru_pointwise_forward_kernel[grid](
            input_gates,
            hidden_gates,
            hidden_prev,
            hidden_next,
            total,
            hidden_size,
            block_size,
        )
        ctx.save_for_backward(input_gates, hidden_gates, hidden_prev)
        ctx.hidden_size = hidden_size
        ctx.block_size = block_size
        return hidden_next

    @staticmethod
    def backward(ctx, grad_hidden_next: torch.Tensor):
        input_gates, hidden_gates, hidden_prev = ctx.saved_tensors
        grad_hidden_next = grad_hidden_next.contiguous()
        grad_input_gates = torch.empty_like(input_gates)
        grad_hidden_gates = torch.empty_like(hidden_gates)
        grad_hidden_prev = torch.empty_like(hidden_prev)
        total = hidden_prev.numel()
        grid = (triton.cdiv(total, ctx.block_size),)
        _gru_pointwise_backward_kernel[grid](
            grad_hidden_next,
            input_gates,
            hidden_gates,
            hidden_prev,
            grad_input_gates,
            grad_hidden_gates,
            grad_hidden_prev,
            total,
            ctx.hidden_size,
            ctx.block_size,
        )
        return grad_input_gates, grad_hidden_gates, grad_hidden_prev


def triton_gru_pointwise(
    input_gates: torch.Tensor,
    hidden_gates: torch.Tensor,
    hidden_prev: torch.Tensor,
) -> torch.Tensor:
    return TritonGRUPointwise.apply(input_gates, hidden_gates, hidden_prev)
