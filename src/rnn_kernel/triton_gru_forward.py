from __future__ import annotations

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice


@triton.jit
def _tanh(x):
    return libdevice.tanh(x)


@triton.jit
def _load_input_gate(
    x,
    weight_ih,
    bias_ih,
    batch_idx,
    step_idx,
    gate_idx: tl.constexpr,
    hidden_offsets,
    hidden_mask,
    seq_len: tl.constexpr,
    input_size: tl.constexpr,
    hidden_size: tl.constexpr,
):
    acc = tl.load(
        bias_ih + gate_idx * hidden_size + hidden_offsets,
        mask=hidden_mask,
        other=0.0,
    )
    for input_idx in tl.range(0, input_size):
        x_value = tl.load(x + (batch_idx * seq_len + step_idx) * input_size + input_idx)
        weight = tl.load(
            weight_ih + (gate_idx * hidden_size + hidden_offsets) * input_size + input_idx,
            mask=hidden_mask,
            other=0.0,
        )
        acc += x_value * weight
    return acc


@triton.jit
def _load_hidden_gate(
    hidden_work,
    weight_hh,
    bias_hh,
    batch_idx,
    gate_idx: tl.constexpr,
    hidden_offsets,
    hidden_mask,
    hidden_size: tl.constexpr,
    block_k: tl.constexpr,
):
    acc = tl.load(
        bias_hh + gate_idx * hidden_size + hidden_offsets,
        mask=hidden_mask,
        other=0.0,
    )
    for k_base in tl.range(0, hidden_size, block_k):
        k_offsets = k_base + tl.arange(0, block_k)
        k_mask = k_offsets < hidden_size
        hidden_values = tl.load(
            hidden_work + batch_idx * hidden_size + k_offsets,
            mask=k_mask,
            other=0.0,
        )
        weights = tl.load(
            weight_hh
            + (gate_idx * hidden_size + hidden_offsets[:, None]) * hidden_size
            + k_offsets[None, :],
            mask=hidden_mask[:, None] & k_mask[None, :],
            other=0.0,
        )
        acc += tl.sum(weights * hidden_values[None, :], axis=1)
    return acc


@triton.jit
def _gru_forward_time_loop_kernel(
    x,
    h0,
    weight_ih,
    weight_hh,
    bias_ih,
    bias_hh,
    output,
    hidden_work,
    seq_len: tl.constexpr,
    input_size: tl.constexpr,
    hidden_size: tl.constexpr,
    block_h: tl.constexpr,
    block_k: tl.constexpr,
):
    batch_idx = tl.program_id(0)
    hidden_offsets = tl.arange(0, block_h)
    hidden_mask = hidden_offsets < hidden_size

    initial_hidden = tl.load(
        h0 + batch_idx * hidden_size + hidden_offsets,
        mask=hidden_mask,
        other=0.0,
    )
    tl.store(
        hidden_work + batch_idx * hidden_size + hidden_offsets,
        initial_hidden,
        mask=hidden_mask,
    )

    for step_idx in tl.range(0, seq_len):
        i_r = _load_input_gate(
            x,
            weight_ih,
            bias_ih,
            batch_idx,
            step_idx,
            0,
            hidden_offsets,
            hidden_mask,
            seq_len,
            input_size,
            hidden_size,
        )
        i_z = _load_input_gate(
            x,
            weight_ih,
            bias_ih,
            batch_idx,
            step_idx,
            1,
            hidden_offsets,
            hidden_mask,
            seq_len,
            input_size,
            hidden_size,
        )
        i_n = _load_input_gate(
            x,
            weight_ih,
            bias_ih,
            batch_idx,
            step_idx,
            2,
            hidden_offsets,
            hidden_mask,
            seq_len,
            input_size,
            hidden_size,
        )
        h_r = _load_hidden_gate(
            hidden_work,
            weight_hh,
            bias_hh,
            batch_idx,
            0,
            hidden_offsets,
            hidden_mask,
            hidden_size,
            block_k,
        )
        h_z = _load_hidden_gate(
            hidden_work,
            weight_hh,
            bias_hh,
            batch_idx,
            1,
            hidden_offsets,
            hidden_mask,
            hidden_size,
            block_k,
        )
        h_n = _load_hidden_gate(
            hidden_work,
            weight_hh,
            bias_hh,
            batch_idx,
            2,
            hidden_offsets,
            hidden_mask,
            hidden_size,
            block_k,
        )
        hidden_prev = tl.load(
            hidden_work + batch_idx * hidden_size + hidden_offsets,
            mask=hidden_mask,
            other=0.0,
        )
        reset_gate = tl.sigmoid(i_r + h_r)
        update_gate = tl.sigmoid(i_z + h_z)
        new_gate = _tanh(i_n + reset_gate * h_n)
        hidden_next = new_gate + update_gate * (hidden_prev - new_gate)

        tl.store(
            hidden_work + batch_idx * hidden_size + hidden_offsets,
            hidden_next,
            mask=hidden_mask,
        )
        tl.store(
            output + (batch_idx * seq_len + step_idx) * hidden_size + hidden_offsets,
            hidden_next,
            mask=hidden_mask,
        )


def triton_gru_forward_layer(
    x: torch.Tensor,
    h0: torch.Tensor,
    weight_ih: torch.Tensor,
    weight_hh: torch.Tensor,
    bias_ih: torch.Tensor,
    bias_hh: torch.Tensor,
) -> torch.Tensor:
    if not x.is_cuda:
        raise RuntimeError("triton_gru_forward_layer requires CUDA tensors.")
    if x.dtype != torch.float32:
        raise RuntimeError("triton_gru_forward_layer currently supports fp32 only.")
    if x.dim() != 3:
        raise ValueError("x must have shape [batch, seq, input].")
    if h0.dim() != 2:
        raise ValueError("h0 must have shape [batch, hidden].")

    x = x.contiguous()
    h0 = h0.contiguous()
    weight_ih = weight_ih.contiguous()
    weight_hh = weight_hh.contiguous()
    bias_ih = bias_ih.contiguous()
    bias_hh = bias_hh.contiguous()

    batch_size, seq_len, input_size = x.shape
    hidden_size = h0.size(1)
    if hidden_size > 256:
        raise ValueError("Prototype supports hidden_size <= 256.")

    block_h = triton.next_power_of_2(hidden_size)
    block_k = 32
    output = torch.empty(batch_size, seq_len, hidden_size, device=x.device, dtype=x.dtype)
    hidden_work = torch.empty(batch_size, hidden_size, device=x.device, dtype=x.dtype)
    _gru_forward_time_loop_kernel[(batch_size,)](
        x,
        h0,
        weight_ih,
        weight_hh,
        bias_ih,
        bias_hh,
        output,
        hidden_work,
        seq_len,
        input_size,
        hidden_size,
        block_h,
        block_k,
        num_warps=8,
    )
    return output

