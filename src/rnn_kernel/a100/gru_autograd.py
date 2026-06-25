from __future__ import annotations

import ctypes
import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .gru_forward import (
    _check_cuda_error,
    _get_a100_kernel,
    a100_gru_forward_from_gates_cooperative_h256_cta8_shmem_gate_cache,
    a100_gru_forward_from_gates_cooperative_h256_cta6_shmem_gate_cache,
    a100_gru_forward_from_gates_cooperative_h256_parallel_update_gate_cache,
    a100_gru_forward_from_gates_cooperative_h256_shmem,
    a100_gru_forward_from_gates_cooperative_h256_shmem_gate_cache,
    a100_gru_forward_from_gates_cooperative_h256_shmem_grad_coeff_cache,
    cuda,
)


def _a100_gru_h256_pointwise_backward(
    grad_hidden_next: torch.Tensor,
    input_gates: torch.Tensor,
    hidden_gates: torch.Tensor,
    h0: torch.Tensor,
    output: torch.Tensor,
    grad_input_gates: torch.Tensor,
    step: int,
    block_threads: int = 256,
    grad_hidden_gates_out: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch_size, seq_len, hidden3 = input_gates.shape
    hidden_size = hidden3 // 3
    if hidden_size != 256:
        raise ValueError("A100 h256 pointwise backward requires hidden_size=256.")

    grad_hidden_next = grad_hidden_next.contiguous()
    hidden_gates = hidden_gates.contiguous()
    if grad_hidden_gates_out is None:
        grad_hidden_gates = torch.empty_like(hidden_gates)
    else:
        if grad_hidden_gates_out.shape != hidden_gates.shape:
            raise ValueError("grad_hidden_gates_out shape must match hidden_gates.")
        if not grad_hidden_gates_out.is_contiguous():
            raise ValueError("grad_hidden_gates_out must be contiguous.")
        grad_hidden_gates = grad_hidden_gates_out
    grad_hidden_prev_direct = torch.empty_like(grad_hidden_next)

    compiled = _get_a100_kernel()
    stream = cuda.CUstream(torch.cuda.current_stream(device=input_gates.device).cuda_stream)
    total = batch_size * hidden_size
    grid_blocks = (total + block_threads - 1) // block_threads
    values = (
        ctypes.c_void_p(grad_hidden_next.data_ptr()),
        ctypes.c_void_p(input_gates.data_ptr()),
        ctypes.c_void_p(hidden_gates.data_ptr()),
        ctypes.c_void_p(h0.data_ptr()),
        ctypes.c_void_p(output.data_ptr()),
        ctypes.c_void_p(grad_input_gates.data_ptr()),
        ctypes.c_void_p(grad_hidden_gates.data_ptr()),
        ctypes.c_void_p(grad_hidden_prev_direct.data_ptr()),
        ctypes.c_int(batch_size),
        ctypes.c_int(seq_len),
        ctypes.c_int(step),
    )
    types = (
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
    )
    err, = cuda.cuLaunchKernel(
        compiled.h256_pointwise_backward_function,
        grid_blocks,
        1,
        1,
        block_threads,
        1,
        1,
        0,
        stream,
        (values, types),
        0,
    )
    _check_cuda_error(err)
    return grad_hidden_gates, grad_hidden_prev_direct


def _a100_gru_h256_recurrent_backward(
    grad_hidden_gates: torch.Tensor,
    weight_hh: torch.Tensor,
    grad_hidden_prev_direct: torch.Tensor,
    block_threads: int = 256,
) -> torch.Tensor:
    batch_size, hidden3 = grad_hidden_gates.shape
    hidden_size = hidden3 // 3
    if hidden_size != 256:
        raise ValueError("A100 h256 recurrent backward requires hidden_size=256.")
    if block_threads != 256:
        raise ValueError("A100 h256 recurrent backward currently requires 256 threads.")

    grad_hidden_gates = grad_hidden_gates.contiguous()
    weight_hh = weight_hh.contiguous()
    grad_hidden_prev_direct = grad_hidden_prev_direct.contiguous()
    grad_hidden_prev = torch.empty_like(grad_hidden_prev_direct)

    compiled = _get_a100_kernel()
    stream = cuda.CUstream(torch.cuda.current_stream(device=grad_hidden_gates.device).cuda_stream)
    shared_bytes = 3 * hidden_size * ctypes.sizeof(ctypes.c_float)
    values = (
        ctypes.c_void_p(grad_hidden_gates.data_ptr()),
        ctypes.c_void_p(weight_hh.data_ptr()),
        ctypes.c_void_p(grad_hidden_prev_direct.data_ptr()),
        ctypes.c_void_p(grad_hidden_prev.data_ptr()),
        ctypes.c_int(batch_size),
    )
    types = (
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int,
    )
    err, = cuda.cuLaunchKernel(
        compiled.h256_recurrent_backward_function,
        batch_size,
        1,
        1,
        block_threads,
        1,
        1,
        shared_bytes,
        stream,
        (values, types),
        0,
    )
    _check_cuda_error(err)
    return grad_hidden_prev


def _a100_gru_h256_recurrent_backward_tiled(
    grad_hidden_gates: torch.Tensor,
    weight_hh: torch.Tensor,
    grad_hidden_prev_direct: torch.Tensor,
    block_threads_x: int = 16,
    block_threads_y: int = 16,
) -> torch.Tensor:
    batch_size, hidden3 = grad_hidden_gates.shape
    hidden_size = hidden3 // 3
    if hidden_size != 256:
        raise ValueError("A100 h256 tiled recurrent backward requires hidden_size=256.")
    if block_threads_x != 16 or block_threads_y != 16:
        raise ValueError("A100 h256 tiled recurrent backward requires 16x16 threads.")

    grad_hidden_gates = grad_hidden_gates.contiguous()
    weight_hh = weight_hh.contiguous()
    grad_hidden_prev_direct = grad_hidden_prev_direct.contiguous()
    grad_hidden_prev = torch.empty_like(grad_hidden_prev_direct)

    compiled = _get_a100_kernel()
    stream = cuda.CUstream(torch.cuda.current_stream(device=grad_hidden_gates.device).cuda_stream)
    shared_floats = block_threads_y * 32 + 32 * block_threads_x
    shared_bytes = shared_floats * ctypes.sizeof(ctypes.c_float)
    values = (
        ctypes.c_void_p(grad_hidden_gates.data_ptr()),
        ctypes.c_void_p(weight_hh.data_ptr()),
        ctypes.c_void_p(grad_hidden_prev_direct.data_ptr()),
        ctypes.c_void_p(grad_hidden_prev.data_ptr()),
        ctypes.c_int(batch_size),
    )
    types = (
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int,
    )
    err, = cuda.cuLaunchKernel(
        compiled.h256_recurrent_backward_tiled_function,
        math.ceil(hidden_size / block_threads_x),
        math.ceil(batch_size / block_threads_y),
        1,
        block_threads_x,
        block_threads_y,
        1,
        shared_bytes,
        stream,
        (values, types),
        0,
    )
    _check_cuda_error(err)
    return grad_hidden_prev


def _a100_gru_h256_recurrent_backward_split(
    grad_hidden_gates: torch.Tensor,
    weight_hh: torch.Tensor,
    grad_hidden_prev_direct: torch.Tensor,
    partial_sums: torch.Tensor,
    split_count: int = 8,
    block_threads_x: int = 16,
    block_threads_y: int = 16,
) -> torch.Tensor:
    batch_size, hidden3 = grad_hidden_gates.shape
    hidden_size = hidden3 // 3
    if hidden_size != 256:
        raise ValueError("A100 h256 split recurrent backward requires hidden_size=256.")
    if block_threads_x != 16 or block_threads_y != 16:
        raise ValueError("A100 h256 split recurrent backward requires 16x16 threads.")
    if partial_sums.shape != (split_count, batch_size, hidden_size):
        raise ValueError("partial_sums must have shape [split_count, batch, hidden].")
    if not partial_sums.is_contiguous():
        raise ValueError("partial_sums must be contiguous.")

    grad_hidden_gates = grad_hidden_gates.contiguous()
    weight_hh = weight_hh.contiguous()
    grad_hidden_prev_direct = grad_hidden_prev_direct.contiguous()
    grad_hidden_prev = torch.empty_like(grad_hidden_prev_direct)

    compiled = _get_a100_kernel()
    stream = cuda.CUstream(torch.cuda.current_stream(device=grad_hidden_gates.device).cuda_stream)
    shared_floats = block_threads_y * 32 + 32 * block_threads_x
    shared_bytes = shared_floats * ctypes.sizeof(ctypes.c_float)
    values = (
        ctypes.c_void_p(grad_hidden_gates.data_ptr()),
        ctypes.c_void_p(weight_hh.data_ptr()),
        ctypes.c_void_p(partial_sums.data_ptr()),
        ctypes.c_int(batch_size),
        ctypes.c_int(split_count),
    )
    types = (
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_int,
    )
    err, = cuda.cuLaunchKernel(
        compiled.h256_recurrent_backward_split_function,
        math.ceil(hidden_size / block_threads_x),
        math.ceil(batch_size / block_threads_y),
        split_count,
        block_threads_x,
        block_threads_y,
        1,
        shared_bytes,
        stream,
        (values, types),
        0,
    )
    _check_cuda_error(err)

    reduce_values = (
        ctypes.c_void_p(partial_sums.data_ptr()),
        ctypes.c_void_p(grad_hidden_prev_direct.data_ptr()),
        ctypes.c_void_p(grad_hidden_prev.data_ptr()),
        ctypes.c_int(batch_size),
        ctypes.c_int(split_count),
    )
    reduce_types = (
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_int,
    )
    total = batch_size * hidden_size
    reduce_threads = 256
    reduce_blocks = (total + reduce_threads - 1) // reduce_threads
    err, = cuda.cuLaunchKernel(
        compiled.h256_recurrent_backward_split_reduce_function,
        reduce_blocks,
        1,
        1,
        reduce_threads,
        1,
        1,
        0,
        stream,
        (reduce_values, reduce_types),
        0,
    )
    _check_cuda_error(err)
    return grad_hidden_prev


def _a100_gru_h256_backward_step(
    grad_hidden_next: torch.Tensor,
    input_gates: torch.Tensor,
    hidden_gates: torch.Tensor,
    h0: torch.Tensor,
    output: torch.Tensor,
    weight_hh: torch.Tensor,
    grad_input_gates: torch.Tensor,
    grad_hidden_gates_out: torch.Tensor,
    step: int,
    block_threads: int = 256,
) -> torch.Tensor:
    batch_size, seq_len, hidden3 = input_gates.shape
    hidden_size = hidden3 // 3
    if hidden_size != 256:
        raise ValueError("A100 h256 backward step requires hidden_size=256.")
    if block_threads != 256:
        raise ValueError("A100 h256 backward step currently requires 256 threads.")
    if grad_hidden_gates_out.shape != (batch_size, 3 * hidden_size):
        raise ValueError("grad_hidden_gates_out must have shape [batch, 3 * hidden].")
    if not grad_hidden_gates_out.is_contiguous():
        raise ValueError("grad_hidden_gates_out must be contiguous.")

    grad_hidden_next = grad_hidden_next.contiguous()
    hidden_gates = hidden_gates.contiguous()
    weight_hh = weight_hh.contiguous()
    grad_hidden_prev = torch.empty_like(grad_hidden_next)

    compiled = _get_a100_kernel()
    stream = cuda.CUstream(torch.cuda.current_stream(device=input_gates.device).cuda_stream)
    shared_bytes = 3 * hidden_size * ctypes.sizeof(ctypes.c_float)
    values = (
        ctypes.c_void_p(grad_hidden_next.data_ptr()),
        ctypes.c_void_p(input_gates.data_ptr()),
        ctypes.c_void_p(hidden_gates.data_ptr()),
        ctypes.c_void_p(h0.data_ptr()),
        ctypes.c_void_p(output.data_ptr()),
        ctypes.c_void_p(weight_hh.data_ptr()),
        ctypes.c_void_p(grad_input_gates.data_ptr()),
        ctypes.c_void_p(grad_hidden_gates_out.data_ptr()),
        ctypes.c_void_p(grad_hidden_prev.data_ptr()),
        ctypes.c_int(batch_size),
        ctypes.c_int(seq_len),
        ctypes.c_int(step),
    )
    types = (
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
    )
    err, = cuda.cuLaunchKernel(
        compiled.h256_backward_step_function,
        batch_size,
        1,
        1,
        block_threads,
        1,
        1,
        shared_bytes,
        stream,
        (values, types),
        0,
    )
    _check_cuda_error(err)
    return grad_hidden_prev


def _a100_gru_h256_backward_step_cooperative_split(
    grad_hidden_next: torch.Tensor,
    input_gates: torch.Tensor,
    hidden_gates: torch.Tensor,
    h0: torch.Tensor,
    output: torch.Tensor,
    weight_hh: torch.Tensor,
    grad_input_gates: torch.Tensor,
    grad_hidden_gates_out: torch.Tensor,
    partial_sums: torch.Tensor,
    step: int,
    split_count: int = 4,
    block_threads: int = 256,
) -> torch.Tensor:
    batch_size, seq_len, hidden3 = input_gates.shape
    hidden_size = hidden3 // 3
    if hidden_size != 256:
        raise ValueError("A100 h256 cooperative split backward step requires hidden_size=256.")
    if block_threads != 256:
        raise ValueError("A100 h256 cooperative split backward step requires 256 threads.")
    if grad_hidden_gates_out.shape != (batch_size, 3 * hidden_size):
        raise ValueError("grad_hidden_gates_out must have shape [batch, 3 * hidden].")
    if partial_sums.shape != (batch_size, split_count, hidden_size):
        raise ValueError("partial_sums must have shape [batch, split_count, hidden].")
    if not grad_hidden_gates_out.is_contiguous() or not partial_sums.is_contiguous():
        raise ValueError("grad_hidden_gates_out and partial_sums must be contiguous.")

    grad_hidden_next = grad_hidden_next.contiguous()
    hidden_gates = hidden_gates.contiguous()
    weight_hh = weight_hh.contiguous()
    grad_hidden_prev = torch.empty_like(grad_hidden_next)

    compiled = _get_a100_kernel()
    stream = cuda.CUstream(torch.cuda.current_stream(device=input_gates.device).cuda_stream)
    shared_bytes = 3 * hidden_size * ctypes.sizeof(ctypes.c_float)
    values = (
        ctypes.c_void_p(grad_hidden_next.data_ptr()),
        ctypes.c_void_p(input_gates.data_ptr()),
        ctypes.c_void_p(hidden_gates.data_ptr()),
        ctypes.c_void_p(h0.data_ptr()),
        ctypes.c_void_p(output.data_ptr()),
        ctypes.c_void_p(weight_hh.data_ptr()),
        ctypes.c_void_p(grad_input_gates.data_ptr()),
        ctypes.c_void_p(grad_hidden_gates_out.data_ptr()),
        ctypes.c_void_p(partial_sums.data_ptr()),
        ctypes.c_void_p(grad_hidden_prev.data_ptr()),
        ctypes.c_int(batch_size),
        ctypes.c_int(seq_len),
        ctypes.c_int(step),
        ctypes.c_int(split_count),
    )
    types = (
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
    )
    err, = cuda.cuLaunchCooperativeKernel(
        compiled.h256_backward_step_cooperative_split_function,
        batch_size * split_count,
        1,
        1,
        block_threads,
        1,
        1,
        shared_bytes,
        stream,
        (values, types),
    )
    _check_cuda_error(err)
    return grad_hidden_prev


def _a100_gru_h256_backward_step_cooperative_split_cached(
    grad_hidden_next: torch.Tensor,
    input_gates: torch.Tensor,
    hidden_gates: torch.Tensor,
    h0: torch.Tensor,
    output: torch.Tensor,
    weight_hh: torch.Tensor,
    grad_input_gates: torch.Tensor,
    grad_hidden_gates_out: torch.Tensor,
    partial_sums: torch.Tensor,
    step: int,
    split_count: int = 2,
    block_threads: int = 256,
) -> torch.Tensor:
    batch_size, seq_len, hidden3 = input_gates.shape
    hidden_size = hidden3 // 3
    if hidden_size != 256:
        raise ValueError("A100 h256 cooperative cached split backward step requires hidden_size=256.")
    if block_threads != 256:
        raise ValueError("A100 h256 cooperative cached split backward step requires 256 threads.")
    if grad_hidden_gates_out.shape != (batch_size, 3 * hidden_size):
        raise ValueError("grad_hidden_gates_out must have shape [batch, 3 * hidden].")
    if partial_sums.shape != (batch_size, split_count, hidden_size):
        raise ValueError("partial_sums must have shape [batch, split_count, hidden].")
    if not grad_hidden_gates_out.is_contiguous() or not partial_sums.is_contiguous():
        raise ValueError("grad_hidden_gates_out and partial_sums must be contiguous.")

    grad_hidden_next = grad_hidden_next.contiguous()
    hidden_gates = hidden_gates.contiguous()
    weight_hh = weight_hh.contiguous()
    grad_hidden_prev = torch.empty_like(grad_hidden_next)

    compiled = _get_a100_kernel()
    stream = cuda.CUstream(torch.cuda.current_stream(device=input_gates.device).cuda_stream)
    shared_bytes = 3 * hidden_size * ctypes.sizeof(ctypes.c_float)
    values = (
        ctypes.c_void_p(grad_hidden_next.data_ptr()),
        ctypes.c_void_p(input_gates.data_ptr()),
        ctypes.c_void_p(hidden_gates.data_ptr()),
        ctypes.c_void_p(h0.data_ptr()),
        ctypes.c_void_p(output.data_ptr()),
        ctypes.c_void_p(weight_hh.data_ptr()),
        ctypes.c_void_p(grad_input_gates.data_ptr()),
        ctypes.c_void_p(grad_hidden_gates_out.data_ptr()),
        ctypes.c_void_p(partial_sums.data_ptr()),
        ctypes.c_void_p(grad_hidden_prev.data_ptr()),
        ctypes.c_int(batch_size),
        ctypes.c_int(seq_len),
        ctypes.c_int(step),
        ctypes.c_int(split_count),
    )
    types = (
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
    )
    err, = cuda.cuLaunchCooperativeKernel(
        compiled.h256_backward_step_cooperative_split_cached_function,
        batch_size * split_count,
        1,
        1,
        block_threads,
        1,
        1,
        shared_bytes,
        stream,
        (values, types),
    )
    _check_cuda_error(err)
    return grad_hidden_prev


def _a100_gru_h256_backward_step_cooperative_split2(
    grad_hidden_next: torch.Tensor,
    input_gates: torch.Tensor,
    hidden_gates: torch.Tensor,
    h0: torch.Tensor,
    output: torch.Tensor,
    weight_hh: torch.Tensor,
    grad_input_gates: torch.Tensor,
    grad_hidden_gates_out: torch.Tensor,
    partial_sums: torch.Tensor,
    step: int,
    block_threads: int = 256,
) -> torch.Tensor:
    batch_size, seq_len, hidden3 = input_gates.shape
    hidden_size = hidden3 // 3
    if hidden_size != 256:
        raise ValueError("A100 h256 cooperative split2 backward step requires hidden_size=256.")
    if block_threads != 256:
        raise ValueError("A100 h256 cooperative split2 backward step requires 256 threads.")
    if grad_hidden_gates_out.shape != (batch_size, 3 * hidden_size):
        raise ValueError("grad_hidden_gates_out must have shape [batch, 3 * hidden].")
    if partial_sums.shape != (batch_size, 2, hidden_size):
        raise ValueError("partial_sums must have shape [batch, 2, hidden].")
    if not grad_hidden_gates_out.is_contiguous() or not partial_sums.is_contiguous():
        raise ValueError("grad_hidden_gates_out and partial_sums must be contiguous.")

    grad_hidden_next = grad_hidden_next.contiguous()
    hidden_gates = hidden_gates.contiguous()
    weight_hh = weight_hh.contiguous()
    grad_hidden_prev = torch.empty_like(grad_hidden_next)

    compiled = _get_a100_kernel()
    stream = cuda.CUstream(torch.cuda.current_stream(device=input_gates.device).cuda_stream)
    shared_bytes = 3 * hidden_size * ctypes.sizeof(ctypes.c_float)
    values = (
        ctypes.c_void_p(grad_hidden_next.data_ptr()),
        ctypes.c_void_p(input_gates.data_ptr()),
        ctypes.c_void_p(hidden_gates.data_ptr()),
        ctypes.c_void_p(h0.data_ptr()),
        ctypes.c_void_p(output.data_ptr()),
        ctypes.c_void_p(weight_hh.data_ptr()),
        ctypes.c_void_p(grad_input_gates.data_ptr()),
        ctypes.c_void_p(grad_hidden_gates_out.data_ptr()),
        ctypes.c_void_p(partial_sums.data_ptr()),
        ctypes.c_void_p(grad_hidden_prev.data_ptr()),
        ctypes.c_int(batch_size),
        ctypes.c_int(seq_len),
        ctypes.c_int(step),
    )
    types = (
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
    )
    err, = cuda.cuLaunchCooperativeKernel(
        compiled.h256_backward_step_cooperative_split2_function,
        batch_size * 2,
        1,
        1,
        block_threads,
        1,
        1,
        shared_bytes,
        stream,
        (values, types),
    )
    _check_cuda_error(err)
    return grad_hidden_prev


def _a100_gru_h256_backward_step_cooperative_split2_cached_local(
    grad_hidden_next: torch.Tensor,
    input_gates: torch.Tensor,
    hidden_gates: torch.Tensor,
    h0: torch.Tensor,
    output: torch.Tensor,
    weight_hh: torch.Tensor,
    grad_input_gates: torch.Tensor,
    grad_hidden_gates_out: torch.Tensor,
    partial_sums: torch.Tensor,
    step: int,
    block_threads: int = 256,
) -> torch.Tensor:
    batch_size, seq_len, hidden3 = input_gates.shape
    hidden_size = hidden3 // 3
    if hidden_size != 256:
        raise ValueError("A100 h256 cooperative split2 cached local requires hidden_size=256.")
    if block_threads != 256:
        raise ValueError("A100 h256 cooperative split2 cached local requires 256 threads.")
    if grad_hidden_gates_out.shape != (batch_size, 3 * hidden_size):
        raise ValueError("grad_hidden_gates_out must have shape [batch, 3 * hidden].")
    if partial_sums.shape != (batch_size, 2, hidden_size):
        raise ValueError("partial_sums must have shape [batch, 2, hidden].")
    if not grad_hidden_gates_out.is_contiguous() or not partial_sums.is_contiguous():
        raise ValueError("grad_hidden_gates_out and partial_sums must be contiguous.")

    grad_hidden_next = grad_hidden_next.contiguous()
    hidden_gates = hidden_gates.contiguous()
    weight_hh = weight_hh.contiguous()
    grad_hidden_prev = torch.empty_like(grad_hidden_next)

    compiled = _get_a100_kernel()
    stream = cuda.CUstream(torch.cuda.current_stream(device=input_gates.device).cuda_stream)
    shared_bytes = 3 * hidden_size * ctypes.sizeof(ctypes.c_float)
    values = (
        ctypes.c_void_p(grad_hidden_next.data_ptr()),
        ctypes.c_void_p(input_gates.data_ptr()),
        ctypes.c_void_p(hidden_gates.data_ptr()),
        ctypes.c_void_p(h0.data_ptr()),
        ctypes.c_void_p(output.data_ptr()),
        ctypes.c_void_p(weight_hh.data_ptr()),
        ctypes.c_void_p(grad_input_gates.data_ptr()),
        ctypes.c_void_p(grad_hidden_gates_out.data_ptr()),
        ctypes.c_void_p(partial_sums.data_ptr()),
        ctypes.c_void_p(grad_hidden_prev.data_ptr()),
        ctypes.c_int(batch_size),
        ctypes.c_int(seq_len),
        ctypes.c_int(step),
    )
    types = (
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
    )
    err, = cuda.cuLaunchCooperativeKernel(
        compiled.h256_backward_step_cooperative_split2_cached_local_function,
        batch_size * 2,
        1,
        1,
        block_threads,
        1,
        1,
        shared_bytes,
        stream,
        (values, types),
    )
    _check_cuda_error(err)
    return grad_hidden_prev


def _a100_gru_h256_backward_step_cooperative_split2_gate_cache(
    grad_hidden_next: torch.Tensor,
    gate_cache: torch.Tensor,
    h0: torch.Tensor,
    output: torch.Tensor,
    weight_hh: torch.Tensor,
    grad_input_gates: torch.Tensor,
    grad_hidden_gates_out: torch.Tensor,
    partial_sums: torch.Tensor,
    step: int,
    block_threads: int = 256,
) -> torch.Tensor:
    batch_size, seq_len, cache_size = gate_cache.shape
    hidden_size = cache_size // 4
    if hidden_size != 256:
        raise ValueError("A100 h256 cooperative split2 gate cache requires hidden_size=256.")
    if block_threads != 256:
        raise ValueError("A100 h256 cooperative split2 gate cache requires 256 threads.")
    if grad_hidden_gates_out.shape != (batch_size, 3 * hidden_size):
        raise ValueError("grad_hidden_gates_out must have shape [batch, 3 * hidden].")
    if partial_sums.shape != (batch_size, 2, hidden_size):
        raise ValueError("partial_sums must have shape [batch, 2, hidden].")
    if not grad_hidden_gates_out.is_contiguous() or not partial_sums.is_contiguous():
        raise ValueError("grad_hidden_gates_out and partial_sums must be contiguous.")

    grad_hidden_next = grad_hidden_next.contiguous()
    gate_cache = gate_cache.contiguous()
    weight_hh = weight_hh.contiguous()
    grad_hidden_prev = torch.empty_like(grad_hidden_next)

    compiled = _get_a100_kernel()
    stream = cuda.CUstream(torch.cuda.current_stream(device=gate_cache.device).cuda_stream)
    shared_bytes = 3 * hidden_size * ctypes.sizeof(ctypes.c_float)
    values = (
        ctypes.c_void_p(grad_hidden_next.data_ptr()),
        ctypes.c_void_p(gate_cache.data_ptr()),
        ctypes.c_void_p(h0.data_ptr()),
        ctypes.c_void_p(output.data_ptr()),
        ctypes.c_void_p(weight_hh.data_ptr()),
        ctypes.c_void_p(grad_input_gates.data_ptr()),
        ctypes.c_void_p(grad_hidden_gates_out.data_ptr()),
        ctypes.c_void_p(partial_sums.data_ptr()),
        ctypes.c_void_p(grad_hidden_prev.data_ptr()),
        ctypes.c_int(batch_size),
        ctypes.c_int(seq_len),
        ctypes.c_int(step),
    )
    types = (
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
    )
    err, = cuda.cuLaunchCooperativeKernel(
        compiled.h256_backward_step_cooperative_split2_gate_cache_function,
        batch_size * 2,
        1,
        1,
        block_threads,
        1,
        1,
        shared_bytes,
        stream,
        (values, types),
    )
    _check_cuda_error(err)
    return grad_hidden_prev


def _a100_gru_h256_backward_sequence_cooperative_split2_cached_local(
    grad_output: torch.Tensor,
    grad_hidden_state: torch.Tensor,
    input_gates: torch.Tensor,
    hidden_gates_steps: torch.Tensor,
    h0: torch.Tensor,
    output: torch.Tensor,
    weight_hh: torch.Tensor,
    grad_input_gates: torch.Tensor,
    grad_hidden_gates_steps: torch.Tensor,
    partial_sums: torch.Tensor,
    block_threads: int = 256,
) -> torch.Tensor:
    batch_size, seq_len, hidden3 = input_gates.shape
    hidden_size = hidden3 // 3
    if hidden_size != 256:
        raise ValueError("A100 h256 persistent backward requires hidden_size=256.")
    if block_threads != 256:
        raise ValueError("A100 h256 persistent backward requires 256 threads.")
    if hidden_gates_steps.shape != (seq_len, batch_size, 3 * hidden_size):
        raise ValueError("hidden_gates_steps must have shape [seq, batch, 3 * hidden].")
    if grad_hidden_gates_steps.shape != (seq_len, batch_size, 3 * hidden_size):
        raise ValueError("grad_hidden_gates_steps must have shape [seq, batch, 3 * hidden].")
    if partial_sums.shape != (batch_size, 2, hidden_size):
        raise ValueError("partial_sums must have shape [batch, 2, hidden].")
    if not hidden_gates_steps.is_contiguous():
        raise ValueError("hidden_gates_steps must be contiguous.")
    if not grad_hidden_gates_steps.is_contiguous() or not partial_sums.is_contiguous():
        raise ValueError("grad_hidden_gates_steps and partial_sums must be contiguous.")

    grad_output = grad_output.contiguous()
    grad_hidden_state = grad_hidden_state.contiguous()
    input_gates = input_gates.contiguous()
    weight_hh = weight_hh.contiguous()

    compiled = _get_a100_kernel()
    stream = cuda.CUstream(torch.cuda.current_stream(device=input_gates.device).cuda_stream)
    shared_bytes = 3 * hidden_size * ctypes.sizeof(ctypes.c_float)
    values = (
        ctypes.c_void_p(grad_output.data_ptr()),
        ctypes.c_void_p(input_gates.data_ptr()),
        ctypes.c_void_p(hidden_gates_steps.data_ptr()),
        ctypes.c_void_p(h0.data_ptr()),
        ctypes.c_void_p(output.data_ptr()),
        ctypes.c_void_p(weight_hh.data_ptr()),
        ctypes.c_void_p(grad_input_gates.data_ptr()),
        ctypes.c_void_p(grad_hidden_gates_steps.data_ptr()),
        ctypes.c_void_p(partial_sums.data_ptr()),
        ctypes.c_void_p(grad_hidden_state.data_ptr()),
        ctypes.c_int(batch_size),
        ctypes.c_int(seq_len),
    )
    types = (
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_int,
    )
    err, = cuda.cuLaunchCooperativeKernel(
        compiled.h256_backward_sequence_cooperative_split2_cached_local_function,
        batch_size * 2,
        1,
        1,
        block_threads,
        1,
        1,
        shared_bytes,
        stream,
        (values, types),
    )
    _check_cuda_error(err)
    return grad_hidden_state


def _a100_gru_h256_backward_sequence_cooperative_split2_state_parts(
    grad_output: torch.Tensor,
    grad_hidden_state: torch.Tensor,
    input_gates: torch.Tensor,
    hidden_gates_steps: torch.Tensor,
    h0: torch.Tensor,
    output: torch.Tensor,
    weight_hh: torch.Tensor,
    grad_input_gates: torch.Tensor,
    grad_hidden_gates_steps: torch.Tensor,
    partial_sums: torch.Tensor,
    block_threads: int = 256,
) -> torch.Tensor:
    batch_size, seq_len, hidden3 = input_gates.shape
    hidden_size = hidden3 // 3
    if hidden_size != 256:
        raise ValueError("A100 h256 persistent state backward requires hidden_size=256.")
    if block_threads != 256:
        raise ValueError("A100 h256 persistent state backward requires 256 threads.")
    if hidden_gates_steps.shape != (seq_len, batch_size, 3 * hidden_size):
        raise ValueError("hidden_gates_steps must have shape [seq, batch, 3 * hidden].")
    if grad_hidden_gates_steps.shape != (seq_len, batch_size, 3 * hidden_size):
        raise ValueError("grad_hidden_gates_steps must have shape [seq, batch, 3 * hidden].")
    if partial_sums.shape != (batch_size, 2, hidden_size):
        raise ValueError("partial_sums must have shape [batch, 2, hidden].")
    if not hidden_gates_steps.is_contiguous():
        raise ValueError("hidden_gates_steps must be contiguous.")
    if not grad_hidden_gates_steps.is_contiguous() or not partial_sums.is_contiguous():
        raise ValueError("grad_hidden_gates_steps and partial_sums must be contiguous.")

    grad_output = grad_output.contiguous()
    grad_hidden_state = grad_hidden_state.contiguous()
    input_gates = input_gates.contiguous()
    weight_hh = weight_hh.contiguous()

    compiled = _get_a100_kernel()
    stream = cuda.CUstream(torch.cuda.current_stream(device=input_gates.device).cuda_stream)
    shared_bytes = 3 * hidden_size * ctypes.sizeof(ctypes.c_float)
    values = (
        ctypes.c_void_p(grad_output.data_ptr()),
        ctypes.c_void_p(input_gates.data_ptr()),
        ctypes.c_void_p(hidden_gates_steps.data_ptr()),
        ctypes.c_void_p(h0.data_ptr()),
        ctypes.c_void_p(output.data_ptr()),
        ctypes.c_void_p(weight_hh.data_ptr()),
        ctypes.c_void_p(grad_input_gates.data_ptr()),
        ctypes.c_void_p(grad_hidden_gates_steps.data_ptr()),
        ctypes.c_void_p(partial_sums.data_ptr()),
        ctypes.c_void_p(grad_hidden_state.data_ptr()),
        ctypes.c_int(batch_size),
        ctypes.c_int(seq_len),
    )
    types = (
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_int,
    )
    err, = cuda.cuLaunchCooperativeKernel(
        compiled.h256_backward_sequence_cooperative_split2_state_parts_function,
        batch_size * 2,
        1,
        1,
        block_threads,
        1,
        1,
        shared_bytes,
        stream,
        (values, types),
    )
    _check_cuda_error(err)
    return grad_hidden_state


def _a100_gru_h256_backward_sequence_cooperative_split2_state_local(
    grad_output: torch.Tensor,
    grad_hidden_state: torch.Tensor,
    input_gates: torch.Tensor,
    hidden_gates_steps: torch.Tensor,
    h0: torch.Tensor,
    output: torch.Tensor,
    weight_hh: torch.Tensor,
    grad_input_gates: torch.Tensor,
    grad_hidden_gates_steps: torch.Tensor,
    partial_sums: torch.Tensor,
    block_threads: int = 256,
) -> torch.Tensor:
    batch_size, seq_len, hidden3 = input_gates.shape
    hidden_size = hidden3 // 3
    if hidden_size != 256:
        raise ValueError("A100 h256 persistent state-local backward requires hidden_size=256.")
    if block_threads != 256:
        raise ValueError("A100 h256 persistent state-local backward requires 256 threads.")
    if hidden_gates_steps.shape != (seq_len, batch_size, 3 * hidden_size):
        raise ValueError("hidden_gates_steps must have shape [seq, batch, 3 * hidden].")
    if grad_hidden_gates_steps.shape != (seq_len, batch_size, 3 * hidden_size):
        raise ValueError("grad_hidden_gates_steps must have shape [seq, batch, 3 * hidden].")
    if partial_sums.shape != (batch_size, 2, hidden_size):
        raise ValueError("partial_sums must have shape [batch, 2, hidden].")
    if not hidden_gates_steps.is_contiguous():
        raise ValueError("hidden_gates_steps must be contiguous.")
    if not grad_hidden_gates_steps.is_contiguous() or not partial_sums.is_contiguous():
        raise ValueError("grad_hidden_gates_steps and partial_sums must be contiguous.")

    grad_output = grad_output.contiguous()
    grad_hidden_state = grad_hidden_state.contiguous()
    input_gates = input_gates.contiguous()
    weight_hh = weight_hh.contiguous()

    compiled = _get_a100_kernel()
    stream = cuda.CUstream(torch.cuda.current_stream(device=input_gates.device).cuda_stream)
    shared_bytes = 3 * hidden_size * ctypes.sizeof(ctypes.c_float)
    values = (
        ctypes.c_void_p(grad_output.data_ptr()),
        ctypes.c_void_p(input_gates.data_ptr()),
        ctypes.c_void_p(hidden_gates_steps.data_ptr()),
        ctypes.c_void_p(h0.data_ptr()),
        ctypes.c_void_p(output.data_ptr()),
        ctypes.c_void_p(weight_hh.data_ptr()),
        ctypes.c_void_p(grad_input_gates.data_ptr()),
        ctypes.c_void_p(grad_hidden_gates_steps.data_ptr()),
        ctypes.c_void_p(partial_sums.data_ptr()),
        ctypes.c_void_p(grad_hidden_state.data_ptr()),
        ctypes.c_int(batch_size),
        ctypes.c_int(seq_len),
    )
    types = (
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_int,
    )
    err, = cuda.cuLaunchCooperativeKernel(
        compiled.h256_backward_sequence_cooperative_split2_state_local_function,
        batch_size * 2,
        1,
        1,
        block_threads,
        1,
        1,
        shared_bytes,
        stream,
        (values, types),
    )
    _check_cuda_error(err)
    return grad_hidden_state


def _a100_gru_h256_backward_sequence_cooperative_split4_state(
    grad_output: torch.Tensor,
    grad_hidden_state: torch.Tensor,
    input_gates: torch.Tensor,
    hidden_gates_steps: torch.Tensor,
    h0: torch.Tensor,
    output: torch.Tensor,
    weight_hh: torch.Tensor,
    grad_input_gates: torch.Tensor,
    grad_hidden_gates_steps: torch.Tensor,
    partial_sums: torch.Tensor,
    block_threads: int = 256,
) -> torch.Tensor:
    batch_size, seq_len, hidden3 = input_gates.shape
    hidden_size = hidden3 // 3
    if hidden_size != 256:
        raise ValueError("A100 h256 persistent split4 backward requires hidden_size=256.")
    if block_threads != 256:
        raise ValueError("A100 h256 persistent split4 backward requires 256 threads.")
    if hidden_gates_steps.shape != (seq_len, batch_size, 3 * hidden_size):
        raise ValueError("hidden_gates_steps must have shape [seq, batch, 3 * hidden].")
    if grad_hidden_gates_steps.shape != (seq_len, batch_size, 3 * hidden_size):
        raise ValueError("grad_hidden_gates_steps must have shape [seq, batch, 3 * hidden].")
    if partial_sums.shape != (batch_size, 4, hidden_size):
        raise ValueError("partial_sums must have shape [batch, 4, hidden].")
    if not hidden_gates_steps.is_contiguous():
        raise ValueError("hidden_gates_steps must be contiguous.")
    if not grad_hidden_gates_steps.is_contiguous() or not partial_sums.is_contiguous():
        raise ValueError("grad_hidden_gates_steps and partial_sums must be contiguous.")

    grad_output = grad_output.contiguous()
    grad_hidden_state = grad_hidden_state.contiguous()
    input_gates = input_gates.contiguous()
    weight_hh = weight_hh.contiguous()

    compiled = _get_a100_kernel()
    stream = cuda.CUstream(torch.cuda.current_stream(device=input_gates.device).cuda_stream)
    shared_bytes = 3 * hidden_size * ctypes.sizeof(ctypes.c_float)
    values = (
        ctypes.c_void_p(grad_output.data_ptr()),
        ctypes.c_void_p(input_gates.data_ptr()),
        ctypes.c_void_p(hidden_gates_steps.data_ptr()),
        ctypes.c_void_p(h0.data_ptr()),
        ctypes.c_void_p(output.data_ptr()),
        ctypes.c_void_p(weight_hh.data_ptr()),
        ctypes.c_void_p(grad_input_gates.data_ptr()),
        ctypes.c_void_p(grad_hidden_gates_steps.data_ptr()),
        ctypes.c_void_p(partial_sums.data_ptr()),
        ctypes.c_void_p(grad_hidden_state.data_ptr()),
        ctypes.c_int(batch_size),
        ctypes.c_int(seq_len),
    )
    types = (
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_int,
    )
    err, = cuda.cuLaunchCooperativeKernel(
        compiled.h256_backward_sequence_cooperative_split4_state_function,
        batch_size * 4,
        1,
        1,
        block_threads,
        1,
        1,
        shared_bytes,
        stream,
        (values, types),
    )
    _check_cuda_error(err)
    return grad_hidden_state


def _a100_gru_h256_backward_sequence_cooperative_split8_state(
    grad_output: torch.Tensor,
    grad_hidden_state: torch.Tensor,
    input_gates: torch.Tensor,
    hidden_gates_steps: torch.Tensor,
    h0: torch.Tensor,
    output: torch.Tensor,
    weight_hh: torch.Tensor,
    grad_input_gates: torch.Tensor,
    grad_hidden_gates_steps: torch.Tensor,
    partial_sums: torch.Tensor,
    block_threads: int = 256,
) -> torch.Tensor:
    batch_size, seq_len, hidden3 = input_gates.shape
    hidden_size = hidden3 // 3
    if hidden_size != 256:
        raise ValueError("A100 h256 persistent split8 backward requires hidden_size=256.")
    if block_threads != 256:
        raise ValueError("A100 h256 persistent split8 backward requires 256 threads.")
    if hidden_gates_steps.shape != (seq_len, batch_size, 3 * hidden_size):
        raise ValueError("hidden_gates_steps must have shape [seq, batch, 3 * hidden].")
    if grad_hidden_gates_steps.shape != (seq_len, batch_size, 3 * hidden_size):
        raise ValueError("grad_hidden_gates_steps must have shape [seq, batch, 3 * hidden].")
    if partial_sums.shape != (batch_size, 8, hidden_size):
        raise ValueError("partial_sums must have shape [batch, 8, hidden].")
    if not hidden_gates_steps.is_contiguous():
        raise ValueError("hidden_gates_steps must be contiguous.")
    if not grad_hidden_gates_steps.is_contiguous() or not partial_sums.is_contiguous():
        raise ValueError("grad_hidden_gates_steps and partial_sums must be contiguous.")

    grad_output = grad_output.contiguous()
    grad_hidden_state = grad_hidden_state.contiguous()
    input_gates = input_gates.contiguous()
    weight_hh = weight_hh.contiguous()

    compiled = _get_a100_kernel()
    stream = cuda.CUstream(torch.cuda.current_stream(device=input_gates.device).cuda_stream)
    shared_bytes = 3 * hidden_size * ctypes.sizeof(ctypes.c_float)
    values = (
        ctypes.c_void_p(grad_output.data_ptr()),
        ctypes.c_void_p(input_gates.data_ptr()),
        ctypes.c_void_p(hidden_gates_steps.data_ptr()),
        ctypes.c_void_p(h0.data_ptr()),
        ctypes.c_void_p(output.data_ptr()),
        ctypes.c_void_p(weight_hh.data_ptr()),
        ctypes.c_void_p(grad_input_gates.data_ptr()),
        ctypes.c_void_p(grad_hidden_gates_steps.data_ptr()),
        ctypes.c_void_p(partial_sums.data_ptr()),
        ctypes.c_void_p(grad_hidden_state.data_ptr()),
        ctypes.c_int(batch_size),
        ctypes.c_int(seq_len),
    )
    types = (
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_int,
    )
    err, = cuda.cuLaunchCooperativeKernel(
        compiled.h256_backward_sequence_cooperative_split8_state_function,
        batch_size * 8,
        1,
        1,
        block_threads,
        1,
        1,
        shared_bytes,
        stream,
        (values, types),
    )
    _check_cuda_error(err)
    return grad_hidden_state


def _a100_gru_h256_backward_sequence_cooperative_split16_state(
    grad_output: torch.Tensor,
    grad_hidden_state: torch.Tensor,
    input_gates: torch.Tensor,
    hidden_gates_steps: torch.Tensor,
    h0: torch.Tensor,
    output: torch.Tensor,
    weight_hh: torch.Tensor,
    grad_input_gates: torch.Tensor,
    grad_hidden_gates_steps: torch.Tensor,
    partial_sums: torch.Tensor,
    block_threads: int = 256,
) -> torch.Tensor:
    batch_size, seq_len, hidden3 = input_gates.shape
    hidden_size = hidden3 // 3
    if hidden_size != 256:
        raise ValueError("A100 h256 persistent split16 backward requires hidden_size=256.")
    if block_threads != 256:
        raise ValueError("A100 h256 persistent split16 backward requires 256 threads.")
    if hidden_gates_steps.shape != (seq_len, batch_size, 3 * hidden_size):
        raise ValueError("hidden_gates_steps must have shape [seq, batch, 3 * hidden].")
    if grad_hidden_gates_steps.shape != (seq_len, batch_size, 3 * hidden_size):
        raise ValueError("grad_hidden_gates_steps must have shape [seq, batch, 3 * hidden].")
    if partial_sums.shape != (batch_size, 16, hidden_size):
        raise ValueError("partial_sums must have shape [batch, 16, hidden].")
    if not hidden_gates_steps.is_contiguous():
        raise ValueError("hidden_gates_steps must be contiguous.")
    if not grad_hidden_gates_steps.is_contiguous() or not partial_sums.is_contiguous():
        raise ValueError("grad_hidden_gates_steps and partial_sums must be contiguous.")

    grad_output = grad_output.contiguous()
    grad_hidden_state = grad_hidden_state.contiguous()
    input_gates = input_gates.contiguous()
    weight_hh = weight_hh.contiguous()

    compiled = _get_a100_kernel()
    stream = cuda.CUstream(torch.cuda.current_stream(device=input_gates.device).cuda_stream)
    shared_bytes = 3 * hidden_size * ctypes.sizeof(ctypes.c_float)
    values = (
        ctypes.c_void_p(grad_output.data_ptr()),
        ctypes.c_void_p(input_gates.data_ptr()),
        ctypes.c_void_p(hidden_gates_steps.data_ptr()),
        ctypes.c_void_p(h0.data_ptr()),
        ctypes.c_void_p(output.data_ptr()),
        ctypes.c_void_p(weight_hh.data_ptr()),
        ctypes.c_void_p(grad_input_gates.data_ptr()),
        ctypes.c_void_p(grad_hidden_gates_steps.data_ptr()),
        ctypes.c_void_p(partial_sums.data_ptr()),
        ctypes.c_void_p(grad_hidden_state.data_ptr()),
        ctypes.c_int(batch_size),
        ctypes.c_int(seq_len),
    )
    types = (
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_int,
    )
    err, = cuda.cuLaunchCooperativeKernel(
        compiled.h256_backward_sequence_cooperative_split16_state_function,
        batch_size * 16,
        1,
        1,
        block_threads,
        1,
        1,
        shared_bytes,
        stream,
        (values, types),
    )
    _check_cuda_error(err)
    return grad_hidden_state


def _a100_gru_h256_backward_sequence_cooperative_split16_gate_cache_state(
    grad_output: torch.Tensor,
    grad_hidden_state: torch.Tensor,
    gate_cache: torch.Tensor,
    h0: torch.Tensor,
    output: torch.Tensor,
    weight_hh: torch.Tensor,
    grad_input_gates: torch.Tensor,
    grad_hidden_gates_steps: torch.Tensor,
    partial_sums: torch.Tensor,
    block_threads: int = 256,
) -> torch.Tensor:
    batch_size, seq_len, cache_size = gate_cache.shape
    hidden_size = cache_size // 4
    if hidden_size != 256:
        raise ValueError(
            "A100 h256 persistent split16 gate-cache backward requires hidden_size=256."
        )
    if block_threads != 256:
        raise ValueError(
            "A100 h256 persistent split16 gate-cache backward requires 256 threads."
        )
    if grad_hidden_gates_steps.shape != (seq_len, batch_size, 3 * hidden_size):
        raise ValueError("grad_hidden_gates_steps must have shape [seq, batch, 3 * hidden].")
    if partial_sums.shape != (batch_size, 16, hidden_size):
        raise ValueError("partial_sums must have shape [batch, 16, hidden].")
    if not gate_cache.is_contiguous():
        raise ValueError("gate_cache must be contiguous.")
    if not grad_hidden_gates_steps.is_contiguous() or not partial_sums.is_contiguous():
        raise ValueError("grad_hidden_gates_steps and partial_sums must be contiguous.")

    grad_output = grad_output.contiguous()
    grad_hidden_state = grad_hidden_state.contiguous()
    gate_cache = gate_cache.contiguous()
    weight_hh = weight_hh.contiguous()

    compiled = _get_a100_kernel()
    stream = cuda.CUstream(torch.cuda.current_stream(device=gate_cache.device).cuda_stream)
    shared_bytes = 3 * hidden_size * ctypes.sizeof(ctypes.c_float)
    values = (
        ctypes.c_void_p(grad_output.data_ptr()),
        ctypes.c_void_p(gate_cache.data_ptr()),
        ctypes.c_void_p(h0.data_ptr()),
        ctypes.c_void_p(output.data_ptr()),
        ctypes.c_void_p(weight_hh.data_ptr()),
        ctypes.c_void_p(grad_input_gates.data_ptr()),
        ctypes.c_void_p(grad_hidden_gates_steps.data_ptr()),
        ctypes.c_void_p(partial_sums.data_ptr()),
        ctypes.c_void_p(grad_hidden_state.data_ptr()),
        ctypes.c_int(batch_size),
        ctypes.c_int(seq_len),
    )
    types = (
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_int,
    )
    err, = cuda.cuLaunchCooperativeKernel(
        compiled.h256_backward_sequence_cooperative_split16_gate_cache_state_function,
        batch_size * 16,
        1,
        1,
        block_threads,
        1,
        1,
        shared_bytes,
        stream,
        (values, types),
    )
    _check_cuda_error(err)
    return grad_hidden_state


def _a100_gru_h256_backward_sequence_cooperative_split16_gate_cache_state_tiled(
    grad_output: torch.Tensor,
    grad_hidden_state: torch.Tensor,
    gate_cache: torch.Tensor,
    h0: torch.Tensor,
    output: torch.Tensor,
    weight_hh: torch.Tensor,
    grad_input_gates: torch.Tensor,
    grad_hidden_gates_steps: torch.Tensor,
    partial_sums: torch.Tensor,
    block_threads: int = 256,
) -> torch.Tensor:
    batch_size, seq_len, cache_size = gate_cache.shape
    hidden_size = cache_size // 4
    if hidden_size != 256:
        raise ValueError(
            "A100 h256 persistent split16 tiled gate-cache backward requires hidden_size=256."
        )
    if block_threads != 256:
        raise ValueError(
            "A100 h256 persistent split16 tiled gate-cache backward requires 256 threads."
        )
    if grad_hidden_gates_steps.shape != (seq_len, batch_size, 3 * hidden_size):
        raise ValueError("grad_hidden_gates_steps must have shape [seq, batch, 3 * hidden].")
    if partial_sums.shape != (batch_size, 16, hidden_size):
        raise ValueError("partial_sums must have shape [batch, 16, hidden].")
    if not gate_cache.is_contiguous():
        raise ValueError("gate_cache must be contiguous.")
    if not grad_hidden_gates_steps.is_contiguous() or not partial_sums.is_contiguous():
        raise ValueError("grad_hidden_gates_steps and partial_sums must be contiguous.")

    grad_output = grad_output.contiguous()
    grad_hidden_state = grad_hidden_state.contiguous()
    gate_cache = gate_cache.contiguous()
    weight_hh = weight_hh.contiguous()

    compiled = _get_a100_kernel()
    stream = cuda.CUstream(torch.cuda.current_stream(device=gate_cache.device).cuda_stream)
    shared_bytes = 48 * ctypes.sizeof(ctypes.c_float)
    values = (
        ctypes.c_void_p(grad_output.data_ptr()),
        ctypes.c_void_p(gate_cache.data_ptr()),
        ctypes.c_void_p(h0.data_ptr()),
        ctypes.c_void_p(output.data_ptr()),
        ctypes.c_void_p(weight_hh.data_ptr()),
        ctypes.c_void_p(grad_input_gates.data_ptr()),
        ctypes.c_void_p(grad_hidden_gates_steps.data_ptr()),
        ctypes.c_void_p(partial_sums.data_ptr()),
        ctypes.c_void_p(grad_hidden_state.data_ptr()),
        ctypes.c_int(batch_size),
        ctypes.c_int(seq_len),
    )
    types = (
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_int,
    )
    err, = cuda.cuLaunchCooperativeKernel(
        compiled.h256_backward_sequence_cooperative_split16_gate_cache_state_tiled_function,
        batch_size * 16,
        1,
        1,
        block_threads,
        1,
        1,
        shared_bytes,
        stream,
        (values, types),
    )
    _check_cuda_error(err)
    return grad_hidden_state


def _a100_gru_h256_backward_sequence_cooperative_split32_gate_cache_state_tiled(
    grad_output: torch.Tensor,
    grad_hidden_state: torch.Tensor,
    gate_cache: torch.Tensor,
    h0: torch.Tensor,
    output: torch.Tensor,
    weight_hh: torch.Tensor,
    grad_input_gates: torch.Tensor,
    grad_hidden_gates_steps: torch.Tensor,
    partial_sums: torch.Tensor,
    block_threads: int = 256,
) -> torch.Tensor:
    batch_size, seq_len, cache_size = gate_cache.shape
    hidden_size = cache_size // 4
    if hidden_size != 256:
        raise ValueError(
            "A100 h256 persistent split32 tiled gate-cache backward requires hidden_size=256."
        )
    if block_threads != 256:
        raise ValueError(
            "A100 h256 persistent split32 tiled gate-cache backward requires 256 threads."
        )
    if grad_hidden_gates_steps.shape != (seq_len, batch_size, 3 * hidden_size):
        raise ValueError("grad_hidden_gates_steps must have shape [seq, batch, 3 * hidden].")
    if partial_sums.shape != (batch_size, 32, hidden_size):
        raise ValueError("partial_sums must have shape [batch, 32, hidden].")
    if not gate_cache.is_contiguous():
        raise ValueError("gate_cache must be contiguous.")
    if not grad_hidden_gates_steps.is_contiguous() or not partial_sums.is_contiguous():
        raise ValueError("grad_hidden_gates_steps and partial_sums must be contiguous.")

    grad_output = grad_output.contiguous()
    grad_hidden_state = grad_hidden_state.contiguous()
    gate_cache = gate_cache.contiguous()
    weight_hh = weight_hh.contiguous()

    compiled = _get_a100_kernel()
    stream = cuda.CUstream(torch.cuda.current_stream(device=gate_cache.device).cuda_stream)
    shared_bytes = 24 * ctypes.sizeof(ctypes.c_float)
    values = (
        ctypes.c_void_p(grad_output.data_ptr()),
        ctypes.c_void_p(gate_cache.data_ptr()),
        ctypes.c_void_p(h0.data_ptr()),
        ctypes.c_void_p(output.data_ptr()),
        ctypes.c_void_p(weight_hh.data_ptr()),
        ctypes.c_void_p(grad_input_gates.data_ptr()),
        ctypes.c_void_p(grad_hidden_gates_steps.data_ptr()),
        ctypes.c_void_p(partial_sums.data_ptr()),
        ctypes.c_void_p(grad_hidden_state.data_ptr()),
        ctypes.c_int(batch_size),
        ctypes.c_int(seq_len),
    )
    types = (
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_int,
    )
    err, = cuda.cuLaunchCooperativeKernel(
        compiled.h256_backward_sequence_cooperative_split32_gate_cache_state_tiled_function,
        batch_size * 32,
        1,
        1,
        block_threads,
        1,
        1,
        shared_bytes,
        stream,
        (values, types),
    )
    _check_cuda_error(err)
    return grad_hidden_state


def _a100_gru_h256_backward_sequence_cooperative_split16_grad_coeff_cache_state_tiled(
    grad_output: torch.Tensor,
    grad_hidden_state: torch.Tensor,
    grad_coeff_cache: torch.Tensor,
    weight_hh: torch.Tensor,
    grad_input_gates: torch.Tensor,
    grad_hidden_gates_steps: torch.Tensor,
    partial_sums: torch.Tensor,
    block_threads: int = 256,
) -> torch.Tensor:
    batch_size, seq_len, cache_size = grad_coeff_cache.shape
    hidden_size = cache_size // 5
    if hidden_size != 256:
        raise ValueError(
            "A100 h256 persistent split16 tiled grad-coeff-cache backward requires hidden_size=256."
        )
    if block_threads != 256:
        raise ValueError(
            "A100 h256 persistent split16 tiled grad-coeff-cache backward requires 256 threads."
        )
    if grad_hidden_gates_steps.shape != (seq_len, batch_size, 3 * hidden_size):
        raise ValueError("grad_hidden_gates_steps must have shape [seq, batch, 3 * hidden].")
    if partial_sums.shape != (batch_size, 16, hidden_size):
        raise ValueError("partial_sums must have shape [batch, 16, hidden].")
    if not grad_coeff_cache.is_contiguous():
        raise ValueError("grad_coeff_cache must be contiguous.")
    if not grad_hidden_gates_steps.is_contiguous() or not partial_sums.is_contiguous():
        raise ValueError("grad_hidden_gates_steps and partial_sums must be contiguous.")

    grad_output = grad_output.contiguous()
    grad_hidden_state = grad_hidden_state.contiguous()
    grad_coeff_cache = grad_coeff_cache.contiguous()
    weight_hh = weight_hh.contiguous()

    compiled = _get_a100_kernel()
    stream = cuda.CUstream(torch.cuda.current_stream(device=grad_coeff_cache.device).cuda_stream)
    shared_bytes = 48 * ctypes.sizeof(ctypes.c_float)
    values = (
        ctypes.c_void_p(grad_output.data_ptr()),
        ctypes.c_void_p(grad_coeff_cache.data_ptr()),
        ctypes.c_void_p(weight_hh.data_ptr()),
        ctypes.c_void_p(grad_input_gates.data_ptr()),
        ctypes.c_void_p(grad_hidden_gates_steps.data_ptr()),
        ctypes.c_void_p(partial_sums.data_ptr()),
        ctypes.c_void_p(grad_hidden_state.data_ptr()),
        ctypes.c_int(batch_size),
        ctypes.c_int(seq_len),
    )
    types = (
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_int,
    )
    err, = cuda.cuLaunchCooperativeKernel(
        compiled.h256_backward_sequence_cooperative_split16_grad_coeff_cache_state_tiled_function,
        batch_size * 16,
        1,
        1,
        block_threads,
        1,
        1,
        shared_bytes,
        stream,
        (values, types),
    )
    _check_cuda_error(err)
    return grad_hidden_state


def _a100_gru_h256_backward_sequence_cooperative_split16_state_global_gates(
    grad_output: torch.Tensor,
    grad_hidden_state: torch.Tensor,
    input_gates: torch.Tensor,
    hidden_gates_steps: torch.Tensor,
    h0: torch.Tensor,
    output: torch.Tensor,
    weight_hh: torch.Tensor,
    grad_input_gates: torch.Tensor,
    grad_hidden_gates_steps: torch.Tensor,
    partial_sums: torch.Tensor,
    block_threads: int = 256,
) -> torch.Tensor:
    batch_size, seq_len, hidden3 = input_gates.shape
    hidden_size = hidden3 // 3
    if hidden_size != 256:
        raise ValueError("A100 h256 persistent split16 global-gates backward requires hidden_size=256.")
    if block_threads != 256:
        raise ValueError("A100 h256 persistent split16 global-gates backward requires 256 threads.")
    if hidden_gates_steps.shape != (seq_len, batch_size, 3 * hidden_size):
        raise ValueError("hidden_gates_steps must have shape [seq, batch, 3 * hidden].")
    if grad_hidden_gates_steps.shape != (seq_len, batch_size, 3 * hidden_size):
        raise ValueError("grad_hidden_gates_steps must have shape [seq, batch, 3 * hidden].")
    if partial_sums.shape != (batch_size, 16, hidden_size):
        raise ValueError("partial_sums must have shape [batch, 16, hidden].")
    if not hidden_gates_steps.is_contiguous():
        raise ValueError("hidden_gates_steps must be contiguous.")
    if not grad_hidden_gates_steps.is_contiguous() or not partial_sums.is_contiguous():
        raise ValueError("grad_hidden_gates_steps and partial_sums must be contiguous.")

    grad_output = grad_output.contiguous()
    grad_hidden_state = grad_hidden_state.contiguous()
    input_gates = input_gates.contiguous()
    weight_hh = weight_hh.contiguous()

    compiled = _get_a100_kernel()
    stream = cuda.CUstream(torch.cuda.current_stream(device=input_gates.device).cuda_stream)
    values = (
        ctypes.c_void_p(grad_output.data_ptr()),
        ctypes.c_void_p(input_gates.data_ptr()),
        ctypes.c_void_p(hidden_gates_steps.data_ptr()),
        ctypes.c_void_p(h0.data_ptr()),
        ctypes.c_void_p(output.data_ptr()),
        ctypes.c_void_p(weight_hh.data_ptr()),
        ctypes.c_void_p(grad_input_gates.data_ptr()),
        ctypes.c_void_p(grad_hidden_gates_steps.data_ptr()),
        ctypes.c_void_p(partial_sums.data_ptr()),
        ctypes.c_void_p(grad_hidden_state.data_ptr()),
        ctypes.c_int(batch_size),
        ctypes.c_int(seq_len),
    )
    types = (
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_int,
    )
    err, = cuda.cuLaunchCooperativeKernel(
        compiled.h256_backward_sequence_cooperative_split16_state_global_gates_function,
        batch_size * 16,
        1,
        1,
        block_threads,
        1,
        1,
        0,
        stream,
        (values, types),
    )
    _check_cuda_error(err)
    return grad_hidden_state


def _a100_gru_h256_backward_sequence_cooperative_split32_state(
    grad_output: torch.Tensor,
    grad_hidden_state: torch.Tensor,
    input_gates: torch.Tensor,
    hidden_gates_steps: torch.Tensor,
    h0: torch.Tensor,
    output: torch.Tensor,
    weight_hh: torch.Tensor,
    grad_input_gates: torch.Tensor,
    grad_hidden_gates_steps: torch.Tensor,
    partial_sums: torch.Tensor,
    block_threads: int = 256,
) -> torch.Tensor:
    batch_size, seq_len, hidden3 = input_gates.shape
    hidden_size = hidden3 // 3
    if hidden_size != 256:
        raise ValueError("A100 h256 persistent split32 backward requires hidden_size=256.")
    if block_threads != 256:
        raise ValueError("A100 h256 persistent split32 backward requires 256 threads.")
    if hidden_gates_steps.shape != (seq_len, batch_size, 3 * hidden_size):
        raise ValueError("hidden_gates_steps must have shape [seq, batch, 3 * hidden].")
    if grad_hidden_gates_steps.shape != (seq_len, batch_size, 3 * hidden_size):
        raise ValueError("grad_hidden_gates_steps must have shape [seq, batch, 3 * hidden].")
    if partial_sums.shape != (batch_size, 32, hidden_size):
        raise ValueError("partial_sums must have shape [batch, 32, hidden].")
    if not hidden_gates_steps.is_contiguous():
        raise ValueError("hidden_gates_steps must be contiguous.")
    if not grad_hidden_gates_steps.is_contiguous() or not partial_sums.is_contiguous():
        raise ValueError("grad_hidden_gates_steps and partial_sums must be contiguous.")

    grad_output = grad_output.contiguous()
    grad_hidden_state = grad_hidden_state.contiguous()
    input_gates = input_gates.contiguous()
    weight_hh = weight_hh.contiguous()

    compiled = _get_a100_kernel()
    stream = cuda.CUstream(torch.cuda.current_stream(device=input_gates.device).cuda_stream)
    shared_bytes = 3 * hidden_size * ctypes.sizeof(ctypes.c_float)
    values = (
        ctypes.c_void_p(grad_output.data_ptr()),
        ctypes.c_void_p(input_gates.data_ptr()),
        ctypes.c_void_p(hidden_gates_steps.data_ptr()),
        ctypes.c_void_p(h0.data_ptr()),
        ctypes.c_void_p(output.data_ptr()),
        ctypes.c_void_p(weight_hh.data_ptr()),
        ctypes.c_void_p(grad_input_gates.data_ptr()),
        ctypes.c_void_p(grad_hidden_gates_steps.data_ptr()),
        ctypes.c_void_p(partial_sums.data_ptr()),
        ctypes.c_void_p(grad_hidden_state.data_ptr()),
        ctypes.c_int(batch_size),
        ctypes.c_int(seq_len),
    )
    types = (
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_int,
    )
    err, = cuda.cuLaunchCooperativeKernel(
        compiled.h256_backward_sequence_cooperative_split32_state_function,
        batch_size * 32,
        1,
        1,
        block_threads,
        1,
        1,
        shared_bytes,
        stream,
        (values, types),
    )
    _check_cuda_error(err)
    return grad_hidden_state


def _a100_gru_h256_backward_step_recompute(
    grad_hidden_next: torch.Tensor,
    input_gates: torch.Tensor,
    h0: torch.Tensor,
    output: torch.Tensor,
    weight_hh: torch.Tensor,
    bias_hh: torch.Tensor,
    grad_input_gates: torch.Tensor,
    grad_hidden_gates_out: torch.Tensor,
    step: int,
    block_threads: int = 256,
) -> torch.Tensor:
    batch_size, seq_len, hidden3 = input_gates.shape
    hidden_size = hidden3 // 3
    if hidden_size != 256:
        raise ValueError("A100 h256 recompute backward step requires hidden_size=256.")
    if block_threads != 256:
        raise ValueError("A100 h256 recompute backward step currently requires 256 threads.")
    if grad_hidden_gates_out.shape != (batch_size, 3 * hidden_size):
        raise ValueError("grad_hidden_gates_out must have shape [batch, 3 * hidden].")
    if not grad_hidden_gates_out.is_contiguous():
        raise ValueError("grad_hidden_gates_out must be contiguous.")

    grad_hidden_next = grad_hidden_next.contiguous()
    weight_hh = weight_hh.contiguous()
    bias_hh = bias_hh.contiguous()
    grad_hidden_prev = torch.empty_like(grad_hidden_next)

    compiled = _get_a100_kernel()
    stream = cuda.CUstream(torch.cuda.current_stream(device=input_gates.device).cuda_stream)
    shared_bytes = 4 * hidden_size * ctypes.sizeof(ctypes.c_float)
    values = (
        ctypes.c_void_p(grad_hidden_next.data_ptr()),
        ctypes.c_void_p(input_gates.data_ptr()),
        ctypes.c_void_p(h0.data_ptr()),
        ctypes.c_void_p(output.data_ptr()),
        ctypes.c_void_p(weight_hh.data_ptr()),
        ctypes.c_void_p(bias_hh.data_ptr()),
        ctypes.c_void_p(grad_input_gates.data_ptr()),
        ctypes.c_void_p(grad_hidden_gates_out.data_ptr()),
        ctypes.c_void_p(grad_hidden_prev.data_ptr()),
        ctypes.c_int(batch_size),
        ctypes.c_int(seq_len),
        ctypes.c_int(step),
    )
    types = (
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
    )
    err, = cuda.cuLaunchKernel(
        compiled.h256_backward_step_recompute_function,
        batch_size,
        1,
        1,
        block_threads,
        1,
        1,
        shared_bytes,
        stream,
        (values, types),
        0,
    )
    _check_cuda_error(err)
    return grad_hidden_prev


class A100GRUH256Function(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        h0: torch.Tensor,
        weight_ih: torch.Tensor,
        weight_hh: torch.Tensor,
        bias_ih: torch.Tensor,
        bias_hh: torch.Tensor,
        block_threads: int = 704,
        use_recurrent_backward_kernel: bool = False,
        recompute_hidden_gates: bool = False,
        use_tiled_recurrent_backward_kernel: bool = False,
        use_split_recurrent_backward_kernel: bool = False,
        split_recurrent_count: int = 8,
        use_cooperative_split_backward_kernel: bool = False,
        cooperative_split_count: int = 4,
        use_cooperative_split_cached_backward_kernel: bool = False,
        use_cooperative_split2_backward_kernel: bool = False,
        use_cooperative_split2_cached_local_backward_kernel: bool = False,
        use_gate_cache_backward_kernel: bool = False,
        use_persistent_backward_kernel: bool = False,
        use_persistent_state_backward_kernel: bool = False,
        use_persistent_state_local_backward_kernel: bool = False,
        use_persistent_state4_backward_kernel: bool = False,
        use_persistent_state8_backward_kernel: bool = False,
        use_persistent_state16_backward_kernel: bool = False,
        use_persistent_state16_gate_cache_backward_kernel: bool = False,
        use_persistent_state16_gate_cache_tiled_backward_kernel: bool = False,
        use_persistent_state16_grad_coeff_cache_tiled_backward_kernel: bool = False,
        use_persistent_state32_gate_cache_tiled_backward_kernel: bool = False,
        use_gate_cache_parallel_update_forward_kernel: bool = False,
        use_gate_cache_cta8_forward_kernel: bool = False,
        use_gate_cache_cta6_forward_kernel: bool = False,
        use_persistent_state16_global_gates_backward_kernel: bool = False,
        use_persistent_state32_backward_kernel: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not x.is_cuda:
            raise RuntimeError("A100GRUH256Function requires CUDA tensors.")
        if x.dtype != torch.float32:
            raise RuntimeError("A100GRUH256Function currently supports fp32 only.")
        if x.dim() != 3:
            raise ValueError("x must have shape [batch, seq, input].")
        if h0.dim() != 2:
            raise ValueError("h0 must have shape [batch, hidden].")
        if h0.size(1) != 256:
            raise ValueError("A100GRUH256Function requires hidden_size=256.")

        x = x.contiguous()
        h0 = h0.contiguous()
        weight_ih = weight_ih.contiguous()
        weight_hh = weight_hh.contiguous()
        bias_ih = bias_ih.contiguous()
        bias_hh = bias_hh.contiguous()

        batch_size, seq_len, input_size = x.shape
        hidden_size = h0.size(1)
        if h0.size(0) != batch_size:
            raise ValueError("h0 batch dimension must match x.")
        if weight_ih.shape != (3 * hidden_size, input_size):
            raise ValueError("weight_ih must have shape [3 * hidden, input].")
        if weight_hh.shape != (3 * hidden_size, hidden_size):
            raise ValueError("weight_hh must have shape [3 * hidden, hidden].")
        if bias_ih.shape != (3 * hidden_size,):
            raise ValueError("bias_ih must have shape [3 * hidden].")
        if bias_hh.shape != (3 * hidden_size,):
            raise ValueError("bias_hh must have shape [3 * hidden].")

        input_gates = F.linear(
            x.reshape(batch_size * seq_len, input_size),
            weight_ih,
            bias_ih,
        ).view(batch_size, seq_len, 3 * hidden_size)
        if (
            (
                use_persistent_state16_gate_cache_backward_kernel
                or use_persistent_state16_gate_cache_tiled_backward_kernel
                or use_persistent_state32_gate_cache_tiled_backward_kernel
            )
            and use_gate_cache_parallel_update_forward_kernel
        ):
            output, gate_cache = (
                a100_gru_forward_from_gates_cooperative_h256_parallel_update_gate_cache(
                    input_gates,
                    h0,
                    weight_hh,
                    bias_hh,
                    block_threads=block_threads,
                )
            )
        elif (
            (
                use_persistent_state16_gate_cache_backward_kernel
                or use_persistent_state16_gate_cache_tiled_backward_kernel
                or use_persistent_state32_gate_cache_tiled_backward_kernel
            )
            and use_gate_cache_cta8_forward_kernel
        ):
            output, gate_cache = a100_gru_forward_from_gates_cooperative_h256_cta8_shmem_gate_cache(
                input_gates,
                h0,
                weight_hh,
                bias_hh,
                block_threads=block_threads,
            )
        elif (
            (
                use_persistent_state16_gate_cache_backward_kernel
                or use_persistent_state16_gate_cache_tiled_backward_kernel
                or use_persistent_state32_gate_cache_tiled_backward_kernel
            )
            and use_gate_cache_cta6_forward_kernel
        ):
            output, gate_cache = a100_gru_forward_from_gates_cooperative_h256_cta6_shmem_gate_cache(
                input_gates,
                h0,
                weight_hh,
                bias_hh,
                block_threads=block_threads,
            )
        elif use_persistent_state16_grad_coeff_cache_tiled_backward_kernel:
            output, gate_cache = (
                a100_gru_forward_from_gates_cooperative_h256_shmem_grad_coeff_cache(
                    input_gates,
                    h0,
                    weight_hh,
                    bias_hh,
                    block_threads=block_threads,
                )
            )
        elif (
            use_gate_cache_backward_kernel
            or use_persistent_state16_gate_cache_backward_kernel
            or use_persistent_state16_gate_cache_tiled_backward_kernel
            or use_persistent_state32_gate_cache_tiled_backward_kernel
        ):
            output, gate_cache = a100_gru_forward_from_gates_cooperative_h256_shmem_gate_cache(
                input_gates,
                h0,
                weight_hh,
                bias_hh,
                block_threads=block_threads,
            )
        else:
            output = a100_gru_forward_from_gates_cooperative_h256_shmem(
                input_gates,
                h0,
                weight_hh,
                bias_hh,
                block_threads=block_threads,
            )
            gate_cache = x.new_empty(0)
        h_n = output[:, -1, :].unsqueeze(0).contiguous()

        ctx.save_for_backward(
            x,
            h0,
            weight_ih,
            weight_hh,
            bias_hh,
            input_gates,
            output,
            gate_cache,
        )
        ctx.block_threads = block_threads
        ctx.use_recurrent_backward_kernel = use_recurrent_backward_kernel
        ctx.recompute_hidden_gates = recompute_hidden_gates
        ctx.use_tiled_recurrent_backward_kernel = use_tiled_recurrent_backward_kernel
        ctx.use_split_recurrent_backward_kernel = use_split_recurrent_backward_kernel
        ctx.split_recurrent_count = split_recurrent_count
        ctx.use_cooperative_split_backward_kernel = use_cooperative_split_backward_kernel
        ctx.cooperative_split_count = cooperative_split_count
        ctx.use_cooperative_split_cached_backward_kernel = (
            use_cooperative_split_cached_backward_kernel
        )
        ctx.use_cooperative_split2_backward_kernel = use_cooperative_split2_backward_kernel
        ctx.use_cooperative_split2_cached_local_backward_kernel = (
            use_cooperative_split2_cached_local_backward_kernel
        )
        ctx.use_gate_cache_backward_kernel = use_gate_cache_backward_kernel
        ctx.use_persistent_backward_kernel = use_persistent_backward_kernel
        ctx.use_persistent_state_backward_kernel = use_persistent_state_backward_kernel
        ctx.use_persistent_state_local_backward_kernel = (
            use_persistent_state_local_backward_kernel
        )
        ctx.use_persistent_state4_backward_kernel = use_persistent_state4_backward_kernel
        ctx.use_persistent_state8_backward_kernel = use_persistent_state8_backward_kernel
        ctx.use_persistent_state16_backward_kernel = use_persistent_state16_backward_kernel
        ctx.use_persistent_state16_gate_cache_backward_kernel = (
            use_persistent_state16_gate_cache_backward_kernel
        )
        ctx.use_persistent_state16_gate_cache_tiled_backward_kernel = (
            use_persistent_state16_gate_cache_tiled_backward_kernel
        )
        ctx.use_persistent_state16_grad_coeff_cache_tiled_backward_kernel = (
            use_persistent_state16_grad_coeff_cache_tiled_backward_kernel
        )
        ctx.use_persistent_state32_gate_cache_tiled_backward_kernel = (
            use_persistent_state32_gate_cache_tiled_backward_kernel
        )
        ctx.use_persistent_state16_global_gates_backward_kernel = (
            use_persistent_state16_global_gates_backward_kernel
        )
        ctx.use_persistent_state32_backward_kernel = use_persistent_state32_backward_kernel
        return output, h_n

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor, grad_h_n: torch.Tensor):
        x, h0, weight_ih, weight_hh, bias_hh, input_gates, output, gate_cache = (
            ctx.saved_tensors
        )
        if grad_output is None:
            grad_output = torch.zeros_like(output)
        grad_output = grad_output.contiguous()
        grad_h_n = grad_h_n.contiguous() if grad_h_n is not None else None

        batch_size, seq_len, input_size = x.shape
        hidden_size = h0.size(1)

        # 跨 time step 预计算 recurrent gates，避免 backward 循环内发起大量小 GEMM。
        hidden_prev_steps = torch.cat(
            (h0.unsqueeze(1), output[:, :-1, :]),
            dim=1,
        ).transpose(0, 1).contiguous()
        if (
            ctx.recompute_hidden_gates
            or ctx.use_gate_cache_backward_kernel
            or ctx.use_persistent_state16_gate_cache_backward_kernel
            or ctx.use_persistent_state16_gate_cache_tiled_backward_kernel
            or ctx.use_persistent_state16_grad_coeff_cache_tiled_backward_kernel
            or ctx.use_persistent_state32_gate_cache_tiled_backward_kernel
        ):
            hidden_gates_steps = None
        else:
            hidden_gates_steps = F.linear(
                hidden_prev_steps.reshape(seq_len * batch_size, hidden_size),
                weight_hh,
                bias_hh,
            ).view(seq_len, batch_size, 3 * hidden_size)

        grad_input_gates = torch.empty_like(input_gates)
        # 使用 [time, batch, gate] 布局，循环后一次 GEMM 累积 weight_hh 梯度。
        grad_hidden_gates_steps = torch.empty(
            seq_len,
            batch_size,
            3 * hidden_size,
            device=x.device,
            dtype=x.dtype,
        )
        recurrent_partial_sums = (
            torch.empty(
                ctx.split_recurrent_count,
                batch_size,
                hidden_size,
                device=x.device,
                dtype=x.dtype,
            )
            if ctx.use_split_recurrent_backward_kernel
            else None
        )
        cooperative_partial_sums = (
            torch.empty(
                batch_size,
                ctx.cooperative_split_count,
                hidden_size,
                device=x.device,
                dtype=x.dtype,
            )
            if ctx.use_cooperative_split_backward_kernel
            else None
        )
        cooperative_cached_partial_sums = (
            torch.empty(
                batch_size,
                ctx.cooperative_split_count,
                hidden_size,
                device=x.device,
                dtype=x.dtype,
            )
            if ctx.use_cooperative_split_cached_backward_kernel
            else None
        )
        cooperative_split2_partial_sums = (
            torch.empty(batch_size, 2, hidden_size, device=x.device, dtype=x.dtype)
            if ctx.use_cooperative_split2_backward_kernel
            else None
        )
        cooperative_split2_cached_local_partial_sums = (
            torch.empty(batch_size, 2, hidden_size, device=x.device, dtype=x.dtype)
            if ctx.use_cooperative_split2_cached_local_backward_kernel
            else None
        )
        cooperative_gate_cache_partial_sums = (
            torch.empty(batch_size, 2, hidden_size, device=x.device, dtype=x.dtype)
            if ctx.use_gate_cache_backward_kernel
            else None
        )
        persistent_partial_sums = (
            torch.empty(batch_size, 2, hidden_size, device=x.device, dtype=x.dtype)
            if ctx.use_persistent_backward_kernel
            else None
        )
        persistent_state_partial_sums = (
            torch.empty(batch_size, 2, hidden_size, device=x.device, dtype=x.dtype)
            if ctx.use_persistent_state_backward_kernel
            else None
        )
        persistent_state_local_partial_sums = (
            torch.empty(batch_size, 2, hidden_size, device=x.device, dtype=x.dtype)
            if ctx.use_persistent_state_local_backward_kernel
            else None
        )
        persistent_state4_partial_sums = (
            torch.empty(batch_size, 4, hidden_size, device=x.device, dtype=x.dtype)
            if ctx.use_persistent_state4_backward_kernel
            else None
        )
        persistent_state8_partial_sums = (
            torch.empty(batch_size, 8, hidden_size, device=x.device, dtype=x.dtype)
            if ctx.use_persistent_state8_backward_kernel
            else None
        )
        persistent_state16_partial_sums = (
            torch.empty(batch_size, 16, hidden_size, device=x.device, dtype=x.dtype)
            if ctx.use_persistent_state16_backward_kernel
            else None
        )
        persistent_state16_gate_cache_partial_sums = (
            torch.empty(batch_size, 16, hidden_size, device=x.device, dtype=x.dtype)
            if ctx.use_persistent_state16_gate_cache_backward_kernel
            else None
        )
        persistent_state16_gate_cache_tiled_partial_sums = (
            torch.empty(batch_size, 16, hidden_size, device=x.device, dtype=x.dtype)
            if ctx.use_persistent_state16_gate_cache_tiled_backward_kernel
            else None
        )
        persistent_state16_grad_coeff_cache_tiled_partial_sums = (
            torch.empty(batch_size, 16, hidden_size, device=x.device, dtype=x.dtype)
            if ctx.use_persistent_state16_grad_coeff_cache_tiled_backward_kernel
            else None
        )
        persistent_state32_gate_cache_tiled_partial_sums = (
            torch.empty(batch_size, 32, hidden_size, device=x.device, dtype=x.dtype)
            if ctx.use_persistent_state32_gate_cache_tiled_backward_kernel
            else None
        )
        persistent_state16_global_gates_partial_sums = (
            torch.empty(batch_size, 16, hidden_size, device=x.device, dtype=x.dtype)
            if ctx.use_persistent_state16_global_gates_backward_kernel
            else None
        )
        persistent_state32_partial_sums = (
            torch.empty(batch_size, 32, hidden_size, device=x.device, dtype=x.dtype)
            if ctx.use_persistent_state32_backward_kernel
            else None
        )
        grad_hidden = torch.zeros_like(h0)
        if grad_h_n is not None:
            grad_hidden = grad_hidden + grad_h_n[0]

        if ctx.use_persistent_state16_grad_coeff_cache_tiled_backward_kernel:
            assert persistent_state16_grad_coeff_cache_tiled_partial_sums is not None
            grad_hidden = (
                _a100_gru_h256_backward_sequence_cooperative_split16_grad_coeff_cache_state_tiled(
                    grad_output,
                    grad_hidden,
                    gate_cache,
                    weight_hh,
                    grad_input_gates,
                    grad_hidden_gates_steps,
                    persistent_state16_grad_coeff_cache_tiled_partial_sums,
                )
            )
        elif ctx.use_persistent_state32_gate_cache_tiled_backward_kernel:
            assert persistent_state32_gate_cache_tiled_partial_sums is not None
            grad_hidden = (
                _a100_gru_h256_backward_sequence_cooperative_split32_gate_cache_state_tiled(
                    grad_output,
                    grad_hidden,
                    gate_cache,
                    h0,
                    output,
                    weight_hh,
                    grad_input_gates,
                    grad_hidden_gates_steps,
                    persistent_state32_gate_cache_tiled_partial_sums,
                )
            )
        elif ctx.use_persistent_state16_gate_cache_tiled_backward_kernel:
            assert persistent_state16_gate_cache_tiled_partial_sums is not None
            grad_hidden = (
                _a100_gru_h256_backward_sequence_cooperative_split16_gate_cache_state_tiled(
                    grad_output,
                    grad_hidden,
                    gate_cache,
                    h0,
                    output,
                    weight_hh,
                    grad_input_gates,
                    grad_hidden_gates_steps,
                    persistent_state16_gate_cache_tiled_partial_sums,
                )
            )
        elif ctx.use_persistent_state16_gate_cache_backward_kernel:
            assert persistent_state16_gate_cache_partial_sums is not None
            grad_hidden = (
                _a100_gru_h256_backward_sequence_cooperative_split16_gate_cache_state(
                    grad_output,
                    grad_hidden,
                    gate_cache,
                    h0,
                    output,
                    weight_hh,
                    grad_input_gates,
                    grad_hidden_gates_steps,
                    persistent_state16_gate_cache_partial_sums,
                )
            )
        elif ctx.use_persistent_state16_global_gates_backward_kernel:
            assert hidden_gates_steps is not None
            assert persistent_state16_global_gates_partial_sums is not None
            grad_hidden = (
                _a100_gru_h256_backward_sequence_cooperative_split16_state_global_gates(
                    grad_output,
                    grad_hidden,
                    input_gates,
                    hidden_gates_steps,
                    h0,
                    output,
                    weight_hh,
                    grad_input_gates,
                    grad_hidden_gates_steps,
                    persistent_state16_global_gates_partial_sums,
                )
            )
        elif ctx.use_persistent_state32_backward_kernel:
            assert hidden_gates_steps is not None
            assert persistent_state32_partial_sums is not None
            grad_hidden = _a100_gru_h256_backward_sequence_cooperative_split32_state(
                grad_output,
                grad_hidden,
                input_gates,
                hidden_gates_steps,
                h0,
                output,
                weight_hh,
                grad_input_gates,
                grad_hidden_gates_steps,
                persistent_state32_partial_sums,
            )
        elif ctx.use_persistent_state16_backward_kernel:
            assert hidden_gates_steps is not None
            assert persistent_state16_partial_sums is not None
            grad_hidden = _a100_gru_h256_backward_sequence_cooperative_split16_state(
                grad_output,
                grad_hidden,
                input_gates,
                hidden_gates_steps,
                h0,
                output,
                weight_hh,
                grad_input_gates,
                grad_hidden_gates_steps,
                persistent_state16_partial_sums,
            )
        elif ctx.use_persistent_state8_backward_kernel:
            assert hidden_gates_steps is not None
            assert persistent_state8_partial_sums is not None
            grad_hidden = _a100_gru_h256_backward_sequence_cooperative_split8_state(
                grad_output,
                grad_hidden,
                input_gates,
                hidden_gates_steps,
                h0,
                output,
                weight_hh,
                grad_input_gates,
                grad_hidden_gates_steps,
                persistent_state8_partial_sums,
            )
        elif ctx.use_persistent_state4_backward_kernel:
            assert hidden_gates_steps is not None
            assert persistent_state4_partial_sums is not None
            grad_hidden = _a100_gru_h256_backward_sequence_cooperative_split4_state(
                grad_output,
                grad_hidden,
                input_gates,
                hidden_gates_steps,
                h0,
                output,
                weight_hh,
                grad_input_gates,
                grad_hidden_gates_steps,
                persistent_state4_partial_sums,
            )
        elif ctx.use_persistent_state_local_backward_kernel:
            assert hidden_gates_steps is not None
            assert persistent_state_local_partial_sums is not None
            grad_hidden = _a100_gru_h256_backward_sequence_cooperative_split2_state_local(
                grad_output,
                grad_hidden,
                input_gates,
                hidden_gates_steps,
                h0,
                output,
                weight_hh,
                grad_input_gates,
                grad_hidden_gates_steps,
                persistent_state_local_partial_sums,
            )
        elif ctx.use_persistent_state_backward_kernel:
            assert hidden_gates_steps is not None
            assert persistent_state_partial_sums is not None
            grad_hidden = _a100_gru_h256_backward_sequence_cooperative_split2_state_parts(
                grad_output,
                grad_hidden,
                input_gates,
                hidden_gates_steps,
                h0,
                output,
                weight_hh,
                grad_input_gates,
                grad_hidden_gates_steps,
                persistent_state_partial_sums,
            )
        elif ctx.use_persistent_backward_kernel:
            assert hidden_gates_steps is not None
            assert persistent_partial_sums is not None
            grad_hidden = _a100_gru_h256_backward_sequence_cooperative_split2_cached_local(
                grad_output,
                grad_hidden,
                input_gates,
                hidden_gates_steps,
                h0,
                output,
                weight_hh,
                grad_input_gates,
                grad_hidden_gates_steps,
                persistent_partial_sums,
            )
        else:
            for step in range(seq_len - 1, -1, -1):
                grad_hidden_next = grad_hidden + grad_output[:, step, :]
                grad_hidden_gates_out = grad_hidden_gates_steps[step]
                if ctx.use_gate_cache_backward_kernel:
                    assert cooperative_gate_cache_partial_sums is not None
                    grad_hidden = _a100_gru_h256_backward_step_cooperative_split2_gate_cache(
                        grad_hidden_next,
                        gate_cache,
                        h0,
                        output,
                        weight_hh,
                        grad_input_gates,
                        grad_hidden_gates_out,
                        cooperative_gate_cache_partial_sums,
                        step,
                    )
                elif ctx.recompute_hidden_gates:
                    grad_hidden = _a100_gru_h256_backward_step_recompute(
                        grad_hidden_next,
                        input_gates,
                        h0,
                        output,
                        weight_hh,
                        bias_hh,
                        grad_input_gates,
                        grad_hidden_gates_out,
                        step,
                    )
                elif ctx.use_recurrent_backward_kernel:
                    assert hidden_gates_steps is not None
                    hidden_gates = hidden_gates_steps[step]
                    grad_hidden = _a100_gru_h256_backward_step(
                        grad_hidden_next,
                        input_gates,
                        hidden_gates,
                        h0,
                        output,
                        weight_hh,
                        grad_input_gates,
                        grad_hidden_gates_out,
                        step,
                    )
                elif ctx.use_tiled_recurrent_backward_kernel:
                    assert hidden_gates_steps is not None
                    hidden_gates = hidden_gates_steps[step]
                    grad_hidden_gates, grad_hidden_prev_direct = _a100_gru_h256_pointwise_backward(
                        grad_hidden_next,
                        input_gates,
                        hidden_gates,
                        h0,
                        output,
                        grad_input_gates,
                        step,
                        grad_hidden_gates_out=grad_hidden_gates_out,
                    )
                    grad_hidden = _a100_gru_h256_recurrent_backward_tiled(
                        grad_hidden_gates,
                        weight_hh,
                        grad_hidden_prev_direct,
                    )
                elif ctx.use_split_recurrent_backward_kernel:
                    assert hidden_gates_steps is not None
                    assert recurrent_partial_sums is not None
                    hidden_gates = hidden_gates_steps[step]
                    grad_hidden_gates, grad_hidden_prev_direct = _a100_gru_h256_pointwise_backward(
                        grad_hidden_next,
                        input_gates,
                        hidden_gates,
                        h0,
                        output,
                        grad_input_gates,
                        step,
                        grad_hidden_gates_out=grad_hidden_gates_out,
                    )
                    grad_hidden = _a100_gru_h256_recurrent_backward_split(
                        grad_hidden_gates,
                        weight_hh,
                        grad_hidden_prev_direct,
                        recurrent_partial_sums,
                        split_count=ctx.split_recurrent_count,
                    )
                elif ctx.use_cooperative_split_backward_kernel:
                    assert hidden_gates_steps is not None
                    assert cooperative_partial_sums is not None
                    hidden_gates = hidden_gates_steps[step]
                    grad_hidden = _a100_gru_h256_backward_step_cooperative_split(
                        grad_hidden_next,
                        input_gates,
                        hidden_gates,
                        h0,
                        output,
                        weight_hh,
                        grad_input_gates,
                        grad_hidden_gates_out,
                        cooperative_partial_sums,
                        step,
                        split_count=ctx.cooperative_split_count,
                    )
                elif ctx.use_cooperative_split_cached_backward_kernel:
                    assert hidden_gates_steps is not None
                    assert cooperative_cached_partial_sums is not None
                    hidden_gates = hidden_gates_steps[step]
                    grad_hidden = _a100_gru_h256_backward_step_cooperative_split_cached(
                        grad_hidden_next,
                        input_gates,
                        hidden_gates,
                        h0,
                        output,
                        weight_hh,
                        grad_input_gates,
                        grad_hidden_gates_out,
                        cooperative_cached_partial_sums,
                        step,
                        split_count=ctx.cooperative_split_count,
                    )
                elif ctx.use_cooperative_split2_backward_kernel:
                    assert hidden_gates_steps is not None
                    assert cooperative_split2_partial_sums is not None
                    hidden_gates = hidden_gates_steps[step]
                    grad_hidden = _a100_gru_h256_backward_step_cooperative_split2(
                        grad_hidden_next,
                        input_gates,
                        hidden_gates,
                        h0,
                        output,
                        weight_hh,
                        grad_input_gates,
                        grad_hidden_gates_out,
                        cooperative_split2_partial_sums,
                        step,
                    )
                elif ctx.use_cooperative_split2_cached_local_backward_kernel:
                    assert hidden_gates_steps is not None
                    assert cooperative_split2_cached_local_partial_sums is not None
                    hidden_gates = hidden_gates_steps[step]
                    grad_hidden = (
                        _a100_gru_h256_backward_step_cooperative_split2_cached_local(
                            grad_hidden_next,
                            input_gates,
                            hidden_gates,
                            h0,
                            output,
                            weight_hh,
                            grad_input_gates,
                            grad_hidden_gates_out,
                            cooperative_split2_cached_local_partial_sums,
                            step,
                        )
                    )
                else:
                    assert hidden_gates_steps is not None
                    hidden_gates = hidden_gates_steps[step]
                    grad_hidden_gates, grad_hidden_prev_direct = _a100_gru_h256_pointwise_backward(
                        grad_hidden_next,
                        input_gates,
                        hidden_gates,
                        h0,
                        output,
                        grad_input_gates,
                        step,
                        grad_hidden_gates_out=grad_hidden_gates_out,
                    )
                    grad_hidden = grad_hidden_prev_direct + grad_hidden_gates.matmul(weight_hh)

        grad_hidden_gates_2d = grad_hidden_gates_steps.reshape(
            seq_len * batch_size,
            3 * hidden_size,
        )
        hidden_prev_2d = hidden_prev_steps.reshape(seq_len * batch_size, hidden_size)
        grad_weight_hh = grad_hidden_gates_2d.transpose(0, 1).matmul(hidden_prev_2d)
        grad_bias_hh = grad_hidden_gates_2d.sum(dim=0)

        grad_input_gates_2d = grad_input_gates.reshape(batch_size * seq_len, 3 * hidden_size)
        x_2d = x.reshape(batch_size * seq_len, input_size)
        grad_x = grad_input_gates_2d.matmul(weight_ih).view(batch_size, seq_len, input_size)
        grad_weight_ih = grad_input_gates_2d.transpose(0, 1).matmul(x_2d)
        grad_bias_ih = grad_input_gates_2d.sum(dim=0)

        return (
            grad_x,
            grad_hidden,
            grad_weight_ih,
            grad_weight_hh,
            grad_bias_ih,
            grad_bias_hh,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


def a100_gru_h256(
    x: torch.Tensor,
    h0: torch.Tensor,
    weight_ih: torch.Tensor,
    weight_hh: torch.Tensor,
    bias_ih: torch.Tensor,
    bias_hh: torch.Tensor,
    block_threads: int = 704,
    use_recurrent_backward_kernel: bool = False,
    recompute_hidden_gates: bool = False,
    use_tiled_recurrent_backward_kernel: bool = False,
    use_split_recurrent_backward_kernel: bool = False,
    split_recurrent_count: int = 8,
    use_cooperative_split_backward_kernel: bool = False,
    cooperative_split_count: int = 4,
    use_cooperative_split_cached_backward_kernel: bool = False,
    use_cooperative_split2_backward_kernel: bool = False,
    use_cooperative_split2_cached_local_backward_kernel: bool = False,
    use_gate_cache_backward_kernel: bool = False,
    use_persistent_backward_kernel: bool = False,
    use_persistent_state_backward_kernel: bool = False,
    use_persistent_state_local_backward_kernel: bool = False,
    use_persistent_state4_backward_kernel: bool = False,
    use_persistent_state8_backward_kernel: bool = False,
    use_persistent_state16_backward_kernel: bool = False,
    use_persistent_state16_gate_cache_backward_kernel: bool = False,
    use_persistent_state16_gate_cache_tiled_backward_kernel: bool = False,
    use_persistent_state16_grad_coeff_cache_tiled_backward_kernel: bool = False,
    use_persistent_state32_gate_cache_tiled_backward_kernel: bool = False,
    use_gate_cache_parallel_update_forward_kernel: bool = False,
    use_gate_cache_cta8_forward_kernel: bool = False,
    use_gate_cache_cta6_forward_kernel: bool = False,
    use_persistent_state16_global_gates_backward_kernel: bool = False,
    use_persistent_state32_backward_kernel: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    return A100GRUH256Function.apply(
        x,
        h0,
        weight_ih,
        weight_hh,
        bias_ih,
        bias_hh,
        block_threads,
        use_recurrent_backward_kernel,
        recompute_hidden_gates,
        use_tiled_recurrent_backward_kernel,
        use_split_recurrent_backward_kernel,
        split_recurrent_count,
        use_cooperative_split_backward_kernel,
        cooperative_split_count,
        use_cooperative_split_cached_backward_kernel,
        use_cooperative_split2_backward_kernel,
        use_cooperative_split2_cached_local_backward_kernel,
        use_gate_cache_backward_kernel,
        use_persistent_backward_kernel,
        use_persistent_state_backward_kernel,
        use_persistent_state_local_backward_kernel,
        use_persistent_state4_backward_kernel,
        use_persistent_state8_backward_kernel,
        use_persistent_state16_backward_kernel,
        use_persistent_state16_gate_cache_backward_kernel,
        use_persistent_state16_gate_cache_tiled_backward_kernel,
        use_persistent_state16_grad_coeff_cache_tiled_backward_kernel,
        use_persistent_state32_gate_cache_tiled_backward_kernel,
        use_gate_cache_parallel_update_forward_kernel,
        use_gate_cache_cta8_forward_kernel,
        use_gate_cache_cta6_forward_kernel,
        use_persistent_state16_global_gates_backward_kernel,
        use_persistent_state32_backward_kernel,
    )


class A100GRUH256(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 256,
        num_layers: int = 1,
        batch_first: bool = True,
        block_threads: int = 704,
        use_recurrent_backward_kernel: bool = False,
        recompute_hidden_gates: bool = False,
        use_tiled_recurrent_backward_kernel: bool = False,
        use_split_recurrent_backward_kernel: bool = False,
        split_recurrent_count: int = 8,
        use_cooperative_split_backward_kernel: bool = False,
        cooperative_split_count: int = 4,
        use_cooperative_split_cached_backward_kernel: bool = False,
        use_cooperative_split2_backward_kernel: bool = False,
        use_cooperative_split2_cached_local_backward_kernel: bool = False,
        use_gate_cache_backward_kernel: bool = False,
        use_persistent_backward_kernel: bool = False,
        use_persistent_state_backward_kernel: bool = False,
        use_persistent_state_local_backward_kernel: bool = False,
        use_persistent_state4_backward_kernel: bool = False,
        use_persistent_state8_backward_kernel: bool = False,
        use_persistent_state16_backward_kernel: bool = False,
        use_persistent_state16_gate_cache_backward_kernel: bool = False,
        use_persistent_state16_gate_cache_tiled_backward_kernel: bool = False,
        use_persistent_state16_grad_coeff_cache_tiled_backward_kernel: bool = False,
        use_persistent_state32_gate_cache_tiled_backward_kernel: bool = False,
        use_gate_cache_parallel_update_forward_kernel: bool = False,
        use_gate_cache_cta8_forward_kernel: bool = False,
        use_gate_cache_cta6_forward_kernel: bool = False,
        use_persistent_state16_global_gates_backward_kernel: bool = False,
        use_persistent_state32_backward_kernel: bool = False,
    ) -> None:
        super().__init__()
        if hidden_size != 256:
            raise ValueError("A100GRUH256 only supports hidden_size=256.")
        if num_layers != 1:
            raise ValueError("A100GRUH256 only supports num_layers=1.")
        if not batch_first:
            raise ValueError("A100GRUH256 only supports batch_first=True.")

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.block_threads = block_threads
        self.use_recurrent_backward_kernel = use_recurrent_backward_kernel
        self.recompute_hidden_gates = recompute_hidden_gates
        self.use_tiled_recurrent_backward_kernel = use_tiled_recurrent_backward_kernel
        self.use_split_recurrent_backward_kernel = use_split_recurrent_backward_kernel
        self.split_recurrent_count = split_recurrent_count
        self.use_cooperative_split_backward_kernel = use_cooperative_split_backward_kernel
        self.cooperative_split_count = cooperative_split_count
        self.use_cooperative_split_cached_backward_kernel = (
            use_cooperative_split_cached_backward_kernel
        )
        self.use_cooperative_split2_backward_kernel = use_cooperative_split2_backward_kernel
        self.use_cooperative_split2_cached_local_backward_kernel = (
            use_cooperative_split2_cached_local_backward_kernel
        )
        self.use_gate_cache_backward_kernel = use_gate_cache_backward_kernel
        self.use_persistent_backward_kernel = use_persistent_backward_kernel
        self.use_persistent_state_backward_kernel = use_persistent_state_backward_kernel
        self.use_persistent_state_local_backward_kernel = (
            use_persistent_state_local_backward_kernel
        )
        self.use_persistent_state4_backward_kernel = use_persistent_state4_backward_kernel
        self.use_persistent_state8_backward_kernel = use_persistent_state8_backward_kernel
        self.use_persistent_state16_backward_kernel = use_persistent_state16_backward_kernel
        self.use_persistent_state16_gate_cache_backward_kernel = (
            use_persistent_state16_gate_cache_backward_kernel
        )
        self.use_persistent_state16_gate_cache_tiled_backward_kernel = (
            use_persistent_state16_gate_cache_tiled_backward_kernel
        )
        self.use_persistent_state16_grad_coeff_cache_tiled_backward_kernel = (
            use_persistent_state16_grad_coeff_cache_tiled_backward_kernel
        )
        self.use_persistent_state32_gate_cache_tiled_backward_kernel = (
            use_persistent_state32_gate_cache_tiled_backward_kernel
        )
        self.use_gate_cache_parallel_update_forward_kernel = (
            use_gate_cache_parallel_update_forward_kernel
        )
        self.use_gate_cache_cta8_forward_kernel = use_gate_cache_cta8_forward_kernel
        self.use_gate_cache_cta6_forward_kernel = use_gate_cache_cta6_forward_kernel
        self.use_persistent_state16_global_gates_backward_kernel = (
            use_persistent_state16_global_gates_backward_kernel
        )
        self.use_persistent_state32_backward_kernel = use_persistent_state32_backward_kernel

        self.weight_ih_l0 = nn.Parameter(torch.empty(3 * hidden_size, input_size))
        self.weight_hh_l0 = nn.Parameter(torch.empty(3 * hidden_size, hidden_size))
        self.bias_ih_l0 = nn.Parameter(torch.empty(3 * hidden_size))
        self.bias_hh_l0 = nn.Parameter(torch.empty(3 * hidden_size))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            nn.init.uniform_(weight, -stdv, stdv)

    def forward(
        self,
        x: torch.Tensor,
        hx: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if hx is None:
            h0 = x.new_zeros(x.size(0), self.hidden_size)
        else:
            if hx.shape != (1, x.size(0), self.hidden_size):
                raise ValueError("hx must have shape [1, batch, 256].")
            h0 = hx[0]

        return a100_gru_h256(
            x,
            h0,
            self.weight_ih_l0,
            self.weight_hh_l0,
            self.bias_ih_l0,
            self.bias_hh_l0,
            block_threads=self.block_threads,
            use_recurrent_backward_kernel=self.use_recurrent_backward_kernel,
            recompute_hidden_gates=self.recompute_hidden_gates,
            use_tiled_recurrent_backward_kernel=self.use_tiled_recurrent_backward_kernel,
            use_split_recurrent_backward_kernel=self.use_split_recurrent_backward_kernel,
            split_recurrent_count=self.split_recurrent_count,
            use_cooperative_split_backward_kernel=self.use_cooperative_split_backward_kernel,
            cooperative_split_count=self.cooperative_split_count,
            use_cooperative_split_cached_backward_kernel=(
                self.use_cooperative_split_cached_backward_kernel
            ),
            use_cooperative_split2_backward_kernel=self.use_cooperative_split2_backward_kernel,
            use_cooperative_split2_cached_local_backward_kernel=(
                self.use_cooperative_split2_cached_local_backward_kernel
            ),
            use_gate_cache_backward_kernel=self.use_gate_cache_backward_kernel,
            use_persistent_backward_kernel=self.use_persistent_backward_kernel,
            use_persistent_state_backward_kernel=self.use_persistent_state_backward_kernel,
            use_persistent_state_local_backward_kernel=(
                self.use_persistent_state_local_backward_kernel
            ),
            use_persistent_state4_backward_kernel=self.use_persistent_state4_backward_kernel,
            use_persistent_state8_backward_kernel=self.use_persistent_state8_backward_kernel,
            use_persistent_state16_backward_kernel=(
                self.use_persistent_state16_backward_kernel
            ),
            use_persistent_state16_gate_cache_backward_kernel=(
                self.use_persistent_state16_gate_cache_backward_kernel
            ),
            use_persistent_state16_gate_cache_tiled_backward_kernel=(
                self.use_persistent_state16_gate_cache_tiled_backward_kernel
            ),
            use_persistent_state16_grad_coeff_cache_tiled_backward_kernel=(
                self.use_persistent_state16_grad_coeff_cache_tiled_backward_kernel
            ),
            use_persistent_state32_gate_cache_tiled_backward_kernel=(
                self.use_persistent_state32_gate_cache_tiled_backward_kernel
            ),
            use_gate_cache_parallel_update_forward_kernel=(
                self.use_gate_cache_parallel_update_forward_kernel
            ),
            use_gate_cache_cta8_forward_kernel=self.use_gate_cache_cta8_forward_kernel,
            use_gate_cache_cta6_forward_kernel=self.use_gate_cache_cta6_forward_kernel,
            use_persistent_state16_global_gates_backward_kernel=(
                self.use_persistent_state16_global_gates_backward_kernel
            ),
            use_persistent_state32_backward_kernel=(
                self.use_persistent_state32_backward_kernel
            ),
        )


def copy_from_torch_gru(a100_gru: A100GRUH256, torch_gru: nn.GRU) -> None:
    if torch_gru.bidirectional:
        raise ValueError("Bidirectional GRU is not supported.")
    if not torch_gru.batch_first:
        raise ValueError("Only batch_first=True GRU is supported.")
    if torch_gru.num_layers != 1:
        raise ValueError("Only num_layers=1 GRU is supported.")
    if torch_gru.hidden_size != 256:
        raise ValueError("Only hidden_size=256 GRU is supported.")
    if torch_gru.input_size != a100_gru.input_size:
        raise ValueError("input_size mismatch.")

    with torch.no_grad():
        a100_gru.weight_ih_l0.copy_(torch_gru.weight_ih_l0)
        a100_gru.weight_hh_l0.copy_(torch_gru.weight_hh_l0)
        a100_gru.bias_ih_l0.copy_(torch_gru.bias_ih_l0)
        a100_gru.bias_hh_l0.copy_(torch_gru.bias_hh_l0)
