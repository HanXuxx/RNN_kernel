from __future__ import annotations

import ctypes
import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .gru_forward import (
    _check_cuda_error,
    _device_attribute,
    _get_a100_kernel,
    a100_gru_forward_from_gates_cooperative_h256_cta8_shmem_gate_cache,
    a100_gru_forward_from_gates_cooperative_h256_cta6_shmem_gate_cache,
    a100_gru_forward_from_gates_cooperative_h256_htile2_shmem_gate_cache,
    a100_gru_forward_from_gates_cooperative_h256_htile4_compact_shmem_gate_cache,
    a100_gru_forward_from_gates_cooperative_h256_htile4_compact_hoist_shmem_gate_cache,
    a100_gru_forward_from_gates_cooperative_h256_htile4_compact_hoist_qwarp_gate_cache,
    a100_gru_forward_from_gates_cooperative_h256_htile4_compact_hoist_row3_shmem_gate_cache,
    a100_gru_forward_from_gates_cooperative_h256_htile4_compact_hoist_row4_hidden_shmem_gate_cache,
    a100_gru_forward_from_gates_cooperative_h256_htile4_compact_hoist_row4_k1_shmem_gate_cache,
    a100_gru_forward_from_gates_cooperative_h256_htile4_compact_hoist_row4_ldg_shmem_gate_cache,
    a100_gru_forward_from_gates_cooperative_h256_htile4_compact_hoist_row4_no_cache,
    a100_gru_forward_from_gates_cooperative_h256_htile4_compact_hoist_row4_no_cache_k1,
    a100_gru_forward_from_gates_cooperative_h256_htile4_compact_hoist_row4_no_cache_k2,
    a100_gru_forward_from_gates_cooperative_h256_htile4_compact_hoist_row4_no_cache_ldg,
    a100_gru_forward_from_gates_cooperative_h256_htile4_compact_hoist_row4_parallel_update_gate_cache,
    a100_gru_forward_from_gates_cooperative_h256_htile4_compact_hoist_row4_prev_cache_shmem_gate_cache,
    a100_gru_forward_from_gates_cooperative_h256_htile4_compact_hoist_row4_shmem_gate_cache,
    a100_gru_forward_from_gates_cooperative_h256_htile4_compact_hoist_row4_weight_shmem_gate_cache,
    a100_gru_forward_from_gates_cooperative_h256_htile4_shmem_gate_cache,
    a100_gru_forward_from_gates_cooperative_h256_htile8_compact_hoist_k1_no_cache,
    a100_gru_forward_from_gates_cooperative_h256_htile8_compact_hoist_k1_shmem_gate_cache,
    a100_gru_forward_from_gates_cooperative_h256_htile8_compact_shmem_gate_cache,
    a100_gru_forward_from_gates_cooperative_h256_parallel_update_gate_cache,
    a100_gru_forward_from_gates_cooperative_h256_shmem,
    a100_gru_forward_from_gates_cooperative_h256_shmem_gate_cache,
    a100_gru_forward_from_gates_cooperative_h256_shmem_grad_coeff_cache,
    a100_gru_h256_stacked_backward_naive,
    a100_gru_h256_stacked_backward_split4,
    a100_gru_h256_stacked_backward_split4_group8,
    a100_gru_h256_stacked_backward_split4_shmem,
    a100_gru_h256_stacked_backward_split6_weight_shmem,
    a100_gru_h256_stacked_backward_split8,
    a100_gru_h256_stacked_row4_forward,
    a100_gru_h256_stacked_row4_k1_train_forward,
    a100_gru_h256_stacked_row4_train_forward,
    a100_gru_h256_stacked_forward_naive,
    cuda,
)


MAX_NUM_LAYERS = 4
MAX_INPUT_SIZE = 16


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


def _a100_gru_h256_backward_sequence_cooperative_split16_gate_cache_state_tiled_weight_shmem(
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
            "A100 h256 persistent split16 weight-shmem gate-cache backward requires hidden_size=256."
        )
    if block_threads != 256:
        raise ValueError(
            "A100 h256 persistent split16 weight-shmem gate-cache backward requires 256 threads."
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
    function = (
        compiled.h256_backward_sequence_cooperative_split16_gate_cache_state_tiled_weight_shmem_function
    )
    stream = cuda.CUstream(torch.cuda.current_stream(device=gate_cache.device).cuda_stream)
    shared_floats = 48 + 48 * hidden_size
    shared_bytes = shared_floats * ctypes.sizeof(ctypes.c_float)
    # 该分支每个 block 需要约 49KB 动态 shared memory，需要显式打开 opt-in 上限。
    err, = cuda.cuFuncSetAttribute(
        function,
        cuda.CUfunction_attribute.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
        shared_bytes,
    )
    _check_cuda_error(err)
    err, = cuda.cuFuncSetAttribute(
        function,
        cuda.CUfunction_attribute.CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT,
        100,
    )
    _check_cuda_error(err)
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
        function,
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


def _a100_gru_h256_backward_sequence_cooperative_split16_gate_cache_state_tiled_weight_shmem_split0_keep(
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
            "A100 h256 persistent split16 split0-keep weight-shmem gate-cache backward requires hidden_size=256."
        )
    if block_threads != 256:
        raise ValueError(
            "A100 h256 persistent split16 split0-keep weight-shmem gate-cache backward requires 256 threads."
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
    function = (
        compiled.h256_backward_sequence_cooperative_split16_gate_cache_state_tiled_weight_shmem_split0_keep_function
    )
    stream = cuda.CUstream(torch.cuda.current_stream(device=gate_cache.device).cuda_stream)
    shared_floats = 48 + 48 * hidden_size
    shared_bytes = shared_floats * ctypes.sizeof(ctypes.c_float)
    # 该实验分支沿用 weight-shmem 的 49KB 动态 shared memory 配置。
    err, = cuda.cuFuncSetAttribute(
        function,
        cuda.CUfunction_attribute.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
        shared_bytes,
    )
    _check_cuda_error(err)
    err, = cuda.cuFuncSetAttribute(
        function,
        cuda.CUfunction_attribute.CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT,
        100,
    )
    _check_cuda_error(err)
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
        function,
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


def _a100_gru_h256_backward_sequence_cooperative_split12_gate_cache_state_tiled_weight_shmem_split0_keep(
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
            "A100 h256 persistent split12 split0-keep weight-shmem gate-cache backward requires hidden_size=256."
        )
    if block_threads != 256:
        raise ValueError(
            "A100 h256 persistent split12 split0-keep weight-shmem gate-cache backward requires 256 threads."
        )
    if grad_hidden_gates_steps.shape != (seq_len, batch_size, 3 * hidden_size):
        raise ValueError("grad_hidden_gates_steps must have shape [seq, batch, 3 * hidden].")
    if partial_sums.shape != (batch_size, 12, hidden_size):
        raise ValueError("partial_sums must have shape [batch, 12, hidden].")
    if not gate_cache.is_contiguous():
        raise ValueError("gate_cache must be contiguous.")
    if not grad_hidden_gates_steps.is_contiguous() or not partial_sums.is_contiguous():
        raise ValueError("grad_hidden_gates_steps and partial_sums must be contiguous.")

    grad_output = grad_output.contiguous()
    grad_hidden_state = grad_hidden_state.contiguous()
    gate_cache = gate_cache.contiguous()
    weight_hh = weight_hh.contiguous()

    compiled = _get_a100_kernel()
    function = (
        compiled.h256_backward_sequence_cooperative_split12_gate_cache_state_tiled_weight_shmem_split0_keep_function
    )
    stream = cuda.CUstream(torch.cuda.current_stream(device=gate_cache.device).cuda_stream)
    shared_floats = 64 + 64 * hidden_size
    shared_bytes = shared_floats * ctypes.sizeof(ctypes.c_float)
    # split12 每个 split 缓存 64x256 recurrent weight tile，约 65.8KB 动态 shared memory。
    err, = cuda.cuFuncSetAttribute(
        function,
        cuda.CUfunction_attribute.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
        shared_bytes,
    )
    _check_cuda_error(err)
    err, = cuda.cuFuncSetAttribute(
        function,
        cuda.CUfunction_attribute.CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT,
        100,
    )
    _check_cuda_error(err)
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
        function,
        batch_size * 12,
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


def _a100_gru_h256_backward_sequence_cooperative_split6_gate_cache_state_tiled_weight_shmem_split0_keep(
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
            "A100 h256 persistent split6 split0-keep weight-shmem gate-cache backward requires hidden_size=256."
        )
    if block_threads != 256:
        raise ValueError(
            "A100 h256 persistent split6 split0-keep weight-shmem gate-cache backward requires 256 threads."
        )
    if grad_hidden_gates_steps.shape != (seq_len, batch_size, 3 * hidden_size):
        raise ValueError("grad_hidden_gates_steps must have shape [seq, batch, 3 * hidden].")
    if partial_sums.shape != (batch_size, 6, hidden_size):
        raise ValueError("partial_sums must have shape [batch, 6, hidden].")
    if not gate_cache.is_contiguous():
        raise ValueError("gate_cache must be contiguous.")
    if not grad_hidden_gates_steps.is_contiguous() or not partial_sums.is_contiguous():
        raise ValueError("grad_hidden_gates_steps and partial_sums must be contiguous.")

    grad_output = grad_output.contiguous()
    grad_hidden_state = grad_hidden_state.contiguous()
    gate_cache = gate_cache.contiguous()
    weight_hh = weight_hh.contiguous()

    compiled = _get_a100_kernel()
    function = (
        compiled.h256_backward_sequence_cooperative_split6_gate_cache_state_tiled_weight_shmem_split0_keep_function
    )
    stream = cuda.CUstream(torch.cuda.current_stream(device=gate_cache.device).cuda_stream)
    shared_floats = 128 + 128 * hidden_size
    shared_bytes = shared_floats * ctypes.sizeof(ctypes.c_float)
    # split6 每个 split 缓存 128x256 recurrent weight tile，接近 A100 单 block shared 上限。
    err, = cuda.cuFuncSetAttribute(
        function,
        cuda.CUfunction_attribute.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
        shared_bytes,
    )
    _check_cuda_error(err)
    err, = cuda.cuFuncSetAttribute(
        function,
        cuda.CUfunction_attribute.CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT,
        100,
    )
    _check_cuda_error(err)
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
        function,
        batch_size * 6,
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


def _a100_gru_h256_backward_sequence_cooperative_split6_gate_cache_state_tiled_weight_shmem_split0_keep_unroll8(
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
            "A100 h256 persistent split6 unroll8 split0-keep weight-shmem gate-cache backward requires hidden_size=256."
        )
    if block_threads != 256:
        raise ValueError(
            "A100 h256 persistent split6 unroll8 split0-keep weight-shmem gate-cache backward requires 256 threads."
        )
    if grad_hidden_gates_steps.shape != (seq_len, batch_size, 3 * hidden_size):
        raise ValueError("grad_hidden_gates_steps must have shape [seq, batch, 3 * hidden].")
    if partial_sums.shape != (batch_size, 6, hidden_size):
        raise ValueError("partial_sums must have shape [batch, 6, hidden].")
    if not gate_cache.is_contiguous():
        raise ValueError("gate_cache must be contiguous.")
    if not grad_hidden_gates_steps.is_contiguous() or not partial_sums.is_contiguous():
        raise ValueError("grad_hidden_gates_steps and partial_sums must be contiguous.")

    grad_output = grad_output.contiguous()
    grad_hidden_state = grad_hidden_state.contiguous()
    gate_cache = gate_cache.contiguous()
    weight_hh = weight_hh.contiguous()

    compiled = _get_a100_kernel()
    function = (
        compiled.h256_backward_sequence_cooperative_split6_gate_cache_state_tiled_weight_shmem_split0_keep_unroll8_function
    )
    stream = cuda.CUstream(torch.cuda.current_stream(device=gate_cache.device).cuda_stream)
    shared_floats = 128 + 128 * hidden_size
    shared_bytes = shared_floats * ctypes.sizeof(ctypes.c_float)
    # split6_unroll8 与默认 split6 使用相同 shared memory，只改变循环展开因子。
    err, = cuda.cuFuncSetAttribute(
        function,
        cuda.CUfunction_attribute.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
        shared_bytes,
    )
    _check_cuda_error(err)
    err, = cuda.cuFuncSetAttribute(
        function,
        cuda.CUfunction_attribute.CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT,
        100,
    )
    _check_cuda_error(err)
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
        function,
        batch_size * 6,
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


def _a100_gru_h256_backward_sequence_cooperative_split8_gate_cache_state_tiled_weight_shmem_split0_keep(
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
            "A100 h256 persistent split8 split0-keep weight-shmem gate-cache backward requires hidden_size=256."
        )
    if block_threads != 256:
        raise ValueError(
            "A100 h256 persistent split8 split0-keep weight-shmem gate-cache backward requires 256 threads."
        )
    if grad_hidden_gates_steps.shape != (seq_len, batch_size, 3 * hidden_size):
        raise ValueError("grad_hidden_gates_steps must have shape [seq, batch, 3 * hidden].")
    if partial_sums.shape != (batch_size, 8, hidden_size):
        raise ValueError("partial_sums must have shape [batch, 8, hidden].")
    if not gate_cache.is_contiguous():
        raise ValueError("gate_cache must be contiguous.")
    if not grad_hidden_gates_steps.is_contiguous() or not partial_sums.is_contiguous():
        raise ValueError("grad_hidden_gates_steps and partial_sums must be contiguous.")

    grad_output = grad_output.contiguous()
    grad_hidden_state = grad_hidden_state.contiguous()
    gate_cache = gate_cache.contiguous()
    weight_hh = weight_hh.contiguous()

    compiled = _get_a100_kernel()
    function = (
        compiled.h256_backward_sequence_cooperative_split8_gate_cache_state_tiled_weight_shmem_split0_keep_function
    )
    stream = cuda.CUstream(torch.cuda.current_stream(device=gate_cache.device).cuda_stream)
    shared_floats = 96 + 96 * hidden_size
    shared_bytes = shared_floats * ctypes.sizeof(ctypes.c_float)
    # split8 缓存 96x256 recurrent weight tile，用更小 shared memory 换更多 CTA。
    err, = cuda.cuFuncSetAttribute(
        function,
        cuda.CUfunction_attribute.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
        shared_bytes,
    )
    _check_cuda_error(err)
    err, = cuda.cuFuncSetAttribute(
        function,
        cuda.CUfunction_attribute.CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT,
        100,
    )
    _check_cuda_error(err)
    err, active_blocks_per_sm = cuda.cuOccupancyMaxActiveBlocksPerMultiprocessor(
        function,
        block_threads,
        shared_bytes,
    )
    _check_cuda_error(err)
    device_index = gate_cache.device.index
    if device_index is None:
        device_index = torch.cuda.current_device()
    sm_count = _device_attribute(
        device_index,
        cuda.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT,
    )
    max_cooperative_blocks = active_blocks_per_sm * sm_count
    grid_blocks = batch_size * 8
    if grid_blocks > max_cooperative_blocks:
        max_batch_per_launch = max_cooperative_blocks // 8
        if max_batch_per_launch < 1:
            raise ValueError(
                "split8 backward cooperative grid is too large for resident launch: "
                f"grid_blocks={grid_blocks}, max={max_cooperative_blocks}."
            )
        # grad_hidden_gates_steps 是 [seq,batch,gate]，按 batch 分块时需要临时 contiguous buffer。
        for batch_start in range(0, batch_size, max_batch_per_launch):
            batch_end = min(batch_start + max_batch_per_launch, batch_size)
            grad_hidden_gates_chunk = torch.empty(
                seq_len,
                batch_end - batch_start,
                3 * hidden_size,
                device=gate_cache.device,
                dtype=gate_cache.dtype,
            )
            grad_hidden_state_chunk = grad_hidden_state[batch_start:batch_end].contiguous()
            _a100_gru_h256_backward_sequence_cooperative_split8_gate_cache_state_tiled_weight_shmem_split0_keep(
                grad_output[batch_start:batch_end],
                grad_hidden_state_chunk,
                gate_cache[batch_start:batch_end],
                h0[batch_start:batch_end],
                output[batch_start:batch_end],
                weight_hh,
                grad_input_gates[batch_start:batch_end],
                grad_hidden_gates_chunk,
                partial_sums[batch_start:batch_end],
                block_threads=block_threads,
            )
            grad_hidden_state[batch_start:batch_end].copy_(grad_hidden_state_chunk)
            grad_hidden_gates_steps[:, batch_start:batch_end, :].copy_(grad_hidden_gates_chunk)
        return grad_hidden_state
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
        function,
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


def _a100_gru_h256_pack_hidden_prev_time_major(
    h0: torch.Tensor,
    output: torch.Tensor,
    block_threads: int = 256,
) -> torch.Tensor:
    batch_size, seq_len, hidden_size = output.shape
    if hidden_size != 256:
        raise ValueError("A100 h256 hidden-prev pack requires hidden_size=256.")
    if h0.shape != (batch_size, hidden_size):
        raise ValueError("h0 must have shape [batch, hidden].")
    if block_threads != 256:
        raise ValueError("A100 h256 hidden-prev pack currently requires 256 threads.")

    h0 = h0.contiguous()
    output = output.contiguous()
    hidden_prev_steps = torch.empty(
        seq_len,
        batch_size,
        hidden_size,
        device=output.device,
        dtype=output.dtype,
    )

    compiled = _get_a100_kernel()
    stream = cuda.CUstream(torch.cuda.current_stream(device=output.device).cuda_stream)
    total_vec4 = seq_len * batch_size * (hidden_size // 4)
    grid_blocks = min(math.ceil(total_vec4 / block_threads), 65535)
    values = (
        ctypes.c_void_p(h0.data_ptr()),
        ctypes.c_void_p(output.data_ptr()),
        ctypes.c_void_p(hidden_prev_steps.data_ptr()),
        ctypes.c_int(batch_size),
        ctypes.c_int(seq_len),
    )
    types = (
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_int,
    )
    err, = cuda.cuLaunchKernel(
        compiled.h256_pack_hidden_prev_time_major_function,
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
    return hidden_prev_steps


def _a100_gru_h256_backward_sequence_cooperative_split5_gate_cache_state_tiled_weight_shmem_split0_keep(
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
            "A100 h256 persistent split5 split0-keep weight-shmem gate-cache backward requires hidden_size=256."
        )
    if block_threads != 256:
        raise ValueError(
            "A100 h256 persistent split5 split0-keep weight-shmem gate-cache backward requires 256 threads."
        )
    if grad_hidden_gates_steps.shape != (seq_len, batch_size, 3 * hidden_size):
        raise ValueError("grad_hidden_gates_steps must have shape [seq, batch, 3 * hidden].")
    if partial_sums.shape != (batch_size, 5, hidden_size):
        raise ValueError("partial_sums must have shape [batch, 5, hidden].")
    if not gate_cache.is_contiguous():
        raise ValueError("gate_cache must be contiguous.")
    if not grad_hidden_gates_steps.is_contiguous() or not partial_sums.is_contiguous():
        raise ValueError("grad_hidden_gates_steps and partial_sums must be contiguous.")

    grad_output = grad_output.contiguous()
    grad_hidden_state = grad_hidden_state.contiguous()
    gate_cache = gate_cache.contiguous()
    weight_hh = weight_hh.contiguous()

    compiled = _get_a100_kernel()
    function = (
        compiled.h256_backward_sequence_cooperative_split5_gate_cache_state_tiled_weight_shmem_split0_keep_function
    )
    stream = cuda.CUstream(torch.cuda.current_stream(device=gate_cache.device).cuda_stream)
    split_size = (3 * hidden_size + 5 - 1) // 5
    shared_floats = split_size + split_size * hidden_size
    shared_bytes = shared_floats * ctypes.sizeof(ctypes.c_float)
    # split5 使用接近 A100 单 block 上限的 dynamic shared memory，减少 backward partial 路数。
    err, = cuda.cuFuncSetAttribute(
        function,
        cuda.CUfunction_attribute.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
        shared_bytes,
    )
    _check_cuda_error(err)
    err, = cuda.cuFuncSetAttribute(
        function,
        cuda.CUfunction_attribute.CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT,
        100,
    )
    _check_cuda_error(err)
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
        function,
        batch_size * 5,
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


def _a100_gru_h256_backward_sequence_cooperative_split24_gate_cache_state_tiled_weight_shmem_split0_keep(
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
            "A100 h256 persistent split24 split0-keep weight-shmem gate-cache backward requires hidden_size=256."
        )
    if block_threads != 256:
        raise ValueError(
            "A100 h256 persistent split24 split0-keep weight-shmem gate-cache backward requires 256 threads."
        )
    if grad_hidden_gates_steps.shape != (seq_len, batch_size, 3 * hidden_size):
        raise ValueError("grad_hidden_gates_steps must have shape [seq, batch, 3 * hidden].")
    if partial_sums.shape != (batch_size, 24, hidden_size):
        raise ValueError("partial_sums must have shape [batch, 24, hidden].")
    if not gate_cache.is_contiguous():
        raise ValueError("gate_cache must be contiguous.")
    if not grad_hidden_gates_steps.is_contiguous() or not partial_sums.is_contiguous():
        raise ValueError("grad_hidden_gates_steps and partial_sums must be contiguous.")

    grad_output = grad_output.contiguous()
    grad_hidden_state = grad_hidden_state.contiguous()
    gate_cache = gate_cache.contiguous()
    weight_hh = weight_hh.contiguous()

    compiled = _get_a100_kernel()
    function = (
        compiled.h256_backward_sequence_cooperative_split24_gate_cache_state_tiled_weight_shmem_split0_keep_function
    )
    stream = cuda.CUstream(torch.cuda.current_stream(device=gate_cache.device).cuda_stream)
    shared_floats = 32 + 32 * hidden_size
    shared_bytes = shared_floats * ctypes.sizeof(ctypes.c_float)
    # split24 每个 split 缓存 32x256 recurrent weight tile，减少 shared memory 压力。
    err, = cuda.cuFuncSetAttribute(
        function,
        cuda.CUfunction_attribute.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
        shared_bytes,
    )
    _check_cuda_error(err)
    err, = cuda.cuFuncSetAttribute(
        function,
        cuda.CUfunction_attribute.CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT,
        100,
    )
    _check_cuda_error(err)
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
        function,
        batch_size * 24,
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


def _a100_gru_h256_backward_sequence_cooperative_split12_gate_cache_state_tiled_weight_shmem_split0_keep_unroll8(
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
            "A100 h256 persistent split12 split0-keep unroll8 weight-shmem gate-cache backward requires hidden_size=256."
        )
    if block_threads != 256:
        raise ValueError(
            "A100 h256 persistent split12 split0-keep unroll8 weight-shmem gate-cache backward requires 256 threads."
        )
    if grad_hidden_gates_steps.shape != (seq_len, batch_size, 3 * hidden_size):
        raise ValueError("grad_hidden_gates_steps must have shape [seq, batch, 3 * hidden].")
    if partial_sums.shape != (batch_size, 12, hidden_size):
        raise ValueError("partial_sums must have shape [batch, 12, hidden].")
    if not gate_cache.is_contiguous():
        raise ValueError("gate_cache must be contiguous.")
    if not grad_hidden_gates_steps.is_contiguous() or not partial_sums.is_contiguous():
        raise ValueError("grad_hidden_gates_steps and partial_sums must be contiguous.")

    grad_output = grad_output.contiguous()
    grad_hidden_state = grad_hidden_state.contiguous()
    gate_cache = gate_cache.contiguous()
    weight_hh = weight_hh.contiguous()

    compiled = _get_a100_kernel()
    function = (
        compiled.h256_backward_sequence_cooperative_split12_gate_cache_state_tiled_weight_shmem_split0_keep_unroll8_function
    )
    stream = cuda.CUstream(torch.cuda.current_stream(device=gate_cache.device).cuda_stream)
    shared_floats = 64 + 64 * hidden_size
    shared_bytes = shared_floats * ctypes.sizeof(ctypes.c_float)
    # 与 split12 主线共享 shared-memory 规模，仅改变 CUDA dot loop 展开因子。
    err, = cuda.cuFuncSetAttribute(
        function,
        cuda.CUfunction_attribute.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
        shared_bytes,
    )
    _check_cuda_error(err)
    err, = cuda.cuFuncSetAttribute(
        function,
        cuda.CUfunction_attribute.CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT,
        100,
    )
    _check_cuda_error(err)
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
        function,
        batch_size * 12,
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


def _a100_gru_h256_backward_sequence_cooperative_split16_gate_cache_state_tiled_weight_shmem_split0_keep_own_shmem(
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
            "A100 h256 persistent split16 split0-keep own-shmem weight-shmem gate-cache backward requires hidden_size=256."
        )
    if block_threads != 256:
        raise ValueError(
            "A100 h256 persistent split16 split0-keep own-shmem weight-shmem gate-cache backward requires 256 threads."
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
    function = (
        compiled.h256_backward_sequence_cooperative_split16_gate_cache_state_tiled_weight_shmem_split0_keep_own_shmem_function
    )
    stream = cuda.CUstream(torch.cuda.current_stream(device=gate_cache.device).cuda_stream)
    shared_floats = 48 + hidden_size + 48 * hidden_size
    shared_bytes = shared_floats * ctypes.sizeof(ctypes.c_float)
    # 本分支额外用 256 个 float shared memory 保存本 split 上一轮 partial。
    err, = cuda.cuFuncSetAttribute(
        function,
        cuda.CUfunction_attribute.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
        shared_bytes,
    )
    _check_cuda_error(err)
    err, = cuda.cuFuncSetAttribute(
        function,
        cuda.CUfunction_attribute.CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT,
        100,
    )
    _check_cuda_error(err)
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
        function,
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


def _a100_gru_h256_backward_sequence_cooperative_split8_gate_cache_state_tiled(
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
            "A100 h256 persistent split8 tiled gate-cache backward requires hidden_size=256."
        )
    if block_threads != 256:
        raise ValueError(
            "A100 h256 persistent split8 tiled gate-cache backward requires 256 threads."
        )
    if grad_hidden_gates_steps.shape != (seq_len, batch_size, 3 * hidden_size):
        raise ValueError("grad_hidden_gates_steps must have shape [seq, batch, 3 * hidden].")
    if partial_sums.shape != (batch_size, 8, hidden_size):
        raise ValueError("partial_sums must have shape [batch, 8, hidden].")
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
    shared_bytes = 96 * ctypes.sizeof(ctypes.c_float)
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
        compiled.h256_backward_sequence_cooperative_split8_gate_cache_state_tiled_function,
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
        use_persistent_state8_gate_cache_tiled_backward_kernel: bool = False,
        use_persistent_state16_gate_cache_tiled_backward_kernel: bool = False,
        use_persistent_state16_gate_cache_tiled_weight_shmem_backward_kernel: bool = False,
        use_persistent_state16_gate_cache_tiled_weight_shmem_split0_keep_backward_kernel: bool = False,
        use_persistent_state5_gate_cache_tiled_weight_shmem_split0_keep_backward_kernel: bool = False,
        use_persistent_state6_gate_cache_tiled_weight_shmem_split0_keep_backward_kernel: bool = False,
        use_persistent_state6_gate_cache_tiled_weight_shmem_split0_keep_unroll8_backward_kernel: bool = False,
        use_persistent_state8_gate_cache_tiled_weight_shmem_split0_keep_backward_kernel: bool = False,
        use_persistent_state12_gate_cache_tiled_weight_shmem_split0_keep_backward_kernel: bool = False,
        use_persistent_state12_gate_cache_tiled_weight_shmem_split0_keep_unroll8_backward_kernel: bool = False,
        use_persistent_state24_gate_cache_tiled_weight_shmem_split0_keep_backward_kernel: bool = False,
        use_persistent_state16_gate_cache_tiled_weight_shmem_split0_keep_own_shmem_backward_kernel: bool = False,
        use_persistent_state16_grad_coeff_cache_tiled_backward_kernel: bool = False,
        use_persistent_state32_gate_cache_tiled_backward_kernel: bool = False,
        use_gate_cache_parallel_update_forward_kernel: bool = False,
        use_gate_cache_cta8_forward_kernel: bool = False,
        use_gate_cache_cta6_forward_kernel: bool = False,
        use_gate_cache_htile2_forward_kernel: bool = False,
        use_gate_cache_htile4_forward_kernel: bool = False,
        use_gate_cache_htile4_compact_forward_kernel: bool = False,
        use_gate_cache_htile4_compact_hoist_forward_kernel: bool = False,
        use_gate_cache_htile4_compact_hoist_qwarp_forward_kernel: bool = False,
        use_gate_cache_htile4_compact_hoist_row3_forward_kernel: bool = False,
        use_gate_cache_htile4_compact_hoist_row4_forward_kernel: bool = False,
        use_gate_cache_htile4_compact_hoist_row4_k1_forward_kernel: bool = False,
        use_gate_cache_htile8_compact_hoist_k1_forward_kernel: bool = False,
        use_gate_cache_htile4_compact_hoist_row4_prev_cache_forward_kernel: bool = False,
        use_pack_hidden_prev_time_major_kernel: bool = False,
        use_gate_cache_htile4_compact_hoist_row4_parallel_update_forward_kernel: bool = False,
        use_gate_cache_htile4_compact_hoist_row4_ldg_forward_kernel: bool = False,
        use_gate_cache_htile4_compact_hoist_row4_hidden_shmem_forward_kernel: bool = False,
        use_gate_cache_htile4_compact_hoist_row4_weight_shmem_forward_kernel: bool = False,
        use_gate_cache_htile8_compact_forward_kernel: bool = False,
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

        use_persistent_state16_weight_shmem_family_backward_kernel = (
            use_persistent_state16_gate_cache_tiled_weight_shmem_backward_kernel
            or use_persistent_state16_gate_cache_tiled_weight_shmem_split0_keep_backward_kernel
            or use_persistent_state5_gate_cache_tiled_weight_shmem_split0_keep_backward_kernel
            or use_persistent_state6_gate_cache_tiled_weight_shmem_split0_keep_backward_kernel
            or use_persistent_state6_gate_cache_tiled_weight_shmem_split0_keep_unroll8_backward_kernel
            or use_persistent_state8_gate_cache_tiled_weight_shmem_split0_keep_backward_kernel
            or use_persistent_state12_gate_cache_tiled_weight_shmem_split0_keep_backward_kernel
            or use_persistent_state12_gate_cache_tiled_weight_shmem_split0_keep_unroll8_backward_kernel
            or use_persistent_state24_gate_cache_tiled_weight_shmem_split0_keep_backward_kernel
            or use_persistent_state16_gate_cache_tiled_weight_shmem_split0_keep_own_shmem_backward_kernel
        )

        input_gates = F.linear(
            x.reshape(batch_size * seq_len, input_size),
            weight_ih,
            bias_ih,
        ).view(batch_size, seq_len, 3 * hidden_size)
        hidden_prev_cache = x.new_empty(0)
        if (
            (
                use_persistent_state16_gate_cache_backward_kernel
                or use_persistent_state8_gate_cache_tiled_backward_kernel
                or use_persistent_state16_gate_cache_tiled_backward_kernel
                or use_persistent_state16_weight_shmem_family_backward_kernel
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
                or use_persistent_state8_gate_cache_tiled_backward_kernel
                or use_persistent_state16_gate_cache_tiled_backward_kernel
                or use_persistent_state16_weight_shmem_family_backward_kernel
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
                or use_persistent_state8_gate_cache_tiled_backward_kernel
                or use_persistent_state16_gate_cache_tiled_backward_kernel
                or use_persistent_state16_weight_shmem_family_backward_kernel
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
        elif (
            (
                use_persistent_state16_gate_cache_backward_kernel
                or use_persistent_state8_gate_cache_tiled_backward_kernel
                or use_persistent_state16_gate_cache_tiled_backward_kernel
                or use_persistent_state16_weight_shmem_family_backward_kernel
                or use_persistent_state32_gate_cache_tiled_backward_kernel
            )
            and use_gate_cache_htile2_forward_kernel
        ):
            output, gate_cache = a100_gru_forward_from_gates_cooperative_h256_htile2_shmem_gate_cache(
                input_gates,
                h0,
                weight_hh,
                bias_hh,
                block_threads=block_threads,
            )
        elif (
            (
                use_persistent_state16_gate_cache_backward_kernel
                or use_persistent_state8_gate_cache_tiled_backward_kernel
                or use_persistent_state16_gate_cache_tiled_backward_kernel
                or use_persistent_state16_weight_shmem_family_backward_kernel
                or use_persistent_state32_gate_cache_tiled_backward_kernel
            )
            and use_gate_cache_htile4_forward_kernel
        ):
            output, gate_cache = a100_gru_forward_from_gates_cooperative_h256_htile4_shmem_gate_cache(
                input_gates,
                h0,
                weight_hh,
                bias_hh,
                block_threads=block_threads,
            )
        elif (
            (
                use_persistent_state16_gate_cache_backward_kernel
                or use_persistent_state8_gate_cache_tiled_backward_kernel
                or use_persistent_state16_gate_cache_tiled_backward_kernel
                or use_persistent_state16_weight_shmem_family_backward_kernel
                or use_persistent_state32_gate_cache_tiled_backward_kernel
            )
            and use_gate_cache_htile4_compact_forward_kernel
        ):
            output, gate_cache = a100_gru_forward_from_gates_cooperative_h256_htile4_compact_shmem_gate_cache(
                input_gates,
                h0,
                weight_hh,
                bias_hh,
                block_threads=block_threads,
            )
        elif (
            (
                use_persistent_state16_gate_cache_backward_kernel
                or use_persistent_state8_gate_cache_tiled_backward_kernel
                or use_persistent_state16_gate_cache_tiled_backward_kernel
                or use_persistent_state16_weight_shmem_family_backward_kernel
                or use_persistent_state32_gate_cache_tiled_backward_kernel
            )
            and use_gate_cache_htile4_compact_hoist_forward_kernel
        ):
            output, gate_cache = a100_gru_forward_from_gates_cooperative_h256_htile4_compact_hoist_shmem_gate_cache(
                input_gates,
                h0,
                weight_hh,
                bias_hh,
                block_threads=block_threads,
            )
        elif (
            (
                use_persistent_state16_gate_cache_backward_kernel
                or use_persistent_state8_gate_cache_tiled_backward_kernel
                or use_persistent_state16_gate_cache_tiled_backward_kernel
                or use_persistent_state16_weight_shmem_family_backward_kernel
                or use_persistent_state32_gate_cache_tiled_backward_kernel
            )
            and use_gate_cache_htile8_compact_hoist_k1_forward_kernel
        ):
            output, gate_cache = (
                a100_gru_forward_from_gates_cooperative_h256_htile8_compact_hoist_k1_shmem_gate_cache(
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
                or use_persistent_state8_gate_cache_tiled_backward_kernel
                or use_persistent_state16_gate_cache_tiled_backward_kernel
                or use_persistent_state16_weight_shmem_family_backward_kernel
                or use_persistent_state32_gate_cache_tiled_backward_kernel
            )
            and use_gate_cache_htile4_compact_hoist_row4_k1_forward_kernel
        ):
            output, gate_cache = a100_gru_forward_from_gates_cooperative_h256_htile4_compact_hoist_row4_k1_shmem_gate_cache(
                input_gates,
                h0,
                weight_hh,
                bias_hh,
                block_threads=block_threads,
            )
        elif (
            (
                use_persistent_state16_gate_cache_backward_kernel
                or use_persistent_state8_gate_cache_tiled_backward_kernel
                or use_persistent_state16_gate_cache_tiled_backward_kernel
                or use_persistent_state16_weight_shmem_family_backward_kernel
                or use_persistent_state32_gate_cache_tiled_backward_kernel
            )
            and use_gate_cache_htile4_compact_hoist_row4_forward_kernel
        ):
            output, gate_cache = a100_gru_forward_from_gates_cooperative_h256_htile4_compact_hoist_row4_shmem_gate_cache(
                input_gates,
                h0,
                weight_hh,
                bias_hh,
                block_threads=block_threads,
            )
        elif (
            (
                use_persistent_state16_gate_cache_backward_kernel
                or use_persistent_state8_gate_cache_tiled_backward_kernel
                or use_persistent_state16_gate_cache_tiled_backward_kernel
                or use_persistent_state16_weight_shmem_family_backward_kernel
                or use_persistent_state32_gate_cache_tiled_backward_kernel
            )
            and use_gate_cache_htile4_compact_hoist_row4_prev_cache_forward_kernel
        ):
            output, gate_cache, hidden_prev_cache = (
                a100_gru_forward_from_gates_cooperative_h256_htile4_compact_hoist_row4_prev_cache_shmem_gate_cache(
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
                or use_persistent_state8_gate_cache_tiled_backward_kernel
                or use_persistent_state16_gate_cache_tiled_backward_kernel
                or use_persistent_state16_weight_shmem_family_backward_kernel
                or use_persistent_state32_gate_cache_tiled_backward_kernel
            )
            and use_gate_cache_htile4_compact_hoist_row4_parallel_update_forward_kernel
        ):
            output, gate_cache = a100_gru_forward_from_gates_cooperative_h256_htile4_compact_hoist_row4_parallel_update_gate_cache(
                input_gates,
                h0,
                weight_hh,
                bias_hh,
                block_threads=block_threads,
            )
        elif (
            (
                use_persistent_state16_gate_cache_backward_kernel
                or use_persistent_state8_gate_cache_tiled_backward_kernel
                or use_persistent_state16_gate_cache_tiled_backward_kernel
                or use_persistent_state16_weight_shmem_family_backward_kernel
                or use_persistent_state32_gate_cache_tiled_backward_kernel
            )
            and use_gate_cache_htile4_compact_hoist_row4_ldg_forward_kernel
        ):
            output, gate_cache = a100_gru_forward_from_gates_cooperative_h256_htile4_compact_hoist_row4_ldg_shmem_gate_cache(
                input_gates,
                h0,
                weight_hh,
                bias_hh,
                block_threads=block_threads,
            )
        elif (
            (
                use_persistent_state16_gate_cache_backward_kernel
                or use_persistent_state8_gate_cache_tiled_backward_kernel
                or use_persistent_state16_gate_cache_tiled_backward_kernel
                or use_persistent_state16_weight_shmem_family_backward_kernel
                or use_persistent_state32_gate_cache_tiled_backward_kernel
            )
            and use_gate_cache_htile4_compact_hoist_qwarp_forward_kernel
        ):
            output, gate_cache = a100_gru_forward_from_gates_cooperative_h256_htile4_compact_hoist_qwarp_gate_cache(
                input_gates,
                h0,
                weight_hh,
                bias_hh,
                block_threads=block_threads,
            )
        elif (
            (
                use_persistent_state16_gate_cache_backward_kernel
                or use_persistent_state8_gate_cache_tiled_backward_kernel
                or use_persistent_state16_gate_cache_tiled_backward_kernel
                or use_persistent_state16_weight_shmem_family_backward_kernel
                or use_persistent_state32_gate_cache_tiled_backward_kernel
            )
            and use_gate_cache_htile4_compact_hoist_row3_forward_kernel
        ):
            output, gate_cache = a100_gru_forward_from_gates_cooperative_h256_htile4_compact_hoist_row3_shmem_gate_cache(
                input_gates,
                h0,
                weight_hh,
                bias_hh,
                block_threads=block_threads,
            )
        elif (
            (
                use_persistent_state16_gate_cache_backward_kernel
                or use_persistent_state8_gate_cache_tiled_backward_kernel
                or use_persistent_state16_gate_cache_tiled_backward_kernel
                or use_persistent_state16_weight_shmem_family_backward_kernel
                or use_persistent_state32_gate_cache_tiled_backward_kernel
            )
            and use_gate_cache_htile4_compact_hoist_row4_hidden_shmem_forward_kernel
        ):
            output, gate_cache = a100_gru_forward_from_gates_cooperative_h256_htile4_compact_hoist_row4_hidden_shmem_gate_cache(
                input_gates,
                h0,
                weight_hh,
                bias_hh,
                block_threads=block_threads,
            )
        elif (
            (
                use_persistent_state16_gate_cache_backward_kernel
                or use_persistent_state8_gate_cache_tiled_backward_kernel
                or use_persistent_state16_gate_cache_tiled_backward_kernel
                or use_persistent_state16_weight_shmem_family_backward_kernel
                or use_persistent_state32_gate_cache_tiled_backward_kernel
            )
            and use_gate_cache_htile4_compact_hoist_row4_weight_shmem_forward_kernel
        ):
            output, gate_cache = a100_gru_forward_from_gates_cooperative_h256_htile4_compact_hoist_row4_weight_shmem_gate_cache(
                input_gates,
                h0,
                weight_hh,
                bias_hh,
                block_threads=block_threads,
            )
        elif (
            (
                use_persistent_state16_gate_cache_backward_kernel
                or use_persistent_state8_gate_cache_tiled_backward_kernel
                or use_persistent_state16_gate_cache_tiled_backward_kernel
                or use_persistent_state16_weight_shmem_family_backward_kernel
                or use_persistent_state32_gate_cache_tiled_backward_kernel
            )
            and use_gate_cache_htile8_compact_forward_kernel
        ):
            output, gate_cache = a100_gru_forward_from_gates_cooperative_h256_htile8_compact_shmem_gate_cache(
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
            or use_persistent_state8_gate_cache_tiled_backward_kernel
            or use_persistent_state16_gate_cache_tiled_backward_kernel
            or use_persistent_state16_weight_shmem_family_backward_kernel
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
        output_only = getattr(ctx, "output_only", False)
        h_n = None if output_only else output[:, -1, :].unsqueeze(0).contiguous()

        ctx.save_for_backward(
            x,
            h0,
            weight_ih,
            weight_hh,
            bias_hh,
            input_gates,
            output,
            gate_cache,
            hidden_prev_cache,
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
        ctx.use_persistent_state8_gate_cache_tiled_backward_kernel = (
            use_persistent_state8_gate_cache_tiled_backward_kernel
        )
        ctx.use_persistent_state16_gate_cache_tiled_backward_kernel = (
            use_persistent_state16_gate_cache_tiled_backward_kernel
        )
        ctx.use_persistent_state16_gate_cache_tiled_weight_shmem_backward_kernel = (
            use_persistent_state16_gate_cache_tiled_weight_shmem_backward_kernel
        )
        ctx.use_persistent_state16_gate_cache_tiled_weight_shmem_split0_keep_backward_kernel = (
            use_persistent_state16_gate_cache_tiled_weight_shmem_split0_keep_backward_kernel
        )
        ctx.use_persistent_state5_gate_cache_tiled_weight_shmem_split0_keep_backward_kernel = (
            use_persistent_state5_gate_cache_tiled_weight_shmem_split0_keep_backward_kernel
        )
        ctx.use_persistent_state6_gate_cache_tiled_weight_shmem_split0_keep_backward_kernel = (
            use_persistent_state6_gate_cache_tiled_weight_shmem_split0_keep_backward_kernel
        )
        ctx.use_persistent_state6_gate_cache_tiled_weight_shmem_split0_keep_unroll8_backward_kernel = (
            use_persistent_state6_gate_cache_tiled_weight_shmem_split0_keep_unroll8_backward_kernel
        )
        ctx.use_persistent_state8_gate_cache_tiled_weight_shmem_split0_keep_backward_kernel = (
            use_persistent_state8_gate_cache_tiled_weight_shmem_split0_keep_backward_kernel
        )
        ctx.use_persistent_state12_gate_cache_tiled_weight_shmem_split0_keep_backward_kernel = (
            use_persistent_state12_gate_cache_tiled_weight_shmem_split0_keep_backward_kernel
        )
        ctx.use_persistent_state12_gate_cache_tiled_weight_shmem_split0_keep_unroll8_backward_kernel = (
            use_persistent_state12_gate_cache_tiled_weight_shmem_split0_keep_unroll8_backward_kernel
        )
        ctx.use_persistent_state24_gate_cache_tiled_weight_shmem_split0_keep_backward_kernel = (
            use_persistent_state24_gate_cache_tiled_weight_shmem_split0_keep_backward_kernel
        )
        ctx.use_persistent_state16_gate_cache_tiled_weight_shmem_split0_keep_own_shmem_backward_kernel = (
            use_persistent_state16_gate_cache_tiled_weight_shmem_split0_keep_own_shmem_backward_kernel
        )
        ctx.use_persistent_state16_grad_coeff_cache_tiled_backward_kernel = (
            use_persistent_state16_grad_coeff_cache_tiled_backward_kernel
        )
        ctx.use_persistent_state32_gate_cache_tiled_backward_kernel = (
            use_persistent_state32_gate_cache_tiled_backward_kernel
        )
        ctx.use_pack_hidden_prev_time_major_kernel = use_pack_hidden_prev_time_major_kernel
        ctx.use_persistent_state16_global_gates_backward_kernel = (
            use_persistent_state16_global_gates_backward_kernel
        )
        ctx.use_persistent_state32_backward_kernel = use_persistent_state32_backward_kernel
        if output_only:
            return output
        return output, h_n

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor, grad_h_n: torch.Tensor):
        x, h0, weight_ih, weight_hh, bias_hh, input_gates, output, gate_cache, hidden_prev_cache = (
            ctx.saved_tensors
        )
        if grad_output is None:
            grad_output = torch.zeros_like(output)
        grad_output = grad_output.contiguous()
        grad_h_n = grad_h_n.contiguous() if grad_h_n is not None else None

        batch_size, seq_len, input_size = x.shape
        hidden_size = h0.size(1)

        # 跨 time step 预计算 recurrent gates，避免 backward 循环内发起大量小 GEMM。
        if hidden_prev_cache.numel() > 0:
            hidden_prev_steps = hidden_prev_cache
        elif ctx.use_pack_hidden_prev_time_major_kernel and hidden_size == 256 and output.is_cuda:
            hidden_prev_steps = _a100_gru_h256_pack_hidden_prev_time_major(h0, output)
        else:
            hidden_prev_steps = torch.cat(
                (h0.unsqueeze(1), output[:, :-1, :]),
                dim=1,
            ).transpose(0, 1).contiguous()
        if (
            ctx.recompute_hidden_gates
            or ctx.use_gate_cache_backward_kernel
            or ctx.use_persistent_state16_gate_cache_backward_kernel
            or ctx.use_persistent_state8_gate_cache_tiled_backward_kernel
            or ctx.use_persistent_state16_gate_cache_tiled_backward_kernel
            or ctx.use_persistent_state16_gate_cache_tiled_weight_shmem_backward_kernel
            or ctx.use_persistent_state16_gate_cache_tiled_weight_shmem_split0_keep_backward_kernel
            or ctx.use_persistent_state5_gate_cache_tiled_weight_shmem_split0_keep_backward_kernel
            or ctx.use_persistent_state6_gate_cache_tiled_weight_shmem_split0_keep_backward_kernel
            or ctx.use_persistent_state6_gate_cache_tiled_weight_shmem_split0_keep_unroll8_backward_kernel
            or ctx.use_persistent_state8_gate_cache_tiled_weight_shmem_split0_keep_backward_kernel
            or ctx.use_persistent_state12_gate_cache_tiled_weight_shmem_split0_keep_backward_kernel
            or ctx.use_persistent_state12_gate_cache_tiled_weight_shmem_split0_keep_unroll8_backward_kernel
            or ctx.use_persistent_state24_gate_cache_tiled_weight_shmem_split0_keep_backward_kernel
            or ctx.use_persistent_state16_gate_cache_tiled_weight_shmem_split0_keep_own_shmem_backward_kernel
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
        persistent_state8_gate_cache_tiled_partial_sums = (
            torch.empty(batch_size, 8, hidden_size, device=x.device, dtype=x.dtype)
            if ctx.use_persistent_state8_gate_cache_tiled_backward_kernel
            else None
        )
        persistent_state16_gate_cache_tiled_partial_sums = (
            torch.empty(batch_size, 16, hidden_size, device=x.device, dtype=x.dtype)
            if ctx.use_persistent_state16_gate_cache_tiled_backward_kernel
            else None
        )
        persistent_state16_gate_cache_tiled_weight_shmem_partial_sums = (
            torch.empty(batch_size, 16, hidden_size, device=x.device, dtype=x.dtype)
            if ctx.use_persistent_state16_gate_cache_tiled_weight_shmem_backward_kernel
            else None
        )
        persistent_state16_gate_cache_tiled_weight_shmem_split0_keep_partial_sums = (
            torch.empty(batch_size, 16, hidden_size, device=x.device, dtype=x.dtype)
            if ctx.use_persistent_state16_gate_cache_tiled_weight_shmem_split0_keep_backward_kernel
            else None
        )
        persistent_state5_gate_cache_tiled_weight_shmem_split0_keep_partial_sums = (
            torch.empty(batch_size, 5, hidden_size, device=x.device, dtype=x.dtype)
            if ctx.use_persistent_state5_gate_cache_tiled_weight_shmem_split0_keep_backward_kernel
            else None
        )
        persistent_state6_gate_cache_tiled_weight_shmem_split0_keep_partial_sums = (
            torch.empty(batch_size, 6, hidden_size, device=x.device, dtype=x.dtype)
            if ctx.use_persistent_state6_gate_cache_tiled_weight_shmem_split0_keep_backward_kernel
            else None
        )
        persistent_state6_gate_cache_tiled_weight_shmem_split0_keep_unroll8_partial_sums = (
            torch.empty(batch_size, 6, hidden_size, device=x.device, dtype=x.dtype)
            if ctx.use_persistent_state6_gate_cache_tiled_weight_shmem_split0_keep_unroll8_backward_kernel
            else None
        )
        persistent_state8_gate_cache_tiled_weight_shmem_split0_keep_partial_sums = (
            torch.empty(batch_size, 8, hidden_size, device=x.device, dtype=x.dtype)
            if ctx.use_persistent_state8_gate_cache_tiled_weight_shmem_split0_keep_backward_kernel
            else None
        )
        persistent_state12_gate_cache_tiled_weight_shmem_split0_keep_partial_sums = (
            torch.empty(batch_size, 12, hidden_size, device=x.device, dtype=x.dtype)
            if ctx.use_persistent_state12_gate_cache_tiled_weight_shmem_split0_keep_backward_kernel
            else None
        )
        persistent_state12_gate_cache_tiled_weight_shmem_split0_keep_unroll8_partial_sums = (
            torch.empty(batch_size, 12, hidden_size, device=x.device, dtype=x.dtype)
            if ctx.use_persistent_state12_gate_cache_tiled_weight_shmem_split0_keep_unroll8_backward_kernel
            else None
        )
        persistent_state24_gate_cache_tiled_weight_shmem_split0_keep_partial_sums = (
            torch.empty(batch_size, 24, hidden_size, device=x.device, dtype=x.dtype)
            if ctx.use_persistent_state24_gate_cache_tiled_weight_shmem_split0_keep_backward_kernel
            else None
        )
        persistent_state16_gate_cache_tiled_weight_shmem_split0_keep_own_shmem_partial_sums = (
            torch.empty(batch_size, 16, hidden_size, device=x.device, dtype=x.dtype)
            if ctx.use_persistent_state16_gate_cache_tiled_weight_shmem_split0_keep_own_shmem_backward_kernel
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
        elif ctx.use_persistent_state8_gate_cache_tiled_backward_kernel:
            assert persistent_state8_gate_cache_tiled_partial_sums is not None
            grad_hidden = (
                _a100_gru_h256_backward_sequence_cooperative_split8_gate_cache_state_tiled(
                    grad_output,
                    grad_hidden,
                    gate_cache,
                    h0,
                    output,
                    weight_hh,
                    grad_input_gates,
                    grad_hidden_gates_steps,
                    persistent_state8_gate_cache_tiled_partial_sums,
                )
            )
        elif ctx.use_persistent_state16_gate_cache_tiled_weight_shmem_split0_keep_backward_kernel:
            assert (
                persistent_state16_gate_cache_tiled_weight_shmem_split0_keep_partial_sums
                is not None
            )
            grad_hidden = (
                _a100_gru_h256_backward_sequence_cooperative_split16_gate_cache_state_tiled_weight_shmem_split0_keep(
                    grad_output,
                    grad_hidden,
                    gate_cache,
                    h0,
                    output,
                    weight_hh,
                    grad_input_gates,
                    grad_hidden_gates_steps,
                    persistent_state16_gate_cache_tiled_weight_shmem_split0_keep_partial_sums,
                )
            )
        elif ctx.use_persistent_state5_gate_cache_tiled_weight_shmem_split0_keep_backward_kernel:
            assert persistent_state5_gate_cache_tiled_weight_shmem_split0_keep_partial_sums is not None
            grad_hidden = (
                _a100_gru_h256_backward_sequence_cooperative_split5_gate_cache_state_tiled_weight_shmem_split0_keep(
                    grad_output,
                    grad_hidden,
                    gate_cache,
                    h0,
                    output,
                    weight_hh,
                    grad_input_gates,
                    grad_hidden_gates_steps,
                    persistent_state5_gate_cache_tiled_weight_shmem_split0_keep_partial_sums,
                )
            )
        elif ctx.use_persistent_state6_gate_cache_tiled_weight_shmem_split0_keep_backward_kernel:
            assert persistent_state6_gate_cache_tiled_weight_shmem_split0_keep_partial_sums is not None
            grad_hidden = (
                _a100_gru_h256_backward_sequence_cooperative_split6_gate_cache_state_tiled_weight_shmem_split0_keep(
                    grad_output,
                    grad_hidden,
                    gate_cache,
                    h0,
                    output,
                    weight_hh,
                    grad_input_gates,
                    grad_hidden_gates_steps,
                    persistent_state6_gate_cache_tiled_weight_shmem_split0_keep_partial_sums,
                )
            )
        elif ctx.use_persistent_state6_gate_cache_tiled_weight_shmem_split0_keep_unroll8_backward_kernel:
            assert (
                persistent_state6_gate_cache_tiled_weight_shmem_split0_keep_unroll8_partial_sums
                is not None
            )
            grad_hidden = (
                _a100_gru_h256_backward_sequence_cooperative_split6_gate_cache_state_tiled_weight_shmem_split0_keep_unroll8(
                    grad_output,
                    grad_hidden,
                    gate_cache,
                    h0,
                    output,
                    weight_hh,
                    grad_input_gates,
                    grad_hidden_gates_steps,
                    persistent_state6_gate_cache_tiled_weight_shmem_split0_keep_unroll8_partial_sums,
                )
            )
        elif ctx.use_persistent_state8_gate_cache_tiled_weight_shmem_split0_keep_backward_kernel:
            assert persistent_state8_gate_cache_tiled_weight_shmem_split0_keep_partial_sums is not None
            grad_hidden = (
                _a100_gru_h256_backward_sequence_cooperative_split8_gate_cache_state_tiled_weight_shmem_split0_keep(
                    grad_output,
                    grad_hidden,
                    gate_cache,
                    h0,
                    output,
                    weight_hh,
                    grad_input_gates,
                    grad_hidden_gates_steps,
                    persistent_state8_gate_cache_tiled_weight_shmem_split0_keep_partial_sums,
                )
            )
        elif ctx.use_persistent_state12_gate_cache_tiled_weight_shmem_split0_keep_backward_kernel:
            assert persistent_state12_gate_cache_tiled_weight_shmem_split0_keep_partial_sums is not None
            grad_hidden = (
                _a100_gru_h256_backward_sequence_cooperative_split12_gate_cache_state_tiled_weight_shmem_split0_keep(
                    grad_output,
                    grad_hidden,
                    gate_cache,
                    h0,
                    output,
                    weight_hh,
                    grad_input_gates,
                    grad_hidden_gates_steps,
                    persistent_state12_gate_cache_tiled_weight_shmem_split0_keep_partial_sums,
                )
            )
        elif ctx.use_persistent_state12_gate_cache_tiled_weight_shmem_split0_keep_unroll8_backward_kernel:
            assert (
                persistent_state12_gate_cache_tiled_weight_shmem_split0_keep_unroll8_partial_sums
                is not None
            )
            grad_hidden = (
                _a100_gru_h256_backward_sequence_cooperative_split12_gate_cache_state_tiled_weight_shmem_split0_keep_unroll8(
                    grad_output,
                    grad_hidden,
                    gate_cache,
                    h0,
                    output,
                    weight_hh,
                    grad_input_gates,
                    grad_hidden_gates_steps,
                    persistent_state12_gate_cache_tiled_weight_shmem_split0_keep_unroll8_partial_sums,
                )
            )
        elif ctx.use_persistent_state24_gate_cache_tiled_weight_shmem_split0_keep_backward_kernel:
            assert persistent_state24_gate_cache_tiled_weight_shmem_split0_keep_partial_sums is not None
            grad_hidden = (
                _a100_gru_h256_backward_sequence_cooperative_split24_gate_cache_state_tiled_weight_shmem_split0_keep(
                    grad_output,
                    grad_hidden,
                    gate_cache,
                    h0,
                    output,
                    weight_hh,
                    grad_input_gates,
                    grad_hidden_gates_steps,
                    persistent_state24_gate_cache_tiled_weight_shmem_split0_keep_partial_sums,
                )
            )
        elif ctx.use_persistent_state16_gate_cache_tiled_weight_shmem_split0_keep_own_shmem_backward_kernel:
            assert (
                persistent_state16_gate_cache_tiled_weight_shmem_split0_keep_own_shmem_partial_sums
                is not None
            )
            grad_hidden = (
                _a100_gru_h256_backward_sequence_cooperative_split16_gate_cache_state_tiled_weight_shmem_split0_keep_own_shmem(
                    grad_output,
                    grad_hidden,
                    gate_cache,
                    h0,
                    output,
                    weight_hh,
                    grad_input_gates,
                    grad_hidden_gates_steps,
                    persistent_state16_gate_cache_tiled_weight_shmem_split0_keep_own_shmem_partial_sums,
                )
            )
        elif ctx.use_persistent_state16_gate_cache_tiled_weight_shmem_backward_kernel:
            assert persistent_state16_gate_cache_tiled_weight_shmem_partial_sums is not None
            grad_hidden = (
                _a100_gru_h256_backward_sequence_cooperative_split16_gate_cache_state_tiled_weight_shmem(
                    grad_output,
                    grad_hidden,
                    gate_cache,
                    h0,
                    output,
                    weight_hh,
                    grad_input_gates,
                    grad_hidden_gates_steps,
                    persistent_state16_gate_cache_tiled_weight_shmem_partial_sums,
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


class A100GRUH256OutputOnlyFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, *args) -> torch.Tensor:
        ctx.output_only = True
        return A100GRUH256Function.forward(ctx, *args)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return A100GRUH256Function.backward(ctx, grad_output, None)


def _a100_gru_h256_apply(
    function: type[torch.autograd.Function],
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
    use_persistent_state8_gate_cache_tiled_backward_kernel: bool = False,
    use_persistent_state16_gate_cache_tiled_backward_kernel: bool = False,
    use_persistent_state16_gate_cache_tiled_weight_shmem_backward_kernel: bool = False,
    use_persistent_state16_gate_cache_tiled_weight_shmem_split0_keep_backward_kernel: bool = False,
    use_persistent_state5_gate_cache_tiled_weight_shmem_split0_keep_backward_kernel: bool = False,
    use_persistent_state6_gate_cache_tiled_weight_shmem_split0_keep_backward_kernel: bool = False,
    use_persistent_state6_gate_cache_tiled_weight_shmem_split0_keep_unroll8_backward_kernel: bool = False,
    use_persistent_state8_gate_cache_tiled_weight_shmem_split0_keep_backward_kernel: bool = False,
    use_persistent_state12_gate_cache_tiled_weight_shmem_split0_keep_backward_kernel: bool = False,
    use_persistent_state12_gate_cache_tiled_weight_shmem_split0_keep_unroll8_backward_kernel: bool = False,
    use_persistent_state24_gate_cache_tiled_weight_shmem_split0_keep_backward_kernel: bool = False,
    use_persistent_state16_gate_cache_tiled_weight_shmem_split0_keep_own_shmem_backward_kernel: bool = False,
    use_persistent_state16_grad_coeff_cache_tiled_backward_kernel: bool = False,
    use_persistent_state32_gate_cache_tiled_backward_kernel: bool = False,
    use_gate_cache_parallel_update_forward_kernel: bool = False,
    use_gate_cache_cta8_forward_kernel: bool = False,
    use_gate_cache_cta6_forward_kernel: bool = False,
    use_gate_cache_htile2_forward_kernel: bool = False,
    use_gate_cache_htile4_forward_kernel: bool = False,
    use_gate_cache_htile4_compact_forward_kernel: bool = False,
    use_gate_cache_htile4_compact_hoist_forward_kernel: bool = False,
    use_gate_cache_htile4_compact_hoist_qwarp_forward_kernel: bool = False,
    use_gate_cache_htile4_compact_hoist_row3_forward_kernel: bool = False,
    use_gate_cache_htile4_compact_hoist_row4_forward_kernel: bool = False,
    use_gate_cache_htile4_compact_hoist_row4_k1_forward_kernel: bool = False,
    use_gate_cache_htile8_compact_hoist_k1_forward_kernel: bool = False,
    use_gate_cache_htile4_compact_hoist_row4_prev_cache_forward_kernel: bool = False,
    use_pack_hidden_prev_time_major_kernel: bool = False,
    use_gate_cache_htile4_compact_hoist_row4_parallel_update_forward_kernel: bool = False,
    use_gate_cache_htile4_compact_hoist_row4_ldg_forward_kernel: bool = False,
    use_gate_cache_htile4_compact_hoist_row4_hidden_shmem_forward_kernel: bool = False,
    use_gate_cache_htile4_compact_hoist_row4_weight_shmem_forward_kernel: bool = False,
    use_gate_cache_htile8_compact_forward_kernel: bool = False,
    use_persistent_state16_global_gates_backward_kernel: bool = False,
    use_persistent_state32_backward_kernel: bool = False,
) -> object:
    return function.apply(
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
        use_persistent_state8_gate_cache_tiled_backward_kernel,
        use_persistent_state16_gate_cache_tiled_backward_kernel,
        use_persistent_state16_gate_cache_tiled_weight_shmem_backward_kernel,
        use_persistent_state16_gate_cache_tiled_weight_shmem_split0_keep_backward_kernel,
        use_persistent_state5_gate_cache_tiled_weight_shmem_split0_keep_backward_kernel,
        use_persistent_state6_gate_cache_tiled_weight_shmem_split0_keep_backward_kernel,
        use_persistent_state6_gate_cache_tiled_weight_shmem_split0_keep_unroll8_backward_kernel,
        use_persistent_state8_gate_cache_tiled_weight_shmem_split0_keep_backward_kernel,
        use_persistent_state12_gate_cache_tiled_weight_shmem_split0_keep_backward_kernel,
        use_persistent_state12_gate_cache_tiled_weight_shmem_split0_keep_unroll8_backward_kernel,
        use_persistent_state24_gate_cache_tiled_weight_shmem_split0_keep_backward_kernel,
        use_persistent_state16_gate_cache_tiled_weight_shmem_split0_keep_own_shmem_backward_kernel,
        use_persistent_state16_grad_coeff_cache_tiled_backward_kernel,
        use_persistent_state32_gate_cache_tiled_backward_kernel,
        use_gate_cache_parallel_update_forward_kernel,
        use_gate_cache_cta8_forward_kernel,
        use_gate_cache_cta6_forward_kernel,
        use_gate_cache_htile2_forward_kernel,
        use_gate_cache_htile4_forward_kernel,
        use_gate_cache_htile4_compact_forward_kernel,
        use_gate_cache_htile4_compact_hoist_forward_kernel,
        use_gate_cache_htile4_compact_hoist_qwarp_forward_kernel,
        use_gate_cache_htile4_compact_hoist_row3_forward_kernel,
        use_gate_cache_htile4_compact_hoist_row4_forward_kernel,
        use_gate_cache_htile4_compact_hoist_row4_k1_forward_kernel,
        use_gate_cache_htile8_compact_hoist_k1_forward_kernel,
        use_gate_cache_htile4_compact_hoist_row4_prev_cache_forward_kernel,
        use_pack_hidden_prev_time_major_kernel,
        use_gate_cache_htile4_compact_hoist_row4_parallel_update_forward_kernel,
        use_gate_cache_htile4_compact_hoist_row4_ldg_forward_kernel,
        use_gate_cache_htile4_compact_hoist_row4_hidden_shmem_forward_kernel,
        use_gate_cache_htile4_compact_hoist_row4_weight_shmem_forward_kernel,
        use_gate_cache_htile8_compact_forward_kernel,
        use_persistent_state16_global_gates_backward_kernel,
        use_persistent_state32_backward_kernel,
    )


def a100_gru_h256(*args, **kwargs) -> tuple[torch.Tensor, torch.Tensor]:
    return _a100_gru_h256_apply(A100GRUH256Function, *args, **kwargs)


def a100_gru_h256_output_only(*args, **kwargs) -> torch.Tensor:
    return _a100_gru_h256_apply(A100GRUH256OutputOnlyFunction, *args, **kwargs)


def _a100_gru_h256_hybrid_k1_train_forward(
    x: torch.Tensor,
    h0: torch.Tensor,
    layer_params: tuple[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], ...],
    block_threads: int = 256,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """实验性 hybrid forward：input projection 走 cuBLAS，recurrent 走 K1 gate-cache kernel。"""
    batch_size, seq_len, _ = x.shape
    num_layers = len(layer_params)
    layer_input = x.contiguous()
    all_outputs = torch.empty(
        num_layers,
        batch_size,
        seq_len,
        256,
        device=x.device,
        dtype=x.dtype,
    )
    gate_cache_all = torch.empty(
        num_layers,
        batch_size,
        seq_len,
        4 * 256,
        device=x.device,
        dtype=x.dtype,
    )
    final_output: torch.Tensor | None = None

    for layer, (weight_ih, weight_hh, bias_ih, bias_hh) in enumerate(layer_params):
        weight_ih = weight_ih.contiguous()
        weight_hh = weight_hh.contiguous()
        bias_ih = bias_ih.contiguous()
        bias_hh = bias_hh.contiguous()
        input_gates = F.linear(layer_input, weight_ih, bias_ih).contiguous()
        output, gate_cache = (
            a100_gru_forward_from_gates_cooperative_h256_htile4_compact_hoist_row4_k1_shmem_gate_cache(
                input_gates,
                h0[layer].contiguous(),
                weight_hh,
                bias_hh,
                block_threads=block_threads,
                output_out=all_outputs[layer],
                gate_cache_out=gate_cache_all[layer],
            )
        )
        del gate_cache
        output = output.contiguous()
        layer_input = output
        final_output = output

    if final_output is None:
        raise RuntimeError("hybrid forward requires at least one layer.")
    h_n = all_outputs[:, :, seq_len - 1, :].contiguous()
    if all_outputs.shape != (num_layers, batch_size, seq_len, 256):
        raise RuntimeError("hybrid all_outputs has an unexpected shape.")
    return final_output, h_n, all_outputs, gate_cache_all


class A100GRUH256StackedFusedNaiveFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        h0: torch.Tensor,
        num_layers: int,
        backward_mode: int,
        forward_mode: int,
        *flat_params: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if len(flat_params) != num_layers * 4:
            raise ValueError("flat_params must contain weight_ih, weight_hh, bias_ih, bias_hh per layer.")
        layer_params = tuple(
            (
                flat_params[layer * 4],
                flat_params[layer * 4 + 1],
                flat_params[layer * 4 + 2],
                flat_params[layer * 4 + 3],
            )
            for layer in range(num_layers)
        )
        if forward_mode == 2:
            output, h_n, all_outputs, gate_cache_all = _a100_gru_h256_hybrid_k1_train_forward(
                x,
                h0,
                layer_params,
                block_threads=256,
            )
        else:
            forward_kernel = (
                a100_gru_h256_stacked_row4_k1_train_forward
                if forward_mode == 1
                else a100_gru_h256_stacked_row4_train_forward
            )
            output, h_n, all_outputs, gate_cache_all = forward_kernel(
                x,
                h0,
                layer_params,
                block_threads=256,
            )
        ctx.num_layers = num_layers
        ctx.backward_mode = backward_mode
        ctx.save_for_backward(x, h0, all_outputs, gate_cache_all, *flat_params)
        return output, h_n

    @staticmethod
    def backward(
        ctx,
        grad_output: Optional[torch.Tensor],
        grad_h_n: Optional[torch.Tensor],
    ):
        saved = ctx.saved_tensors
        x = saved[0]
        h0 = saved[1]
        all_outputs = saved[2]
        gate_cache_all = saved[3]
        flat_params = saved[4:]
        num_layers = ctx.num_layers
        layer_params = tuple(
            (
                flat_params[layer * 4],
                flat_params[layer * 4 + 1],
                flat_params[layer * 4 + 2],
                flat_params[layer * 4 + 3],
            )
            for layer in range(num_layers)
        )
        if grad_output is None:
            grad_output = torch.zeros_like(all_outputs[-1])
        if grad_h_n is None:
            grad_h_n = torch.zeros_like(h0)

        backward_kernel = (
            a100_gru_h256_stacked_backward_split4
            if ctx.backward_mode == 1
            else a100_gru_h256_stacked_backward_split8
            if ctx.backward_mode == 2
            else a100_gru_h256_stacked_backward_split4_shmem
            if ctx.backward_mode == 3
            else a100_gru_h256_stacked_backward_split4_group8
            if ctx.backward_mode == 4
            else a100_gru_h256_stacked_backward_split6_weight_shmem
            if ctx.backward_mode == 5
            else a100_gru_h256_stacked_backward_naive
        )
        grad_x, grad_h0, grad_input_gates_all, grad_hidden_gates_all, _ = (
            backward_kernel(
                grad_output.contiguous(),
                grad_h_n.contiguous(),
                x,
                h0,
                layer_params,
                all_outputs,
                gate_cache_all,
                block_threads=256,
            )
        )

        param_grads: list[torch.Tensor] = []
        batch_size, seq_len, _ = grad_output.shape
        for layer, (weight_ih, weight_hh, bias_ih, bias_hh) in enumerate(layer_params):
            del weight_ih, weight_hh, bias_ih, bias_hh
            layer_input = x if layer == 0 else all_outputs[layer - 1]
            hidden_prev = torch.cat(
                (h0[layer].unsqueeze(1), all_outputs[layer, :, :-1, :]),
                dim=1,
            )
            grad_input_gates = grad_input_gates_all[layer]
            grad_hidden_gates = grad_hidden_gates_all[layer]
            grad_input_2d = grad_input_gates.reshape(batch_size * seq_len, 3 * 256)
            grad_hidden_2d = grad_hidden_gates.reshape(batch_size * seq_len, 3 * 256)
            layer_input_2d = layer_input.reshape(batch_size * seq_len, layer_input.size(-1))
            hidden_prev_2d = hidden_prev.reshape(batch_size * seq_len, 256)

            grad_weight_ih = grad_input_2d.transpose(0, 1).matmul(layer_input_2d)
            grad_weight_hh = grad_hidden_2d.transpose(0, 1).matmul(hidden_prev_2d)
            grad_bias_ih = grad_input_2d.sum(dim=0)
            grad_bias_hh = grad_hidden_2d.sum(dim=0)
            param_grads.extend((grad_weight_ih, grad_weight_hh, grad_bias_ih, grad_bias_hh))

        return (grad_x, grad_h0, None, None, None, *param_grads)


class A100GRUH256(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 256,
        num_layers: int = 1,
        batch_first: bool = True,
        bias: bool = True,
        dropout: float = 0.0,
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
        use_persistent_state8_gate_cache_tiled_backward_kernel: bool = False,
        use_persistent_state16_gate_cache_tiled_backward_kernel: bool = False,
        use_persistent_state16_gate_cache_tiled_weight_shmem_backward_kernel: bool = False,
        use_persistent_state16_gate_cache_tiled_weight_shmem_split0_keep_backward_kernel: bool = False,
        use_persistent_state5_gate_cache_tiled_weight_shmem_split0_keep_backward_kernel: bool = False,
        use_persistent_state6_gate_cache_tiled_weight_shmem_split0_keep_backward_kernel: bool = False,
        use_persistent_state6_gate_cache_tiled_weight_shmem_split0_keep_unroll8_backward_kernel: bool = False,
        use_persistent_state8_gate_cache_tiled_weight_shmem_split0_keep_backward_kernel: bool = False,
        use_persistent_state12_gate_cache_tiled_weight_shmem_split0_keep_backward_kernel: bool = False,
        use_persistent_state12_gate_cache_tiled_weight_shmem_split0_keep_unroll8_backward_kernel: bool = False,
        use_persistent_state24_gate_cache_tiled_weight_shmem_split0_keep_backward_kernel: bool = False,
        use_persistent_state16_gate_cache_tiled_weight_shmem_split0_keep_own_shmem_backward_kernel: bool = False,
        use_persistent_state16_grad_coeff_cache_tiled_backward_kernel: bool = False,
        use_persistent_state32_gate_cache_tiled_backward_kernel: bool = False,
        use_gate_cache_parallel_update_forward_kernel: bool = False,
        use_gate_cache_cta8_forward_kernel: bool = False,
        use_gate_cache_cta6_forward_kernel: bool = False,
        use_gate_cache_htile2_forward_kernel: bool = False,
        use_gate_cache_htile4_forward_kernel: bool = False,
        use_gate_cache_htile4_compact_forward_kernel: bool = False,
        use_gate_cache_htile4_compact_hoist_forward_kernel: bool = False,
        use_gate_cache_htile4_compact_hoist_qwarp_forward_kernel: bool = False,
        use_gate_cache_htile4_compact_hoist_row3_forward_kernel: bool = False,
        use_gate_cache_htile4_compact_hoist_row4_forward_kernel: bool = False,
        use_gate_cache_htile4_compact_hoist_row4_k1_forward_kernel: bool = False,
        use_gate_cache_htile8_compact_hoist_k1_forward_kernel: bool = False,
        use_gate_cache_htile4_compact_hoist_row4_prev_cache_forward_kernel: bool = False,
        use_pack_hidden_prev_time_major_kernel: bool = False,
        use_gate_cache_htile4_compact_hoist_row4_parallel_update_forward_kernel: bool = False,
        use_gate_cache_htile4_compact_hoist_row4_ldg_forward_kernel: bool = False,
        use_gate_cache_htile4_compact_hoist_row4_hidden_shmem_forward_kernel: bool = False,
        use_gate_cache_htile4_compact_hoist_row4_weight_shmem_forward_kernel: bool = False,
        use_gate_cache_htile8_compact_forward_kernel: bool = False,
        use_persistent_state16_global_gates_backward_kernel: bool = False,
        use_persistent_state32_backward_kernel: bool = False,
    ) -> None:
        super().__init__()
        if hidden_size != 256:
            raise ValueError("A100GRUH256 only supports hidden_size=256.")
        if not 1 <= input_size <= MAX_INPUT_SIZE:
            raise ValueError("A100GRUH256 only supports 1 <= input_size <= 16.")
        if not 1 <= num_layers <= MAX_NUM_LAYERS:
            raise ValueError("A100GRUH256 only supports 1 <= num_layers <= 4.")
        if not batch_first:
            raise ValueError("A100GRUH256 only supports batch_first=True.")
        if not bias:
            raise ValueError("A100GRUH256 currently requires bias=True.")
        if dropout != 0.0:
            raise ValueError("A100GRUH256 currently requires dropout=0.")

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bias = bias
        self.dropout = float(dropout)
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
        self.use_persistent_state8_gate_cache_tiled_backward_kernel = (
            use_persistent_state8_gate_cache_tiled_backward_kernel
        )
        self.use_persistent_state16_gate_cache_tiled_backward_kernel = (
            use_persistent_state16_gate_cache_tiled_backward_kernel
        )
        self.use_persistent_state16_gate_cache_tiled_weight_shmem_backward_kernel = (
            use_persistent_state16_gate_cache_tiled_weight_shmem_backward_kernel
        )
        self.use_persistent_state16_gate_cache_tiled_weight_shmem_split0_keep_backward_kernel = (
            use_persistent_state16_gate_cache_tiled_weight_shmem_split0_keep_backward_kernel
        )
        self.use_persistent_state5_gate_cache_tiled_weight_shmem_split0_keep_backward_kernel = (
            use_persistent_state5_gate_cache_tiled_weight_shmem_split0_keep_backward_kernel
        )
        self.use_persistent_state6_gate_cache_tiled_weight_shmem_split0_keep_backward_kernel = (
            use_persistent_state6_gate_cache_tiled_weight_shmem_split0_keep_backward_kernel
        )
        self.use_persistent_state6_gate_cache_tiled_weight_shmem_split0_keep_unroll8_backward_kernel = (
            use_persistent_state6_gate_cache_tiled_weight_shmem_split0_keep_unroll8_backward_kernel
        )
        self.use_persistent_state8_gate_cache_tiled_weight_shmem_split0_keep_backward_kernel = (
            use_persistent_state8_gate_cache_tiled_weight_shmem_split0_keep_backward_kernel
        )
        self.use_persistent_state12_gate_cache_tiled_weight_shmem_split0_keep_backward_kernel = (
            use_persistent_state12_gate_cache_tiled_weight_shmem_split0_keep_backward_kernel
        )
        self.use_persistent_state12_gate_cache_tiled_weight_shmem_split0_keep_unroll8_backward_kernel = (
            use_persistent_state12_gate_cache_tiled_weight_shmem_split0_keep_unroll8_backward_kernel
        )
        self.use_persistent_state24_gate_cache_tiled_weight_shmem_split0_keep_backward_kernel = (
            use_persistent_state24_gate_cache_tiled_weight_shmem_split0_keep_backward_kernel
        )
        self.use_persistent_state16_gate_cache_tiled_weight_shmem_split0_keep_own_shmem_backward_kernel = (
            use_persistent_state16_gate_cache_tiled_weight_shmem_split0_keep_own_shmem_backward_kernel
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
        self.use_gate_cache_htile2_forward_kernel = use_gate_cache_htile2_forward_kernel
        self.use_gate_cache_htile4_forward_kernel = use_gate_cache_htile4_forward_kernel
        self.use_gate_cache_htile4_compact_forward_kernel = (
            use_gate_cache_htile4_compact_forward_kernel
        )
        self.use_gate_cache_htile4_compact_hoist_forward_kernel = (
            use_gate_cache_htile4_compact_hoist_forward_kernel
        )
        self.use_gate_cache_htile4_compact_hoist_qwarp_forward_kernel = (
            use_gate_cache_htile4_compact_hoist_qwarp_forward_kernel
        )
        self.use_gate_cache_htile4_compact_hoist_row3_forward_kernel = (
            use_gate_cache_htile4_compact_hoist_row3_forward_kernel
        )
        self.use_gate_cache_htile4_compact_hoist_row4_forward_kernel = (
            use_gate_cache_htile4_compact_hoist_row4_forward_kernel
        )
        self.use_gate_cache_htile4_compact_hoist_row4_k1_forward_kernel = (
            use_gate_cache_htile4_compact_hoist_row4_k1_forward_kernel
        )
        self.use_gate_cache_htile8_compact_hoist_k1_forward_kernel = (
            use_gate_cache_htile8_compact_hoist_k1_forward_kernel
        )
        self.use_gate_cache_htile4_compact_hoist_row4_prev_cache_forward_kernel = (
            use_gate_cache_htile4_compact_hoist_row4_prev_cache_forward_kernel
        )
        self.use_pack_hidden_prev_time_major_kernel = use_pack_hidden_prev_time_major_kernel
        self.use_gate_cache_htile4_compact_hoist_row4_parallel_update_forward_kernel = (
            use_gate_cache_htile4_compact_hoist_row4_parallel_update_forward_kernel
        )
        self.use_gate_cache_htile4_compact_hoist_row4_ldg_forward_kernel = (
            use_gate_cache_htile4_compact_hoist_row4_ldg_forward_kernel
        )
        self.use_gate_cache_htile4_compact_hoist_row4_hidden_shmem_forward_kernel = (
            use_gate_cache_htile4_compact_hoist_row4_hidden_shmem_forward_kernel
        )
        self.use_gate_cache_htile4_compact_hoist_row4_weight_shmem_forward_kernel = (
            use_gate_cache_htile4_compact_hoist_row4_weight_shmem_forward_kernel
        )
        self.use_gate_cache_htile8_compact_forward_kernel = (
            use_gate_cache_htile8_compact_forward_kernel
        )
        self.use_persistent_state16_global_gates_backward_kernel = (
            use_persistent_state16_global_gates_backward_kernel
        )
        self.use_persistent_state32_backward_kernel = use_persistent_state32_backward_kernel

        for layer in range(num_layers):
            layer_input_size = input_size if layer == 0 else hidden_size
            setattr(self, f"weight_ih_l{layer}", nn.Parameter(torch.empty(3 * hidden_size, layer_input_size)))
            setattr(self, f"weight_hh_l{layer}", nn.Parameter(torch.empty(3 * hidden_size, hidden_size)))
            setattr(self, f"bias_ih_l{layer}", nn.Parameter(torch.empty(3 * hidden_size)))
            setattr(self, f"bias_hh_l{layer}", nn.Parameter(torch.empty(3 * hidden_size)))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            nn.init.uniform_(weight, -stdv, stdv)

    def _layer_parameters(self, layer: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            getattr(self, f"weight_ih_l{layer}"),
            getattr(self, f"weight_hh_l{layer}"),
            getattr(self, f"bias_ih_l{layer}"),
            getattr(self, f"bias_hh_l{layer}"),
        )

    def _prepare_hx(self, x: torch.Tensor, hx: Optional[torch.Tensor]) -> torch.Tensor:
        if hx is None:
            return x.new_zeros(self.num_layers, x.size(0), self.hidden_size)
        expected_shape = (self.num_layers, x.size(0), self.hidden_size)
        if hx.shape != expected_shape:
            raise ValueError(f"hx must have shape {expected_shape}.")
        return hx

    def _forward_train_layer(
        self,
        layer_input: torch.Tensor,
        h0: torch.Tensor,
        layer: int,
        output_only: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor] | torch.Tensor:
        weight_ih, weight_hh, bias_ih, bias_hh = self._layer_parameters(layer)
        runner = a100_gru_h256_output_only if output_only else a100_gru_h256
        return runner(
            layer_input,
            h0,
            weight_ih,
            weight_hh,
            bias_ih,
            bias_hh,
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
            use_persistent_state8_gate_cache_tiled_backward_kernel=(
                self.use_persistent_state8_gate_cache_tiled_backward_kernel
            ),
            use_persistent_state16_gate_cache_tiled_backward_kernel=(
                self.use_persistent_state16_gate_cache_tiled_backward_kernel
            ),
            use_persistent_state16_gate_cache_tiled_weight_shmem_backward_kernel=(
                self.use_persistent_state16_gate_cache_tiled_weight_shmem_backward_kernel
            ),
            use_persistent_state16_gate_cache_tiled_weight_shmem_split0_keep_backward_kernel=(
                self.use_persistent_state16_gate_cache_tiled_weight_shmem_split0_keep_backward_kernel
            ),
            use_persistent_state5_gate_cache_tiled_weight_shmem_split0_keep_backward_kernel=(
                self.use_persistent_state5_gate_cache_tiled_weight_shmem_split0_keep_backward_kernel
            ),
            use_persistent_state6_gate_cache_tiled_weight_shmem_split0_keep_backward_kernel=(
                self.use_persistent_state6_gate_cache_tiled_weight_shmem_split0_keep_backward_kernel
            ),
            use_persistent_state6_gate_cache_tiled_weight_shmem_split0_keep_unroll8_backward_kernel=(
                self.use_persistent_state6_gate_cache_tiled_weight_shmem_split0_keep_unroll8_backward_kernel
            ),
            use_persistent_state8_gate_cache_tiled_weight_shmem_split0_keep_backward_kernel=(
                self.use_persistent_state8_gate_cache_tiled_weight_shmem_split0_keep_backward_kernel
            ),
            use_persistent_state12_gate_cache_tiled_weight_shmem_split0_keep_backward_kernel=(
                self.use_persistent_state12_gate_cache_tiled_weight_shmem_split0_keep_backward_kernel
            ),
            use_persistent_state12_gate_cache_tiled_weight_shmem_split0_keep_unroll8_backward_kernel=(
                self.use_persistent_state12_gate_cache_tiled_weight_shmem_split0_keep_unroll8_backward_kernel
            ),
            use_persistent_state24_gate_cache_tiled_weight_shmem_split0_keep_backward_kernel=(
                self.use_persistent_state24_gate_cache_tiled_weight_shmem_split0_keep_backward_kernel
            ),
            use_persistent_state16_gate_cache_tiled_weight_shmem_split0_keep_own_shmem_backward_kernel=(
                self.use_persistent_state16_gate_cache_tiled_weight_shmem_split0_keep_own_shmem_backward_kernel
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
            use_gate_cache_htile2_forward_kernel=(
                self.use_gate_cache_htile2_forward_kernel
            ),
            use_gate_cache_htile4_forward_kernel=(
                self.use_gate_cache_htile4_forward_kernel
            ),
            use_gate_cache_htile4_compact_forward_kernel=(
                self.use_gate_cache_htile4_compact_forward_kernel
            ),
            use_gate_cache_htile4_compact_hoist_forward_kernel=(
                self.use_gate_cache_htile4_compact_hoist_forward_kernel
            ),
            use_gate_cache_htile4_compact_hoist_qwarp_forward_kernel=(
                self.use_gate_cache_htile4_compact_hoist_qwarp_forward_kernel
            ),
            use_gate_cache_htile4_compact_hoist_row3_forward_kernel=(
                self.use_gate_cache_htile4_compact_hoist_row3_forward_kernel
            ),
            use_gate_cache_htile4_compact_hoist_row4_forward_kernel=(
                self.use_gate_cache_htile4_compact_hoist_row4_forward_kernel
            ),
            use_gate_cache_htile4_compact_hoist_row4_k1_forward_kernel=(
                self.use_gate_cache_htile4_compact_hoist_row4_k1_forward_kernel
            ),
            use_gate_cache_htile8_compact_hoist_k1_forward_kernel=(
                self.use_gate_cache_htile8_compact_hoist_k1_forward_kernel
            ),
            use_gate_cache_htile4_compact_hoist_row4_prev_cache_forward_kernel=(
                self.use_gate_cache_htile4_compact_hoist_row4_prev_cache_forward_kernel
            ),
            use_pack_hidden_prev_time_major_kernel=(
                self.use_pack_hidden_prev_time_major_kernel
            ),
            use_gate_cache_htile4_compact_hoist_row4_parallel_update_forward_kernel=(
                self.use_gate_cache_htile4_compact_hoist_row4_parallel_update_forward_kernel
            ),
            use_gate_cache_htile4_compact_hoist_row4_ldg_forward_kernel=(
                self.use_gate_cache_htile4_compact_hoist_row4_ldg_forward_kernel
            ),
            use_gate_cache_htile4_compact_hoist_row4_hidden_shmem_forward_kernel=(
                self.use_gate_cache_htile4_compact_hoist_row4_hidden_shmem_forward_kernel
            ),
            use_gate_cache_htile4_compact_hoist_row4_weight_shmem_forward_kernel=(
                self.use_gate_cache_htile4_compact_hoist_row4_weight_shmem_forward_kernel
            ),
            use_gate_cache_htile8_compact_forward_kernel=(
                self.use_gate_cache_htile8_compact_forward_kernel
            ),
            use_persistent_state16_global_gates_backward_kernel=(
                self.use_persistent_state16_global_gates_backward_kernel
            ),
            use_persistent_state32_backward_kernel=(
                self.use_persistent_state32_backward_kernel
            ),
        )

    def _forward_inference_layer(
        self,
        layer_input: torch.Tensor,
        h0: torch.Tensor,
        layer: int,
        output_only: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor] | torch.Tensor:
        weight_ih, weight_hh, bias_ih, bias_hh = self._layer_parameters(layer)
        with torch.no_grad():
            layer_input = layer_input.contiguous()
            h0 = h0.contiguous()
            weight_ih = weight_ih.contiguous()
            weight_hh = weight_hh.contiguous()
            bias_ih = bias_ih.contiguous()
            bias_hh = bias_hh.contiguous()
            batch_size, seq_len, input_size = layer_input.shape
            input_gates = F.linear(
                layer_input.reshape(batch_size * seq_len, input_size),
                weight_ih,
                bias_ih,
            ).view(batch_size, seq_len, 3 * self.hidden_size)
            output = a100_gru_forward_from_gates_cooperative_h256_htile4_compact_hoist_row4_no_cache_k1(
                input_gates,
                h0,
                weight_hh,
                bias_hh,
                block_threads=256,
            )
            if output_only:
                return output
            h_n = output[:, -1, :].unsqueeze(0).contiguous()
        return output, h_n

    def _forward_inference_layer_ldg(
        self,
        layer_input: torch.Tensor,
        h0: torch.Tensor,
        layer: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        weight_ih, weight_hh, bias_ih, bias_hh = self._layer_parameters(layer)
        with torch.no_grad():
            layer_input = layer_input.contiguous()
            h0 = h0.contiguous()
            weight_ih = weight_ih.contiguous()
            weight_hh = weight_hh.contiguous()
            bias_ih = bias_ih.contiguous()
            bias_hh = bias_hh.contiguous()
            batch_size, seq_len, input_size = layer_input.shape
            input_gates = F.linear(
                layer_input.reshape(batch_size * seq_len, input_size),
                weight_ih,
                bias_ih,
            ).view(batch_size, seq_len, 3 * self.hidden_size)
            output = a100_gru_forward_from_gates_cooperative_h256_htile4_compact_hoist_row4_no_cache_ldg(
                input_gates,
                h0,
                weight_hh,
                bias_hh,
                block_threads=256,
            )
            h_n = output[:, -1, :].unsqueeze(0).contiguous()
        return output, h_n

    def _forward_inference_layer_k2(
        self,
        layer_input: torch.Tensor,
        h0: torch.Tensor,
        layer: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        weight_ih, weight_hh, bias_ih, bias_hh = self._layer_parameters(layer)
        with torch.no_grad():
            layer_input = layer_input.contiguous()
            h0 = h0.contiguous()
            weight_ih = weight_ih.contiguous()
            weight_hh = weight_hh.contiguous()
            bias_ih = bias_ih.contiguous()
            bias_hh = bias_hh.contiguous()
            batch_size, seq_len, input_size = layer_input.shape
            input_gates = F.linear(
                layer_input.reshape(batch_size * seq_len, input_size),
                weight_ih,
                bias_ih,
            ).view(batch_size, seq_len, 3 * self.hidden_size)
            output = a100_gru_forward_from_gates_cooperative_h256_htile4_compact_hoist_row4_no_cache_k2(
                input_gates,
                h0,
                weight_hh,
                bias_hh,
                block_threads=256,
            )
            h_n = output[:, -1, :].unsqueeze(0).contiguous()
        return output, h_n

    def _forward_inference_layer_k1(
        self,
        layer_input: torch.Tensor,
        h0: torch.Tensor,
        layer: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        weight_ih, weight_hh, bias_ih, bias_hh = self._layer_parameters(layer)
        with torch.no_grad():
            layer_input = layer_input.contiguous()
            h0 = h0.contiguous()
            weight_ih = weight_ih.contiguous()
            weight_hh = weight_hh.contiguous()
            bias_ih = bias_ih.contiguous()
            bias_hh = bias_hh.contiguous()
            batch_size, seq_len, input_size = layer_input.shape
            input_gates = F.linear(
                layer_input.reshape(batch_size * seq_len, input_size),
                weight_ih,
                bias_ih,
            ).view(batch_size, seq_len, 3 * self.hidden_size)
            output = a100_gru_forward_from_gates_cooperative_h256_htile4_compact_hoist_row4_no_cache_k1(
                input_gates,
                h0,
                weight_hh,
                bias_hh,
                block_threads=256,
            )
            h_n = output[:, -1, :].unsqueeze(0).contiguous()
        return output, h_n

    def _forward_inference_layer_k1_htile8(
        self,
        layer_input: torch.Tensor,
        h0: torch.Tensor,
        layer: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        weight_ih, weight_hh, bias_ih, bias_hh = self._layer_parameters(layer)
        with torch.no_grad():
            layer_input = layer_input.contiguous()
            h0 = h0.contiguous()
            weight_ih = weight_ih.contiguous()
            weight_hh = weight_hh.contiguous()
            bias_ih = bias_ih.contiguous()
            bias_hh = bias_hh.contiguous()
            batch_size, seq_len, input_size = layer_input.shape
            input_gates = F.linear(
                layer_input.reshape(batch_size * seq_len, input_size),
                weight_ih,
                bias_ih,
            ).view(batch_size, seq_len, 3 * self.hidden_size)
            output = a100_gru_forward_from_gates_cooperative_h256_htile8_compact_hoist_k1_no_cache(
                input_gates,
                h0,
                weight_hh,
                bias_hh,
                block_threads=256,
            )
            h_n = output[:, -1, :].unsqueeze(0).contiguous()
        return output, h_n

    def _forward_inference_k1_htile8(self, x: torch.Tensor, hx: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        output = x
        h_n_parts = []
        for layer in range(self.num_layers):
            output, h_n = self._forward_inference_layer_k1_htile8(output, hx[layer], layer)
            h_n_parts.append(h_n)
        return output, torch.cat(h_n_parts, dim=0)

    def _forward_train_1(self, x: torch.Tensor, hx: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        output0, h0 = self._forward_train_layer(x, hx[0], 0)
        return output0, h0

    def _forward_train_2(self, x: torch.Tensor, hx: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        output0, h0 = self._forward_train_layer(x, hx[0], 0)
        output1, h1 = self._forward_train_layer(output0, hx[1], 1)
        return output1, torch.cat((h0, h1), dim=0)

    def _forward_train_3(self, x: torch.Tensor, hx: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        output0, h0 = self._forward_train_layer(x, hx[0], 0)
        output1, h1 = self._forward_train_layer(output0, hx[1], 1)
        output2, h2 = self._forward_train_layer(output1, hx[2], 2)
        return output2, torch.cat((h0, h1, h2), dim=0)

    def _forward_train_4(self, x: torch.Tensor, hx: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        output0, h0 = self._forward_train_layer(x, hx[0], 0)
        output1, h1 = self._forward_train_layer(output0, hx[1], 1)
        output2, h2 = self._forward_train_layer(output1, hx[2], 2)
        output3, h3 = self._forward_train_layer(output2, hx[3], 3)
        return output3, torch.cat((h0, h1, h2, h3), dim=0)

    def _forward_train(self, x: torch.Tensor, hx: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.num_layers == 1:
            return self._forward_train_1(x, hx)
        if self.num_layers == 2:
            return self._forward_train_2(x, hx)
        if self.num_layers == 3:
            return self._forward_train_3(x, hx)
        return self._forward_train_4(x, hx)

    def _forward_train_output_only_1(self, x: torch.Tensor, hx: torch.Tensor) -> torch.Tensor:
        return self._forward_train_layer(x, hx[0], 0, output_only=True)

    def _forward_train_output_only_2(self, x: torch.Tensor, hx: torch.Tensor) -> torch.Tensor:
        output0 = self._forward_train_layer(x, hx[0], 0, output_only=True)
        return self._forward_train_layer(output0, hx[1], 1, output_only=True)

    def _forward_train_output_only_3(self, x: torch.Tensor, hx: torch.Tensor) -> torch.Tensor:
        output0 = self._forward_train_layer(x, hx[0], 0, output_only=True)
        output1 = self._forward_train_layer(output0, hx[1], 1, output_only=True)
        return self._forward_train_layer(output1, hx[2], 2, output_only=True)

    def _forward_train_output_only_4(self, x: torch.Tensor, hx: torch.Tensor) -> torch.Tensor:
        output0 = self._forward_train_layer(x, hx[0], 0, output_only=True)
        output1 = self._forward_train_layer(output0, hx[1], 1, output_only=True)
        output2 = self._forward_train_layer(output1, hx[2], 2, output_only=True)
        return self._forward_train_layer(output2, hx[3], 3, output_only=True)

    def _forward_train_output_only(self, x: torch.Tensor, hx: torch.Tensor) -> torch.Tensor:
        if self.num_layers == 1:
            return self._forward_train_output_only_1(x, hx)
        if self.num_layers == 2:
            return self._forward_train_output_only_2(x, hx)
        if self.num_layers == 3:
            return self._forward_train_output_only_3(x, hx)
        return self._forward_train_output_only_4(x, hx)

    def _forward_inference_1(self, x: torch.Tensor, hx: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        output0, h0 = self._forward_inference_layer(x, hx[0], 0)
        return output0, h0

    def _forward_inference_2(self, x: torch.Tensor, hx: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        output0, h0 = self._forward_inference_layer(x, hx[0], 0)
        output1, h1 = self._forward_inference_layer(output0, hx[1], 1)
        return output1, torch.cat((h0, h1), dim=0)

    def _forward_inference_3(self, x: torch.Tensor, hx: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        output0, h0 = self._forward_inference_layer(x, hx[0], 0)
        output1, h1 = self._forward_inference_layer(output0, hx[1], 1)
        output2, h2 = self._forward_inference_layer(output1, hx[2], 2)
        return output2, torch.cat((h0, h1, h2), dim=0)

    def _forward_inference_4(self, x: torch.Tensor, hx: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        output0, h0 = self._forward_inference_layer(x, hx[0], 0)
        output1, h1 = self._forward_inference_layer(output0, hx[1], 1)
        output2, h2 = self._forward_inference_layer(output1, hx[2], 2)
        output3, h3 = self._forward_inference_layer(output2, hx[3], 3)
        return output3, torch.cat((h0, h1, h2, h3), dim=0)

    def _forward_inference_ldg_1(self, x: torch.Tensor, hx: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        output0, h0 = self._forward_inference_layer_ldg(x, hx[0], 0)
        return output0, h0

    def _forward_inference_ldg_2(self, x: torch.Tensor, hx: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        output0, h0 = self._forward_inference_layer_ldg(x, hx[0], 0)
        output1, h1 = self._forward_inference_layer_ldg(output0, hx[1], 1)
        return output1, torch.cat((h0, h1), dim=0)

    def _forward_inference_ldg_3(self, x: torch.Tensor, hx: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        output0, h0 = self._forward_inference_layer_ldg(x, hx[0], 0)
        output1, h1 = self._forward_inference_layer_ldg(output0, hx[1], 1)
        output2, h2 = self._forward_inference_layer_ldg(output1, hx[2], 2)
        return output2, torch.cat((h0, h1, h2), dim=0)

    def _forward_inference_ldg_4(self, x: torch.Tensor, hx: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        output0, h0 = self._forward_inference_layer_ldg(x, hx[0], 0)
        output1, h1 = self._forward_inference_layer_ldg(output0, hx[1], 1)
        output2, h2 = self._forward_inference_layer_ldg(output1, hx[2], 2)
        output3, h3 = self._forward_inference_layer_ldg(output2, hx[3], 3)
        return output3, torch.cat((h0, h1, h2, h3), dim=0)

    def _forward_inference_ldg(self, x: torch.Tensor, hx: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.num_layers == 1:
            return self._forward_inference_ldg_1(x, hx)
        if self.num_layers == 2:
            return self._forward_inference_ldg_2(x, hx)
        if self.num_layers == 3:
            return self._forward_inference_ldg_3(x, hx)
        return self._forward_inference_ldg_4(x, hx)

    def _forward_inference_k2_1(self, x: torch.Tensor, hx: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        output0, h0 = self._forward_inference_layer_k2(x, hx[0], 0)
        return output0, h0

    def _forward_inference_k2_2(self, x: torch.Tensor, hx: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        output0, h0 = self._forward_inference_layer_k2(x, hx[0], 0)
        output1, h1 = self._forward_inference_layer_k2(output0, hx[1], 1)
        return output1, torch.cat((h0, h1), dim=0)

    def _forward_inference_k2_3(self, x: torch.Tensor, hx: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        output0, h0 = self._forward_inference_layer_k2(x, hx[0], 0)
        output1, h1 = self._forward_inference_layer_k2(output0, hx[1], 1)
        output2, h2 = self._forward_inference_layer_k2(output1, hx[2], 2)
        return output2, torch.cat((h0, h1, h2), dim=0)

    def _forward_inference_k2_4(self, x: torch.Tensor, hx: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        output0, h0 = self._forward_inference_layer_k2(x, hx[0], 0)
        output1, h1 = self._forward_inference_layer_k2(output0, hx[1], 1)
        output2, h2 = self._forward_inference_layer_k2(output1, hx[2], 2)
        output3, h3 = self._forward_inference_layer_k2(output2, hx[3], 3)
        return output3, torch.cat((h0, h1, h2, h3), dim=0)

    def _forward_inference_k2(self, x: torch.Tensor, hx: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.num_layers == 1:
            return self._forward_inference_k2_1(x, hx)
        if self.num_layers == 2:
            return self._forward_inference_k2_2(x, hx)
        if self.num_layers == 3:
            return self._forward_inference_k2_3(x, hx)
        return self._forward_inference_k2_4(x, hx)

    def _forward_inference_k1_1(self, x: torch.Tensor, hx: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        output0, h0 = self._forward_inference_layer_k1(x, hx[0], 0)
        return output0, h0

    def _forward_inference_k1_2(self, x: torch.Tensor, hx: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        output0, h0 = self._forward_inference_layer_k1(x, hx[0], 0)
        output1, h1 = self._forward_inference_layer_k1(output0, hx[1], 1)
        return output1, torch.cat((h0, h1), dim=0)

    def _forward_inference_k1_3(self, x: torch.Tensor, hx: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        output0, h0 = self._forward_inference_layer_k1(x, hx[0], 0)
        output1, h1 = self._forward_inference_layer_k1(output0, hx[1], 1)
        output2, h2 = self._forward_inference_layer_k1(output1, hx[2], 2)
        return output2, torch.cat((h0, h1, h2), dim=0)

    def _forward_inference_k1_4(self, x: torch.Tensor, hx: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        output0, h0 = self._forward_inference_layer_k1(x, hx[0], 0)
        output1, h1 = self._forward_inference_layer_k1(output0, hx[1], 1)
        output2, h2 = self._forward_inference_layer_k1(output1, hx[2], 2)
        output3, h3 = self._forward_inference_layer_k1(output2, hx[3], 3)
        return output3, torch.cat((h0, h1, h2, h3), dim=0)

    def _forward_inference_k1(self, x: torch.Tensor, hx: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.num_layers == 1:
            return self._forward_inference_k1_1(x, hx)
        if self.num_layers == 2:
            return self._forward_inference_k1_2(x, hx)
        if self.num_layers == 3:
            return self._forward_inference_k1_3(x, hx)
        return self._forward_inference_k1_4(x, hx)

    def _forward_inference(self, x: torch.Tensor, hx: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.num_layers == 1:
            return self._forward_inference_1(x, hx)
        if self.num_layers == 2:
            return self._forward_inference_2(x, hx)
        if self.num_layers == 3:
            return self._forward_inference_3(x, hx)
        return self._forward_inference_4(x, hx)

    def _forward_inference_output_only_1(self, x: torch.Tensor, hx: torch.Tensor) -> torch.Tensor:
        return self._forward_inference_layer(x, hx[0], 0, output_only=True)

    def _forward_inference_output_only_2(self, x: torch.Tensor, hx: torch.Tensor) -> torch.Tensor:
        output0 = self._forward_inference_layer(x, hx[0], 0, output_only=True)
        return self._forward_inference_layer(output0, hx[1], 1, output_only=True)

    def _forward_inference_output_only_3(self, x: torch.Tensor, hx: torch.Tensor) -> torch.Tensor:
        output0 = self._forward_inference_layer(x, hx[0], 0, output_only=True)
        output1 = self._forward_inference_layer(output0, hx[1], 1, output_only=True)
        return self._forward_inference_layer(output1, hx[2], 2, output_only=True)

    def _forward_inference_output_only_4(self, x: torch.Tensor, hx: torch.Tensor) -> torch.Tensor:
        output0 = self._forward_inference_layer(x, hx[0], 0, output_only=True)
        output1 = self._forward_inference_layer(output0, hx[1], 1, output_only=True)
        output2 = self._forward_inference_layer(output1, hx[2], 2, output_only=True)
        return self._forward_inference_layer(output2, hx[3], 3, output_only=True)

    def _forward_inference_output_only(self, x: torch.Tensor, hx: torch.Tensor) -> torch.Tensor:
        if self.num_layers == 1:
            return self._forward_inference_output_only_1(x, hx)
        if self.num_layers == 2:
            return self._forward_inference_output_only_2(x, hx)
        if self.num_layers == 3:
            return self._forward_inference_output_only_3(x, hx)
        return self._forward_inference_output_only_4(x, hx)

    def forward(
        self,
        x: torch.Tensor,
        hx: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        hx = self._prepare_hx(x, hx)
        if not torch.is_grad_enabled():
            return self._forward_inference(x, hx)
        return self._forward_train(x, hx)

    def forward_inference(
        self,
        x: torch.Tensor,
        hx: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """显式运行不保存 backward gate cache 的推理路径。"""
        hx = self._prepare_hx(x, hx)
        return self._forward_inference(x, hx)

    def forward_inference_ldg(
        self,
        x: torch.Tensor,
        hx: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """实验性 no-cache 推理路径，对只读 input/weight/bias 使用 __ldg。"""
        hx = self._prepare_hx(x, hx)
        return self._forward_inference_ldg(x, hx)

    def forward_inference_k2(
        self,
        x: torch.Tensor,
        hx: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """实验性 no-cache 推理路径，把 recurrent K 维从 4 个 CTA 改为 2 个 CTA。"""
        hx = self._prepare_hx(x, hx)
        return self._forward_inference_k2(x, hx)

    def forward_inference_k1(
        self,
        x: torch.Tensor,
        hx: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """实验性 no-cache 推理路径，把 recurrent K 维压到单个 CTA。"""
        hx = self._prepare_hx(x, hx)
        return self._forward_inference_k1(x, hx)

    def forward_inference_k1_htile8(
        self,
        x: torch.Tensor,
        hx: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """实验性 htile8/K1 no-cache 推理路径，面向 batch=16 提高 CTA 数量。"""
        hx = self._prepare_hx(x, hx)
        return self._forward_inference_k1_htile8(x, hx)

    def forward_inference_stacked_naive(
        self,
        x: torch.Tensor,
        hx: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """实验性多层 fused forward-only kernel，用于验证新的 layer 2/3/4 调度。"""
        hx = self._prepare_hx(x, hx)
        layer_params = tuple(
            tuple(param.contiguous() for param in self._layer_parameters(layer))
            for layer in range(self.num_layers)
        )
        with torch.no_grad():
            return a100_gru_h256_stacked_forward_naive(
                x.contiguous(),
                hx.contiguous(),
                layer_params,
                block_threads=256,
            )

    def forward_inference_stacked_row4(
        self,
        x: torch.Tensor,
        hx: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """实验性多层 fused row4 forward-only kernel。"""
        hx = self._prepare_hx(x, hx)
        layer_params = tuple(
            tuple(param.contiguous() for param in self._layer_parameters(layer))
            for layer in range(self.num_layers)
        )
        with torch.no_grad():
            return a100_gru_h256_stacked_row4_forward(
                x.contiguous(),
                hx.contiguous(),
                layer_params,
                block_threads=256,
            )

    def forward_train_stacked_row4_cache(
        self,
        x: torch.Tensor,
        hx: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """实验性多层 fused 训练 forward，返回 output、h_n、all_outputs 和 gate cache。"""
        hx = self._prepare_hx(x, hx)
        layer_params = tuple(
            tuple(param.contiguous() for param in self._layer_parameters(layer))
            for layer in range(self.num_layers)
        )
        return a100_gru_h256_stacked_row4_train_forward(
            x.contiguous(),
            hx.contiguous(),
            layer_params,
            block_threads=256,
        )

    def forward_train_stacked_row4_k1_cache(
        self,
        x: torch.Tensor,
        hx: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """实验性多层 fused K1 训练 forward，返回 output、h_n、all_outputs 和 gate cache。"""
        hx = self._prepare_hx(x, hx)
        layer_params = tuple(
            tuple(param.contiguous() for param in self._layer_parameters(layer))
            for layer in range(self.num_layers)
        )
        return a100_gru_h256_stacked_row4_k1_train_forward(
            x.contiguous(),
            hx.contiguous(),
            layer_params,
            block_threads=256,
        )

    def forward_train_stacked_fused_naive(
        self,
        x: torch.Tensor,
        hx: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """实验性多层 fused forward/backward 原型，当前只用于正确性和调度研究。"""
        hx = self._prepare_hx(x, hx)
        flat_params: list[torch.Tensor] = []
        for layer in range(self.num_layers):
            flat_params.extend(self._layer_parameters(layer))
        return A100GRUH256StackedFusedNaiveFunction.apply(
            x.contiguous(),
            hx.contiguous(),
            self.num_layers,
            0,
            0,
            *flat_params,
        )

    def forward_train_stacked_fused_split4(
        self,
        x: torch.Tensor,
        hx: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """实验性多层 fused forward/backward split4 路径，提升 fused backward 并行度。"""
        hx = self._prepare_hx(x, hx)
        flat_params: list[torch.Tensor] = []
        for layer in range(self.num_layers):
            flat_params.extend(self._layer_parameters(layer))
        return A100GRUH256StackedFusedNaiveFunction.apply(
            x.contiguous(),
            hx.contiguous(),
            self.num_layers,
            1,
            0,
            *flat_params,
        )

    def forward_train_stacked_fused_split8(
        self,
        x: torch.Tensor,
        hx: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """实验性多层 fused forward/backward split8 路径，继续提高 CTA 并行度。"""
        hx = self._prepare_hx(x, hx)
        flat_params: list[torch.Tensor] = []
        for layer in range(self.num_layers):
            flat_params.extend(self._layer_parameters(layer))
        return A100GRUH256StackedFusedNaiveFunction.apply(
            x.contiguous(),
            hx.contiguous(),
            self.num_layers,
            2,
            0,
            *flat_params,
        )

    def forward_train_stacked_fused_split4_shmem(
        self,
        x: torch.Tensor,
        hx: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """实验性 fused split4 变体，用 shared memory 缓存本 CTA 的 gate 梯度。"""
        hx = self._prepare_hx(x, hx)
        flat_params: list[torch.Tensor] = []
        for layer in range(self.num_layers):
            flat_params.extend(self._layer_parameters(layer))
        return A100GRUH256StackedFusedNaiveFunction.apply(
            x.contiguous(),
            hx.contiguous(),
            self.num_layers,
            3,
            0,
            *flat_params,
        )

    def forward_train_stacked_fused_split4_k1_forward(
        self,
        x: torch.Tensor,
        hx: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """实验性 fused 路径，K1 stacked forward + split4 backward。"""
        hx = self._prepare_hx(x, hx)
        flat_params: list[torch.Tensor] = []
        for layer in range(self.num_layers):
            flat_params.extend(self._layer_parameters(layer))
        return A100GRUH256StackedFusedNaiveFunction.apply(
            x.contiguous(),
            hx.contiguous(),
            self.num_layers,
            1,
            1,
            *flat_params,
        )

    def forward_train_stacked_fused_split4_group8(
        self,
        x: torch.Tensor,
        hx: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """实验性 fused 路径，K1 stacked forward + split4/group8 backward。"""
        hx = self._prepare_hx(x, hx)
        flat_params: list[torch.Tensor] = []
        for layer in range(self.num_layers):
            flat_params.extend(self._layer_parameters(layer))
        return A100GRUH256StackedFusedNaiveFunction.apply(
            x.contiguous(),
            hx.contiguous(),
            self.num_layers,
            4,
            1,
            *flat_params,
        )

    def forward_train_stacked_fused_split6_weight_shmem(
        self,
        x: torch.Tensor,
        hx: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """实验性 fused 路径，K1 forward + split6 weight-shmem backward。"""
        hx = self._prepare_hx(x, hx)
        flat_params: list[torch.Tensor] = []
        for layer in range(self.num_layers):
            flat_params.extend(self._layer_parameters(layer))
        return A100GRUH256StackedFusedNaiveFunction.apply(
            x.contiguous(),
            hx.contiguous(),
            self.num_layers,
            5,
            1,
            *flat_params,
        )

    def forward_train_stacked_fused_split6_hybrid_forward(
        self,
        x: torch.Tensor,
        hx: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """实验性 fused 路径，cuBLAS input projection + K1 recurrent forward + split6 backward。"""
        hx = self._prepare_hx(x, hx)
        flat_params: list[torch.Tensor] = []
        for layer in range(self.num_layers):
            flat_params.extend(self._layer_parameters(layer))
        return A100GRUH256StackedFusedNaiveFunction.apply(
            x.contiguous(),
            hx.contiguous(),
            self.num_layers,
            5,
            2,
            *flat_params,
        )

    def forward_output_only(
        self,
        x: torch.Tensor,
        hx: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """显式运行只返回 output 的实验路径，适合调用方完全不使用 h_n 的训练。"""
        hx = self._prepare_hx(x, hx)
        if not torch.is_grad_enabled():
            return self._forward_inference_output_only(x, hx)
        return self._forward_train_output_only(x, hx)

    @classmethod
    def from_torch_gru(cls, gru: nn.GRU) -> A100GRUH256:
        """从 torch.nn.GRU 创建同权重的实验 A100GRUH256。"""
        return from_torch_gru(gru)


def is_supported_gru(gru: nn.GRU) -> bool:
    """判断 torch.nn.GRU 是否能无损替换为实验 A100 h256 实现。"""
    if not isinstance(gru, nn.GRU) or not gru.bias:
        return False
    try:
        first_param = next(gru.parameters())
    except StopIteration:
        return False
    return (
        1 <= gru.input_size <= MAX_INPUT_SIZE
        and gru.hidden_size == 256
        and 1 <= gru.num_layers <= MAX_NUM_LAYERS
        and gru.batch_first
        and not gru.bidirectional
        and gru.dropout == 0.0
        and first_param.dtype == torch.float32
    )


def from_torch_gru(gru: nn.GRU) -> A100GRUH256:
    """从支持范围内的 torch.nn.GRU 创建当前实验最优组合。"""
    if not is_supported_gru(gru):
        raise ValueError(
            "Only input_size<=16, 1-4 layer unidirectional batch_first fp32 GRU "
            "with hidden_size=256, dropout=0 and bias=True is supported."
        )
    device = next(gru.parameters()).device
    module = A100GRUH256(
        input_size=gru.input_size,
        num_layers=gru.num_layers,
        dropout=gru.dropout,
        block_threads=256,
        use_persistent_state6_gate_cache_tiled_weight_shmem_split0_keep_unroll8_backward_kernel=True,
        use_gate_cache_htile4_compact_hoist_row4_k1_forward_kernel=True,
        use_pack_hidden_prev_time_major_kernel=True,
    ).to(device=device)
    copy_from_torch_gru(module, gru)
    return module


def copy_from_torch_gru(a100_gru: A100GRUH256, torch_gru: nn.GRU) -> None:
    if torch_gru.bidirectional:
        raise ValueError("Bidirectional GRU is not supported.")
    if not torch_gru.batch_first:
        raise ValueError("Only batch_first=True GRU is supported.")
    if not torch_gru.bias:
        raise ValueError("Only bias=True GRU is supported.")
    if torch_gru.dropout != 0.0:
        raise ValueError("Only dropout=0 GRU is supported.")
    if not 1 <= torch_gru.num_layers <= MAX_NUM_LAYERS:
        raise ValueError("Only 1 <= num_layers <= 4 GRU is supported.")
    if torch_gru.num_layers != a100_gru.num_layers:
        raise ValueError("num_layers mismatch.")
    if torch_gru.hidden_size != 256:
        raise ValueError("Only hidden_size=256 GRU is supported.")
    if not 1 <= torch_gru.input_size <= MAX_INPUT_SIZE:
        raise ValueError("Only 1 <= input_size <= 16 GRU is supported.")
    if torch_gru.input_size != a100_gru.input_size:
        raise ValueError("input_size mismatch.")

    with torch.no_grad():
        for layer in range(torch_gru.num_layers):
            getattr(a100_gru, f"weight_ih_l{layer}").copy_(getattr(torch_gru, f"weight_ih_l{layer}"))
            getattr(a100_gru, f"weight_hh_l{layer}").copy_(getattr(torch_gru, f"weight_hh_l{layer}"))
            getattr(a100_gru, f"bias_ih_l{layer}").copy_(getattr(torch_gru, f"bias_ih_l{layer}"))
            getattr(a100_gru, f"bias_hh_l{layer}").copy_(getattr(torch_gru, f"bias_hh_l{layer}"))
