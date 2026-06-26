from __future__ import annotations

import ctypes
import math
from importlib import resources
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from cuda import cuda
except ImportError:  # pragma: no cover - 运行时给出更清晰错误。
    cuda = None


BEST_BLOCK_THREADS = 256
HIDDEN_SIZE = 256
_CUBIN_PACKAGE = "a100_gru_h256.kernels"
_CUBIN_NAME = "a100_gru_h256_sm80.cubin"
_SOURCE_TREE_CUBIN_PATH = Path(__file__).resolve().parent / "kernels" / _CUBIN_NAME

_FORWARD_KERNEL_NAME = (
    b"a100_gru_forward_from_gates_cooperative_h256_htile4_compact_hoist_row4_shmem_gate_cache_kernel"
)
_FORWARD_INFERENCE_KERNEL_NAME = (
    b"a100_gru_forward_from_gates_cooperative_h256_htile4_compact_hoist_row4_no_cache_kernel"
)
_BACKWARD_KERNEL_NAME = (
    b"a100_gru_h256_backward_sequence_cooperative_split6_gate_cache_state_tiled_weight_shmem_split0_keep_kernel"
)
_PACK_KERNEL_NAME = b"a100_gru_h256_pack_hidden_prev_time_major_kernel"


@dataclass(frozen=True)
class _ProdKernels:
    module: object
    forward_function: object
    forward_inference_function: object
    backward_function: object
    pack_function: object


def _require_cuda_python() -> None:
    if cuda is None:
        raise RuntimeError(
            "a100_gru_h256 requires cuda-python at runtime to load prebuilt cubin."
        )


def _check_cuda_error(err: object, detail: str = "") -> None:
    _require_cuda_python()
    if isinstance(err, cuda.CUresult):
        if err == cuda.CUresult.CUDA_SUCCESS:
            return
        raise RuntimeError(f"CUDA driver error: {err}. {detail}".strip())
    raise RuntimeError(f"Unknown CUDA error type: {err}. {detail}".strip())


def _device_capability(device_index: int) -> tuple[int, int]:
    _require_cuda_python()
    err, = cuda.cuInit(0)
    _check_cuda_error(err)
    err, device = cuda.cuDeviceGet(device_index)
    _check_cuda_error(err)
    err, major = cuda.cuDeviceGetAttribute(
        cuda.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
        device,
    )
    _check_cuda_error(err)
    err, minor = cuda.cuDeviceGetAttribute(
        cuda.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,
        device,
    )
    _check_cuda_error(err)
    return major, minor


def _device_attribute(device_index: int, attribute: object) -> int:
    _require_cuda_python()
    err, = cuda.cuInit(0)
    _check_cuda_error(err)
    err, device = cuda.cuDeviceGet(device_index)
    _check_cuda_error(err)
    err, value = cuda.cuDeviceGetAttribute(attribute, device)
    _check_cuda_error(err)
    return value


def _device_index_from_optional(device: Optional[torch.device | int | str]) -> int:
    if device is None:
        return torch.cuda.current_device()
    if isinstance(device, int):
        return device
    torch_device = torch.device(device)
    if torch_device.type != "cuda":
        raise ValueError("device must be a CUDA device.")
    if torch_device.index is None:
        return torch.cuda.current_device()
    return torch_device.index


def is_a100_available(device: Optional[torch.device | int | str] = None) -> bool:
    """检查当前设备是否为 A100/SM80。"""
    if not torch.cuda.is_available():
        return False
    try:
        device_index = _device_index_from_optional(device)
    except ValueError:
        return False
    return torch.cuda.get_device_capability(device_index) == (8, 0)


def is_supported_gru(gru: nn.GRU) -> bool:
    """判断 torch.nn.GRU 是否能无损替换为当前 A100 h256 实现。"""
    if not isinstance(gru, nn.GRU) or not gru.bias:
        return False
    try:
        first_param = next(gru.parameters())
    except StopIteration:
        return False
    return (
        gru.input_size > 0
        and gru.hidden_size == HIDDEN_SIZE
        and gru.num_layers == 1
        and gru.batch_first
        and not gru.bidirectional
        and first_param.dtype == torch.float32
    )


def _validate_a100_device(device_index: int) -> None:
    capability = _device_capability(device_index)
    if capability != (8, 0):
        raise RuntimeError(f"A100GRU prod requires SM80/A100, got sm_{capability[0]}{capability[1]}.")
    cooperative_launch = _device_attribute(
        device_index,
        cuda.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_COOPERATIVE_LAUNCH,
    )
    if cooperative_launch == 0:
        raise RuntimeError("Current CUDA device does not support cooperative launch.")


@lru_cache(maxsize=None)
def _load_prod_kernels(device_index: int) -> _ProdKernels:
    _require_cuda_python()
    cubin_resource = resources.files(_CUBIN_PACKAGE).joinpath(_CUBIN_NAME)
    if not cubin_resource.is_file():
        raise RuntimeError(
            "Prebuilt A100 prod cubin is missing. Run "
            "`python a100_gru_h256/scripts/build_cubin.py` in the build environment and include "
            f"`{_SOURCE_TREE_CUBIN_PATH}` in the package."
        )
    torch.cuda.set_device(device_index)
    err, = cuda.cuInit(0)
    _check_cuda_error(err)
    image = cubin_resource.read_bytes()
    err, module = cuda.cuModuleLoadData(image)
    _check_cuda_error(err, f"{_CUBIN_PACKAGE}/{_CUBIN_NAME}")
    err, forward_function = cuda.cuModuleGetFunction(module, _FORWARD_KERNEL_NAME)
    _check_cuda_error(err)
    err, forward_inference_function = cuda.cuModuleGetFunction(module, _FORWARD_INFERENCE_KERNEL_NAME)
    _check_cuda_error(err)
    err, backward_function = cuda.cuModuleGetFunction(module, _BACKWARD_KERNEL_NAME)
    _check_cuda_error(err)
    err, pack_function = cuda.cuModuleGetFunction(module, _PACK_KERNEL_NAME)
    _check_cuda_error(err)
    return _ProdKernels(
        module=module,
        forward_function=forward_function,
        forward_inference_function=forward_inference_function,
        backward_function=backward_function,
        pack_function=pack_function,
    )


def _validate_inputs(
    x: torch.Tensor,
    h0: torch.Tensor,
    weight_ih: torch.Tensor,
    weight_hh: torch.Tensor,
    bias_ih: torch.Tensor,
    bias_hh: torch.Tensor,
) -> tuple[int, int, int, int]:
    if not x.is_cuda:
        raise RuntimeError("A100GRU prod requires CUDA tensors.")
    if x.dtype != torch.float32:
        raise RuntimeError("A100GRU prod currently supports fp32 only.")
    if x.dim() != 3:
        raise ValueError("x must have shape [batch, seq, input].")
    if h0.dim() != 2:
        raise ValueError("h0 must have shape [batch, hidden].")
    if h0.size(1) != HIDDEN_SIZE:
        raise ValueError("A100GRU prod requires hidden_size=256.")

    tensors = (x, h0, weight_ih, weight_hh, bias_ih, bias_hh)
    devices = {tensor.device for tensor in tensors}
    if len(devices) != 1:
        raise ValueError("All tensors must be on the same CUDA device.")
    if any(tensor.dtype != torch.float32 for tensor in tensors):
        raise RuntimeError("A100GRU prod currently supports fp32 only.")

    batch_size, seq_len, input_size = x.shape
    if h0.size(0) != batch_size:
        raise ValueError("h0 batch dimension must match x.")
    if weight_ih.shape != (3 * HIDDEN_SIZE, input_size):
        raise ValueError("weight_ih must have shape [3 * hidden, input].")
    if weight_hh.shape != (3 * HIDDEN_SIZE, HIDDEN_SIZE):
        raise ValueError("weight_hh must have shape [3 * hidden, hidden].")
    if bias_ih.shape != (3 * HIDDEN_SIZE,) or bias_hh.shape != (3 * HIDDEN_SIZE,):
        raise ValueError("bias tensors must have shape [3 * hidden].")

    device_index = x.device.index
    if device_index is None:
        device_index = torch.cuda.current_device()
    return batch_size, seq_len, input_size, device_index


def _launch_forward(
    input_gates: torch.Tensor,
    h0: torch.Tensor,
    weight_hh: torch.Tensor,
    bias_hh: torch.Tensor,
    device_index: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch_size, seq_len, hidden3 = input_gates.shape
    hidden_tile = HIDDEN_SIZE // 4
    ctas_per_batch = 16
    compact_partials_per_batch = 12
    output = torch.empty(batch_size, seq_len, HIDDEN_SIZE, device=input_gates.device, dtype=input_gates.dtype)
    gate_cache = torch.empty(batch_size, seq_len, 4 * HIDDEN_SIZE, device=input_gates.device, dtype=input_gates.dtype)
    partial_gates = torch.empty(
        batch_size * compact_partials_per_batch,
        3 * hidden_tile,
        device=input_gates.device,
        dtype=input_gates.dtype,
    )
    hidden_state = torch.empty(batch_size, HIDDEN_SIZE, device=input_gates.device, dtype=input_gates.dtype)

    kernels = _load_prod_kernels(device_index)
    block_threads = BEST_BLOCK_THREADS
    shared_bytes = 3 * hidden_tile * ctypes.sizeof(ctypes.c_float)
    err, active_blocks_per_sm = cuda.cuOccupancyMaxActiveBlocksPerMultiprocessor(
        kernels.forward_function,
        block_threads,
        shared_bytes,
    )
    _check_cuda_error(err)
    sm_count = _device_attribute(device_index, cuda.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)
    grid_blocks = batch_size * ctas_per_batch
    max_cooperative_blocks = active_blocks_per_sm * sm_count
    if grid_blocks > max_cooperative_blocks:
        raise ValueError(
            "A100GRU prod forward cooperative grid is too large for resident launch: "
            f"grid_blocks={grid_blocks}, max={max_cooperative_blocks}."
        )

    stream = cuda.CUstream(torch.cuda.current_stream(device=input_gates.device).cuda_stream)
    values = (
        ctypes.c_void_p(input_gates.data_ptr()),
        ctypes.c_void_p(h0.data_ptr()),
        ctypes.c_void_p(weight_hh.data_ptr()),
        ctypes.c_void_p(bias_hh.data_ptr()),
        ctypes.c_void_p(partial_gates.data_ptr()),
        ctypes.c_void_p(hidden_state.data_ptr()),
        ctypes.c_void_p(output.data_ptr()),
        ctypes.c_void_p(gate_cache.data_ptr()),
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
        ctypes.c_int,
    )
    err, = cuda.cuLaunchCooperativeKernel(
        kernels.forward_function,
        grid_blocks,
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
    return output, gate_cache


def _launch_forward_no_cache(
    input_gates: torch.Tensor,
    h0: torch.Tensor,
    weight_hh: torch.Tensor,
    bias_hh: torch.Tensor,
    device_index: int,
) -> torch.Tensor:
    batch_size, seq_len, hidden3 = input_gates.shape
    hidden_tile = HIDDEN_SIZE // 4
    ctas_per_batch = 16
    compact_partials_per_batch = 12
    output = torch.empty(batch_size, seq_len, HIDDEN_SIZE, device=input_gates.device, dtype=input_gates.dtype)
    partial_gates = torch.empty(
        batch_size * compact_partials_per_batch,
        3 * hidden_tile,
        device=input_gates.device,
        dtype=input_gates.dtype,
    )
    hidden_state = torch.empty(batch_size, HIDDEN_SIZE, device=input_gates.device, dtype=input_gates.dtype)

    kernels = _load_prod_kernels(device_index)
    block_threads = BEST_BLOCK_THREADS
    shared_bytes = 3 * hidden_tile * ctypes.sizeof(ctypes.c_float)
    err, active_blocks_per_sm = cuda.cuOccupancyMaxActiveBlocksPerMultiprocessor(
        kernels.forward_inference_function,
        block_threads,
        shared_bytes,
    )
    _check_cuda_error(err)
    sm_count = _device_attribute(device_index, cuda.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)
    grid_blocks = batch_size * ctas_per_batch
    max_cooperative_blocks = active_blocks_per_sm * sm_count
    if grid_blocks > max_cooperative_blocks:
        raise ValueError(
            "A100GRU prod inference cooperative grid is too large for resident launch: "
            f"grid_blocks={grid_blocks}, max={max_cooperative_blocks}."
        )

    stream = cuda.CUstream(torch.cuda.current_stream(device=input_gates.device).cuda_stream)
    values = (
        ctypes.c_void_p(input_gates.data_ptr()),
        ctypes.c_void_p(h0.data_ptr()),
        ctypes.c_void_p(weight_hh.data_ptr()),
        ctypes.c_void_p(bias_hh.data_ptr()),
        ctypes.c_void_p(partial_gates.data_ptr()),
        ctypes.c_void_p(hidden_state.data_ptr()),
        ctypes.c_void_p(output.data_ptr()),
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
    )
    err, = cuda.cuLaunchCooperativeKernel(
        kernels.forward_inference_function,
        grid_blocks,
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
    return output


def _launch_pack_hidden_prev(h0: torch.Tensor, output: torch.Tensor, device_index: int) -> torch.Tensor:
    batch_size, seq_len, hidden_size = output.shape
    if hidden_size != HIDDEN_SIZE:
        raise ValueError("A100GRU prod hidden-prev pack requires hidden_size=256.")
    hidden_prev_steps = torch.empty(seq_len, batch_size, HIDDEN_SIZE, device=output.device, dtype=output.dtype)

    kernels = _load_prod_kernels(device_index)
    block_threads = BEST_BLOCK_THREADS
    total_vec4 = seq_len * batch_size * (HIDDEN_SIZE // 4)
    grid_blocks = min(math.ceil(total_vec4 / block_threads), 65535)
    stream = cuda.CUstream(torch.cuda.current_stream(device=output.device).cuda_stream)
    values = (
        ctypes.c_void_p(h0.data_ptr()),
        ctypes.c_void_p(output.data_ptr()),
        ctypes.c_void_p(hidden_prev_steps.data_ptr()),
        ctypes.c_int(batch_size),
        ctypes.c_int(seq_len),
    )
    types = (ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int)
    err, = cuda.cuLaunchKernel(
        kernels.pack_function,
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


def _launch_backward_sequence(
    grad_output: torch.Tensor,
    grad_hidden_state: torch.Tensor,
    gate_cache: torch.Tensor,
    h0: torch.Tensor,
    output: torch.Tensor,
    weight_hh: torch.Tensor,
    grad_input_gates: torch.Tensor,
    grad_hidden_gates_steps: torch.Tensor,
    partial_sums: torch.Tensor,
    device_index: int,
) -> torch.Tensor:
    batch_size, seq_len, cache_size = gate_cache.shape
    if cache_size != 4 * HIDDEN_SIZE:
        raise ValueError("gate_cache must have shape [batch, seq, 4 * hidden].")

    kernels = _load_prod_kernels(device_index)
    shared_floats = 128 + 128 * HIDDEN_SIZE
    shared_bytes = shared_floats * ctypes.sizeof(ctypes.c_float)
    err, = cuda.cuFuncSetAttribute(
        kernels.backward_function,
        cuda.CUfunction_attribute.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
        shared_bytes,
    )
    _check_cuda_error(err)
    err, = cuda.cuFuncSetAttribute(
        kernels.backward_function,
        cuda.CUfunction_attribute.CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT,
        100,
    )
    _check_cuda_error(err)

    stream = cuda.CUstream(torch.cuda.current_stream(device=gate_cache.device).cuda_stream)
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
        kernels.backward_function,
        batch_size * 6,
        1,
        1,
        BEST_BLOCK_THREADS,
        1,
        1,
        shared_bytes,
        stream,
        (values, types),
    )
    _check_cuda_error(err)
    return grad_hidden_state


class _A100GRUFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        h0: torch.Tensor,
        weight_ih: torch.Tensor,
        weight_hh: torch.Tensor,
        bias_ih: torch.Tensor,
        bias_hh: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len, input_size, device_index = _validate_inputs(
            x,
            h0,
            weight_ih,
            weight_hh,
            bias_ih,
            bias_hh,
        )
        torch.cuda.set_device(device_index)
        _validate_a100_device(device_index)

        x = x.contiguous()
        h0 = h0.contiguous()
        weight_ih = weight_ih.contiguous()
        weight_hh = weight_hh.contiguous()
        bias_ih = bias_ih.contiguous()
        bias_hh = bias_hh.contiguous()

        input_gates = F.linear(
            x.reshape(batch_size * seq_len, input_size),
            weight_ih,
            bias_ih,
        ).view(batch_size, seq_len, 3 * HIDDEN_SIZE)
        output, gate_cache = _launch_forward(input_gates, h0, weight_hh, bias_hh, device_index)
        h_n = output[:, -1, :].unsqueeze(0).contiguous()

        ctx.save_for_backward(x, h0, weight_ih, weight_hh, input_gates, output, gate_cache)
        ctx.device_index = device_index
        return output, h_n

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor, grad_h_n: torch.Tensor):
        x, h0, weight_ih, weight_hh, input_gates, output, gate_cache = ctx.saved_tensors
        if grad_output is None:
            grad_output = torch.zeros_like(output)
        grad_output = grad_output.contiguous()
        grad_h_n = grad_h_n.contiguous() if grad_h_n is not None else None

        batch_size, seq_len, input_size = x.shape
        device_index = ctx.device_index
        hidden_prev_steps = _launch_pack_hidden_prev(h0, output, device_index)

        grad_input_gates = torch.empty_like(input_gates)
        grad_hidden_gates_steps = torch.empty(
            seq_len,
            batch_size,
            3 * HIDDEN_SIZE,
            device=x.device,
            dtype=x.dtype,
        )
        partial_sums = torch.empty(batch_size, 6, HIDDEN_SIZE, device=x.device, dtype=x.dtype)
        grad_hidden = torch.zeros_like(h0)
        if grad_h_n is not None:
            grad_hidden = grad_hidden + grad_h_n[0]

        grad_hidden = _launch_backward_sequence(
            grad_output,
            grad_hidden,
            gate_cache,
            h0,
            output,
            weight_hh,
            grad_input_gates,
            grad_hidden_gates_steps,
            partial_sums,
            device_index,
        )

        grad_hidden_gates_2d = grad_hidden_gates_steps.reshape(seq_len * batch_size, 3 * HIDDEN_SIZE)
        hidden_prev_2d = hidden_prev_steps.reshape(seq_len * batch_size, HIDDEN_SIZE)
        grad_weight_hh = grad_hidden_gates_2d.transpose(0, 1).matmul(hidden_prev_2d)
        grad_bias_hh = grad_hidden_gates_2d.sum(dim=0)

        grad_input_gates_2d = grad_input_gates.reshape(batch_size * seq_len, 3 * HIDDEN_SIZE)
        x_2d = x.reshape(batch_size * seq_len, input_size)
        grad_x = grad_input_gates_2d.matmul(weight_ih).view(batch_size, seq_len, input_size)
        grad_weight_ih = grad_input_gates_2d.transpose(0, 1).matmul(x_2d)
        grad_bias_ih = grad_input_gates_2d.sum(dim=0)

        return grad_x, grad_hidden, grad_weight_ih, grad_weight_hh, grad_bias_ih, grad_bias_hh


def _a100_gru_forward_inference(
    x: torch.Tensor,
    h0: torch.Tensor,
    weight_ih: torch.Tensor,
    weight_hh: torch.Tensor,
    bias_ih: torch.Tensor,
    bias_hh: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch_size, seq_len, input_size, device_index = _validate_inputs(
        x,
        h0,
        weight_ih,
        weight_hh,
        bias_ih,
        bias_hh,
    )
    torch.cuda.set_device(device_index)
    _validate_a100_device(device_index)

    with torch.no_grad():
        x = x.contiguous()
        h0 = h0.contiguous()
        weight_ih = weight_ih.contiguous()
        weight_hh = weight_hh.contiguous()
        bias_ih = bias_ih.contiguous()
        bias_hh = bias_hh.contiguous()

        input_gates = F.linear(
            x.reshape(batch_size * seq_len, input_size),
            weight_ih,
            bias_ih,
        ).view(batch_size, seq_len, 3 * HIDDEN_SIZE)
        output = _launch_forward_no_cache(input_gates, h0, weight_hh, bias_hh, device_index)
        h_n = output[:, -1, :].unsqueeze(0).contiguous()
    return output, h_n


class A100GRU(nn.Module):
    """固定为当前最快 A100 h256 训练路径的独立 prod GRU 模块。"""

    def __init__(
        self,
        input_size: int,
        hidden_size: int = HIDDEN_SIZE,
        num_layers: int = 1,
        batch_first: bool = True,
        bias: bool = True,
    ) -> None:
        super().__init__()
        if hidden_size != HIDDEN_SIZE:
            raise ValueError("A100GRU only supports hidden_size=256.")
        if num_layers != 1:
            raise ValueError("A100GRU only supports num_layers=1.")
        if not batch_first:
            raise ValueError("A100GRU only supports batch_first=True.")
        if not bias:
            raise ValueError("A100GRU currently requires bias=True.")

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bias = bias
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
        if not torch.is_grad_enabled():
            return _a100_gru_forward_inference(
                x,
                h0,
                self.weight_ih_l0,
                self.weight_hh_l0,
                self.bias_ih_l0,
                self.bias_hh_l0,
            )
        return _A100GRUFunction.apply(
            x,
            h0,
            self.weight_ih_l0,
            self.weight_hh_l0,
            self.bias_ih_l0,
            self.bias_hh_l0,
        )

    def forward_inference(
        self,
        x: torch.Tensor,
        hx: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """显式运行 forward-only no-cache 推理路径。"""
        if hx is None:
            h0 = x.new_zeros(x.size(0), self.hidden_size)
        else:
            if hx.shape != (1, x.size(0), self.hidden_size):
                raise ValueError("hx must have shape [1, batch, 256].")
            h0 = hx[0]
        return _a100_gru_forward_inference(
            x,
            h0,
            self.weight_ih_l0,
            self.weight_hh_l0,
            self.bias_ih_l0,
            self.bias_hh_l0,
        )

    @classmethod
    def from_torch_gru(cls, gru: nn.GRU) -> A100GRU:
        """从 torch.nn.GRU 创建同权重的 A100GRU。"""
        return from_torch_gru(gru)


A100GRUH256 = A100GRU


def from_torch_gru(gru: nn.GRU) -> A100GRU:
    """从支持范围内的 torch.nn.GRU 创建生产试用封装。"""
    if not is_supported_gru(gru):
        raise ValueError(
            "Only single-layer unidirectional batch_first fp32 GRU with hidden_size=256 "
            "and bias=True is supported."
        )
    device = next(gru.parameters()).device
    module = A100GRU(input_size=gru.input_size).to(device=device)
    with torch.no_grad():
        module.weight_ih_l0.copy_(gru.weight_ih_l0)
        module.weight_hh_l0.copy_(gru.weight_hh_l0)
        module.bias_ih_l0.copy_(gru.bias_ih_l0)
        module.bias_hh_l0.copy_(gru.bias_hh_l0)
    return module
