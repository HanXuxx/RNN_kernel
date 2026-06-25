from __future__ import annotations

import ctypes
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import torch
import torch.nn.functional as F

try:
    import nvidia.cuda_nvcc
    import nvidia.cuda_runtime
    from cuda import cuda, cudart, nvrtc
except ImportError:  # pragma: no cover - 由运行时错误给出更清晰的提示。
    cuda = None
    cudart = None
    nvrtc = None
    nvidia = None


_KERNEL_NAME = b"a100_gru_forward_layer_kernel"
_GATES_KERNEL_NAME = b"a100_gru_forward_from_gates_kernel"
_HALF_WARP_GATES_KERNEL_NAME = b"a100_gru_forward_from_gates_half_warp_kernel"
_QUARTER_WARP_GATES_KERNEL_NAME = b"a100_gru_forward_from_gates_quarter_warp_kernel"
_FUSED_HALF_WARP_GATES_KERNEL_NAME = b"a100_gru_forward_from_gates_fused_half_warp_kernel"
_FUSED_PINGPONG_GATES_KERNEL_NAME = b"a100_gru_forward_from_gates_fused_pingpong_half_warp_kernel"
_FUSED_SPECIALIZED_H128_GATES_KERNEL_NAME = (
    b"a100_gru_forward_from_gates_fused_specialized_h128_kernel"
)
_FUSED_SPECIALIZED_H130_GATES_KERNEL_NAME = (
    b"a100_gru_forward_from_gates_fused_specialized_h130_kernel"
)
_FUSED_SPECIALIZED_H160_GATES_KERNEL_NAME = (
    b"a100_gru_forward_from_gates_fused_specialized_h160_kernel"
)
_COOPERATIVE_GATES_KERNEL_NAME = b"a100_gru_forward_from_gates_cooperative_kernel"
_COOPERATIVE_H256_GATES_KERNEL_NAME = b"a100_gru_forward_from_gates_cooperative_h256_kernel"
_COOPERATIVE_H256_PARALLEL_UPDATE_GATES_KERNEL_NAME = (
    b"a100_gru_forward_from_gates_cooperative_h256_parallel_update_kernel"
)
_COOPERATIVE_H256_SHMEM_GATES_KERNEL_NAME = (
    b"a100_gru_forward_from_gates_cooperative_h256_shmem_kernel"
)
_COOPERATIVE_H256_QWARP_SHMEM_GATES_KERNEL_NAME = (
    b"a100_gru_forward_from_gates_cooperative_h256_qwarp_shmem_kernel"
)
_COOPERATIVE_H256_CACHED_SHMEM_GATES_KERNEL_NAME = (
    b"a100_gru_forward_from_gates_cooperative_h256_cached_shmem_kernel"
)
_SOURCE_PATH = Path(__file__).with_name("gru_forward_kernel.cu")


@dataclass(frozen=True)
class _CompiledKernel:
    module: object
    layer_function: object
    gates_function: object
    half_warp_gates_function: object
    quarter_warp_gates_function: object
    fused_half_warp_gates_function: object
    fused_pingpong_gates_function: object
    fused_specialized_h128_gates_function: object
    fused_specialized_h130_gates_function: object
    fused_specialized_h160_gates_function: object
    cooperative_gates_function: object
    cooperative_h256_gates_function: object
    cooperative_h256_parallel_update_gates_function: object
    cooperative_h256_shmem_gates_function: object
    cooperative_h256_qwarp_shmem_gates_function: object
    cooperative_h256_cached_shmem_gates_function: object


def _require_cuda_python() -> None:
    if cuda is None or nvrtc is None:
        raise RuntimeError(
            "a100_gru_forward_layer requires cuda-python and nvidia-cuda-nvcc-cu12 "
            "installed in the active virtual environment."
        )


def _check_cuda_error(err: object, detail: str = "") -> None:
    if cuda is not None and isinstance(err, cuda.CUresult):
        if err == cuda.CUresult.CUDA_SUCCESS:
            return
        raise RuntimeError(f"CUDA driver error: {err}. {detail}".strip())
    if cudart is not None and isinstance(err, cudart.cudaError_t):
        if err == cudart.cudaError_t.cudaSuccess:
            return
        raise RuntimeError(f"CUDA runtime error: {err}. {detail}".strip())
    if nvrtc is not None and isinstance(err, nvrtc.nvrtcResult):
        if err == nvrtc.nvrtcResult.NVRTC_SUCCESS:
            return
        raise RuntimeError(f"NVRTC error: {err}. {detail}".strip())
    raise RuntimeError(f"Unknown CUDA error type: {err}. {detail}".strip())


def _cuda_include_dirs() -> list[Path]:
    _require_cuda_python()
    return [
        Path(nvidia.cuda_nvcc.__file__).resolve().parent / "include",
        Path(nvidia.cuda_runtime.__file__).resolve().parent / "include",
    ]


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


def _compile_a100_kernel() -> _CompiledKernel:
    source = _SOURCE_PATH.read_text(encoding="utf-8")
    options = [
        b"--std=c++17",
        b"--gpu-architecture=sm_80",
    ]
    options.extend(
        f"--include-path={include_dir}".encode("utf-8")
        for include_dir in _cuda_include_dirs()
    )

    err, program = nvrtc.nvrtcCreateProgram(
        source.encode("utf-8"),
        str(_SOURCE_PATH.name).encode("utf-8"),
        0,
        [],
        [],
    )
    _check_cuda_error(err)

    err_compile, = nvrtc.nvrtcCompileProgram(program, len(options), options)
    err_log, log_size = nvrtc.nvrtcGetProgramLogSize(program)
    _check_cuda_error(err_log)
    log_buffer = b" " * log_size
    err_log, = nvrtc.nvrtcGetProgramLog(program, log_buffer)
    _check_cuda_error(err_log)
    compile_log = log_buffer.decode("utf-8", errors="replace").strip()
    if err_compile != nvrtc.nvrtcResult.NVRTC_SUCCESS:
        _check_cuda_error(err_compile, compile_log)

    err, image_size = nvrtc.nvrtcGetCUBINSize(program)
    _check_cuda_error(err)
    image = b" " * image_size
    err, = nvrtc.nvrtcGetCUBIN(program, image)
    _check_cuda_error(err)
    err, = nvrtc.nvrtcDestroyProgram(program)
    _check_cuda_error(err)

    err, module = cuda.cuModuleLoadData(image)
    _check_cuda_error(err)
    err, layer_function = cuda.cuModuleGetFunction(module, _KERNEL_NAME)
    _check_cuda_error(err)
    err, gates_function = cuda.cuModuleGetFunction(module, _GATES_KERNEL_NAME)
    _check_cuda_error(err)
    err, half_warp_gates_function = cuda.cuModuleGetFunction(
        module,
        _HALF_WARP_GATES_KERNEL_NAME,
    )
    _check_cuda_error(err)
    err, quarter_warp_gates_function = cuda.cuModuleGetFunction(
        module,
        _QUARTER_WARP_GATES_KERNEL_NAME,
    )
    _check_cuda_error(err)
    err, fused_half_warp_gates_function = cuda.cuModuleGetFunction(
        module,
        _FUSED_HALF_WARP_GATES_KERNEL_NAME,
    )
    _check_cuda_error(err)
    err, fused_pingpong_gates_function = cuda.cuModuleGetFunction(
        module,
        _FUSED_PINGPONG_GATES_KERNEL_NAME,
    )
    _check_cuda_error(err)
    err, fused_specialized_h128_gates_function = cuda.cuModuleGetFunction(
        module,
        _FUSED_SPECIALIZED_H128_GATES_KERNEL_NAME,
    )
    _check_cuda_error(err)
    err, fused_specialized_h130_gates_function = cuda.cuModuleGetFunction(
        module,
        _FUSED_SPECIALIZED_H130_GATES_KERNEL_NAME,
    )
    _check_cuda_error(err)
    err, fused_specialized_h160_gates_function = cuda.cuModuleGetFunction(
        module,
        _FUSED_SPECIALIZED_H160_GATES_KERNEL_NAME,
    )
    _check_cuda_error(err)
    err, cooperative_gates_function = cuda.cuModuleGetFunction(
        module,
        _COOPERATIVE_GATES_KERNEL_NAME,
    )
    _check_cuda_error(err)
    err, cooperative_h256_gates_function = cuda.cuModuleGetFunction(
        module,
        _COOPERATIVE_H256_GATES_KERNEL_NAME,
    )
    _check_cuda_error(err)
    err, cooperative_h256_parallel_update_gates_function = cuda.cuModuleGetFunction(
        module,
        _COOPERATIVE_H256_PARALLEL_UPDATE_GATES_KERNEL_NAME,
    )
    _check_cuda_error(err)
    err, cooperative_h256_shmem_gates_function = cuda.cuModuleGetFunction(
        module,
        _COOPERATIVE_H256_SHMEM_GATES_KERNEL_NAME,
    )
    _check_cuda_error(err)
    err, cooperative_h256_qwarp_shmem_gates_function = cuda.cuModuleGetFunction(
        module,
        _COOPERATIVE_H256_QWARP_SHMEM_GATES_KERNEL_NAME,
    )
    _check_cuda_error(err)
    err, cooperative_h256_cached_shmem_gates_function = cuda.cuModuleGetFunction(
        module,
        _COOPERATIVE_H256_CACHED_SHMEM_GATES_KERNEL_NAME,
    )
    _check_cuda_error(err)
    return _CompiledKernel(
        module=module,
        layer_function=layer_function,
        gates_function=gates_function,
        half_warp_gates_function=half_warp_gates_function,
        quarter_warp_gates_function=quarter_warp_gates_function,
        fused_half_warp_gates_function=fused_half_warp_gates_function,
        fused_pingpong_gates_function=fused_pingpong_gates_function,
        fused_specialized_h128_gates_function=fused_specialized_h128_gates_function,
        fused_specialized_h130_gates_function=fused_specialized_h130_gates_function,
        fused_specialized_h160_gates_function=fused_specialized_h160_gates_function,
        cooperative_gates_function=cooperative_gates_function,
        cooperative_h256_gates_function=cooperative_h256_gates_function,
        cooperative_h256_parallel_update_gates_function=(
            cooperative_h256_parallel_update_gates_function
        ),
        cooperative_h256_shmem_gates_function=cooperative_h256_shmem_gates_function,
        cooperative_h256_qwarp_shmem_gates_function=(
            cooperative_h256_qwarp_shmem_gates_function
        ),
        cooperative_h256_cached_shmem_gates_function=(
            cooperative_h256_cached_shmem_gates_function
        ),
    )


@lru_cache(maxsize=1)
def _get_a100_kernel() -> _CompiledKernel:
    return _compile_a100_kernel()


def _validate_tensors(
    x: torch.Tensor,
    h0: torch.Tensor,
    weight_ih: torch.Tensor,
    weight_hh: torch.Tensor,
    bias_ih: torch.Tensor,
    bias_hh: torch.Tensor,
) -> tuple[int, int, int]:
    if not x.is_cuda:
        raise RuntimeError("a100_gru_forward_layer requires CUDA tensors.")
    if x.dtype != torch.float32:
        raise RuntimeError("a100_gru_forward_layer currently supports fp32 only.")
    if x.dim() != 3:
        raise ValueError("x must have shape [batch, seq, input].")
    if h0.dim() != 2:
        raise ValueError("h0 must have shape [batch, hidden].")

    tensors = (x, h0, weight_ih, weight_hh, bias_ih, bias_hh)
    devices = {tensor.device for tensor in tensors}
    if len(devices) != 1:
        raise ValueError("All tensors must be on the same CUDA device.")
    if any(tensor.dtype != torch.float32 for tensor in tensors):
        raise RuntimeError("a100_gru_forward_layer currently supports fp32 only.")

    batch_size, seq_len, input_size = x.shape
    hidden_size = h0.size(1)
    if h0.size(0) != batch_size:
        raise ValueError("h0 batch dimension must match x.")
    if hidden_size > 256:
        raise ValueError("A100 prototype supports hidden_size <= 256.")
    if weight_ih.shape != (3 * hidden_size, input_size):
        raise ValueError("weight_ih must have shape [3 * hidden, input].")
    if weight_hh.shape != (3 * hidden_size, hidden_size):
        raise ValueError("weight_hh must have shape [3 * hidden, hidden].")
    if bias_ih.shape != (3 * hidden_size,):
        raise ValueError("bias_ih must have shape [3 * hidden].")
    if bias_hh.shape != (3 * hidden_size,):
        raise ValueError("bias_hh must have shape [3 * hidden].")
    return batch_size, seq_len, input_size


def _validate_block_threads(block_threads: int) -> None:
    if block_threads < 128 or block_threads > 1024 or block_threads % 32 != 0:
        raise ValueError("block_threads must be a multiple of 32 between 128 and 1024.")


def _validate_subwarp_size(subwarp_size: int) -> None:
    if subwarp_size not in {8, 16, 32}:
        raise ValueError("subwarp_size must be one of 8, 16, 32.")


def _validate_ctas_per_batch(ctas_per_batch: int) -> None:
    if ctas_per_batch < 1 or ctas_per_batch > 8:
        raise ValueError("ctas_per_batch must be between 1 and 8.")


def a100_gru_forward_layer(
    x: torch.Tensor,
    h0: torch.Tensor,
    weight_ih: torch.Tensor,
    weight_hh: torch.Tensor,
    bias_ih: torch.Tensor,
    bias_hh: torch.Tensor,
    block_threads: int = 1024,
) -> torch.Tensor:
    """运行 A100/SM80 专用单层 fp32 GRU forward-only 原型。"""
    _require_cuda_python()
    _validate_block_threads(block_threads)
    batch_size, seq_len, input_size = _validate_tensors(
        x,
        h0,
        weight_ih,
        weight_hh,
        bias_ih,
        bias_hh,
    )

    device_index = x.device.index
    if device_index is None:
        device_index = torch.cuda.current_device()
    torch.cuda.set_device(device_index)
    capability = _device_capability(device_index)
    if capability != (8, 0):
        raise RuntimeError(
            f"a100_gru_forward_layer requires SM80/A100, got sm_{capability[0]}{capability[1]}."
        )

    x = x.contiguous()
    h0 = h0.contiguous()
    weight_ih = weight_ih.contiguous()
    weight_hh = weight_hh.contiguous()
    bias_ih = bias_ih.contiguous()
    bias_hh = bias_hh.contiguous()

    hidden_size = h0.size(1)
    output = torch.empty(batch_size, seq_len, hidden_size, device=x.device, dtype=x.dtype)
    compiled = _get_a100_kernel()
    stream = cuda.CUstream(torch.cuda.current_stream(device=x.device).cuda_stream)

    shared_bytes = 4 * hidden_size * ctypes.sizeof(ctypes.c_float)
    values = (
        ctypes.c_void_p(x.data_ptr()),
        ctypes.c_void_p(h0.data_ptr()),
        ctypes.c_void_p(weight_ih.data_ptr()),
        ctypes.c_void_p(weight_hh.data_ptr()),
        ctypes.c_void_p(bias_ih.data_ptr()),
        ctypes.c_void_p(bias_hh.data_ptr()),
        ctypes.c_void_p(output.data_ptr()),
        ctypes.c_int(seq_len),
        ctypes.c_int(input_size),
        ctypes.c_int(hidden_size),
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
        ctypes.c_int,
    )
    err, = cuda.cuLaunchKernel(
        compiled.layer_function,
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
    return output


def _validate_input_gates(
    input_gates: torch.Tensor,
    h0: torch.Tensor,
    weight_hh: torch.Tensor,
    bias_hh: torch.Tensor,
) -> tuple[int, int, int]:
    if not input_gates.is_cuda:
        raise RuntimeError("a100_gru_forward_from_gates requires CUDA tensors.")
    if input_gates.dtype != torch.float32:
        raise RuntimeError("a100_gru_forward_from_gates currently supports fp32 only.")
    if input_gates.dim() != 3:
        raise ValueError("input_gates must have shape [batch, seq, 3 * hidden].")
    if h0.dim() != 2:
        raise ValueError("h0 must have shape [batch, hidden].")

    tensors = (input_gates, h0, weight_hh, bias_hh)
    devices = {tensor.device for tensor in tensors}
    if len(devices) != 1:
        raise ValueError("All tensors must be on the same CUDA device.")
    if any(tensor.dtype != torch.float32 for tensor in tensors):
        raise RuntimeError("a100_gru_forward_from_gates currently supports fp32 only.")

    batch_size, seq_len, gate_size = input_gates.shape
    hidden_size = h0.size(1)
    if gate_size != 3 * hidden_size:
        raise ValueError("input_gates last dimension must equal 3 * hidden.")
    if h0.size(0) != batch_size:
        raise ValueError("h0 batch dimension must match input_gates.")
    if hidden_size > 256:
        raise ValueError("A100 prototype supports hidden_size <= 256.")
    if weight_hh.shape != (3 * hidden_size, hidden_size):
        raise ValueError("weight_hh must have shape [3 * hidden, hidden].")
    if bias_hh.shape != (3 * hidden_size,):
        raise ValueError("bias_hh must have shape [3 * hidden].")
    return batch_size, seq_len, hidden_size


def a100_gru_forward_from_gates(
    input_gates: torch.Tensor,
    h0: torch.Tensor,
    weight_hh: torch.Tensor,
    bias_hh: torch.Tensor,
    block_threads: int = 1024,
) -> torch.Tensor:
    """运行已预计算 input projection 的 A100/SM80 GRU forward 原型。"""
    _require_cuda_python()
    _validate_block_threads(block_threads)
    return _a100_gru_forward_from_gates_impl(
        input_gates,
        h0,
        weight_hh,
        bias_hh,
        block_threads,
        subwarp_size=32,
        fuse_gates=False,
    )


def a100_gru_forward_from_gates_subwarp(
    input_gates: torch.Tensor,
    h0: torch.Tensor,
    weight_hh: torch.Tensor,
    bias_hh: torch.Tensor,
    block_threads: int = 1024,
    subwarp_size: int = 16,
) -> torch.Tensor:
    """运行 sub-warp recurrent projection 的 A100/SM80 GRU forward 原型。"""
    _require_cuda_python()
    _validate_block_threads(block_threads)
    _validate_subwarp_size(subwarp_size)
    return _a100_gru_forward_from_gates_impl(
        input_gates,
        h0,
        weight_hh,
        bias_hh,
        block_threads,
        subwarp_size=subwarp_size,
        fuse_gates=False,
    )


def a100_gru_forward_from_gates_fused(
    input_gates: torch.Tensor,
    h0: torch.Tensor,
    weight_hh: torch.Tensor,
    bias_hh: torch.Tensor,
    block_threads: int = 1024,
) -> torch.Tensor:
    """运行 fused r/z/n recurrent projection 的 A100/SM80 GRU forward 原型。"""
    _require_cuda_python()
    _validate_block_threads(block_threads)
    return _a100_gru_forward_from_gates_impl(
        input_gates,
        h0,
        weight_hh,
        bias_hh,
        block_threads,
        subwarp_size=16,
        fuse_gates=True,
        pingpong=False,
    )


def a100_gru_forward_from_gates_fused_pingpong(
    input_gates: torch.Tensor,
    h0: torch.Tensor,
    weight_hh: torch.Tensor,
    bias_hh: torch.Tensor,
    block_threads: int = 1024,
) -> torch.Tensor:
    """运行 ping-pong shared buffer 的 fused recurrent kernel。"""
    _require_cuda_python()
    _validate_block_threads(block_threads)
    return _a100_gru_forward_from_gates_impl(
        input_gates,
        h0,
        weight_hh,
        bias_hh,
        block_threads,
        subwarp_size=16,
        fuse_gates=True,
        pingpong=True,
    )


def a100_gru_forward_from_gates_fused_specialized(
    input_gates: torch.Tensor,
    h0: torch.Tensor,
    weight_hh: torch.Tensor,
    bias_hh: torch.Tensor,
    block_threads: int = 1024,
) -> torch.Tensor:
    """对 hidden_size=128/130/160 使用固定 H 的 fused recurrent kernel。"""
    _require_cuda_python()
    _validate_block_threads(block_threads)
    return _a100_gru_forward_from_gates_impl(
        input_gates,
        h0,
        weight_hh,
        bias_hh,
        block_threads,
        subwarp_size=16,
        fuse_gates=True,
        pingpong=False,
        specialize_hidden=True,
    )


def a100_gru_forward_from_gates_cooperative(
    input_gates: torch.Tensor,
    h0: torch.Tensor,
    weight_hh: torch.Tensor,
    bias_hh: torch.Tensor,
    block_threads: int = 1024,
    ctas_per_batch: int = 4,
) -> torch.Tensor:
    """运行 cooperative multi-CTA recurrent projection 原型。"""
    _require_cuda_python()
    _validate_block_threads(block_threads)
    _validate_ctas_per_batch(ctas_per_batch)
    batch_size, seq_len, hidden_size = _validate_input_gates(
        input_gates,
        h0,
        weight_hh,
        bias_hh,
    )

    device_index = input_gates.device.index
    if device_index is None:
        device_index = torch.cuda.current_device()
    torch.cuda.set_device(device_index)
    capability = _device_capability(device_index)
    if capability != (8, 0):
        raise RuntimeError(
            "a100_gru_forward_from_gates_cooperative requires SM80/A100, "
            f"got sm_{capability[0]}{capability[1]}."
        )
    cooperative_launch = _device_attribute(
        device_index,
        cuda.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_COOPERATIVE_LAUNCH,
    )
    if cooperative_launch == 0:
        raise RuntimeError("Current CUDA device does not support cooperative launch.")

    input_gates = input_gates.contiguous()
    h0 = h0.contiguous()
    weight_hh = weight_hh.contiguous()
    bias_hh = bias_hh.contiguous()

    output = torch.empty(
        batch_size,
        seq_len,
        hidden_size,
        device=input_gates.device,
        dtype=input_gates.dtype,
    )
    partial_gates = torch.empty(
        batch_size * ctas_per_batch,
        3 * hidden_size,
        device=input_gates.device,
        dtype=input_gates.dtype,
    )
    hidden_state = torch.empty(
        batch_size,
        hidden_size,
        device=input_gates.device,
        dtype=input_gates.dtype,
    )

    compiled = _get_a100_kernel()
    shared_bytes = 0
    err, active_blocks_per_sm = cuda.cuOccupancyMaxActiveBlocksPerMultiprocessor(
        compiled.cooperative_gates_function,
        block_threads,
        shared_bytes,
    )
    _check_cuda_error(err)
    sm_count = _device_attribute(
        device_index,
        cuda.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT,
    )
    grid_blocks = batch_size * ctas_per_batch
    max_cooperative_blocks = active_blocks_per_sm * sm_count
    if grid_blocks > max_cooperative_blocks:
        raise ValueError(
            "cooperative grid is too large for resident launch: "
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
        ctypes.c_int(hidden_size),
        ctypes.c_int(ctas_per_batch),
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
        ctypes.c_int,
    )
    err, = cuda.cuLaunchCooperativeKernel(
        compiled.cooperative_gates_function,
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


def a100_gru_forward_from_gates_cooperative_h256(
    input_gates: torch.Tensor,
    h0: torch.Tensor,
    weight_hh: torch.Tensor,
    bias_hh: torch.Tensor,
    block_threads: int = 704,
) -> torch.Tensor:
    """运行 hidden_size=256 专用 cooperative recurrent kernel。"""
    _require_cuda_python()
    _validate_block_threads(block_threads)
    batch_size, seq_len, hidden_size = _validate_input_gates(
        input_gates,
        h0,
        weight_hh,
        bias_hh,
    )
    if hidden_size != 256:
        raise ValueError("a100_gru_forward_from_gates_cooperative_h256 requires hidden_size=256.")

    device_index = input_gates.device.index
    if device_index is None:
        device_index = torch.cuda.current_device()
    torch.cuda.set_device(device_index)
    capability = _device_capability(device_index)
    if capability != (8, 0):
        raise RuntimeError(
            "a100_gru_forward_from_gates_cooperative_h256 requires SM80/A100, "
            f"got sm_{capability[0]}{capability[1]}."
        )
    cooperative_launch = _device_attribute(
        device_index,
        cuda.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_COOPERATIVE_LAUNCH,
    )
    if cooperative_launch == 0:
        raise RuntimeError("Current CUDA device does not support cooperative launch.")

    input_gates = input_gates.contiguous()
    h0 = h0.contiguous()
    weight_hh = weight_hh.contiguous()
    bias_hh = bias_hh.contiguous()

    ctas_per_batch = 4
    output = torch.empty(
        batch_size,
        seq_len,
        hidden_size,
        device=input_gates.device,
        dtype=input_gates.dtype,
    )
    partial_gates = torch.empty(
        batch_size * ctas_per_batch,
        3 * hidden_size,
        device=input_gates.device,
        dtype=input_gates.dtype,
    )
    hidden_state = torch.empty(
        batch_size,
        hidden_size,
        device=input_gates.device,
        dtype=input_gates.dtype,
    )

    compiled = _get_a100_kernel()
    shared_bytes = 0
    err, active_blocks_per_sm = cuda.cuOccupancyMaxActiveBlocksPerMultiprocessor(
        compiled.cooperative_h256_gates_function,
        block_threads,
        shared_bytes,
    )
    _check_cuda_error(err)
    sm_count = _device_attribute(
        device_index,
        cuda.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT,
    )
    grid_blocks = batch_size * ctas_per_batch
    max_cooperative_blocks = active_blocks_per_sm * sm_count
    if grid_blocks > max_cooperative_blocks:
        raise ValueError(
            "h256 cooperative grid is too large for resident launch: "
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
        compiled.cooperative_h256_gates_function,
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


def a100_gru_forward_from_gates_cooperative_h256_parallel_update(
    input_gates: torch.Tensor,
    h0: torch.Tensor,
    weight_hh: torch.Tensor,
    bias_hh: torch.Tensor,
    block_threads: int = 704,
) -> torch.Tensor:
    """运行 hidden update 由 4 个 CTA 分摊的 h256 cooperative kernel。"""
    _require_cuda_python()
    _validate_block_threads(block_threads)
    batch_size, seq_len, hidden_size = _validate_input_gates(
        input_gates,
        h0,
        weight_hh,
        bias_hh,
    )
    if hidden_size != 256:
        raise ValueError(
            "a100_gru_forward_from_gates_cooperative_h256_parallel_update "
            "requires hidden_size=256."
        )

    device_index = input_gates.device.index
    if device_index is None:
        device_index = torch.cuda.current_device()
    torch.cuda.set_device(device_index)
    capability = _device_capability(device_index)
    if capability != (8, 0):
        raise RuntimeError(
            "a100_gru_forward_from_gates_cooperative_h256_parallel_update "
            f"requires SM80/A100, got sm_{capability[0]}{capability[1]}."
        )
    cooperative_launch = _device_attribute(
        device_index,
        cuda.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_COOPERATIVE_LAUNCH,
    )
    if cooperative_launch == 0:
        raise RuntimeError("Current CUDA device does not support cooperative launch.")

    input_gates = input_gates.contiguous()
    h0 = h0.contiguous()
    weight_hh = weight_hh.contiguous()
    bias_hh = bias_hh.contiguous()

    ctas_per_batch = 4
    output = torch.empty(
        batch_size,
        seq_len,
        hidden_size,
        device=input_gates.device,
        dtype=input_gates.dtype,
    )
    partial_gates = torch.empty(
        batch_size * ctas_per_batch,
        3 * hidden_size,
        device=input_gates.device,
        dtype=input_gates.dtype,
    )
    hidden_state = torch.empty(
        batch_size,
        hidden_size,
        device=input_gates.device,
        dtype=input_gates.dtype,
    )

    compiled = _get_a100_kernel()
    shared_bytes = 0
    err, active_blocks_per_sm = cuda.cuOccupancyMaxActiveBlocksPerMultiprocessor(
        compiled.cooperative_h256_parallel_update_gates_function,
        block_threads,
        shared_bytes,
    )
    _check_cuda_error(err)
    sm_count = _device_attribute(
        device_index,
        cuda.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT,
    )
    grid_blocks = batch_size * ctas_per_batch
    max_cooperative_blocks = active_blocks_per_sm * sm_count
    if grid_blocks > max_cooperative_blocks:
        raise ValueError(
            "h256 parallel-update cooperative grid is too large for resident launch: "
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
        compiled.cooperative_h256_parallel_update_gates_function,
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


def a100_gru_forward_from_gates_cooperative_h256_shmem(
    input_gates: torch.Tensor,
    h0: torch.Tensor,
    weight_hh: torch.Tensor,
    bias_hh: torch.Tensor,
    block_threads: int = 704,
) -> torch.Tensor:
    """运行 CTA0 partial 使用 shared memory 的 hidden_size=256 cooperative kernel。"""
    _require_cuda_python()
    _validate_block_threads(block_threads)
    batch_size, seq_len, hidden_size = _validate_input_gates(
        input_gates,
        h0,
        weight_hh,
        bias_hh,
    )
    if hidden_size != 256:
        raise ValueError("a100_gru_forward_from_gates_cooperative_h256_shmem requires hidden_size=256.")

    device_index = input_gates.device.index
    if device_index is None:
        device_index = torch.cuda.current_device()
    torch.cuda.set_device(device_index)
    capability = _device_capability(device_index)
    if capability != (8, 0):
        raise RuntimeError(
            "a100_gru_forward_from_gates_cooperative_h256_shmem requires SM80/A100, "
            f"got sm_{capability[0]}{capability[1]}."
        )
    cooperative_launch = _device_attribute(
        device_index,
        cuda.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_COOPERATIVE_LAUNCH,
    )
    if cooperative_launch == 0:
        raise RuntimeError("Current CUDA device does not support cooperative launch.")

    input_gates = input_gates.contiguous()
    h0 = h0.contiguous()
    weight_hh = weight_hh.contiguous()
    bias_hh = bias_hh.contiguous()

    ctas_per_batch = 4
    output = torch.empty(
        batch_size,
        seq_len,
        hidden_size,
        device=input_gates.device,
        dtype=input_gates.dtype,
    )
    partial_gates = torch.empty(
        batch_size * ctas_per_batch,
        3 * hidden_size,
        device=input_gates.device,
        dtype=input_gates.dtype,
    )
    hidden_state = torch.empty(
        batch_size,
        hidden_size,
        device=input_gates.device,
        dtype=input_gates.dtype,
    )

    compiled = _get_a100_kernel()
    shared_bytes = 3 * hidden_size * ctypes.sizeof(ctypes.c_float)
    err, active_blocks_per_sm = cuda.cuOccupancyMaxActiveBlocksPerMultiprocessor(
        compiled.cooperative_h256_shmem_gates_function,
        block_threads,
        shared_bytes,
    )
    _check_cuda_error(err)
    sm_count = _device_attribute(
        device_index,
        cuda.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT,
    )
    grid_blocks = batch_size * ctas_per_batch
    max_cooperative_blocks = active_blocks_per_sm * sm_count
    if grid_blocks > max_cooperative_blocks:
        raise ValueError(
            "h256 shmem cooperative grid is too large for resident launch: "
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
        compiled.cooperative_h256_shmem_gates_function,
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


def a100_gru_forward_from_gates_cooperative_h256_qwarp_shmem(
    input_gates: torch.Tensor,
    h0: torch.Tensor,
    weight_hh: torch.Tensor,
    bias_hh: torch.Tensor,
    block_threads: int = 704,
) -> torch.Tensor:
    """运行 quarter-warp dot-product 的 h256 shmem cooperative kernel。"""
    _require_cuda_python()
    _validate_block_threads(block_threads)
    batch_size, seq_len, hidden_size = _validate_input_gates(
        input_gates,
        h0,
        weight_hh,
        bias_hh,
    )
    if hidden_size != 256:
        raise ValueError(
            "a100_gru_forward_from_gates_cooperative_h256_qwarp_shmem "
            "requires hidden_size=256."
        )

    device_index = input_gates.device.index
    if device_index is None:
        device_index = torch.cuda.current_device()
    torch.cuda.set_device(device_index)
    capability = _device_capability(device_index)
    if capability != (8, 0):
        raise RuntimeError(
            "a100_gru_forward_from_gates_cooperative_h256_qwarp_shmem "
            f"requires SM80/A100, got sm_{capability[0]}{capability[1]}."
        )
    cooperative_launch = _device_attribute(
        device_index,
        cuda.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_COOPERATIVE_LAUNCH,
    )
    if cooperative_launch == 0:
        raise RuntimeError("Current CUDA device does not support cooperative launch.")

    input_gates = input_gates.contiguous()
    h0 = h0.contiguous()
    weight_hh = weight_hh.contiguous()
    bias_hh = bias_hh.contiguous()

    ctas_per_batch = 4
    output = torch.empty(
        batch_size,
        seq_len,
        hidden_size,
        device=input_gates.device,
        dtype=input_gates.dtype,
    )
    partial_gates = torch.empty(
        batch_size * ctas_per_batch,
        3 * hidden_size,
        device=input_gates.device,
        dtype=input_gates.dtype,
    )
    hidden_state = torch.empty(
        batch_size,
        hidden_size,
        device=input_gates.device,
        dtype=input_gates.dtype,
    )

    compiled = _get_a100_kernel()
    shared_bytes = 3 * hidden_size * ctypes.sizeof(ctypes.c_float)
    err, active_blocks_per_sm = cuda.cuOccupancyMaxActiveBlocksPerMultiprocessor(
        compiled.cooperative_h256_qwarp_shmem_gates_function,
        block_threads,
        shared_bytes,
    )
    _check_cuda_error(err)
    sm_count = _device_attribute(
        device_index,
        cuda.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT,
    )
    grid_blocks = batch_size * ctas_per_batch
    max_cooperative_blocks = active_blocks_per_sm * sm_count
    if grid_blocks > max_cooperative_blocks:
        raise ValueError(
            "h256 qwarp shmem cooperative grid is too large for resident launch: "
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
        compiled.cooperative_h256_qwarp_shmem_gates_function,
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


def a100_gru_forward_from_gates_cooperative_h256_cached_shmem(
    input_gates: torch.Tensor,
    h0: torch.Tensor,
    weight_hh: torch.Tensor,
    bias_hh: torch.Tensor,
    block_threads: int = 704,
) -> torch.Tensor:
    """运行 hidden k-tile 使用 shared memory 缓存的 h256 cooperative kernel。"""
    _require_cuda_python()
    _validate_block_threads(block_threads)
    batch_size, seq_len, hidden_size = _validate_input_gates(
        input_gates,
        h0,
        weight_hh,
        bias_hh,
    )
    if hidden_size != 256:
        raise ValueError(
            "a100_gru_forward_from_gates_cooperative_h256_cached_shmem "
            "requires hidden_size=256."
        )

    device_index = input_gates.device.index
    if device_index is None:
        device_index = torch.cuda.current_device()
    torch.cuda.set_device(device_index)
    capability = _device_capability(device_index)
    if capability != (8, 0):
        raise RuntimeError(
            "a100_gru_forward_from_gates_cooperative_h256_cached_shmem "
            f"requires SM80/A100, got sm_{capability[0]}{capability[1]}."
        )
    cooperative_launch = _device_attribute(
        device_index,
        cuda.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_COOPERATIVE_LAUNCH,
    )
    if cooperative_launch == 0:
        raise RuntimeError("Current CUDA device does not support cooperative launch.")

    input_gates = input_gates.contiguous()
    h0 = h0.contiguous()
    weight_hh = weight_hh.contiguous()
    bias_hh = bias_hh.contiguous()

    ctas_per_batch = 4
    output = torch.empty(
        batch_size,
        seq_len,
        hidden_size,
        device=input_gates.device,
        dtype=input_gates.dtype,
    )
    partial_gates = torch.empty(
        batch_size * ctas_per_batch,
        3 * hidden_size,
        device=input_gates.device,
        dtype=input_gates.dtype,
    )
    hidden_state = torch.empty(
        batch_size,
        hidden_size,
        device=input_gates.device,
        dtype=input_gates.dtype,
    )

    compiled = _get_a100_kernel()
    shared_bytes = (3 * hidden_size + 64) * ctypes.sizeof(ctypes.c_float)
    err, active_blocks_per_sm = cuda.cuOccupancyMaxActiveBlocksPerMultiprocessor(
        compiled.cooperative_h256_cached_shmem_gates_function,
        block_threads,
        shared_bytes,
    )
    _check_cuda_error(err)
    sm_count = _device_attribute(
        device_index,
        cuda.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT,
    )
    grid_blocks = batch_size * ctas_per_batch
    max_cooperative_blocks = active_blocks_per_sm * sm_count
    if grid_blocks > max_cooperative_blocks:
        raise ValueError(
            "h256 cached shmem cooperative grid is too large for resident launch: "
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
        compiled.cooperative_h256_cached_shmem_gates_function,
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


def _a100_gru_forward_from_gates_impl(
    input_gates: torch.Tensor,
    h0: torch.Tensor,
    weight_hh: torch.Tensor,
    bias_hh: torch.Tensor,
    block_threads: int,
    subwarp_size: int,
    fuse_gates: bool,
    pingpong: bool = False,
    specialize_hidden: bool = False,
) -> torch.Tensor:
    batch_size, seq_len, hidden_size = _validate_input_gates(
        input_gates,
        h0,
        weight_hh,
        bias_hh,
    )

    device_index = input_gates.device.index
    if device_index is None:
        device_index = torch.cuda.current_device()
    torch.cuda.set_device(device_index)
    capability = _device_capability(device_index)
    if capability != (8, 0):
        raise RuntimeError(
            f"a100_gru_forward_from_gates requires SM80/A100, got sm_{capability[0]}{capability[1]}."
        )

    input_gates = input_gates.contiguous()
    h0 = h0.contiguous()
    weight_hh = weight_hh.contiguous()
    bias_hh = bias_hh.contiguous()

    output = torch.empty(
        batch_size,
        seq_len,
        hidden_size,
        device=input_gates.device,
        dtype=input_gates.dtype,
    )
    compiled = _get_a100_kernel()
    if specialize_hidden and hidden_size == 128:
        kernel_function = compiled.fused_specialized_h128_gates_function
    elif specialize_hidden and hidden_size == 130:
        kernel_function = compiled.fused_specialized_h130_gates_function
    elif specialize_hidden and hidden_size == 160:
        kernel_function = compiled.fused_specialized_h160_gates_function
    elif pingpong:
        kernel_function = compiled.fused_pingpong_gates_function
    elif fuse_gates:
        kernel_function = compiled.fused_half_warp_gates_function
    elif subwarp_size == 8:
        kernel_function = compiled.quarter_warp_gates_function
    elif subwarp_size == 16:
        kernel_function = compiled.half_warp_gates_function
    else:
        kernel_function = compiled.gates_function
    stream = cuda.CUstream(torch.cuda.current_stream(device=input_gates.device).cuda_stream)

    shared_multiplier = 2 if fuse_gates else 4
    shared_bytes = shared_multiplier * hidden_size * ctypes.sizeof(ctypes.c_float)
    values = (
        ctypes.c_void_p(input_gates.data_ptr()),
        ctypes.c_void_p(h0.data_ptr()),
        ctypes.c_void_p(weight_hh.data_ptr()),
        ctypes.c_void_p(bias_hh.data_ptr()),
        ctypes.c_void_p(output.data_ptr()),
        ctypes.c_int(seq_len),
        ctypes.c_int(hidden_size),
    )
    types = (
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_int,
    )
    err, = cuda.cuLaunchKernel(
        kernel_function,
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
    return output


def a100_gru_forward_layer_precompute_input(
    x: torch.Tensor,
    h0: torch.Tensor,
    weight_ih: torch.Tensor,
    weight_hh: torch.Tensor,
    bias_ih: torch.Tensor,
    bias_hh: torch.Tensor,
    block_threads: int = 1024,
) -> torch.Tensor:
    """用 cuBLAS 预计算 input gates，再运行 A100/SM80 recurrent kernel。"""
    batch_size, seq_len, input_size = _validate_tensors(
        x,
        h0,
        weight_ih,
        weight_hh,
        bias_ih,
        bias_hh,
    )
    hidden_size = h0.size(1)
    x_2d = x.contiguous().reshape(batch_size * seq_len, input_size)
    input_gates = F.linear(x_2d, weight_ih, bias_ih).view(
        batch_size,
        seq_len,
        3 * hidden_size,
    )
    return a100_gru_forward_from_gates(
        input_gates,
        h0,
        weight_hh,
        bias_hh,
        block_threads=block_threads,
    )


def a100_gru_forward_layer_precompute_input_subwarp(
    x: torch.Tensor,
    h0: torch.Tensor,
    weight_ih: torch.Tensor,
    weight_hh: torch.Tensor,
    bias_ih: torch.Tensor,
    bias_hh: torch.Tensor,
    block_threads: int = 1024,
    subwarp_size: int = 16,
) -> torch.Tensor:
    """用 cuBLAS 预计算 input gates，再运行 sub-warp recurrent kernel。"""
    batch_size, seq_len, input_size = _validate_tensors(
        x,
        h0,
        weight_ih,
        weight_hh,
        bias_ih,
        bias_hh,
    )
    hidden_size = h0.size(1)
    x_2d = x.contiguous().reshape(batch_size * seq_len, input_size)
    input_gates = F.linear(x_2d, weight_ih, bias_ih).view(
        batch_size,
        seq_len,
        3 * hidden_size,
    )
    return a100_gru_forward_from_gates_subwarp(
        input_gates,
        h0,
        weight_hh,
        bias_hh,
        block_threads=block_threads,
        subwarp_size=subwarp_size,
    )


def a100_gru_forward_layer_precompute_input_fused(
    x: torch.Tensor,
    h0: torch.Tensor,
    weight_ih: torch.Tensor,
    weight_hh: torch.Tensor,
    bias_ih: torch.Tensor,
    bias_hh: torch.Tensor,
    block_threads: int = 1024,
) -> torch.Tensor:
    """用 cuBLAS 预计算 input gates，再运行 fused recurrent kernel。"""
    batch_size, seq_len, input_size = _validate_tensors(
        x,
        h0,
        weight_ih,
        weight_hh,
        bias_ih,
        bias_hh,
    )
    hidden_size = h0.size(1)
    x_2d = x.contiguous().reshape(batch_size * seq_len, input_size)
    input_gates = F.linear(x_2d, weight_ih, bias_ih).view(
        batch_size,
        seq_len,
        3 * hidden_size,
    )
    return a100_gru_forward_from_gates_fused(
        input_gates,
        h0,
        weight_hh,
        bias_hh,
        block_threads=block_threads,
    )


def a100_gru_forward_layer_precompute_input_fused_pingpong(
    x: torch.Tensor,
    h0: torch.Tensor,
    weight_ih: torch.Tensor,
    weight_hh: torch.Tensor,
    bias_ih: torch.Tensor,
    bias_hh: torch.Tensor,
    block_threads: int = 1024,
) -> torch.Tensor:
    """用 cuBLAS 预计算 input gates，再运行 ping-pong fused recurrent kernel。"""
    batch_size, seq_len, input_size = _validate_tensors(
        x,
        h0,
        weight_ih,
        weight_hh,
        bias_ih,
        bias_hh,
    )
    hidden_size = h0.size(1)
    x_2d = x.contiguous().reshape(batch_size * seq_len, input_size)
    input_gates = F.linear(x_2d, weight_ih, bias_ih).view(
        batch_size,
        seq_len,
        3 * hidden_size,
    )
    return a100_gru_forward_from_gates_fused_pingpong(
        input_gates,
        h0,
        weight_hh,
        bias_hh,
        block_threads=block_threads,
    )


def a100_gru_forward_layer_precompute_input_fused_specialized(
    x: torch.Tensor,
    h0: torch.Tensor,
    weight_ih: torch.Tensor,
    weight_hh: torch.Tensor,
    bias_ih: torch.Tensor,
    bias_hh: torch.Tensor,
    block_threads: int = 1024,
) -> torch.Tensor:
    """用 cuBLAS 预计算 input gates，再运行固定 hidden size 的 fused kernel。"""
    batch_size, seq_len, input_size = _validate_tensors(
        x,
        h0,
        weight_ih,
        weight_hh,
        bias_ih,
        bias_hh,
    )
    hidden_size = h0.size(1)
    x_2d = x.contiguous().reshape(batch_size * seq_len, input_size)
    input_gates = F.linear(x_2d, weight_ih, bias_ih).view(
        batch_size,
        seq_len,
        3 * hidden_size,
    )
    return a100_gru_forward_from_gates_fused_specialized(
        input_gates,
        h0,
        weight_hh,
        bias_hh,
        block_threads=block_threads,
    )


def a100_gru_forward_layer_precompute_input_cooperative(
    x: torch.Tensor,
    h0: torch.Tensor,
    weight_ih: torch.Tensor,
    weight_hh: torch.Tensor,
    bias_ih: torch.Tensor,
    bias_hh: torch.Tensor,
    block_threads: int = 1024,
    ctas_per_batch: int = 4,
) -> torch.Tensor:
    """用 cuBLAS 预计算 input gates，再运行 cooperative multi-CTA recurrent kernel。"""
    batch_size, seq_len, input_size = _validate_tensors(
        x,
        h0,
        weight_ih,
        weight_hh,
        bias_ih,
        bias_hh,
    )
    hidden_size = h0.size(1)
    x_2d = x.contiguous().reshape(batch_size * seq_len, input_size)
    input_gates = F.linear(x_2d, weight_ih, bias_ih).view(
        batch_size,
        seq_len,
        3 * hidden_size,
    )
    return a100_gru_forward_from_gates_cooperative(
        input_gates,
        h0,
        weight_hh,
        bias_hh,
        block_threads=block_threads,
        ctas_per_batch=ctas_per_batch,
    )


def a100_gru_forward_layer_precompute_input_cooperative_h256(
    x: torch.Tensor,
    h0: torch.Tensor,
    weight_ih: torch.Tensor,
    weight_hh: torch.Tensor,
    bias_ih: torch.Tensor,
    bias_hh: torch.Tensor,
    block_threads: int = 704,
) -> torch.Tensor:
    """用 cuBLAS 预计算 input gates，再运行 h256 专用 cooperative kernel。"""
    batch_size, seq_len, input_size = _validate_tensors(
        x,
        h0,
        weight_ih,
        weight_hh,
        bias_ih,
        bias_hh,
    )
    hidden_size = h0.size(1)
    if hidden_size != 256:
        raise ValueError("a100_gru_forward_layer_precompute_input_cooperative_h256 requires hidden_size=256.")
    x_2d = x.contiguous().reshape(batch_size * seq_len, input_size)
    input_gates = F.linear(x_2d, weight_ih, bias_ih).view(
        batch_size,
        seq_len,
        3 * hidden_size,
    )
    return a100_gru_forward_from_gates_cooperative_h256(
        input_gates,
        h0,
        weight_hh,
        bias_hh,
        block_threads=block_threads,
    )


def a100_gru_forward_layer_precompute_input_cooperative_h256_parallel_update(
    x: torch.Tensor,
    h0: torch.Tensor,
    weight_ih: torch.Tensor,
    weight_hh: torch.Tensor,
    bias_ih: torch.Tensor,
    bias_hh: torch.Tensor,
    block_threads: int = 704,
) -> torch.Tensor:
    """用 cuBLAS 预计算 input gates，再运行 h256 parallel-update kernel。"""
    batch_size, seq_len, input_size = _validate_tensors(
        x,
        h0,
        weight_ih,
        weight_hh,
        bias_ih,
        bias_hh,
    )
    hidden_size = h0.size(1)
    if hidden_size != 256:
        raise ValueError(
            "a100_gru_forward_layer_precompute_input_cooperative_h256_parallel_update "
            "requires hidden_size=256."
        )
    x_2d = x.contiguous().reshape(batch_size * seq_len, input_size)
    input_gates = F.linear(x_2d, weight_ih, bias_ih).view(
        batch_size,
        seq_len,
        3 * hidden_size,
    )
    return a100_gru_forward_from_gates_cooperative_h256_parallel_update(
        input_gates,
        h0,
        weight_hh,
        bias_hh,
        block_threads=block_threads,
    )


def a100_gru_forward_layer_precompute_input_cooperative_h256_shmem(
    x: torch.Tensor,
    h0: torch.Tensor,
    weight_ih: torch.Tensor,
    weight_hh: torch.Tensor,
    bias_ih: torch.Tensor,
    bias_hh: torch.Tensor,
    block_threads: int = 704,
) -> torch.Tensor:
    """用 cuBLAS 预计算 input gates，再运行 h256 shmem cooperative kernel。"""
    batch_size, seq_len, input_size = _validate_tensors(
        x,
        h0,
        weight_ih,
        weight_hh,
        bias_ih,
        bias_hh,
    )
    hidden_size = h0.size(1)
    if hidden_size != 256:
        raise ValueError(
            "a100_gru_forward_layer_precompute_input_cooperative_h256_shmem "
            "requires hidden_size=256."
        )
    x_2d = x.contiguous().reshape(batch_size * seq_len, input_size)
    input_gates = F.linear(x_2d, weight_ih, bias_ih).view(
        batch_size,
        seq_len,
        3 * hidden_size,
    )
    return a100_gru_forward_from_gates_cooperative_h256_shmem(
        input_gates,
        h0,
        weight_hh,
        bias_hh,
        block_threads=block_threads,
    )


def a100_gru_forward_layer_precompute_input_cooperative_h256_qwarp_shmem(
    x: torch.Tensor,
    h0: torch.Tensor,
    weight_ih: torch.Tensor,
    weight_hh: torch.Tensor,
    bias_ih: torch.Tensor,
    bias_hh: torch.Tensor,
    block_threads: int = 704,
) -> torch.Tensor:
    """用 cuBLAS 预计算 input gates，再运行 h256 qwarp shmem kernel。"""
    batch_size, seq_len, input_size = _validate_tensors(
        x,
        h0,
        weight_ih,
        weight_hh,
        bias_ih,
        bias_hh,
    )
    hidden_size = h0.size(1)
    if hidden_size != 256:
        raise ValueError(
            "a100_gru_forward_layer_precompute_input_cooperative_h256_qwarp_shmem "
            "requires hidden_size=256."
        )
    x_2d = x.contiguous().reshape(batch_size * seq_len, input_size)
    input_gates = F.linear(x_2d, weight_ih, bias_ih).view(
        batch_size,
        seq_len,
        3 * hidden_size,
    )
    return a100_gru_forward_from_gates_cooperative_h256_qwarp_shmem(
        input_gates,
        h0,
        weight_hh,
        bias_hh,
        block_threads=block_threads,
    )


def a100_gru_forward_layer_precompute_input_cooperative_h256_cached_shmem(
    x: torch.Tensor,
    h0: torch.Tensor,
    weight_ih: torch.Tensor,
    weight_hh: torch.Tensor,
    bias_ih: torch.Tensor,
    bias_hh: torch.Tensor,
    block_threads: int = 704,
) -> torch.Tensor:
    """用 cuBLAS 预计算 input gates，再运行 h256 cached-shmem kernel。"""
    batch_size, seq_len, input_size = _validate_tensors(
        x,
        h0,
        weight_ih,
        weight_hh,
        bias_ih,
        bias_hh,
    )
    hidden_size = h0.size(1)
    if hidden_size != 256:
        raise ValueError(
            "a100_gru_forward_layer_precompute_input_cooperative_h256_cached_shmem "
            "requires hidden_size=256."
        )
    x_2d = x.contiguous().reshape(batch_size * seq_len, input_size)
    input_gates = F.linear(x_2d, weight_ih, bias_ih).view(
        batch_size,
        seq_len,
        3 * hidden_size,
    )
    return a100_gru_forward_from_gates_cooperative_h256_cached_shmem(
        input_gates,
        h0,
        weight_hh,
        bias_hh,
        block_threads=block_threads,
    )
