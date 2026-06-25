from __future__ import annotations

import ctypes
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import torch

try:
    import nvidia.cuda_nvcc
    import nvidia.cuda_runtime
    from cuda import cuda, cudart, nvrtc
except ImportError:  # pragma: no cover - 由运行时错误给出更清晰的提示。
    cuda = None
    cudart = None
    nvrtc = None
    nvidia = None


_CUDA_GRU_FORWARD_SOURCE = r"""
#include <math_functions.h>

extern "C" __global__
void gru_forward_layer(
    const float* __restrict__ x,
    const float* __restrict__ h0,
    const float* __restrict__ weight_ih,
    const float* __restrict__ weight_hh,
    const float* __restrict__ bias_ih,
    const float* __restrict__ bias_hh,
    float* __restrict__ output,
    int seq_len,
    int input_size,
    int hidden_size)
{
    extern __shared__ float hidden[];
    const int batch_idx = blockIdx.x;
    const int hid = threadIdx.x;

    if (hid < hidden_size) {
        hidden[hid] = h0[batch_idx * hidden_size + hid];
    }
    __syncthreads();

    for (int step = 0; step < seq_len; ++step) {
        float hidden_next = 0.0f;
        if (hid < hidden_size) {
            float i_r = bias_ih[hid];
            float i_z = bias_ih[hidden_size + hid];
            float i_n = bias_ih[2 * hidden_size + hid];

            const float* x_step = x + (batch_idx * seq_len + step) * input_size;
            for (int k = 0; k < input_size; ++k) {
                const float x_value = x_step[k];
                i_r += x_value * weight_ih[hid * input_size + k];
                i_z += x_value * weight_ih[(hidden_size + hid) * input_size + k];
                i_n += x_value * weight_ih[(2 * hidden_size + hid) * input_size + k];
            }

            float h_r = bias_hh[hid];
            float h_z = bias_hh[hidden_size + hid];
            float h_n = bias_hh[2 * hidden_size + hid];
            for (int k = 0; k < hidden_size; ++k) {
                const float hidden_value = hidden[k];
                h_r += hidden_value * weight_hh[hid * hidden_size + k];
                h_z += hidden_value * weight_hh[(hidden_size + hid) * hidden_size + k];
                h_n += hidden_value * weight_hh[(2 * hidden_size + hid) * hidden_size + k];
            }

            const float hidden_prev = hidden[hid];
            const float reset_gate = 1.0f / (1.0f + expf(-(i_r + h_r)));
            const float update_gate = 1.0f / (1.0f + expf(-(i_z + h_z)));
            const float new_gate = tanhf(i_n + reset_gate * h_n);
            hidden_next = new_gate + update_gate * (hidden_prev - new_gate);
        }

        __syncthreads();
        if (hid < hidden_size) {
            hidden[hid] = hidden_next;
            output[(batch_idx * seq_len + step) * hidden_size + hid] = hidden_next;
        }
        __syncthreads();
    }
}
"""


@dataclass(frozen=True)
class _CompiledKernel:
    module: object
    function: object


def _require_cuda_python() -> None:
    if cuda is None or nvrtc is None:
        raise RuntimeError(
            "cuda_gru_forward_layer requires cuda-python and nvidia-cuda-nvcc-cu12 "
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


def _get_device_arch(device_index: int) -> tuple[int, int]:
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


def _compile_cuda_source(source: str, kernel_name: bytes, device_index: int) -> _CompiledKernel:
    major, minor = _get_device_arch(device_index)
    err, _, nvrtc_minor = nvrtc.nvrtcVersion()
    _check_cuda_error(err)

    # NVRTC 12.x 可以直接产出 cubin；旧版本保留 PTX fallback。
    use_cubin = nvrtc_minor >= 1
    arch_prefix = "sm" if use_cubin else "compute"
    options = [
        b"--std=c++17",
        f"--gpu-architecture={arch_prefix}_{major}{minor}".encode("ascii"),
    ]
    options.extend(
        f"--include-path={include_dir}".encode("utf-8")
        for include_dir in _cuda_include_dirs()
    )

    err, program = nvrtc.nvrtcCreateProgram(
        source.encode("utf-8"),
        b"cuda_gru_forward.cu",
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

    if use_cubin:
        err, image_size = nvrtc.nvrtcGetCUBINSize(program)
        _check_cuda_error(err)
        image = b" " * image_size
        err, = nvrtc.nvrtcGetCUBIN(program, image)
        _check_cuda_error(err)
    else:
        err, image_size = nvrtc.nvrtcGetPTXSize(program)
        _check_cuda_error(err)
        image = b" " * image_size
        err, = nvrtc.nvrtcGetPTX(program, image)
        _check_cuda_error(err)

    err, = nvrtc.nvrtcDestroyProgram(program)
    _check_cuda_error(err)

    err, module = cuda.cuModuleLoadData(image)
    _check_cuda_error(err)
    err, function = cuda.cuModuleGetFunction(module, kernel_name)
    _check_cuda_error(err)
    return _CompiledKernel(module=module, function=function)


@lru_cache(maxsize=16)
def _get_forward_kernel(device_index: int, major: int, minor: int) -> _CompiledKernel:
    del major, minor
    return _compile_cuda_source(
        _CUDA_GRU_FORWARD_SOURCE,
        b"gru_forward_layer",
        device_index,
    )


def _next_power_of_2(value: int) -> int:
    return 1 << (value - 1).bit_length()


def cuda_gru_forward_layer(
    x: torch.Tensor,
    h0: torch.Tensor,
    weight_ih: torch.Tensor,
    weight_hh: torch.Tensor,
    bias_ih: torch.Tensor,
    bias_hh: torch.Tensor,
) -> torch.Tensor:
    """运行单层 fp32 GRU forward-only CUDA C/NVRTC 原型。"""
    _require_cuda_python()
    if not x.is_cuda:
        raise RuntimeError("cuda_gru_forward_layer requires CUDA tensors.")
    if x.dtype != torch.float32:
        raise RuntimeError("cuda_gru_forward_layer currently supports fp32 only.")
    if x.dim() != 3:
        raise ValueError("x must have shape [batch, seq, input].")
    if h0.dim() != 2:
        raise ValueError("h0 must have shape [batch, hidden].")

    tensors = (x, h0, weight_ih, weight_hh, bias_ih, bias_hh)
    devices = {tensor.device for tensor in tensors}
    if len(devices) != 1:
        raise ValueError("All tensors must be on the same CUDA device.")
    if any(tensor.dtype != torch.float32 for tensor in tensors):
        raise RuntimeError("cuda_gru_forward_layer currently supports fp32 only.")

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
    if hidden_size > 256:
        raise ValueError("Prototype supports hidden_size <= 256.")
    if weight_ih.shape != (3 * hidden_size, input_size):
        raise ValueError("weight_ih must have shape [3 * hidden, input].")
    if weight_hh.shape != (3 * hidden_size, hidden_size):
        raise ValueError("weight_hh must have shape [3 * hidden, hidden].")
    if bias_ih.shape != (3 * hidden_size,):
        raise ValueError("bias_ih must have shape [3 * hidden].")
    if bias_hh.shape != (3 * hidden_size,):
        raise ValueError("bias_hh must have shape [3 * hidden].")

    device_index = x.device.index
    if device_index is None:
        device_index = torch.cuda.current_device()
    torch.cuda.set_device(device_index)
    major, minor = _get_device_arch(device_index)
    compiled = _get_forward_kernel(device_index, major, minor)

    output = torch.empty(batch_size, seq_len, hidden_size, device=x.device, dtype=x.dtype)
    stream = cuda.CUstream(torch.cuda.current_stream(device=x.device).cuda_stream)
    block_h = _next_power_of_2(hidden_size)
    shared_bytes = hidden_size * ctypes.sizeof(ctypes.c_float)

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
        compiled.function,
        batch_size,
        1,
        1,
        block_h,
        1,
        1,
        shared_bytes,
        stream,
        (values, types),
        0,
    )
    _check_cuda_error(err)
    return output
