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
_COOPERATIVE_H256_SHMEM_GATE_CACHE_GATES_KERNEL_NAME = (
    b"a100_gru_forward_from_gates_cooperative_h256_shmem_gate_cache_kernel"
)
_COOPERATIVE_H256_SHMEM_GRAD_COEFF_CACHE_GATES_KERNEL_NAME = (
    b"a100_gru_forward_from_gates_cooperative_h256_shmem_grad_coeff_cache_kernel"
)
_COOPERATIVE_H256_PARALLEL_UPDATE_GATE_CACHE_GATES_KERNEL_NAME = (
    b"a100_gru_forward_from_gates_cooperative_h256_parallel_update_gate_cache_kernel"
)
_COOPERATIVE_H256_CTA8_SHMEM_GATE_CACHE_GATES_KERNEL_NAME = (
    b"a100_gru_forward_from_gates_cooperative_h256_cta8_shmem_gate_cache_kernel"
)
_COOPERATIVE_H256_CTA6_SHMEM_GATE_CACHE_GATES_KERNEL_NAME = (
    b"a100_gru_forward_from_gates_cooperative_h256_cta6_shmem_gate_cache_kernel"
)
_COOPERATIVE_H256_HTILE2_SHMEM_GATE_CACHE_GATES_KERNEL_NAME = (
    b"a100_gru_forward_from_gates_cooperative_h256_htile2_shmem_gate_cache_kernel"
)
_COOPERATIVE_H256_HTILE4_SHMEM_GATE_CACHE_GATES_KERNEL_NAME = (
    b"a100_gru_forward_from_gates_cooperative_h256_htile4_shmem_gate_cache_kernel"
)
_COOPERATIVE_H256_HTILE4_COMPACT_SHMEM_GATE_CACHE_GATES_KERNEL_NAME = (
    b"a100_gru_forward_from_gates_cooperative_h256_htile4_compact_shmem_gate_cache_kernel"
)
_COOPERATIVE_H256_HTILE4_COMPACT_HOIST_SHMEM_GATE_CACHE_GATES_KERNEL_NAME = (
    b"a100_gru_forward_from_gates_cooperative_h256_htile4_compact_hoist_shmem_gate_cache_kernel"
)
_COOPERATIVE_H256_HTILE4_COMPACT_HOIST_ROW4_SHMEM_GATE_CACHE_GATES_KERNEL_NAME = (
    b"a100_gru_forward_from_gates_cooperative_h256_htile4_compact_hoist_row4_shmem_gate_cache_kernel"
)
_COOPERATIVE_H256_HTILE4_COMPACT_HOIST_ROW4_PREV_CACHE_SHMEM_GATE_CACHE_GATES_KERNEL_NAME = (
    b"a100_gru_forward_from_gates_cooperative_h256_htile4_compact_hoist_row4_prev_cache_shmem_gate_cache_kernel"
)
_COOPERATIVE_H256_HTILE4_COMPACT_HOIST_ROW4_PARALLEL_UPDATE_GATE_CACHE_GATES_KERNEL_NAME = (
    b"a100_gru_forward_from_gates_cooperative_h256_htile4_compact_hoist_row4_parallel_update_gate_cache_kernel"
)
_COOPERATIVE_H256_HTILE4_COMPACT_HOIST_ROW4_LDG_SHMEM_GATE_CACHE_GATES_KERNEL_NAME = (
    b"a100_gru_forward_from_gates_cooperative_h256_htile4_compact_hoist_row4_ldg_shmem_gate_cache_kernel"
)
_COOPERATIVE_H256_HTILE4_COMPACT_HOIST_ROW3_SHMEM_GATE_CACHE_GATES_KERNEL_NAME = (
    b"a100_gru_forward_from_gates_cooperative_h256_htile4_compact_hoist_row3_shmem_gate_cache_kernel"
)
_COOPERATIVE_H256_HTILE4_COMPACT_HOIST_QWARP_GATE_CACHE_GATES_KERNEL_NAME = (
    b"a100_gru_forward_from_gates_cooperative_h256_htile4_compact_hoist_qwarp_gate_cache_kernel"
)
_COOPERATIVE_H256_HTILE4_COMPACT_HOIST_ROW4_HIDDEN_SHMEM_GATE_CACHE_GATES_KERNEL_NAME = (
    b"a100_gru_forward_from_gates_cooperative_h256_htile4_compact_hoist_row4_hidden_shmem_gate_cache_kernel"
)
_COOPERATIVE_H256_HTILE4_COMPACT_HOIST_ROW4_WEIGHT_SHMEM_GATE_CACHE_GATES_KERNEL_NAME = (
    b"a100_gru_forward_from_gates_cooperative_h256_htile4_compact_hoist_row4_weight_shmem_gate_cache_kernel"
)
_COOPERATIVE_H256_HTILE8_COMPACT_SHMEM_GATE_CACHE_GATES_KERNEL_NAME = (
    b"a100_gru_forward_from_gates_cooperative_h256_htile8_compact_shmem_gate_cache_kernel"
)
_COOPERATIVE_H256_QWARP_SHMEM_GATES_KERNEL_NAME = (
    b"a100_gru_forward_from_gates_cooperative_h256_qwarp_shmem_kernel"
)
_COOPERATIVE_H256_CACHED_SHMEM_GATES_KERNEL_NAME = (
    b"a100_gru_forward_from_gates_cooperative_h256_cached_shmem_kernel"
)
_H256_POINTWISE_BACKWARD_KERNEL_NAME = b"a100_gru_h256_pointwise_backward_kernel"
_H256_RECURRENT_BACKWARD_KERNEL_NAME = b"a100_gru_h256_recurrent_backward_kernel"
_H256_RECURRENT_BACKWARD_TILED_KERNEL_NAME = (
    b"a100_gru_h256_recurrent_backward_tiled_kernel"
)
_H256_RECURRENT_BACKWARD_SPLIT_KERNEL_NAME = (
    b"a100_gru_h256_recurrent_backward_split_kernel"
)
_H256_RECURRENT_BACKWARD_SPLIT_REDUCE_KERNEL_NAME = (
    b"a100_gru_h256_recurrent_backward_split_reduce_kernel"
)
_H256_BACKWARD_STEP_KERNEL_NAME = b"a100_gru_h256_backward_step_kernel"
_H256_BACKWARD_STEP_COOPERATIVE_SPLIT_KERNEL_NAME = (
    b"a100_gru_h256_backward_step_cooperative_split_kernel"
)
_H256_BACKWARD_STEP_COOPERATIVE_SPLIT2_KERNEL_NAME = (
    b"a100_gru_h256_backward_step_cooperative_split2_kernel"
)
_H256_BACKWARD_STEP_COOPERATIVE_SPLIT_CACHED_KERNEL_NAME = (
    b"a100_gru_h256_backward_step_cooperative_split_cached_kernel"
)
_H256_BACKWARD_STEP_COOPERATIVE_SPLIT2_CACHED_LOCAL_KERNEL_NAME = (
    b"a100_gru_h256_backward_step_cooperative_split2_cached_local_kernel"
)
_H256_BACKWARD_STEP_COOPERATIVE_SPLIT2_GATE_CACHE_KERNEL_NAME = (
    b"a100_gru_h256_backward_step_cooperative_split2_gate_cache_kernel"
)
_H256_BACKWARD_SEQUENCE_COOPERATIVE_SPLIT2_CACHED_LOCAL_KERNEL_NAME = (
    b"a100_gru_h256_backward_sequence_cooperative_split2_cached_local_kernel"
)
_H256_BACKWARD_SEQUENCE_COOPERATIVE_SPLIT2_STATE_PARTS_KERNEL_NAME = (
    b"a100_gru_h256_backward_sequence_cooperative_split2_state_parts_kernel"
)
_H256_BACKWARD_SEQUENCE_COOPERATIVE_SPLIT2_STATE_LOCAL_KERNEL_NAME = (
    b"a100_gru_h256_backward_sequence_cooperative_split2_state_local_kernel"
)
_H256_BACKWARD_SEQUENCE_COOPERATIVE_SPLIT4_STATE_KERNEL_NAME = (
    b"a100_gru_h256_backward_sequence_cooperative_split4_state_kernel"
)
_H256_BACKWARD_SEQUENCE_COOPERATIVE_SPLIT8_STATE_KERNEL_NAME = (
    b"a100_gru_h256_backward_sequence_cooperative_split8_state_kernel"
)
_H256_BACKWARD_SEQUENCE_COOPERATIVE_SPLIT16_STATE_KERNEL_NAME = (
    b"a100_gru_h256_backward_sequence_cooperative_split16_state_kernel"
)
_H256_BACKWARD_SEQUENCE_COOPERATIVE_SPLIT16_GATE_CACHE_STATE_KERNEL_NAME = (
    b"a100_gru_h256_backward_sequence_cooperative_split16_gate_cache_state_kernel"
)
_H256_BACKWARD_SEQUENCE_COOPERATIVE_SPLIT8_GATE_CACHE_STATE_TILED_KERNEL_NAME = (
    b"a100_gru_h256_backward_sequence_cooperative_split8_gate_cache_state_tiled_kernel"
)
_H256_BACKWARD_SEQUENCE_COOPERATIVE_SPLIT16_GATE_CACHE_STATE_TILED_KERNEL_NAME = (
    b"a100_gru_h256_backward_sequence_cooperative_split16_gate_cache_state_tiled_kernel"
)
_H256_BACKWARD_SEQUENCE_COOPERATIVE_SPLIT16_GATE_CACHE_STATE_TILED_WEIGHT_SHMEM_KERNEL_NAME = (
    b"a100_gru_h256_backward_sequence_cooperative_split16_gate_cache_state_tiled_weight_shmem_kernel"
)
_H256_BACKWARD_SEQUENCE_COOPERATIVE_SPLIT16_GATE_CACHE_STATE_TILED_WEIGHT_SHMEM_SPLIT0_KEEP_KERNEL_NAME = (
    b"a100_gru_h256_backward_sequence_cooperative_split16_gate_cache_state_tiled_weight_shmem_split0_keep_kernel"
)
_H256_BACKWARD_SEQUENCE_COOPERATIVE_SPLIT16_GATE_CACHE_STATE_TILED_WEIGHT_SHMEM_SPLIT0_KEEP_OWN_SHMEM_KERNEL_NAME = (
    b"a100_gru_h256_backward_sequence_cooperative_split16_gate_cache_state_tiled_weight_shmem_split0_keep_own_shmem_kernel"
)
_H256_BACKWARD_SEQUENCE_COOPERATIVE_SPLIT5_GATE_CACHE_STATE_TILED_WEIGHT_SHMEM_SPLIT0_KEEP_KERNEL_NAME = (
    b"a100_gru_h256_backward_sequence_cooperative_split5_gate_cache_state_tiled_weight_shmem_split0_keep_kernel"
)
_H256_BACKWARD_SEQUENCE_COOPERATIVE_SPLIT6_GATE_CACHE_STATE_TILED_WEIGHT_SHMEM_SPLIT0_KEEP_KERNEL_NAME = (
    b"a100_gru_h256_backward_sequence_cooperative_split6_gate_cache_state_tiled_weight_shmem_split0_keep_kernel"
)
_H256_BACKWARD_SEQUENCE_COOPERATIVE_SPLIT12_GATE_CACHE_STATE_TILED_WEIGHT_SHMEM_SPLIT0_KEEP_KERNEL_NAME = (
    b"a100_gru_h256_backward_sequence_cooperative_split12_gate_cache_state_tiled_weight_shmem_split0_keep_kernel"
)
_H256_BACKWARD_SEQUENCE_COOPERATIVE_SPLIT12_GATE_CACHE_STATE_TILED_WEIGHT_SHMEM_SPLIT0_KEEP_UNROLL8_KERNEL_NAME = (
    b"a100_gru_h256_backward_sequence_cooperative_split12_gate_cache_state_tiled_weight_shmem_split0_keep_unroll8_kernel"
)
_H256_BACKWARD_SEQUENCE_COOPERATIVE_SPLIT24_GATE_CACHE_STATE_TILED_WEIGHT_SHMEM_SPLIT0_KEEP_KERNEL_NAME = (
    b"a100_gru_h256_backward_sequence_cooperative_split24_gate_cache_state_tiled_weight_shmem_split0_keep_kernel"
)
_H256_BACKWARD_SEQUENCE_COOPERATIVE_SPLIT16_GRAD_COEFF_CACHE_STATE_TILED_KERNEL_NAME = (
    b"a100_gru_h256_backward_sequence_cooperative_split16_grad_coeff_cache_state_tiled_kernel"
)
_H256_BACKWARD_SEQUENCE_COOPERATIVE_SPLIT32_GATE_CACHE_STATE_TILED_KERNEL_NAME = (
    b"a100_gru_h256_backward_sequence_cooperative_split32_gate_cache_state_tiled_kernel"
)
_H256_BACKWARD_SEQUENCE_COOPERATIVE_SPLIT16_STATE_GLOBAL_GATES_KERNEL_NAME = (
    b"a100_gru_h256_backward_sequence_cooperative_split16_state_global_gates_kernel"
)
_H256_BACKWARD_SEQUENCE_COOPERATIVE_SPLIT32_STATE_KERNEL_NAME = (
    b"a100_gru_h256_backward_sequence_cooperative_split32_state_kernel"
)
_H256_BACKWARD_STEP_RECOMPUTE_KERNEL_NAME = (
    b"a100_gru_h256_backward_step_recompute_kernel"
)
_H256_PACK_HIDDEN_PREV_TIME_MAJOR_KERNEL_NAME = (
    b"a100_gru_h256_pack_hidden_prev_time_major_kernel"
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
    cooperative_h256_shmem_gate_cache_gates_function: object
    cooperative_h256_shmem_grad_coeff_cache_gates_function: object
    cooperative_h256_parallel_update_gate_cache_gates_function: object
    cooperative_h256_cta8_shmem_gate_cache_gates_function: object
    cooperative_h256_cta6_shmem_gate_cache_gates_function: object
    cooperative_h256_htile2_shmem_gate_cache_gates_function: object
    cooperative_h256_htile4_shmem_gate_cache_gates_function: object
    cooperative_h256_htile4_compact_shmem_gate_cache_gates_function: object
    cooperative_h256_htile4_compact_hoist_shmem_gate_cache_gates_function: object
    cooperative_h256_htile4_compact_hoist_row4_shmem_gate_cache_gates_function: object
    cooperative_h256_htile4_compact_hoist_row4_prev_cache_shmem_gate_cache_gates_function: object
    cooperative_h256_htile4_compact_hoist_row4_parallel_update_gate_cache_gates_function: object
    cooperative_h256_htile4_compact_hoist_row4_ldg_shmem_gate_cache_gates_function: object
    cooperative_h256_htile4_compact_hoist_row3_shmem_gate_cache_gates_function: object
    cooperative_h256_htile4_compact_hoist_qwarp_gate_cache_gates_function: object
    cooperative_h256_htile4_compact_hoist_row4_hidden_shmem_gate_cache_gates_function: object
    cooperative_h256_htile4_compact_hoist_row4_weight_shmem_gate_cache_gates_function: object
    cooperative_h256_htile8_compact_shmem_gate_cache_gates_function: object
    cooperative_h256_qwarp_shmem_gates_function: object
    cooperative_h256_cached_shmem_gates_function: object
    h256_pointwise_backward_function: object
    h256_recurrent_backward_function: object
    h256_recurrent_backward_tiled_function: object
    h256_recurrent_backward_split_function: object
    h256_recurrent_backward_split_reduce_function: object
    h256_backward_step_function: object
    h256_backward_step_cooperative_split_function: object
    h256_backward_step_cooperative_split2_function: object
    h256_backward_step_cooperative_split_cached_function: object
    h256_backward_step_cooperative_split2_cached_local_function: object
    h256_backward_step_cooperative_split2_gate_cache_function: object
    h256_backward_sequence_cooperative_split2_cached_local_function: object
    h256_backward_sequence_cooperative_split2_state_parts_function: object
    h256_backward_sequence_cooperative_split2_state_local_function: object
    h256_backward_sequence_cooperative_split4_state_function: object
    h256_backward_sequence_cooperative_split8_state_function: object
    h256_backward_sequence_cooperative_split16_state_function: object
    h256_backward_sequence_cooperative_split16_gate_cache_state_function: object
    h256_backward_sequence_cooperative_split8_gate_cache_state_tiled_function: object
    h256_backward_sequence_cooperative_split16_gate_cache_state_tiled_function: object
    h256_backward_sequence_cooperative_split16_gate_cache_state_tiled_weight_shmem_function: object
    h256_backward_sequence_cooperative_split16_gate_cache_state_tiled_weight_shmem_split0_keep_function: object
    h256_backward_sequence_cooperative_split16_gate_cache_state_tiled_weight_shmem_split0_keep_own_shmem_function: object
    h256_backward_sequence_cooperative_split5_gate_cache_state_tiled_weight_shmem_split0_keep_function: object
    h256_backward_sequence_cooperative_split6_gate_cache_state_tiled_weight_shmem_split0_keep_function: object
    h256_backward_sequence_cooperative_split12_gate_cache_state_tiled_weight_shmem_split0_keep_function: object
    h256_backward_sequence_cooperative_split12_gate_cache_state_tiled_weight_shmem_split0_keep_unroll8_function: object
    h256_backward_sequence_cooperative_split24_gate_cache_state_tiled_weight_shmem_split0_keep_function: object
    h256_backward_sequence_cooperative_split16_grad_coeff_cache_state_tiled_function: object
    h256_backward_sequence_cooperative_split32_gate_cache_state_tiled_function: object
    h256_backward_sequence_cooperative_split16_state_global_gates_function: object
    h256_backward_sequence_cooperative_split32_state_function: object
    h256_backward_step_recompute_function: object
    h256_pack_hidden_prev_time_major_function: object


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
    err, cooperative_h256_shmem_gate_cache_gates_function = cuda.cuModuleGetFunction(
        module,
        _COOPERATIVE_H256_SHMEM_GATE_CACHE_GATES_KERNEL_NAME,
    )
    _check_cuda_error(err)
    err, cooperative_h256_shmem_grad_coeff_cache_gates_function = (
        cuda.cuModuleGetFunction(
            module,
            _COOPERATIVE_H256_SHMEM_GRAD_COEFF_CACHE_GATES_KERNEL_NAME,
        )
    )
    _check_cuda_error(err)
    err, cooperative_h256_parallel_update_gate_cache_gates_function = (
        cuda.cuModuleGetFunction(
            module,
            _COOPERATIVE_H256_PARALLEL_UPDATE_GATE_CACHE_GATES_KERNEL_NAME,
        )
    )
    _check_cuda_error(err)
    err, cooperative_h256_cta8_shmem_gate_cache_gates_function = (
        cuda.cuModuleGetFunction(
            module,
            _COOPERATIVE_H256_CTA8_SHMEM_GATE_CACHE_GATES_KERNEL_NAME,
        )
    )
    _check_cuda_error(err)
    err, cooperative_h256_cta6_shmem_gate_cache_gates_function = (
        cuda.cuModuleGetFunction(
            module,
            _COOPERATIVE_H256_CTA6_SHMEM_GATE_CACHE_GATES_KERNEL_NAME,
        )
    )
    _check_cuda_error(err)
    err, cooperative_h256_htile2_shmem_gate_cache_gates_function = (
        cuda.cuModuleGetFunction(
            module,
            _COOPERATIVE_H256_HTILE2_SHMEM_GATE_CACHE_GATES_KERNEL_NAME,
        )
    )
    _check_cuda_error(err)
    err, cooperative_h256_htile4_shmem_gate_cache_gates_function = (
        cuda.cuModuleGetFunction(
            module,
            _COOPERATIVE_H256_HTILE4_SHMEM_GATE_CACHE_GATES_KERNEL_NAME,
        )
    )
    _check_cuda_error(err)
    err, cooperative_h256_htile4_compact_shmem_gate_cache_gates_function = (
        cuda.cuModuleGetFunction(
            module,
            _COOPERATIVE_H256_HTILE4_COMPACT_SHMEM_GATE_CACHE_GATES_KERNEL_NAME,
        )
    )
    _check_cuda_error(err)
    err, cooperative_h256_htile4_compact_hoist_shmem_gate_cache_gates_function = (
        cuda.cuModuleGetFunction(
            module,
            _COOPERATIVE_H256_HTILE4_COMPACT_HOIST_SHMEM_GATE_CACHE_GATES_KERNEL_NAME,
        )
    )
    _check_cuda_error(err)
    err, cooperative_h256_htile4_compact_hoist_row4_shmem_gate_cache_gates_function = (
        cuda.cuModuleGetFunction(
            module,
            _COOPERATIVE_H256_HTILE4_COMPACT_HOIST_ROW4_SHMEM_GATE_CACHE_GATES_KERNEL_NAME,
        )
    )
    _check_cuda_error(err)
    err, (
        cooperative_h256_htile4_compact_hoist_row4_prev_cache_shmem_gate_cache_gates_function
    ) = cuda.cuModuleGetFunction(
        module,
        _COOPERATIVE_H256_HTILE4_COMPACT_HOIST_ROW4_PREV_CACHE_SHMEM_GATE_CACHE_GATES_KERNEL_NAME,
    )
    _check_cuda_error(err)
    err, (
        cooperative_h256_htile4_compact_hoist_row4_parallel_update_gate_cache_gates_function
    ) = cuda.cuModuleGetFunction(
        module,
        _COOPERATIVE_H256_HTILE4_COMPACT_HOIST_ROW4_PARALLEL_UPDATE_GATE_CACHE_GATES_KERNEL_NAME,
    )
    _check_cuda_error(err)
    err, cooperative_h256_htile4_compact_hoist_row4_ldg_shmem_gate_cache_gates_function = (
        cuda.cuModuleGetFunction(
            module,
            _COOPERATIVE_H256_HTILE4_COMPACT_HOIST_ROW4_LDG_SHMEM_GATE_CACHE_GATES_KERNEL_NAME,
        )
    )
    _check_cuda_error(err)
    err, cooperative_h256_htile4_compact_hoist_row3_shmem_gate_cache_gates_function = (
        cuda.cuModuleGetFunction(
            module,
            _COOPERATIVE_H256_HTILE4_COMPACT_HOIST_ROW3_SHMEM_GATE_CACHE_GATES_KERNEL_NAME,
        )
    )
    _check_cuda_error(err)
    err, cooperative_h256_htile4_compact_hoist_qwarp_gate_cache_gates_function = (
        cuda.cuModuleGetFunction(
            module,
            _COOPERATIVE_H256_HTILE4_COMPACT_HOIST_QWARP_GATE_CACHE_GATES_KERNEL_NAME,
        )
    )
    _check_cuda_error(err)
    err, (
        cooperative_h256_htile4_compact_hoist_row4_hidden_shmem_gate_cache_gates_function
    ) = cuda.cuModuleGetFunction(
        module,
        _COOPERATIVE_H256_HTILE4_COMPACT_HOIST_ROW4_HIDDEN_SHMEM_GATE_CACHE_GATES_KERNEL_NAME,
    )
    _check_cuda_error(err)
    err, (
        cooperative_h256_htile4_compact_hoist_row4_weight_shmem_gate_cache_gates_function
    ) = cuda.cuModuleGetFunction(
        module,
        _COOPERATIVE_H256_HTILE4_COMPACT_HOIST_ROW4_WEIGHT_SHMEM_GATE_CACHE_GATES_KERNEL_NAME,
    )
    _check_cuda_error(err)
    err, cooperative_h256_htile8_compact_shmem_gate_cache_gates_function = (
        cuda.cuModuleGetFunction(
            module,
            _COOPERATIVE_H256_HTILE8_COMPACT_SHMEM_GATE_CACHE_GATES_KERNEL_NAME,
        )
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
    err, h256_pointwise_backward_function = cuda.cuModuleGetFunction(
        module,
        _H256_POINTWISE_BACKWARD_KERNEL_NAME,
    )
    _check_cuda_error(err)
    err, h256_recurrent_backward_function = cuda.cuModuleGetFunction(
        module,
        _H256_RECURRENT_BACKWARD_KERNEL_NAME,
    )
    _check_cuda_error(err)
    err, h256_recurrent_backward_tiled_function = cuda.cuModuleGetFunction(
        module,
        _H256_RECURRENT_BACKWARD_TILED_KERNEL_NAME,
    )
    _check_cuda_error(err)
    err, h256_recurrent_backward_split_function = cuda.cuModuleGetFunction(
        module,
        _H256_RECURRENT_BACKWARD_SPLIT_KERNEL_NAME,
    )
    _check_cuda_error(err)
    err, h256_recurrent_backward_split_reduce_function = cuda.cuModuleGetFunction(
        module,
        _H256_RECURRENT_BACKWARD_SPLIT_REDUCE_KERNEL_NAME,
    )
    _check_cuda_error(err)
    err, h256_backward_step_function = cuda.cuModuleGetFunction(
        module,
        _H256_BACKWARD_STEP_KERNEL_NAME,
    )
    _check_cuda_error(err)
    err, h256_backward_step_cooperative_split_function = cuda.cuModuleGetFunction(
        module,
        _H256_BACKWARD_STEP_COOPERATIVE_SPLIT_KERNEL_NAME,
    )
    _check_cuda_error(err)
    err, h256_backward_step_cooperative_split2_function = cuda.cuModuleGetFunction(
        module,
        _H256_BACKWARD_STEP_COOPERATIVE_SPLIT2_KERNEL_NAME,
    )
    _check_cuda_error(err)
    err, h256_backward_step_cooperative_split_cached_function = cuda.cuModuleGetFunction(
        module,
        _H256_BACKWARD_STEP_COOPERATIVE_SPLIT_CACHED_KERNEL_NAME,
    )
    _check_cuda_error(err)
    err, h256_backward_step_cooperative_split2_cached_local_function = cuda.cuModuleGetFunction(
        module,
        _H256_BACKWARD_STEP_COOPERATIVE_SPLIT2_CACHED_LOCAL_KERNEL_NAME,
    )
    _check_cuda_error(err)
    err, h256_backward_step_cooperative_split2_gate_cache_function = cuda.cuModuleGetFunction(
        module,
        _H256_BACKWARD_STEP_COOPERATIVE_SPLIT2_GATE_CACHE_KERNEL_NAME,
    )
    _check_cuda_error(err)
    err, h256_backward_sequence_cooperative_split2_cached_local_function = (
        cuda.cuModuleGetFunction(
            module,
            _H256_BACKWARD_SEQUENCE_COOPERATIVE_SPLIT2_CACHED_LOCAL_KERNEL_NAME,
        )
    )
    _check_cuda_error(err)
    err, h256_backward_sequence_cooperative_split2_state_parts_function = (
        cuda.cuModuleGetFunction(
            module,
            _H256_BACKWARD_SEQUENCE_COOPERATIVE_SPLIT2_STATE_PARTS_KERNEL_NAME,
        )
    )
    _check_cuda_error(err)
    err, h256_backward_sequence_cooperative_split2_state_local_function = (
        cuda.cuModuleGetFunction(
            module,
            _H256_BACKWARD_SEQUENCE_COOPERATIVE_SPLIT2_STATE_LOCAL_KERNEL_NAME,
        )
    )
    _check_cuda_error(err)
    err, h256_backward_sequence_cooperative_split4_state_function = (
        cuda.cuModuleGetFunction(
            module,
            _H256_BACKWARD_SEQUENCE_COOPERATIVE_SPLIT4_STATE_KERNEL_NAME,
        )
    )
    _check_cuda_error(err)
    err, h256_backward_sequence_cooperative_split8_state_function = (
        cuda.cuModuleGetFunction(
            module,
            _H256_BACKWARD_SEQUENCE_COOPERATIVE_SPLIT8_STATE_KERNEL_NAME,
        )
    )
    _check_cuda_error(err)
    err, h256_backward_sequence_cooperative_split16_state_function = (
        cuda.cuModuleGetFunction(
            module,
            _H256_BACKWARD_SEQUENCE_COOPERATIVE_SPLIT16_STATE_KERNEL_NAME,
        )
    )
    _check_cuda_error(err)
    err, h256_backward_sequence_cooperative_split16_gate_cache_state_function = (
        cuda.cuModuleGetFunction(
            module,
            _H256_BACKWARD_SEQUENCE_COOPERATIVE_SPLIT16_GATE_CACHE_STATE_KERNEL_NAME,
        )
    )
    _check_cuda_error(err)
    err, h256_backward_sequence_cooperative_split8_gate_cache_state_tiled_function = (
        cuda.cuModuleGetFunction(
            module,
            _H256_BACKWARD_SEQUENCE_COOPERATIVE_SPLIT8_GATE_CACHE_STATE_TILED_KERNEL_NAME,
        )
    )
    _check_cuda_error(err)
    err, h256_backward_sequence_cooperative_split16_gate_cache_state_tiled_function = (
        cuda.cuModuleGetFunction(
            module,
            _H256_BACKWARD_SEQUENCE_COOPERATIVE_SPLIT16_GATE_CACHE_STATE_TILED_KERNEL_NAME,
        )
    )
    _check_cuda_error(err)
    err, (
        h256_backward_sequence_cooperative_split16_gate_cache_state_tiled_weight_shmem_function
    ) = cuda.cuModuleGetFunction(
        module,
        _H256_BACKWARD_SEQUENCE_COOPERATIVE_SPLIT16_GATE_CACHE_STATE_TILED_WEIGHT_SHMEM_KERNEL_NAME,
    )
    _check_cuda_error(err)
    err, (
        h256_backward_sequence_cooperative_split16_gate_cache_state_tiled_weight_shmem_split0_keep_function
    ) = cuda.cuModuleGetFunction(
        module,
        _H256_BACKWARD_SEQUENCE_COOPERATIVE_SPLIT16_GATE_CACHE_STATE_TILED_WEIGHT_SHMEM_SPLIT0_KEEP_KERNEL_NAME,
    )
    _check_cuda_error(err)
    err, (
        h256_backward_sequence_cooperative_split16_gate_cache_state_tiled_weight_shmem_split0_keep_own_shmem_function
    ) = cuda.cuModuleGetFunction(
        module,
        _H256_BACKWARD_SEQUENCE_COOPERATIVE_SPLIT16_GATE_CACHE_STATE_TILED_WEIGHT_SHMEM_SPLIT0_KEEP_OWN_SHMEM_KERNEL_NAME,
    )
    _check_cuda_error(err)
    err, (
        h256_backward_sequence_cooperative_split5_gate_cache_state_tiled_weight_shmem_split0_keep_function
    ) = cuda.cuModuleGetFunction(
        module,
        _H256_BACKWARD_SEQUENCE_COOPERATIVE_SPLIT5_GATE_CACHE_STATE_TILED_WEIGHT_SHMEM_SPLIT0_KEEP_KERNEL_NAME,
    )
    _check_cuda_error(err)
    err, (
        h256_backward_sequence_cooperative_split6_gate_cache_state_tiled_weight_shmem_split0_keep_function
    ) = cuda.cuModuleGetFunction(
        module,
        _H256_BACKWARD_SEQUENCE_COOPERATIVE_SPLIT6_GATE_CACHE_STATE_TILED_WEIGHT_SHMEM_SPLIT0_KEEP_KERNEL_NAME,
    )
    _check_cuda_error(err)
    err, (
        h256_backward_sequence_cooperative_split12_gate_cache_state_tiled_weight_shmem_split0_keep_function
    ) = cuda.cuModuleGetFunction(
        module,
        _H256_BACKWARD_SEQUENCE_COOPERATIVE_SPLIT12_GATE_CACHE_STATE_TILED_WEIGHT_SHMEM_SPLIT0_KEEP_KERNEL_NAME,
    )
    _check_cuda_error(err)
    err, (
        h256_backward_sequence_cooperative_split12_gate_cache_state_tiled_weight_shmem_split0_keep_unroll8_function
    ) = cuda.cuModuleGetFunction(
        module,
        _H256_BACKWARD_SEQUENCE_COOPERATIVE_SPLIT12_GATE_CACHE_STATE_TILED_WEIGHT_SHMEM_SPLIT0_KEEP_UNROLL8_KERNEL_NAME,
    )
    _check_cuda_error(err)
    err, (
        h256_backward_sequence_cooperative_split24_gate_cache_state_tiled_weight_shmem_split0_keep_function
    ) = cuda.cuModuleGetFunction(
        module,
        _H256_BACKWARD_SEQUENCE_COOPERATIVE_SPLIT24_GATE_CACHE_STATE_TILED_WEIGHT_SHMEM_SPLIT0_KEEP_KERNEL_NAME,
    )
    _check_cuda_error(err)
    err, h256_backward_sequence_cooperative_split16_grad_coeff_cache_state_tiled_function = (
        cuda.cuModuleGetFunction(
            module,
            _H256_BACKWARD_SEQUENCE_COOPERATIVE_SPLIT16_GRAD_COEFF_CACHE_STATE_TILED_KERNEL_NAME,
        )
    )
    _check_cuda_error(err)
    err, h256_backward_sequence_cooperative_split32_gate_cache_state_tiled_function = (
        cuda.cuModuleGetFunction(
            module,
            _H256_BACKWARD_SEQUENCE_COOPERATIVE_SPLIT32_GATE_CACHE_STATE_TILED_KERNEL_NAME,
        )
    )
    _check_cuda_error(err)
    err, h256_backward_sequence_cooperative_split16_state_global_gates_function = (
        cuda.cuModuleGetFunction(
            module,
            _H256_BACKWARD_SEQUENCE_COOPERATIVE_SPLIT16_STATE_GLOBAL_GATES_KERNEL_NAME,
        )
    )
    _check_cuda_error(err)
    err, h256_backward_sequence_cooperative_split32_state_function = (
        cuda.cuModuleGetFunction(
            module,
            _H256_BACKWARD_SEQUENCE_COOPERATIVE_SPLIT32_STATE_KERNEL_NAME,
        )
    )
    _check_cuda_error(err)
    err, h256_backward_step_recompute_function = cuda.cuModuleGetFunction(
        module,
        _H256_BACKWARD_STEP_RECOMPUTE_KERNEL_NAME,
    )
    _check_cuda_error(err)
    err, h256_pack_hidden_prev_time_major_function = cuda.cuModuleGetFunction(
        module,
        _H256_PACK_HIDDEN_PREV_TIME_MAJOR_KERNEL_NAME,
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
        cooperative_h256_shmem_gate_cache_gates_function=(
            cooperative_h256_shmem_gate_cache_gates_function
        ),
        cooperative_h256_shmem_grad_coeff_cache_gates_function=(
            cooperative_h256_shmem_grad_coeff_cache_gates_function
        ),
        cooperative_h256_parallel_update_gate_cache_gates_function=(
            cooperative_h256_parallel_update_gate_cache_gates_function
        ),
        cooperative_h256_cta8_shmem_gate_cache_gates_function=(
            cooperative_h256_cta8_shmem_gate_cache_gates_function
        ),
        cooperative_h256_cta6_shmem_gate_cache_gates_function=(
            cooperative_h256_cta6_shmem_gate_cache_gates_function
        ),
        cooperative_h256_htile2_shmem_gate_cache_gates_function=(
            cooperative_h256_htile2_shmem_gate_cache_gates_function
        ),
        cooperative_h256_htile4_shmem_gate_cache_gates_function=(
            cooperative_h256_htile4_shmem_gate_cache_gates_function
        ),
        cooperative_h256_htile4_compact_shmem_gate_cache_gates_function=(
            cooperative_h256_htile4_compact_shmem_gate_cache_gates_function
        ),
        cooperative_h256_htile4_compact_hoist_shmem_gate_cache_gates_function=(
            cooperative_h256_htile4_compact_hoist_shmem_gate_cache_gates_function
        ),
        cooperative_h256_htile4_compact_hoist_row4_shmem_gate_cache_gates_function=(
            cooperative_h256_htile4_compact_hoist_row4_shmem_gate_cache_gates_function
        ),
        cooperative_h256_htile4_compact_hoist_row4_prev_cache_shmem_gate_cache_gates_function=(
            cooperative_h256_htile4_compact_hoist_row4_prev_cache_shmem_gate_cache_gates_function
        ),
        cooperative_h256_htile4_compact_hoist_row4_parallel_update_gate_cache_gates_function=(
            cooperative_h256_htile4_compact_hoist_row4_parallel_update_gate_cache_gates_function
        ),
        cooperative_h256_htile4_compact_hoist_row4_ldg_shmem_gate_cache_gates_function=(
            cooperative_h256_htile4_compact_hoist_row4_ldg_shmem_gate_cache_gates_function
        ),
        cooperative_h256_htile4_compact_hoist_row3_shmem_gate_cache_gates_function=(
            cooperative_h256_htile4_compact_hoist_row3_shmem_gate_cache_gates_function
        ),
        cooperative_h256_htile4_compact_hoist_qwarp_gate_cache_gates_function=(
            cooperative_h256_htile4_compact_hoist_qwarp_gate_cache_gates_function
        ),
        cooperative_h256_htile4_compact_hoist_row4_hidden_shmem_gate_cache_gates_function=(
            cooperative_h256_htile4_compact_hoist_row4_hidden_shmem_gate_cache_gates_function
        ),
        cooperative_h256_htile4_compact_hoist_row4_weight_shmem_gate_cache_gates_function=(
            cooperative_h256_htile4_compact_hoist_row4_weight_shmem_gate_cache_gates_function
        ),
        cooperative_h256_htile8_compact_shmem_gate_cache_gates_function=(
            cooperative_h256_htile8_compact_shmem_gate_cache_gates_function
        ),
        cooperative_h256_qwarp_shmem_gates_function=(
            cooperative_h256_qwarp_shmem_gates_function
        ),
        cooperative_h256_cached_shmem_gates_function=(
            cooperative_h256_cached_shmem_gates_function
        ),
        h256_pointwise_backward_function=h256_pointwise_backward_function,
        h256_recurrent_backward_function=h256_recurrent_backward_function,
        h256_recurrent_backward_tiled_function=h256_recurrent_backward_tiled_function,
        h256_recurrent_backward_split_function=h256_recurrent_backward_split_function,
        h256_recurrent_backward_split_reduce_function=(
            h256_recurrent_backward_split_reduce_function
        ),
        h256_backward_step_function=h256_backward_step_function,
        h256_backward_step_cooperative_split_function=(
            h256_backward_step_cooperative_split_function
        ),
        h256_backward_step_cooperative_split2_function=(
            h256_backward_step_cooperative_split2_function
        ),
        h256_backward_step_cooperative_split_cached_function=(
            h256_backward_step_cooperative_split_cached_function
        ),
        h256_backward_step_cooperative_split2_cached_local_function=(
            h256_backward_step_cooperative_split2_cached_local_function
        ),
        h256_backward_step_cooperative_split2_gate_cache_function=(
            h256_backward_step_cooperative_split2_gate_cache_function
        ),
        h256_backward_sequence_cooperative_split2_cached_local_function=(
            h256_backward_sequence_cooperative_split2_cached_local_function
        ),
        h256_backward_sequence_cooperative_split2_state_parts_function=(
            h256_backward_sequence_cooperative_split2_state_parts_function
        ),
        h256_backward_sequence_cooperative_split2_state_local_function=(
            h256_backward_sequence_cooperative_split2_state_local_function
        ),
        h256_backward_sequence_cooperative_split4_state_function=(
            h256_backward_sequence_cooperative_split4_state_function
        ),
        h256_backward_sequence_cooperative_split8_state_function=(
            h256_backward_sequence_cooperative_split8_state_function
        ),
        h256_backward_sequence_cooperative_split16_state_function=(
            h256_backward_sequence_cooperative_split16_state_function
        ),
        h256_backward_sequence_cooperative_split16_gate_cache_state_function=(
            h256_backward_sequence_cooperative_split16_gate_cache_state_function
        ),
        h256_backward_sequence_cooperative_split8_gate_cache_state_tiled_function=(
            h256_backward_sequence_cooperative_split8_gate_cache_state_tiled_function
        ),
        h256_backward_sequence_cooperative_split16_gate_cache_state_tiled_function=(
            h256_backward_sequence_cooperative_split16_gate_cache_state_tiled_function
        ),
        h256_backward_sequence_cooperative_split16_gate_cache_state_tiled_weight_shmem_function=(
            h256_backward_sequence_cooperative_split16_gate_cache_state_tiled_weight_shmem_function
        ),
        h256_backward_sequence_cooperative_split16_gate_cache_state_tiled_weight_shmem_split0_keep_function=(
            h256_backward_sequence_cooperative_split16_gate_cache_state_tiled_weight_shmem_split0_keep_function
        ),
        h256_backward_sequence_cooperative_split16_gate_cache_state_tiled_weight_shmem_split0_keep_own_shmem_function=(
            h256_backward_sequence_cooperative_split16_gate_cache_state_tiled_weight_shmem_split0_keep_own_shmem_function
        ),
        h256_backward_sequence_cooperative_split5_gate_cache_state_tiled_weight_shmem_split0_keep_function=(
            h256_backward_sequence_cooperative_split5_gate_cache_state_tiled_weight_shmem_split0_keep_function
        ),
        h256_backward_sequence_cooperative_split6_gate_cache_state_tiled_weight_shmem_split0_keep_function=(
            h256_backward_sequence_cooperative_split6_gate_cache_state_tiled_weight_shmem_split0_keep_function
        ),
        h256_backward_sequence_cooperative_split12_gate_cache_state_tiled_weight_shmem_split0_keep_function=(
            h256_backward_sequence_cooperative_split12_gate_cache_state_tiled_weight_shmem_split0_keep_function
        ),
        h256_backward_sequence_cooperative_split12_gate_cache_state_tiled_weight_shmem_split0_keep_unroll8_function=(
            h256_backward_sequence_cooperative_split12_gate_cache_state_tiled_weight_shmem_split0_keep_unroll8_function
        ),
        h256_backward_sequence_cooperative_split24_gate_cache_state_tiled_weight_shmem_split0_keep_function=(
            h256_backward_sequence_cooperative_split24_gate_cache_state_tiled_weight_shmem_split0_keep_function
        ),
        h256_backward_sequence_cooperative_split16_grad_coeff_cache_state_tiled_function=(
            h256_backward_sequence_cooperative_split16_grad_coeff_cache_state_tiled_function
        ),
        h256_backward_sequence_cooperative_split32_gate_cache_state_tiled_function=(
            h256_backward_sequence_cooperative_split32_gate_cache_state_tiled_function
        ),
        h256_backward_sequence_cooperative_split16_state_global_gates_function=(
            h256_backward_sequence_cooperative_split16_state_global_gates_function
        ),
        h256_backward_sequence_cooperative_split32_state_function=(
            h256_backward_sequence_cooperative_split32_state_function
        ),
        h256_backward_step_recompute_function=h256_backward_step_recompute_function,
        h256_pack_hidden_prev_time_major_function=h256_pack_hidden_prev_time_major_function,
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


def a100_gru_forward_from_gates_cooperative_h256_shmem_gate_cache(
    input_gates: torch.Tensor,
    h0: torch.Tensor,
    weight_hh: torch.Tensor,
    bias_hh: torch.Tensor,
    block_threads: int = 704,
) -> tuple[torch.Tensor, torch.Tensor]:
    """运行 h256 shmem cooperative forward，并保存 backward 需要的 gate cache。"""
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
            "a100_gru_forward_from_gates_cooperative_h256_shmem_gate_cache "
            "requires hidden_size=256."
        )

    device_index = input_gates.device.index
    if device_index is None:
        device_index = torch.cuda.current_device()
    torch.cuda.set_device(device_index)
    capability = _device_capability(device_index)
    if capability != (8, 0):
        raise RuntimeError(
            "a100_gru_forward_from_gates_cooperative_h256_shmem_gate_cache "
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
    gate_cache = torch.empty(
        batch_size,
        seq_len,
        4 * hidden_size,
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
        compiled.cooperative_h256_shmem_gate_cache_gates_function,
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
            "h256 gate-cache shmem cooperative grid is too large for resident launch: "
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
        compiled.cooperative_h256_shmem_gate_cache_gates_function,
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


def a100_gru_forward_from_gates_cooperative_h256_shmem_grad_coeff_cache(
    input_gates: torch.Tensor,
    h0: torch.Tensor,
    weight_hh: torch.Tensor,
    bias_hh: torch.Tensor,
    block_threads: int = 704,
) -> tuple[torch.Tensor, torch.Tensor]:
    """运行 h256 shmem cooperative forward，并保存 backward 导数系数 cache。"""
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
            "a100_gru_forward_from_gates_cooperative_h256_shmem_grad_coeff_cache "
            "requires hidden_size=256."
        )

    device_index = input_gates.device.index
    if device_index is None:
        device_index = torch.cuda.current_device()
    torch.cuda.set_device(device_index)
    capability = _device_capability(device_index)
    if capability != (8, 0):
        raise RuntimeError(
            "a100_gru_forward_from_gates_cooperative_h256_shmem_grad_coeff_cache "
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
    grad_coeff_cache = torch.empty(
        batch_size,
        seq_len,
        5 * hidden_size,
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
        compiled.cooperative_h256_shmem_grad_coeff_cache_gates_function,
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
            "h256 grad-coeff-cache shmem cooperative grid is too large for resident launch: "
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
        ctypes.c_void_p(grad_coeff_cache.data_ptr()),
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
        compiled.cooperative_h256_shmem_grad_coeff_cache_gates_function,
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
    return output, grad_coeff_cache


def a100_gru_forward_from_gates_cooperative_h256_parallel_update_gate_cache(
    input_gates: torch.Tensor,
    h0: torch.Tensor,
    weight_hh: torch.Tensor,
    bias_hh: torch.Tensor,
    block_threads: int = 704,
) -> tuple[torch.Tensor, torch.Tensor]:
    """运行 4 CTA 分摊 hidden update 的 h256 gate-cache cooperative kernel。"""
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
            "a100_gru_forward_from_gates_cooperative_h256_parallel_update_gate_cache "
            "requires hidden_size=256."
        )

    device_index = input_gates.device.index
    if device_index is None:
        device_index = torch.cuda.current_device()
    torch.cuda.set_device(device_index)
    capability = _device_capability(device_index)
    if capability != (8, 0):
        raise RuntimeError(
            "a100_gru_forward_from_gates_cooperative_h256_parallel_update_gate_cache "
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
    gate_cache = torch.empty(
        batch_size,
        seq_len,
        4 * hidden_size,
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
        compiled.cooperative_h256_parallel_update_gate_cache_gates_function,
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
            "h256 parallel-update gate-cache cooperative grid is too large "
            "for resident launch: "
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
        compiled.cooperative_h256_parallel_update_gate_cache_gates_function,
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


def a100_gru_forward_from_gates_cooperative_h256_cta8_shmem_gate_cache(
    input_gates: torch.Tensor,
    h0: torch.Tensor,
    weight_hh: torch.Tensor,
    bias_hh: torch.Tensor,
    block_threads: int = 704,
) -> tuple[torch.Tensor, torch.Tensor]:
    """运行 8 CTA h256 shmem cooperative forward，并保存 gate cache。"""
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
            "a100_gru_forward_from_gates_cooperative_h256_cta8_shmem_gate_cache "
            "requires hidden_size=256."
        )

    device_index = input_gates.device.index
    if device_index is None:
        device_index = torch.cuda.current_device()
    torch.cuda.set_device(device_index)
    capability = _device_capability(device_index)
    if capability != (8, 0):
        raise RuntimeError(
            "a100_gru_forward_from_gates_cooperative_h256_cta8_shmem_gate_cache "
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

    ctas_per_batch = 8
    output = torch.empty(
        batch_size,
        seq_len,
        hidden_size,
        device=input_gates.device,
        dtype=input_gates.dtype,
    )
    gate_cache = torch.empty(
        batch_size,
        seq_len,
        4 * hidden_size,
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
        compiled.cooperative_h256_cta8_shmem_gate_cache_gates_function,
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
            "h256 cta8 gate-cache shmem cooperative grid is too large "
            "for resident launch: "
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
        compiled.cooperative_h256_cta8_shmem_gate_cache_gates_function,
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


def a100_gru_forward_from_gates_cooperative_h256_cta6_shmem_gate_cache(
    input_gates: torch.Tensor,
    h0: torch.Tensor,
    weight_hh: torch.Tensor,
    bias_hh: torch.Tensor,
    block_threads: int = 704,
) -> tuple[torch.Tensor, torch.Tensor]:
    """运行 6 CTA/batch 的 h256 gate-cache cooperative kernel。"""
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
            "a100_gru_forward_from_gates_cooperative_h256_cta6_shmem_gate_cache "
            "requires hidden_size=256."
        )

    device_index = input_gates.device.index
    if device_index is None:
        device_index = torch.cuda.current_device()
    torch.cuda.set_device(device_index)
    capability = _device_capability(device_index)
    if capability != (8, 0):
        raise RuntimeError(
            "a100_gru_forward_from_gates_cooperative_h256_cta6_shmem_gate_cache "
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

    ctas_per_batch = 6
    output = torch.empty(
        batch_size,
        seq_len,
        hidden_size,
        device=input_gates.device,
        dtype=input_gates.dtype,
    )
    gate_cache = torch.empty(
        batch_size,
        seq_len,
        4 * hidden_size,
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
        compiled.cooperative_h256_cta6_shmem_gate_cache_gates_function,
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
            "h256 cta6 gate-cache shmem cooperative grid is too large "
            "for resident launch: "
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
        compiled.cooperative_h256_cta6_shmem_gate_cache_gates_function,
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


def a100_gru_forward_from_gates_cooperative_h256_htile2_shmem_gate_cache(
    input_gates: torch.Tensor,
    h0: torch.Tensor,
    weight_hh: torch.Tensor,
    bias_hh: torch.Tensor,
    block_threads: int = 704,
) -> tuple[torch.Tensor, torch.Tensor]:
    """运行 2 个 hidden tile 的 h256 gate-cache cooperative kernel。"""
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
            "a100_gru_forward_from_gates_cooperative_h256_htile2_shmem_gate_cache "
            "requires hidden_size=256."
        )

    device_index = input_gates.device.index
    if device_index is None:
        device_index = torch.cuda.current_device()
    torch.cuda.set_device(device_index)
    capability = _device_capability(device_index)
    if capability != (8, 0):
        raise RuntimeError(
            "a100_gru_forward_from_gates_cooperative_h256_htile2_shmem_gate_cache "
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

    hidden_tile = hidden_size // 2
    ctas_per_batch = 8
    output = torch.empty(
        batch_size,
        seq_len,
        hidden_size,
        device=input_gates.device,
        dtype=input_gates.dtype,
    )
    gate_cache = torch.empty(
        batch_size,
        seq_len,
        4 * hidden_size,
        device=input_gates.device,
        dtype=input_gates.dtype,
    )
    partial_gates = torch.empty(
        batch_size * ctas_per_batch,
        3 * hidden_tile,
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
    shared_bytes = 3 * hidden_tile * ctypes.sizeof(ctypes.c_float)
    err, active_blocks_per_sm = cuda.cuOccupancyMaxActiveBlocksPerMultiprocessor(
        compiled.cooperative_h256_htile2_shmem_gate_cache_gates_function,
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
            "h256 htile2 gate-cache shmem cooperative grid is too large "
            "for resident launch: "
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
        compiled.cooperative_h256_htile2_shmem_gate_cache_gates_function,
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


def a100_gru_forward_from_gates_cooperative_h256_htile4_shmem_gate_cache(
    input_gates: torch.Tensor,
    h0: torch.Tensor,
    weight_hh: torch.Tensor,
    bias_hh: torch.Tensor,
    block_threads: int = 256,
) -> tuple[torch.Tensor, torch.Tensor]:
    """运行 4 个 hidden tile 的 h256 gate-cache cooperative kernel。"""
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
            "a100_gru_forward_from_gates_cooperative_h256_htile4_shmem_gate_cache "
            "requires hidden_size=256."
        )

    device_index = input_gates.device.index
    if device_index is None:
        device_index = torch.cuda.current_device()
    torch.cuda.set_device(device_index)
    capability = _device_capability(device_index)
    if capability != (8, 0):
        raise RuntimeError(
            "a100_gru_forward_from_gates_cooperative_h256_htile4_shmem_gate_cache "
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

    hidden_tile = hidden_size // 4
    ctas_per_batch = 16
    output = torch.empty(
        batch_size,
        seq_len,
        hidden_size,
        device=input_gates.device,
        dtype=input_gates.dtype,
    )
    gate_cache = torch.empty(
        batch_size,
        seq_len,
        4 * hidden_size,
        device=input_gates.device,
        dtype=input_gates.dtype,
    )
    partial_gates = torch.empty(
        batch_size * ctas_per_batch,
        3 * hidden_tile,
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
    shared_bytes = 3 * hidden_tile * ctypes.sizeof(ctypes.c_float)
    err, active_blocks_per_sm = cuda.cuOccupancyMaxActiveBlocksPerMultiprocessor(
        compiled.cooperative_h256_htile4_shmem_gate_cache_gates_function,
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
            "h256 htile4 gate-cache shmem cooperative grid is too large "
            "for resident launch: "
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
        compiled.cooperative_h256_htile4_shmem_gate_cache_gates_function,
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


def a100_gru_forward_from_gates_cooperative_h256_htile4_compact_shmem_gate_cache(
    input_gates: torch.Tensor,
    h0: torch.Tensor,
    weight_hh: torch.Tensor,
    bias_hh: torch.Tensor,
    block_threads: int = 256,
) -> tuple[torch.Tensor, torch.Tensor]:
    """运行 compact partial buffer 的 4 hidden tile h256 gate-cache kernel。"""
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
            "a100_gru_forward_from_gates_cooperative_h256_htile4_compact_shmem_gate_cache "
            "requires hidden_size=256."
        )

    device_index = input_gates.device.index
    if device_index is None:
        device_index = torch.cuda.current_device()
    torch.cuda.set_device(device_index)
    capability = _device_capability(device_index)
    if capability != (8, 0):
        raise RuntimeError(
            "a100_gru_forward_from_gates_cooperative_h256_htile4_compact_shmem_gate_cache "
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

    hidden_tile = hidden_size // 4
    ctas_per_batch = 16
    compact_partials_per_batch = 12
    output = torch.empty(
        batch_size,
        seq_len,
        hidden_size,
        device=input_gates.device,
        dtype=input_gates.dtype,
    )
    gate_cache = torch.empty(
        batch_size,
        seq_len,
        4 * hidden_size,
        device=input_gates.device,
        dtype=input_gates.dtype,
    )
    partial_gates = torch.empty(
        batch_size * compact_partials_per_batch,
        3 * hidden_tile,
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
    shared_bytes = 3 * hidden_tile * ctypes.sizeof(ctypes.c_float)
    err, active_blocks_per_sm = cuda.cuOccupancyMaxActiveBlocksPerMultiprocessor(
        compiled.cooperative_h256_htile4_compact_shmem_gate_cache_gates_function,
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
            "h256 htile4 compact gate-cache shmem cooperative grid is too large "
            "for resident launch: "
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
        compiled.cooperative_h256_htile4_compact_shmem_gate_cache_gates_function,
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


def a100_gru_forward_from_gates_cooperative_h256_htile4_compact_hoist_shmem_gate_cache(
    input_gates: torch.Tensor,
    h0: torch.Tensor,
    weight_hh: torch.Tensor,
    bias_hh: torch.Tensor,
    block_threads: int = 256,
) -> tuple[torch.Tensor, torch.Tensor]:
    """运行 hoist partial 地址计算的 4 hidden tile compact kernel。"""
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
            "a100_gru_forward_from_gates_cooperative_h256_htile4_compact_hoist_shmem_gate_cache "
            "requires hidden_size=256."
        )

    device_index = input_gates.device.index
    if device_index is None:
        device_index = torch.cuda.current_device()
    torch.cuda.set_device(device_index)
    capability = _device_capability(device_index)
    if capability != (8, 0):
        raise RuntimeError(
            "a100_gru_forward_from_gates_cooperative_h256_htile4_compact_hoist_shmem_gate_cache "
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

    hidden_tile = hidden_size // 4
    ctas_per_batch = 16
    compact_partials_per_batch = 12
    output = torch.empty(
        batch_size,
        seq_len,
        hidden_size,
        device=input_gates.device,
        dtype=input_gates.dtype,
    )
    gate_cache = torch.empty(
        batch_size,
        seq_len,
        4 * hidden_size,
        device=input_gates.device,
        dtype=input_gates.dtype,
    )
    partial_gates = torch.empty(
        batch_size * compact_partials_per_batch,
        3 * hidden_tile,
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
    shared_bytes = 3 * hidden_tile * ctypes.sizeof(ctypes.c_float)
    err, active_blocks_per_sm = cuda.cuOccupancyMaxActiveBlocksPerMultiprocessor(
        compiled.cooperative_h256_htile4_compact_hoist_shmem_gate_cache_gates_function,
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
            "h256 htile4 compact hoist gate-cache shmem cooperative grid is too large "
            "for resident launch: "
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
        compiled.cooperative_h256_htile4_compact_hoist_shmem_gate_cache_gates_function,
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


def a100_gru_forward_from_gates_cooperative_h256_htile4_compact_hoist_row4_shmem_gate_cache(
    input_gates: torch.Tensor,
    h0: torch.Tensor,
    weight_hh: torch.Tensor,
    bias_hh: torch.Tensor,
    block_threads: int = 256,
) -> tuple[torch.Tensor, torch.Tensor]:
    """运行 row4 hidden 行分配的 htile4 compact hoist kernel。"""
    _require_cuda_python()
    _validate_block_threads(block_threads)
    if block_threads != 256:
        raise ValueError(
            "a100_gru_forward_from_gates_cooperative_h256_htile4_compact_hoist_row4_shmem_gate_cache "
            "requires block_threads=256."
        )
    batch_size, seq_len, hidden_size = _validate_input_gates(
        input_gates,
        h0,
        weight_hh,
        bias_hh,
    )
    if hidden_size != 256:
        raise ValueError(
            "a100_gru_forward_from_gates_cooperative_h256_htile4_compact_hoist_row4_shmem_gate_cache "
            "requires hidden_size=256."
        )

    device_index = input_gates.device.index
    if device_index is None:
        device_index = torch.cuda.current_device()
    torch.cuda.set_device(device_index)
    capability = _device_capability(device_index)
    if capability != (8, 0):
        raise RuntimeError(
            "a100_gru_forward_from_gates_cooperative_h256_htile4_compact_hoist_row4_shmem_gate_cache "
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

    hidden_tile = hidden_size // 4
    ctas_per_batch = 16
    compact_partials_per_batch = 12
    output = torch.empty(
        batch_size,
        seq_len,
        hidden_size,
        device=input_gates.device,
        dtype=input_gates.dtype,
    )
    gate_cache = torch.empty(
        batch_size,
        seq_len,
        4 * hidden_size,
        device=input_gates.device,
        dtype=input_gates.dtype,
    )
    partial_gates = torch.empty(
        batch_size * compact_partials_per_batch,
        3 * hidden_tile,
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
    shared_bytes = 3 * hidden_tile * ctypes.sizeof(ctypes.c_float)
    err, active_blocks_per_sm = cuda.cuOccupancyMaxActiveBlocksPerMultiprocessor(
        compiled.cooperative_h256_htile4_compact_hoist_row4_shmem_gate_cache_gates_function,
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
            "h256 htile4 compact hoist row4 gate-cache shmem cooperative grid is too large "
            "for resident launch: "
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
        compiled.cooperative_h256_htile4_compact_hoist_row4_shmem_gate_cache_gates_function,
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


def a100_gru_forward_from_gates_cooperative_h256_htile4_compact_hoist_row4_prev_cache_shmem_gate_cache(
    input_gates: torch.Tensor,
    h0: torch.Tensor,
    weight_hh: torch.Tensor,
    bias_hh: torch.Tensor,
    block_threads: int = 256,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """运行额外保存 time-major hidden_prev cache 的 row4 kernel。"""
    _require_cuda_python()
    _validate_block_threads(block_threads)
    if block_threads != 256:
        raise ValueError(
            "a100_gru_forward_from_gates_cooperative_h256_htile4_compact_hoist_row4_prev_cache_shmem_gate_cache "
            "requires block_threads=256."
        )
    batch_size, seq_len, hidden_size = _validate_input_gates(
        input_gates,
        h0,
        weight_hh,
        bias_hh,
    )
    if hidden_size != 256:
        raise ValueError(
            "a100_gru_forward_from_gates_cooperative_h256_htile4_compact_hoist_row4_prev_cache_shmem_gate_cache "
            "requires hidden_size=256."
        )

    device_index = input_gates.device.index
    if device_index is None:
        device_index = torch.cuda.current_device()
    torch.cuda.set_device(device_index)
    capability = _device_capability(device_index)
    if capability != (8, 0):
        raise RuntimeError(
            "a100_gru_forward_from_gates_cooperative_h256_htile4_compact_hoist_row4_prev_cache_shmem_gate_cache "
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

    hidden_tile = hidden_size // 4
    ctas_per_batch = 16
    compact_partials_per_batch = 12
    output = torch.empty(
        batch_size,
        seq_len,
        hidden_size,
        device=input_gates.device,
        dtype=input_gates.dtype,
    )
    gate_cache = torch.empty(
        batch_size,
        seq_len,
        4 * hidden_size,
        device=input_gates.device,
        dtype=input_gates.dtype,
    )
    hidden_prev_cache = torch.empty(
        seq_len,
        batch_size,
        hidden_size,
        device=input_gates.device,
        dtype=input_gates.dtype,
    )
    partial_gates = torch.empty(
        batch_size * compact_partials_per_batch,
        3 * hidden_tile,
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
    function = (
        compiled.cooperative_h256_htile4_compact_hoist_row4_prev_cache_shmem_gate_cache_gates_function
    )
    shared_bytes = 3 * hidden_tile * ctypes.sizeof(ctypes.c_float)
    err, active_blocks_per_sm = cuda.cuOccupancyMaxActiveBlocksPerMultiprocessor(
        function,
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
            "h256 htile4 compact hoist row4 prev-cache cooperative grid is too large "
            "for resident launch: "
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
        ctypes.c_void_p(hidden_prev_cache.data_ptr()),
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
    )
    err, = cuda.cuLaunchCooperativeKernel(
        function,
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
    return output, gate_cache, hidden_prev_cache


def a100_gru_forward_from_gates_cooperative_h256_htile4_compact_hoist_row4_parallel_update_gate_cache(
    input_gates: torch.Tensor,
    h0: torch.Tensor,
    weight_hh: torch.Tensor,
    bias_hh: torch.Tensor,
    block_threads: int = 256,
) -> tuple[torch.Tensor, torch.Tensor]:
    """运行让 4 个 k CTA 分摊 gate/update/cache 的 row4 kernel。"""
    _require_cuda_python()
    _validate_block_threads(block_threads)
    if block_threads != 256:
        raise ValueError(
            "a100_gru_forward_from_gates_cooperative_h256_htile4_compact_hoist_row4_parallel_update_gate_cache "
            "requires block_threads=256."
        )
    batch_size, seq_len, hidden_size = _validate_input_gates(
        input_gates,
        h0,
        weight_hh,
        bias_hh,
    )
    if hidden_size != 256:
        raise ValueError(
            "a100_gru_forward_from_gates_cooperative_h256_htile4_compact_hoist_row4_parallel_update_gate_cache "
            "requires hidden_size=256."
        )

    device_index = input_gates.device.index
    if device_index is None:
        device_index = torch.cuda.current_device()
    torch.cuda.set_device(device_index)
    capability = _device_capability(device_index)
    if capability != (8, 0):
        raise RuntimeError(
            "a100_gru_forward_from_gates_cooperative_h256_htile4_compact_hoist_row4_parallel_update_gate_cache "
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

    hidden_tile = hidden_size // 4
    ctas_per_batch = 16
    partials_per_batch = 16
    output = torch.empty(
        batch_size,
        seq_len,
        hidden_size,
        device=input_gates.device,
        dtype=input_gates.dtype,
    )
    gate_cache = torch.empty(
        batch_size,
        seq_len,
        4 * hidden_size,
        device=input_gates.device,
        dtype=input_gates.dtype,
    )
    partial_gates = torch.empty(
        batch_size * partials_per_batch,
        3 * hidden_tile,
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
    function = (
        compiled.cooperative_h256_htile4_compact_hoist_row4_parallel_update_gate_cache_gates_function
    )
    shared_bytes = 0
    err, active_blocks_per_sm = cuda.cuOccupancyMaxActiveBlocksPerMultiprocessor(
        function,
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
            "h256 htile4 compact hoist row4 parallel-update cooperative grid is too large "
            "for resident launch: "
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
        function,
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


def a100_gru_forward_from_gates_cooperative_h256_htile4_compact_hoist_row4_ldg_shmem_gate_cache(
    input_gates: torch.Tensor,
    h0: torch.Tensor,
    weight_hh: torch.Tensor,
    bias_hh: torch.Tensor,
    block_threads: int = 256,
) -> tuple[torch.Tensor, torch.Tensor]:
    """运行对只读输入显式使用 ldg 的 row4 htile4 compact hoist kernel。"""
    _require_cuda_python()
    _validate_block_threads(block_threads)
    if block_threads != 256:
        raise ValueError(
            "a100_gru_forward_from_gates_cooperative_h256_htile4_compact_hoist_row4_ldg_shmem_gate_cache "
            "requires block_threads=256."
        )
    batch_size, seq_len, hidden_size = _validate_input_gates(
        input_gates,
        h0,
        weight_hh,
        bias_hh,
    )
    if hidden_size != 256:
        raise ValueError(
            "a100_gru_forward_from_gates_cooperative_h256_htile4_compact_hoist_row4_ldg_shmem_gate_cache "
            "requires hidden_size=256."
        )

    device_index = input_gates.device.index
    if device_index is None:
        device_index = torch.cuda.current_device()
    torch.cuda.set_device(device_index)
    capability = _device_capability(device_index)
    if capability != (8, 0):
        raise RuntimeError(
            "a100_gru_forward_from_gates_cooperative_h256_htile4_compact_hoist_row4_ldg_shmem_gate_cache "
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

    hidden_tile = hidden_size // 4
    ctas_per_batch = 16
    compact_partials_per_batch = 12
    output = torch.empty(
        batch_size,
        seq_len,
        hidden_size,
        device=input_gates.device,
        dtype=input_gates.dtype,
    )
    gate_cache = torch.empty(
        batch_size,
        seq_len,
        4 * hidden_size,
        device=input_gates.device,
        dtype=input_gates.dtype,
    )
    partial_gates = torch.empty(
        batch_size * compact_partials_per_batch,
        3 * hidden_tile,
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
    function = (
        compiled.cooperative_h256_htile4_compact_hoist_row4_ldg_shmem_gate_cache_gates_function
    )
    shared_bytes = 3 * hidden_tile * ctypes.sizeof(ctypes.c_float)
    err, active_blocks_per_sm = cuda.cuOccupancyMaxActiveBlocksPerMultiprocessor(
        function,
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
            "h256 htile4 compact hoist row4 ldg gate-cache shmem cooperative grid is too large "
            "for resident launch: "
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
        function,
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


def a100_gru_forward_from_gates_cooperative_h256_htile4_compact_hoist_qwarp_gate_cache(
    input_gates: torch.Tensor,
    h0: torch.Tensor,
    weight_hh: torch.Tensor,
    bias_hh: torch.Tensor,
    block_threads: int = 256,
) -> tuple[torch.Tensor, torch.Tensor]:
    """运行 quarter-warp row 分配的 htile4 compact hoist gate-cache kernel。"""
    _require_cuda_python()
    _validate_block_threads(block_threads)
    if block_threads != 256:
        raise ValueError(
            "a100_gru_forward_from_gates_cooperative_h256_htile4_compact_hoist_qwarp_gate_cache "
            "requires block_threads=256."
        )
    batch_size, seq_len, hidden_size = _validate_input_gates(
        input_gates,
        h0,
        weight_hh,
        bias_hh,
    )
    if hidden_size != 256:
        raise ValueError(
            "a100_gru_forward_from_gates_cooperative_h256_htile4_compact_hoist_qwarp_gate_cache "
            "requires hidden_size=256."
        )

    device_index = input_gates.device.index
    if device_index is None:
        device_index = torch.cuda.current_device()
    torch.cuda.set_device(device_index)
    capability = _device_capability(device_index)
    if capability != (8, 0):
        raise RuntimeError(
            "a100_gru_forward_from_gates_cooperative_h256_htile4_compact_hoist_qwarp_gate_cache "
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

    hidden_tile = hidden_size // 4
    ctas_per_batch = 16
    compact_partials_per_batch = 12
    output = torch.empty(
        batch_size,
        seq_len,
        hidden_size,
        device=input_gates.device,
        dtype=input_gates.dtype,
    )
    gate_cache = torch.empty(
        batch_size,
        seq_len,
        4 * hidden_size,
        device=input_gates.device,
        dtype=input_gates.dtype,
    )
    partial_gates = torch.empty(
        batch_size * compact_partials_per_batch,
        3 * hidden_tile,
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
    function = compiled.cooperative_h256_htile4_compact_hoist_qwarp_gate_cache_gates_function
    shared_bytes = 3 * hidden_tile * ctypes.sizeof(ctypes.c_float)
    err, active_blocks_per_sm = cuda.cuOccupancyMaxActiveBlocksPerMultiprocessor(
        function,
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
            "h256 htile4 compact hoist qwarp gate-cache cooperative grid is too large "
            "for resident launch: "
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
        function,
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


def a100_gru_forward_from_gates_cooperative_h256_htile4_compact_hoist_row3_shmem_gate_cache(
    input_gates: torch.Tensor,
    h0: torch.Tensor,
    weight_hh: torch.Tensor,
    bias_hh: torch.Tensor,
    block_threads: int = 256,
) -> tuple[torch.Tensor, torch.Tensor]:
    """运行 3+1 hidden 行分配的 htile4 compact hoist gate-cache kernel。"""
    _require_cuda_python()
    _validate_block_threads(block_threads)
    if block_threads != 256:
        raise ValueError(
            "a100_gru_forward_from_gates_cooperative_h256_htile4_compact_hoist_row3_shmem_gate_cache "
            "requires block_threads=256."
        )
    batch_size, seq_len, hidden_size = _validate_input_gates(
        input_gates,
        h0,
        weight_hh,
        bias_hh,
    )
    if hidden_size != 256:
        raise ValueError(
            "a100_gru_forward_from_gates_cooperative_h256_htile4_compact_hoist_row3_shmem_gate_cache "
            "requires hidden_size=256."
        )

    device_index = input_gates.device.index
    if device_index is None:
        device_index = torch.cuda.current_device()
    torch.cuda.set_device(device_index)
    capability = _device_capability(device_index)
    if capability != (8, 0):
        raise RuntimeError(
            "a100_gru_forward_from_gates_cooperative_h256_htile4_compact_hoist_row3_shmem_gate_cache "
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

    hidden_tile = hidden_size // 4
    ctas_per_batch = 16
    compact_partials_per_batch = 12
    output = torch.empty(
        batch_size,
        seq_len,
        hidden_size,
        device=input_gates.device,
        dtype=input_gates.dtype,
    )
    gate_cache = torch.empty(
        batch_size,
        seq_len,
        4 * hidden_size,
        device=input_gates.device,
        dtype=input_gates.dtype,
    )
    partial_gates = torch.empty(
        batch_size * compact_partials_per_batch,
        3 * hidden_tile,
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
    function = compiled.cooperative_h256_htile4_compact_hoist_row3_shmem_gate_cache_gates_function
    shared_bytes = 3 * hidden_tile * ctypes.sizeof(ctypes.c_float)
    err, active_blocks_per_sm = cuda.cuOccupancyMaxActiveBlocksPerMultiprocessor(
        function,
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
            "h256 htile4 compact hoist row3 gate-cache cooperative grid is too large "
            "for resident launch: "
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
        function,
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


def a100_gru_forward_from_gates_cooperative_h256_htile4_compact_hoist_row4_hidden_shmem_gate_cache(
    input_gates: torch.Tensor,
    h0: torch.Tensor,
    weight_hh: torch.Tensor,
    bias_hh: torch.Tensor,
    block_threads: int = 256,
) -> tuple[torch.Tensor, torch.Tensor]:
    """运行把每步 hidden k tile 放入 shared memory 的 row4 kernel。"""
    _require_cuda_python()
    _validate_block_threads(block_threads)
    if block_threads != 256:
        raise ValueError(
            "a100_gru_forward_from_gates_cooperative_h256_htile4_compact_hoist_row4_hidden_shmem_gate_cache "
            "requires block_threads=256."
        )
    batch_size, seq_len, hidden_size = _validate_input_gates(
        input_gates,
        h0,
        weight_hh,
        bias_hh,
    )
    if hidden_size != 256:
        raise ValueError(
            "a100_gru_forward_from_gates_cooperative_h256_htile4_compact_hoist_row4_hidden_shmem_gate_cache "
            "requires hidden_size=256."
        )

    device_index = input_gates.device.index
    if device_index is None:
        device_index = torch.cuda.current_device()
    torch.cuda.set_device(device_index)
    capability = _device_capability(device_index)
    if capability != (8, 0):
        raise RuntimeError(
            "a100_gru_forward_from_gates_cooperative_h256_htile4_compact_hoist_row4_hidden_shmem_gate_cache "
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

    hidden_tile = hidden_size // 4
    k_tile = hidden_size // 4
    ctas_per_batch = 16
    compact_partials_per_batch = 12
    output = torch.empty(
        batch_size,
        seq_len,
        hidden_size,
        device=input_gates.device,
        dtype=input_gates.dtype,
    )
    gate_cache = torch.empty(
        batch_size,
        seq_len,
        4 * hidden_size,
        device=input_gates.device,
        dtype=input_gates.dtype,
    )
    partial_gates = torch.empty(
        batch_size * compact_partials_per_batch,
        3 * hidden_tile,
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
    function = (
        compiled.cooperative_h256_htile4_compact_hoist_row4_hidden_shmem_gate_cache_gates_function
    )
    shared_bytes = (3 * hidden_tile + k_tile) * ctypes.sizeof(ctypes.c_float)
    err, active_blocks_per_sm = cuda.cuOccupancyMaxActiveBlocksPerMultiprocessor(
        function,
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
            "h256 htile4 compact hoist row4 hidden-shmem gate-cache cooperative grid is too large "
            "for resident launch: "
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
        function,
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


def a100_gru_forward_from_gates_cooperative_h256_htile4_compact_hoist_row4_weight_shmem_gate_cache(
    input_gates: torch.Tensor,
    h0: torch.Tensor,
    weight_hh: torch.Tensor,
    bias_hh: torch.Tensor,
    block_threads: int = 256,
) -> tuple[torch.Tensor, torch.Tensor]:
    """运行把 row4 recurrent weight tile 常驻 shared memory 的 forward kernel。"""
    _require_cuda_python()
    _validate_block_threads(block_threads)
    if block_threads != 256:
        raise ValueError(
            "a100_gru_forward_from_gates_cooperative_h256_htile4_compact_hoist_row4_weight_shmem_gate_cache "
            "requires block_threads=256."
        )
    batch_size, seq_len, hidden_size = _validate_input_gates(
        input_gates,
        h0,
        weight_hh,
        bias_hh,
    )
    if hidden_size != 256:
        raise ValueError(
            "a100_gru_forward_from_gates_cooperative_h256_htile4_compact_hoist_row4_weight_shmem_gate_cache "
            "requires hidden_size=256."
        )

    device_index = input_gates.device.index
    if device_index is None:
        device_index = torch.cuda.current_device()
    torch.cuda.set_device(device_index)
    capability = _device_capability(device_index)
    if capability != (8, 0):
        raise RuntimeError(
            "a100_gru_forward_from_gates_cooperative_h256_htile4_compact_hoist_row4_weight_shmem_gate_cache "
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

    hidden_tile = hidden_size // 4
    k_tile = hidden_size // 4
    ctas_per_batch = 16
    compact_partials_per_batch = 12
    output = torch.empty(
        batch_size,
        seq_len,
        hidden_size,
        device=input_gates.device,
        dtype=input_gates.dtype,
    )
    gate_cache = torch.empty(
        batch_size,
        seq_len,
        4 * hidden_size,
        device=input_gates.device,
        dtype=input_gates.dtype,
    )
    partial_gates = torch.empty(
        batch_size * compact_partials_per_batch,
        3 * hidden_tile,
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
    function = (
        compiled.cooperative_h256_htile4_compact_hoist_row4_weight_shmem_gate_cache_gates_function
    )
    shared_floats = 3 * hidden_tile + 3 * hidden_tile * k_tile
    shared_bytes = shared_floats * ctypes.sizeof(ctypes.c_float)
    # 每个 CTA 缓存 3 个 gate 的 64x64 recurrent weight tile，需要打开 >48KB 动态 shared memory。
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
    sm_count = _device_attribute(
        device_index,
        cuda.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT,
    )
    grid_blocks = batch_size * ctas_per_batch
    max_cooperative_blocks = active_blocks_per_sm * sm_count
    if grid_blocks > max_cooperative_blocks:
        raise ValueError(
            "h256 htile4 compact hoist row4 weight-shmem gate-cache cooperative grid is too large "
            "for resident launch: "
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
        function,
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


def a100_gru_forward_from_gates_cooperative_h256_htile8_compact_shmem_gate_cache(
    input_gates: torch.Tensor,
    h0: torch.Tensor,
    weight_hh: torch.Tensor,
    bias_hh: torch.Tensor,
    block_threads: int = 128,
) -> tuple[torch.Tensor, torch.Tensor]:
    """运行 compact partial buffer 的 8 hidden tile h256 gate-cache kernel。"""
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
            "a100_gru_forward_from_gates_cooperative_h256_htile8_compact_shmem_gate_cache "
            "requires hidden_size=256."
        )

    device_index = input_gates.device.index
    if device_index is None:
        device_index = torch.cuda.current_device()
    torch.cuda.set_device(device_index)
    capability = _device_capability(device_index)
    if capability != (8, 0):
        raise RuntimeError(
            "a100_gru_forward_from_gates_cooperative_h256_htile8_compact_shmem_gate_cache "
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

    hidden_tile = hidden_size // 8
    ctas_per_batch = 32
    compact_partials_per_batch = 24
    output = torch.empty(
        batch_size,
        seq_len,
        hidden_size,
        device=input_gates.device,
        dtype=input_gates.dtype,
    )
    gate_cache = torch.empty(
        batch_size,
        seq_len,
        4 * hidden_size,
        device=input_gates.device,
        dtype=input_gates.dtype,
    )
    partial_gates = torch.empty(
        batch_size * compact_partials_per_batch,
        3 * hidden_tile,
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
    shared_bytes = 3 * hidden_tile * ctypes.sizeof(ctypes.c_float)
    err, active_blocks_per_sm = cuda.cuOccupancyMaxActiveBlocksPerMultiprocessor(
        compiled.cooperative_h256_htile8_compact_shmem_gate_cache_gates_function,
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
            "h256 htile8 compact gate-cache shmem cooperative grid is too large "
            "for resident launch: "
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
        compiled.cooperative_h256_htile8_compact_shmem_gate_cache_gates_function,
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
