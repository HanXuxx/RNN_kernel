"""A100/SM80 specific GRU kernels."""

_AUTOGRAD_EXPORTS = {
    "A100GRUH256",
    "a100_gru_h256",
    "copy_from_torch_gru",
}

_FORWARD_EXPORTS = {
    "a100_gru_forward_from_gates",
    "a100_gru_forward_from_gates_cooperative",
    "a100_gru_forward_from_gates_cooperative_h256",
    "a100_gru_forward_from_gates_cooperative_h256_cached_shmem",
    "a100_gru_forward_from_gates_cooperative_h256_cta8_shmem_gate_cache",
    "a100_gru_forward_from_gates_cooperative_h256_parallel_update",
    "a100_gru_forward_from_gates_cooperative_h256_parallel_update_gate_cache",
    "a100_gru_forward_from_gates_cooperative_h256_qwarp_shmem",
    "a100_gru_forward_from_gates_cooperative_h256_shmem",
    "a100_gru_forward_from_gates_cooperative_h256_shmem_gate_cache",
    "a100_gru_forward_from_gates_fused",
    "a100_gru_forward_from_gates_fused_pingpong",
    "a100_gru_forward_from_gates_fused_specialized",
    "a100_gru_forward_from_gates_subwarp",
    "a100_gru_forward_layer",
    "a100_gru_forward_layer_precompute_input",
    "a100_gru_forward_layer_precompute_input_cooperative",
    "a100_gru_forward_layer_precompute_input_cooperative_h256",
    "a100_gru_forward_layer_precompute_input_cooperative_h256_cached_shmem",
    "a100_gru_forward_layer_precompute_input_cooperative_h256_parallel_update",
    "a100_gru_forward_layer_precompute_input_cooperative_h256_qwarp_shmem",
    "a100_gru_forward_layer_precompute_input_cooperative_h256_shmem",
    "a100_gru_forward_layer_precompute_input_fused",
    "a100_gru_forward_layer_precompute_input_fused_pingpong",
    "a100_gru_forward_layer_precompute_input_fused_specialized",
    "a100_gru_forward_layer_precompute_input_subwarp",
}

__all__ = sorted(_AUTOGRAD_EXPORTS | _FORWARD_EXPORTS)


def __getattr__(name: str):
    """按需加载实验模块，避免导入 prod 子包时触发 NVRTC/NVCC 依赖。"""
    if name in _AUTOGRAD_EXPORTS:
        from . import gru_autograd

        value = getattr(gru_autograd, name)
        globals()[name] = value
        return value
    if name in _FORWARD_EXPORTS:
        from . import gru_forward

        value = getattr(gru_forward, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
