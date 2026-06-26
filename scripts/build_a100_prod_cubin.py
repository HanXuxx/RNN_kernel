#!/usr/bin/env python3
"""构建 A100 prod 运行时需要的预编译 cubin。"""

from __future__ import annotations

import argparse
import ctypes
from pathlib import Path

import nvidia.cuda_nvcc
import nvidia.cuda_nvrtc
import nvidia.cuda_runtime
from cuda import nvrtc


ROOT = Path(__file__).resolve().parents[1]
SOURCE_PATH = ROOT / "src" / "rnn_kernel" / "a100" / "prod" / "kernels" / "a100_gru_h256_kernels.cu"
OUTPUT_PATH = ROOT / "src" / "rnn_kernel" / "a100" / "prod" / "kernels" / "a100_gru_h256_sm80.cubin"


def _check_nvrtc(err: object, detail: str = "") -> None:
    if err == nvrtc.nvrtcResult.NVRTC_SUCCESS:
        return
    raise RuntimeError(f"NVRTC error: {err}. {detail}".strip())


def _include_dirs() -> list[Path]:
    return [
        Path(nvidia.cuda_nvcc.__file__).resolve().parent / "include",
        Path(nvidia.cuda_runtime.__file__).resolve().parent / "include",
    ]


def _preload_nvrtc() -> None:
    lib_dir = Path(nvidia.cuda_nvrtc.__file__).resolve().parent / "lib"
    ctypes.CDLL(str(lib_dir / "libnvrtc-builtins.so.12.4"), mode=ctypes.RTLD_GLOBAL)
    ctypes.CDLL(str(lib_dir / "libnvrtc.so.12"), mode=ctypes.RTLD_GLOBAL)


def build_cubin(source_path: Path, output_path: Path) -> None:
    _preload_nvrtc()
    source = source_path.read_text(encoding="utf-8")
    options = [
        b"--std=c++17",
        b"--gpu-architecture=sm_80",
    ]
    options.extend(f"--include-path={include_dir}".encode("utf-8") for include_dir in _include_dirs())

    err, program = nvrtc.nvrtcCreateProgram(
        source.encode("utf-8"),
        source_path.name.encode("utf-8"),
        0,
        [],
        [],
    )
    _check_nvrtc(err)

    err_compile, = nvrtc.nvrtcCompileProgram(program, len(options), options)
    err_log, log_size = nvrtc.nvrtcGetProgramLogSize(program)
    _check_nvrtc(err_log)
    log_buffer = b" " * log_size
    err_log, = nvrtc.nvrtcGetProgramLog(program, log_buffer)
    _check_nvrtc(err_log)
    compile_log = log_buffer.decode("utf-8", errors="replace").strip()
    if err_compile != nvrtc.nvrtcResult.NVRTC_SUCCESS:
        _check_nvrtc(err_compile, compile_log)

    err, image_size = nvrtc.nvrtcGetCUBINSize(program)
    _check_nvrtc(err)
    image = b" " * image_size
    err, = nvrtc.nvrtcGetCUBIN(program, image)
    _check_nvrtc(err)
    err, = nvrtc.nvrtcDestroyProgram(program)
    _check_nvrtc(err)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(image)
    if compile_log:
        print(compile_log)
    print(f"wrote {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=SOURCE_PATH)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    build_cubin(args.source, args.output)


if __name__ == "__main__":
    main()
