#!/usr/bin/env bash

export PYTHONPATH="${PWD}/src:${PYTHONPATH:-}"
export MPLCONFIGDIR="${PWD}/.cache/matplotlib"
export PIP_CACHE_DIR="${PWD}/.cache/pip"
export TORCH_CUDA_ARCH_LIST="8.0;9.0"
