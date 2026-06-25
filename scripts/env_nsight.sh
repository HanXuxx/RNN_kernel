#!/usr/bin/env bash
# 加载项目本地 Nsight CLI，不修改系统 PATH 或系统包数据库。

# 默认使用与当前 CUDA 12.4 / driver 550 更匹配的 2024.x CLI。
export NSIGHT_COMPUTE_HOME="$PWD/tools/nsight/extract/compute-2024/opt/nvidia/nsight-compute/2024.2.1"
export NSIGHT_SYSTEMS_HOME="$PWD/tools/nsight/extract/systems-2024/opt/nvidia/nsight-systems/2024.2.3"
export PATH="$NSIGHT_COMPUTE_HOME:$NSIGHT_SYSTEMS_HOME/bin:$PATH"
