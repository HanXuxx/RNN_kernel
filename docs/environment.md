# 环境说明

## 当前本地环境

本地虚拟环境安装在 `.venv`。

当前机器观测到的环境：

- Python：3.12.3
- GPU：4 x NVIDIA A100 80GB PCIe
- NVIDIA 驱动：550.163.01
- NVIDIA-SMI CUDA 版本：12.4
- PyTorch：2.6.0+cu124
- PyTorch CUDA 运行时：12.4
- PyTorch cuDNN：9.1.0
- cuda-python：12.4.0
- `.venv` 内 NVRTC/NVCC 辅助包：`nvidia-cuda-nvcc-cu12==12.4.131`
- 项目本地 Nsight CLI：默认使用 Nsight Compute 2024.2.1 和 Nsight Systems 2024.2.3
- Codex 沙箱外 CUDA 可用性：true

没有修改系统驱动。兼容性修复只发生在 `.venv` 内：之前的 CUDA 13 PyTorch
软件包已替换为与当前驱动匹配的 CUDA 12.4 软件包。通过 Codex 运行命令时，
GPU 访问可能需要沙箱外执行，因为沙箱可能隐藏 `/dev/nvidia*`；在普通 shell
中使用同一个 `.venv` 可以直接访问 GPU。

当前系统没有依赖全局 `nvcc`。CUDA C 原型通过 `.venv` 内的 `cuda-python` 调用
NVRTC，在运行时编译 cubin 并加载到 PyTorch CUDA context。`nvidia-cuda-nvcc-cu12`
提供头文件、libdevice 和 `ptxas`，但不提供完整 `nvcc` 命令。

Nsight CLI 没有通过系统包管理器安装。`.deb` 包下载并解压在 `tools/nsight/` 下，
该目录被 git 忽略。使用时显式加载：

```bash
source scripts/env_nsight.sh
ncu --version
nsys --version
```

当前默认版本选择 2024.x，是因为 2026.x Nsight Systems 在 driver 550/CUDA 12.4
环境中生成 report 时出现 CUDA metadata 导入错误。Nsight Compute 可以启动，但当前
机器未开放 GPU performance counter 权限，因此 `ncu` 会报 `ERR_NVGPUCTRPERM`；
Nsight Systems 的 CUDA timeline 和 kernel grid/block 信息可以正常采集。

## 目标 GPU 假设

优化目标是 NVIDIA A100 和 H200：

- A100：SM80
- H200：SM90
- 当前 PyTorch 软件包的架构列表包含 `sm_80` 和 `sm_90`。

后续构建 CUDA/C++ 或 Triton 扩展时，优先使用：

```bash
export TORCH_CUDA_ARCH_LIST="8.0;9.0"
```

使用支持 CUDA 12.4 或更新版本的数据中心驱动。当前 A100 机器的驱动
`550.163.01` 已满足这个要求。

## 可复现性约定

- `requirements.txt` 记录最小运行时依赖。
- `requirements-dev.txt` 增加基准测试分析和测试工具。
- `requirements-lock.txt` 记录当前已安装的精确包集合。
- 不提交 `.venv`、性能分析输出或原始基准测试 CSV，除非某个结果文件被明确
  提升为文档化结果。
