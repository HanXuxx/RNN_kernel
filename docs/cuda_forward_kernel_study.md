# CUDA C/NVRTC GRU forward-only 原型研究

## 目标

上一轮 Triton forward-only time-loop 原型证明：只把时间循环放进一个 GPU kernel
还不够，瓶颈会转移到手写 hidden projection。本轮继续验证 CUDA C 路线，但遵守
环境约束：

- 不修改系统 NVIDIA 驱动。
- 不依赖系统 `/usr/local/cuda` 或系统 `nvcc`。
- 只在 `.venv` 内安装 CUDA 编译相关 Python wheel。
- 保持 fp32，不使用 TF32、bf16、AMP 或 fast-math。

## 环境增量

已在 `.venv` 内增加：

- `cuda-python==12.4.0`：提供 CUDA Driver API 和 NVRTC Python binding。
- `nvidia-cuda-nvcc-cu12==12.4.131`：提供 NVRTC 需要的头文件、libdevice 和
  `ptxas`。该 wheel 没有提供完整 `nvcc` 可执行文件。

当前实现通过 NVRTC 在运行时把 CUDA C 源码编译为 cubin，并加载到 PyTorch 已创建
的 CUDA context 中执行。kernel 启动使用 PyTorch 当前 CUDA stream，因此不会破坏
PyTorch 的异步执行顺序。

## 已实现内容

新增 forward-only 原型：

- `src/rnn_kernel/cuda_gru_forward.py`
- `scripts/benchmark_cuda_forward.py`
- `tests/test_cuda_gru_forward.py`

支持范围：

- 单层 GRU
- 单向
- batch-first
- fp32
- `hidden_size <= 256`
- forward-only，不提供 backward

实现方式：

- 每个 CUDA block 处理一个 batch item。
- kernel 内部循环整个 sequence。
- hidden state 放在 shared memory。
- input projection 和 hidden projection 都在 CUDA kernel 内部逐元素累加。
- 没有使用 cuBLAS、CUTLASS、tensor core、warp-level MMA 或 tiled GEMM。

## 正确性验证

命令：

```bash
source .venv/bin/activate
source scripts/env.sh
.venv/bin/python -m pytest tests/test_cuda_gru_forward.py -q
```

结果：

```text
4 passed
```

测试覆盖：

- `hidden_size=16`
- `hidden_size=33`
- `hidden_size=130`
- 随机输入和随机初始 hidden
- 与 `torch.nn.GRU` forward 输出对齐
- 不支持的 `hidden_size=257` 明确报错

当前测试容差：

```text
atol=2e-4
rtol=1e-4
```

在本轮 benchmark 中，`max_abs_diff=0.000000`，说明该 forward 原型与 cuDNN forward
在测试形状上对齐到打印精度。

全量测试：

```bash
source .venv/bin/activate
source scripts/env.sh
.venv/bin/python -m pytest -q
```

结果：

```text
12 passed
```

## 性能结果

硬件和软件：

- GPU：NVIDIA A100 80GB PCIe
- NVIDIA 驱动：550.163.01
- PyTorch：2.6.0+cu124
- PyTorch CUDA runtime：12.4
- cuDNN：9.1.0
- NVRTC：12.4
- dtype：fp32
- TF32：关闭

### seq_len=256

命令：

```bash
source .venv/bin/activate
source scripts/env.sh
.venv/bin/python scripts/benchmark_cuda_forward.py \
  --hidden-sizes 128,130,160 \
  --batch-size 16 \
  --seq-len 256 \
  --input-dim 9 \
  --warmup-steps 3 \
  --timed-steps 10 \
  --output-csv results/cuda_forward_seq256.csv
```

结果：

| hidden_size | cuDNN forward ms | CUDA C/NVRTC forward ms | CUDA C / cuDNN | max_abs_diff |
| ---: | ---: | ---: | ---: | ---: |
| 128 | 0.329 | 9.712 | 29.52x slower | 0.000000 |
| 130 | 0.846 | 6.285 | 7.43x slower | 0.000000 |
| 160 | 0.930 | 15.222 | 16.37x slower | 0.000000 |

### seq_len=8000

命令：

```bash
source .venv/bin/activate
source scripts/env.sh
.venv/bin/python scripts/benchmark_cuda_forward.py \
  --hidden-sizes 128,130,160 \
  --batch-size 16 \
  --seq-len 8000 \
  --input-dim 9 \
  --warmup-steps 1 \
  --timed-steps 3 \
  --output-csv results/cuda_forward_seq8000.csv
```

结果：

| hidden_size | cuDNN forward ms | CUDA C/NVRTC forward ms | CUDA C / cuDNN | max_abs_diff |
| ---: | ---: | ---: | ---: | ---: |
| 128 | 8.850 | 304.511 | 34.41x slower | 0.000000 |
| 130 | 28.808 | 197.231 | 6.85x slower | 0.000000 |
| 160 | 29.488 | 476.169 | 16.15x slower | 0.000000 |

## 判断

本轮回答了两个问题。

第一，环境链路已经打通。即使没有系统 `nvcc`，也可以只依赖 `.venv` 内的
`cuda-python` 和 NVIDIA CUDA wheel，在 A100 上运行 CUDA C 原型。这满足“不影响整台
机器默认驱动”的要求。

第二，朴素 CUDA C time-loop kernel 仍然不是性能路线。虽然它比之前的 Triton
time-loop 在部分形状上更快，但由于 hidden projection 仍是每个 batch block 内的
标量矩阵向量累加，没有利用 tensor core 或高效 GEMM 组织，因此目标形状
`seq_len=8000, hidden_size=130` 仍比 cuDNN forward 慢约 `6.85x`。

因此，不建议在当前朴素 CUDA C/NVRTC forward 原型上继续实现 backward。backward
会引入更多矩阵乘、跨时间步依赖和中间状态读写；如果 forward 已经明显慢于 cuDNN，
补 backward 不能解决训练速度断崖。

## 下一步

继续自定义 kernel 路线时，应从“手写 per-batch matvec”切换到“高效矩阵乘组织”：

1. 保留当前 NVRTC 原型，作为 CUDA C 开发链和公式正确性的最小样例。
2. 下一版 forward 应优先重构为 tiled GEMM / CUTLASS / warp-level MMA 思路，而不是
   给当前 matvec kernel 补 backward。
3. 只有 forward 在 `hidden_size=130/160`、`seq_len=8000` 上接近或超过 cuDNN 后，
   才投入自定义 backward。
4. 如果必须继续闭环 backward，应先做 CPU/PyTorch 公式版 backward 的保存张量清单，
   再映射到 CUDA kernel；不要直接从当前慢 forward 派生完整训练实现。
