# Triton GRU forward-only time-loop 原型研究

## 目标

上一轮 `custom_gru_triton` 只融合了每个 time step 的 gate pointwise，仍然有
`seq_len * num_layers` 级别的 Python 循环和 kernel launch。本轮尝试下一步：
把单层 GRU 的整个时间循环放进一个 Triton GPU kernel 内部，验证 persistent
kernel 思路中“减少 time-step launch”这一点是否足够。

本轮仍保持 fp32，并显式关闭 TF32：

- `cuda_matmul_allow_tf32=False`
- `cudnn_allow_tf32=False`

## 已实现内容

新增 forward-only 原型：

- `src/rnn_kernel/triton_gru_forward.py`
- `scripts/benchmark_triton_forward.py`
- `tests/test_triton_gru_forward.py`

支持范围：

- 单层 GRU
- 单向
- batch-first
- fp32
- `hidden_size <= 256`
- forward-only，不提供 backward

实现方式：

- 每个 Triton program 处理一个 batch item。
- kernel 内部循环整个 sequence。
- input projection 和 hidden projection 都在 Triton kernel 内部计算。
- hidden projection 使用逐元素矩阵向量累加，没有使用 cuBLAS/CUTLASS/tensor core。

## 正确性验证

已增加测试：

```bash
source .venv/bin/activate
source scripts/env.sh
pytest -q tests/test_triton_gru_forward.py
```

测试覆盖：

- `hidden_size=16`
- `hidden_size=33`
- `hidden_size=130`
- 随机输入和随机初始 hidden
- 与 `torch.nn.GRU` forward 输出对齐

当前测试容差：

```text
atol=2e-4
rtol=1e-4
```

在 A100 实测 benchmark 中，`max_abs_diff=0.000000`，说明当前 forward 原型与
cuDNN forward 在这些测试形状上完全对齐到打印精度。

## 性能结果

### seq_len=256

命令：

```bash
source .venv/bin/activate
source scripts/env.sh
.venv/bin/python scripts/benchmark_triton_forward.py \
  --hidden-sizes 128,130,160 \
  --batch-size 16 \
  --seq-len 256 \
  --input-dim 9 \
  --warmup-steps 3 \
  --timed-steps 10 \
  --output-csv results/a100_triton_forward_seq256.csv
```

结果：

| hidden_size | cuDNN forward ms | Triton forward ms | Triton / cuDNN | max_abs_diff |
| ---: | ---: | ---: | ---: | ---: |
| 128 | 0.330 | 1.603 | 4.86x slower | 0.000000 |
| 130 | 0.865 | 5.222 | 6.04x slower | 0.000000 |
| 160 | 0.941 | 2.741 | 2.91x slower | 0.000000 |

### seq_len=8000

命令：

```bash
source .venv/bin/activate
source scripts/env.sh
.venv/bin/python scripts/benchmark_triton_forward.py \
  --hidden-sizes 128,130 \
  --batch-size 16 \
  --seq-len 8000 \
  --input-dim 9 \
  --warmup-steps 1 \
  --timed-steps 3 \
  --output-csv results/a100_triton_forward_seq8000.csv
```

结果：

| hidden_size | cuDNN forward ms | Triton forward ms | Triton / cuDNN | max_abs_diff |
| ---: | ---: | ---: | ---: | ---: |
| 128 | 8.769 | 50.604 | 5.77x slower | 0.000000 |
| 130 | 29.043 | 163.862 | 5.64x slower | 0.000000 |

## 判断

本轮 forward-only 原型回答了一个关键问题：

> 只把 time loop 放进单个 GPU kernel，是否足以接近或超过 cuDNN？

答案是否定的。

原因：

1. 该 Triton kernel 减少了 time-step launch，但 hidden projection 变成了手写
   矩阵向量累加。
2. cuDNN 在 RNN forward 中使用高度优化的 persistent kernel 和 GEMM 路径。
3. 当前 Triton 原型没有使用 tensor core，也没有做 shared memory tiling 或 warp
   级矩阵乘优化。
4. 对目标形状 `seq_len=8000, batch_size=16, hidden_size=130`，Triton forward
   仍比 cuDNN forward 慢约 `5.6x`。

因此，当前 Triton time-loop 原型正确，但不是可用的性能路线。

## 对下一步的影响

可以保留：

- `triton_gru_forward.py` 作为 forward 公式和 time-loop-in-kernel 的正确性原型。
- `benchmark_triton_forward.py` 作为后续 forward-only 原型的基准脚本。

不建议继续：

- 在当前 Triton kernel 上直接补 backward。
- 继续优化没有高效矩阵乘的 per-batch matvec 实现。

建议下一步：

1. 如果要继续自定义 kernel，必须引入高效矩阵乘策略：CUTLASS/CUDA C++、
   warp-level MMA 或专门的 tiled GEMM。
2. 优先实现 forward-only CUDA/C++ extension；只有 forward 接近 cuDNN 后，再投入
   backward。
3. 如果没有 `nvcc`，可以先使用 `.venv` 内 `cuda-python` + NVRTC 验证 CUDA C
   原型；本路线的后续结果见 `docs/cuda_forward_kernel_study.md`。
4. 对 backward 的优化重点应是避免 `hidden_size > 128` 后 cuDNN 退化成大量小 GEMM
   和 elementwise kernel。
