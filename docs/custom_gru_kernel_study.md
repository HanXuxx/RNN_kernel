# 自定义 GRU kernel/backward 可行性研究

## 目标

本轮目标是启动自定义 kernel/backward 路线，并用可运行实验判断哪条方向值得继续。
本轮仍保持 fp32，且默认关闭 TF32：

- `cuda_matmul_allow_tf32=False`
- `cudnn_allow_tf32=False`

## 已实现候选

### `custom_gru`

纯 PyTorch 展开版 GRU：

- 使用 PyTorch GRU 相同的 gate 顺序和公式。
- 每层先预计算所有 time step 的 input projection。
- recurrent projection 和 gate pointwise 逐 time step 执行。
- backward 由 PyTorch autograd 生成。

它的价值是作为透明的正确性基准，不是最终性能方案。

### `custom_gru_triton`

在 `custom_gru` 基础上，把 GRU gate pointwise forward/backward 换成 Triton
自定义 autograd kernel：

- forward kernel 计算 reset/update/new gate 和 hidden update。
- backward kernel 返回 input gate、hidden gate 和 hidden direct path 的梯度。
- recurrent GEMM 仍然使用 PyTorch `F.linear`。
- 时间维仍然在 Python 循环中推进。

它的价值是量化“只融合 pointwise backward 是否足够”。

## 相关文件

- `src/rnn_kernel/custom_gru.py`
- `src/rnn_kernel/triton_gru_pointwise.py`
- `tests/test_custom_gru.py`
- `scripts/run_custom_gru_study.sh`
- `results/a100_custom_study_*.csv`

## 正确性

已通过测试：

```bash
source .venv/bin/activate
source scripts/env.sh
pytest -q tests/test_custom_gru.py tests/test_sequence_chunking.py
```

测试覆盖：

- CPU 上 `custom_gru` 与 `torch.nn.GRU` 的 forward、hidden、输入梯度和参数梯度。
- CUDA 上 `custom_gru_triton` 与 `custom_gru` torch backend 的严格对齐。
- CUDA 上 `custom_gru_triton` 与 cuDNN GRU 的 fp32 容差对齐。

说明：cuDNN GPU GRU 与逐步展开公式之间存在约 `1e-4` 到 `1e-3` 的 fp32 舍入差异；
纯 PyTorch 展开版也有同样差异，因此测试对 cuDNN 使用 GPU fp32 容差。

## 性能结果

### 短序列对比

配置：

- GPU：NVIDIA A100 80GB PCIe
- batch size：16
- sequence length：256
- num layers：4
- input dim：9
- dtype：fp32

| 实现 | hidden 128 ms/step | hidden 130 ms/step | hidden 128 tokens/s | hidden 130 tokens/s |
| --- | ---: | ---: | ---: | ---: |
| cuDNN GRU | 3.857 | 16.198 | 1061871 | 252878 |
| PyTorch native，禁用 cuDNN | 304.443 | 270.563 | 13454 | 15139 |
| `custom_gru` | 537.590 | 530.234 | 7619 | 7725 |
| `custom_gru_triton` | 477.657 | 441.713 | 8575 | 9273 |

观察：

- Triton pointwise backward 比纯 PyTorch 展开版快约 `11-17%`。
- 但它仍然比 cuDNN hidden 130 慢约 `27x`，比 cuDNN hidden 128 慢约 `124x`。
- 禁用 cuDNN 的 PyTorch native 路径也远慢于 cuDNN。

### 长序列单层估算

配置：

- batch size：16
- sequence length：8000
- num layers：1
- hidden size：130
- input dim：9
- dtype：fp32

| 实现 | ms/step | tokens/s | peak memory GB |
| --- | ---: | ---: | ---: |
| cuDNN GRU | 152.387 | 839968 | 1.598 |
| `custom_gru_triton` | 4985.410 | 25675 | 0.899 |

观察：

- `custom_gru_triton` 单层长序列比 cuDNN 慢约 `32.7x`。
- 显存更低，但速度差距过大，不能作为训练加速路线。

## 判断

本轮结论：

1. 自定义 pointwise forward/backward 可以正确工作。
2. 只融合 GRU gate pointwise 不足以解决 hidden size 断崖。
3. 当前慢点不是单个 pointwise 公式本身，而是每个 time step/layer 的 Python 循环、
   recurrent 小 GEMM 和大量 kernel launch。
4. 对 `seq_len=8000`、`num_layers=4` 这种目标 workload，必须把时间递归放进更少的
   GPU kernel 内部，接近 cuDNN persistent RNN 的执行方式。
5. 继续优化当前 `custom_gru_triton` 路线收益有限；它应保留为正确性和公式基准。

## 下一步

下一阶段应转向 CUDA/C++ persistent kernel 原型：

1. 先做单层、单向、batch-first、fp32、固定 hidden size 的 forward-only CUDA kernel。
2. 每个 CUDA block 或 cooperative group 在 kernel 内循环 time steps，避免
   `seq_len * num_layers` 级别的 Python 循环和 kernel launch。
3. 第一版先专门覆盖 `hidden_size=130/160`、`batch_size=16`、`input_dim=9`。
4. 正确性对齐 `custom_gru`，再对齐 cuDNN 容差。
5. forward-only 有收益后，再做 backward；否则不投入完整 backward。

当前不建议继续投入：

- 纯 PyTorch 展开版优化。
- 只替换 pointwise gate 的 Triton autograd kernel。
- 禁用 cuDNN 的 PyTorch native 路径。

## 后续 forward-only time-loop 原型

已继续实现并验证单层 Triton forward-only time-loop kernel，结果见
`docs/triton_forward_kernel_study.md`。

补充结论：把 time loop 放进单个 Triton kernel 是正确的，但由于 hidden projection
没有高效矩阵乘，目标形状上仍比 cuDNN forward 慢约 `5.6x`，因此不建议在该 Triton
原型上继续补 backward。

后续 CUDA C/NVRTC forward-only 原型见 `docs/cuda_forward_kernel_study.md`。该路线
证明 `.venv` 内 CUDA C 开发链可用，但朴素 per-batch matvec 仍慢于 cuDNN，因此同样
不建议在没有高效矩阵乘组织前直接补 backward。
