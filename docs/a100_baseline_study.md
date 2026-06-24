# A100 GRU hidden size 断崖基线研究

## 结论摘要

在当前 A100 环境下，PyTorch/cuDNN 的 GRU 训练在 `hidden_size=128` 到
`hidden_size=129` 之间出现明确性能断崖：

- GRU 总 step 时间从 `80.390 ms` 增加到 `470.238 ms`，变慢 `5.85x`。
- `hidden_size=130`、`144`、`160` 稳定在约 `383-386 ms/step`，说明断崖边界
  正好发生在超过 128 后。
- 分段计时显示主要瓶颈是 backward：GRU 128 的 backward 为 `46.661 ms`，
  GRU 129 的 backward 为 `320.891 ms`，增加约 `6.88x`。
- loss 和 optimizer 基本无关：optimizer 只有约 `0.12-0.16 ms/step`。
- LSTM 也存在 128 到 129 的断崖，说明这更像 cuDNN RNN kernel 选择问题，
  不是 GRU head、loss 或 AdamW 特有问题。
- profiler 显示 `hidden_size=128` 使用 `RNN_blockPersist_bp_GRU` backward kernel；
  `hidden_size=129/130` 的 backward 退化为大量小 GEMM 和 elementwise kernel。

因此，第一轮研究判断：当前优化对象应聚焦在 cuDNN RNN backward 路径切换，
而不是优化 loss、optimizer、数据加载或显存占用。

## 环境

- GPU：4 x NVIDIA A100 80GB PCIe
- 驱动：550.163.01
- NVIDIA-SMI CUDA 版本：12.4
- PyTorch：2.6.0+cu124
- PyTorch CUDA 运行时：12.4
- cuDNN：90100
- batch size：16
- sequence length：8000
- input dim：9
- num layers：4
- dtype：fp32
- deterministic：false

## 复现命令

完整 A100 基线与 GRU 分段计时：

```bash
source .venv/bin/activate
scripts/run_a100_baseline.sh
```

代表性 profiler：

```bash
source .venv/bin/activate
source scripts/env.sh
.venv/bin/python scripts/profile_rnn_case.py --cell-type GRU --hidden-size 128 --warmup-steps 2 --profile-steps 2 --row-limit 40
.venv/bin/python scripts/profile_rnn_case.py --cell-type GRU --hidden-size 129 --warmup-steps 2 --profile-steps 1 --row-limit 40
.venv/bin/python scripts/profile_rnn_case.py --cell-type GRU --hidden-size 130 --warmup-steps 2 --profile-steps 2 --row-limit 40
```

原始输出：

- `results/a100_baseline_total.csv`
- `results/a100_gru_breakdown.csv`
- `profiles/gru_hidden_128_profiler_table.txt`
- `profiles/gru_hidden_129_profiler_table.txt`
- `profiles/gru_hidden_130_profiler_table.txt`

## 总 step 基线

### GRU

| hidden_size | ms/step | tokens/s | peak memory GB | 相对 128 |
| ---: | ---: | ---: | ---: | ---: |
| 96 | 70.636 | 1812111.743 | 2.434 | 0.879x |
| 112 | 81.163 | 1577067.809 | 2.809 | 1.010x |
| 120 | 82.718 | 1547422.618 | 2.996 | 1.029x |
| 128 | 80.390 | 1592235.922 | 3.184 | 1.000x |
| 129 | 470.238 | 272202.601 | 3.079 | 5.849x |
| 130 | 382.852 | 334333.111 | 3.102 | 4.762x |
| 144 | 384.401 | 332985.515 | 3.417 | 4.782x |
| 160 | 386.311 | 331338.892 | 3.778 | 4.805x |
| 192 | 484.687 | 264087.839 | 4.499 | 6.029x |
| 256 | 564.043 | 226932.864 | 4.239 | 7.016x |

### LSTM 对照

| hidden_size | ms/step | tokens/s | peak memory GB | 相对 128 |
| ---: | ---: | ---: | ---: | ---: |
| 96 | 69.263 | 1848024.604 | 2.339 | 0.452x |
| 112 | 154.807 | 826837.590 | 2.699 | 1.010x |
| 120 | 158.968 | 805195.740 | 2.880 | 1.037x |
| 128 | 153.339 | 834750.369 | 3.060 | 1.000x |
| 129 | 514.874 | 248604.344 | 2.954 | 3.358x |
| 130 | 469.848 | 272428.504 | 2.976 | 3.064x |
| 144 | 471.527 | 271458.203 | 3.278 | 3.075x |
| 160 | 481.229 | 265985.539 | 3.623 | 3.138x |
| 192 | 549.787 | 232817.530 | 4.314 | 3.585x |
| 256 | 691.519 | 185099.718 | 3.758 | 4.510x |

## GRU 分段计时

分段计时会在每个 step 后同步，用于定位瓶颈，不直接替代总吞吐结果。

| hidden_size | ms/step | forward ms | backward ms | optimizer ms | forward 占比 | backward 占比 | 相对 128 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 128 | 81.918 | 34.725 | 46.661 | 0.121 | 42.503% | 57.113% | 1.000x |
| 129 | 437.338 | 115.954 | 320.891 | 0.153 | 26.524% | 73.402% | 5.339x |
| 130 | 386.486 | 116.026 | 269.917 | 0.144 | 30.036% | 69.875% | 4.718x |
| 160 | 383.416 | 119.209 | 263.638 | 0.153 | 31.108% | 68.796% | 4.680x |
| 256 | 616.869 | 320.212 | 296.073 | 0.164 | 51.923% | 48.008% | 7.530x |

分段计时结论：

- `hidden_size=128` 时，forward 和 backward 都在合理范围。
- `hidden_size=129/130/160` 时，forward 约变为 `3.3x`，backward 约变为
  `5.8-6.9x`。
- `hidden_size=256` 时 forward 也明显变重，说明更大 hidden size 下不仅是
  backward，forward kernel 计算路径也开始显著放大。
- loss 与 optimizer 占比接近 0，可以从优化目标中排除。

## Profiler 证据

### hidden_size=128

关键 profiler 行：

- `aten::_cudnn_rnn_backward`：self CUDA `286.138 ms`，2 次调用。
- `RNN_blockPersist_bp_GRU`：self CUDA `186.792 ms`，8 次调用。
- `aten::_cudnn_rnn`：self CUDA `146.239 ms`，2 次调用。
- `RNN_blockPersist_fp_GRU`：self CUDA `132.170 ms`，8 次调用。

解释：128 使用 cuDNN 的 persistent forward/backward GRU kernel，kernel 数量少，
路径紧凑。

### hidden_size=129

关键 profiler 行：

- `aten::_cudnn_rnn_backward`：self CUDA `677.919 ms`，1 次调用。
- `cutlass::Kernel2...`：self CUDA `459.756 ms`，`33000` 次调用。
- `GRU_elementWise_bp1`：self CUDA `100.550 ms`，`32000` 次调用。
- `aten::_cudnn_rnn`：self CUDA `112.585 ms`，1 次调用。
- `RNN_blockPersist_fp_GRU`：self CUDA `110.222 ms`，4 次调用。

解释：超过 128 后，forward 仍能看到 persistent forward kernel，但 backward
不再走 `RNN_blockPersist_bp_GRU`，而是退化为大量小 GEMM 和 elementwise kernel。
这解释了 backward 成为主要瓶颈。

### hidden_size=130

关键 profiler 行：

- `aten::_cudnn_rnn_backward`：self CUDA `1.388 s`，2 次调用。
- `cutlass::Kernel2...`：self CUDA `935.376 ms`，`66000` 次调用。
- `GRU_elementWise_bp1`：self CUDA `201.799 ms`，`64000` 次调用。
- `aten::_cudnn_rnn`：self CUDA `228.463 ms`，2 次调用。
- `RNN_blockPersist_fp_GRU`：self CUDA `223.688 ms`，8 次调用。

解释：130 与 129 的路径一致，说明断崖不是 129 的偶发现象，而是 hidden size
超过 128 后的稳定路径切换。

## 判断

当前证据支持以下判断：

1. 性能断崖边界在 `hidden_size > 128`。
2. 主要问题是 cuDNN GRU backward kernel 选择变化。
3. forward 也会变慢，但对 `129/130/160` 来说 backward 是主因。
4. optimizer、loss、数据生成和显存峰值不是主因。
5. LSTM 也有类似断崖，说明这是 cuDNN RNN 层面的通用阈值行为。

## 下一步优化路线

优先级从低风险到高成本：

1. 先做 PyTorch 参数层实验：TF32、bf16/AMP、deterministic、不同 PyTorch/cuDNN
   版本、不同 batch/sequence shape。目标是确认是否存在无需自定义 kernel 的
   规避方案。
2. 如果业务允许模型结构变化，评估是否能把大 hidden size 拆成多个不超过 128 的
   GRU block。但这会改变模型表达能力，不能默认等价。
3. 如果必须保持 `torch.nn.GRU` 的完整语义，优化重点应转向自定义 backward 或
   自定义完整 GRU kernel。
4. Triton 原型应先验证 gate 计算和 backward 融合机会；如果不能覆盖完整递归依赖，
   价值有限。
5. CUDA/C++ extension 是最终高成本路线，应以 `hidden_size > 128`、A100/H200、
   batch-first、单向 GRU、固定 dtype 为第一版支持面。

第一轮不建议直接优化 AdamW 或 loss，也不建议把大量时间投入数据加载，因为当前
benchmark 已经把这些因素排除。

## 第二轮 fp32-only 研究

不降低精度的后端开关和 shape 实验见 `docs/a100_fp32_method_study.md`。

补充结论：

- TF32、bf16、AMP 已排除，所有第二轮结果均为 fp32。
- deterministic、关闭 cuDNN benchmark、torch.compile 都不能消除断崖。
- 禁用 cuDNN 会极慢，不可行。
- 完整 BPTT 的 sequence chunking 可以降低显存，但不能根本解决速度问题。
- 如果任务允许较短 BPTT，`batch_size=32, seq_len=4000` 这类形状能显著提高吞吐；
  如果必须保持单段 `seq_len=8000`，下一步应转向自定义 backward/kernel。
