# A100 fp32-only 优化方法研究

## 目标

本轮只测试不降低数值精度的方法，明确排除 TF32、bf16 和 AMP。所有 benchmark 输出
均记录：

- `cuda_matmul_allow_tf32=False`
- `cudnn_allow_tf32=False`
- dtype 仍为 fp32

目标是确认是否存在无需自定义 kernel、无需牺牲精度即可规避 `hidden_size > 128`
性能断崖的方法。

## 复现命令

后端开关、torch.compile、短序列 cuDNN 对照和形状敏感性：

```bash
source .venv/bin/activate
scripts/run_fp32_backend_study.sh
```

完整 BPTT sequence chunking 扫描：

```bash
source .venv/bin/activate
scripts/run_fp32_chunk_study.sh
```

相关输出：

- `results/a100_fp32_default.csv`
- `results/a100_fp32_cudnn_benchmark_off.csv`
- `results/a100_fp32_deterministic.csv`
- `results/a100_fp32_compile.csv`
- `results/a100_fp32_cudnn_enabled_short_seq.csv`
- `results/a100_fp32_cudnn_disabled_short_seq.csv`
- `results/a100_fp32_shape_bs32_seq4000.csv`
- `results/a100_fp32_shape_bs8_seq16000.csv`
- `results/a100_fp32_chunk_*.csv`

## 后端开关结论

| 方法 | hidden 128 ms/step | hidden 130 ms/step | 结论 |
| --- | ---: | ---: | --- |
| 默认 fp32，cuDNN benchmark on | 88.525 | 357.700 | 断崖仍在 |
| cuDNN benchmark off | 88.225 | 362.411 | 基本无改善 |
| deterministic | 88.018 | 360.671 | 对 130 无改善；129 的异常高值下降 |
| torch.compile | 88.278 | 435.862 | 无改善，反而更慢 |

判断：

- `torch.backends.cudnn.benchmark` 不是主因。
- deterministic 不解决 `hidden_size > 128` 的根本问题，只让 129 的表现接近
  130。
- `torch.compile` 没有穿透 cuDNN RNN kernel，不能优化该路径。

## 禁用 cuDNN

短序列对照，`batch_size=16`、`seq_len=256`、`num_layers=4`：

| 方法 | hidden 128 ms/step | hidden 130 ms/step |
| --- | ---: | ---: |
| cuDNN enabled | 3.534 | 16.826 |
| cuDNN disabled | 184.990 | 182.331 |

判断：

- 禁用 cuDNN 后不再出现 128/130 断崖，但整体速度比 cuDNN 慢一个数量级以上。
- 对完整 `seq_len=8000` 训练不可行，不能作为优化路线。

## 完整 BPTT sequence chunking

实现方式：`--sequence-chunk-len` 会把输入按时间维切片，多次调用同一个 RNN，
并把 hidden state 传给下一段。没有 detach hidden，因此保留完整 BPTT。正确性测试
已比较完整序列和 chunked 序列的 loss、输入梯度、参数梯度。

| chunk_len | hidden 128 ms/step | hidden 130 ms/step | hidden 128 memory GB | hidden 130 memory GB |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 87.452 | 453.939 | 3.050 | 2.968 |
| 4000 | 88.630 | 462.265 | 2.433 | 2.373 |
| 2000 | 89.577 | 441.384 | 2.122 | 2.076 |
| 1000 | 91.339 | 446.873 | 1.968 | 1.926 |
| 500 | 94.692 | 360.035 | 1.898 | 1.915 |
| 250 | 101.938 | 460.909 | 1.874 | 1.918 |
| 125 | 115.055 | 494.464 | 1.874 | 1.904 |

判断：

- chunking 可以显著降低显存，hidden 130 从约 `2.97 GB` 降到约 `1.91 GB`。
- `chunk_len=500` 对 hidden 130 有一定速度改善，但仍约 `360 ms/step`，没有消除
  断崖。
- chunk 太短会增加 cuDNN 调用开销，使 hidden 128 和 hidden 130 都变慢。
- chunking 更适合作为显存优化或辅助策略，不是主要速度解法。

## batch/sequence 形状敏感性

保持 tokens/step 为 `128000`，但改变 batch 和 sequence：

| 形状 | hidden 128 ms/step | hidden 130 ms/step | hidden 128 tokens/s | hidden 130 tokens/s |
| --- | ---: | ---: | ---: | ---: |
| batch 16, seq 8000 | 88.525 | 357.700 | 1445917 | 357842 |
| batch 32, seq 4000 | 51.630 | 202.373 | 2479159 | 632496 |
| batch 8, seq 16000 | 160.531 | 742.909 | 797356 | 172296 |

判断：

- 更短 sequence、更大 batch 可以显著提高吞吐，但这改变了训练形状。
- 如果真实任务允许改变 BPTT 长度或 batch 组织，`batch=32, seq_len=4000`
  是当前最有效的无降精度规避方法。
- 如果必须保持完整 `seq_len=8000` 的单段训练语义，这不是等价替代。

## 综合判断

本轮没有找到“保持 fp32、保持原始单段 `seq_len=8000` 语义、只靠 PyTorch 后端开关”
即可消除断崖的方法。

可采用的低风险结论：

1. 后续所有 benchmark 默认关闭 TF32，避免精度策略混淆。
2. `deterministic`、`cudnn_benchmark off`、`torch.compile` 都不应作为主优化方向。
3. 禁用 cuDNN 不可行。
4. 如果任务允许较短 BPTT，优先尝试更大 batch、更短 sequence。
5. 如果任务必须保持原始 shape 和完整语义，下一步应进入自定义 kernel/backward
   路线。

