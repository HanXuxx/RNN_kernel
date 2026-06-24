# RNN Kernel 优化

这个仓库用于优化 PyTorch 环境下的 GRU 训练性能，目标 GPU 是 NVIDIA A100/H200。
当前基线测试脚本是 `rnn_benchmark.py`，用于测量不同 hidden size 下的训练速度。

## 环境

创建并激活本地虚拟环境：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements-dev.txt
source scripts/env.sh
```

当前准备好的环境使用：

- Python 3.12
- PyTorch 2.6.0，CUDA 12.4 运行时包
- Triton 3.2.0
- pytest、pandas、matplotlib、ninja

当前机器是 A100 GPU，NVIDIA 驱动版本为 `550.163.01`，支持 CUDA 12.4。
虚拟环境固定为 `torch 2.6.0+cu124`，因此可以在不修改系统驱动的前提下运行
GPU 基准测试。当前选择的 PyTorch 软件包包含 `sm_80` 和 `sm_90` 支持，可以
覆盖现在的 A100 和后续的 H200。

## 快速检查

CPU 冒烟测试：

```bash
python rnn_benchmark.py \
  --device cpu \
  --cell-types GRU \
  --hidden-sizes 16 \
  --batch-size 2 \
  --seq-len 8 \
  --num-layers 1 \
  --dataset-batches 2 \
  --warmup-steps 1 \
  --timed-steps 1
```

A100/H200 上的 GPU 基线测试：

```bash
source .venv/bin/activate
source scripts/env.sh
python rnn_benchmark.py \
  --device cuda \
  --cell-types GRU,LSTM \
  --hidden-sizes 64,96,128,129,130,160,192,256 \
  --output-csv results/rnn_baseline.csv
```

## 目录结构

- `rnn_benchmark.py`：当前最小化的 GRU/LSTM 训练基准测试。
- `src/rnn_kernel/`：后续 Python 包和扩展入口。
- `benchmarks/`：超出当前单文件脚本后的基准测试驱动。
- `tests/`：正确性测试和性能回归测试。
- `profiles/`：本地性能分析输出，除 `.gitkeep` 外被 git 忽略。
- `results/`：本地基准测试 CSV，除 `.gitkeep` 外被 git 忽略。
- `docs/`：环境说明、Codex 工作规范和优化计划。
- `scripts/`：可复用 shell/Python 辅助脚本。

修改代码前先阅读 `docs/codex_development.md`，选择下一步优化任务前先阅读
`docs/optimization_plan.md`。

当前 A100 第一轮闭环研究结论见 `docs/a100_baseline_study.md`。
不降精度的 fp32-only 方法研究见 `docs/a100_fp32_method_study.md`。
