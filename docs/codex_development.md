# Codex 开发规范

## 工作规则

1. 每次修改前先阅读相关基准测试、测试和文档。
2. 修改范围要限定在当前优化问题内。
3. 不要在没有记录原因的情况下改写基准测试参数或模型语义。
4. 在添加自定义 kernel 前，始终保留一个可运行的基线。
5. 每个性能结论都必须包含命令、硬件、PyTorch 版本、CUDA 运行时、驱动版本
   和测量指标。
6. 每个 kernel 或算法变更都需要和 `torch.nn.GRU` 做前向输出与梯度的正确性
   对比。
7. 优先使用小型、可复现的脚本，不依赖只存在于 notebook 中的实验。
8. 性能分析输出和 CSV 视为生成产物；把长期有效的发现总结到 `docs/`，不直接
   提交大量原始文件。
9. 代码使用英文，注释使用中文；不要把中文用于变量名、函数名、类名或文件名。
10. A100 相关优化必须先在 `src/rnn_kernel/a100` 实验区完成实现、正确性验证和
    性能验证；实验通过后，才允许把成熟的最优代码迁移到
    `src/rnn_kernel/a100/prod`。
11. 顶层 `/home/xuh/RNN_kernel/prod` 只允许由
    `src/rnn_kernel/a100/prod/scripts/export_a100_gru_h256.py` 一键导出生成，
    不手动复制或手动修改。

## 代码与注释语言

- 变量名、函数名、类名、模块名、文件名、命令行参数、配置键和公开 API 使用英文。
- 代码注释和面向项目维护者的解释性文字使用中文。
- 第三方库 API、PyTorch/CUDA/Triton 术语、报错文本和 profiler 原始字段保持原文。
- 只有在解释复杂逻辑、性能假设、数值容差或硬件相关限制时才添加注释。
- 不为了翻译而改动已有 API 名称；对外接口稳定性优先。

## 最低验证要求

仅修改 Python 代码时：

```bash
source .venv/bin/activate
source scripts/env.sh
python rnn_benchmark.py --device cpu --cell-types GRU --hidden-sizes 16 --batch-size 2 --seq-len 8 --num-layers 1 --dataset-batches 2 --warmup-steps 1 --timed-steps 1
```

修改 GPU 基准测试相关代码时，在 A100/H200 上运行：

```bash
source .venv/bin/activate
source scripts/env.sh
nvidia-smi
python rnn_benchmark.py --device cuda --cell-types GRU,LSTM --hidden-sizes 64,96,128,129,130,160,192,256 --output-csv results/rnn_baseline.csv
```

后续添加自定义 kernel 时：

```bash
pytest -q
python rnn_benchmark.py --device cuda --cell-types GRU --hidden-sizes 128,160,192,256 --output-csv results/gru_candidate.csv
```

## 目录职责

- 当基线和探索性基准测试超出单文件脚本后，放到 `benchmarks/`。
- 可复用 Python 模块放到 `src/rnn_kernel/`。
- 需要 CUDA/C++ 扩展源码时，放到 `src/rnn_kernel/csrc/`。
- 正确性测试和容差检查放到 `tests/`。
- 可复用性能分析命令放到 `scripts/`。
- 生成的性能结果放到 `results/` 或 `profiles/`。

## A100 prod 迁移流程

1. 在 `src/rnn_kernel/a100` 中实现实验 kernel、Python launcher 和局部 benchmark。
2. 使用 A100 跑正确性验证，至少覆盖 forward 输出；如果涉及训练路径，还必须覆盖
   backward 梯度。
3. 记录关键性能数字和命令。只把稳定结论写入 `docs/`，不把临时实验输出作为
   唯一依据。
4. 将通过验证的最优实现迁移到 `src/rnn_kernel/a100/prod`。prod 运行时不能依赖
   实验模块、NVRTC 或系统 `nvcc`。
5. 在 `src/rnn_kernel/a100/prod/a100_gru_h256` 中同步独立包源码、脚本和文档。
6. 运行 `src/rnn_kernel/a100/prod/scripts/export_a100_gru_h256.py` 生成顶层
   `prod/a100_gru_h256` 和 wheel。顶层 `prod` 目录视为构建产物。
7. 从顶层 `prod` 和 wheel 分别运行 functional test，确认导出产物和源目录一致。

## 性能评审检查表

接受一个优化前，需要检查：

1. 在同一张 GPU 上与相同 PyTorch 基线对比。
2. 报告多次运行的中位数或截尾均值，不使用单次幸运结果。
3. 尽量拆分前向、反向、优化器和数据生成时间。
4. 检查 hidden size 在 128 以下和以上紧邻位置的表现。
5. 在为某个架构做专门优化前，同时检查 A100 和 H200。
6. 验证 fp32 和目标混合精度模式下的数值容差。
