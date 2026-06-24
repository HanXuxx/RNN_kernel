# GRU 训练优化计划

## 目标

降低 PyTorch GRU 在 `hidden_size` 超过 128 后出现的训练速度下降。优化重点是
A100/H200 GPU，以及 `rnn_benchmark.py` 所代表的基准测试形状。

## 阶段 0：建立基线

当前 A100 第一轮结果已记录在 `docs/a100_baseline_study.md`。

1. 在 A100 和 H200 上运行当前基准测试，重点覆盖疑似性能断崖附近的 hidden
   size：`96,112,120,128,129,130,144,160,192,256`。
2. 记录 GPU 型号、驱动、PyTorch 版本、CUDA 运行时、cuDNN 版本、批大小、
   序列长度、层数、dtype 和 deterministic 设置。
3. 分别收集 GRU 和 LSTM 结果。LSTM 是对照组，因为当前断崖可能与 GRU 的
   kernel 选择有关。
4. 对代表性案例增加性能分析：`hidden_size=128`、`130`、`160` 和 `256`。
5. 确认性能断崖主要来自前向、反向、优化器还是内存行为。

退出标准：形成一张文档化的基线表和性能分析摘要，并且至少在一类目标 GPU 上
复现该性能下降。

## 阶段 1：低风险 PyTorch 层优化

1. 确认输入、权重和 hidden state 保持 contiguous，并且仍走 cuDNN 路径。
2. deterministic、cuDNN benchmark、禁用 cuDNN、torch.compile、完整 BPTT
   sequence chunking 和 batch/sequence shape 变化已在
   `docs/a100_fp32_method_study.md` 中完成第一轮验证；这些方法没有在保持原始
   单段 `seq_len=8000` 语义且不降精度的前提下消除断崖。
3. 只有在隔离 RNN kernel 时间后，再测量融合优化器替代方案。
4. 测试 hidden size 对齐策略，例如填充到更利于 tensor core 的倍数，再在 head
   输出处切片；前提是数值语义可接受。
5. 只把 `torch.compile` 当成测量项，不预设它能解决问题，因为 cuDNN RNN kernel
   通常对编译器融合不透明。

退出标准：明确现有 PyTorch 参数和运行路径能否追回 5 倍回退中的显著部分。

## 阶段 2：基准测试和测试框架

1. 在不改变模型语义的前提下，把可复用基准测试逻辑移动到 `benchmarks/` 或
   `src/rnn_kernel/`。
2. 增加测试，用小形状对比候选实现和 `torch.nn.GRU` 的前向与反向。
3. 增加计时工具，分别报告前向、反向、优化器和总 step 时间。
4. 增加 PyTorch 性能分析器、Nsight Systems、Nsight Compute 的性能分析脚本。

退出标准：有一个可重复运行的框架，可以快速拒绝不正确或更慢的候选方案。

## 阶段 3：算法和布局实验

1. 检查 PyTorch/cuDNN 在 `hidden_size=128` 附近的 kernel 选择，确认是否只有
   较小 hidden size 使用 persistent RNN kernel。
2. 如果真实训练任务允许，测试序列分块或 truncated BPTT。demo 中序列长度为
   8000，时间步递归成本会占主导。
3. 探索权重打包和 gate 布局调整，减少每个时间步的开销。
4. 如果反向的内存流量是主要瓶颈，评估 recomputation/checkpointing 的取舍。

退出标准：判断最佳路线是模型层 reshape、自定义前向/反向 kernel，还是接受
cuDNN 当前行为。

## 阶段 4：Triton 原型

1. 先用 Triton 原型化独立的 GRU gate pointwise 操作。
2. 只在小型受控形状上先和 PyTorch 对比基准测试。
3. 在前向和梯度测试稳定前，不尝试完整替换训练实现。
4. 在投入 CUDA extension 前，主要用 Triton 验证数据布局、融合机会，以及
   H200/A100 的敏感性。

退出标准：证明在计入反向后，自定义融合对目标 hidden size 有机会超过
PyTorch。

## 阶段 5：CUDA/C++ 扩展

1. 在 `src/rnn_kernel/csrc/` 下添加 PyTorch extension，构建目标包含 SM80 和
   SM90。
2. 初始阶段专门覆盖真实基准测试约束：单向 GRU、batch-first 输入、固定 dtype
   模式，以及 hidden size 大于 128。
3. 先实现前向，再实现反向，最后整合优化器计时。
4. 与 cuDNN GRU 对比速度、显存和数值容差。
5. 增加类似 CI 的本地检查：构建、单元测试、CPU 冒烟基准测试和 GPU 基准测试
   命令模板。

退出标准：候选 kernel 在 A100 和 H200 的目标工作负载上都正确，并且快于
PyTorch。

## 阶段 6：生产化加固

1. 定义支持的形状、dtype、GPU 和回退路径。
2. 增加运行时检查，在不支持的输入范围回退到 `torch.nn.GRU`。
3. 增加针对 hidden-size 性能断崖的回归基准测试。
4. 文档化维护成本和任何数值差异。

退出标准：形成可维护的实现路径，而不是一次性的基准测试胜利。
