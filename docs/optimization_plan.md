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
2. 当前 Triton pointwise forward/backward 研究见
   `docs/custom_gru_kernel_study.md`。该路线正确但远慢于 cuDNN，不能作为主优化
   方向。
3. 单层 forward-only time-loop 原型见 `docs/triton_forward_kernel_study.md`。该
   路线正确，但由于缺少高效矩阵乘，仍比 cuDNN forward 慢，不建议直接补 backward。
4. 在投入 CUDA extension 前，主要用 Triton 验证数据布局、融合机会，以及
   H200/A100 的敏感性。

退出标准：证明在计入反向后，自定义融合对目标 hidden size 有机会超过
PyTorch。

## 阶段 5：CUDA/C++ 扩展

1. 当前机器没有检测到系统 `nvcc`，但已打通 `.venv` 内 CUDA C/NVRTC 原型，见
   `docs/cuda_forward_kernel_study.md`。
2. 初始 NVRTC forward-only 原型正确，但由于 hidden projection 是朴素 per-batch
   matvec，目标形状上仍慢于 cuDNN，不建议直接补 backward。
3. 下一版 CUDA 路线必须引入高效矩阵乘组织：CUTLASS、tiled GEMM、warp-level MMA
   或 PyTorch extension 中可维护的等价实现。
4. A100/SM80 专用 forward-only 原型见 `docs/a100_forward_kernel_study.md`。该路线
   已用 Nsight Systems 验证调度形状：自定义 recurrent kernel 是主要瓶颈，单纯匹配
   cuDNN 的 block size 不能追上 cuDNN。当前主目标已收敛到 `hidden_size=256`。
5. 当前机器没有开放 GPU performance counter 权限，Nsight Compute 会报
   `ERR_NVGPUCTRPERM`。如果要继续做 occupancy、warp stall、memory throughput 层面的
   定量优化，需要管理员开放 counter 权限，或在允许 profiling 的机器上运行。
6. cooperative groups/multi-CTA recurrent projection 已完成 h256 多版验证。
   h256 forward-only 已经快于 cuDNN：通用 cooperative4 为 `81.918 ms`，h256 专用
   shmem cooperative 为 `73.540 ms`，cuDNN 为 `111.976 ms`。
7. 下一步聚焦 h256，不再把 h130/160/192 作为主 benchmark。h256 backward 正确性
   原型和第一轮性能优化已完成，见 `docs/a100_h256_backward_study.md`。
8. 初始阶段专门覆盖真实基准测试约束：单向 GRU、batch-first 输入、固定 dtype
   模式，以及 hidden size 大于 128。
9. h256 forward 已经达到继续投入 backward 的门槛；当前 pointwise CUDA backward、
   跨 time step batched GEMM、fused backward step kernel、cooperative split
   backward step、persistent backward sequence kernel、persistent-state kernel、
   split16 persistent-state kernel、split16 persistent-state gate-cache kernel 和
   split16 persistent-state gate-cache tiled kernel
   已完成。seq8000 timed_steps=10 顺序复测下最佳自定义训练结果为
   `137.471 ms/step`，快于同配置 cuDNN 复测的 `251.414 ms/step`，约 `1.83x`。
   `recompute_hidden_gates`、tiled recurrent、外部分离 split recurrent partial-buffer、
   split2 gate-cache、persistent-state-local、split32、split16 global-gates、
   split32 gate-cache tiled、grad-coeff-cache tiled、gate-cache parallel-update forward、
   gate-cache CTA8 forward 和 gate-cache CTA6 forward 路径都已验证为负向或中性实验；
   split16 persistent-state gate-cache tiled 是当前正向候选，下一步应优化 CTA4 shmem
   forward gate-cache kernel、backward split16 gate-cache tiled kernel、剩余 grid sync、
   partial 规约和 reserve-space 显存/带宽成本。
10. 与 cuDNN GRU 对比速度、显存和数值容差。
11. 增加类似 CI 的本地检查：构建、单元测试、CPU 冒烟基准测试和 GPU 基准测试
   命令模板。

退出标准：候选 kernel 在 A100 和 H200 的目标工作负载上都正确，并且快于
PyTorch。

## 阶段 6：生产化加固

1. 定义支持的形状、dtype、GPU 和回退路径。
2. 增加运行时检查，在不支持的输入范围回退到 `torch.nn.GRU`。
3. 增加针对 hidden-size 性能断崖的回归基准测试。
4. 文档化维护成本和任何数值差异。

退出标准：形成可维护的实现路径，而不是一次性的基准测试胜利。
