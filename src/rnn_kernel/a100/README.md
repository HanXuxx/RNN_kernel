# A100 实验区开发规范

该目录是 A100/SM80 GRU h256 优化的实验源头。所有新 kernel、Python 封装、
benchmark 接口和正确性验证应先在这里完成，不直接在 `prod/` 中开发。

## 开发流程

1. 在 `src/rnn_kernel/a100` 中实现实验代码，并通过实验测试验证正确性。
2. 用 `rnn_benchmark.py` 或专用脚本确认性能收益，记录 batch、seq、层数和实现名。
3. 只有在实验区结果稳定后，才把成熟实现迁移到 `src/rnn_kernel/a100/prod`。
4. 通过 `src/rnn_kernel/a100/prod/scripts/export_a100_gru_h256.py` 一键导出到顶层 `prod/`。
5. 顶层 `prod/a100_gru_h256` 是导出结果，不作为手写开发源头。

## 当前多层策略

`A100GRUH256` 支持 `num_layers=1..4`，对外保持一个统一接口，对内使用
`_forward_train_1/2/3/4` 和 `_forward_inference_1/2/3/4` 四组固定分支。
这种方式先避免动态循环进入产品接口，后续如果新增 fused 多层 kernel，也应先在这里替换
对应实验分支。

当前输入维度支持范围固定为 `input_size=1..16`。第一层输入投影仍由 PyTorch
`F.linear` 完成，A100 recurrent kernel 消费预计算后的 `[batch, seq, 3 * hidden]`
gate cache，因此 `input_size<=16` 不需要修改 recurrent CUDA kernel。

当前训练最优组合仍是：

- forward：`htile4 compact hoist row4`
- backward：`split6 persistent-state gate-cache tiled weight-shmem split0-keep unroll8`
- block threads：`256`
- sequence：full sequence，不做 chunk

chunking 可以降低显存峰值，但当前 A100 测试中会降低多层训练速度。

当前推理默认路径已切到 `row4 no-cache K1`：

- input projection 仍由 `F.linear` 完成。
- recurrent kernel 每个 batch 使用 `4` 个 CTA，而不是旧 row4 no-cache 的 `16` 个 CTA。
- K1 在单个 CTA 内完成完整 hidden dot，去掉跨 K CTA 的 global partial 读写。
- dot 完成后必须做一次 `grid.sync()`，防止其他 hidden tile 仍在读取旧 hidden 时提前写回。
- batch 超过 cooperative resident 上限时，Python wrapper 会按 batch 维自动分块兜底。

当前训练 forward 默认路径已切到 `row4 gate-cache K1`：

- backward 仍复用当前最优 `split6 persistent-state gate-cache tiled weight-shmem split0-keep unroll8`。
- gate cache layout 保持 `[batch, seq, 4 * hidden]`，因此 backward 公式和精度不变。
- batch=32 训练目前不是完整支持范围；forward 可分块，但当前 split6 系列 backward 会触发
  cooperative launch 失败。

## 多层复测记录

目标配置：A100 80GB、`input_size=16`、`hidden_size=256`、`batch=16`、`seq=8000`、
`num_layers=2..4`、fp32、TF32 关闭。

已复测的多层变体：

- `split5/6/12/24`：`split6` 仍然最快。
- `split6_unroll8`：只把 split6 recurrent dot 的 unroll 因子从 4 提到 8，sequence
  kernel 约快 3%，完整训练 total 约快 0.5%-0.9%，已作为当前默认 backward。
- `split8 weight-shmem split0-keep`：新增实验路径，batch=16 直接 resident grid 超限，
  分块后 sequence kernel 约 53.859 ms，明显慢于 split6 的 28.724 ms，不作为默认路径。
- `prev_cache`：显存更高，但 2/3/4 层都比当前 `split6_unroll8 + pack` 慢。
- `parallel_update`：2/3/4 层都慢于当前组合。
- `ldg`：2/3/4 层都明显慢于当前组合。
- `forward_output_only()`：只适合调用方完全不使用 `h_n` 的实验路径；当前测试中
  2 层只有极小收益，3/4 层无稳定收益，因此不作为默认路径，也不迁移到 prod。
- `row4 no-cache K2`：推理可用，快于旧 K4，但慢于 K1。
- `row4 no-cache K1`：推理默认路径。加入安全 `grid.sync()` 后，当前 batch=16、
  seq=8000 下约 1.07x 快于旧 K4。
- `row4 gate-cache K1`：训练 forward 默认路径，在当前 split6_unroll8 backward 组合下约 1.05x
  快于旧 K4 gate-cache forward。
- `row4 no-cache LDG`：小 batch 正确，但 batch=16 时 occupancy 下降导致 cooperative
  resident 上限不足，不作为主路径。
- `row4 K1 hidden 双缓冲`：尝试用 ping-pong hidden buffer 去掉 dot 后、update 前的
  `grid.sync()`，functional test 通过，但 batch=16 下更慢，因此已恢复为单 buffer
  双同步安全版本。
- `htile8/K1`：新增独立实验路径，把每个 batch 的 CTA 从 4 增加到 8；functional test
  通过，但同步和调度成本上升，batch=16 明显慢于当前 row4/K1，不作为默认路径。

当前推理性能：

| 层数 | 旧 row4 no-cache K4 | 当前默认 K1 | 加速 |
| --- | ---: | ---: | ---: |
| 1 | 56.667 ms | 52.491 ms | 1.08x |
| 2 | 114.765 ms | 107.934 ms | 1.06x |
| 3 | 173.111 ms | 161.093 ms | 1.07x |
| 4 | 231.230 ms | 216.714 ms | 1.07x |

batch=32、seq=8000 推理时 K1 通过 batch 分块兜底，正确性通过；当前实测
1/2/3/4 层约为 88.807/182.826/274.357/365.366 ms。

当前训练性能：

| 层数 | 旧 row4 gate-cache K4 | 当前默认 K1 + split6_unroll8 | K1/K4 加速 |
| --- | ---: | ---: | ---: |
| 1 | 93.813 ms | 88.150 ms | 1.06x |
| 2 | 191.467 ms | 181.994 ms | 1.05x |
| 3 | 289.752 ms | 274.456 ms | 1.06x |
| 4 | 388.061 ms | 369.306 ms | 1.05x |

本轮 backward 侧 batch=16 复测中，sequence kernel 拆解显示主要瓶颈在 recurrent
backward kernel，而不是后处理 GEMM。单层 `input_size=16` 下：

| 阶段 | 耗时 |
| --- | ---: |
| pack hidden-prev | 0.205 ms |
| split6 sequence kernel | 28.792 ms |
| grad_weight_hh GEMM | 2.693 ms |
| grad_x GEMM | 0.439 ms |
| grad_weight_ih GEMM | 0.377 ms |

split6_unroll8 对比旧 split6 的完整训练复测：

| 层数 | 旧 split6 total | split6_unroll8 total | 加速 |
| --- | ---: | ---: | ---: |
| 1 | 88.810 ms | 88.150 ms | 1.007x |
| 2 | 183.712 ms | 181.994 ms | 1.009x |
| 3 | 275.690 ms | 274.456 ms | 1.004x |
| 4 | 371.198 ms | 369.306 ms | 1.005x |

本轮 batch=16 复测中，恢复后的当前默认 row4/K1 仍是最优。A100 80GB、`input_size=16`、
`hidden_size=256`、`batch=16`、`seq=8000`、fp32、TF32 关闭下：

| 层数 | row4/K1 推理 | htile8/K1 推理 | row4/K1 训练 total | htile8/K1 训练 total |
| --- | ---: | ---: | ---: | ---: |
| 1 | 52.381 ms | 69.342 ms | 89.310 ms | 108.139 ms |
| 2 | 106.981 ms | 139.564 ms | 183.623 ms | 220.829 ms |
| 3 | 161.433 ms | 210.959 ms | 277.076 ms | 333.662 ms |
| 4 | 216.343 ms | 283.635 ms | 370.847 ms | 446.474 ms |

结论：batch=16 虽然 row4/K1 只有 `16 * 4 = 64` 个 CTA，未完全覆盖 A100 的 SM 数量，
但增加 hidden tile 到 8 个 CTA 会让每个 CTA 的有效计算变少，并放大每步 cooperative
同步和调度成本，整体约慢 1.31x。因此当前不要为了提高 resident CTA 数量切到 htile8。

同配置下与 `torch.nn.GRU` 的训练 forward+backward 对比：

| 层数 | torch.nn.GRU | 当前默认 A100 | 加速 |
| --- | ---: | ---: | ---: |
| 1 | 242.522 ms | 88.494 ms | 2.74x |
| 2 | 302.694 ms | 182.593 ms | 1.66x |
| 3 | 465.242 ms | 277.563 ms | 1.68x |
| 4 | 560.712 ms | 371.534 ms | 1.51x |

## 新多层 Kernel 实验

新增两个只用于 forward-only/inference 的多层 fused kernel：

- `forward_inference_stacked_naive()`：
  一个 CUDA block 负责一个 batch item，在 kernel 内按 `time -> layer` 推进。
  该版本用于验证 fused 多层调度和公式正确性；内部 256 维 dot-product 由单线程串行完成，
  性能不可作为优化目标。
- `forward_inference_stacked_row4()`：
  一个 cooperative kernel 使用 `batch * 16` 个 CTA，复用 row4 的 hidden/K 分片；
  在 kernel 内按 layer 串行推进，layer>0 的 r/z input projection 与 recurrent projection
  合并进同一次 partial reduction，n gate 保持 `input_n + reset * recurrent_n` 的公式分离。

当前 `forward_inference_stacked_row4()` 已通过 `num_layers=2/3/4` 的 functional test，
但还不是性能收益路径。A100 80GB、`input_size=16`、`hidden_size=256`、`batch=16`、
`seq=8000`、fp32、TF32 关闭下：

| 层数 | 当前默认 K1 forward_inference | stacked row4 | 结论 |
| --- | ---: | ---: | --- |
| 1 | 52.529 ms | 93.131 ms | 慢 1.77x |
| 2 | 107.343 ms | 193.057 ms | 慢 1.80x |
| 3 | 160.839 ms | 280.426 ms | 慢 1.74x |
| 4 | 216.133 ms | 375.470 ms | 慢 1.74x |

主要瓶颈是 layer>0 的 input projection 仍未达到 cuBLAS/F.linear 的效率，且每个 time/layer
仍有多次 `grid.sync()`。`stacked_row4` 已经给 n gate 的 `input_n` 增加独立 partial
分片，去掉了 update 阶段的串行 256 维 dot；该改动把 2/3/4 层从
401.370/726.897/1030.952 ms 降到 193.728/278.802/373.072 ms，但仍慢于逐层 K1。

新增训练版真正 fused 原型：

- `forward_train_stacked_row4_cache()`：
  单个 cooperative CUDA kernel 在 forward 中跨 layer 推进，同时写出最终 output、
  `h_n`、每层 `all_outputs` 和每层 `gate_cache_all`。`num_layers=2/3/4`
  小规模正确性已和 `torch.nn.GRU` 对齐，最大 forward 误差约 `3.6e-7`。
- `forward_train_stacked_fused_naive()`：
  基于上面的 fused forward cache，backward 使用
  `a100_gru_h256_stacked_backward_naive_kernel` 在单个 CUDA kernel 内按
  `layer: top -> bottom`、`time: last -> first` 传播 `grad_x`、`grad_h0`、
  `grad_input_gates_all` 和 `grad_hidden_gates_all`。再用 PyTorch matmul 组合参数梯度。
  `num_layers=2/3/4` 的 end-to-end autograd functional test 已通过，输入、h0 和参数梯度
  最大误差约 `1e-6` 量级。

该训练 fused 原型目前只是正确性和调度基线，不能替换默认性能路径。A100 80GB、
`batch=16`、`seq=512`、`input_size=16`、`hidden_size=256`、fp32、TF32 关闭下：

| 层数 | 当前默认训练 | stacked fused naive | 结论 |
| --- | ---: | ---: | --- |
| 2 | 12.141 ms | 119.705 ms | 慢 9.86x |
| 3 | 18.450 ms | 153.780 ms | 慢 8.34x |
| 4 | 24.749 ms | 187.498 ms | 慢 7.58x |

负收益原因：naive backward 每个 batch 只用一个 block，虽然跨 layer 的梯度传播已经在
单个 kernel 内完成，但 recurrent/input 两个矩阵向量反传由每个 hidden lane 串行遍历
`3 * hidden`，吞吐远低于当前 split6_unroll8 分片 backward 和 PyTorch/cuBLAS 的权重梯度
GEMM。因此下一步如果继续推进真正 fused backward，必须把该原型改成 split6/row4 风格的
多 CTA 分片 reduce，而不是在当前 naive kernel 上做微调。

新增 split-style fused backward 实验：

- `forward_train_stacked_fused_split4()`：
  使用 `a100_gru_h256_stacked_backward_split4_kernel`，每个 batch 从 1 个 CTA 提升到
  4 个 CTA，每个 hidden 维由 4 个线程协作完成 recurrent/input 反传。该路径仍然在单个
  cooperative kernel 内按 layer/time 传播梯度，是真正 fused backward。当前版本已移除
  time step 末尾的 post-step `grid.sync()`，只保留 gate 梯度生成后的同步和 layer 边界同步。
- `forward_train_stacked_fused_split8()`：
  使用 8 个 CTA/批次、8 个线程协作一个 hidden 维。移除 post-step `grid.sync()` 后，
  2 层单项 backward 接近 split4，但 3/4 层和完整 step 仍慢于 split4。
- `forward_train_stacked_fused_split4_shmem()`：
  缓存本 CTA 负责的 64 个 hidden gate 梯度到 shared memory，减少本 tile 的 global read。
  该方向正确但略慢，说明当前瓶颈更偏向同步/调度，而不是本 tile gate 梯度 global read。
- `forward_train_stacked_row4_k1_cache()`：
  新增 stacked K1 训练 forward。每个 batch 使用 4 个 h-tile CTA，不再使用 4 个 K 分片 CTA，
  去掉跨 K partial reduce 和对应的 grid-wide 同步；layer>0 的 input/recurrent projection
  仍在同一个 kernel 内完成。该路径会写出 `all_outputs` 和 `gate_cache_all`，可直接接
  split4 fused backward。
- `forward_train_stacked_fused_split4_k1_forward()`：
  使用 K1 stacked forward + split4 fused backward，是当前 fused 训练路径中最快的实验分支。
- `forward_train_stacked_fused_split4_group8()`：
  保持 4 个 CTA/批次，但每 8 个线程协作两个 hidden 维。该路径在 2 层 backward 有小幅收益，
  3/4 层为负收益，只保留为实验入口。
- `forward_train_stacked_fused_split6_weight_shmem()`：
  将当前默认单层 backward 的 split6/weight-shmem 思路迁移到 stacked fused backward：
  每个 batch 使用 6 个 CTA 分担 `3 * hidden` gate row，recurrent weight tile 常驻 shared
  memory，input partial 用双 buffer 防止 step 间覆盖。该路径是当前最快的真正 fused
  多层 backward 实验，但仍慢于默认逐层训练路径。当前版本增加了
  `__launch_bounds__(256, 1)`，匹配 132KB shared memory 下实际 1 block/SM 的 resident
  约束，避免编译器按更高 occupancy 压低寄存器分配。该快版本放在
  `gru_fused_split6_kernel.cu` 独立 NVRTC 编译单元中；直接放进主 CUDA module 会影响默认
  pack hidden-prev helper 的长序列重复 launch 稳定性。

A100 80GB、`batch=16`、`seq=512`、`input_size=16`、`hidden_size=256`、fp32、
TF32 关闭下，单独 backward kernel：

| 层数 | naive backward | split4 backward | split4_group8 backward | split6_weight_shmem backward | split6 对 split4 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 2 | 106.477 ms | 20.131 ms | 19.570 ms | 7.069 ms | 2.85x |
| 3 | 133.261 ms | 30.277 ms | 30.888 ms | 10.580 ms | 2.86x |
| 4 | 160.662 ms | 40.031 ms | 42.407 ms | 14.179 ms | 2.82x |

完整 forward+backward step，不含 optimizer：

| 层数 | 当前默认 | fused split4 K1 | fused split4_group8 | fused split6_weight_shmem | 结论 |
| --- | ---: | ---: | ---: | ---: | --- |
| 2 | 11.829 ms | 29.231 ms | 28.697 ms | 16.170 ms | split6 仍慢 1.37x |
| 3 | 17.920 ms | 44.862 ms | 45.489 ms | 25.034 ms | split6 仍慢 1.40x |
| 4 | 24.144 ms | 59.908 ms | 62.270 ms | 34.100 ms | split6 仍慢 1.41x |

结论：split6_weight_shmem 是当前 fused backward 最佳实验分支，相比 K1+split4 fused
完整 step 提升约 43%-45%，单独 backward 提升约 2.8x。它仍慢于当前默认逐层
`K1 forward + split6_unroll8 backward`，主要剩余瓶颈是 fused stacked forward 和
backward 中 layer>0 input projection 仍无法达到 PyTorch/cuBLAS 的 GEMM 吞吐。
`split4_group8` 仅 2 层小幅正收益，3/4 层为负收益；第 0 层无效 input partial 跳过实验
也略慢，已撤回。`__ldg` 只读权重加载会把 fused split6 backward 从约
`10.105/14.758/19.444 ms` 退化到 `33.615/49.181/65.004 ms`，已撤回；
`unroll4` 也慢于 `unroll8`，已撤回。

新增 `forward_train_stacked_fused_split6_hybrid_forward()` 实验入口：

- layer input projection 回到 PyTorch/cuBLAS `F.linear`，recurrent forward 复用单层
  row4/K1 gate-cache kernel。
- fused backward 仍使用当前最快的 `split6_weight_shmem` 独立编译单元。
- 单层 row4/K1 wrapper 新增可选 `output_out/gate_cache_out`，hybrid 路径直接写入
  `[layer,batch,seq,*]` 的最终缓存，避免最后 `torch.stack` 整层拷贝。

A100 80GB、`batch=16`、`seq=512`、`input_size=16`、`hidden_size=256`、fp32、
TF32 关闭下，完整 forward+backward step：

| 层数 | fused split6_weight_shmem | split6 hybrid forward | hybrid 加速 |
| --- | ---: | ---: | ---: |
| 2 | 16.273 ms | 15.134 ms | 1.075x |
| 3 | 25.114 ms | 22.912 ms | 1.096x |
| 4 | 34.242 ms | 30.576 ms | 1.120x |

结论：hybrid forward 证明 fused split6 backward 可以配合更高效的 layer input projection，
让真正 fused 训练路径继续接近默认逐层路径；但当前仍慢于默认
`逐层 K1 forward + split6_unroll8 backward`，因此只保留为实验入口，不切默认。
尝试用复用 scratch buffer 的 `addmm(out=...)` 替代 `F.linear`，2 层只有噪声级收益，
3/4 层略慢，已撤回。

stacked K1 forward 复测，A100 80GB、`batch=16`、`seq=512`、`input_size=16`、
`hidden_size=256`、fp32、TF32 关闭：

| 层数 | K4 fused forward | K1 fused forward | forward 提升 | K4 split4 step | K1 split4 step | step 提升 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 2 | 12.606 ms | 8.394 ms | 1.50x | 33.914 ms | 29.948 ms | 1.13x |
| 3 | 18.919 ms | 13.343 ms | 1.42x | 50.404 ms | 44.936 ms | 1.12x |
| 4 | 25.276 ms | 18.558 ms | 1.36x | 67.164 ms | 60.513 ms | 1.11x |

同配置下和当前默认路径对比：

| 层数 | 当前默认 | K1 fused split4 | 差距 |
| --- | ---: | ---: | ---: |
| 2 | 12.093 ms | 29.650 ms | 慢 2.45x |
| 3 | 18.720 ms | 45.345 ms | 慢 2.42x |
| 4 | 24.780 ms | 60.526 ms | 慢 2.44x |

结论：K1 stacked forward 是正收益，说明 fused forward 的同步/partial reduce 成本确实很高；
但 fused 训练仍明显慢于当前默认逐层 K1 forward + split6_unroll8 backward。下一步继续做
fused 时，应优先降低 fused backward 的矩阵向量反传成本，或把 stacked forward 的 layer>0
projection 改成更接近 GEMM 的批量计算，而不是回到 K4 partial reduce。

下一步若继续推进，应优先：

- 继续围绕 `forward_train_stacked_fused_split6_weight_shmem()` 做瓶颈拆分，优先确认
  `grad_input_gates @ W_ih` 的 global weight 访问是否已经成为主要瓶颈。
- 不要简单增加 split 数或 hidden tile 数；batch=16 下 cooperative resident grid 已经约束
  很强，双 weight tile 方案通常会因为 block 数或 shared memory 上限不可 resident。
- 用 Nsight 观察 split6_weight_shmem 的 `grid.sync()`、input/recurrent partial_sums global
  读写和 shared-memory replay，判断是否还能减少每个 time step 的同步/partial 成本。
- 对比 K1/K2/K4 在 Nsight 下的 grid sync、occupancy 和 memory transaction，确认 K1
  安全同步后的剩余瓶颈。
- 若继续做 stacked kernel，需要让 layer>0 input projection 接近 cuBLAS/F.linear 的吞吐，
  否则 fused 多层路径没有性能优势。
