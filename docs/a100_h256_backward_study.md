# A100 h256 GRU backward 原型研究

## 目标

在 A100/SM80 的 `hidden_size=256` 路线上，把 forward-only 原型推进到训练闭环。
本轮目标不是直接得到最快 backward，而是先验证：

- A100 h256 forward kernel 能否接入 `torch.autograd.Function`。
- 单层、单向、batch-first、fp32 GRU 的 backward 公式是否完整覆盖所有梯度。
- 完整训练 step 中 forward、backward、optimizer 的真实耗时比例。

仍然保持以下约束：

- 不使用 TF32、bf16 或 AMP。
- 不修改系统 NVIDIA 驱动。
- 当前只覆盖 A100/SM80 和 `hidden_size=256`。
- 代码标识符用英文，注释和文档用中文。

## 已实现内容

新增文件：

- `src/rnn_kernel/a100/gru_autograd.py`
- `tests/test_a100_gru_autograd.py`
- `scripts/profile_a100_h256_step.py`

新增 CUDA kernel：

- `a100_gru_h256_pointwise_backward_kernel`
- `a100_gru_h256_recurrent_backward_kernel`
- `a100_gru_h256_recurrent_backward_tiled_kernel`
- `a100_gru_h256_recurrent_backward_split_kernel`
- `a100_gru_h256_recurrent_backward_split_reduce_kernel`
- `a100_gru_h256_backward_step_kernel`
- `a100_gru_h256_backward_step_cooperative_split_kernel`
- `a100_gru_h256_backward_step_cooperative_split2_kernel`
- `a100_gru_h256_backward_step_cooperative_split_cached_kernel`
- `a100_gru_h256_backward_step_cooperative_split2_cached_local_kernel`
- `a100_gru_h256_backward_step_cooperative_split2_gate_cache_kernel`
- `a100_gru_h256_backward_sequence_cooperative_split2_cached_local_kernel`
- `a100_gru_h256_backward_sequence_cooperative_split2_state_parts_kernel`
- `a100_gru_h256_backward_sequence_cooperative_split2_state_local_kernel`
- `a100_gru_h256_backward_sequence_cooperative_split4_state_kernel`
- `a100_gru_h256_backward_sequence_cooperative_split8_state_kernel`
- `a100_gru_h256_backward_sequence_cooperative_split16_state_kernel`
- `a100_gru_h256_backward_sequence_cooperative_split16_gate_cache_state_kernel`
- `a100_gru_h256_backward_sequence_cooperative_split16_gate_cache_state_tiled_kernel`
- `a100_gru_h256_backward_sequence_cooperative_split16_grad_coeff_cache_state_tiled_kernel`
- `a100_gru_h256_backward_sequence_cooperative_split16_state_global_gates_kernel`
- `a100_gru_h256_backward_sequence_cooperative_split32_state_kernel`
- `a100_gru_h256_backward_sequence_cooperative_split32_gate_cache_state_tiled_kernel`
- `a100_gru_h256_backward_step_recompute_kernel`
- `a100_gru_forward_from_gates_cooperative_h256_shmem_gate_cache_kernel`
- `a100_gru_forward_from_gates_cooperative_h256_shmem_grad_coeff_cache_kernel`
- `a100_gru_forward_from_gates_cooperative_h256_parallel_update_gate_cache_kernel`
- `a100_gru_forward_from_gates_cooperative_h256_cta8_shmem_gate_cache_kernel`
- `a100_gru_forward_from_gates_cooperative_h256_cta6_shmem_gate_cache_kernel`

新增接口：

- `A100GRUH256Function`
- `a100_gru_h256`
- `A100GRUH256`
- `copy_from_torch_gru`

`rnn_benchmark.py` 新增 implementation：

```bash
--implementation a100_gru_h256
--implementation a100_gru_h256_recurrent_kernel
--implementation a100_gru_h256_tiled_recurrent
--implementation a100_gru_h256_split_recurrent
--implementation a100_gru_h256_split4_recurrent
--implementation a100_gru_h256_coop_split2
--implementation a100_gru_h256_coop_split4
--implementation a100_gru_h256_coop_split2_cached
--implementation a100_gru_h256_coop_split2_cached_local
--implementation a100_gru_h256_coop_split2_gate_cache
--implementation a100_gru_h256_coop_split2_persistent
--implementation a100_gru_h256_coop_split2_persistent_state
--implementation a100_gru_h256_coop_split2_persistent_state_local
--implementation a100_gru_h256_coop_split4_persistent_state
--implementation a100_gru_h256_coop_split8_persistent_state
--implementation a100_gru_h256_coop_split16_persistent_state
--implementation a100_gru_h256_coop_split16_persistent_state_gate_cache
--implementation a100_gru_h256_coop_split16_persistent_state_gate_cache_tiled
--implementation a100_gru_h256_coop_split16_persistent_state_gate_cache_tiled_cta6
--implementation a100_gru_h256_coop_split16_persistent_state_grad_coeff_cache_tiled
--implementation a100_gru_h256_coop_split32_persistent_state_gate_cache_tiled
--implementation a100_gru_h256_coop_split16_persistent_state_gate_cache_parallel_update
--implementation a100_gru_h256_coop_split16_persistent_state_gate_cache_cta8
--implementation a100_gru_h256_coop_split16_persistent_state_global_gates
--implementation a100_gru_h256_coop_split32_persistent_state
--implementation a100_gru_h256_coop_split2_specialized
--implementation a100_gru_h256_recompute
```

该实现只支持：

- `cell_type=GRU`
- `hidden_size=256`
- `num_layers=1`
- `batch_first=True`
- `fp32`

## 实现策略

forward 路径：

1. 用 PyTorch `F.linear` 预计算 input gates。
2. 调用当前最快的 A100 h256 shmem cooperative recurrent kernel。
3. 返回完整 output 和 final hidden。

backward 路径当前分为两条：

1. 保存 `x`、`h0`、`weight_ih`、`weight_hh`、`bias_hh`、`input_gates` 和 `output`。
2. 使用保存的 `output[:, t - 1]` 作为 `h_{t-1}`，跨 time step 一次性重算
   recurrent gates，避免 backward 循环内发起大量小 GEMM。
3. 从最后一个 time step 反向遍历。
4. 默认对照路径调用 `a100_gru_h256_pointwise_backward_kernel` 计算每步 pointwise 梯度：
   - `grad_input_gates[:, t, :]`
   - `grad_hidden_gates`
   - direct `grad_h_prev`
5. 实验优化路径调用 `a100_gru_h256_backward_step_kernel`，把每步 pointwise backward 和
   recurrent input-gradient 合并在一个 CUDA kernel 内。
6. 循环结束后，把 `grad_hidden_gates.T @ h_prev` 聚合成跨 time step 的单次大 GEMM，
   计算 `grad_weight_hh` 和 `grad_bias_hh`。
7. 用 PyTorch matmul 继续计算：
   - `grad_x`
   - `grad_h0`
   - `grad_weight_ih`
   - `grad_bias_ih`

第一版 backward 是 PyTorch 公式原型，每个 time step 会发起多个 PyTorch CUDA op。
当前版本已经完成三次收敛：

1. gate 的逐元素反向融合成 CUDA pointwise kernel。
2. recurrent gate 重算和 `weight_hh` 梯度累积改成跨 time step 大 GEMM。
3. `a100_gru_h256_recurrent_kernel` 路径把 pointwise backward 与 recurrent
   input-gradient 合并为 `a100_gru_h256_backward_step_kernel`。

本轮追加了多条保留分支：

1. `a100_gru_h256_recompute`：在 backward step kernel 内重算 recurrent gates，
   减少 `hidden_gates_steps` 显存，但由于每步重复做低效 row-wise recurrent matvec，
   长序列速度明显变差。
2. `a100_gru_h256_tiled_recurrent`：把 recurrent input-gradient 拆成 tiled shared
   memory kernel，减少 `weight_hh` 重复读取，但多一次 per-step launch 后整体仍慢于
   fused backward step。
3. `a100_gru_h256_split_recurrent` / `a100_gru_h256_split4_recurrent`：使用额外
   partial buffer，把 recurrent input-gradient 沿 gate/K 维拆分成更多 CTA 并行计算，
   再 reduce 回 `grad_hidden`。该方向提高了单步并行度，但每步变成 pointwise、split、
   reduce 三次 launch，长序列仍慢于 fused backward step。
4. `a100_gru_h256_coop_split2` / `a100_gru_h256_coop_split4`：把 split partial 和
   reduce 放回同一个 cooperative kernel 内，用更多 CTA 和 partial buffer 换取更高
   单步并行度，同时保持每个 time step 只有一次 backward-step launch。
5. `a100_gru_h256_coop_split2_cached`：只让 split0 计算 pointwise backward，其它
   split CTA 在 `grid.sync()` 后复用 `grad_hidden_gates`，减少重复 sigmoid/tanh。
6. `a100_gru_h256_coop_split2_cached_local`：在 cached 基础上，让 split0 复用自己
   shared memory 中的 gate 梯度，减少一次全局回读。
7. `a100_gru_h256_coop_split2_specialized`：把 split2 边界和 reduce 写成固定形状，
   去掉通用 split_count 分支。
8. `a100_gru_h256_coop_split2_gate_cache`：forward 额外保存
   `reset/update/new/recurrent_new`，backward 直接读 cache，尝试用更多显存换掉
   backward 中的 sigmoid/tanh 和 hidden-gates 读取。实测全局 cache 写读开销更大，
   是负向实验。
9. `a100_gru_h256_coop_split2_persistent`：把 backward 的倒序 time loop 合并进
   单个 cooperative kernel，保留 cached-local 的 split2 计算结构，但消除每个
   time step 一次 Python/CUDA launch 的外层调度成本。该分支证明 persistent
   组织是正确方向。
10. `a100_gru_h256_coop_split2_persistent_state`：把 `grad_hidden_prev_direct` 合入
    split0 的 partial state，下一步直接从两个 partial state 还原 `grad_hidden`。
    这个版本每个 time step 只需要一次 `grid.sync()`，代价是 split0/split1 都重算
    pointwise backward。该分支曾是 split2 阶段最佳。
11. `a100_gru_h256_coop_split2_persistent_state_local`：在 state 基础上让每个 split
    用寄存器保留自己的 partial state，只读对侧 partial。实测基本持平略慢，说明
    state partial 的全局读回不是主要瓶颈。
12. `a100_gru_h256_coop_split4_persistent_state`：把 recurrent input-gradient 拆成
    4 个 CTA，每个 CTA 处理 192 个 gate 项，继续保持每步一次 `grid.sync()`。该分支
    明显超过 cuDNN。
13. `a100_gru_h256_coop_split8_persistent_state`：把 recurrent input-gradient 拆成
    8 个 CTA，每个 CTA 处理 96 个 gate 项。虽然重复 pointwise 和 partial 读写更多，
    但 h256 目标形状上 recurrent dot-product 仍是主导，该分支相对 split4 继续收益。
14. `a100_gru_h256_coop_split16_persistent_state`：继续把 recurrent input-gradient
    拆成 16 个 CTA，每个 CTA 处理 48 个 gate 项。该分支相对 split8 仍有小幅收益。
15. `a100_gru_h256_coop_split32_persistent_state`：继续拆到 32 个 CTA，每个 CTA 处理
    24 个 gate 项。实测同步、partial 规约和重复 pointwise 成本超过收益，是负向实验。
16. `a100_gru_h256_coop_split16_persistent_state_global_gates`：只让 split0 计算
    gate 梯度并写全局缓存，其它 split 复用，目标是用显存读写减少重复 pointwise。
    实测额外 `grid.sync()` 和全局读成本更高，是负向实验。
17. `a100_gru_h256_coop_split16_persistent_state_gate_cache`：forward 额外保存
    `reset/update/new/recurrent_new`，backward persistent split16 直接读取 cache，跳过
    `hidden_gates_steps = F.linear(...)` 重算和 sigmoid/tanh。
18. `a100_gru_h256_coop_split16_persistent_state_gate_cache_parallel_update`：forward
    让 4 个 CTA 分摊 hidden update 和 gate-cache 写入，但 CTA0 partial 也要落全局
    内存。实测 forward 变慢，是负向实验。
19. `a100_gru_h256_coop_split16_persistent_state_gate_cache_cta8`：forward 改成每个
    batch 8 个 CTA，提高常驻 CTA 数，尝试改善 A100 SM 利用率。`block_threads=704`
    下无法 cooperative resident launch，降低 block 后 forward 仍明显变慢，是负向实验。
20. `a100_gru_h256_coop_split16_persistent_state_gate_cache_tiled`：在 split16 gate-cache
    backward 中只把本 split 需要的 48 个 gate 梯度放入 shared memory，避免每个 split
    都搬运完整 3H gate。seq8000 timed_steps=10 降到 `137.471 ms/step`，是当前最佳。
21. `a100_gru_h256_coop_split32_persistent_state_gate_cache_tiled`：继续拆到 32 个 CTA，
    每个 split 只处理 24 个 gate 项。seq8000 复测为 `146.225 ms/step`，同步和 partial
    规约成本超过收益，是负向实验。
22. `a100_gru_h256_coop_split16_persistent_state_grad_coeff_cache_tiled`：forward 额外保存
    5H backward 导数系数，尝试用显存换掉 backward pointwise 乘法链。seq8000 复测为
    `139.136 ms/step`，峰值显存 `2.23 GB`，略慢于当前最佳。
23. `a100_gru_h256_coop_split16_persistent_state_gate_cache_tiled_cta6`：forward 改成每个
    batch 6 个 CTA，形成 96 个 cooperative blocks。seq8000 复测为 `148.113 ms/step`，
    非整除 k-tile 和 6-way partial 规约成本更高，是负向实验。

cached-local 仍不是最终性能实现，因为每个 time step 至少有一个 backward kernel
launch。persistent 分支已经把这部分 launch 合并，但仍显式保存 recurrent gates 与
`grad_hidden_gates`，峰值显存高于 cuDNN。

## 正确性验证

命令：

```bash
source scripts/env.sh
CUDA_VISIBLE_DEVICES=2 .venv/bin/python -m pytest tests/test_a100_gru_autograd.py -q
```

结果：

```text
33 passed
```

完整测试套件：

```bash
source scripts/env.sh
CUDA_VISIBLE_DEVICES=2 .venv/bin/python -m pytest -q
```

结果：

```text
84 passed
```

测试覆盖：

- output 与 `torch.nn.GRU` 对齐
- final hidden 与 `torch.nn.GRU` 对齐
- `x` 梯度对齐
- `h0` 梯度对齐
- `weight_ih_l0`、`weight_hh_l0`、`bias_ih_l0`、`bias_hh_l0` 梯度对齐
- `a100_gru_h256_recurrent_backward_kernel` 与
  `grad_hidden_prev_direct + grad_hidden_gates.matmul(weight_hh)` 对齐
- `a100_gru_h256_backward_step_kernel` 与
  `pointwise backward + grad_hidden_gates.matmul(weight_hh)` 单步对齐
- `a100_gru_h256_backward_step_recompute_kernel` 与显式 hidden gates 的 fused step
  单步对齐
- `a100_gru_h256_recurrent_backward_tiled_kernel` 与
  `grad_hidden_prev_direct + grad_hidden_gates.matmul(weight_hh)` 对齐
- `a100_gru_h256_recurrent_backward_split_kernel` 与
  `grad_hidden_prev_direct + grad_hidden_gates.matmul(weight_hh)` 对齐，覆盖
  `split_count=8` 和 `split_count=4`
- `a100_gru_h256_backward_step_cooperative_split_kernel` 与 fused step 单步对齐，
  覆盖 `split_count=4` 和 `split_count=2`
- `a100_gru_h256_backward_step_cooperative_split_cached_kernel` 与 fused step 单步对齐
- `a100_gru_h256_backward_step_cooperative_split2_cached_local_kernel` 与 fused step 单步对齐
- `a100_gru_h256_backward_step_cooperative_split2_gate_cache_kernel` 与 fused step 单步对齐
- `a100_gru_h256_backward_step_cooperative_split2_kernel` 与 fused step 单步对齐
- `use_recurrent_backward_kernel=True` 的完整 autograd 路径与 `torch.nn.GRU` 对齐
- `recompute_hidden_gates=True` 的完整 autograd 路径与 `torch.nn.GRU` 对齐
- `use_tiled_recurrent_backward_kernel=True` 的完整 autograd 路径与 `torch.nn.GRU` 对齐
- `use_split_recurrent_backward_kernel=True` 的完整 autograd 路径与 `torch.nn.GRU` 对齐
- `use_cooperative_split_backward_kernel=True` 的完整 autograd 路径与
  `torch.nn.GRU` 对齐
- `use_cooperative_split_cached_backward_kernel=True` 的完整 autograd 路径与
  `torch.nn.GRU` 对齐
- `use_cooperative_split2_cached_local_backward_kernel=True` 的完整 autograd 路径与
  `torch.nn.GRU` 对齐
- `use_cooperative_split2_backward_kernel=True` 的完整 autograd 路径与
  `torch.nn.GRU` 对齐
- `use_gate_cache_backward_kernel=True` 的完整 autograd 路径与 `torch.nn.GRU` 对齐
- `use_persistent_backward_kernel=True` 的完整 autograd 路径与 `torch.nn.GRU` 对齐
- `use_persistent_state_backward_kernel=True` 的完整 autograd 路径与
  `torch.nn.GRU` 对齐
- `use_persistent_state_local_backward_kernel=True` 的完整 autograd 路径与
  `torch.nn.GRU` 对齐
- `use_persistent_state4_backward_kernel=True` 的完整 autograd 路径与
  `torch.nn.GRU` 对齐
- `use_persistent_state8_backward_kernel=True` 的完整 autograd 路径与
  `torch.nn.GRU` 对齐
- `use_persistent_state16_backward_kernel=True` 的完整 autograd 路径与
  `torch.nn.GRU` 对齐
- `use_persistent_state32_backward_kernel=True` 的完整 autograd 路径与
  `torch.nn.GRU` 对齐
- `use_persistent_state16_global_gates_backward_kernel=True` 的完整 autograd 路径与
  `torch.nn.GRU` 对齐
- `use_persistent_state16_gate_cache_backward_kernel=True` 的完整 autograd 路径与
  `torch.nn.GRU` 对齐
- `use_persistent_state16_gate_cache_backward_kernel=True` 且
  `use_gate_cache_parallel_update_forward_kernel=True` 的完整 autograd 路径与
  `torch.nn.GRU` 对齐
- `use_persistent_state16_gate_cache_backward_kernel=True` 且
  `use_gate_cache_cta8_forward_kernel=True` 的完整 autograd 路径与 `torch.nn.GRU` 对齐
- `use_persistent_state16_gate_cache_tiled_backward_kernel=True` 的完整 autograd 路径与
  `torch.nn.GRU` 对齐
- `use_persistent_state32_gate_cache_tiled_backward_kernel=True` 的完整 autograd 路径与
  `torch.nn.GRU` 对齐
- `use_persistent_state16_grad_coeff_cache_tiled_backward_kernel=True` 的完整 autograd 路径与
  `torch.nn.GRU` 对齐
- `use_persistent_state16_gate_cache_tiled_backward_kernel=True` 且
  `use_gate_cache_cta6_forward_kernel=True` 的完整 autograd 路径与 `torch.nn.GRU` 对齐

常规路径参数梯度使用 `atol=3e-3, rtol=5e-4`，实验 backward kernel 路径最高使用到
`atol=5e-3, rtol=1e-3`。原因是 forward 使用 A100 4-CTA fp32 规约，而 backward
原型和实验 kernel 的规约顺序不同，会放大到 `weight_hh_l0` 的最大绝对误差。实测最大
误差约 `2.4e-3`，平均误差约 `7e-5`。

## 训练拆分结果

硬件和软件：

- GPU：NVIDIA A100 80GB PCIe
- PyTorch：2.6.0+cu124
- CUDA runtime：12.4
- cuDNN：9.1.0
- dtype：fp32
- TF32：关闭

### seq_len=256

PyTorch cuDNN：

```bash
CUDA_VISIBLE_DEVICES=1 .venv/bin/python rnn_benchmark.py \
  --implementation torch \
  --cell-types GRU \
  --hidden-sizes 256 \
  --num-layers 1 \
  --batch-size 16 \
  --seq-len 256 \
  --input-dim 9 \
  --dataset-batches 4 \
  --warmup-steps 1 \
  --timed-steps 3 \
  --breakdown-timing \
  --output-csv results/a100_h256_train_torch_seq256.csv
```

A100 custom autograd：

```bash
CUDA_VISIBLE_DEVICES=1 .venv/bin/python rnn_benchmark.py \
  --implementation a100_gru_h256 \
  --cell-types GRU \
  --hidden-sizes 256 \
  --num-layers 1 \
  --batch-size 16 \
  --seq-len 256 \
  --input-dim 9 \
  --dataset-batches 4 \
  --warmup-steps 1 \
  --timed-steps 3 \
  --breakdown-timing \
  --output-csv results/a100_h256_train_custom_seq256.csv
```

表中的 fused backward step 使用 `--implementation a100_gru_h256_recurrent_kernel`。

结果：

| implementation | step ms | forward ms | backward ms | optimizer ms | peak memory |
| --- | ---: | ---: | ---: | ---: | ---: |
| PyTorch cuDNN | 9.251 | 4.051 | 4.717 | 0.220 | 0.08 GB |
| PyTorch cuDNN（复测） | 9.955 | 3.889 | 5.407 | 0.433 | 0.08 GB |
| A100 custom autograd，Python 公式 backward | 148.387 | 2.746 | 144.504 | 0.799 | 0.07 GB |
| A100 custom autograd，pointwise + batched GEMM | 27.365 | 2.682 | 23.758 | 0.656 | 0.10 GB |
| A100 custom autograd，fused backward step | 19.165 | 2.706 | 15.366 | 0.794 | 0.10 GB |
| A100 custom autograd，cooperative split2 step | 22.572 | 4.079 | 17.306 | 0.593 | 0.10 GB |
| A100 custom autograd，cooperative split2 cached | 18.248 | 2.681 | 14.498 | 0.744 | 0.10 GB |
| A100 custom autograd，cooperative split2 cached-local | 17.926 | 2.731 | 14.081 | 0.725 | 0.10 GB |
| A100 custom autograd，cooperative split2 persistent | 9.351 | 2.705 | 6.251 | 0.148 | 0.10 GB |
| A100 custom autograd，cooperative split2 persistent-state | 9.101 | 2.772 | 5.884 | 0.148 | 0.10 GB |
| A100 custom autograd，cooperative split2 persistent-state-local | 9.358 | 2.914 | 5.886 | 0.148 | 0.10 GB |
| A100 custom autograd，cooperative split4 persistent-state | 6.945 | 2.859 | 3.593 | 0.148 | 0.10 GB |
| A100 custom autograd，cooperative split8 persistent-state | 5.641 | 2.704 | 2.557 | 0.148 | 0.10 GB |
| A100 custom autograd，cooperative split16 persistent-state | 5.472 | 2.647 | 2.471 | 0.148 | 0.10 GB |
| A100 custom autograd，cooperative split16 persistent-state gate-cache | 5.117 | 2.609 | 2.215 | 0.147 | 0.11 GB |
| A100 custom autograd，cooperative split16 persistent-state gate-cache tiled | 5.118 | 2.708 | 2.031 | 0.148 | 0.11 GB |
| A100 custom autograd，cooperative split16 persistent-state global-gates | 6.540 | 2.782 | 3.314 | 0.147 | 0.10 GB |
| A100 custom autograd，cooperative split32 persistent-state | 6.660 | 2.898 | 3.252 | 0.147 | 0.10 GB |
| A100 custom autograd，cooperative split2 specialized | 21.760 | 2.637 | 18.154 | 0.716 | 0.10 GB |
| A100 custom autograd，tiled recurrent 实验 | 45.713 | 4.813 | 38.598 | 1.015 | 0.10 GB |
| A100 custom autograd，split4 recurrent 实验 | 33.679 | 4.863 | 26.684 | 0.317 | 0.10 GB |
| A100 custom autograd，split8 recurrent 实验 | 31.183 | 4.858 | 24.784 | 0.434 | 0.10 GB |
| A100 custom autograd，recompute hidden gates 实验 | 93.256 | 4.807 | 86.881 | 0.152 | 0.09 GB |

结论：A100 forward 在训练图中也更快。CUDA pointwise backward、跨 time step
batched GEMM 和 fused backward step 逐步把 seq256 backward 从 `144.504 ms` 降到
`15.366 ms`。cooperative split2 cached-local 继续降到 `14.081 ms`。persistent
分支把每步 launch 合并后，seq256 backward 降到 `6.251 ms`，总 step `9.351 ms`。
进一步用 partial state 合并 direct 梯度后，persistent-state 总 step 降到 `9.101 ms`。
split4 和 split8 继续提高 recurrent input-gradient 并行度，分别降到 `6.945 ms` 和
`5.641 ms`；split16 进一步降到 `5.472 ms`。split16 gate-cache persistent 降到
`5.117 ms`，新的 split16 gate-cache tiled 把 backward 从 `2.215 ms` 继续降到
`2.031 ms`，总 step 基本持平为 `5.118 ms`，已经明显快于本轮 cuDNN 复测的
`9.955 ms`。split32 和 global-gates 在短序列上都退化，说明更多 CTA 或额外 grid sync
不是当前主线。
state-local 没有继续提升，说明对侧 partial 读回不是短序列主瓶颈。recompute hidden
gates 在短序列上明显变慢。
seq256 由于 timed steps 少、kernel launch 占比较高，长序列结论更有代表性。

### seq_len=8000

带 warmup 的 3-step 均值结果：

| implementation | step ms | forward ms | backward ms | optimizer ms | peak memory |
| --- | ---: | ---: | ---: | ---: | ---: |
| PyTorch cuDNN | 243.737 | 113.874 | 129.433 | 0.146 | 1.64 GB |
| PyTorch cuDNN（复测） | 242.647 | 113.816 | 128.348 | 0.147 | 1.65 GB |
| PyTorch cuDNN（timed_steps=10 复测） | 251.414 | 114.062 | 136.804 | 0.147 | 1.65 GB |
| A100 custom autograd，Python 公式 backward | 4821.995 | 74.411 | 4745.219 | 1.686 | 1.12 GB |
| A100 custom autograd，pointwise + batched GEMM | 975.206 | 120.547 | 853.240 | 0.771 | 1.97 GB |
| A100 custom autograd，fused backward step | 661.825 | 110.420 | 549.944 | 0.467 | 1.97 GB |
| A100 custom autograd，cooperative split2 step | 657.316 | 74.258 | 582.463 | 0.150 | 1.97 GB |
| A100 custom autograd，cooperative split4 step | 682.180 | 112.411 | 568.128 | 0.152 | 1.97 GB |
| A100 custom autograd，cooperative split2 cached | 514.936 | 74.277 | 440.195 | 0.148 | 1.97 GB |
| A100 custom autograd，cooperative split2 cached-local | 510.658 | 74.274 | 435.895 | 0.149 | 1.97 GB |
| A100 custom autograd，cooperative split2 gate-cache | 547.562 | 77.081 | 469.918 | 0.148 | 2.11 GB |
| A100 custom autograd，cooperative split2 persistent | 267.939 | 77.624 | 189.624 | 0.149 | 1.99 GB |
| A100 custom autograd，cooperative split2 persistent-state | 256.554 | 77.720 | 177.997 | 0.149 | 1.99 GB |
| A100 custom autograd，cooperative split2 persistent-state-local | 256.759 | 77.638 | 178.341 | 0.149 | 1.99 GB |
| A100 custom autograd，cooperative split4 persistent-state | 185.327 | 77.748 | 106.688 | 0.148 | 1.99 GB |
| A100 custom autograd，cooperative split4 persistent-state（timed_steps=10） | 185.026 | 77.626 | 106.726 | 0.149 | 1.99 GB |
| A100 custom autograd，cooperative split8 persistent-state | 153.787 | 77.653 | 75.499 | 0.148 | 1.99 GB |
| A100 custom autograd，cooperative split8 persistent-state（timed_steps=10） | 153.701 | 77.614 | 75.429 | 0.148 | 1.99 GB |
| A100 custom autograd，cooperative split16 persistent-state | 150.202 | 77.491 | 71.907 | 0.149 | 1.99 GB |
| A100 custom autograd，cooperative split16 persistent-state（timed_steps=10） | 150.099 | 77.492 | 71.911 | 0.148 | 1.99 GB |
| A100 custom autograd，cooperative split16 persistent-state gate-cache | 143.265 | 76.855 | 65.803 | 0.148 | 2.11 GB |
| A100 custom autograd，cooperative split16 persistent-state gate-cache（timed_steps=10） | 143.204 | 76.813 | 65.852 | 0.147 | 2.11 GB |
| A100 custom autograd，cooperative split16 persistent-state gate-cache tiled | 137.722 | 76.881 | 60.001 | 0.149 | 2.11 GB |
| A100 custom autograd，cooperative split16 persistent-state gate-cache tiled（timed_steps=10） | 137.471 | 76.864 | 59.966 | 0.149 | 2.11 GB |
| A100 custom autograd，cooperative split16 persistent-state gate-cache tiled（GPU2 timed_steps=10） | 138.155 | 77.579 | 59.906 | 0.149 | 2.11 GB |
| A100 custom autograd，cooperative split32 persistent-state gate-cache tiled（GPU2） | 146.225 | 77.954 | 67.447 | 0.149 | 2.11 GB |
| A100 custom autograd，cooperative split16 persistent-state grad-coeff-cache tiled（GPU2） | 139.136 | 77.712 | 60.805 | 0.149 | 2.23 GB |
| A100 custom autograd，cooperative split16 persistent-state gate-cache tiled CTA6（GPU2） | 148.113 | 87.504 | 59.945 | 0.149 | 2.11 GB |
| A100 custom autograd，cooperative split16 persistent-state gate-cache parallel-update | 147.506 | 80.966 | 65.768 | 0.148 | 2.11 GB |
| A100 custom autograd，cooperative split16 persistent-state gate-cache CTA8（block=512） | 161.569 | 95.189 | 65.781 | 0.149 | 2.11 GB |
| A100 custom autograd，cooperative split16 persistent-state global-gates | 177.040 | 77.419 | 98.971 | 0.148 | 1.99 GB |
| A100 custom autograd，cooperative split32 persistent-state | 171.662 | 77.664 | 93.285 | 0.148 | 1.99 GB |
| A100 custom autograd，cooperative split2 specialized | 566.031 | 74.260 | 491.270 | 0.148 | 1.97 GB |
| A100 custom autograd，tiled recurrent 实验 | 817.929 | 121.791 | 694.579 | 0.154 | 1.97 GB |
| A100 custom autograd，split4 recurrent 实验 | 838.608 | 126.595 | 710.395 | 0.153 | 1.97 GB |
| A100 custom autograd，split8 recurrent 实验 | 1578.526 | 98.713 | 1477.228 | 0.971 | 1.97 GB |
| A100 custom autograd，recompute hidden gates 实验 | 2493.820 | 122.709 | 2369.842 | 0.154 | 1.61 GB |

split16 gate-cache forward block size 扫描，均为 seq8000、timed_steps=3：

| block threads | step ms | forward ms | backward ms |
| ---: | ---: | ---: | ---: |
| 512 | 145.257 | 78.795 | 65.710 |
| 640 | 143.412 | 76.618 | 65.815 |
| 672 | 146.213 | 79.737 | 65.822 |
| 704 | 143.265 | 76.855 | 65.803 |
| 736 | 143.379 | 76.634 | 65.836 |
| 768 | 144.970 | 78.365 | 65.838 |
| 1024 | 146.854 | 80.401 | 65.795 |

结论：forward block size 对 backward 几乎没有影响，主要影响 forward recurrent
kernel。`640/704/736` 接近，但没有稳定超过默认 `704`；因此默认仍保持 `704`。

forward gate-cache 结构实验，均为 seq8000、timed_steps=3：

| forward 结构 | block threads | step ms | forward ms | backward ms | 结论 |
| --- | ---: | ---: | ---: | ---: | --- |
| CTA4 shmem gate-cache | 704 | 143.265 | 76.855 | 65.803 | 旧 gate-cache 主线 |
| CTA4 shmem gate-cache tiled backward | 704 | 137.722 | 76.881 | 60.001 | 当前最佳 |
| CTA4 grad-coeff-cache tiled backward | 704 | 139.136 | 77.712 | 60.805 | 增加 5H cache 后略慢 |
| CTA6 shmem gate-cache tiled backward | 704 | 148.113 | 87.504 | 59.945 | 6-way partial 规约让 forward 变慢 |
| CTA4 parallel-update gate-cache | 704 | 147.506 | 80.966 | 65.768 | 失去 CTA0 shmem partial 后变慢 |
| CTA4 parallel-update gate-cache | 512 | 153.605 | 87.056 | 65.768 | 更慢 |
| CTA4 parallel-update gate-cache | 1024 | 148.009 | 81.327 | 65.772 | 仍慢 |
| CTA8 shmem gate-cache | 704 | launch 失败 | - | - | `grid_blocks=128` 超过 resident 上限 `108` |
| CTA8 shmem gate-cache | 512 | 161.569 | 95.189 | 65.781 | 更慢 |
| CTA8 shmem gate-cache | 384 | 163.604 | 97.047 | 65.800 | 更慢 |

结论：forward 的瓶颈不是 CTA0 hidden update/cache 写入，也不是 CTA 数不足可以直接解决。
CTA4 shmem 结构仍是当前 forward 主线；保留 CTA8 和 parallel-update 作为负向实验。

结果文件：

- `results/a100_h256_train_torch_seq8000.csv`
- `results/a100_h256_train_custom_seq8000.csv`
- `results/a100_h256_train_custom_recurrent_kernel_seq8000.csv`
- `results/a100_h256_train_custom_coop_split2_seq8000.csv`
- `results/a100_h256_train_custom_coop_split4_seq8000.csv`
- `results/a100_h256_train_custom_coop_split2_cached_seq8000.csv`
- `results/a100_h256_train_custom_coop_split2_cached_local_seq8000.csv`
- `results/a100_h256_train_custom_coop_split2_gate_cache_seq8000.csv`
- `results/a100_h256_train_custom_coop_split2_persistent_seq8000.csv`
- `results/a100_h256_train_custom_coop_split2_persistent_state_seq8000.csv`
- `results/a100_h256_train_custom_coop_split2_persistent_state_local_seq8000.csv`
- `results/a100_h256_train_custom_coop_split4_persistent_state_seq8000.csv`
- `results/a100_h256_train_custom_coop_split4_persistent_state_seq8000_t10_rerun.csv`
- `results/a100_h256_train_custom_coop_split8_persistent_state_seq8000.csv`
- `results/a100_h256_train_custom_coop_split8_persistent_state_seq8000_t10.csv`
- `results/a100_h256_train_custom_coop_split16_persistent_state_seq8000.csv`
- `results/a100_h256_train_custom_coop_split16_persistent_state_seq8000_t10.csv`
- `results/a100_h256_train_custom_coop_split16_persistent_state_gate_cache_seq8000.csv`
- `results/a100_h256_train_custom_coop_split16_persistent_state_gate_cache_seq8000_t10.csv`
- `results/a100_h256_train_custom_coop_split16_persistent_state_gate_cache_tiled_seq8000.csv`
- `results/a100_h256_train_custom_coop_split16_persistent_state_gate_cache_tiled_seq8000_t10.csv`
- `results/a100_h256_train_custom_coop_split16_persistent_state_gate_cache_tiled_seq8000_gpu2.csv`
- `results/a100_h256_train_custom_coop_split16_persistent_state_gate_cache_tiled_seq8000_t10_gpu2.csv`
- `results/a100_h256_train_custom_coop_split32_persistent_state_gate_cache_tiled_seq8000_gpu2.csv`
- `results/a100_h256_train_custom_coop_split16_persistent_state_grad_coeff_cache_tiled_seq8000_gpu2.csv`
- `results/a100_h256_train_custom_coop_split16_persistent_state_gate_cache_tiled_cta6_seq8000_gpu2.csv`
- `results/a100_h256_train_custom_coop_split16_persistent_state_gate_cache_parallel_update_seq8000.csv`
- `results/a100_h256_train_custom_coop_split16_gate_cache_parallel_update_seq8000_bt512.csv`
- `results/a100_h256_train_custom_coop_split16_gate_cache_parallel_update_seq8000_bt1024.csv`
- `results/a100_h256_train_custom_coop_split16_gate_cache_cta8_seq8000_bt512.csv`
- `results/a100_h256_train_custom_coop_split16_gate_cache_cta8_seq8000_bt384.csv`
- `results/a100_h256_train_custom_coop_split16_persistent_state_global_gates_seq8000.csv`
- `results/a100_h256_train_custom_coop_split32_persistent_state_seq8000.csv`
- `results/a100_h256_train_torch_seq8000_t10_rerun.csv`
- `results/a100_h256_train_custom_coop_split2_specialized_seq8000.csv`
- `results/a100_h256_train_custom_tiled_recurrent_seq8000.csv`
- `results/a100_h256_train_custom_split4_recurrent_seq8000.csv`
- `results/a100_h256_train_custom_split_recurrent_seq8000.csv`
- `results/a100_h256_train_custom_recompute_seq8000.csv`

结论：

1. A100 forward 仍然有效，但训练 breakdown 对单步波动敏感；forward-only benchmark
   仍以 `scripts/benchmark_a100_forward.py`
   为准。
2. pointwise + batched GEMM 把 backward 从 Python 公式原型的 `4745.219 ms`
   降到 `853.240 ms`。
3. fused backward step 继续把 backward 降到 `549.944 ms`。
4. cooperative split2 / split4 把单 kernel 时间显著降下来；cached split2 进一步
   减少重复 pointwise 计算，cached-local 继续减少 split0 的全局回读，seq8000 旧主线
   端到端结果为 `510.658 ms/step`。
5. tiled recurrent 实验没有超过 fused backward step，说明在当前 Python/autograd
   组织下，多一次 per-step launch 抵消了 weight 读取复用。
6. split recurrent 使用额外 partial buffer 提高单步并行度，但 split4 为
   `838.608 ms/step`，split8 为 `1578.526 ms/step`，都没有超过 fused backward step。
7. recompute hidden gates 把峰值显存从 `1.97 GB` 降到 `1.61 GB`，但 backward
   增加到 `2369.842 ms`，不能作为主线。
8. 旧的 split2 gate-cache 尝试用额外 `0.14 GB` 峰值显存换掉 backward 的部分
   pointwise 计算，但仍保留 per-step launch，总 step 退化到 `547.562 ms`，不是主线。
9. persistent backward 把 8000 个 backward-step launch 合并为 1 个 cooperative
   kernel，backward 从 cached-local 的 `435.895 ms` 降到 `189.624 ms`，总 step
   降到 `267.939 ms`。与同配置 cuDNN 复测 `242.647 ms` 相比，差距缩小到约 `10%`。
10. persistent-state 把每步 `grid.sync()` 从 2 次降到 1 次，虽然重复 pointwise
    backward，但总 step 继续降到 `256.554 ms`，与 cuDNN 差距约 `5.7%`。
11. persistent-state-local 尝试减少 state partial 的全局读回，总 step `256.759 ms`，
    与 state 基本持平略慢，说明下一步应继续压同步和 recurrent dot-product，而不是
    只减少这两个 partial 读。
12. split4 persistent-state 把总 step 降到 `185.327 ms`，timed_steps=10 顺序复测为
    `185.026 ms`，说明结果稳定且已经超过 cuDNN。
13. split8 persistent-state 继续降到 `153.787 ms`，timed_steps=10 顺序复测为
    `153.701 ms`。与 cuDNN timed_steps=10 复测 `251.414 ms` 相比，约 `1.64x`
    更快。
14. split16 persistent-state 仍有小幅收益，timed_steps=10 顺序复测为 `150.099 ms`，
    backward `71.911 ms`。相比 cuDNN timed_steps=10 复测约 `1.68x` 更快。
15. split32 persistent-state 退化到 `171.662 ms`，backward `93.285 ms`。这说明
    h256 上 split16 附近已经接近 CTA 分裂收益上限，继续增加 split 会放大重复
    pointwise、partial 规约和同步成本。
16. split16 global-gates 退化到 `177.040 ms`，backward `98.971 ms`。只计算一次
    gate 梯度的思路正确，但额外 grid sync 和全局 cache 读取成本超过收益。
17. split16 persistent-state gate-cache 把 forward 的 gate activation 作为 reserve
    space 保存，峰值显存从 `1.99 GB` 增到 `2.11 GB`，timed_steps=10 总 step
    降到 `143.204 ms`，backward 降到 `65.852 ms`。
18. split16 persistent-state gate-cache tiled 继续减少 backward shared memory 搬运，
    timed_steps=10 总 step 降到 `137.471 ms`，backward 降到 `59.966 ms`。相比
    cuDNN timed_steps=10 复测 `251.414 ms`，约 `1.83x` 更快，是当前最佳。GPU2
    复测为 `138.155 ms/step`，说明结果在空闲 A100 上稳定。
19. gate-cache parallel-update forward 退化到 `147.506 ms`，forward `80.966 ms`。
    这说明 CTA0 hidden update/cache 写入不是当前 forward 主瓶颈；失去 CTA0 shared
    partial 后，全局 partial 写读成本更高。
20. gate-cache CTA8 forward 在 `block_threads=704` 下无法常驻 cooperative launch；
    降到 `512/384` 后分别为 `161.569/163.604 ms`，明显慢于 CTA4。增加 CTA 数带来
    的 partial 写读、规约和同步成本超过了提高 SM 覆盖率的收益。
21. split32 gate-cache tiled 在 GPU2 上为 `146.225 ms/step`，backward `67.447 ms`，
    比 split16 tiled 慢，说明继续拆分 recurrent input-gradient 已超过收益上限。
22. grad-coeff-cache tiled 在 GPU2 上为 `139.136 ms/step`，峰值显存 `2.23 GB`。
    用 5H cache 预存导数系数没有超过 4H gate-cache tiled，说明额外 cache 带宽不值得。
23. gate-cache tiled CTA6 在 GPU2 上为 `148.113 ms/step`，forward `87.504 ms`。
    96 个 resident blocks 没有换来收益，非整除 k-tile 与 6-way partial 规约更贵。

## Nsight Systems 观察

新增 profiling 脚本：

```bash
CUDA_VISIBLE_DEVICES=1 tools/nsight/extract/systems-2024/opt/nvidia/nsight-systems/2024.2.3/target-linux-x64/nsys profile \
  --force-overwrite=true \
  --capture-range=cudaProfilerApi \
  --capture-range-end=stop \
  --trace=cuda,nvtx \
  --sample=none \
  --cpuctxsw=none \
  --output=results/nsys/a100_h256_fused_seq256_range \
  .venv/bin/python scripts/profile_a100_h256_step.py \
  --implementation a100_gru_h256_recurrent_kernel \
  --seq-len 256 \
  --warmup-steps 1
```

报告文件：

- `results/nsys/a100_h256_fused_seq256_range.nsys-rep`

`cuda_gpu_kern_sum` 关键结果：

| kernel | instances | total ms | avg us | time |
| --- | ---: | ---: | ---: | ---: |
| `a100_gru_h256_backward_step_kernel` | 256 | 22.752 | 88.873 | 76.1% |
| `a100_gru_forward_from_gates_cooperative_h256_shmem_kernel` | 1 | 4.818 | 4818.179 | 16.1% |
| `ampere_sgemm_32x128_nt` | 1 | 0.949 | 948.653 | 3.2% |

`cuda_api_sum` 显示：

- `cudaLaunchKernel`：290 次，总计约 `2.397 ms`
- `cuLaunchKernel`：257 次，总计约 `1.825 ms`
- `cuLaunchCooperativeKernel`：1 次

结论：当时 fused-step 路径的主要 GPU 时间已经集中在
`a100_gru_h256_backward_step_kernel`，不是 `weight_hh` 梯度 GEMM，也不是 optimizer。
下一轮如果继续追 cuDNN，需要减少每步 recurrent backward 的 kernel 时间，而不是再把
recurrent input-gradient 拆成更多小 kernel。

cooperative split2 的同类报告：

- `results/nsys/a100_h256_coop_split2_seq256_range.nsys-rep`

| kernel | instances | total ms | avg us | time |
| --- | ---: | ---: | ---: | ---: |
| `a100_gru_h256_backward_step_cooperative_split_kernel` | 256 | 5.936 | 23.189 | 48.1% |
| `a100_gru_forward_from_gates_cooperative_h256_shmem_kernel` | 1 | 4.935 | 4934.866 | 40.0% |

结论：cooperative split2 把每步 backward kernel 平均时间从约 `88.873 us`
降到约 `23.189 us`。端到端 benchmark 仍受 launch、forward 计时波动和 PyTorch
autograd 其它 op 影响，但 kernel 层面已经证明用 partial buffer 和 cooperative
grid-sync 换单步并行度是有效方向。

cooperative split2 cached 的同类报告：

- `results/nsys/a100_h256_coop_split2_cached_seq256_range.nsys-rep`

| kernel | instances | total ms | avg us | time |
| --- | ---: | ---: | ---: | ---: |
| `a100_gru_h256_backward_step_cooperative_split_cached_kernel` | 256 | 6.258 | 24.446 | 61.5% |
| `a100_gru_forward_from_gates_cooperative_h256_shmem_kernel` | 1 | 2.457 | 2456.949 | 24.1% |

cached 版本的单 kernel 时间略慢于 non-cached split2，但旧版端到端 seq8000 最好，
说明减少重复 pointwise 的收益在长序列 benchmark 中仍可能抵消额外 grid sync。

split16 persistent-state gate-cache 旧主线的同类报告：

- `results/nsys/a100_h256_gate_cache_seq256_range.nsys-rep`
- `results/nsys/a100_h256_gate_cache_seq8000_range.nsys-rep`

seq256 单 step 的 `cuda_gpu_kern_sum` 关键结果：

| kernel | instances | total ms | avg us | time |
| --- | ---: | ---: | ---: | ---: |
| `a100_gru_forward_from_gates_cooperative_h256_shmem_gate_cache_kernel` | 1 | 2.428 | 2428.280 | 49.7% |
| `a100_gru_h256_backward_sequence_cooperative_split16_gate_cache_state_kernel` | 1 | 2.007 | 2006.938 | 41.1% |
| `ampere_sgemm_32x128_nt` | 1 | 0.116 | 116.448 | 2.4% |

seq8000 单 step 的 `cuda_gpu_kern_sum` 关键结果：

| kernel | instances | total ms | avg us | time |
| --- | ---: | ---: | ---: | ---: |
| `a100_gru_forward_from_gates_cooperative_h256_shmem_gate_cache_kernel` | 1 | 76.544 | 76543.862 | 52.7% |
| `a100_gru_h256_backward_sequence_cooperative_split16_gate_cache_state_kernel` | 1 | 62.088 | 62088.446 | 42.7% |
| `ampere_sgemm_128x32_sliced1x4_nt` | 1 | 2.666 | 2665.648 | 1.8% |
| `ampere_sgemm_128x64_tn` | 1 | 1.781 | 1781.227 | 1.2% |

seq8000 的 launch 形状：

| kernel | grid | block | regs/thread | dyn smem |
| --- | ---: | ---: | ---: | ---: |
| forward gate-cache | `64x1x1` | `704x1x1` | 43 | 3072 |
| backward split16 gate-cache state | `256x1x1` | `256x1x1` | 32 | 3072 |

split16 persistent-state gate-cache tiled 当前主线的同类报告：

- `results/nsys/a100_h256_gate_cache_tiled_seq8000_range.nsys-rep`
- `results/nsys/a100_h256_gate_cache_tiled_seq8000_range.sqlite`

seq8000 单 step 的 `cuda_gpu_kern_sum` 关键结果：

| kernel | total ms | time |
| --- | ---: | ---: |
| `a100_gru_forward_from_gates_cooperative_h256_shmem_gate_cache_kernel` | 76.393 | 54.7% |
| `a100_gru_h256_backward_sequence_cooperative_split16_gate_cache_state_tiled_kernel` | 56.475 | 40.5% |
| 最大单个 GEMM | 2.668 | 1.9% |

seq8000 的 launch 形状：

| kernel | grid | block | regs/thread | dyn smem |
| --- | ---: | ---: | ---: | ---: |
| forward gate-cache | `64x1x1` | `704x1x1` | 43 | 3072 |
| backward split16 gate-cache tiled | `256x1x1` | `256x1x1` | 48 | 192 |

结论：当前主线的 GPU 时间已经高度集中在 forward gate-cache cooperative kernel 和
backward split16 gate-cache tiled persistent kernel。tiled backward 把动态 shared
memory 从 `3072` 字节降到 `192` 字节，并把 backward kernel 从旧主线约 `62.088 ms`
降到约 `56.475 ms`。`grad_weight_hh` 等 PyTorch/cuBLAS GEMM 在 seq8000 下最大的单
kernel 约 `2.668 ms`，不是主要瓶颈。继续优化应优先压两个自定义 cooperative kernel
内的 grid sync、partial buffer 读写和 recurrent dot-product 组织。

## 判断

本轮完成了 backward 的正确性闭环，并实现了三类有效优化：pointwise CUDA backward、
跨 time step batched GEMM，以及 fused backward step kernel。结果证明 h256
训练路径已经从纯 Python 公式原型进入可继续优化的 CUDA backward 路线。

当前状态：

1. **已完成 pointwise backward kernel**：每个 time step 计算 `grad_input_gates`、
   `grad_hidden_gates` 和 direct `grad_h_prev`。
2. **已完成 recurrent gate 重算聚合**：循环内小 `F.linear` 改成一次大 GEMM。
3. **已完成 weight_hh gradient 聚合**：`grad_hidden_gates.T @ h_prev` 改成跨 time
   step 的单次大 GEMM。
4. **已完成 fused backward step kernel**：`a100_gru_h256_recurrent_kernel` 路径把
   pointwise backward 与 recurrent input-gradient 合并。
5. **仍保留 PyTorch GEMM 的部分**：`grad_input_gates.T @ x` 和
   `grad_input_gates @ weight_ih` 继续交给 cuBLAS，这是合理基线。
6. **已完成两个负向实验**：`recompute_hidden_gates` 降低显存但过慢；
   tiled recurrent 减少 weight 读取但输给 fused backward step。
7. **已完成 partial-buffer 负向实验**：split recurrent 增加中间 partial buffer 和
   CTA 并行度，但 per-step 三次 launch 让长序列仍慢于 fused backward step。
8. **已完成 cooperative split 正向实验**：cooperative split2 在同一 kernel 内完成
   partial 和 reduce，显著降低 backward-step kernel 时间；cached-local split2 曾把
   seq8000 降到 `510.658 ms/step`。
9. **已完成 persistent backward 正向实验**：把 backward time loop 合并进单个
   cooperative kernel，seq8000 降到 `267.939 ms/step`，当前距离 cuDNN 复测
   `242.647 ms/step` 约 `10%`。
10. **已完成 persistent-state 正向实验**：每步只保留一次 `grid.sync()`，seq8000
    继续降到 `256.554 ms/step`，距离 cuDNN 复测约 `5.7%`。
11. **已完成 persistent-state-local 中性实验**：寄存器保留本 split 的 partial state
    后没有继续提升，seq8000 为 `256.759 ms/step`。
12. **已完成 split4 persistent-state 正向实验**：seq8000 降到 `185.026 ms/step`
    的 10-step 稳定结果，已经超过 cuDNN。
13. **已完成 split8 persistent-state 正向实验**：seq8000 降到 `153.701 ms/step`
    的 10-step 稳定结果。
14. **已完成 split16 persistent-state 正向实验**：seq8000 降到 `150.099 ms/step`
    的 10-step 稳定结果。
15. **已完成 split32 persistent-state 负向实验**：seq8000 为 `171.662 ms/step`，
    说明继续增加 CTA split 已经超过收益上限。
16. **已完成 split16 global-gates 负向实验**：seq8000 为 `177.040 ms/step`，说明
    额外 grid sync 和全局 gate cache 读写不值得替代重复 pointwise。
17. **已完成 split16 persistent-state gate-cache 正向实验**：用额外 reserve-space
    保存 gate activation，seq8000 timed_steps=10 降到 `143.204 ms/step`，峰值显存
    为 `2.11 GB`。
18. **已完成 split16 persistent-state gate-cache tiled 正向实验**：只把每个 split
    需要的 48 个 gate 梯度放入 shared memory，seq8000 timed_steps=10 降到
    `137.471 ms/step`，backward `59.966 ms`，是当前最佳。
19. **已完成 gate-cache parallel-update forward 负向实验**：forward 变慢到
    `80.966 ms`，说明分摊 hidden update/cache 写入不是主矛盾。
20. **已完成 gate-cache CTA8 forward 负向实验**：`704` threads 下超过 cooperative
    resident 上限，`512/384` threads 下仍明显慢于 CTA4 shmem。
21. **已完成 split32 gate-cache tiled 负向实验**：GPU2 复测为 `146.225 ms/step`，
    慢于 split16 tiled。
22. **已完成 grad-coeff-cache tiled 中性偏负实验**：GPU2 复测为 `139.136 ms/step`，
    峰值显存增到 `2.23 GB`，没有超过 4H gate-cache tiled。
23. **已完成 CTA6 forward 负向实验**：GPU2 复测为 `148.113 ms/step`，forward
    `87.504 ms`，说明单纯提高 resident block 数不够。

下一步应优先研究：

1. 继续优化当前最佳的 split16 persistent-state gate-cache tiled kernel：重点是
   backward split16 gate-cache tiled 的 recurrent dot-product、partial 规约和 grid
   sync 成本。
2. 继续优化 forward gate-cache kernel：Nsight Systems 显示它在 seq8000 下占
   `54.7%` GPU 时间，block size 扫描确认默认 `704` 仍是当前最佳；parallel-update、
   CTA8 和 CTA6 都已验证为负向，后续应关注 CTA4 shmem 结构内部的 recurrent
   dot-product 和 partial buffer 布局。
3. 降低或压缩 gate-cache reserve-space：当前峰值显存从 split16 的 `1.99 GB` 增到
   `2.11 GB`，但换来了 `137.471 ms/step` 的最佳速度。
4. 如果要继续追更高性能，需要考虑更接近 cuDNN reserve-space 的 forward/backward
   联合组织，而不是继续增加 split 数或额外全局同步。

当前最重要的结论是：h256 forward 优化已经成立，backward 也有明确下降曲线，并且
split16 persistent-state gate-cache tiled 已经超过 cuDNN 训练闭环。以 timed_steps=10
的顺序复测为准，自定义路径 `137.471 ms/step`，cuDNN `251.414 ms/step`，约 `1.83x`
更快。瓶颈已经从 Python/autograd 逐步 launch 外层组织，转向 split16 gate-cache
cooperative kernel 内部的 recurrent dot-product、剩余同步、partial 规约和显式保存中间量带来的
显存/带宽成本。
