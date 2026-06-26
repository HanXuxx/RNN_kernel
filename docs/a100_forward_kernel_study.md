# A100/SM80 GRU forward-only 专用原型研究

## 目标

本轮在 `src/rnn_kernel/a100/` 下建立 A100 专用实现分支，继续优化前一轮
CUDA C/NVRTC forward-only 原型。该分支仍然遵守环境约束：

- 不修改系统 NVIDIA 驱动。
- 不依赖系统 `nvcc`。
- CUDA 源码放在真实 `.cu` 文件中，Python 只负责 NVRTC 编译、加载和 launch。
- 保持 fp32，不使用 TF32、bf16、AMP 或 fast-math。
- 当前只覆盖 A100/SM80；在 H200/SM90 上会显式报错。

## 已实现内容

新增文件：

- `src/rnn_kernel/a100/__init__.py`
- `src/rnn_kernel/a100/gru_forward.py`
- `src/rnn_kernel/a100/gru_forward_kernel.cu`
- `tests/test_a100_gru_forward.py`
- `scripts/benchmark_a100_forward.py`
- `scripts/profile_a100_forward.py`
- `scripts/run_nsight_a100_forward.sh`
- `scripts/extract_nsys_kernel_table.py`

新增接口：

- `a100_gru_forward_layer`
- `a100_gru_forward_from_gates`
- `a100_gru_forward_from_gates_subwarp`
- `a100_gru_forward_from_gates_fused`
- `a100_gru_forward_from_gates_fused_pingpong`
- `a100_gru_forward_from_gates_fused_specialized`
- `a100_gru_forward_from_gates_cooperative`
- `a100_gru_forward_from_gates_cooperative_h256`
- `a100_gru_forward_from_gates_cooperative_h256_parallel_update`
- `a100_gru_forward_from_gates_cooperative_h256_shmem`
- `a100_gru_forward_from_gates_cooperative_h256_shmem_gate_cache`
- `a100_gru_forward_from_gates_cooperative_h256_qwarp_shmem`
- `a100_gru_forward_from_gates_cooperative_h256_cached_shmem`
- `a100_gru_forward_layer_precompute_input`
- `a100_gru_forward_layer_precompute_input_subwarp`
- `a100_gru_forward_layer_precompute_input_fused`
- `a100_gru_forward_layer_precompute_input_fused_pingpong`
- `a100_gru_forward_layer_precompute_input_fused_specialized`
- `a100_gru_forward_layer_precompute_input_cooperative`
- `a100_gru_forward_layer_precompute_input_cooperative_h256`
- `a100_gru_forward_layer_precompute_input_cooperative_h256_parallel_update`
- `a100_gru_forward_layer_precompute_input_cooperative_h256_shmem`
- `a100_gru_forward_layer_precompute_input_cooperative_h256_qwarp_shmem`
- `a100_gru_forward_layer_precompute_input_cooperative_h256_cached_shmem`

支持范围：

- 单层 GRU
- 单向
- batch-first
- fp32
- `hidden_size <= 256`
- forward-only，不提供 backward

## 优化点

### 1. warp-level recurrent projection

上一版通用 CUDA/NVRTC kernel 是每个线程串行计算一个 hidden output 的 dot-product。
A100 版本改为：

- 每个 CUDA block 处理一个 batch item。
- block 内使用 1024 threads，也就是 32 个 warp。
- 每个 warp 负责一个 `gate/hid` recurrent dot-product。
- lane 维度并行遍历 hidden dimension，并用 `__shfl_down_sync` 做 warp 归约。

这改善了权重读取连续性，也减少了单线程串行 dot-product 的压力。

### 2. input projection 从 warp 路径移除

`input_dim=9` 很小，不值得为 input projection 占用 warp dot-product 调度槽。
因此第二版 A100 kernel 只用 warp 做 recurrent projection：

- hidden projection：warp-level dot-product。
- input projection：每个 hidden thread 在 pointwise 阶段串行计算 3 个小 dot。
- shared memory 从 `5H` 降到 `4H`。
- 每个 time step 的 warp dot 数从 `4H` 降到 `3H`。

### 3. cuBLAS 预计算 input gates 变体

另一个变体先用 PyTorch `F.linear` 预计算完整 input gates：

```text
[batch, seq, input] x [3 * hidden, input] -> [batch, seq, 3 * hidden]
```

然后 A100 kernel 只处理 recurrent projection 和 gate update。该变体用于判断 input
projection 是否仍是瓶颈。

### 4. block threads 调参

先测试过 `256/512/1024` threads per block。后续根据 Nsight Systems 看到 cuDNN 在
`hidden_size=130` 使用 `block=480`，又补测 `384/480/640/768/1024`。
当前这个 warp-dot kernel 上，`1024` 仍然最快，因此默认值保持 `1024`。这说明单纯
匹配 cuDNN 的 block size 不是主要矛盾，cuDNN 内部的 persistent recurrent 组织更高效。

### 5. sub-warp recurrent projection

Nsight Systems 显示主要时间仍在 recurrent kernel 内部。上一版是 32-lane warp 负责
一个 `gate/hid` dot-product。本轮继续实现两个变体：

- `subwarp_size=16`：一个 warp 拆成两个 half-warp，同时计算两个输出。
- `subwarp_size=8`：一个 warp 拆成四个 quarter-warp，同时计算四个输出。

实现时必须给 `__shfl_down_sync` 使用正确的子组 mask。早期版本使用 full-warp mask，
在最后一轮输出维循环中会因为部分 sub-warp 不参与而死锁；已修正为按 half/quarter
子组生成 mask。

当前结果：

- `hidden_size=128/130` 上 `subwarp_size=16` 最快。
- `hidden_size=160` 上 `subwarp_size=8` 略快。
- 综合目标断崖点 `hidden_size=130`，后续默认关注 `subwarp_size=16`。

### 6. fused r/z/n recurrent projection

subwarp16 仍然会先把 `3H` 个 recurrent gate 写到 shared memory，再由 pointwise 阶段
读出。fused 版本改成一个 half-warp 负责一个 hidden index，并在同一个循环里同时计算
r/z/n 三个 recurrent dot-product：

- 每个 half-warp 只读一遍 hidden vector。
- 不再写 `3H` 个 `hidden_gates` shared 中间量。
- lane 0 在三个归约结束后直接计算 GRU gate update。

这是目前对 `hidden_size=130` 最有效的 CTA 内优化。`seq_len=8000` 下，h130 从
subwarp16 的 `69.681 ms` 降到 `61.816 ms`。

### 7. ping-pong shared buffer

fused 版本每个 time step 仍有两次 `__syncthreads()`，并且需要把 `next_hidden` 拷回
`hidden`。ping-pong 版本使用两个 shared hidden buffer，当前 step 读一个、写另一个，
并直接写 output，目标是减少一次同步和 shared copy。

结果：

- h130 长序列有小幅收益：`61.816 ms` -> `61.258 ms`。
- h128/h160 略慢或基本持平。
- Nsight Systems 中 h130 短序列主 kernel 和普通 fused 基本同一量级，说明同步/拷贝
  不是当前最大瓶颈。

因此 ping-pong 只作为 h130 长序列的实验性微优化保留，不作为决定性路线。

### 8. fixed hidden-size 专用化

又实现了固定 `hidden_size=128/130/160` 的 fused kernel，让编译器看到常量 hidden
size，消除动态边界和部分索引计算。

结果有明显分化：

- h128 收益稳定：`50.450 ms` -> `47.063 ms`。
- h160 小幅收益：`82.895 ms` -> `80.746 ms`。
- h130 反而退化：`61.816 ms` -> `69.911 ms`。

h130 的第一次 fixed 版本更慢，Nsight Systems 显示主 kernel 约 `2579.830 us`。给 h130
内层 hidden loop 增加 `unroll 1` 后，主 kernel 降到约 `2023.310 us`，但仍慢于普通
fused 的约 `1764.832 us`。这说明对非 16 对齐的 `hidden_size=130`，简单固定常量会让
编译器生成不利的尾部循环代码；该方向不能作为 h130 主路径。

### 9. cooperative multi-CTA recurrent projection

新增 cooperative kernel，使用 `cooperative_groups::this_grid().sync()` 在单个 kernel
内部跨 CTA 同步。每个 batch item 分配多个 CTA：

- `ctas_per_batch` 个 CTA 按 hidden 维切分 recurrent dot-product。
- 每个 CTA 写 `3H` 个 partial recurrent gate 到全局临时 buffer。
- grid-level sync 后，由该 batch 的第一个 CTA 规约 partial gates、计算 GRU gate
  update、写 hidden state 和 output。
- 再次 grid-level sync 后进入下一个 time step。

Python launcher 使用 `cuLaunchCooperativeKernel`，并用
`cuOccupancyMaxActiveBlocksPerMultiprocessor` 和 SM 数检查 cooperative grid 是否能
常驻，避免 launch 规模超过硬件限制。当前目标形状下最佳配置是
`ctas_per_batch=4, block_threads=1024`。

结果：

- h130 只追平 fused/ping-pong：`61.262 ms` vs cooperative4 `61.350 ms`。
- h160 明显收益：`80.724 ms` -> `64.165 ms`。
- h192/h256 收益继续扩大；h256 上 cooperative4 `81.935 ms`，快于 cuDNN forward
  `109.961 ms`。

这说明 multi-CTA 方向是有效的，但收益门槛大约在 h160 之后。对 h130 来说，grid sync
和全局 partial buffer 的开销基本抵消了分摊 dot-product 的收益。

### 10. h256 专用 cooperative kernel

当前优化目标收敛到 `hidden_size=256`。在通用 cooperative4 的基础上新增多版 h256
专用 kernel：

- `cooperative_h256`：固定 `H=256, ctas_per_batch=4`，每个 CTA 固定处理 64 维
  k-tile，内层循环可展开。
- `cooperative_h256_shmem`：在 `cooperative_h256` 基础上，让 CTA0 的 partial gates
  留在 shared memory，避免 CTA0 partial 先写全局内存再读回。
- `cooperative_h256_parallel_update`：让 4 个 CTA 分摊 hidden update，验证 CTA0
  串行更新是否是瓶颈；实测没有超过 shmem 路径。
- `cooperative_h256_qwarp_shmem`：把 recurrent dot-product 从 half-warp 改成
  quarter-warp；实测变慢，说明更细 sub-warp 粒度增加的访存和每线程工作不划算。
- `cooperative_h256_cached_shmem`：每步先把 64 维 hidden k-tile 缓存到 shared
  memory；多出来的 block 同步抵消了 hidden 复用收益，实测慢于 shmem 路径。
- `cooperative_h256_shmem_gate_cache`：训练 backward 实验使用的 forward 变体，会额外
  保存 gate activation。它属于显存换速度实验，forward-only 主结果仍以 shmem 路径为准。

block size 扫描结果显示，h256 专用路径的最佳 `block_threads` 更新为 `704`，不是通用
cooperative 的 `1024`。当前 h256 长序列结果：

- cuDNN forward：`111.976 ms`
- 通用 cooperative4：`81.918 ms`
- h256 cooperative：`73.989 ms`
- h256 cooperative parallel update：`73.885 ms`
- h256 cooperative shmem：`73.540 ms`
- h256 cooperative qwarp shmem：`91.476 ms`
- h256 cooperative cached shmem：`81.120 ms`

因此 h256 当前最佳 forward-only 自定义实现比 cuDNN 快约 `1.52x`。

## Nsight CLI

Nsight CLI 安装在项目本地，不修改系统包和系统驱动：

- Nsight Compute：2024.2.1
- Nsight Systems：2024.2.3
- 加载脚本：`scripts/env_nsight.sh`
- 下载和解压目录：`tools/nsight/`，该目录被 git 忽略

曾尝试 2026.x Nsight CLI，但 Nsight Systems 2026.1.3 在当前 driver 550/CUDA 12.4
环境中生成 report 时出现 CUDA metadata 导入错误，因此默认切到 2024.x。

Nsight Compute 当前状态：

```text
ERR_NVGPUCTRPERM
```

这表示当前用户没有权限访问 GPU performance counters。该权限需要系统级 driver
配置，本项目没有擅自修改。因此本轮没有使用 NCU 的 occupancy、warp stall 和 memory
throughput 指标。

Nsight Systems 当前状态：

- CUDA timeline 可采集。
- CUDA kernel summary 可导出。
- SQLite 中可读取 kernel 名称、耗时、grid/block、寄存器和 shared memory。

可复现命令：

```bash
source .venv/bin/activate
bash scripts/run_nsight_a100_forward.sh
```

也可以单独导出已生成的 SQLite：

```bash
.venv/bin/python scripts/extract_nsys_kernel_table.py \
  profiles/nsys2024_a100_precompute_h130_s256.sqlite \
  --limit 8
```

## Nsight Systems 调度观察

采集配置：

- `batch_size=16`
- `seq_len=256`
- `input_dim=9`
- forward-only
- `profile_steps=1`
- warmup 和 profile 各一次，因此同一 kernel 通常出现 2 个 instances

### cuDNN hidden_size=128

主要 kernel：

| kernel | avg_us | grid | block | regs/thread | dynamic smem | static smem |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `RNN_blockPersist_fp_GRU<128>` | 288.066 | `16x1x1` | `384x1x1` | 160 | 0 | 3584 |
| `ampere_sgemm_128x32_tn` | 9.648 | `3x128x1` | `256x1x1` | 57 | 0 | 16384 |

### cuDNN hidden_size=130

主要 kernel：

| kernel | avg_us | grid | block | regs/thread | dynamic smem | static smem |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `RNN_blockPersist_fp_GRU<160>` | 806.420 | `16x1x1` | `480x1x1` | 128 | 0 | 4480 |
| `ampere_sgemm_32x128_tn` | 10.352 | `13x32x1` | `256x1x1` | 57 | 0 | 16384 |

这说明 cuDNN forward 在 `hidden_size=130` 并没有退回普通小 GEMM time loop，而是
继续使用 persistent GRU kernel，只是模板从 `128` 跳到 `160`，forward 时间约从
`288 us` 增加到 `806 us`。

### A100 precompute-input hidden_size=130

主要 kernel：

| kernel | avg_us | grid | block | regs/thread | dynamic smem | static smem |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `a100_gru_forward_from_gates_kernel` | 2754.847 | `16x1x1` | `1024x1x1` | 34 | 2080 | 0 |
| `ampere_sgemm_32x128_tn` | 47.888 | `13x32x1` | `256x1x1` | 57 | 0 | 16384 |

### A100 subwarp16 precompute-input hidden_size=130

主要 kernel：

| kernel | avg_us | grid | block | regs/thread | dynamic smem | static smem |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `a100_gru_forward_from_gates_half_warp_kernel` | 2085.045 | `16x1x1` | `1024x1x1` | 32 | 2080 | 0 |
| `ampere_sgemm_32x128_tn` | 47.713 | `13x32x1` | `256x1x1` | 57 | 0 | 16384 |

### A100 fused precompute-input hidden_size=130

主要 kernel：

| kernel | avg_us | grid | block | regs/thread | dynamic smem | static smem |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `a100_gru_forward_from_gates_fused_half_warp_kernel` | 1764.832 | `16x1x1` | `1024x1x1` | 48 | 1040 | 0 |
| `ampere_sgemm_32x128_tn` | 47.728 | `13x32x1` | `256x1x1` | 57 | 0 | 16384 |

### A100 fused ping-pong precompute-input hidden_size=130

主要 kernel：

| kernel | avg_us | grid | block | regs/thread | dynamic smem | static smem |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `a100_gru_forward_from_gates_fused_pingpong_half_warp_kernel` | 1776.080 | `16x1x1` | `1024x1x1` | 48 | 1040 | 0 |
| `ampere_sgemm_32x128_tn` | 48.016 | `13x32x1` | `256x1x1` | 57 | 0 | 16384 |

### A100 fixed h130 fused precompute-input hidden_size=130

主要 kernel：

| kernel | avg_us | grid | block | regs/thread | dynamic smem | static smem |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `a100_gru_forward_from_gates_fused_specialized_h130_kernel` | 2023.310 | `16x1x1` | `1024x1x1` | 35 | 1040 | 0 |
| `ampere_sgemm_32x128_tn` | 47.680 | `13x32x1` | `256x1x1` | 57 | 0 | 16384 |

### A100 cooperative4 precompute-input hidden_size=130

主要 kernel：

| kernel | avg_us | grid | block | regs/thread | dynamic smem | static smem |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `a100_gru_forward_from_gates_cooperative_kernel` | 1820.404 | `64x1x1` | `1024x1x1` | 64 | 0 | 0 |
| `ampere_sgemm_32x128_tn` | 47.697 | `13x32x1` | `256x1x1` | 57 | 0 | 16384 |

### A100 cooperative4 precompute-input hidden_size=256

主要 kernel：

| kernel | avg_us | grid | block | regs/thread | dynamic smem | static smem |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `a100_gru_forward_from_gates_cooperative_kernel` | 2435.967 | `64x1x1` | `1024x1x1` | 64 | 0 | 0 |
| `ampere_sgemm_128x64_tn` | 44.785 | `6x64x1` | `128x1x1` | 122 | 0 | 12800 |

### A100 h256 cooperative shmem precompute-input hidden_size=256

主要 kernel：

| kernel | avg_us | grid | block | regs/thread | dynamic smem | static smem |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `a100_gru_forward_from_gates_cooperative_h256_shmem_kernel` | 2200.947 | `64x1x1` | `704x1x1` | 43 | 3072 | 0 |
| `ampere_sgemm_128x64_tn` | 37.377 | `6x64x1` | `128x1x1` | 122 | 0 | 12800 |

结论：

1. 我们的 input precompute 不是主要瓶颈；cuBLAS input gate GEMM 只有约 `48 us`。
2. 自定义 A100 recurrent kernel 才是主要瓶颈，占 GPU kernel 时间约 `98%`。
3. cuDNN 和自定义 kernel 都是 `grid=16`，即每个 batch item 一个 CTA；但 cuDNN 的
   CTA 内部组织明显更高效。
4. subwarp16 把 recurrent 主 kernel 从约 `2755 us` 降到约 `2085 us`，说明减少
   warp 级输出循环和 shuffle 归约有实际收益。
5. fused half-warp 继续把主 kernel 降到约 `1765 us`，说明减少 hidden 重读和
   `hidden_gates` shared 往返是有效优化。
6. ping-pong 在短序列 profile 中没有降低主 kernel 时间，说明少一次同步并不是当前
   主瓶颈。
7. fixed h130 即使禁用内层 unroll 后仍慢于普通 fused，说明 h130 的非 16 对齐尾部
   循环需要更细的手写组织，不能只靠常量传播。
8. cooperative4 把 h130 grid 从 `16` 个 CTA 扩到 `64` 个 CTA，但主 kernel 仍约
   `1820 us`，略慢于 fused 的约 `1765 us`；h130 的计算量不足以抵消 grid sync 和
   partial buffer 开销。
9. h256 cooperative4 主 kernel 约 `2436 us`，长序列整体已经快于 cuDNN forward，
   说明 multi-CTA 在更大 hidden size 上是正确方向。
10. h256 shmem 专用 kernel 把主 kernel 继续降到约 `2201 us`，主要收益来自固定
    H/CTA 配置、`block=704` 调参、基址 hoist，以及 CTA0 partial 留在 shared memory。
11. 由于 NCU counters 未开放，暂时不能定量确认 warp stall/occupancy；但 Nsight
   Systems 已足以否定“只靠匹配 block size 能追上 cuDNN”的假设。

## 正确性验证

命令：

```bash
source .venv/bin/activate
source scripts/env.sh
.venv/bin/python -m pytest tests/test_a100_gru_forward.py -q
```

结果：

```text
39 passed
```

全量测试：

```bash
source .venv/bin/activate
source scripts/env.sh
.venv/bin/python -m pytest -q
```

结果：

```text
90 passed
```

测试覆盖：

- `hidden_size=16`
- `hidden_size=33`
- `hidden_size=130`
- cooperative 路径额外覆盖 `hidden_size=160/256`
- inline-input A100 kernel 与 `torch.nn.GRU` forward 对齐
- precompute-input A100 kernel 与 `torch.nn.GRU` forward 对齐
- subwarp8/subwarp16 precompute-input A100 kernel 与 `torch.nn.GRU` forward 对齐
- fused、ping-pong fused、fixed hidden-size fused kernel 与 `torch.nn.GRU` forward 对齐
- cooperative multi-CTA kernel 与 `torch.nn.GRU` forward 对齐
- h256 cooperative/h256 shmem cooperative kernel 与 `torch.nn.GRU` forward 对齐
- 不支持的 `hidden_size=257` 明确报错

当前测试容差：

```text
atol=2e-4
rtol=1e-4
```

cooperative h256 由于 fp32 规约顺序不同，测试使用：

```text
atol=3e-4
rtol=1e-4
```

实测 `ctas_per_batch=2, hidden_size=256` 的最大绝对误差约 `2.405e-4`，平均绝对误差
约 `1.326e-5`。

## 性能结果

硬件和软件：

- GPU：NVIDIA A100 80GB PCIe
- 实测设备：`CUDA_VISIBLE_DEVICES=1`，避免 GPU0 上其他进程干扰
- NVIDIA 驱动：550.163.01
- PyTorch：2.6.0+cu124
- PyTorch CUDA runtime：12.4
- cuDNN：9.1.0
- dtype：fp32
- TF32：关闭
- 通用 A100 kernel：`block_threads=1024`
- h256 专用 cooperative kernel：`block_threads=704`

### h256 seq_len=256

命令：

```bash
source .venv/bin/activate
source scripts/env.sh
CUDA_VISIBLE_DEVICES=1 .venv/bin/python scripts/benchmark_a100_forward.py \
  --hidden-sizes 256 \
  --batch-size 16 \
  --seq-len 256 \
  --input-dim 9 \
  --warmup-steps 3 \
  --timed-steps 10 \
  --block-threads 1024 \
  --subwarp-sizes 16 \
  --cooperative-ctas 4 \
  --cooperative-block-threads 1024 \
  --skip-generic \
  --output-csv results/a100_forward_h256_seq256.csv
```

结果：

| hidden_size | cuDNN ms | fused ms | cooperative4 ms | h256 coop ms | parallel update ms | h256 shmem ms | qwarp shmem ms | cached shmem ms | 最佳自定义 / cuDNN |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 256 | 4.056 | 4.047 | 2.509 | 2.266 | 2.248 | 2.244 | 2.837 | 2.482 | 1.81x faster |

### h256 seq_len=8000

命令：

```bash
source .venv/bin/activate
source scripts/env.sh
CUDA_VISIBLE_DEVICES=1 .venv/bin/python scripts/benchmark_a100_forward.py \
  --hidden-sizes 256 \
  --batch-size 16 \
  --seq-len 8000 \
  --input-dim 9 \
  --warmup-steps 1 \
  --timed-steps 5 \
  --block-threads 1024 \
  --subwarp-sizes 16 \
  --cooperative-ctas 4 \
  --cooperative-block-threads 1024 \
  --skip-generic \
  --output-csv results/a100_forward_h256_seq8000.csv
```

结果：

| hidden_size | cuDNN ms | fused ms | cooperative4 ms | h256 coop ms | parallel update ms | h256 shmem ms | qwarp shmem ms | cached shmem ms | 最佳自定义 / cuDNN |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 256 | 111.976 | 141.294 | 81.918 | 73.989 | 73.885 | 73.540 | 91.476 | 81.120 | 1.52x faster |

## 判断

A100 专用 forward-only 优化有效，已经快于 cuDNN forward。后续 h256 backward 训练
闭环已经在 `docs/a100_h256_backward_study.md` 中推进到超过 cuDNN 的自定义路径；当前
训练闭环最佳为 htile4 compact hoist row4 forward + split12 weight-shmem split0-keep
backward，seq8000 timed_steps=10 为 `94.286 ms/step`。本文档保留 forward-only 阶段的设计和
负向实验结论；训练闭环的最新数字以 backward 研究文档为准。

已经确认的有效点：

1. warp-level recurrent dot-product 比上一版线程串行 dot-product 明显更快。
2. 把 input projection 从 warp 路径移除后，A100 kernel 进一步提速。
3. 通用 single-CTA warp-dot kernel 上 `block_threads=1024` 仍然最好；h256
   cooperative 专用路径上 `block_threads=704` 最好。
4. cuBLAS 预计算 input gates 对 `hidden_size=130` 有小幅收益，但不是决定性瓶颈。
5. Nsight Systems 确认 input gate GEMM 不是主要耗时，自定义 recurrent kernel 才是
   主要瓶颈。
6. cooperative multi-CTA 对 h256 明确有效：通用 cooperative4 从 cuDNN `111.976 ms`
   降到 `81.918 ms`。
7. h256 固定配置和 `block_threads=704` 继续降低开销：`81.918 ms` -> `73.989 ms`。
8. h256 shmem partial 和基址 hoist 进一步降低到 `73.540 ms`，当前 forward-only
   最佳实现比 cuDNN 快约 `1.52x`。
9. parallel update、quarter-warp shmem、hidden-tile cached shmem 都没有超过
   half-warp shmem 路径，后续不作为主线。

仍然存在的核心限制：

1. h256 cooperative kernel 每个 time step 仍需要两次 grid sync。
2. 另外 3 个 CTA 的 partial gates 仍经过全局内存，CTA0 shared-memory 优化只消除了
   CTA0 自己的 partial 全局写读。
3. 当前实现仍然是 sub-warp dot-product，不是 tiled GEMM，也没有使用 tensor core。
4. Nsight Compute performance counters 未开放，当前还缺少 occupancy 和 warp stall
   的硬件计数器证据。

因此，当前 A100 forward-only 分支在 h256 上已经达到继续投入 backward 的门槛。
h256 backward 正确性原型、pointwise CUDA backward、跨 time step batched GEMM、
fused backward step kernel、cooperative split backward step、persistent-state gate-cache
tiled backward，以及 htile4 compact hoist row4 forward 均已完成，见
`docs/a100_h256_backward_study.md`。

## 下一步

继续 A100/h256 路线时，建议按优先级做：

1. 继续优化 h256 forward partial buffer：尝试让 CTA1/2/3 的 partial gates 使用更紧凑
   布局或 vectorized store/load，减少全局内存往返。
2. 评估 h256 tile-by-hidden 的多 CTA 组织，避免 k 维 partial reduce，但需要解决下一
   time step hidden 可见性。
3. 为 h256 forward 增加更完整的 benchmark 维度：不同 batch size、seq_len 和输入维度。
4. backward 已完成第一轮有效优化：pointwise CUDA kernel、recurrent gate 聚合、
   `weight_hh` 梯度聚合和 fused backward step。recompute hidden gates 与 tiled
   recurrent、外部分离 split recurrent partial-buffer 路径都没有超过 fused step。
   当前 h256 训练主线已经推进到 htile4 compact hoist row4 forward + split12
   weight-shmem split0-keep backward，seq8000 timed_steps=10 最佳为
   `94.286 ms/step`，forward `59.349 ms`，backward `34.450 ms`。下一步需要同时优化 row4 的
   寄存器/occupancy 权衡、forward partial buffer 写读路径、weight-shmem backward
   的 shared-memory tile 细节、partial 规约读写，以及 reserve-space 带来的显存/带宽成本。本轮已验证
   gate-cache parallel-update、CTA8 forward、CTA6 forward、row4 forward weight-shmem、
   row4 forward hidden-shmem、qwarp forward、row3 forward、split8 backward、split24 backward、
   split32 gate-cache tiled 和 grad-coeff-cache tiled 都没有超过当前主线；htile8-compact
   也已验证为负向边界。
   htile4-compact-hoist-row4-256 是当前长序列最快 forward 分支，htile2 的 seq_len
   扫描显示 1024 之后收益更稳定。
5. 如果保持 NVRTC 路线，继续把 CUDA 源码放在 `.cu` 文件中；如果补齐项目内 `nvcc`，
   再迁移到 PyTorch C++ extension。
