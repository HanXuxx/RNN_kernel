# A100 h256 GRU 生产试用封装源目录

该目录是 A100GRUH256 产品包的唯一源目录。顶层 `/home/xuh/RNN_kernel/prod`
由这里导出生成，不应手动维护。

目录职责：

- `a100_gru_h256/`：外部可直接复制和安装的独立 Python 包源码。
- `a100_gru_h256/pyproject.toml` / `a100_gru_h256/MANIFEST.in`：独立包的打包元数据。
- `scripts/export_a100_gru_h256.py`：复制 `a100_gru_h256/` 到顶层 `prod/` 并构建 wheel/tar.gz。
- `gru.py` / `kernels/`：保留给仓库内部 `rnn_kernel.a100.prod` 稳定入口使用。

内部稳定入口把当前最快 A100 h256 训练路径封装起来，避免外部代码直接依赖实验
implementation string 或大量 kernel 开关。prod 运行时不 import `rnn_kernel.a100.gru_autograd`
或 `rnn_kernel.a100.gru_forward`，只加载随包发布的预编译 `sm80` cubin。

当前固定组合：

- forward：逐层 `htile4 compact hoist row4 K1 gate-cache`
- forward-only：逐层 `htile4 compact hoist row4 K1 no-cache`
- backward：逐层 `split6 persistent-state gate-cache tiled weight-shmem split0-keep unroll8`
- 布局准备：`hidden-prev pack`
- block threads：`256`

该 prod 包同步的是当前最优非 fused 路径。真正 fused 多层 kernel 仍保留在
`src/rnn_kernel/a100` 实验区，不作为本次产品默认路径。

支持范围：

- A100/SM80
- `torch.float32`
- `input_size=1..16`
- `hidden_size=256`
- `num_layers=1..4`
- 单向 GRU
- `batch_first=True`
- `bias=True`
- `dropout=0.0`

示例：

```python
import torch
from rnn_kernel.a100.prod import A100GRU, from_torch_gru

gru = torch.nn.GRU(16, 256, num_layers=2, batch_first=True).cuda()
fast_gru = from_torch_gru(gru)

x = torch.randn(16, 8000, 16, device="cuda")
output, h_n = fast_gru(x)

with torch.no_grad():
    output, h_n = fast_gru(x)

output, h_n = fast_gru.forward_inference(x)
```

该封装复用当前最快算法组合，但 CUDA 源码、cubin 加载和 autograd/module 入口已经独立；
外部调用只依赖 `rnn_kernel.a100.prod` 的稳定 API。

## 导出独立产品包

从仓库根目录执行：

```bash
python src/rnn_kernel/a100/prod/scripts/export_a100_gru_h256.py
```

该命令会重建：

```text
prod/a100_gru_h256/
prod/a100_gru_h256/dist/a100_gru_h256-0.1.0-py3-none-any.whl
prod/a100_gru_h256/dist/a100_gru_h256-0.1.0.tar.gz
```

只复制源码、不构建 wheel 和 tar.gz：

```bash
python src/rnn_kernel/a100/prod/scripts/export_a100_gru_h256.py --no-build-artifacts
```

## 打包和无 nvcc 运行

构建发布包前，在开发环境生成 cubin：

```bash
python scripts/build_a100_prod_cubin.py
```

该命令需要开发环境安装 `cuda-python`、`nvidia-cuda-nvrtc-cu12`、`nvidia-cuda-nvcc-cu12`
和 `nvidia-cuda-runtime-cu12`，但不依赖系统全局 `nvcc`。

运行环境不需要安装 `nvcc`，也不需要 NVRTC/NVCC wheel；需要：

- NVIDIA driver 支持 A100/SM80
- CUDA 版 PyTorch
- `cuda-python`
- wheel 内包含 `a100_gru_h256/kernels/a100_gru_h256_sm80.cubin`

`pyproject.toml` 已把 `*.cubin` 和 `*.cu` 配为 package data，正常构建 wheel 时会打入包内。
