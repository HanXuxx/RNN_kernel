# A100GRUH256 独立库

这是一个可直接复制走的 A100/SM80 专用 GRU 库，固定支持 `hidden_size=256`、fp32、
单层、单向、`batch_first=True` 的训练场景。

## 目录内容

- `__init__.py` / `gru.py`：独立 Python API，外部模块直接引用这里。
- `pyproject.toml` / `MANIFEST.in`：自包含打包元数据，解压后可直接安装本目录。
- `kernels/a100_gru_h256_sm80.cubin`：预编译 A100/SM80 kernel，运行时直接加载。
- `kernels/a100_gru_h256_kernels.cu`：prod 版最小 CUDA 源码，便于重新生成 cubin。
- `scripts/check_env.py`：环境检查。
- `scripts/functional_test.py`：功能正确性测试。
- `scripts/benchmark.py`：训练 step benchmark。
- `scripts/demo.py`：最小使用示例。
- `scripts/build_cubin.py`：构建期生成 cubin。
- `scripts/package_wheel.sh`：构建 wheel。
- `scripts/package_archive.sh`：生成只包含 `a100_gru_h256` 目录的 tar.gz。
- `docs/`：API、打包和限制说明。

## 运行环境

运行环境不需要 `nvcc`，也不需要 NVRTC/NVCC wheel。需要：

- NVIDIA A100/SM80
- NVIDIA driver 支持当前 PyTorch CUDA 版本
- CUDA 版 PyTorch
- `cuda-python`

安装运行依赖：

```bash
python -m pip install -r a100_gru_h256/requirements-runtime.txt
```

如果目标机没有 CUDA 版 PyTorch，请先按该机器的 CUDA/驱动环境安装 CUDA 版 PyTorch；
不要误装 CPU-only PyTorch。

## 压缩目录分发

在源机器上：

```bash
cd /home/xuh/RNN_kernel/prod
mkdir -p a100_gru_h256/dist
tar --exclude='a100_gru_h256/dist' -czf a100_gru_h256/dist/a100_gru_h256-0.1.0.tar.gz a100_gru_h256
```

或使用脚本：

```bash
bash a100_gru_h256/scripts/package_archive.sh
```

在目标机器上：

```bash
tar -xzf a100_gru_h256-0.1.0.tar.gz
python -m pip install -r a100_gru_h256/requirements-runtime.txt
python a100_gru_h256/scripts/check_env.py --run-smoke
python a100_gru_h256/scripts/functional_test.py
```

如果目标环境已经安装好 CUDA 版 PyTorch 和 `cuda-python`，可以跳过
`requirements-runtime.txt`。

## 无安装直接引用

可以不执行 `pip install ./a100_gru_h256`。要求是把 `a100_gru_h256` 的父目录加入
`PYTHONPATH`，注意不是把 `a100_gru_h256` 目录本身加入 `PYTHONPATH`。

例如目录结构为：

```text
/opt/a100_gru_lib/
  a100_gru_h256/
    __init__.py
    gru.py
    kernels/
```

运行：

```bash
export PYTHONPATH=/opt/a100_gru_lib:${PYTHONPATH:-}
python - <<'PY'
from a100_gru_h256 import A100GRUH256
print(A100GRUH256)
PY
```

也可以在业务代码里显式加入父目录：

```python
import sys
from pathlib import Path

sys.path.insert(0, "/opt/a100_gru_lib")

from a100_gru_h256 import A100GRUH256, from_torch_gru
```

如果把库放进项目内，推荐结构是：

```text
my_project/
  vendor/
    a100_gru_h256/
  train.py
```

`train.py` 中：

```python
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "vendor"))

from a100_gru_h256 import A100GRUH256, from_torch_gru
```

## 使用方式

安装或配置 `PYTHONPATH` 后，可以直接：

```python
from a100_gru_h256 import A100GRUH256, from_torch_gru

fast_gru = A100GRUH256(input_size=5).cuda()

# 或者从 torch.nn.GRU 复制权重
fast_gru = from_torch_gru(torch_gru)
output, h_n = fast_gru(x)

# 推理路径不分配 backward gate cache
with torch.no_grad():
    output, h_n = fast_gru(x)

# 也可以显式调用 forward-only 路径
output, h_n = fast_gru.forward_inference(x)
```

也可以安装 wheel：

```bash
bash a100_gru_h256/scripts/package_wheel.sh
python -m pip install a100_gru_h256/dist/a100_gru_h256-0.1.0-py3-none-any.whl --no-deps
```

## 检查和测试

```bash
python a100_gru_h256/scripts/check_env.py --run-smoke
python a100_gru_h256/scripts/functional_test.py
python a100_gru_h256/scripts/benchmark.py --timed-steps 10 --include-inference
```

## 支持范围

- `hidden_size=256`
- `dtype=torch.float32`
- `batch_first=True`
- `num_layers=1`
- 单向 GRU
- `bias=True`
- A100/SM80

不满足上述条件时应回退到 `torch.nn.GRU`。
