# 构建和打包

## 源目录和导出方式

本包的源目录是：

```text
src/rnn_kernel/a100/prod/a100_gru_h256
```

顶层 `prod/a100_gru_h256` 是导出结果，不应手动修改。从仓库根目录执行：

```bash
python src/rnn_kernel/a100/prod/scripts/export_a100_gru_h256.py
```

该命令会复制包源码，并在包目录内构建：

```text
prod/a100_gru_h256/dist/a100_gru_h256-0.1.0-py3-none-any.whl
prod/a100_gru_h256/dist/a100_gru_h256-0.1.0.tar.gz
```

导出的 `prod/a100_gru_h256` 目录本身也是自包含包目录，可单独压缩、解压和安装。

## 构建 cubin

运行环境不需要编译，但发布包前需要在构建环境生成一次 cubin：

```bash
python -m pip install -r a100_gru_h256/requirements-build.txt
python a100_gru_h256/scripts/build_cubin.py
```

该脚本使用 Python wheel 中的 NVRTC/NVCC 头文件和库，不要求系统安装全局 `nvcc`。

## 打包 wheel

```bash
bash a100_gru_h256/scripts/package_wheel.sh
```

生成物：

```text
a100_gru_h256/dist/a100_gru_h256-0.1.0-py3-none-any.whl
```

wheel 内必须包含：

```text
a100_gru_h256/kernels/a100_gru_h256_sm80.cubin
```

可以用以下命令检查：

```bash
python - <<'PY'
import zipfile
from pathlib import Path
wheel = next(Path("a100_gru_h256/dist").glob("a100_gru_h256-*.whl"))
with zipfile.ZipFile(wheel) as zf:
    print("a100_gru_h256/kernels/a100_gru_h256_sm80.cubin" in zf.namelist())
PY
```

## 打包可复制目录

只分发 `a100_gru_h256` 目录时，在源机器执行：

```bash
cd /home/xuh/RNN_kernel/prod
bash a100_gru_h256/scripts/package_archive.sh
```

或直接：

```bash
tar -czf a100_gru_h256-0.1.0.tar.gz a100_gru_h256
```

目标机器解压后执行：

```bash
tar -xzf a100_gru_h256-0.1.0.tar.gz
python -m pip install -r a100_gru_h256/requirements-runtime.txt
python -m pip install ./a100_gru_h256 --no-deps
python a100_gru_h256/scripts/check_env.py --run-smoke
```
