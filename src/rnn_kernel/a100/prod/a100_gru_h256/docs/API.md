# API 说明

## `A100GRUH256`

```python
from a100_gru_h256 import A100GRUH256

gru = A100GRUH256(input_size=5).cuda()
output, h_n = gru(x, hx=None)
```

参数：

- `input_size`：输入维度。
- `hidden_size`：必须为 `256`。
- `num_layers`：必须为 `1`。
- `batch_first`：必须为 `True`。
- `bias`：必须为 `True`。

输入：

- `x`：`[batch, seq, input_size]`，CUDA fp32 tensor。
- `hx`：可选，`[1, batch, 256]`，CUDA fp32 tensor。

输出：

- `output`：`[batch, seq, 256]`
- `h_n`：`[1, batch, 256]`

训练时普通 `forward()` 会使用带 `gate_cache` 的路径，以支持 backward。处于
`torch.no_grad()` 或 `torch.inference_mode()` 时，普通 `forward()` 会自动切到
forward-only no-cache 路径，避免分配 backward 缓存。

## `forward_inference`

```python
output, h_n = gru.forward_inference(x, hx=None)
```

显式运行 forward-only no-cache 推理路径。该路径不构建 autograd graph，也不分配
`gate_cache`，适合只做推理或验证输出的场景。

## `from_torch_gru`

```python
from a100_gru_h256 import from_torch_gru

fast_gru = from_torch_gru(torch_gru)
```

该函数会检查 `torch_gru` 是否满足支持范围，并复制 `weight_ih_l0`、`weight_hh_l0`、
`bias_ih_l0`、`bias_hh_l0`。

## `is_supported_gru`

```python
from a100_gru_h256 import is_supported_gru

if is_supported_gru(torch_gru):
    fast_gru = from_torch_gru(torch_gru)
```

## `is_a100_available`

```python
from a100_gru_h256 import is_a100_available

if is_a100_available():
    ...
```
