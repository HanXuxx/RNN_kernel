# 限制和回退策略

当前库是 A100/SM80 + GRU h256 的专用实现，不是通用 RNN 替代品。

必须满足：

- GPU compute capability 为 `sm_80`
- GRU input size 为 `1..16`
- GRU hidden size 为 `256`
- fp32
- `num_layers=1..4`
- 单向
- `batch_first=True`
- `bias=True`
- `dropout=0.0`

建议外部模块在替换前使用：

```python
from a100_gru_h256 import is_a100_available, is_supported_gru, from_torch_gru

if is_a100_available() and is_supported_gru(torch_gru):
    gru = from_torch_gru(torch_gru)
else:
    gru = torch_gru
```

如果未来扩展到 H200，应新增独立 cubin 和运行时选择逻辑，不要复用 `sm80` cubin。
