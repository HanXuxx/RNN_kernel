from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from .triton_gru_pointwise import triton_gru_pointwise
except ImportError:  # pragma: no cover - 允许没有安装 Triton 时使用 torch backend。
    triton_gru_pointwise = None


class CustomGRU(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        pointwise_backend: str = "torch",
        batch_first: bool = True,
    ) -> None:
        super().__init__()
        if not batch_first:
            raise ValueError("CustomGRU only supports batch_first=True.")
        if pointwise_backend not in {"torch", "triton"}:
            raise ValueError(f"Unsupported pointwise backend: {pointwise_backend}")

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.pointwise_backend = pointwise_backend
        self.batch_first = batch_first

        for layer in range(num_layers):
            layer_input_size = input_size if layer == 0 else hidden_size
            weight_ih = nn.Parameter(torch.empty(3 * hidden_size, layer_input_size))
            weight_hh = nn.Parameter(torch.empty(3 * hidden_size, hidden_size))
            bias_ih = nn.Parameter(torch.empty(3 * hidden_size))
            bias_hh = nn.Parameter(torch.empty(3 * hidden_size))
            setattr(self, f"weight_ih_l{layer}", weight_ih)
            setattr(self, f"weight_hh_l{layer}", weight_hh)
            setattr(self, f"bias_ih_l{layer}", bias_ih)
            setattr(self, f"bias_hh_l{layer}", bias_hh)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            nn.init.uniform_(weight, -stdv, stdv)

    def forward(
        self,
        x: torch.Tensor,
        hx: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if hx is None:
            hx = x.new_zeros(self.num_layers, x.size(0), self.hidden_size)

        layer_input = x
        next_hidden = []
        for layer in range(self.num_layers):
            output, hidden = self._forward_layer(layer_input, hx[layer], layer)
            layer_input = output
            next_hidden.append(hidden)
        return layer_input, torch.stack(next_hidden, dim=0)

    def _forward_layer(
        self,
        x: torch.Tensor,
        hidden: torch.Tensor,
        layer: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        weight_ih = getattr(self, f"weight_ih_l{layer}")
        weight_hh = getattr(self, f"weight_hh_l{layer}")
        bias_ih = getattr(self, f"bias_ih_l{layer}")
        bias_hh = getattr(self, f"bias_hh_l{layer}")

        batch_size, seq_len, input_size = x.shape
        input_gates = F.linear(
            x.reshape(batch_size * seq_len, input_size),
            weight_ih,
            bias_ih,
        ).view(batch_size, seq_len, 3 * self.hidden_size)
        input_gates = input_gates.transpose(0, 1).contiguous()

        outputs = []
        for step in range(seq_len):
            hidden_gates = F.linear(hidden, weight_hh, bias_hh)
            hidden = self._pointwise(input_gates[step], hidden_gates, hidden)
            outputs.append(hidden)
        output = torch.stack(outputs, dim=1)
        return output, hidden

    def _pointwise(
        self,
        input_gates: torch.Tensor,
        hidden_gates: torch.Tensor,
        hidden: torch.Tensor,
    ) -> torch.Tensor:
        if (
            self.pointwise_backend == "triton"
            and input_gates.is_cuda
            and input_gates.dtype == torch.float32
            and triton_gru_pointwise is not None
        ):
            return triton_gru_pointwise(input_gates, hidden_gates, hidden)

        i_r, i_z, i_n = input_gates.chunk(3, dim=1)
        h_r, h_z, h_n = hidden_gates.chunk(3, dim=1)
        reset_gate = torch.sigmoid(i_r + h_r)
        update_gate = torch.sigmoid(i_z + h_z)
        new_gate = torch.tanh(i_n + reset_gate * h_n)
        return new_gate + update_gate * (hidden - new_gate)


def copy_from_torch_gru(custom_gru: CustomGRU, torch_gru: nn.GRU) -> None:
    if torch_gru.bidirectional:
        raise ValueError("Bidirectional GRU is not supported.")
    if not torch_gru.batch_first:
        raise ValueError("Only batch_first=True GRU is supported.")
    if custom_gru.num_layers != torch_gru.num_layers:
        raise ValueError("num_layers mismatch.")
    if custom_gru.hidden_size != torch_gru.hidden_size:
        raise ValueError("hidden_size mismatch.")
    if custom_gru.input_size != torch_gru.input_size:
        raise ValueError("input_size mismatch.")

    with torch.no_grad():
        for layer in range(custom_gru.num_layers):
            for name in ("weight_ih", "weight_hh", "bias_ih", "bias_hh"):
                target = getattr(custom_gru, f"{name}_l{layer}")
                source = getattr(torch_gru, f"{name}_l{layer}")
                target.copy_(source)
