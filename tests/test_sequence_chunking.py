import torch
import torch.nn.functional as F

from rnn_benchmark import RNNBenchmarkModel


def _clone_model(cell_type: str, chunk_len: int) -> RNNBenchmarkModel:
    return RNNBenchmarkModel(
        cell_type=cell_type,
        input_dim=3,
        hidden_size=5,
        num_layers=2,
        sequence_chunk_len=chunk_len,
    )


def test_sequence_chunking_matches_full_bptt_gru() -> None:
    torch.manual_seed(123)
    full_model = _clone_model("GRU", chunk_len=0)
    chunk_model = _clone_model("GRU", chunk_len=4)
    chunk_model.load_state_dict(full_model.state_dict())

    x_full = torch.randn(2, 11, 3, requires_grad=True)
    x_chunk = x_full.detach().clone().requires_grad_(True)
    y = torch.randn(2, 11)

    full_loss = F.mse_loss(full_model(x_full), y)
    chunk_loss = F.mse_loss(chunk_model(x_chunk), y)
    full_loss.backward()
    chunk_loss.backward()

    assert torch.allclose(full_loss, chunk_loss, atol=1e-6, rtol=1e-6)
    assert torch.allclose(x_full.grad, x_chunk.grad, atol=1e-6, rtol=1e-6)
    for full_param, chunk_param in zip(full_model.parameters(), chunk_model.parameters()):
        assert torch.allclose(full_param.grad, chunk_param.grad, atol=1e-6, rtol=1e-6)


def test_sequence_chunking_matches_full_bptt_lstm() -> None:
    torch.manual_seed(456)
    full_model = _clone_model("LSTM", chunk_len=0)
    chunk_model = _clone_model("LSTM", chunk_len=3)
    chunk_model.load_state_dict(full_model.state_dict())

    x_full = torch.randn(2, 10, 3, requires_grad=True)
    x_chunk = x_full.detach().clone().requires_grad_(True)
    y = torch.randn(2, 10)

    full_loss = F.mse_loss(full_model(x_full), y)
    chunk_loss = F.mse_loss(chunk_model(x_chunk), y)
    full_loss.backward()
    chunk_loss.backward()

    assert torch.allclose(full_loss, chunk_loss, atol=1e-6, rtol=1e-6)
    assert torch.allclose(x_full.grad, x_chunk.grad, atol=1e-6, rtol=1e-6)
    for full_param, chunk_param in zip(full_model.parameters(), chunk_model.parameters()):
        assert torch.allclose(full_param.grad, chunk_param.grad, atol=1e-6, rtol=1e-6)

