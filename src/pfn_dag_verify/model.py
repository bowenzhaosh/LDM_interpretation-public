from collections import OrderedDict

import numpy as np
import torch
from torch import nn

from .constants import BASE_MODEL_CONFIG, N_BINS, NULL_TOKEN
from .registry import verify_file_record


class PFNModel(nn.Module):
    """Checkpoint-compatible e21/dose PFN with no positional embedding."""

    def __init__(self, d_model: int, d_ff: int, n_heads: int, n_layers: int):
        super().__init__()
        self.point_embed = nn.Linear(2, d_model)
        self.query_embed = nn.Linear(1, d_model)
        self.token_embed = nn.Embedding(3, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model,
            n_heads,
            d_ff,
            batch_first=True,
            dropout=0.0,
        )
        self.transformer = nn.TransformerEncoder(layer, n_layers)
        self.out_head = nn.Linear(d_model, N_BINS)

    def forward(self, context: torch.Tensor, queries: torch.Tensor, token: torch.Tensor):
        context_embedding = self.point_embed(context)
        token_embedding = self.token_embed(token).unsqueeze(1)
        outputs = []
        for query_index in range(queries.shape[1]):
            query_embedding = self.query_embed(queries[:, query_index, :]).unsqueeze(1)
            encoded = self.transformer(
                torch.cat([token_embedding, context_embedding, query_embedding], dim=1)
            )
            outputs.append(self.out_head(encoded[:, -1, :]))
        return torch.stack(outputs, dim=1)


def configure_determinism(seed: int = 0) -> None:
    torch.manual_seed(int(seed))
    torch.use_deterministic_algorithms(True)
    torch.set_float32_matmul_precision("highest")
    torch.backends.mha.set_fastpath_enabled(True)
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        # PyTorch permits setting inter-op threads only before parallel work starts.
        if torch.get_num_interop_threads() != 1:
            raise


def state_schema(state_dict: OrderedDict | dict) -> list[dict]:
    return [
        {
            "key": str(key),
            "shape": list(value.shape),
            "dtype": str(value.dtype),
        }
        for key, value in state_dict.items()
    ]


def load_registered_checkpoint(record: dict) -> PFNModel:
    verify_file_record(record)
    config = record.get("model_config")
    if config != BASE_MODEL_CONFIG:
        raise ValueError("checkpoint model_config does not match the locked base architecture")
    state = torch.load(
        record.get("_resolved_path", record["path"]),
        map_location="cpu",
        weights_only=True,
    )
    if not isinstance(state, dict) or not state:
        raise ValueError("checkpoint must contain a nonempty state dictionary")
    actual_schema = state_schema(state)
    if actual_schema != record.get("state_schema"):
        raise ValueError("checkpoint state schema mismatch")
    if any(not torch.isfinite(value).all() for value in state.values()):
        raise ValueError("checkpoint contains a non-finite tensor")
    model = PFNModel(**config)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


def predict_probabilities(
    model: PFNModel,
    contexts: np.ndarray,
    queries: np.ndarray,
    *,
    batch_size: int = 64,
) -> np.ndarray:
    contexts = np.asarray(contexts, dtype=np.float32)
    queries = np.asarray(queries, dtype=np.float32)
    if contexts.ndim != 3 or contexts.shape[2] != 2 or not np.isfinite(contexts).all():
        raise ValueError("contexts must be finite with shape (batch, rows, 2)")
    if queries.ndim != 1 or queries.size < 2 or not np.isfinite(queries).all():
        raise ValueError("queries must be a finite one-dimensional bank")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    output = np.empty((len(contexts), len(queries), N_BINS), dtype=np.float32)
    with torch.inference_mode():
        for start in range(0, len(contexts), batch_size):
            stop = min(len(contexts), start + batch_size)
            context_tensor = torch.from_numpy(contexts[start:stop])
            query_tensor = torch.from_numpy(
                np.broadcast_to(queries[None, :, None], (stop - start, len(queries), 1)).copy()
            )
            token = torch.full((stop - start,), NULL_TOKEN, dtype=torch.long)
            logits = model(context_tensor, query_tensor, token)
            probabilities = torch.softmax(logits, dim=-1).cpu().numpy().astype(np.float32)
            if not np.isfinite(probabilities).all():
                raise FloatingPointError("model produced non-finite probabilities")
            output[start:stop] = probabilities
    if not np.allclose(output.sum(axis=2), 1.0, atol=1e-6, rtol=0):
        raise FloatingPointError("model probabilities do not normalize")
    return output
