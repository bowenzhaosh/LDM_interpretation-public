import numpy as np
import torch

from pathlib import Path

from pfn_dag_verify.generative import generate_group
from pfn_dag_verify.model import (
    PFNModel,
    configure_determinism,
    load_registered_checkpoint,
    predict_probabilities,
)
from pfn_dag_verify.provenance import derive_seed
from pfn_dag_verify.query_bank import CANDIDATE_QUERIES, select_symmetric_query_bank
from pfn_dag_verify.registry import expanded_checkpoint_record, load_checkpoint_registry


def test_model_probability_shape_determinism_and_row_permutation():
    configure_determinism(5)
    model = PFNModel(d_model=16, d_ff=32, n_heads=4, n_layers=1).eval()
    rng = np.random.default_rng(6)
    contexts = rng.normal(size=(3, 20, 2)).astype(np.float32)
    queries = np.array([-2.0, -0.5, 0.5, 2.0], dtype=np.float32)
    first = predict_probabilities(model, contexts, queries, batch_size=2)
    second = predict_probabilities(model, contexts, queries, batch_size=3)
    np.testing.assert_allclose(first, second, atol=1e-7, rtol=0)
    permutation = np.random.default_rng(8).permutation(contexts.shape[1])
    permuted = predict_probabilities(model, contexts[:, permutation], queries, batch_size=3)
    np.testing.assert_allclose(first, permuted, atol=1e-6, rtol=0)
    np.testing.assert_allclose(first.sum(axis=2), 1.0, atol=1e-6, rtol=0)
    assert first.shape == (3, 4, 100)
    assert all(not parameter.requires_grad or torch.isfinite(parameter).all() for parameter in model.parameters())


def test_query_selection_is_symmetric_deterministic_and_uses_tie_break():
    rng = np.random.default_rng(21)
    f0 = rng.gamma(2, 1, size=(40, len(CANDIDATE_QUERIES), 100))
    f1 = rng.gamma(2, 1, size=f0.shape)
    f0 /= f0.sum(axis=2, keepdims=True)
    f1 /= f1.sum(axis=2, keepdims=True)
    first = select_symmetric_query_bank(f0, f1)
    second = select_symmetric_query_bank(f0, f1)
    np.testing.assert_array_equal(first.queries, second.queries)
    np.testing.assert_array_equal(first.queries, -first.queries[::-1])
    assert len(first.queries) == 8
    assert np.all(np.diff(first.objective_trace) >= -1e-12)


def test_commit_seed_derivation_is_labeled_and_stable():
    commit = "0123456789abcdef" * 4
    assert derive_seed(commit, "groups") == derive_seed(commit, "groups")
    assert derive_seed(commit, "groups") != derive_seed(commit, "bootstrap")


def test_registered_real_checkpoint_loads_and_is_permutation_invariant():
    root = Path(__file__).resolve().parents[1]
    registry = load_checkpoint_registry(root / "config" / "checkpoint_registry.json")
    record = expanded_checkpoint_record(registry, seed=0, step=12000)
    model = load_registered_checkpoint(record)
    group = generate_group(np.random.default_rng(99), n_continuations=2)
    context = np.concatenate([group.core, group.reference])[None, :, :]
    queries = np.array([-4.0, -2.5, 2.5, 4.0], dtype=np.float32)
    original = predict_probabilities(model, context, queries)
    permutation = np.random.default_rng(100).permutation(context.shape[1])
    shuffled = predict_probabilities(model, context[:, permutation], queries)
    np.testing.assert_allclose(original, shuffled, atol=1e-6, rtol=0)
