"""Scientific invariants, using the actual finite prior and original functions."""
import hashlib
import json
from pathlib import Path
import unittest

import numpy as np
import torch

from pfn_dag_verify.corrected_models import ModelConfig, PFN
from pfn_dag_verify.pilot_shared import production_quadrature
from pfn_dag_verify.evaluation import reconstruct
from pfn_dag_verify.training import API, prepare

ROOT = Path(__file__).resolve().parents[1]
from pfn_dag_verify.target_construction import forward, loss_for, quadrature, query_components


class ScientificIdentities(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)
        cls.config = json.loads((ROOT / "configs/reviewer/tiny.json").read_text())
        cls.prepared = prepare(cls.config)

    def test_pinned_portable_scientific_files(self):
        record = json.loads((ROOT / "manifests/scientific-provenance.json").read_text())
        for path, expected in record["portable_files_sha256"].items():
            self.assertEqual(hashlib.sha256((ROOT / "src" / (path if path.startswith("pfn_dag_verify/") else "pfn_dag_verify/" + path)).read_bytes()).hexdigest(), expected)

    def test_target_means_covariances_and_common_state_gradient_covariances(self):
        X, _, _, _, _, _, world = self.prepared
        a, q, Q = query_components(API, world, X[0])[1][0]
        self.assertEqual(q.shape, (48, 100))
        A = q - Q
        B = (A.T * a) @ A
        W = np.diag(Q) - (q.T * a) @ q
        np.testing.assert_allclose(B + W, np.diag(Q) - np.outer(Q, Q), atol=1e-14)
        np.testing.assert_allclose(a @ A, 0, atol=1e-14)
        one_hot = np.eye(100)
        mass = (a[:, None] * q).ravel()
        signed = (Q + one_hot[None, :, :] - q[:, None, :]).reshape(-1, 100)
        np.testing.assert_allclose(signed.sum(-1), 1, atol=1e-14)
        self.assertTrue((signed < 0).any())
        np.testing.assert_allclose(mass @ signed, Q, atol=1e-14)
        # J is common across interventions: no separately trained models enter.
        rng = np.random.default_rng(812)
        J = rng.normal(size=(100, 7))
        logits = rng.normal(size=100)
        p = np.exp(logits - logits.max()); p /= p.sum()
        for targets, weights, covariance in (
                (Q[None], np.ones(1), np.zeros((100, 100))),
                (q, a, B),
                (signed, mass, W),
                (one_hot, Q, B + W)):
            expected_loss = weights @ (-targets @ np.log(p))
            self.assertAlmostEqual(float(expected_loss), float(-Q @ np.log(p)), places=12)
            gradients = (p - targets) @ J
            centered = gradients - weights @ gradients
            observed = (centered.T * weights) @ centered
            np.testing.assert_allclose(observed, J.T @ covariance @ J, atol=1e-12)

    def test_signed_loss_and_autograd_match_direct_signed_target(self):
        _, Q, q, y, _, _, _ = self.prepared
        target = Q, q[:, 0, 0], y[:, 0, 0]
        torch.manual_seed(18)
        logits = torch.randn(len(Q), 7, 100, dtype=torch.float64, requires_grad=True)
        actual = loss_for(logits, target, "signed", torch.float64)
        signed = (torch.tensor(target[0]) + torch.nn.functional.one_hot(
                  torch.tensor(target[2].astype(np.int64)), 100) - torch.tensor(target[1]))
        direct = -(signed * logits.log_softmax(-1)).sum(-1).mean()
        np.testing.assert_allclose(signed.sum(-1).numpy(), 1, atol=2e-6)
        self.assertTrue(torch.allclose(actual, direct, atol=1e-12, rtol=1e-12))
        g_actual = torch.autograd.grad(actual, logits, retain_graph=True)[0]
        g_direct = torch.autograd.grad(direct, logits)[0]
        self.assertTrue(torch.allclose(g_actual, g_direct, atol=1e-12, rtol=1e-12))

    def test_query_labels_do_not_enter_model(self):
        X = self.prepared[0]
        altered = X.copy()
        altered[:, 20:, 2] += 1000
        torch.manual_seed(17)
        model = PFN(ModelConfig(name="base", d=3)).eval()
        with torch.no_grad():
            self.assertTrue(torch.equal(forward(model, X, "cpu"), forward(model, altered, "cpu")))

    def test_quadrature_adapter_and_fixed_input_generation(self):
        from pfn_dag_verify.corrected_world import world_metadata
        metadata = world_metadata(self.prepared[-1], self.config["world_seed"])
        self.assertEqual(metadata["accepted_atom_count"], 8)
        self.assertEqual(len(metadata["library_sha256"]), 64)
        for actual, expected in zip(quadrature(32), production_quadrature()):
            np.testing.assert_array_equal(actual, expected)
        from pfn_dag_verify.corrected_models import gen_batch
        from pfn_dag_verify.target_construction import input_key
        key, count = input_key("pilot", 0)
        repeated = gen_batch(self.prepared[-1], np.random.default_rng(np.random.SeedSequence(key)),
                             count, 20, 7)[0].astype(np.float32)
        np.testing.assert_array_equal(repeated[:self.config["train_rows"]], self.prepared[0])

    def test_reported_point_contrasts(self):
        result = reconstruct(ROOT / "examples/reviewer/fixtures/withheld_seed_risks.json")
        self.assertGreater(result["point_estimates"]["source"], 0)
        self.assertLess(result["point_estimates"]["coef"], 0)


if __name__ == "__main__":
    unittest.main()
