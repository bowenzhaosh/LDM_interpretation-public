"""Reconstruct the reported point contrasts from the frozen five-seed fixture."""
import json
from pathlib import Path
import numpy as np


def reconstruct(fixture_path):
    fixture = json.loads(Path(fixture_path).read_text())
    risks = np.asarray(fixture["seed_KL"], dtype=np.float64)
    assert risks.shape == (5, 3, 4)
    arms = fixture["arms"]
    delta = risks[:, :, arms.index("FreshY1")] - risks[:, :, arms.index("qz")]
    means = dict(zip(fixture["regimes"], delta.mean(0).tolist()))
    assert np.isclose(means["source"], 0.009754286649962193, rtol=0, atol=1e-14)
    assert round(means["coef"], 6) == -0.004131
    return {"contrast": "FreshY1 minus qz; positive means Fresh has higher KL",
            "point_estimates": means,
            "scope": "Reconstructs published means, not checkpoint predictions or confidence intervals.",
            "per_seed": delta.tolist(), "source_export_sha256": fixture["source_export_sha256"]}

