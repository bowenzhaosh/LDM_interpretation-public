from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class DecisionInputs:
    instrument_pass: bool
    identifiable: bool
    mapping_pass: bool
    reconstruction_confirmation_pass: bool
    canary_intervals: tuple[tuple[float, float], tuple[float, float]]
    slope_intervals: tuple[tuple[float, float], tuple[float, float]]
    nrmse_upper: tuple[float, float]


@dataclass(frozen=True)
class Decision:
    code: str
    reason: str


def decide_primary(inputs: DecisionInputs) -> Decision:
    if not (
        len(inputs.canary_intervals)
        == len(inputs.slope_intervals)
        == len(inputs.nrmse_upper)
        == 2
    ):
        raise ValueError("the decision requires exactly two query-bank inputs")
    numeric = np.asarray(
        [
            *[value for interval in inputs.canary_intervals for value in interval],
            *[value for interval in inputs.slope_intervals for value in interval],
            *inputs.nrmse_upper,
        ],
        dtype=float,
    )
    if not np.isfinite(numeric).all():
        raise ValueError("decision inputs must be finite")
    if any(low > high for low, high in (*inputs.canary_intervals, *inputs.slope_intervals)):
        raise ValueError("decision intervals must be ordered")
    if not inputs.instrument_pass:
        return Decision("INCONCLUSIVE_INSTRUMENT", "an instrument fixture or oracle guard failed")
    if not inputs.identifiable:
        return Decision("INCONCLUSIVE_IDENTIFIABILITY", "the fixed panel lacked the required eligible groups")
    if not inputs.mapping_pass:
        return Decision("INCONCLUSIVE_MAPPING", "native predictions failed a preregistered mapping gate")
    if not all(lo >= -0.15 and lo <= 0.0 <= hi and hi <= 0.15 for lo, hi in inputs.canary_intervals):
        return Decision("INCONCLUSIVE_CANARY", "the continuation-swap canary did not collapse")

    below = all(hi < 0.8 for _lo, hi in inputs.slope_intervals)
    above = all(lo > 1.2 for lo, _hi in inputs.slope_intervals)
    if below or above:
        return Decision("INCOMPATIBLE_ON_TESTED_REGIME", "both banks place the primary slope outside the locked band")
    compatible = all(lo >= 0.8 and hi <= 1.2 for lo, hi in inputs.slope_intervals)
    compatible = compatible and all(value <= 0.35 for value in inputs.nrmse_upper)
    if compatible and not inputs.reconstruction_confirmation_pass:
        return Decision(
            "INCONCLUSIVE_RECONSTRUCTION",
            "coordinate gates passed but exact-evidence reconstruction did not confirm",
        )
    if compatible:
        return Decision("COMPATIBLE_ON_TESTED_REGIME", "both banks satisfy the locked compatibility gates")
    return Decision("INCONCLUSIVE", "the estimates do not meet a preregistered terminal rule")
