from __future__ import annotations

import numpy as np
from tests.factories import credit_frame

from creditmon import simulation
from creditmon.schema import TARGET, validate_batch


def test_control_month_is_untouched() -> None:
    frame = credit_frame()
    out = simulation.apply_drift(frame, 0.0, np.random.default_rng(0))
    assert out.equals(frame)


def test_full_shock_moves_the_documented_variables() -> None:
    frame = credit_frame(4000)
    out = simulation.apply_drift(frame, 1.0, np.random.default_rng(0))
    assert out["BILL_AMT1"].mean() > frame["BILL_AMT1"].mean() * 1.4
    assert out["PAY_AMT1"].mean() < frame["PAY_AMT1"].mean() * 0.7
    assert out["PAY_0"].mean() > frame["PAY_0"].mean()
    assert out["AGE"].mean() < frame["AGE"].mean() - 3
    assert out[TARGET].mean() > frame[TARGET].mean()


def test_drifted_batches_still_pass_the_contract() -> None:
    batches = simulation.simulate_production(credit_frame(3000), seed=1)
    assert list(batches) == simulation.MONTHS
    for month, batch in batches.items():
        validate_batch(batch, name=month)  # drift != dado inválido


def test_batches_are_disjoint_slices_of_the_holdout() -> None:
    batches = simulation.simulate_production(credit_frame(3000), seed=1)
    ids = np.concatenate([b["ID"].to_numpy() for b in batches.values()])
    assert len(ids) == len(set(ids)) == 3000
