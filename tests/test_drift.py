from __future__ import annotations

import numpy as np
import pandas as pd
from tests.factories import credit_frame

from creditmon import drift, simulation
from creditmon.schema import ID_COLUMN, TARGET

FEATURES = [c for c in credit_frame(1).columns if c not in {ID_COLUMN, TARGET}]


def test_psi_is_zero_for_identical_distributions() -> None:
    values = pd.Series(np.random.default_rng(0).normal(size=5000))
    assert drift.psi(values, values) < 1e-6


def test_psi_grows_with_the_shift() -> None:
    rng = np.random.default_rng(0)
    ref = pd.Series(rng.normal(0, 1, 5000))
    assert drift.psi(ref, pd.Series(rng.normal(0.1, 1, 5000))) < 0.10
    assert drift.psi(ref, pd.Series(rng.normal(1.0, 1, 5000))) > 0.25


def test_psi_handles_discrete_variables() -> None:
    ref = pd.Series([0] * 80 + [1] * 20)
    cur = pd.Series([0] * 40 + [1] * 60)
    assert drift.psi(ref, cur) > 0.25


def test_only_the_shifted_feature_is_flagged() -> None:
    ref, cur = credit_frame(3000, seed=1), credit_frame(3000, seed=2)
    cur["LIMIT_BAL"] = cur["LIMIT_BAL"] * 3
    table = drift.feature_drift(ref, cur, FEATURES)
    assert drift.summarize(table)["drifted_features"] == ["LIMIT_BAL"]


def test_simulated_shock_is_detected_and_control_month_is_clean() -> None:
    pool = credit_frame(6000, seed=3)
    batches = simulation.simulate_production(pool, seed=3)
    first = drift.summarize(drift.feature_drift(pool, batches["2026-01"], FEATURES))
    last = drift.summarize(drift.feature_drift(pool, batches["2026-06"], FEATURES))
    assert first["n_drifted"] == 0
    assert last["n_drifted"] >= 2


def test_ks_separation_is_one_for_a_perfect_score() -> None:
    assert drift.ks_separation([0, 0, 1, 1], np.array([0.1, 0.2, 0.8, 0.9])) == 1.0


def test_report_shows_predictions_per_month(tmp_path) -> None:
    table = drift.feature_drift(credit_frame(500, seed=1), credit_frame(500, seed=2), FEATURES)
    table["psi_evidently"] = table["psi"]
    path = tmp_path / "t.csv"
    table.to_csv(path, index=False)
    record = {
        "month": "2026-01",
        "drift": drift.summarize(table),
        "prediction_psi": 0.01,
        "performance": {"mean_score": 0.31, "approval_rate": 0.7, "roc_auc": 0.78,
                        "default_rate": 0.22},
        "table_path": str(path),
    }  # fmt: skip
    md = drift.drift_markdown([record], {"roc_auc": 0.78})
    assert "Score médio" in md and "Aprovação" in md and "70.0%" in md


def test_alerts_flag_each_rule_and_assign_blocked_batch_to_its_month() -> None:
    # EDUCATION já falhava no holdout: disparidade pré-existente não vira alerta mensal.
    baseline_fairness = {"SEX": 0.95, "AGE": 0.86, "EDUCATION": 0.62, "MARRIAGE": None}
    record = {
        "month": "2026-06",
        "drift": {"critical_features": ["PAY_0", "LIMIT_BAL"], "dataset_drift": True},
        "performance": {"roc_auc": 0.70},
        "fairness": {"SEX": 0.97, "AGE": 0.75, "EDUCATION": 0.5, "MARRIAGE": 0.5},
    }
    assert drift.alerts(record, 0.78, ["2026-06-corrupted"], baseline_fairness) == {
        "psi_critical": 2,
        "dataset_drift": 1,
        "auc_drop": 1,
        "contract_blocked": 1,
        "fairness": 1,
    }
    calm = {**record, "month": "2026-01", "performance": {"roc_auc": 0.77},
            "drift": {"critical_features": [], "dataset_drift": False},
            "fairness": {"SEX": 0.9, "AGE": 0.85, "EDUCATION": 0.6, "MARRIAGE": None}}
    assert sum(drift.alerts(calm, 0.78, ["2026-06-corrupted"], baseline_fairness).values()) == 0
