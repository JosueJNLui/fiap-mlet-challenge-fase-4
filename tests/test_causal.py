from __future__ import annotations

from tests.factories import credit_frame

from creditmon import causal
from creditmon.train import fit_model, model_features


def test_scenarios_switch_one_mechanism_at_a_time() -> None:
    sc = causal.scenarios()
    assert len(sc) == 2 + 2 * len(causal.MECHANISMS)
    only_c1 = sc["só C1 inadimplência silenciosa"][1]
    assert only_c1.silent_default_prob > 0
    assert only_c1.inflation_bill == only_c1.unemployment_share == only_c1.new_profile_share == 0
    without_d1 = sc["sem D1 inflação"][1]
    assert without_d1.inflation_bill == without_d1.income_squeeze_pay == 0
    assert without_d1.silent_default_prob > 0


def test_concept_drift_alone_is_invisible_to_data_drift() -> None:
    reference, holdout = credit_frame(3000, seed=0), credit_frame(1500, seed=1)
    model, _ = fit_model(reference, model_features(reference.columns), seed=0)

    result = causal.ablation(model, reference, holdout, seed=0)
    rows = {r["scenario"]: r for r in result["ablation"]}
    base, c1 = rows[causal.NO_SHOCK], rows["só C1 inadimplência silenciosa"]

    assert base["drift_share"] == c1["drift_share"] == 0  # C1 só muda o rótulo
    assert c1["default_rate"] > base["default_rate"]
    assert rows[causal.FULL_SHOCK]["drift_share"] > 0
    assert [d["severity"] for d in result["dose_response"]] == [0, 0.2, 0.4, 0.6, 0.8, 1]
    lo, hi = result["dose_response"][0]["batch_auc_ci95"]
    assert lo <= hi
    assert "Dose-resposta" in causal.causal_markdown(result)
