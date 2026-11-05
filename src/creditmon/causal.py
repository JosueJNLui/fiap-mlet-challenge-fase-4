"""Análise causal da degradação: ablação dos mecanismos de drift da simulação.

Como o cenário de produção é simulado, a causa da degradação pode ser medida por
**intervenção**, não só inferida por correlação. Sobre o holdout inteiro (6.000 clientes,
intensidade 1,0) rodamos:

* **isolado**: só um mecanismo ligado (efeito do mecanismo sozinho);
* **remoção**: choque completo com um mecanismo desligado (quanto da degradação some
  sem ele);
* **dose-resposta**: choque completo nas intensidades dos 6 meses simulados.

Todos os cenários usam a mesma seed. Como ``apply_drift`` sorteia os mesmos números
aleatórios em qualquer cenário, a única diferença entre dois cenários é o mecanismo
ligado ou desligado (números aleatórios comuns). O cenário "sem choque" tem todos os
mecanismos zerados, mas mantém o ruído individual de ``D1`` (±5% nas faturas e
pagamentos), igual em todos os cenários.

Métricas por cenário, contra a referência de treino: drift share e PSI do score (o que
o monitor de data drift enxerga), ROC AUC, recall, aprovação e default real (o que o
negócio sente). ``make causal`` grava ``reports/causal/causal_summary.json`` e
``causal_report.md``.
"""

from __future__ import annotations

import json
from dataclasses import fields, replace
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from creditmon.config import Settings
from creditmon.drift import feature_columns, feature_drift, performance, psi, score, summarize
from creditmon.observability import logger, setup_logging, stage
from creditmon.schema import TARGET
from creditmon.simulation import MONTHS, Scenario, apply_drift, severity

MECHANISMS = {
    "D1 inflação": ("inflation_bill", "income_squeeze_pay"),
    "D2 desemprego": ("unemployment_share",),
    "D3 novo perfil": ("new_profile_share",),
    "C1 inadimplência silenciosa": ("silent_default_prob",),
}
NO_SHOCK, FULL_SHOCK = "sem choque", "choque completo"
BATCH_SIZE, BOOTSTRAP_DRAWS = 1000, 300


def scenarios() -> dict[str, tuple[str, Scenario]]:
    """Cenários da ablação: nome -> (tipo, parâmetros)."""
    full = Scenario()
    off = replace(full, **{f.name: 0.0 for f in fields(Scenario)})
    out = {NO_SHOCK: ("controle", off), FULL_SHOCK: ("controle", full)}
    for name, params in MECHANISMS.items():
        out[f"só {name}"] = ("isolado", replace(off, **{p: getattr(full, p) for p in params}))
    for name, params in MECHANISMS.items():
        out[f"sem {name}"] = ("remoção", replace(full, **dict.fromkeys(params, 0.0)))
    return out


def batch_auc_interval(scored: pd.DataFrame, seed: int) -> list[float]:
    """IC 95% (bootstrap) da ROC AUC de um lote mensal de ``BATCH_SIZE`` clientes."""
    rng = np.random.default_rng(seed)
    y, p = scored[TARGET].to_numpy(), scored["score"].to_numpy()
    aucs = []
    for _ in range(BOOTSTRAP_DRAWS):
        idx = rng.choice(len(scored), size=min(BATCH_SIZE, len(scored)), replace=False)
        aucs.append(roc_auc_score(y[idx], p[idx]))
    return [round(float(q), 4) for q in np.percentile(aucs, [2.5, 97.5])]


def evaluate(
    model,
    reference_scored: pd.DataFrame,
    frame: pd.DataFrame,
    features: list[str],
    *,
    interval_seed: int | None = None,
) -> dict[str, Any]:
    """O que o monitor enxerga (drift) e o que o negócio sente (performance) num lote."""
    scored = score(model, frame)
    drift = summarize(feature_drift(reference_scored, scored, features))
    perf = performance(scored)
    return {
        "drift_share": drift["drift_share"],
        "n_drifted": drift["n_drifted"],
        "prediction_psi": round(psi(reference_scored["score"], scored["score"]), 4),
        **{k: round(perf[k], 4) for k in ("roc_auc", "recall", "approval_rate", "default_rate")},
        **({} if interval_seed is None else {
            "batch_auc_ci95": batch_auc_interval(scored, interval_seed)
        }),
    }


def ablation(
    model, reference: pd.DataFrame, holdout: pd.DataFrame, seed: int
) -> dict[str, list[dict[str, Any]]]:
    """Roda a ablação dos mecanismos e a curva dose-resposta do choque completo."""
    features = feature_columns(model)
    reference_scored = score(model, reference)

    def run(s: float, sc: Scenario, interval_seed: int | None = None) -> dict[str, Any]:
        frame = apply_drift(holdout, s, np.random.default_rng(seed), sc)
        return evaluate(model, reference_scored, frame, features, interval_seed=interval_seed)

    rows = [{"scenario": n, "kind": k, **run(1.0, sc)} for n, (k, sc) in scenarios().items()]
    dose = [
        {"month": m, "severity": round(severity(i), 2), **run(severity(i), Scenario(), seed)}
        for i, m in enumerate(MONTHS)
    ]
    return {"ablation": rows, "dose_response": dose}


def causal_markdown(result: dict[str, list[dict[str, Any]]]) -> str:
    """Relatório: ablação com deltas vs. "sem choque" e curva dose-resposta."""
    rows = {r["scenario"]: r for r in result["ablation"]}
    base, full = rows[NO_SHOCK], rows[FULL_SHOCK]
    lines = [
        "# Análise causal da degradação (ablação dos mecanismos)",
        "",
        "Holdout inteiro (6.000 clientes), intensidade 1,0, mesma seed em todos os cenários. "
        "Δ AUC = ROC AUC do cenário menos a do cenário sem choque. Na remoção, "
        "\"AUC recuperada\" é a fração da queda do choque completo que some ao desligar o "
        "mecanismo.",
        "",
        "| Cenário | Tipo | Drift share | PSI do score | ROC AUC | Δ AUC | AUC recuperada "
        "| Recall | Aprovação | Default real |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    total_drop = base["roc_auc"] - full["roc_auc"]
    for r in result["ablation"]:
        recovered = (
            f"{(r['roc_auc'] - full['roc_auc']) / total_drop:.0%}"
            if r["kind"] == "remoção" and total_drop > 0
            else ""
        )
        lines.append(
            f"| {r['scenario']} | {r['kind']} | {r['drift_share']:.0%} | "
            f"{r['prediction_psi']:.3f} | {r['roc_auc']:.4f} | "
            f"{r['roc_auc'] - base['roc_auc']:+.4f} | {recovered} | {r['recall']:.1%} | "
            f"{r['approval_rate']:.1%} | {r['default_rate']:.1%} |"
        )
    lines += [
        "",
        "## Dose-resposta do choque completo",
        "",
        "Mesmas intensidades dos lotes mensais, mas sobre o holdout inteiro: separa o efeito "
        f"do choque da variação amostral dos lotes de {BATCH_SIZE} clientes. \"IC 95% "
        f"num lote\" é o intervalo bootstrap ({BOOTSTRAP_DRAWS} sorteios) da ROC AUC de um "
        f"lote mensal de {BATCH_SIZE} clientes na mesma intensidade.",
        "",
        "| Mês | Intensidade | Drift share | PSI do score | ROC AUC | IC 95% num lote "
        "| Aprovação | Default real |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for d in result["dose_response"]:
        lo, hi = d["batch_auc_ci95"]
        lines.append(
            f"| {d['month']} | {d['severity']:.0%} | {d['drift_share']:.0%} | "
            f"{d['prediction_psi']:.3f} | {d['roc_auc']:.4f} | {lo:.3f} a {hi:.3f} | "
            f"{d['approval_rate']:.1%} | "
            f"{d['default_rate']:.1%} |"
        )
    return "\n".join(lines) + "\n"


def run(settings: Settings) -> dict[str, Any]:
    from creditmon.data import load_reference
    from creditmon.drift import load_model

    reference, holdout = load_reference(settings)
    result = ablation(load_model(), reference, holdout, settings.seed)
    out_dir = settings.paths.reports / "causal"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "causal_summary.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False)
    )
    (out_dir / "causal_report.md").write_text(causal_markdown(result), encoding="utf-8")
    logger.info(f"relatórios em {out_dir}/")
    return result


def main() -> int:
    import mlflow

    from creditmon.config import load_settings
    from creditmon.tracking import log_stage, tracked_run

    setup_logging()
    settings = load_settings()
    with tracked_run(settings, "causal") as tracking:
        with stage("causal") as st:
            result = run(settings)
            st.update(scenarios=len(result["ablation"]))
        if tracking:
            mlflow.log_artifacts(str(settings.paths.reports / "causal"), "causal")
            log_stage(st)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
