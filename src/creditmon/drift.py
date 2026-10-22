"""Detecção estatística de drift: PSI + Kolmogorov-Smirnov e relatórios do Evidently AI.

Dois métodos independentes, de propósito:

* **PSI** (Population Stability Index): padrão de mercado em risco de crédito, diz
  *quanto* a distribuição mudou.
* **KS** (Kolmogorov-Smirnov, duas amostras): teste de hipótese, diz *se* a mudança é
  estatisticamente significativa.

Uma feature é marcada com drift quando PSI >= 0,25 (crítico), ou quando PSI >= 0,10
**e** p-valor KS < 0,05. Exigir as duas evidências evita o falso positivo típico do KS
em lotes grandes, que acusa diferenças irrelevantes.

O Evidently roda em paralelo (``DataDriftPreset`` com PSI + ``ClassificationPreset``) e
gera os HTMLs em ``reports/drift/``. O PSI dele é gravado ao lado do nosso em
``feature_drift_<mes>.csv`` como validação cruzada da implementação.

Uso: ``make drift`` (depois de ``make train`` e ``make simulate``). Além dos relatórios,
grava as predições do baseline para cada lote em
``data/processed/production/predictions_<mes>.csv`` (``ID``, ``score``, ``prediction``).
"""

from __future__ import annotations

import argparse
import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

from creditmon.config import Settings
from creditmon.schema import CATEGORICAL_COLUMNS, ID_COLUMN, TARGET
from creditmon.train import METRICS_PATH, MODEL_PATH

EPS = 1e-4
SCORE, PREDICTION = "score", "prediction"


@dataclass(frozen=True)
class Thresholds:
    """Limiares de drift e de alerta (valores usuais em risco de crédito)."""

    psi_warning: float = 0.10  # 0,10-0,25: mudança moderada, investigar
    psi_critical: float = 0.25  # > 0,25: mudança significativa, ação obrigatória
    ks_pvalue: float = 0.05
    drift_share_critical: float = 0.30  # fração de features com drift = dataset drift
    auc_drop_critical: float = 0.05  # queda absoluta de AUC vs. baseline
    default_rate_shift: float = 0.05  # variação da inadimplência real vs. holdout


THRESHOLDS = Thresholds()


# ── Estatísticas ─────────────────────────────────────────────────────────────


def psi(reference: pd.Series, current: pd.Series, bins: int = 10) -> float:
    """Population Stability Index.

    Variáveis discretas (<= 12 valores distintos na referência) usam as próprias
    categorias; contínuas usam os decis da referência.
    """
    ref = pd.Series(reference).dropna().to_numpy(dtype=float)
    cur = pd.Series(current).dropna().to_numpy(dtype=float)
    if len(ref) == 0 or len(cur) == 0:
        return float("nan")
    if len(np.unique(ref)) <= 12:
        values = np.union1d(np.unique(ref), np.unique(cur))
        ref_pct = np.array([(ref == v).mean() for v in values])
        cur_pct = np.array([(cur == v).mean() for v in values])
    else:
        edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
        edges[0], edges[-1] = -np.inf, np.inf
        ref_pct = np.histogram(ref, edges)[0] / len(ref)
        cur_pct = np.histogram(cur, edges)[0] / len(cur)
    ref_pct = np.clip(ref_pct, EPS, None)
    cur_pct = np.clip(cur_pct, EPS, None)
    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


def psi_severity(value: float, th: Thresholds = THRESHOLDS) -> str:
    if value >= th.psi_critical:
        return "critical"
    if value >= th.psi_warning:
        return "warning"
    return "ok"


def feature_drift(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    features: list[str],
    th: Thresholds = THRESHOLDS,
) -> pd.DataFrame:
    """Tabela de drift por feature (PSI, KS, médias), ordenada pelo PSI."""
    rows = []
    for col in features:
        p = psi(reference[col], current[col])
        ks = ks_2samp(reference[col].dropna(), current[col].dropna())
        drift = p >= th.psi_critical or (p >= th.psi_warning and ks.pvalue < th.ks_pvalue)
        rows.append(
            {
                "feature": col,
                "psi": round(p, 4),
                "ks_stat": round(float(ks.statistic), 4),
                "ks_pvalue": float(ks.pvalue),
                "ref_mean": round(float(reference[col].mean()), 2),
                "cur_mean": round(float(current[col].mean()), 2),
                "severity": psi_severity(p, th),
                "drift": bool(drift),
            }
        )
    return pd.DataFrame(rows).sort_values("psi", ascending=False).reset_index(drop=True)


def summarize(table: pd.DataFrame, th: Thresholds = THRESHOLDS) -> dict[str, Any]:
    drifted = table.loc[table["drift"], "feature"].tolist()
    share = len(drifted) / max(len(table), 1)
    return {
        "n_features": len(table),
        "n_drifted": len(drifted),
        "drift_share": round(share, 3),
        "drifted_features": drifted,
        "critical_features": table.loc[table["severity"] == "critical", "feature"].tolist(),
        "max_psi_feature": str(table.iloc[0]["feature"]),
        "max_psi": float(table.iloc[0]["psi"]),
        "dataset_drift": share >= th.drift_share_critical,
    }


# ── Modelo ───────────────────────────────────────────────────────────────────


def load_model(model_path: str = MODEL_PATH):
    if not Path(model_path).exists():
        raise FileNotFoundError(f"Modelo não encontrado em {model_path}. Rode `make train`.")
    with open(model_path, "rb") as handle:
        return pickle.load(handle)


def feature_columns(model) -> list[str]:
    """Features na mesma ordem do treino (``ID`` e alvo ficam de fora)."""
    return [str(c) for c in model.feature_names_in_]


def score(model, frame: pd.DataFrame) -> pd.DataFrame:
    """Anexa probabilidade de default (``score``) e decisão (``prediction``) ao lote."""
    out = frame.copy()
    out[SCORE] = model.predict_proba(out[feature_columns(model)])[:, 1]
    out[PREDICTION] = model.predict(out[feature_columns(model)]).astype(int)
    return out


def ks_separation(y_true, proba) -> float:
    """KS de separação entre bons e maus (métrica clássica de credit scoring)."""
    y = np.asarray(y_true)[np.argsort(proba)]
    cum_bad = np.cumsum(y) / max(y.sum(), 1)
    cum_good = np.cumsum(1 - y) / max((1 - y).sum(), 1)
    return float(np.max(np.abs(cum_bad - cum_good)))


def performance(scored: pd.DataFrame) -> dict[str, float]:
    """Desempenho do modelo num lote com rótulo (a safra já "maturou")."""
    y = scored[TARGET].astype(int)
    pred = scored[PREDICTION]
    return {
        "roc_auc": float(roc_auc_score(y, scored[SCORE])),
        "ks": ks_separation(y, scored[SCORE].to_numpy()),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "default_rate": float(y.mean()),
        "mean_score": float(scored[SCORE].mean()),
        "approval_rate": float(1 - pred.mean()),
    }


def load_baseline_metrics(path: str = METRICS_PATH) -> dict[str, Any]:
    if not Path(path).exists():
        raise FileNotFoundError(f"Métricas do baseline ausentes em {path}. Rode `make train`.")
    return json.loads(Path(path).read_text())


# ── Evidently ────────────────────────────────────────────────────────────────


def evidently_report(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    features: list[str],
    name: str,
    out_dir: Path | None,
    th: Thresholds = THRESHOLDS,
) -> dict[str, Any]:
    """Roda o Evidently (DataDrift + Classification).

    Salva o HTML em ``out_dir`` (se informado) e devolve o resumo: colunas com drift e
    PSI por coluna. O HTML tem ~5 MB (JS embutido), por isso o pipeline só salva o mês
    de controle e o último mês.
    """
    from evidently import BinaryClassification, DataDefinition, Dataset, Report
    from evidently.presets import ClassificationPreset, DataDriftPreset

    categorical = [c for c in CATEGORICAL_COLUMNS if c in features]
    definition = DataDefinition(
        numerical_columns=[*(c for c in features if c not in categorical), SCORE],
        categorical_columns=[*categorical, PREDICTION, TARGET],
        classification=[
            BinaryClassification(
                target=TARGET, prediction_labels=PREDICTION, prediction_probas=SCORE
            )
        ],
    )
    cols = [*features, SCORE, PREDICTION, TARGET]
    report = Report(
        [
            DataDriftPreset(
                columns=[*features, SCORE],
                num_method="psi",
                cat_method="psi",
                threshold=th.psi_warning,
                drift_share=th.drift_share_critical,
            ),
            ClassificationPreset(),
        ],
        include_tests=True,
    )
    snapshot = report.run(
        reference_data=Dataset.from_pandas(reference[cols].copy(), data_definition=definition),
        current_data=Dataset.from_pandas(current[cols].copy(), data_definition=definition),
        name=name,
    )
    summary: dict[str, Any] = {"evidently_psi": {}}
    for metric in snapshot.dict().get("metrics", []):
        metric_name, value = str(metric.get("metric_name", "")), metric.get("value")
        if metric_name.startswith("DriftedColumnsCount"):
            summary["evidently_drifted_columns"] = int(value.get("count", 0))
            summary["evidently_drift_share"] = round(float(value.get("share", 0.0)), 4)
        elif metric_name.startswith("ValueDrift(column="):
            column = metric_name.split("column=", 1)[1].split(",", 1)[0]
            summary["evidently_psi"][column] = round(float(value), 4)
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"evidently_{name}.html"
        snapshot.save_html(str(path))
        summary["html"] = str(path)
    return summary


# ── Análise de um lote ───────────────────────────────────────────────────────


def analyze_batch(
    month: str,
    reference_scored: pd.DataFrame,
    batch_scored: pd.DataFrame,
    features: list[str],
    out_dir: Path,
    *,
    save_html: bool,
) -> dict[str, Any]:
    """Drift de features, drift do score, performance e Evidently de um lote mensal."""
    table = feature_drift(reference_scored, batch_scored, features)
    evidently = evidently_report(
        reference_scored, batch_scored, features, month, out_dir if save_html else None
    )
    table["psi_evidently"] = table["feature"].map(evidently.pop("evidently_psi"))
    out_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(out_dir / f"feature_drift_{month}.csv", index=False)
    return {
        "month": month,
        "rows": len(batch_scored),
        "drift": summarize(table),
        "psi_by_feature": dict(zip(table["feature"], table["psi"], strict=True)),
        "prediction_psi": round(psi(reference_scored[SCORE], batch_scored[SCORE]), 4),
        "performance": {k: round(v, 4) for k, v in performance(batch_scored).items()},
        "evidently": evidently,
    }


def _num(value: float) -> str:
    return f"{value:,.0f}" if abs(value) >= 100 else f"{value:.2f}"


def drift_markdown(records: list[dict[str, Any]], baseline: dict[str, Any]) -> str:
    """Resumo legível (Markdown) do relatório de drift, por mês."""
    lines = [
        "# Relatório de drift: produção vs. referência",
        "",
        f"Baseline (holdout): ROC AUC = **{baseline['roc_auc']:.3f}**. "
        f"Critério: PSI ≥ {THRESHOLDS.psi_critical} ou "
        f"(PSI ≥ {THRESHOLDS.psi_warning} e KS p < {THRESHOLDS.ks_pvalue}).",
        "",
        "| Mês | Features com drift | Drift share | Score médio | Aprovação | PSI do score "
        "| ROC AUC | Default real |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in records:
        d, p = r["drift"], r["performance"]
        feats = ", ".join(d["drifted_features"]) or "nenhuma"
        lines.append(
            f"| {r['month']} | {feats} | {d['drift_share']:.0%} | {p['mean_score']:.3f} "
            f"| {p['approval_rate']:.1%} | {r['prediction_psi']:.3f} "
            f"| {p['roc_auc']:.3f} | {p['default_rate']:.1%} |"
        )
    last = records[-1]
    lines += [
        "",
        f"## Variáveis degradadas no último mês ({last['month']})",
        "",
        "| Feature | PSI | PSI (Evidently) | KS p-valor | Média ref. | Média atual | Severidade |",
        "|---|---|---|---|---|---|---|",
    ]
    table = pd.read_csv(Path(last["table_path"]))
    for _, row in table[table["drift"]].iterrows():
        lines.append(
            f"| {row['feature']} | {row['psi']:.3f} | {row['psi_evidently']:.3f} "
            f"| {row['ks_pvalue']:.1e} | {_num(row['ref_mean'])} | {_num(row['cur_mean'])} "
            f"| {row['severity']} |"
        )
    return "\n".join(lines) + "\n"


def run(settings: Settings, *, all_html: bool = False) -> list[dict[str, Any]]:
    """Compara cada lote de produção com a referência e grava os relatórios."""
    from creditmon.data import load_reference
    from creditmon.simulation import MONTHS, load_batch, production_dir

    model = load_model()
    features = feature_columns(model)
    reference, _ = load_reference(settings)
    reference_scored = score(model, reference)
    baseline = load_baseline_metrics()
    out_dir = settings.paths.reports / "drift"

    records = []
    for i, month in enumerate(MONTHS):
        batch_scored = score(model, load_batch(settings, month))
        # Predições do modelo para o lote de produção (probabilidade e decisão por cliente).
        predictions = batch_scored[[ID_COLUMN, SCORE, PREDICTION]]
        predictions.to_csv(production_dir(settings) / f"predictions_{month}.csv", index=False)
        save = all_html or i in (0, len(MONTHS) - 1)
        record = analyze_batch(
            month, reference_scored, batch_scored, features, out_dir, save_html=save
        )
        record["table_path"] = str(out_dir / f"feature_drift_{month}.csv")
        records.append(record)
        d = record["drift"]
        print(
            f"{month}: {d['n_drifted']}/{d['n_features']} features com drift "
            f"{d['drifted_features'][:5]} | AUC {record['performance']['roc_auc']:.3f}"
        )

    (out_dir / "drift_summary.json").write_text(
        json.dumps({"baseline": baseline, "months": records}, indent=2, ensure_ascii=False)
    )
    (out_dir / "drift_report.md").write_text(drift_markdown(records, baseline), encoding="utf-8")
    print(f"relatórios em {out_dir}/")
    return records


def main() -> int:
    from creditmon.config import load_settings

    parser = argparse.ArgumentParser(description="Detecta drift nos lotes de produção.")
    parser.add_argument("--all-html", action="store_true", help="salva o HTML de todos os meses")
    args = parser.parse_args()
    run(load_settings(), all_html=args.all_html)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
