"""Equidade do modelo por grupo: auditoria de vieses e decisão sobre variáveis sensíveis.

Para cada atributo protegido (``SEX``, ``EDUCATION``, ``MARRIAGE`` e faixa de ``AGE``),
calcula por grupo a taxa de aprovação, o recall (inadimplentes barrados) e o FPR (bons
pagadores negados, o erro que mais pesa para o cliente). A disparidade é a **razão de
aprovação** entre o grupo menos e o mais aprovado (regra dos 4/5: < 0,8 indica impacto
desproporcional). Grupos com menos de ``MIN_GROUP_SIZE`` clientes ficam fora da razão:
num lote mensal de 1.000 clientes, um grupo de 6 pessoas não sustenta conclusão.

``make fairness`` grava em ``reports/fairness/``:

* comparação de variantes do baseline (com e sem as variáveis sensíveis), a evidência
  da decisão de excluir ``SEX`` e ``MARRIAGE`` (``train.EXCLUDED_FEATURES``);
* tabelas por grupo do modelo em produção no holdout e no último mês simulado.

O monitoramento mensal (razões por mês e alerta de equidade) roda em ``make drift``,
reusando ``audit``.
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from creditmon.config import Settings
from creditmon.observability import logger, setup_logging, stage
from creditmon.schema import TARGET

MIN_GROUP_SIZE = 30
FOUR_FIFTHS = 0.8
AGE_BINS = [17, 25, 35, 45, 60, 100]
AGE_LABELS = ["18-25", "26-35", "36-45", "46-60", "61+"]
VARIANTS = {
    "todas as features": (),
    "sem SEX e MARRIAGE (adotada)": ("SEX", "MARRIAGE"),
    "sem SEX, MARRIAGE e EDUCATION": ("SEX", "MARRIAGE", "EDUCATION"),
}


def protected_groups(frame: pd.DataFrame) -> dict[str, pd.Series]:
    """Rótulo de grupo de cada cliente por atributo (códigos da UCI; 0/4/5/6 = outros)."""
    return {
        "SEX": frame["SEX"].map({1: "masculino", 2: "feminino"}),
        "EDUCATION": frame["EDUCATION"]
        .map({1: "pós-graduação", 2: "universidade", 3: "ensino médio"})
        .fillna("outros"),
        "MARRIAGE": frame["MARRIAGE"].map({1: "casado", 2: "solteiro"}).fillna("outros"),
        "AGE": pd.cut(frame["AGE"], AGE_BINS, labels=AGE_LABELS).astype(str),
    }


def group_metrics(y: pd.Series, denied: pd.Series, group: pd.Series) -> pd.DataFrame:
    """Aprovação, recall e FPR por grupo (``denied`` = 1 quando o modelo nega o crédito)."""
    t = pd.DataFrame({"group": group.to_numpy(), "y": y.to_numpy(), "d": denied.to_numpy()})
    rows = []
    for name, g in t.groupby("group", sort=True):
        bad, good = g["y"] == 1, g["y"] == 0
        rows.append(
            {
                "group": name,
                "n": len(g),
                "approval_rate": round(1 - g["d"].mean(), 4),
                "recall": round(g.loc[bad, "d"].mean(), 4) if bad.any() else None,
                "fpr": round(g.loc[good, "d"].mean(), 4) if good.any() else None,
                "default_rate": round(g["y"].mean(), 4),
            }
        )
    return pd.DataFrame(rows)


def approval_ratio(table: pd.DataFrame, min_n: int = MIN_GROUP_SIZE) -> float | None:
    """Razão min/máx da aprovação entre grupos com pelo menos ``min_n`` clientes."""
    rates = table.loc[table["n"] >= min_n, "approval_rate"]
    if len(rates) < 2 or rates.max() == 0:
        return None
    return round(float(rates.min() / rates.max()), 4)


def audit(frame: pd.DataFrame, denied: pd.Series) -> dict[str, dict[str, Any]]:
    """Tabela por grupo e razão de aprovação de cada atributo protegido."""
    out = {}
    for attribute, group in protected_groups(frame).items():
        table = group_metrics(frame[TARGET].astype(int), denied, group)
        out[attribute] = {
            "approval_ratio": approval_ratio(table),
            "groups": table.to_dict(orient="records"),
        }
    return out


def ratios(result: dict[str, dict[str, Any]]) -> dict[str, float | None]:
    return {attribute: r["approval_ratio"] for attribute, r in result.items()}


# ── Relatório (make fairness) ────────────────────────────────────────────────


def compare_variants(
    reference: pd.DataFrame, evaluations: dict[str, pd.DataFrame], seed: int
) -> list[dict[str, Any]]:
    """Treina cada variante do baseline e mede ROC AUC e razões de aprovação."""
    from creditmon.train import fit_model, model_features

    rows = []
    for name, exclude in VARIANTS.items():
        features = model_features(reference.columns, exclude)
        model, _ = fit_model(reference, features, seed)
        row: dict[str, Any] = {"variant": name, "excluded": list(exclude)}
        for split, frame in evaluations.items():
            proba = model.predict_proba(frame[features])[:, 1]
            denied = pd.Series(model.predict(frame[features]), index=frame.index)
            row[split] = {
                "roc_auc": round(float(roc_auc_score(frame[TARGET], proba)), 4),
                "approval_rate": round(float(1 - denied.mean()), 4),
                "approval_ratio": ratios(audit(frame, denied)),
            }
        rows.append(row)
    return rows


def _ratio(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"**{value:.3f}**" if value < FOUR_FIFTHS else f"{value:.3f}"


def fairness_markdown(variants: list[dict[str, Any]], audits: dict[str, dict]) -> str:
    """Relatório legível: comparação de variantes e tabelas por grupo."""
    splits = list(audits)
    attributes = list(next(iter(audits.values())))
    lines = [
        "# Relatório de equidade do modelo de crédito",
        "",
        f"Razão de aprovação = aprovação do grupo menos aprovado / do mais aprovado, entre "
        f"grupos com n ≥ {MIN_GROUP_SIZE}. Abaixo de {FOUR_FIFTHS} (em negrito) a regra "
        "dos 4/5 indica impacto desproporcional.",
        "",
        "## Variantes do baseline",
        "",
        "| Variante | " + " | ".join(
            f"ROC AUC {s} | Aprovação {s} | " + " | ".join(f"Razão {a} {s}" for a in attributes)
            for s in splits
        ) + " |",
        "|---|" + "---|" * (len(splits) * (2 + len(attributes))),
    ]
    for v in variants:
        cells = []
        for s in splits:
            m = v[s]
            cells += [f"{m['roc_auc']:.4f}", f"{m['approval_rate']:.1%}"]
            cells += [_ratio(m["approval_ratio"][a]) for a in attributes]
        lines.append(f"| {v['variant']} | " + " | ".join(cells) + " |")
    for split, result in audits.items():
        lines += ["", f"## Modelo em produção por grupo: {split}", ""]
        for attribute, r in result.items():
            lines += [
                f"### {attribute} (razão de aprovação {_ratio(r['approval_ratio'])})",
                "",
                "| Grupo | n | Aprovação | Recall | FPR | Default real |",
                "|---|---|---|---|---|---|",
            ]
            for g in r["groups"]:
                recall = "n/a" if g["recall"] is None else f"{g['recall']:.1%}"
                fpr = "n/a" if g["fpr"] is None else f"{g['fpr']:.1%}"
                lines.append(
                    f"| {g['group']} | {g['n']} | {g['approval_rate']:.1%} | {recall} "
                    f"| {fpr} | {g['default_rate']:.1%} |"
                )
            lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def run(settings: Settings) -> dict[str, Any]:
    from creditmon.data import load_reference
    from creditmon.drift import load_model
    from creditmon.simulation import MONTHS, load_batch
    from creditmon.train import EXCLUDED_FEATURES

    reference, holdout = load_reference(settings)
    evaluations = {"holdout": holdout, MONTHS[-1]: load_batch(settings, MONTHS[-1])}
    model = load_model()
    features = list(model.feature_names_in_)
    audits = {
        split: audit(frame, pd.Series(model.predict(frame[features]), index=frame.index))
        for split, frame in evaluations.items()
    }
    summary = {
        "min_group_size": MIN_GROUP_SIZE,
        "production_features_excluded": list(EXCLUDED_FEATURES),
        "variants": compare_variants(reference, evaluations, settings.seed),
        "audits": audits,
    }
    out_dir = settings.paths.reports / "fairness"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "fairness_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, default=_json_default)
    )
    (out_dir / "fairness_report.md").write_text(
        fairness_markdown(summary["variants"], audits), encoding="utf-8"
    )
    logger.info(f"relatórios em {out_dir}/")
    return summary


def _json_default(value: Any) -> Any:
    return value.item() if isinstance(value, np.generic) else str(value)


def main() -> int:
    import mlflow

    from creditmon.config import load_settings
    from creditmon.tracking import log_stage, tracked_run

    setup_logging()
    settings = load_settings()
    with tracked_run(settings, "fairness") as tracking:
        with stage("fairness") as st:
            summary = run(settings)
            st.update(variants=len(summary["variants"]))
        if tracking:
            mlflow.log_artifacts(str(settings.paths.reports / "fairness"), "fairness")
            log_stage(st)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
