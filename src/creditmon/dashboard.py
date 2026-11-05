"""Dashboard de monitoramento (Etapa 3): HTML estático, arquivo único, abre offline.

Fontes:

* ``reports/monitoring/pipeline_runs.jsonl``: eventos ``stage`` do pipeline (status,
  duração, volume, lotes bloqueados pelo contrato);
* ``reports/drift/drift_summary.json``: drift, performance, equidade e alertas por mês.

Saída: ``reports/monitoring/dashboard.html``, com os gráficos do matplotlib embutidos em
base64 e uma tabela ao lado de cada gráfico (os números legíveis sem depender da cor).
O JSONL é append-only; a tabela de saúde mostra o último evento de cada par
(etapa, lote), então execuções antigas não poluem o painel.

Uso: ``make dashboard`` (também é gerado no fim de ``make drift``).
"""

from __future__ import annotations

import base64
import html
import io
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap

from creditmon.drift import THRESHOLDS
from creditmon.observability import LOG_PATH

SUMMARY_PATH = Path("reports/drift/drift_summary.json")
DASHBOARD_PATH = Path("reports/monitoring/dashboard.html")

# Paleta de referência da skill de visualização (tema claro).
INK, INK_2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS, SURFACE = "#e1e0d9", "#c3c2b7", "#fcfcfb"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
BLUES = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
STATUS = {"good": "#0ca30c", "warning": "#fab219", "serious": "#ec835a", "critical": "#d03b3b"}

STAGE_STATUS = {
    "success": ("good", "✓ sucesso"),
    "blocked": ("serious", "⛔ bloqueado"),
    "failed": ("critical", "✗ falhou"),
}
ALERT_LABELS = {
    "psi_critical": "Feature com PSI crítico",
    "dataset_drift": "Dataset drift",
    "auc_drop": "Queda de ROC AUC",
    "default_shift": "Inadimplência real fora do holdout",
    "contract_blocked": "Lote bloqueado",
    "fairness": "Equidade (regra dos 4/5)",
}
ALERT_SHORT = {
    "psi_critical": "PSI crít.",
    "dataset_drift": "Dataset drift",
    "auc_drop": "Queda AUC",
    "default_shift": "Inadimpl.",
    "contract_blocked": "Lote bloq.",
    "fairness": "Equidade",
}


# ── Dados ────────────────────────────────────────────────────────────────────


def load_stage_events(path: Path = LOG_PATH) -> list[dict[str, Any]]:
    """Eventos ``stage`` do JSONL (linhas de log comuns são ignoradas)."""
    if not path.exists():
        return []
    lines = path.read_text(encoding="utf-8").splitlines()
    return [e for e in map(json.loads, filter(None, lines)) if e.get("event") == "stage"]


def latest_by_stage(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Último evento de cada (etapa, lote), na ordem da primeira ocorrência."""
    latest: dict[tuple[str, str | None], dict[str, Any]] = {}
    for event in events:
        latest[(event["stage"], event.get("batch"))] = event  # dict mantém a ordem de inserção
    return list(latest.values())


def month_status(alerts: dict[str, int]) -> tuple[str, str]:
    """Estado do mês: crítico se modelo ou ingestão foram afetados; atenção com PSI, equidade
    ou inadimplência real fora do holdout."""
    if alerts["auc_drop"] or alerts["dataset_drift"] or alerts["contract_blocked"]:
        return "critical", "✗ crítico"
    if alerts["psi_critical"] or alerts["fairness"] or alerts["default_shift"]:
        return "warning", "⚠ atenção"
    return "good", "✓ estável"


# ── Gráficos ─────────────────────────────────────────────────────────────────


def _style(ax) -> None:
    ax.set_facecolor(SURFACE)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(AXIS)
    ax.tick_params(colors=MUTED, labelsize=9, length=0)


def _png(fig) -> str:
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=110, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    return base64.b64encode(buffer.getvalue()).decode()


def _limit(ax, y: float, label: str, left: bool = False) -> None:
    ax.axhline(y, color=MUTED, linewidth=1, linestyle="--")
    ax.annotate(label, (0 if left else 1, y), xycoords=("axes fraction", "data"), xytext=(0, 3),
                textcoords="offset points", ha="left" if left else "right", fontsize=8,
                color=INK_2)


def _line(ax, months, values, title: str, fmt: str, color: str = SERIES[0]) -> None:
    _style(ax)
    ax.plot(months, values, color=color, linewidth=2, marker="o", markersize=6,
            markeredgecolor=SURFACE, markeredgewidth=2)
    ax.annotate(format(values[-1], fmt), (len(months) - 1, values[-1]), xytext=(6, 0),
                textcoords="offset points", va="center", fontsize=9, color=INK)
    ax.set_title(title, loc="left", fontsize=11, color=INK)


def trend_chart(summary: dict[str, Any]) -> str:
    """Três pequenos múltiplos: drift share, PSI do score e ROC AUC por mês."""
    months = [m["month"] for m in summary["months"]]
    baseline_auc = summary["baseline"]["roc_auc"]
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.4), facecolor=SURFACE)

    _line(axes[0], months, [m["drift"]["drift_share"] for m in summary["months"]],
          "Drift share (fração de features com drift)", ".0%")
    _limit(axes[0], THRESHOLDS.drift_share_critical, "dataset drift")
    axes[0].yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))

    _line(axes[1], months, [m["prediction_psi"] for m in summary["months"]],
          "PSI do score (prediction drift)", ".3f")
    _limit(axes[1], THRESHOLDS.psi_warning, "moderado")
    _limit(axes[1], THRESHOLDS.psi_critical, "crítico")

    _line(axes[2], months, [m["performance"]["roc_auc"] for m in summary["months"]],
          "ROC AUC por safra (com rótulo)", ".3f")
    _limit(axes[2], baseline_auc, f"baseline {baseline_auc:.3f}")
    _limit(axes[2], baseline_auc - THRESHOLDS.auc_drop_critical, "limite de alerta")
    fig.tight_layout(w_pad=3)
    return _png(fig)


def business_chart(summary: dict[str, Any]) -> str:
    """Taxa de aprovação vs. inadimplência real por mês (mesma escala: proporção)."""
    months = [m["month"] for m in summary["months"]]
    fig, ax = plt.subplots(figsize=(6.4, 3.4), facecolor=SURFACE)
    series = [
        ("Aprovação do modelo", [m["performance"]["approval_rate"] for m in summary["months"]]),
        ("Inadimplência real", [m["performance"]["default_rate"] for m in summary["months"]]),
    ]
    for (label, values), color in zip(series, SERIES, strict=False):
        _line(ax, months, values, "Decisão vs. inadimplência real", ".0%", color)
        ax.lines[-1].set_label(label)
    base_default = summary["baseline"]["default_rate"]
    _limit(ax, base_default + THRESHOLDS.default_rate_shift, "limite de inadimplência", left=True)
    ax.set_ylim(0, 1)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.legend(frameon=False, fontsize=9, labelcolor=INK_2, loc="center left")
    return _png(fig)


def alerts_chart(summary: dict[str, Any]) -> str:
    """Barras empilhadas de alertas por mês e tipo, com o total no topo."""
    months = [m["month"] for m in summary["months"]]
    fig, ax = plt.subplots(figsize=(6.4, 3.4), facecolor=SURFACE)
    _style(ax)
    bottom = [0] * len(months)
    for kind, color in zip(ALERT_LABELS, SERIES, strict=True):
        values = [m["alerts"][kind] for m in summary["months"]]
        ax.bar(months, values, bottom=bottom, color=color, width=0.6, label=ALERT_LABELS[kind],
               edgecolor=SURFACE, linewidth=2)
        bottom = [b + v for b, v in zip(bottom, values, strict=True)]
    for i, total in enumerate(bottom):
        ax.annotate(str(total), (i, total), xytext=(0, 3), textcoords="offset points",
                    ha="center", fontsize=9, color=INK)
    ax.set_title("Alertas por mês e tipo", loc="left", fontsize=11, color=INK)
    ax.yaxis.get_major_locator().set_params(integer=True)
    ax.legend(frameon=False, fontsize=8, labelcolor=INK_2, loc="upper left")
    return _png(fig)


def psi_heatmap(summary: dict[str, Any]) -> str:
    """Heatmap feature × mês (azul sequencial); só as células críticas levam o número."""
    months = [m["month"] for m in summary["months"]]
    last = summary["months"][-1]["psi_by_feature"]
    features = sorted(last, key=last.get, reverse=True)
    grid = [[m["psi_by_feature"][f] for m in summary["months"]] for f in features]

    bounds = [0, 0.025, 0.05, THRESHOLDS.psi_warning, 0.175, THRESHOLDS.psi_critical, 0.5, 10]
    norm = BoundaryNorm(bounds, len(BLUES))
    fig, ax = plt.subplots(figsize=(6.4, 7.2), facecolor=SURFACE)
    image = ax.imshow(grid, cmap=ListedColormap(BLUES), norm=norm, aspect="auto")
    ax.set_xticks(range(len(months)), months)
    ax.set_yticks(range(len(features)), features)
    ax.tick_params(colors=INK_2, labelsize=8, length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xticks([x - 0.5 for x in range(1, len(months))], minor=True)
    ax.set_yticks([y - 0.5 for y in range(1, len(features))], minor=True)
    ax.grid(which="minor", color=SURFACE, linewidth=2)
    ax.tick_params(which="minor", length=0)
    for y, row in enumerate(grid):
        for x, value in enumerate(row):
            if value >= THRESHOLDS.psi_critical:
                ax.text(x, y, f"{value:.2f}", ha="center", va="center", fontsize=7,
                        color="#ffffff")
    bar = fig.colorbar(image, ax=ax, fraction=0.04, pad=0.02, ticks=bounds[:-1])
    bar.ax.tick_params(labelsize=7, colors=MUTED, length=0)
    bar.outline.set_visible(False)
    ax.set_title("PSI por feature e mês (ordenado pelo último mês)", loc="left", fontsize=11,
                 color=INK)
    return _png(fig)


# ── HTML ─────────────────────────────────────────────────────────────────────

CSS = """
.viz-root { color-scheme: light; --surface: #fcfcfb; --page: #f9f9f7; --ink: #0b0b0b;
  --ink-2: #52514e; --muted: #898781; --grid: #e1e0d9; --good: #0ca30c; --warning: #fab219;
  --serious: #ec835a; --critical: #d03b3b; }
body { margin: 0; background: var(--page); color: var(--ink);
  font-family: system-ui, -apple-system, "Segoe UI", sans-serif; }
main { max-width: 1240px; margin: 0 auto; padding: 24px; }
h1 { font-size: 22px; margin: 0 0 4px; } h2 { font-size: 16px; margin: 32px 0 8px; }
p.sub { color: var(--ink-2); margin: 0 0 16px; font-size: 14px; }
.kpis { display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 12px; }
.tile { background: var(--surface); border: 1px solid rgba(11,11,11,.10); border-radius: 8px;
  padding: 12px 16px; }
.tile .label { color: var(--ink-2); font-size: 12px; } .tile .value { font-size: 26px; }
.tile .note { color: var(--muted); font-size: 12px; }
.card { background: var(--surface); border: 1px solid rgba(11,11,11,.10); border-radius: 8px;
  padding: 12px; margin-top: 12px; overflow-x: auto; }
.row { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }
@media (max-width: 900px) { .row { grid-template-columns: 1fr; } }
img { max-width: 100%; display: block; }
table { border-collapse: collapse; width: 100%; font-size: 13px;
  font-variant-numeric: tabular-nums; }
th, td { text-align: left; padding: 6px 8px; border-bottom: 1px solid var(--grid); }
th { color: var(--ink-2); font-weight: 600; } td.num { text-align: right; }
.badge { display: inline-block; padding: 1px 8px; border-radius: 10px; font-size: 12px;
  color: #ffffff; white-space: nowrap; }
.badge.warning { color: #0b0b0b; }
.good { background: var(--good); } .warning { background: var(--warning); }
.serious { background: var(--serious); } .critical { background: var(--critical); }
footer { color: var(--muted); font-size: 12px; margin-top: 32px; }
"""


def _badge(kind: str, label: str) -> str:
    return f'<span class="badge {kind}">{html.escape(label)}</span>'


def _table(headers: list[str], rows: list[list[str]], numeric: set[int] = frozenset()) -> str:
    head = "".join(f"<th>{h}</th>" for h in headers)
    body = "".join(
        "<tr>" + "".join(
            f'<td class="num">{c}</td>' if i in numeric else f"<td>{c}</td>"
            for i, c in enumerate(row)
        ) + "</tr>"
        for row in rows
    )
    return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


def _num(value: Any, fmt: str = "") -> str:
    return "–" if value is None else format(value, fmt)


def _img(data: str, alt: str) -> str:
    return f'<img src="data:image/png;base64,{data}" alt="{html.escape(alt)}">'


def _ratio_badge(ratio: float | None) -> str:
    if ratio is None:
        return "–"
    ok = ratio >= THRESHOLDS.fairness_ratio_min
    return _badge("good" if ok else "warning", f"{'✓' if ok else '⚠'} {ratio:.2f}")


def render(summary: dict[str, Any], events: list[dict[str, Any]]) -> str:
    """Monta o HTML do painel a partir do resumo de drift e dos eventos do pipeline."""
    months = summary["months"]
    last, baseline = months[-1], summary["baseline"]
    latest = latest_by_stage(events)
    counts = {s: sum(e["status"] == s for e in latest) for s in STAGE_STATUS}
    alert_months = [m["month"] for m in months if m["alerts_total"]]
    auc_delta = last["performance"]["roc_auc"] - baseline["roc_auc"]

    tiles = [
        ("Etapas executadas", str(len(latest)),
         f"{counts['success']} sucesso · {counts['blocked']} bloqueadas · "
         f"{counts['failed']} falhas"),
        ("Lotes bloqueados pelo contrato", str(len(summary.get("blocked_batches", []))),
         ", ".join(summary.get("blocked_batches", [])) or "nenhum"),
        (f"ROC AUC {last['month']}", f"{last['performance']['roc_auc']:.3f}",
         f"{auc_delta:+.3f} vs. baseline {baseline['roc_auc']:.3f}"),
        (f"Drift share {last['month']}", f"{last['drift']['drift_share']:.0%}",
         f"{last['drift']['n_drifted']} de {last['drift']['n_features']} features"),
        ("Alertas no período", str(sum(m["alerts_total"] for m in months)),
         f"meses com alerta: {', '.join(alert_months) or 'nenhum'}"),
    ]
    kpis = "".join(
        f'<div class="tile"><div class="label">{html.escape(label)}</div>'
        f'<div class="value">{value}</div><div class="note">{html.escape(note)}</div></div>'
        for label, value, note in tiles
    )

    health_rows = [
        [
            html.escape(e["stage"]),
            html.escape(e.get("batch") or "–"),
            _badge(*STAGE_STATUS.get(e["status"], ("critical", e["status"]))),
            _num(e.get("duration_s"), ".2f"),
            _num(e.get("rows_in"), ","),
            _num(e.get("rows_out"), ","),
            _num(e.get("failure_count")),
            e["ts"][:19].replace("T", " "),
        ]
        for e in latest
    ]
    health = _table(
        ["Etapa", "Lote", "Status", "Duração (s)", "Linhas entrada", "Linhas saída",
         "Violações do contrato", "Horário (UTC)"],
        health_rows, numeric={3, 4, 5, 6},
    )

    month_rows = [
        [
            m["month"],
            _badge(*month_status(m["alerts"])),
            f"{m['drift']['drift_share']:.1%}",
            html.escape(", ".join(m["drift"]["critical_features"]) or "–"),
            f"{m['prediction_psi']:.3f}",
            f"{m['performance']['roc_auc']:.3f}",
            f"{m['performance']['approval_rate']:.1%}",
            f"{m['performance']['default_rate']:.1%}",
            *(str(m["alerts"][k]) for k in ALERT_LABELS),
            str(m["alerts_total"]),
        ]
        for m in months
    ]
    monthly = _table(
        ["Mês", "Estado", "Drift share", "Features críticas", "PSI score", "ROC AUC",
         "Aprovação", "Default real", *(ALERT_SHORT[k] for k in ALERT_LABELS), "Total"],
        month_rows, numeric=set(range(4, 9 + len(ALERT_LABELS))),
    )

    attributes = list(baseline["fairness"])
    fair_rows = [
        [label, *(_ratio_badge(ratios.get(a)) for a in attributes)]
        for label, ratios in [
            ("holdout (baseline)", baseline["fairness"]),
            *((m["month"], m["fairness"]) for m in months),
        ]
    ]
    fairness_table = _table(["Lote", *attributes], fair_rows)

    generated = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
    return f"""<!DOCTYPE html>
<html lang="pt-BR">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Monitoramento: credit scoring</title>
<style>{CSS}</style>
</head>
<body class="viz-root">
<main>
<h1>Monitoramento do modelo de credit scoring</h1>
<p class="sub">Saúde do pipeline, drift de dados e de predição, performance por safra e
alertas. Referência: treino de 24.000 clientes; produção: 6 lotes mensais simulados de
1.000 clientes (jan/2026 é o mês de controle, sem drift).</p>
<div class="kpis">{kpis}</div>

<h2>Saúde do pipeline</h2>
<p class="sub">Último evento de cada etapa e lote em <code>pipeline_runs.jsonl</code>.
Bloqueado = contrato de dados rejeitou o lote antes de escorar.</p>
<div class="card">{health}</div>

<h2>Drift e performance ao longo do tempo</h2>
<div class="card">{_img(trend_chart(summary), "Drift share, PSI do score e ROC AUC por mês")}</div>
<div class="row">
<div class="card">{_img(alerts_chart(summary), "Alertas por mês e tipo")}</div>
<div class="card">{_img(business_chart(summary), "Aprovação e inadimplência por mês")}</div>
</div>

<h2>Estado por mês</h2>
<p class="sub">Crítico: queda de ROC AUC &gt; {THRESHOLDS.auc_drop_critical}, drift share
&ge; {THRESHOLDS.drift_share_critical:.0%} ou lote bloqueado. Atenção: features com
PSI &ge; {THRESHOLDS.psi_critical}, equidade abaixo de {THRESHOLDS.fairness_ratio_min} ou
inadimplência real a mais de {THRESHOLDS.default_rate_shift * 100:.0f} p.p. do holdout.</p>
<div class="card">{monthly}</div>

<h2>Equidade por atributo protegido</h2>
<p class="sub">Razão de aprovação entre o grupo menos e o mais aprovado (grupos com pelo
menos 30 clientes). Abaixo de {THRESHOLDS.fairness_ratio_min} a regra dos 4/5 indica impacto
desproporcional; o alerta dispara quando um atributo que passava no holdout deixa de passar.
<code>SEX</code> e <code>MARRIAGE</code> não são features do modelo, só auditadas.</p>
<div class="card">{fairness_table}</div>

<h2>Variáveis degradadas</h2>
<div class="card">{_img(psi_heatmap(summary), "PSI por feature e mês")}</div>

<footer>Gerado em {generated} por <code>make dashboard</code> a partir de
<code>reports/drift/drift_summary.json</code> e
<code>reports/monitoring/pipeline_runs.jsonl</code>. Relatórios do Evidently em
<code>reports/drift/</code>; runs no MLflow do DagsHub.</footer>
</main>
</body>
</html>
"""


def build(
    summary_path: Path = SUMMARY_PATH,
    log_path: Path = LOG_PATH,
    out_path: Path = DASHBOARD_PATH,
) -> Path:
    """Lê as fontes e grava o dashboard; devolve o caminho do HTML."""
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(render(summary, load_stage_events(log_path)), encoding="utf-8")
    return out_path


def main() -> int:
    # print, não logger: o dashboard só lê o JSONL e não deve gravar eventos nele.
    print(f"dashboard: {build()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
