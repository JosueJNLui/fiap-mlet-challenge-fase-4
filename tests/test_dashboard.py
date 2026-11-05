from __future__ import annotations

import json

from creditmon import dashboard

NO_ALERTS = {
    "psi_critical": 0, "dataset_drift": 0, "auc_drop": 0, "default_shift": 0,
    "contract_blocked": 0, "fairness": 0,
}


def month(name: str, share: float, auc: float, alerts: dict[str, int]) -> dict:
    return {
        "month": name,
        "rows": 1000,
        "drift": {"n_features": 2, "n_drifted": int(share * 2), "drift_share": share,
                  "critical_features": ["AGE"] if alerts["psi_critical"] else []},
        "psi_by_feature": {"AGE": 0.3 if alerts["psi_critical"] else 0.01, "SEX": 0.0},
        "prediction_psi": 0.02,
        "performance": {"roc_auc": auc, "approval_rate": 0.7, "default_rate": 0.2},
        "fairness": {"SEX": 0.95, "AGE": 0.7 if alerts["fairness"] else 0.9},
        "alerts": alerts,
        "alerts_total": sum(alerts.values()),
    }


def stage_event(stage: str, status: str, batch: str | None = None, ts: str = "T0") -> dict:
    return {"event": "stage", "stage": stage, "status": status, "batch": batch, "ts": ts,
            "duration_s": 0.1, "rows_in": 10, "rows_out": None}


def test_latest_by_stage_keeps_last_event_per_stage_and_batch() -> None:
    events = [
        stage_event("drift", "failed", ts="old"),
        stage_event("ingest", "success", "2026-01"),
        stage_event("ingest", "blocked", "2026-06-corrupted"),
        stage_event("drift", "success", ts="new"),
    ]
    latest = dashboard.latest_by_stage(events)
    assert [(e["stage"], e["batch"]) for e in latest] == [
        ("drift", None), ("ingest", "2026-01"), ("ingest", "2026-06-corrupted")
    ]
    assert latest[0]["ts"] == "new"


def test_month_status_levels() -> None:
    assert dashboard.month_status(NO_ALERTS)[0] == "good"
    assert dashboard.month_status({**NO_ALERTS, "psi_critical": 2})[0] == "warning"
    assert dashboard.month_status({**NO_ALERTS, "fairness": 1})[0] == "warning"
    assert dashboard.month_status({**NO_ALERTS, "default_shift": 1})[0] == "warning"
    assert dashboard.month_status({**NO_ALERTS, "auc_drop": 1})[0] == "critical"


def test_build_writes_self_contained_html(tmp_path) -> None:
    summary = {
        "baseline": {"roc_auc": 0.78, "default_rate": 0.22, "fairness": {"SEX": 0.95, "AGE": 0.86}},
        "months": [
            month("2026-01", 0.0, 0.80, NO_ALERTS),
            month("2026-02", 0.5, 0.65,
                  {**NO_ALERTS, "psi_critical": 1, "auc_drop": 1, "fairness": 1}),
        ],
        "blocked_batches": ["2026-02-corrupted"],
    }
    summary_path, log_path = tmp_path / "summary.json", tmp_path / "runs.jsonl"
    summary_path.write_text(json.dumps(summary))
    log_path.write_text(
        json.dumps({"message": "linha comum de log"}) + "\n"
        + json.dumps(stage_event("ingest", "blocked", "2026-02-corrupted")) + "\n"
    )

    out = dashboard.build(summary_path, log_path, tmp_path / "dashboard.html")

    page = out.read_text(encoding="utf-8")
    assert page.count("data:image/png;base64,") == 4
    assert "✓ estável" in page and "✗ crítico" in page and "⛔ bloqueado" in page
    assert "2026-02-corrupted" in page
    # Cada tipo de alerta tem cor, rótulo e coluna própria na tabela por mês.
    assert set(dashboard.ALERT_LABELS) == set(dashboard.ALERT_SHORT) == set(NO_ALERTS)
    assert "Inadimpl." in page
    assert "⚠ 0.70" in page and "✓ 0.86" in page
