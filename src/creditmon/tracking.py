"""Tracking de experimentos e registro de modelos no MLflow hospedado no DagsHub.

Autenticação vai por `MLFLOW_TRACKING_USERNAME`/`MLFLOW_TRACKING_PASSWORD` em vez do
fluxo interativo do pacote `dagshub`, para funcionar em execução headless (CI, DVC).
Sem token, `init_mlflow` falha alto: tracking local mascararia a ausência do DagsHub.
`tracked_run` é a versão tolerante para as etapas do pipeline: sem token, avisa no log
e segue só com os relatórios locais.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime
from numbers import Number
from typing import Any

import mlflow

from creditmon.config import Settings
from creditmon.observability import RUN_ID, logger


def init_mlflow(settings: Settings) -> str:
    """Aponta o MLflow para o DagsHub. Exige `DAGSHUB_TOKEN` no `.env`.

    Returns:
        A URI de tracking configurada.
    """
    if not settings.dagshub_token:
        raise RuntimeError(
            "DAGSHUB_TOKEN ausente: defina no .env "
            "(cp .env.example .env). O tracking de experimentos exige o DagsHub."
        )
    os.environ["MLFLOW_TRACKING_USERNAME"] = settings.dagshub_user or settings.dagshub_token
    os.environ["MLFLOW_TRACKING_PASSWORD"] = settings.dagshub_token
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment(settings.mlflow.experiment_name)
    return settings.mlflow_tracking_uri


def pipeline_tags(stage_name: str, **tags: Any) -> dict[str, str]:
    """Tags comuns: ligam a run do MLflow aos eventos de `pipeline_runs.jsonl` (`run_id`)."""
    return {"pipeline_run_id": RUN_ID, "pipeline_stage": stage_name, **tags}


@contextmanager
def tracked_run(settings: Settings, stage_name: str, **tags: Any) -> Iterator[bool]:
    """Abre uma run `<etapa>-<timestamp>` no DagsHub; devolve `False` (e não abre) sem token.

    Exceção dentro do bloco fecha a run como `FAILED` no MLflow.
    """
    if not settings.dagshub_token:
        logger.warning(
            f"DAGSHUB_TOKEN ausente: etapa '{stage_name}' NÃO registrada no MLflow "
            "(relatórios locais seguem sendo gerados)"
        )
        yield False
        return
    init_mlflow(settings)
    run_name = f"{stage_name}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    with mlflow.start_run(run_name=run_name, tags=pipeline_tags(stage_name, **tags)):
        yield True


def log_stage(info: dict[str, Any]) -> None:
    """Loga o resultado de um `stage()` na run ativa: `status` como tag, números como métricas."""
    if mlflow.active_run() is None:
        return
    mlflow.set_tag("status", info.get("status", "unknown"))
    metrics = {
        k: float(v)
        for k, v in info.items()
        if isinstance(v, Number) and not isinstance(v, bool)
    }
    mlflow.log_metrics(metrics)
