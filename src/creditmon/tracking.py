"""Tracking de experimentos e registro de modelos no MLflow hospedado no DagsHub.

Autenticação vai por `MLFLOW_TRACKING_USERNAME`/`MLFLOW_TRACKING_PASSWORD` em vez do
fluxo interativo do pacote `dagshub`, para funcionar em execução headless (CI, DVC).
Sem token, `init_mlflow` falha alto: tracking local mascararia a ausência do DagsHub.
"""

from __future__ import annotations

import os

import mlflow

from creditmon.config import Settings


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
