"""Treino do baseline (XGBoost) para credit scoring.

Treina a referência congelada, avalia no holdout limpo, loga métricas e o modelo no
MLflow (DagsHub), registra e promove para `production`.

O objetivo: evidenciar que o contrato e o baseline funcionam antes de simular drift.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from xgboost import XGBClassifier

from creditmon.config import Settings
from creditmon.data import HOLDOUT_FRACTION, TARGET, load_reference
from creditmon.schema import ID_COLUMN, validate_batch
from creditmon.tracking import init_mlflow

DEFAULT_MODEL_NAME = "credit_scoring_baseline"
DEFAULT_ALIAS = "production"
MODEL_PATH = "models/baseline.pkl"
METRICS_PATH = "models/baseline_metrics.json"


def train_baseline(settings: Settings, *, track: bool = True) -> dict:
    """Treina o baseline, loga no MLflow e grava artefatos locais.

    Args:
        settings: Configuração do projeto.
        track: Se `True`, configura o MLflow e loga run/registro.

    Returns:
        Dicionário com métricas no conjunto de test (holdout).
    """
    reference, holdout = load_reference(settings)

    if track and settings.dagshub_token:
        init_mlflow(settings)

    feature_columns = [c for c in reference.columns if c not in {ID_COLUMN, TARGET}]
    X_train = reference[feature_columns]
    y_train = reference[TARGET].astype(int)
    X_test = holdout[feature_columns]
    y_test = holdout[TARGET].astype(int)

    neg = (y_train == 0).sum()
    pos = (y_train == 1).sum()
    scale_pos_weight = (neg / pos) if pos else 1.0

    params = {
        "seed": settings.seed,
        "n_estimators": 200,
        "max_depth": 4,
        "learning_rate": 0.1,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "eval_metric": "logloss",
        "scale_pos_weight": float(scale_pos_weight),
    }

    model = XGBClassifier(**params)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "roc_auc": float(roc_auc_score(y_test, proba)),
        "accuracy": float(accuracy_score(y_test, preds)),
        "precision": float(precision_score(y_test, preds, zero_division=0)),
        "recall": float(recall_score(y_test, preds, zero_division=0)),
        "f1": float(f1_score(y_test, preds, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_test, preds).tolist(),
        "test_size": int(len(X_test)),
        "train_size": int(len(X_train)),
        "holdout_fraction": HOLDOUT_FRACTION,
    }

    # Grava artefatos locais para Etapa 2
    Path(MODEL_PATH).parent.mkdir(parents=True, exist_ok=True)
    import pickle

    with open(MODEL_PATH, "wb") as handle:
        pickle.dump(model, handle)

    # Salva como objeto JSON (não como array) para ser fácil de ler em outros lugares
    import json

    with open(METRICS_PATH, "w") as handle:
        json.dump(metrics, handle, indent=2)
    print(f"modelo salvo: {MODEL_PATH}")
    print(f"metricas salvas: {METRICS_PATH}")

    if track:
        run_name = f"credit-baseline-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        with mlflow.start_run(run_name=run_name):
            mlflow.log_params(params)
            mlflow.log_metrics(
                {
                    k: v
                    for k, v in metrics.items()
                    if not isinstance(v, list)
                }
            )
            mlflow.log_dict(metrics, "metrics/metrics.json")
            try:
                mlflow.sklearn.log_model(
                    sk_model=model,
                    name="model",
                    registered_model_name=DEFAULT_MODEL_NAME,
                    input_example=X_test.head(10),
                    signature=mlflow.models.infer_signature(X_test, proba),
                    skops_trusted_types=[
                        "xgboost.sklearn.XGBClassifier",
                        "xgboost.core.Booster",
                    ],
                )
            except Exception:
                mlflow.sklearn.log_model(
                    sk_model=model,
                    name="model",
                    registered_model_name=DEFAULT_MODEL_NAME,
                    input_example=X_test.head(10),
                    signature=mlflow.models.infer_signature(X_test, proba),
                )
            try:
                client = mlflow.MlflowClient()
                versions = client.search_model_versions(
                    f"name='{DEFAULT_MODEL_NAME}'"
                )
                if versions:
                    latest = max(versions, key=lambda v: int(v.version))
                    client.set_registered_model_alias(
                        DEFAULT_MODEL_NAME, DEFAULT_ALIAS, latest.version
                    )
                    print(
                        f"registro: {DEFAULT_MODEL_NAME} "
                        f"v{latest.version} -> alias '{DEFAULT_ALIAS}'"
                    )
            except Exception:
                print("alias não promovido (continuando)")

    print("treino concluído")
    return metrics


def predict_from_model(
    frame: pd.DataFrame,
    settings: Settings,
    *,
    model_path: str = MODEL_PATH,
) -> np.ndarray:
    """Gera predições do baseline para um lote validado.

    A validação não é re-executada aqui: quem chama esta função (Etapa 2) deve validar
    primeiro. Mantemos essa separação intencional: o contrato bloqueia a ingestão,
    a predição assume lote válido.
    """
    import pickle

    if not Path(model_path).exists():
        raise FileNotFoundError(
            f"Modelo não encontrado em {model_path}. Rode `make train`."
        )
    with open(model_path, "rb") as handle:
        model = pickle.load(handle)

    feature_columns = [
        c for c in frame.columns if c not in {ID_COLUMN, TARGET}
    ]
    return model.predict(frame[feature_columns])


def validate_and_predict(frame: pd.DataFrame, settings: Settings) -> np.ndarray:
    """Wrapper de conveniência: valida o lote e retorna predições."""
    validated = validate_batch(frame, name="production")
    return predict_from_model(validated, settings)


def main() -> int:
    from creditmon.config import load_settings

    settings = load_settings()
    train_baseline(settings, track=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
