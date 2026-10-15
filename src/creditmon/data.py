"""Carga, separação e versionamento do dataset de crédito.

Duas responsabilidades:

1. Carregar o CSV cru e passá-lo pelo contrato de dados (nada entra sem validação).
2. Congelar o **dataset de referência** (treino) e um **holdout** limpo. A referência é
   o que a Etapa 2 usa como linha de base para medir drift contra o dataset de produção.
   Sem esse congelamento, "drift" vira uma comparação sem ponto de partida.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from creditmon.config import Settings
from creditmon.schema import TARGET, validate_batch

RAW_CSV_NAME = "default_of_credit_card_clients.csv"
REFERENCE_CSV_NAME = "reference.csv"
HOLDOUT_CSV_NAME = "holdout.csv"

# Fração reservada para avaliar o baseline. O restante é a referência de treino.
HOLDOUT_FRACTION = 0.2


def load_raw(settings: Settings) -> pd.DataFrame:
    """Carrega o CSV cru baixado e valida contra o contrato de dados."""
    csv_path = settings.paths.raw / RAW_CSV_NAME
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Dataset ausente em {csv_path}. Rode `make data-download`."
        )
    return validate_batch(pd.read_csv(csv_path), name="raw")


def reference_paths(settings: Settings) -> tuple[Path, Path]:
    """Caminhos do dataset de referência e do holdout."""
    directory = settings.paths.processed.parent / "reference"
    return directory / REFERENCE_CSV_NAME, directory / HOLDOUT_CSV_NAME


def split_reference(frame: pd.DataFrame, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Separa o dataset em referência (treino) e holdout, estratificado pelo alvo."""
    holdout = frame.sample(frac=HOLDOUT_FRACTION, random_state=seed)
    reference = frame.drop(holdout.index)
    return reference.reset_index(drop=True), holdout.reset_index(drop=True)


def build_reference(settings: Settings) -> tuple[Path, Path]:
    """Valida o dataset cru e grava a referência e o holdout congelados.

    Returns:
        Os caminhos de `reference.csv` e `holdout.csv`.
    """
    reference, holdout = split_reference(load_raw(settings), settings.seed)

    reference_csv, holdout_csv = reference_paths(settings)
    reference_csv.parent.mkdir(parents=True, exist_ok=True)
    reference.to_csv(reference_csv, index=False)
    holdout.to_csv(holdout_csv, index=False)

    print(f"{reference_csv} ({len(reference):,} linhas)")
    print(f"{holdout_csv} ({len(holdout):,} linhas)")
    print(f"taxa de default - referencia: {reference[TARGET].mean():.4f}")
    print(f"taxa de default - holdout:    {holdout[TARGET].mean():.4f}")
    return reference_csv, holdout_csv


def load_reference(settings: Settings) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Carrega a referência e o holdout congelados (rode `make reference` antes)."""
    reference_csv, holdout_csv = reference_paths(settings)
    missing = [p for p in (reference_csv, holdout_csv) if not p.exists()]
    if missing:
        raise FileNotFoundError(
            f"Dataset de referência ausente ({', '.join(str(p) for p in missing)}). "
            "Rode `make reference`."
        )
    return pd.read_csv(reference_csv), pd.read_csv(holdout_csv)


def main() -> int:
    """Congela a referência e o holdout a partir do dataset cru."""
    from creditmon.config import load_settings

    settings = load_settings()
    if not settings.dagshub_token:
        # A validação da referência não precisa de DagsHub; o treino, sim.
        print("(aviso) DAGSHUB_TOKEN ausente: a referência é gerada, mas `make train` vai exigir.")
    build_reference(settings)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
