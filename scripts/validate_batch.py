#!/usr/bin/env python3
"""Valida um lote CSV contra o contrato de dados (Pandera).

Uso:
  uv run python scripts/validate_batch.py data/bad_batch/bad_batch.csv
  uv run python scripts/validate_batch.py data/raw/default_of_credit_card_clients.csv

O script falha com código != 0 caso o lote seja rejeitado: é isso que
demonstra que o contrato "bloqueia" a ingestão de lixo.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pandas as pd

from creditmon.config import load_settings  # noqa: E402
from creditmon.observability import setup_logging, stage  # noqa: E402
from creditmon.schema import DataContractError, validate_batch  # noqa: E402
from creditmon.tracking import log_stage, tracked_run  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Valida um lote CSV contra o contrato de dados.")
    parser.add_argument(
        "csv",
        nargs="?",
        default="data/raw/default_of_credit_card_clients.csv",
        help="Caminho para o CSV a ser validado.",
    )
    args = parser.parse_args()
    log = setup_logging()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        print(f"ERRO: arquivo não encontrado: {csv_path}", file=sys.stderr)
        return 2

    try:
        frame = pd.read_csv(csv_path)
    except Exception as error:  # noqa: BLE001
        print(f"ERRO: não foi possível ler {csv_path}: {error}", file=sys.stderr)
        return 2

    with tracked_run(load_settings(), "validate_batch", batch=csv_path.name):
        try:
            with stage("validate_batch", batch=csv_path.name, rows_in=len(frame)) as st:
                st["rows_out"] = len(validate_batch(frame, name=csv_path.name))
            log.info(f"✓ OK: {csv_path.name} passou no contrato ({len(frame):,} linhas)")
            return 0
        except DataContractError as error:
            log.warning(f"✗ BLOQUEADO: {csv_path.name} NÃO passou no contrato\n{error.summary}")
            return 1
        finally:
            log_stage(st)

if __name__ == "__main__":
    raise SystemExit(main())
