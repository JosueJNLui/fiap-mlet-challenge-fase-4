#!/usr/bin/env python3
"""Baixa o dataset Default of Credit Card Clients (UCI) para `data/raw/`.

A UCI distribui um `.xls` binário OLE2 (não é CSV renomeado) e a série histórica de
faturas tem uma linha de metadados antes do cabeçalho real. O script baixa o zip, extrai
o `.xls` e converte para CSV, que é o formato em que o resto do pipeline trabalha.

Idempotente: se o CSV já existe com o mesmo número de linhas esperado, não rebaixa.
"""

from __future__ import annotations

import io
import sys
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from creditmon.config import load_settings  # noqa: E402
from creditmon.observability import logger, setup_logging, stage  # noqa: E402

UCI_URL = "https://archive.ics.uci.edu/static/public/350/default+of+credit+card+clients.zip"
XLS_NAME = "default of credit card clients.xls"
CSV_NAME = "default_of_credit_card_clients.csv"
EXPECTED_ROWS = 30_000
# `header=1`: a linha 0 do .xls é metadado da UCI ("This research was accepted by..."),
# a linha 1 é o cabeçalho real das 25 colunas.
HEADER_ROW = 1


def download_zip(url: str = UCI_URL) -> bytes:
    """Baixa o zip do dataset e devolve os bytes."""
    with urllib.request.urlopen(url) as response:  # noqa: S310 - URL fixa do UCI
        return response.read()


def xls_bytes_from_zip(payload: bytes) -> bytes:
    """Extrai o `.xls` de dentro do zip."""
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        return archive.read(XLS_NAME)


def to_dataframe(xls_payload: bytes) -> pd.DataFrame:
    """Lê o `.xls` e devolve o DataFrame com o cabeçalho real."""
    return pd.read_excel(io.BytesIO(xls_payload), header=HEADER_ROW)


def download(st: dict) -> int:
    """Baixa e converte o dataset, escrevendo o CSV em `data/raw/`."""
    settings = load_settings()
    raw_dir = settings.paths.raw
    raw_dir.mkdir(parents=True, exist_ok=True)
    csv_path = raw_dir / CSV_NAME

    if csv_path.exists():
        rows = sum(1 for _ in csv_path.open()) - 1
        if rows == EXPECTED_ROWS:
            logger.info(f"Dataset já presente: {csv_path} ({rows} linhas). Nada a fazer.")
            st["rows_out"] = rows
            return 0
        logger.info(f"CSV com {rows} linhas (esperado {EXPECTED_ROWS}); rebaixando.")

    logger.info(f"Baixando {UCI_URL}")
    xls_payload = xls_bytes_from_zip(download_zip())
    frame = to_dataframe(xls_payload)

    if len(frame) != EXPECTED_ROWS:
        raise ValueError(
            f"esperado {EXPECTED_ROWS} linhas, veio {len(frame)}. "
            "A UCI pode ter trocado o arquivo."
        )

    xls_path = raw_dir / XLS_NAME
    xls_path.write_bytes(xls_payload)
    frame.to_csv(csv_path, index=False)

    logger.info(f"{xls_path} ({xls_path.stat().st_size:,} bytes)")
    logger.info(f"{csv_path} ({csv_path.stat().st_size:,} bytes)")
    logger.info(f"{len(frame)} linhas, {len(frame.columns)} colunas")
    logger.info(f"Colunas: {', '.join(frame.columns)}")
    st["rows_out"] = len(frame)
    return 0


def main() -> int:
    setup_logging()
    try:
        with stage("data_download") as st:
            return download(st)
    except Exception:  # noqa: BLE001 - o evento `failed` já foi registrado pelo stage
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
