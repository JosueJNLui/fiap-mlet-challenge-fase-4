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


def main() -> int:
    """Baixa, converte e valida o dataset, escrevendo o CSV em `data/raw/`."""
    settings = load_settings()
    raw_dir = settings.paths.raw
    raw_dir.mkdir(parents=True, exist_ok=True)
    csv_path = raw_dir / CSV_NAME

    if csv_path.exists():
        rows = sum(1 for _ in csv_path.open()) - 1
        if rows == EXPECTED_ROWS:
            print(f"Dataset ja presente: {csv_path} ({rows} linhas). Nada a fazer.")
            return 0
        print(f"CSV com {rows} linhas (esperado {EXPECTED_ROWS}); rebaixando.")

    print(f"Baixando {UCI_URL}")
    xls_payload = xls_bytes_from_zip(download_zip())
    frame = to_dataframe(xls_payload)

    if len(frame) != EXPECTED_ROWS:
        print(
            f"ERRO: esperado {EXPECTED_ROWS} linhas, veio {len(frame)}. "
            "A UCI pode ter trocado o arquivo.",
            file=sys.stderr,
        )
        return 1

    xls_path = raw_dir / XLS_NAME
    xls_path.write_bytes(xls_payload)
    frame.to_csv(csv_path, index=False)

    print(f"{xls_path} ({xls_path.stat().st_size:,} bytes)")
    print(f"{csv_path} ({csv_path.stat().st_size:,} bytes)")
    print(f"{len(frame)} linhas, {len(frame.columns)} colunas")
    print(f"Colunas: {', '.join(frame.columns)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
