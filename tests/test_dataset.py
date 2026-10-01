from __future__ import annotations

import csv
from pathlib import Path

import pytest

from creditmon.config import load_settings

# 25 colunas da UCI: 1 ID, 4 categoricas, 1 idade, 6 status de pagamento,
# 6 faturas, 6 pagamentos e o alvo binario.
EXPECTED_COLUMNS = [
    "ID",
    "LIMIT_BAL",
    "SEX",
    "EDUCATION",
    "MARRIAGE",
    "AGE",
    "PAY_0",
    "PAY_2",
    "PAY_3",
    "PAY_4",
    "PAY_5",
    "PAY_6",
    "BILL_AMT1",
    "BILL_AMT2",
    "BILL_AMT3",
    "BILL_AMT4",
    "BILL_AMT5",
    "BILL_AMT6",
    "PAY_AMT1",
    "PAY_AMT2",
    "PAY_AMT3",
    "PAY_AMT4",
    "PAY_AMT5",
    "PAY_AMT6",
    "default payment next month",
]


@pytest.fixture(scope="module")
def raw_dir() -> Path:
    """Diretorio raw do projeto, pulando o teste se o dataset nao foi baixado."""
    path = load_settings().paths.raw
    if not (path / "default_of_credit_card_clients.csv").exists():
        pytest.skip("dataset ausente: rode `make data-download`")
    return path


def test_csv_has_the_documented_schema(raw_dir: Path) -> None:
    """O CSV baixado tem as 25 colunas da UCI, na ordem original."""
    with (raw_dir / "default_of_credit_card_clients.csv").open() as handle:
        header = next(csv.reader(handle))
    assert header == EXPECTED_COLUMNS


def test_csv_has_30k_rows_and_binary_target(raw_dir: Path) -> None:
    """30.000 linhas e alvo em {0, 1}: o minimo de 5.000 do brief, com folga."""
    with (raw_dir / "default_of_credit_card_clients.csv").open() as handle:
        reader = csv.DictReader(handle)
        target = "default payment next month"
        rows = list(reader)
    assert len(rows) == 30_000
    assert {row[target] for row in rows} == {"0", "1"}


def test_xls_source_is_kept_next_to_the_csv(raw_dir: Path) -> None:
    """O `.xls` original fica em data/raw ao lado do CSV, para reprodutibilidade."""
    source = raw_dir / "default of credit card clients.xls"
    assert source.exists()
    assert source.read_bytes()[:8] == b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1", "esperado OLE2"
