"""Fábrica de lotes sintéticos no formato do contrato (25 colunas da UCI)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from creditmon.schema import (
    BILL_AMOUNT_COLUMNS,
    ID_COLUMN,
    PAY_AMOUNT_COLUMNS,
    PAY_STATUS_COLUMNS,
    TARGET,
)


def credit_frame(n: int = 1000, seed: int = 0) -> pd.DataFrame:
    """Lote válido pelo contrato, com relação fraca entre atraso e default."""
    rng = np.random.default_rng(seed)
    frame = pd.DataFrame(
        {
            ID_COLUMN: np.arange(1, n + 1),
            "LIMIT_BAL": rng.choice(np.arange(10_000, 500_001, 10_000), n),
            "SEX": rng.choice([1, 2], n),
            "EDUCATION": rng.choice([1, 2, 3, 4], n),
            "MARRIAGE": rng.choice([1, 2, 3], n),
            "AGE": rng.integers(21, 70, n),
        }
    )
    for col in PAY_STATUS_COLUMNS:
        frame[col] = rng.integers(-2, 4, n)
    for col in BILL_AMOUNT_COLUMNS:
        frame[col] = rng.integers(0, 100_000, n)
    for col in PAY_AMOUNT_COLUMNS:
        frame[col] = rng.integers(0, 10_000, n)
    frame[TARGET] = (rng.random(n) < 0.2 + 0.15 * (frame["PAY_0"] > 0)).astype(int)
    return frame
