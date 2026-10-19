"""Gera um lote com erros para demonstrar que o contrato de dados bloqueia a ingestão.

São injetadas violações escolhidas para serem óbvias e reportáveis pelo Pandera
(idade de menor, nulo, domínio inválido, alvo inválido e duplicidade de ID). O batch
é construído a partir de um pequeno subconjunto do holdout limpo, então ele nunca
entra no dataset de referência.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from creditmon.config import Settings
from creditmon.data import load_reference
from creditmon.schema import ID_COLUMN, TARGET

BAD_BATCH_NAME = "bad_batch.csv"


def create_bad_batch(settings: Settings, *, n_rows: int = 5) -> Path:
    """Cria um lote sujo e retorna o caminho para o arquivo gerado."""
    _, holdout = load_reference(settings)
    bad = holdout.head(n_rows).copy()

    # 1) Nulo em LIMIT_BAL (viola "não pode ser nulo" + coerção se vier como NaN)
    bad.loc[bad.index[0], "LIMIT_BAL"] = pd.NA

    # 2) Idade < 18
    bad.loc[bad.index[1], "AGE"] = 15

    # 3) Fora do domínio de EDUCATION (0..6)
    bad.loc[bad.index[2], "EDUCATION"] = 99

    # 4) Alvo fora de {0,1}
    bad.loc[bad.index[3], TARGET] = 2

    # 5) Duplicidade de ID + limite de renda plausível, mas o ID já existe no batch
    bad.loc[bad.index[4], ID_COLUMN] = bad.loc[bad.index[0], ID_COLUMN]

    bad_batch_dir = settings.paths.raw.parent / "bad_batch"
    bad_batch_dir.mkdir(parents=True, exist_ok=True)
    path = bad_batch_dir / BAD_BATCH_NAME
    bad.to_csv(path, index=False)
    print(f"lote sujo gerado: {path}")
    print(f"  {n_rows} linhas, violações deliberadas no índice 0–4")
    return path


def main() -> int:
    from creditmon.config import load_settings

    settings = load_settings()
    create_bad_batch(settings)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
