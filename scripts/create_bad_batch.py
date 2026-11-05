"""Gera um lote com erros para demonstrar que o contrato de dados bloqueia a ingestão.

São injetadas violações escolhidas para serem óbvias e reportáveis pelo Pandera
(idade de menor, nulo, domínio inválido, alvo inválido e duplicidade de ID). O batch
é construído a partir de um pequeno subconjunto do holdout limpo, então ele nunca
entra no dataset de referência.
"""

from __future__ import annotations

from pathlib import Path

from creditmon.config import Settings
from creditmon.data import load_reference
from creditmon.observability import logger, setup_logging
from creditmon.simulation import inject_contract_violations

BAD_BATCH_NAME = "bad_batch.csv"


def create_bad_batch(settings: Settings, *, n_rows: int = 5) -> Path:
    """Cria um lote sujo e retorna o caminho para o arquivo gerado."""
    _, holdout = load_reference(settings)
    bad = inject_contract_violations(holdout.head(n_rows))

    bad_batch_dir = settings.paths.raw.parent / "bad_batch"
    bad_batch_dir.mkdir(parents=True, exist_ok=True)
    path = bad_batch_dir / BAD_BATCH_NAME
    bad.to_csv(path, index=False)
    logger.info(f"lote sujo gerado: {path}")
    logger.info(f"  {n_rows} linhas, violações deliberadas nas linhas 0 a 4")
    return path


def main() -> int:
    from creditmon.config import load_settings

    setup_logging()
    settings = load_settings()
    create_bad_batch(settings)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
