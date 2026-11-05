"""Logs estruturados e métricas de execução das etapas do pipeline.

Usa só o ``logging`` da stdlib. Duas saídas:

* terminal: mensagem legível, como os antigos ``print``;
* ``reports/monitoring/pipeline_runs.jsonl``: um JSON por linha (com ``run_id`` da
  execução), fonte do dashboard e do MLflow da Etapa 3.

``stage(name)`` envolve uma etapa e grava um evento ``{"event": "stage", ...}`` com
``status`` (``success``/``failed``/``blocked``), ``duration_s``, ``rows_in``,
``rows_out`` e o erro, se houver. Lote rejeitado pelo contrato vira ``blocked``, não
``failed``: o pipeline funcionou, quem falhou foi o dado.
"""

from __future__ import annotations

import json
import logging
import sys
import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from creditmon.schema import DataContractError

LOG_PATH = Path("reports/monitoring/pipeline_runs.jsonl")
LOGGER_NAME = "creditmon"
RUN_ID = uuid.uuid4().hex[:12]

logger = logging.getLogger(LOGGER_NAME)


class JsonFormatter(logging.Formatter):
    """Uma linha JSON por registro; campos de ``extra={"event": {...}}`` vão para o topo."""

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": datetime.fromtimestamp(record.created, UTC).isoformat(timespec="milliseconds"),
            "run_id": RUN_ID,
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            **getattr(record, "event", {}),
        }
        return json.dumps(payload, ensure_ascii=False, default=str)


def setup_logging(log_path: Path = LOG_PATH) -> logging.Logger:
    """Configura o logger ``creditmon`` (terminal + JSONL). Chamar uma vez, no ``main``."""
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()
    log_path.parent.mkdir(parents=True, exist_ok=True)

    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(logging.Formatter("%(message)s"))
    file = logging.FileHandler(log_path, encoding="utf-8")
    file.setFormatter(JsonFormatter())

    logger.addHandler(console)
    logger.addHandler(file)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    return logger


@contextmanager
def stage(name: str, **fields: Any) -> Iterator[dict[str, Any]]:
    """Mede e registra uma etapa do pipeline.

    O dicionário devolvido pode ser preenchido dentro do bloco (``rows_in``,
    ``rows_out`` ou qualquer métrica extra) e vai junto no evento. Exceções são
    registradas e relançadas: quem chama decide se o pipeline segue.
    """
    info: dict[str, Any] = {"rows_in": None, "rows_out": None, **fields}
    status, error = "success", None
    start = time.perf_counter()
    try:
        yield info
    except DataContractError as exc:
        status, error = "blocked", exc.summary
        info["failure_count"] = exc.failure_count
        raise
    except Exception as exc:
        status, error = "failed", f"{type(exc).__name__}: {exc}"
        raise
    finally:
        duration = round(time.perf_counter() - start, 3)
        event = {
            "event": "stage",
            "stage": name,
            "status": status,
            "duration_s": duration,
            **info,
            "error": error,
        }
        level = {"success": logging.INFO, "blocked": logging.WARNING}.get(status, logging.ERROR)
        label = " ".join(f"{k}={v}" for k, v in fields.items())
        logger.log(
            level,
            f"[{name}{' ' + label if label else ''}] {status} em {duration:.2f}s",
            extra={"event": event},
        )
