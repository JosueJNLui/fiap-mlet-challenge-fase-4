#!/usr/bin/env python3
"""Valida o ambiente: versao do Python, deps criticas, .env e credenciais DagsHub.

Falhas *hard* (versao do Python, imports faltando) retornam codigo != 0. As demais
(`.env`, token) sao avisos: o script passa logo apos `make install`, antes de o usuario
criar o `.env`.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from creditmon.config import load_settings  # noqa: E402

CHECK, CROSS, WARN = "✓", "✗", "!"
CRITICAL_IMPORTS = ("pandas", "sklearn", "xgboost", "pandera", "evidently", "mlflow")
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def check_python() -> tuple[bool, str, str]:
    """Confere se a versao do Python esta em >=3.12,<3.14 (ver pyproject.toml)."""
    major, minor = sys.version_info[:2]
    ok = major == 3 and minor in (12, 13)
    hint = "" if ok else "Use Python 3.12.x ou 3.13.x (requires-python = >=3.12,<3.14)."
    return ok, f"Python {major}.{minor} (esperado 3.12.x ou 3.13.x)", hint


def check_import(module: str) -> tuple[bool, str, str]:
    """Verifica se um modulo critico e importavel no ambiente atual."""
    ok = importlib.util.find_spec(module) is not None
    hint = "" if ok else f"Dep '{module}' ausente. Rode `make install`."
    return ok, f"import {module}", hint


def check_env_file() -> tuple[bool, str, str]:
    """Avisa se o .env ainda nao existe."""
    ok = (PROJECT_ROOT / ".env").exists()
    return ok, ".env presente", "" if ok else "Rode `cp .env.example .env` e preencha."


def check_token() -> tuple[bool, str, str]:
    """Avisa se DAGSHUB_TOKEN nao esta definido (necessario para MLflow e DVC)."""
    ok = bool(load_settings().dagshub_token)
    return ok, "DAGSHUB_TOKEN definido", "" if ok else "Defina DAGSHUB_TOKEN no .env."


def report(label: str, ok: bool, hint: str, hard: bool) -> None:
    """Imprime uma linha do relatorio com o simbolo adequado."""
    mark = CHECK if ok else (CROSS if hard else WARN)
    tail = f"  → {hint}" if hint and not ok else ""
    print(f"  {mark} {label}{tail}")


def main() -> int:
    """Roda todos os checks; retorna 1 se algum check *hard* falhar."""
    hard = [check_python(), *(check_import(m) for m in CRITICAL_IMPORTS)]
    soft = [check_env_file(), check_token()]

    print("Checks obrigatorios:")
    hard_failed = False
    for ok, label, hint in hard:
        report(label, ok, hint, hard=True)
        hard_failed = hard_failed or not ok

    print("\nAvisos (nao bloqueiam):")
    for ok, label, hint in soft:
        report(label, ok, hint, hard=False)

    if hard_failed:
        print("\nAmbiente incompleto: corrija os itens marcados com ✗.", file=sys.stderr)
        return 1
    print("\nAmbiente OK.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
