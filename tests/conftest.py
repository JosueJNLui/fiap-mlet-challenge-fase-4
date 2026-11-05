from __future__ import annotations

import pytest

from creditmon import data
from creditmon.config import load_settings


@pytest.fixture(scope="session")
def ensure_reference():
    """Garante que data/reference/ existe, construindo se necessário.

    Num clone sem o dataset (ex.: CI), os testes que dependem dele são pulados em vez de
    falhar.
    """
    settings = load_settings()
    try:
        data.load_reference(settings)
    except FileNotFoundError:
        try:
            data.build_reference(settings)
        except FileNotFoundError as error:
            pytest.skip(f"{error} Ou `make data-pull`.")
