from __future__ import annotations

import pytest

from creditmon import data
from creditmon.config import load_settings


@pytest.fixture(scope="session")
def ensure_reference():
    """Garante que data/reference/ existe, construindo se necessario."""
    settings = load_settings()
    try:
        data.load_reference(settings)
    except FileNotFoundError:
        data.build_reference(settings)
