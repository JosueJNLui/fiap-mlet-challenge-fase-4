from __future__ import annotations

import pytest

from creditmon import train
from creditmon.config import load_settings

pytestmark = pytest.mark.usefixtures("ensure_reference")


def test_train_smoke_no_tracking() -> None:
    settings = load_settings()
    settings.dagshub_token = None
    metrics = train.train_baseline(settings, track=False)
    assert metrics["test_size"] == 6000
    assert metrics["train_size"] == 24000
    assert 0.7 < metrics["roc_auc"] < 0.8
    assert len(metrics["confusion_matrix"]) == 2
