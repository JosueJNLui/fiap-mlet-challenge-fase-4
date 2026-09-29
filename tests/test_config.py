from __future__ import annotations

import pytest

from creditmon.config import Settings
from creditmon.tracking import init_mlflow


def test_dagshub_urls_are_derived_from_owner_and_repo() -> None:
    settings = Settings(dagshub_repo_owner="acme", dagshub_repo_name="fase-4")
    assert settings.dagshub_repo_slug == "acme/fase-4"
    assert settings.mlflow_tracking_uri == "https://dagshub.com/acme/fase-4.mlflow"
    assert settings.dvc_remote_url == "https://dagshub.com/acme/fase-4.dvc"


def test_init_mlflow_fails_without_token() -> None:
    settings = Settings(dagshub_token=None)
    with pytest.raises(RuntimeError, match="DAGSHUB_TOKEN"):
        init_mlflow(settings)
