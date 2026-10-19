from __future__ import annotations

import pandas as pd
import pytest

from creditmon import data, schema
from creditmon.config import load_settings
from creditmon.schema import DataContractError

pytestmark = pytest.mark.usefixtures("ensure_reference")


def _load_holdout() -> pd.DataFrame:
    _, holdout = data.load_reference(load_settings())
    return holdout


def test_schema_accepts_clean_reference() -> None:
    schema.validate_batch(_load_holdout().head(10), name="clean")


def test_schema_rejects_age_under_18() -> None:
    bad = _load_holdout().head(2).copy()
    bad.loc[bad.index[0], "AGE"] = 17
    with pytest.raises(DataContractError):
        schema.validate_batch(bad, name="age-violation")


def test_schema_rejects_limit_bal_null() -> None:
    bad = _load_holdout().head(2).copy()
    bad.loc[bad.index[0], "LIMIT_BAL"] = pd.NA
    with pytest.raises(DataContractError):
        schema.validate_batch(bad, name="null-violation")


def test_schema_rejects_invalid_education() -> None:
    bad = _load_holdout().head(2).copy()
    bad.loc[bad.index[0], "EDUCATION"] = 99
    with pytest.raises(DataContractError):
        schema.validate_batch(bad, name="edu-violation")


def test_schema_rejects_invalid_target() -> None:
    bad = _load_holdout().head(2).copy()
    bad.loc[bad.index[0], "default payment next month"] = 2
    with pytest.raises(DataContractError):
        schema.validate_batch(bad, name="target-violation")


def test_schema_rejects_duplicate_id() -> None:
    bad = _load_holdout().head(2).copy()
    bad.loc[bad.index[1], "ID"] = bad.loc[bad.index[0], "ID"]
    with pytest.raises(DataContractError):
        schema.validate_batch(bad, name="dup-violation")
