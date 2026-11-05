from __future__ import annotations

import json
import logging

import pytest
from tests.factories import credit_frame

from creditmon import observability
from creditmon.observability import setup_logging, stage
from creditmon.schema import DataContractError, validate_batch
from creditmon.simulation import inject_contract_violations


@pytest.fixture
def log_file(tmp_path):
    path = tmp_path / "pipeline_runs.jsonl"
    setup_logging(path)
    yield path
    for handler in list(observability.logger.handlers):
        observability.logger.removeHandler(handler)
        handler.close()


def events(path) -> list[dict]:
    lines = path.read_text(encoding="utf-8").splitlines()
    return [e for e in map(json.loads, lines) if e.get("event") == "stage"]


def test_stage_records_success_with_volume(log_file) -> None:
    with stage("reference", batch="x") as st:
        st.update(rows_in=10, rows_out=8)

    (event,) = events(log_file)
    assert event["stage"] == "reference"
    assert event["status"] == "success"
    assert (event["rows_in"], event["rows_out"], event["batch"]) == (10, 8, "x")
    assert event["duration_s"] >= 0 and event["error"] is None
    assert event["run_id"] == observability.RUN_ID and event["level"] == "INFO"


def test_stage_records_failure_and_reraises(log_file) -> None:
    with pytest.raises(RuntimeError), stage("train"):
        raise RuntimeError("boom")

    (event,) = events(log_file)
    assert event["status"] == "failed"
    assert event["error"] == "RuntimeError: boom"
    assert event["level"] == logging.getLevelName(logging.ERROR)


def test_corrupted_batch_is_blocked_by_the_contract(log_file) -> None:
    corrupted = inject_contract_violations(credit_frame(50))

    with pytest.raises(DataContractError), stage("ingest_production", rows_in=50):
        validate_batch(corrupted, name="corrupted")

    (event,) = events(log_file)
    assert event["status"] == "blocked"
    assert event["failure_count"] >= 5
    assert event["rows_out"] is None
    assert "AGE" in event["error"] and "EDUCATION" in event["error"]


def test_clean_batch_passes_the_contract() -> None:
    frame = credit_frame(50)
    assert len(validate_batch(frame, name="clean")) == 50
