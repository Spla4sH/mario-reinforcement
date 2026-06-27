"""Tests für das CSV-Logging (metrics.MetricsLogger)."""

import csv

from metrics import FIELDNAMES, MetricsLogger


def test_writes_header_and_rows(tmp_path):
    log = MetricsLogger(log_dir=str(tmp_path), run_name="t")
    log.log_episode(1, 10.0, 100, 50, 0.5, 0.1, 60.0, False)
    log.log_episode(2, 20.0, 200, 60, 0.4, None, 55.0, True)
    log.close()

    csv_path = tmp_path / "t.csv"
    assert csv_path.exists()

    with open(csv_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    assert len(rows) == 2
    assert list(rows[0].keys()) == FIELDNAMES
    assert rows[0]["reward"] == "10.0"
    assert rows[1]["flag_get"] == "1"
    # None-Loss wird als leeres Feld geschrieben
    assert rows[1]["avg_loss"] == ""


def test_flag_get_is_zero_when_false(tmp_path):
    log = MetricsLogger(log_dir=str(tmp_path), run_name="t2")
    log.log_episode(1, 1.0, 1, 1, 1.0, 0.0, 1.0, False)
    log.close()
    with open(tmp_path / "t2.csv", newline="", encoding="utf-8") as f:
        row = next(csv.DictReader(f))
    assert row["flag_get"] == "0"
