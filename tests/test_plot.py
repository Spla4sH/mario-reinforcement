"""Tests für die Plot-Helfer (plot)."""

from pathlib import Path

import numpy as np

from mario.metrics import MetricsLogger
from mario.plot import _moving_average, _read_csv, generate_plots


def test_moving_average_basic():
    vals = np.array([1.0, 2.0, 3.0, 4.0])
    ma = _moving_average(vals, window=2)
    assert ma[0] == 1.0
    assert ma[1] == 1.5
    assert ma[3] == 3.5


def test_moving_average_ignores_nan():
    vals = np.array([np.nan, 2.0, 4.0])
    ma = _moving_average(vals, window=3)
    # nan wird ignoriert -> Mittel aus 2.0 und 4.0
    assert ma[2] == 3.0


def test_read_csv_roundtrip_and_plot(tmp_path):
    log = MetricsLogger(log_dir=str(tmp_path), run_name="run_x")
    for i in range(1, 6):
        log.log_episode(i, float(i), i * 10, 100, 1.0 - i * 0.1, 0.5 / i, 60.0, i == 5)
    log.close()

    csv_path = tmp_path / "run_x.csv"
    data = _read_csv(str(csv_path))
    assert len(data["episode"]) == 5
    assert data["x_pos"][-1] == 50

    png = generate_plots(str(csv_path), out_dir=str(tmp_path / "plots"))
    assert png is not None
    assert Path(png).exists()
