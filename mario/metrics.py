"""Persistentes Logging der Trainingsmetriken in eine CSV-Datei."""

import csv
import time
from datetime import datetime
from pathlib import Path


# Spalten der CSV (Reihenfolge = Schreibreihenfolge)
FIELDNAMES = [
    "episode",
    "reward",
    "x_pos",
    "steps",
    "epsilon",
    "avg_loss",
    "fps",
    "flag_get",
    "elapsed_total",
]


class MetricsLogger:
    """Schreibt pro Episode eine Zeile mit Trainingsstatistiken in eine CSV.

    Die Datei wird nach jeder Episode geflusht, damit die Daten auch bei
    einem Abbruch (z. B. Strg+C) erhalten bleiben.
    """

    def __init__(self, log_dir="logs", run_name=None):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)

        if run_name is None:
            run_name = f"run_{datetime.now():%Y%m%d_%H%M%S}"
        self.run_name = run_name
        self.csv_path = self.log_dir / f"{run_name}.csv"

        self._start_time = time.time()
        self._file = open(self.csv_path, "w", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._file, fieldnames=FIELDNAMES)
        self._writer.writeheader()
        self._file.flush()

    def log_episode(
        self, episode, reward, x_pos, steps, epsilon, avg_loss, fps, flag_get
    ):
        """Hängt eine Episode-Zeile an und flusht sofort."""
        self._writer.writerow(
            {
                "episode": episode,
                "reward": round(float(reward), 2),
                "x_pos": int(x_pos),
                "steps": int(steps),
                "epsilon": round(float(epsilon), 4),
                "avg_loss": round(float(avg_loss), 5) if avg_loss is not None else "",
                "fps": round(float(fps), 1),
                "flag_get": int(bool(flag_get)),
                "elapsed_total": round(time.time() - self._start_time, 1),
            }
        )
        self._file.flush()

    def close(self):
        if not self._file.closed:
            self._file.close()
