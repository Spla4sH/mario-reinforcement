"""Erzeugt Graphen aus einer Trainings-CSV (siehe metrics.py).

Aufruf:
    python plot.py [csv_pfad]

Ohne Argument wird automatisch die neueste CSV in logs/ verwendet.
"""

import sys
import csv
import glob
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")  # Kein GUI-Backend nötig, wir speichern nur PNGs
import matplotlib.pyplot as plt


def latest_csv(log_dir="logs"):
    """Gibt den Pfad zur neuesten run_*.csv zurück (oder None)."""
    files = sorted(glob.glob(str(Path(log_dir) / "run_*.csv")))
    return files[-1] if files else None


def _read_csv(csv_path):
    """Liest die CSV als Spalten-Dictionary aus Floats (leere Felder -> nan)."""
    columns = {}
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for field in reader.fieldnames:
            columns[field] = []
        for row in reader:
            for field in reader.fieldnames:
                value = row[field]
                columns[field].append(float(value) if value != "" else np.nan)
    return {k: np.array(v, dtype=np.float64) for k, v in columns.items()}


def _moving_average(values, window=100):
    """Gleitender Durchschnitt; ignoriert nan-Werte."""
    if len(values) == 0:
        return values
    window = min(window, len(values))
    result = np.full(len(values), np.nan)
    for i in range(len(values)):
        chunk = values[max(0, i - window + 1) : i + 1]
        chunk = chunk[~np.isnan(chunk)]
        if len(chunk) > 0:
            result[i] = chunk.mean()
    return result


def generate_plots(csv_path, out_dir="plots", window=100):
    """Liest die CSV und schreibt ein PNG mit vier Subplots.

    Gibt den Pfad zum erzeugten PNG zurück.
    """
    data = _read_csv(csv_path)
    episodes = data.get("episode", np.array([]))
    if len(episodes) == 0:
        print(f"Keine Datenzeilen in {csv_path} – überspringe Plot.")
        return None

    out_path = Path(out_dir)
    out_path.mkdir(exist_ok=True)
    png_path = out_path / (Path(csv_path).stem + ".png")

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle(f"Mario RL – {Path(csv_path).stem}", fontsize=14)

    # Belohnung
    ax = axes[0, 0]
    ax.plot(episodes, data["reward"], color="#9ecae1", linewidth=0.8, label="Reward")
    ax.plot(
        episodes,
        _moving_average(data["reward"], window),
        color="#08519c",
        linewidth=1.8,
        label=f"Ø {window}",
    )
    ax.set_title("Belohnung pro Episode")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Reward")
    ax.legend()
    ax.grid(alpha=0.3)

    # x_pos (Level-Fortschritt)
    ax = axes[0, 1]
    ax.plot(episodes, data["x_pos"], color="#a1d99b", linewidth=0.8, label="x_pos")
    ax.plot(
        episodes,
        _moving_average(data["x_pos"], window),
        color="#006d2c",
        linewidth=1.8,
        label=f"Ø {window}",
    )
    ax.set_title("Level-Fortschritt (max. x-Position)")
    ax.set_xlabel("Episode")
    ax.set_ylabel("x_pos")
    ax.legend()
    ax.grid(alpha=0.3)

    # Epsilon
    ax = axes[1, 0]
    ax.plot(episodes, data["epsilon"], color="#fd8d3c", linewidth=1.2)
    ax.set_title("Exploration (Epsilon)")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Epsilon")
    ax.grid(alpha=0.3)

    # Loss
    ax = axes[1, 1]
    ax.plot(episodes, data["avg_loss"], color="#dd6e6e", linewidth=0.8, label="Loss")
    ax.plot(
        episodes,
        _moving_average(data["avg_loss"], window),
        color="#a50f15",
        linewidth=1.8,
        label=f"Ø {window}",
    )
    ax.set_title("Durchschnittlicher Loss pro Episode")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Loss")
    ax.legend()
    ax.grid(alpha=0.3)

    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(png_path, dpi=100)
    plt.close(fig)
    return png_path


def main():
    csv_path = sys.argv[1] if len(sys.argv) > 1 else latest_csv()
    if not csv_path:
        print("Keine CSV gefunden. Starte zuerst ein Training (python train.py).")
        sys.exit(1)
    png = generate_plots(csv_path)
    if png:
        print(f"Graphen gespeichert: {png}")


if __name__ == "__main__":
    main()
