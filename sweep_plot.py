"""Vergleichsgraph über die Sweep-Varianten (K8s-Hyperparameter-Sweep).

Erwartet die eingesammelten Logs des Sweeps (siehe k8s/reader-pod.yaml):

    sweep_results/<variante>/run_*.csv   # neueste CSV je Variante zählt

und zeichnet x-Position und Reward pro Episode als geglättete Kurven in einen
gemeinsamen Plot (plots/sweep.png) – der eigentliche Zweck des Sweeps: Varianten
auf einen Blick vergleichen statt vier einzelne Logdateien zu lesen.

    python sweep_plot.py --dir sweep_results --out plots/sweep.png
"""

from __future__ import annotations

import argparse
import csv
import glob
import os

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# Referenzpalette (dataviz): feste Kategorien-Reihenfolge für die Varianten.
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
COLORS = ["#2a78d6", "#e05e4e", "#2e9e78", "#b58f1e"]


def _load(path: str) -> dict[str, np.ndarray]:
    cols: dict[str, list[float]] = {}
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            for k, v in row.items():
                cols.setdefault(k, []).append(float(v) if v else 0.0)
    return {k: np.array(v) for k, v in cols.items()}


def _smooth(a: np.ndarray, window: int) -> np.ndarray:
    if len(a) < 2:
        return a
    kernel = np.ones(min(window, len(a))) / min(window, len(a))
    pad = np.concatenate([np.full(min(window, len(a)) - 1, a[0]), a])
    return np.convolve(pad, kernel, mode="valid")


def main() -> None:
    parser = argparse.ArgumentParser(description="Sweep-Varianten vergleichen.")
    parser.add_argument("--dir", default="sweep_results", help="Ordner mit <variante>/run_*.csv")
    parser.add_argument("--out", default="plots/sweep.png")
    parser.add_argument("--window", type=int, default=20, help="Glättungsfenster (Episoden)")
    args = parser.parse_args()

    variants: dict[str, dict[str, np.ndarray]] = {}
    for sub in sorted(os.listdir(args.dir)):
        csvs = sorted(glob.glob(os.path.join(args.dir, sub, "run_*.csv")))
        if csvs:
            variants[sub] = _load(csvs[-1])  # neueste CSV der Variante
    if not variants:
        raise SystemExit(f"Keine run_*.csv unter {args.dir}/<variante>/ gefunden.")

    # Drei Panels erzählen die Geschichte auch in der Frühphase des Trainings:
    # Epsilon zeigt den *Mechanismus* der decay-Variante, der Loss die
    # (In-)Stabilität der Lernraten, x_pos das (noch verrauschte) Ergebnis.
    panels = [
        ("epsilon", "Exploration: Epsilon", 1),
        ("avg_loss", "Stabilität: Trainings-Loss (geglättet)", args.window),
        ("x_pos", "Ergebnis: erreichte x-Position (geglättet)", args.window),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4), dpi=140)
    fig.patch.set_facecolor(SURFACE)
    for i, (name, df) in enumerate(variants.items()):
        color = COLORS[i % len(COLORS)]
        for ax, (col, _, window) in zip(axes, panels):
            ax.plot(df["episode"], _smooth(df[col], window),
                    label=name, linewidth=2, color=color)
    for ax, (_, title, _) in zip(axes, panels):
        ax.set_facecolor(SURFACE)
        ax.set_title(title, fontsize=10.5, color=INK)
        ax.set_xlabel("Episode", fontsize=9, color=INK_2)
        ax.grid(alpha=0.3)
        ax.tick_params(colors=MUTED, labelsize=8)
    axes[0].legend(fontsize=9)
    fig.suptitle("Hyperparameter-Sweep: 4 parallele Kubernetes-Jobs, gleiche Seed",
                 fontsize=12.5, color=INK, fontweight="bold")
    fig.tight_layout()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.savefig(args.out, facecolor=SURFACE, bbox_inches="tight")
    print(f"Plot gespeichert: {args.out} ({len(variants)} Varianten)")
    for name, df in variants.items():
        print(
            f"  {name:12s} | Episoden: {len(df['episode']):3.0f}"
            f" | best x: {df['x_pos'].max():4.0f}"
            f" | Ø Reward (letzte 20): {df['reward'][-20:].mean():7.1f}"
        )


if __name__ == "__main__":
    main()
