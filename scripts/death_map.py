"""„Wo stirbt Mario?" – Todes-Dichtekurven über einem echten Level-Panorama.

Zwei Teile:
1. **Panorama**: Ein deterministischer Greedy-Lauf (mario_best.pt) durch 1-1.
   Pro Frame wird die Kameraposition aus dem NES-RAM abgeleitet
   (Level-x aus info minus Marios Bildschirm-x aus RAM $03AD) und die
   sichtbaren Bildspalten in einen langen Level-Streifen kopiert.
2. **Ridgeline**: Die Todespositionen aus dem DQN-Trainingslog (5.409 Episoden),
   in sechs Trainingsphasen geteilt und als geglättete Dichtekurven über dem
   Panorama gezeichnet – man sieht, wie die Todes-Hotspots vom Rohr über die
   Grube nach rechts wandern, während die Flaggen-Quote steigt.

Ausführen im DQN-venv (braucht checkpoints/mario_best.pt):
    python death_map.py            # schreibt death_heatmap.png (README-Asset)
"""

from __future__ import annotations

import argparse
import csv
import sys

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

for _s in (sys.stdout,):
    try:
        _s.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass

# --- Referenzpalette (dataviz): Surface/Ink + sequentieller Blau-Ramp ---------
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
BASELINE = "#c3c2b7"
PHASE_RAMP = ["#9ec5f4", "#6da7ec", "#3987e5", "#2a78d6", "#1c5cab", "#0d366b"]

HUD_ROWS = 32  # Statuszeile oben abschneiden (Punkte/Zeit gehören nicht ins Level)
SKY = (104, 136, 252)  # SMB-Himmelblau für nie gesehene Spalten


def build_panorama(x_max: int, checkpoint: str) -> np.ndarray:
    """Greedy-Lauf aufnehmen und die Frames zu einem Level-Streifen vernähen."""
    from mario.agent import MarioAgent
    from mario.wrappers import create_env

    env = create_env(world=1, stage=1, render=False)
    nes = env.unwrapped
    agent = MarioAgent(env.action_space.n)
    agent.load(checkpoint)

    height = 240 - HUD_ROWS
    canvas = np.full((height, x_max, 3), SKY, dtype=np.uint8)
    filled = np.zeros(x_max, dtype=bool)

    state = env.reset()
    done, info = False, {}
    while not done:
        frame = np.array(env.render(mode="rgb_array"), copy=True)[HUD_ROWS:]
        x_level = int(info.get("x_pos", 40))
        camera = max(0, x_level - int(nes.ram[0x03AD]))
        lo, hi = camera, min(camera + frame.shape[1], x_max)
        new = ~filled[lo:hi]
        canvas[:, lo:hi][:, new] = frame[:, : hi - lo][:, new]
        filled[lo:hi] |= True
        state, _, done, info = env.step(agent.act(state))
    env.close()
    print(f"Panorama: Lauf bis x={info.get('x_pos')} | Flagge: {bool(info.get('flag_get'))}"
          f" | {filled.sum()} von {x_max} Spalten gesehen")
    return canvas


def death_curves(csv_path: str, n_phases: int, x_max: int, bin_px: int = 25):
    """Todespositionen je Trainingsphase als geglättete %-Dichten."""
    episodes, xs, flags = [], [], []
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            episodes.append(int(row["episode"]))
            xs.append(int(float(row["x_pos"])))
            flags.append(row["flag_get"] in ("1", "True", "true"))
    xs, flags = np.array(xs), np.array(flags)
    n = len(xs)
    print(f"{n} Episoden, {flags.sum()} Flaggen")

    edges = np.arange(0, x_max + bin_px, bin_px)
    centers = (edges[:-1] + edges[1:]) / 2
    kernel = np.exp(-0.5 * (np.arange(-4, 5) / 1.6) ** 2)
    kernel /= kernel.sum()

    phase_edges = np.linspace(0, n, n_phases + 1, dtype=int)
    curves, rates, labels = [], [], []
    for p in range(n_phases):
        sl = slice(phase_edges[p], phase_edges[p + 1])
        deaths = xs[sl][~flags[sl]]
        hist, _ = np.histogram(np.clip(deaths, 0, x_max - 1), bins=edges)
        dens = hist / (phase_edges[p + 1] - phase_edges[p]) * 100.0  # % der Episoden
        curves.append(np.convolve(dens, kernel, mode="same"))
        rates.append(flags[sl].mean() * 100.0)
        labels.append(f"Ep. {phase_edges[p] + 1}–{phase_edges[p + 1]}")
    return centers, curves, rates, labels


def main() -> None:
    parser = argparse.ArgumentParser(description="Todes-Dichten über dem Level-Panorama.")
    parser.add_argument("--csv", default="logs/run_20260703_133635.csv")
    parser.add_argument("--checkpoint", default="checkpoints/mario_best.pt")
    parser.add_argument("--out", default="assets/death_heatmap.png")
    parser.add_argument("--phases", type=int, default=6)
    args = parser.parse_args()

    x_max = 3400  # Levelende (Flagge 3161 + Schloss)
    flag_x = 3161
    panorama = build_panorama(x_max, args.checkpoint)
    centers, curves, rates, labels = death_curves(args.csv, args.phases, x_max)
    peak = max(c.max() for c in curves)

    fig = plt.figure(figsize=(13, 7.2), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    grid = fig.add_gridspec(
        args.phases + 1, 1, hspace=0.0,
        height_ratios=[1.0] * args.phases + [1.9], left=0.09, right=0.9, top=0.86, bottom=0.04,
    )

    for p in range(args.phases):
        ax = fig.add_subplot(grid[p])
        ax.set_facecolor(SURFACE)
        color = PHASE_RAMP[p % len(PHASE_RAMP)]
        ax.fill_between(centers, 0, curves[p], color=color, alpha=0.85, lw=0)
        ax.plot(centers, curves[p], color=color, lw=1.4)
        ax.set_xlim(0, x_max)
        ax.set_ylim(0, peak * 1.05)
        ax.axvline(flag_x, color=INK_2, lw=1.0, ls=(0, (4, 3)), alpha=0.6)
        ax.set_yticks([])
        ax.set_xticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.text(-0.012, 0.15, labels[p], transform=ax.transAxes, ha="right",
                fontsize=9, color=INK_2)
        ax.text(1.008, 0.15, f"{rates[p]:.1f}%", transform=ax.transAxes, ha="left",
                fontsize=9.5, color=INK_2, fontweight="bold" if rates[p] >= 1 else "normal")
        if p == 0:
            ax.text(flag_x, peak * 1.12, "Flagge (x=3161)", ha="center", va="bottom",
                    fontsize=8.5, color=INK_2)
            ax.text(1.008, 1.05, "Flaggen-Quote", transform=ax.transAxes, ha="left",
                    fontsize=8, color=MUTED)

    # Panorama als unterste Zeile – gleiche x-Skala wie die Kurven darüber.
    ax_pan = fig.add_subplot(grid[args.phases])
    ax_pan.imshow(panorama, extent=[0, x_max, 0, panorama.shape[0]], aspect="auto",
                  interpolation="nearest")
    ax_pan.axvline(flag_x, color="white", lw=1.2, ls=(0, (4, 3)), alpha=0.9)
    ax_pan.set_yticks([])
    ax_pan.set_xticks(np.arange(0, x_max + 1, 400))
    ax_pan.tick_params(colors=MUTED, labelsize=8, length=3)
    for spine in ax_pan.spines.values():
        spine.set_color(BASELINE)
    ax_pan.set_xlabel("x-Position im Level", fontsize=10, color=INK_2)

    fig.suptitle("Wo stirbt Mario? — Todesorte über das DQN-Training (Level 1-1)",
                 fontsize=14, color=INK, x=0.09, ha="left", fontweight="bold")
    fig.text(0.09, 0.895, "Je Trainingsphase: Anteil der Episoden, die an dieser Position enden "
             "(geglättet) — die Hotspots wandern nach rechts, die Flaggen-Quote steigt.",
             fontsize=9.5, color=MUTED)

    fig.savefig(args.out, facecolor=SURFACE, bbox_inches="tight")
    print(f"gespeichert: {args.out}")


if __name__ == "__main__":
    main()
