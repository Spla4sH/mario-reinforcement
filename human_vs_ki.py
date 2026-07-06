"""Mensch vs. KI: eigenen Mario-Lauf aufnehmen und Seite an Seite mit dem Agenten zeigen.

Zwei Schritte (beide im DQN-venv, Schritt 1 braucht ein Fenster):

1. Eigenen Lauf aufnehmen (Steuerung: Pfeiltasten = laufen, O = springen,
   P = Feuer/Rennen; ESC beendet). Aufgenommen wird bis zum ersten Tod/Flaggen-Erfolg:

       python human_vs_ki.py record --out mensch_1-1.npz

2. Side-by-Side-GIF bauen (links du, rechts der Agent aus mario_best.pt):

       python human_vs_ki.py compose --human mensch_1-1.npz --out mensch_vs_ki.gif
"""

from __future__ import annotations

import argparse

import numpy as np

import config


def _make_raw_env(world: int, stage: int):
    """Rohes NES-Env + SIMPLE_MOVEMENT – dieselben 7 Aktionen wie der Agent (fair)."""
    import gym_super_mario_bros
    from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
    from nes_py.wrappers import JoypadSpace

    env = gym_super_mario_bros.make(f"SuperMarioBros-{world}-{stage}-v0")
    return JoypadSpace(env, SIMPLE_MOVEMENT)


def record_human(world: int, stage: int, out: str) -> None:
    """Öffnet das Spiel-Fenster und zeichnet die erste Episode (bis done) auf."""
    from nes_py.app.play_human import play_human

    env = _make_raw_env(world, stage)
    frames: list[np.ndarray] = []
    state = {"done": False, "info": {}}

    orig_step = env.step

    def recording_step(action):
        obs, reward, done, info = orig_step(action)
        if not state["done"]:
            frames.append(np.array(env.screen, copy=True))
            if done:
                state["done"] = True
                state["info"] = dict(info)
        return obs, reward, done, info

    env.step = recording_step
    print("Fenster öffnet sich. Pfeiltasten = laufen, O = springen, P = rennen/Feuer.")
    print("Aufgenommen wird bis zum ersten Tod / zur Flagge. Danach mit ESC schließen.")
    try:
        play_human(env)
    finally:
        env.close()

    if not frames:
        print("Keine Frames aufgenommen – Fenster zu früh geschlossen?")
        return
    info = state["info"]
    np.savez_compressed(out, frames=np.array(frames[:: 4]))  # alle 4 NES-Frames (wie SkipFrame)
    print(
        f"{len(frames)} Frames aufgenommen (gespeichert: jede 4.) -> {out} | "
        f"x_pos {info.get('x_pos')} | Flagge: {bool(info.get('flag_get'))}"
    )


def compose(human_path: str, out: str, checkpoint: str, world: int, stage: int) -> None:
    """Baut das Side-by-Side-GIF: links Mensch, rechts Agent (greedy)."""
    from PIL import Image, ImageDraw

    from agent import MarioAgent
    from record import record_frames, save_gif
    from wrappers import create_env

    human = list(np.load(human_path)["frames"])
    print(f"Mensch: {len(human)} Frames geladen")

    env = create_env(world=world, stage=stage, render=False)
    agent = MarioAgent(env.action_space.n)
    agent.load(checkpoint)
    ki, info = record_frames(agent, env)  # SkipFrame(4) -> gleiche Frame-Rate wie human[::4]
    env.close()
    print(f"KI: {len(ki)} Frames | x_pos {info.get('x_pos')} | Flagge: {bool(info.get('flag_get'))}")

    # Kürzeren Lauf mit letztem Frame auffüllen, dann nebeneinander montieren.
    n = max(len(human), len(ki))
    human += [human[-1]] * (n - len(human))
    ki += [ki[-1]] * (n - len(ki))

    combined = []
    divider = np.full((human[0].shape[0], 4, 3), 40, dtype=np.uint8)
    for h, k in zip(human, ki):
        img = Image.fromarray(np.hstack([h, divider, k]))
        draw = ImageDraw.Draw(img)
        for x, label in ((6, "MENSCH"), (human[0].shape[1] + 10, "KI (DQN)")):
            draw.rectangle([x - 3, 4, x + 6.5 * len(label), 16], fill=(0, 0, 0))
            draw.text((x, 5), label, fill=(255, 255, 255))
        combined.append(np.array(img))

    # Unter ~9.5 MB bleiben: notfalls Frames ausdünnen.
    import os

    for keep in (1.0, 0.7, 0.5, 0.35):
        sel = [f for i, f in enumerate(combined) if (i * keep) % 1 < keep]
        save_gif(sel, out, fps=max(int(15 * keep), 8))
        size = os.path.getsize(out) / 1e6
        print(f"keep={keep}: {len(sel)} Frames, {size:.1f} MB")
        if size <= 9.5:
            break
    print(f"GIF gespeichert: {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Mensch-vs-KI-Vergleich.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_rec = sub.add_parser("record", help="eigenen Lauf aufnehmen (öffnet Fenster)")
    p_rec.add_argument("--world", type=int, default=config.WORLD)
    p_rec.add_argument("--stage", type=int, default=config.STAGE)
    p_rec.add_argument("--out", default="mensch_1-1.npz")

    p_cmp = sub.add_parser("compose", help="Side-by-Side-GIF bauen")
    p_cmp.add_argument("--human", default="mensch_1-1.npz")
    p_cmp.add_argument("--checkpoint", default="checkpoints/mario_best.pt")
    p_cmp.add_argument("--world", type=int, default=config.WORLD)
    p_cmp.add_argument("--stage", type=int, default=config.STAGE)
    p_cmp.add_argument("--out", default="mensch_vs_ki.gif")

    args = parser.parse_args()
    if args.cmd == "record":
        record_human(args.world, args.stage, args.out)
    else:
        compose(args.human, args.out, args.checkpoint, args.world, args.stage)


if __name__ == "__main__":
    main()
