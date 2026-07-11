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
    state = {"done": False}

    def on_step(obs, action, reward, done, next_obs):
        # Offizieller play_human-Callback: nach jedem Schritt aufgerufen.
        # WICHTIG: sofort beim Episodenende speichern – play_human schließt das
        # Env beim Beenden selbst, auf den Code NACH play_human ist kein Verlass.
        if state["done"]:
            return
        # Erst ab der ersten echten Eingabe aufnehmen (Aktion 0 = NOOP) –
        # Fenster zurechtziehen/orientieren landet so nicht im GIF.
        if not frames and action == 0:
            return
        frames.append(np.array(env.unwrapped.screen, copy=True))
        if done:
            state["done"] = True
            np.savez_compressed(out, frames=np.array(frames[::4]))  # jede 4. (wie SkipFrame)
            print(f"\nEpisode beendet – {len(frames)} Frames gespeichert -> {out}")
            print("Fenster jetzt mit ESC schließen, dann 'compose' ausführen.")

    print("Fenster öffnet sich. Pfeiltasten = laufen, O = springen, P = rennen/Feuer.")
    print("Aufgenommen wird bis zum ersten Tod / zur Flagge. Danach mit ESC schließen.")
    try:
        play_human(env, callback=on_step)
    finally:
        # play_human schließt das Env normalerweise selbst – doppelter close
        # wirft in nes-py einen ValueError, daher hier nur als Absicherung.
        try:
            env.close()
        except Exception:
            pass

    if not state["done"]:
        print("Keine vollständige Episode aufgenommen – Fenster zu früh geschlossen?")


def compose(
    human_path: str, out: str, checkpoint: str, world: int, stage: int,
    trim_start: int = 0,
) -> None:
    """Baut das Side-by-Side-GIF: links Mensch, rechts Agent (greedy)."""
    from PIL import Image, ImageDraw

    from agent import MarioAgent
    from record import record_frames, save_gif
    from wrappers import create_env

    import os

    if not os.path.exists(human_path):
        print(f"Aufnahme nicht gefunden: {human_path}")
        print("Zuerst den eigenen Lauf aufnehmen:  python human_vs_ki.py record")
        return
    human = list(np.load(human_path)["frames"])
    if trim_start:
        human = human[trim_start:]
        print(f"Mensch: {len(human)} Frames (erste {trim_start} abgeschnitten)")
    else:
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
            # Textgröße messen statt schätzen – die Default-Schrift ist je nach
            # Pillow-Version unterschiedlich groß (sonst endet die Box im Text).
            box = draw.textbbox((x, 5), label)
            draw.rectangle([box[0] - 3, box[1] - 2, box[2] + 3, box[3] + 2], fill=(0, 0, 0))
            draw.text((x, 5), label, fill=(255, 255, 255))
        combined.append(np.array(img))

    # Unter ~9.5 MB bleiben: notfalls Frames ausdünnen.
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
    p_cmp.add_argument(
        "--trim-start", type=int, default=0,
        help="So viele Frames vom Anfang der Mensch-Aufnahme abschneiden (15 ≈ 1 Sek.)",
    )

    args = parser.parse_args()
    if args.cmd == "record":
        record_human(args.world, args.stage, args.out)
    else:
        compose(args.human, args.out, args.checkpoint, args.world, args.stage, args.trim_start)


if __name__ == "__main__":
    main()
