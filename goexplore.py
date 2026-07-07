"""Savestate-Suche gegen Hard-Exploration-Stellen (Go-Explore-Idee, minimal).

PPO scheiterte am Trampolin-Turm von 2-1 trotz 12M Steps, Entropie-Boosts und
BC einer menschlichen Demo: Die Policy erreicht die Stelle zuverlässig, findet
den Abprall-Sprung durch Zufalls-Exploration aber nie – jede Probe kostet einen
kompletten Anlauf, die Chance auf die richtige Sprungsequenz ist damit praktisch null.

Der Trick (wie bei Go-Explore, Ecoffet et al. 2019): **Zustand sichern statt
immer neu anlaufen.**

  1. Policy spielt greedy bis kurz vor die Hängestelle (``--backup-x``).
  2. NES-Savestate sichern (nes-py ``_backup``).
  3. Tausende zufällige Aktionssequenzen ab dem Savestate testen
     (``_restore`` ist sofortig, kein Replay nötig).
  4. Gefundene Lösung als .npz speichern → mit ``imitate.py bc-seq``
     in die Policy klonen.

Beispiel 2-1 (fand den Trampolin-Superbounce nach 81 Kandidaten):
    python goexplore.py --model checkpoints_ppo/mario_ppo_2-1_bc_ft.zip \\
        --world 2 --stage 1 --backup-x 2940 --success-x 3150 \\
        --save-best tower_seq_2-1.npz

Nur im `.venv-ppo` lauffähig. FRAME_SKIP beachten – die gefundene Sequenz
gilt für den Skip, mit dem gesucht wurde (wird in der .npz mitgespeichert).
"""

from __future__ import annotations

import argparse
import sys

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description="Zufallssuche ab Savestate (Go-Explore-Idee).")
    parser.add_argument("--model", required=True, help="PPO-.zip für den Anlauf (greedy)")
    parser.add_argument("--world", type=int, default=2)
    parser.add_argument("--stage", type=int, default=1)
    parser.add_argument("--backup-x", type=int, required=True, help="Savestate, sobald x erreicht")
    parser.add_argument("--success-x", type=int, required=True, help="Erfolg, sobald x erreicht (oder Flagge)")
    parser.add_argument("--candidates", type=int, default=5000)
    parser.add_argument("--horizon", type=int, default=70, help="Agent-Steps je Kandidat")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save-best", default="", help=".npz für die Lösungssequenz")
    args = parser.parse_args()

    from stable_baselines3 import PPO

    import config
    from wrappers import create_env

    print(f"FRAME_SKIP={config.FRAME_SKIP} | Modell: {args.model}")

    model = PPO.load(args.model)
    env = create_env(world=args.world, stage=args.stage, render=False)
    nes = env.unwrapped

    # 1) Anlauf: Policy greedy bis zum Backup-Punkt
    obs = env.reset()
    head_actions: list[int] = []
    x = 0
    for _ in range(5000):
        action, _ = model.predict(obs, deterministic=True)
        obs, _, done, info = env.step(int(action))
        head_actions.append(int(action))
        x = int(info["x_pos"])
        if x >= args.backup_x or done:
            break
    if x < args.backup_x:
        print(f"Anlauf gescheitert (x={x}) – Policy erreicht --backup-x nicht.")
        sys.exit(1)
    print(f"Anlauf: x={x} nach {len(head_actions)} Agent-Steps – Savestate gesichert.")
    nes._backup()

    # 2) Zufallssuche ab Savestate: Segmente aus (Aktion, Haltedauer), damit
    # auch längere Sprünge/Anläufe entstehen statt reinem Aktions-Rauschen.
    rng = np.random.default_rng(args.seed)
    actions = [0, 1, 2, 3, 4, 5, 6]  # NOOP, R, R+A, R+B, R+A+B, A, L (SIMPLE_MOVEMENT)
    weights = [0.06, 0.08, 0.12, 0.28, 0.28, 0.10, 0.08]

    best_x, best_seq = 0, []
    solved = False
    info = {}
    for cand in range(1, args.candidates + 1):
        nes._restore()
        nes.done = False  # Python-seitiges done-Flag zurücksetzen
        seq: list[int] = []
        cand_max = 0
        done = False
        while len(seq) < args.horizon and not done:
            a = int(rng.choice(actions, p=weights))
            hold = int(rng.integers(1, 9))
            for _ in range(hold):
                if len(seq) >= args.horizon:
                    break
                _, _, done, info = env.step(a)
                seq.append(a)
                cand_max = max(cand_max, int(info["x_pos"]))
                if done:
                    break
        if cand_max > best_x:
            best_x, best_seq = cand_max, list(seq)
            print(f"Kandidat {cand}: neues Best-x {best_x}" + (" | FLAGGE!" if info.get("flag_get") else ""))
        if info.get("flag_get") or cand_max >= args.success_x:
            solved = True
            print(f"DURCHBRUCH bei Kandidat {cand}: x={cand_max}, Flagge={bool(info.get('flag_get'))}")
            break
        if cand % 500 == 0:
            print(f"... {cand} Kandidaten, Best-x bisher: {best_x}")

    env.close()
    print("=" * 50)
    print(f"ERGEBNIS: Best-x {best_x} | Durchbruch: {solved} | Skip {config.FRAME_SKIP}")
    if solved and args.save_best:
        np.savez_compressed(
            args.save_best,
            head=np.array(head_actions, dtype=np.int8),
            tail=np.array(best_seq, dtype=np.int8),
            frame_skip=config.FRAME_SKIP,
            world=args.world,
            stage=args.stage,
        )
        print(f"Sequenz gespeichert: {args.save_best} (head={len(head_actions)}, tail={len(best_seq)})")
        print("Nächster Schritt:  python imitate.py bc-seq --seq " + args.save_best)


if __name__ == "__main__":
    main()
