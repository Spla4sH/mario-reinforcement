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
    parser.add_argument("--model", default="", help="PPO-.zip für den Anlauf (greedy)")
    parser.add_argument("--head-seq", default="",
                        help="Anlauf stattdessen aus .npz-Sequenz (head+tail) abspielen – "
                             "für mehrstufige Suchen (z. B. Labyrinth 4-4)")
    parser.add_argument("--world", type=int, default=2)
    parser.add_argument("--stage", type=int, default=1)
    parser.add_argument("--backup-x", type=int, required=True, help="Savestate, sobald x erreicht")
    parser.add_argument("--success-x", type=int, default=10**9,
                        help="Erfolg, sobald x erreicht (oder Flagge)")
    parser.add_argument("--success-area", action="store_true",
                        help="Erfolg = Bereichswechsel (SMB-RAM $0760) statt x-Schwelle. "
                             "Pflicht fuer Loop-Level wie 8-4: dort zaehlt x_pos beim "
                             "Im-Kreis-Laufen einfach weiter, jede x-Metrik luegt.")
    parser.add_argument("--candidates", type=int, default=5000)
    parser.add_argument("--horizon", type=int, default=70, help="Agent-Steps je Kandidat")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save-best", default="", help=".npz für die Lösungssequenz")
    args = parser.parse_args()

    if not args.model and not args.head_seq:
        parser.error("--model oder --head-seq angeben")

    import config
    from wrappers import create_env

    print(f"FRAME_SKIP={config.FRAME_SKIP} | Anlauf: {args.head_seq or args.model}")

    env = create_env(world=args.world, stage=args.stage, render=False)
    nes = env.unwrapped

    # 1) Anlauf bis zum Backup-Punkt: Policy greedy – oder eine bereits gefundene
    # Sequenz abspielen (mehrstufige Suche: das Ergebnis von Stufe N ist der Anlauf
    # von Stufe N+1; die finale .npz enthält dann den kompletten Weg für bc-seq).
    obs = env.reset()
    head_actions: list[int] = []
    x = 0
    if args.head_seq:
        data = np.load(args.head_seq)
        if int(data["frame_skip"]) != config.FRAME_SKIP:
            print(f"FRAME_SKIP-Konflikt: Sequenz={int(data['frame_skip'])}, config={config.FRAME_SKIP}")
            sys.exit(1)
        for action in np.concatenate([data["head"], data["tail"]]).astype(int).tolist():
            obs, _, done, info = env.step(action)
            head_actions.append(action)
            x = int(info["x_pos"])
            if x >= args.backup_x or done:
                break
    else:
        from stable_baselines3 import PPO

        model = PPO.load(args.model)
        for _ in range(5000):
            action, _ = model.predict(obs, deterministic=True)
            obs, _, done, info = env.step(int(action))
            head_actions.append(int(action))
            x = int(info["x_pos"])
            if x >= args.backup_x or done:
                break
    if x < args.backup_x:
        print(f"Anlauf gescheitert (x={x}) – Anlauf erreicht --backup-x nicht.")
        sys.exit(1)
    print(f"Anlauf: x={x} nach {len(head_actions)} Agent-Steps – Savestate gesichert.")
    nes._backup()
    backup_x_val = x
    # SMB haelt in $0760 den "AreaPointer" – welche Karte gerade geladen ist.
    # In Roehren-Labyrinthen (8-4) ist das die EINZIGE ehrliche Fortschrittsmessung:
    # x_pos zaehlt beim Loop stur weiter (Page-Nummer wird nicht zurueckgesetzt),
    # ein Bereichswechsel dagegen passiert nur durch eine echte Roehre.
    start_area = int(nes.ram[0x0760])
    if args.success_area:
        print(f"Erfolgskriterium: Bereichswechsel (Start-Area {start_area})")

    # 2) Zufallssuche ab Savestate: Segmente aus (Aktion, Haltedauer), damit
    # auch längere Sprünge/Anläufe entstehen statt reinem Aktions-Rauschen.
    rng = np.random.default_rng(args.seed)
    actions = [0, 1, 2, 3, 4, 5, 6]  # NOOP, R, R+A, R+B, R+A+B, A, L (SIMPLE_MOVEMENT)
    weights = [0.06, 0.08, 0.12, 0.28, 0.28, 0.10, 0.08]
    # Mit ACTION_SET=down (8-4: Roehren) kommt ['down'] als 8. Aktion dazu – ohne
    # sie in der Auswahl wuerde die Suche nie eine Roehre betreten. Anteil bewusst
    # klein: 'down' ist nur an wenigen Stellen sinnvoll, verlangsamt sonst nur.
    if env.action_space.n == 8:
        actions = actions + [7]
        weights = [w * 0.92 for w in weights] + [0.08]

    best_x, best_seq = 0, []
    solved = False
    info = {}
    for cand in range(1, args.candidates + 1):
        nes._restore()
        nes.done = False  # Python-seitiges done-Flag zurücksetzen
        seq: list[int] = []
        cand_max = 0
        prev_x = backup_x_val
        done = False
        area_hit = False
        while len(seq) < args.horizon and not done:
            a = int(rng.choice(actions, p=weights))
            hold = int(rng.integers(1, 9))
            for _ in range(hold):
                if len(seq) >= args.horizon:
                    break
                _, _, done, info = env.step(a)
                seq.append(a)
                # Nur physikalisch plausible Schritte zählen (wie ProgressReward):
                # der 16-Bit-x-Glitch (x_pos springt auf ~65535) wäre sonst ein
                # falscher "Durchbruch". Echte Rücksprünge (Loop-Reset im
                # Labyrinth) werden übernommen, damit prev_x stimmig bleibt.
                x_now = int(info["x_pos"])
                delta = x_now - prev_x
                if 0 < delta <= 64:
                    cand_max = max(cand_max, x_now)
                    prev_x = x_now
                elif delta <= 0 and x_now < 60000:
                    prev_x = x_now
                if args.success_area and not done and int(nes.ram[0x0760]) != start_area:
                    area_hit = True
                    break
                if done:
                    break
            if area_hit:
                break
        # Bereichswechsel schlaegt jede x-Bewertung: nach der Roehre ist x_pos klein
        # (neue Karte) – ohne diesen Zweig wuerden die x-Regeln unten den einzigen
        # echten Treffer der ganzen Suche wegwerfen.
        if area_hit:
            cand_max = 10**6
        # Tote Kandidaten verwerfen: Beim Sturz in eine Grube laeuft x_pos noch
        # weiter hoch (Mario fliegt im Fall nach vorn) – ohne diese Pruefung
        # gewinnt ein weiter Sturz gegen einen sicheren Stand, und die Suche
        # optimiert das Sterben (bei 5-3 genau so passiert).
        elif done and not info.get("flag_get"):
            cand_max = 0
        # In Loop-Leveln (Labyrinth-Schloesser) zaehlt die ENDposition, nicht das
        # Maximum: Ein Kandidat, der kurz weit kommt und dann zurueckgeworfen wird,
        # ist als Anlauf fuer die naechste Stufe wertlos. Bei 7-4 meldete die Suche
        # so einen "Durchbruch" auf x 2344, dessen Sequenz bei x 1477 endete.
        elif not info.get("flag_get"):
            cand_max = min(cand_max, int(info.get("x_pos", 0)))
        if cand_max > best_x:
            best_x, best_seq = cand_max, list(seq)
            if not area_hit:
                print(f"Kandidat {cand}: neues Best-x {best_x}"
                      + (" | FLAGGE!" if info.get("flag_get") else ""))
        if area_hit:
            solved = True
            print(f"BEREICHSWECHSEL bei Kandidat {cand}: Area {start_area} -> "
                  f"{int(nes.ram[0x0760])} nach {len(seq)} Steps (x={int(info.get('x_pos', 0))})")
            break
        if info.get("flag_get") or cand_max >= args.success_x:
            solved = True
            print(f"DURCHBRUCH bei Kandidat {cand}: x={cand_max}, Flagge={bool(info.get('flag_get'))}")
            break
        if cand % 500 == 0:
            print(f"... {cand} Kandidaten, Best-x bisher: {best_x}")

    env.close()
    print("=" * 50)
    ergebnis = "Bereichswechsel" if best_x == 10**6 else f"Best-x {best_x}"
    print(f"ERGEBNIS: {ergebnis} | Durchbruch: {solved} | Skip {config.FRAME_SKIP}")
    # Auch ohne Durchbruch speichern: Bei mehrstufigen Suchen (Labyrinthe mit
    # mehreren Weichen) ist der beste Zwischenstand die Grundlage der nächsten
    # Stufe – ihn zu verwerfen würde die ganze Suche wiederholen.
    if args.save_best and best_seq:
        np.savez_compressed(
            args.save_best,
            head=np.array(head_actions, dtype=np.int8),
            tail=np.array(best_seq, dtype=np.int8),
            frame_skip=config.FRAME_SKIP,
            world=args.world,
            stage=args.stage,
        )
        print(f"Sequenz gespeichert: {args.save_best} (head={len(head_actions)}, tail={len(best_seq)})")
        if solved:
            print("Nächster Schritt:  python imitate.py bc-seq --seq " + args.save_best)
        else:
            print(f"Kein Durchbruch – bester Stand x={best_x} als Anlauf für die nächste Stufe:")
            print(f"  python goexplore.py --head-seq {args.save_best} --backup-x <kurz vor {best_x}> ...")


if __name__ == "__main__":
    main()
