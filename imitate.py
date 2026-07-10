"""Imitation-Warmstart: menschliche Demo aufnehmen und der PPO-Policy beibringen.

Für Stellen, an denen zufällige Exploration scheitert (z. B. der Trampolin-Turm
in 2-1): Ein Mensch zeigt die Passage einmal, Behavior Cloning bringt die
Policy auf diese Spur, PPO-Feintuning macht sie robust.

Ablauf (alles im .venv-ppo):
  1) Demo aufnehmen (Fenster; bis Tod/Flagge spielen – mindestens über die Hängestelle):
       python imitate.py record --world 2 --stage 1 --out demo_2-1.npz
  2) Behavior Cloning auf ein bestehendes Modell:
       python imitate.py bc --demo demo_2-1.npz --model checkpoints_ppo/mario_ppo_2-1.zip \
           --out checkpoints_ppo/mario_ppo_2-1_bc.zip
  3) PPO-Feintuning wie gewohnt:
       python train_ppo.py --world 2 --stage 1 --resume-from checkpoints_ppo/mario_ppo_2-1_bc.zip ...

Drittes Subkommando `bc-seq`: kloniert eine von `goexplore.py` gefundene
Lösungssequenz (Policy-Anlauf + Such-Tail) in die Policy. Anders als beim
menschlichen Demo-BC gibt es hier keinen Distribution Shift – der Anlauf IST
das greedy-Verhalten der Policy. Trainiert bis die Policy den Lauf greedy
exakt reproduziert (deterministisches Env → Flagge):
       python imitate.py bc-seq --seq tower_seq_2-1.npz \
           --model checkpoints_ppo/mario_ppo_2-1.zip \
           --out checkpoints_ppo/mario_ppo_2-1_tower.zip
"""

from __future__ import annotations

import argparse
from collections import Counter, deque

import numpy as np


def _make_raw_env(world: int, stage: int):
    """Rohes NES-Env + SIMPLE_MOVEMENT (60 FPS, keine Wrapper) – für Aufnahme & Replay."""
    import gym_super_mario_bros
    from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
    from nes_py.wrappers import JoypadSpace

    env = gym_super_mario_bros.make(f"SuperMarioBros-{world}-{stage}-v0")
    return JoypadSpace(env, SIMPLE_MOVEMENT)


def _process(frame: np.ndarray, size: int = 84) -> np.ndarray:
    """Graustufen + 84x84 – exakt wie wrappers.GrayScaleResize."""
    import cv2

    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    return cv2.resize(gray, (size, size), interpolation=cv2.INTER_AREA)[:, :, np.newaxis]


def record(world: int, stage: int, out: str) -> None:
    """Zeichnet die KOMPLETTE Aktionssequenz ab Levelstart auf (kein Trimming!).

    Der Emulator ist deterministisch: dieselbe Aktionsfolge ab Reset reproduziert
    den Lauf exakt – auch Gegner-Phasen. Deshalb keine Frames nötig, nur Aktionen.
    """
    from nes_py.app.play_human import play_human

    env = _make_raw_env(world, stage)
    actions: list[int] = []
    state = {"attempt": 1, "best_x": 0}

    def on_step(obs, action, reward, done, next_obs):
        actions.append(int(action))
        if done:
            # Bester Versuch gewinnt: einfach weiterspielen, bis einer sitzt.
            x = int(getattr(env.unwrapped, "_x_position", 0))
            if x > state["best_x"]:
                state["best_x"] = x
                np.savez_compressed(
                    out, actions=np.array(actions, dtype=np.int8), world=world, stage=stage
                )
                print(f"\nVersuch {state['attempt']}: x={x} – NEUER BESTER, gespeichert -> {out}")
            else:
                print(f"\nVersuch {state['attempt']}: x={x} – verworfen (best: {state['best_x']})")
            state["attempt"] += 1
            actions.clear()

    print("Fenster öffnet sich. Pfeiltasten = laufen, O = springen, P = rennen.")
    print("Es zählt automatisch der BESTE Versuch (weiteste x-Position) –")
    print("einfach spielen, bis einer über die Hängestelle kommt, dann ESC.")
    try:
        play_human(env, callback=on_step)
    finally:
        try:
            env.close()
        except Exception:
            pass
    if state["best_x"] == 0:
        print("Nichts gespeichert – Fenster vor dem ersten Episodenende geschlossen?")
    else:
        print(f"Bester Versuch: x={state['best_x']} -> {out}")


def _demo_to_pairs(demo_path: str) -> tuple[np.ndarray, np.ndarray, dict]:
    """Re-simuliert die Demo und baut (Beobachtung, Aktion)-Paare im Agenten-Format.

    Agent-Sicht nachgebaut: Beobachtung = Stack der letzten 4 Graustufen-Frames,
    aktualisiert alle 4 Roh-Frames (wie SkipFrame). Label = häufigste menschliche
    Aktion im 4-Frame-Fenster NACH der Beobachtung (das, was der Agent dann täte).
    """
    data = np.load(demo_path)
    actions = data["actions"].astype(int)
    world, stage = int(data["world"]), int(data["stage"])
    print(f"Demo: {len(actions)} Aktionen, Level {world}-{stage}")

    env = _make_raw_env(world, stage)
    frame = env.reset()
    stack: deque = deque([_process(frame)] * 4, maxlen=4)

    obs_list, label_list = [], []
    info: dict = {}
    i = 0
    done = False
    while i < len(actions) and not done:
        window = actions[i : i + 4]
        label = Counter(window).most_common(1)[0][0]
        obs_list.append(np.concatenate(list(stack), axis=-1))
        label_list.append(label)
        for a in window:
            frame, _, done, info = env.step(int(a))
            if done:
                break
        stack.append(_process(frame))
        i += 4
    env.close()
    print(
        f"Replay: x_pos {info.get('x_pos')} | Flagge: {bool(info.get('flag_get'))} | "
        f"{len(obs_list)} (obs, action)-Paare"
    )
    return np.array(obs_list, dtype=np.uint8), np.array(label_list, dtype=np.int64), info


def bc(demo: str, model_path: str, out: str, epochs: int, lr: float, batch_size: int) -> None:
    """Behavior Cloning: Policy supervised auf die menschliche Demo nachtrainieren."""
    import shutil

    import torch
    from stable_baselines3 import PPO

    obs, labels, _ = _demo_to_pairs(demo)

    model = PPO.load(model_path)
    policy = model.policy
    policy.set_training_mode(True)
    optimizer = torch.optim.Adam(policy.parameters(), lr=lr)

    n = len(obs)
    idx = np.arange(n)
    for epoch in range(1, epochs + 1):
        np.random.shuffle(idx)
        losses, correct = [], 0
        for start in range(0, n, batch_size):
            batch = idx[start : start + batch_size]
            obs_t, _ = policy.obs_to_tensor(obs[batch])
            act_t = torch.as_tensor(labels[batch], device=policy.device)
            dist = policy.get_distribution(obs_t)
            loss = -dist.log_prob(act_t).mean()
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
            optimizer.step()
            losses.append(float(loss))
            correct += int((dist.distribution.probs.argmax(dim=1) == act_t).sum())
        print(
            f"Epoche {epoch}/{epochs} | Loss {np.mean(losses):.4f} | "
            f"Demo-Trefferquote {correct / n * 100:.1f}%"
        )

    policy.set_training_mode(False)
    model.save(out)
    # VecNormalize-Statistik mitkopieren, damit das Feintuning sie laden kann.
    src_pkl = model_path.replace(".zip", "_vecnormalize.pkl")
    dst_pkl = out.replace(".zip", "_vecnormalize.pkl")
    if src_pkl != dst_pkl:
        try:
            shutil.copyfile(src_pkl, dst_pkl)
            print(f"VecNormalize-Statistik kopiert -> {dst_pkl}")
        except FileNotFoundError:
            pass
    print(f"BC-Modell gespeichert: {out}")


def bc_seq(seq_path: str, model_path: str, out: str, epochs: int, lr: float, batch_size: int) -> None:
    """Kloniert eine goexplore-Lösungssequenz in die Policy (bis zur Reproduktion).

    Fehler-Steps werden übergewichtet und jede Epoche wird greedy verifiziert –
    Stopp, sobald der Rollout die Flagge erreicht. Entscheidend ist nicht die
    Trefferquote, sondern der Greedy-Lauf selbst (lange Zufalls-Tails wie bei 4-4
    sind nie zu 100 % klonbar); gespeichert wird der beste Stand nach Greedy-x.
    """
    import shutil

    import torch
    from stable_baselines3 import PPO

    import config
    from wrappers import create_env

    data = np.load(seq_path)
    if int(data["frame_skip"]) != config.FRAME_SKIP:
        raise SystemExit(
            f"Sequenz wurde mit FRAME_SKIP={int(data['frame_skip'])} gesucht, "
            f"aktuell ist {config.FRAME_SKIP} – FRAME_SKIP passend setzen!"
        )
    seq = np.concatenate([data["head"], data["tail"]]).astype(int)
    world, stage = int(data["world"]), int(data["stage"])
    print(f"Sequenz {world}-{stage}: {len(data['head'])} Anlauf- + {len(data['tail'])} Tail-Steps")

    # 1) Replay durch den Wrapper-Stack -> (obs, action)-Paare im Agenten-Format
    env = create_env(world=world, stage=stage, render=False)
    obs = env.reset()
    obs_list, act_list = [], []
    info: dict = {}
    for a in seq:
        obs_list.append(obs)
        act_list.append(int(a))
        obs, _, done, info = env.step(int(a))
        if done:
            break
    print(f"Replay: x={info.get('x_pos')} | Flagge: {bool(info.get('flag_get'))} | {len(obs_list)} Paare")
    if not info.get("flag_get"):
        raise SystemExit("Replay erreicht die Flagge nicht – Sequenz ungültig (FRAME_SKIP? Env-Version?).")
    obs_arr = np.array(obs_list, dtype=np.uint8)
    act_arr = np.array(act_list, dtype=np.int64)

    # 2) BC bis die Policy den Lauf greedy exakt reproduziert
    model = PPO.load(model_path)
    policy = model.policy
    optimizer = torch.optim.Adam(policy.parameters(), lr=lr)

    def verify() -> tuple[int, bool]:
        policy.set_training_mode(False)
        o = env.reset()
        done = False
        while not done:
            action, _ = model.predict(o, deterministic=True)
            o, _, done, i = env.step(int(action))
        policy.set_training_mode(True)
        return int(i.get("x_pos", 0)), bool(i.get("flag_get"))

    policy.set_training_mode(True)
    n = len(obs_arr)
    wrong = np.arange(n)  # Fehler-Steps der letzten Epoche (Start: alle)
    flag = False
    best_x, best_state = -1, None
    for epoch in range(1, epochs + 1):
        idx = np.concatenate([np.arange(n)] + [wrong] * 4)  # Fehler 4x übergewichten
        np.random.shuffle(idx)
        losses = []
        for start in range(0, len(idx), batch_size):
            batch = idx[start : start + batch_size]
            obs_t, _ = policy.obs_to_tensor(obs_arr[batch])
            act_t = torch.as_tensor(act_arr[batch], device=policy.device)
            dist = policy.get_distribution(obs_t)
            loss = -dist.log_prob(act_t).mean()
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
            optimizer.step()
            losses.append(float(loss))
        with torch.no_grad():
            preds = []
            for start in range(0, n, 256):
                obs_t, _ = policy.obs_to_tensor(obs_arr[start : start + 256])
                preds.append(policy.get_distribution(obs_t).distribution.probs.argmax(dim=1).cpu().numpy())
        wrong = np.where(np.concatenate(preds) != act_arr)[0]
        acc = (n - len(wrong)) / n * 100
        x, flag = verify()
        if x > best_x:
            best_x = x
            best_state = {k: v.detach().clone() for k, v in policy.state_dict().items()}
        print(f"Epoche {epoch:2d} | Loss {np.mean(losses):.4f} | Treffer {acc:.1f}%"
              f" | greedy: x={x}{' FLAGGE!' if flag else ''}")
        if flag:
            break

    # Besten Stand behalten (der letzte ist nicht automatisch der beste).
    if not flag and best_state is not None:
        policy.load_state_dict(best_state)
    policy.set_training_mode(False)
    x, flag = verify()
    env.close()
    print(f"Greedy-Verifikation: x={x} | Flagge: {flag}")

    model.save(out)
    src_pkl = model_path.replace(".zip", "_vecnormalize.pkl")
    dst_pkl = out.replace(".zip", "_vecnormalize.pkl")
    if src_pkl != dst_pkl:
        try:
            shutil.copyfile(src_pkl, dst_pkl)
        except FileNotFoundError:
            pass
    print(f"Modell gespeichert: {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Imitation-Warmstart (Demo + Behavior Cloning).")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_rec = sub.add_parser("record", help="menschliche Demo aufnehmen (öffnet Fenster)")
    p_rec.add_argument("--world", type=int, default=2)
    p_rec.add_argument("--stage", type=int, default=1)
    p_rec.add_argument("--out", default="demo_2-1.npz")

    p_bc = sub.add_parser("bc", help="Behavior Cloning auf ein PPO-Modell")
    p_bc.add_argument("--demo", default="demo_2-1.npz")
    p_bc.add_argument("--model", default="checkpoints_ppo/mario_ppo_2-1.zip")
    p_bc.add_argument("--out", default="checkpoints_ppo/mario_ppo_2-1_bc.zip")
    p_bc.add_argument("--epochs", type=int, default=8)
    p_bc.add_argument("--lr", type=float, default=3e-5)
    p_bc.add_argument("--batch-size", type=int, default=64)

    p_seq = sub.add_parser("bc-seq", help="goexplore-Lösungssequenz in die Policy klonen")
    p_seq.add_argument("--seq", required=True, help=".npz aus goexplore.py --save-best")
    p_seq.add_argument("--model", default="checkpoints_ppo/mario_ppo_2-1.zip")
    p_seq.add_argument("--out", default="checkpoints_ppo/mario_ppo_2-1_tower.zip")
    p_seq.add_argument("--epochs", type=int, default=60)
    p_seq.add_argument("--lr", type=float, default=3e-5)
    p_seq.add_argument("--batch-size", type=int, default=64)

    args = parser.parse_args()
    if args.cmd == "record":
        record(args.world, args.stage, args.out)
    elif args.cmd == "bc":
        bc(args.demo, args.model, args.out, args.epochs, args.lr, args.batch_size)
    else:
        bc_seq(args.seq, args.model, args.out, args.epochs, args.lr, args.batch_size)


if __name__ == "__main__":
    main()
