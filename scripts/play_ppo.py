"""Einem trainierten PPO-Agenten live zuschauen (Original-NES-Fenster).

Läuft im PPO-venv und funktioniert mit jedem SB3-Checkpoint – auch mit den
Zwischenständen, die das Training alle ~50k Steps ablegt (Zuschauen während
das Training weiterläuft):

    .venv-ppo\\Scripts\\python play_ppo.py --model checkpoints_ppo/mario_ppo_gen_welt1.zip --stage 3
"""

from __future__ import annotations

import argparse
import time


def main() -> None:
    parser = argparse.ArgumentParser(description="PPO-Agenten live zuschauen.")
    parser.add_argument("--model", default="checkpoints_ppo/mario_ppo_gen_welt1.zip")
    parser.add_argument("--world", type=int, default=1)
    parser.add_argument("--stage", type=int, default=1)
    parser.add_argument("--episodes", type=int, default=3)
    args = parser.parse_args()

    from stable_baselines3 import PPO

    from mario.wrappers import create_env

    # Altes gym-Env reicht hier: SB3s predict() arbeitet direkt auf der
    # Beobachtung (transponiert Bilder selbst), die 5er-Tupel-API ist egal.
    model = PPO.load(args.model)
    env = create_env(world=args.world, stage=args.stage, render=True)

    print(f"Modell: {args.model} | Level {args.world}-{args.stage} | deterministisch")
    for episode in range(1, args.episodes + 1):
        obs = env.reset()
        total = 0.0
        info: dict = {}
        while True:
            env.render()
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(int(action))
            total += reward
            time.sleep(0.016)  # ~60 FPS Spielgeschwindigkeit
            if done:
                break
        flag = "  *** FLAGGE! ***" if info.get("flag_get", False) else ""
        print(f"Episode {episode} | Reward: {total:7.1f} | x: {info.get('x_pos', 0)}{flag}")

    env.close()
    print("Fertig.")


if __name__ == "__main__":
    main()
