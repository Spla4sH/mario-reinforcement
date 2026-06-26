"""Spielt einen trainierten Agenten greedy ab – Live im Original-Spielbild.

Lädt einen Checkpoint und lässt Mario ohne Exploration (epsilon=0) und ohne
Weitertraining spielen. Das NES-Originalbild wird in einem Fenster angezeigt.

Aufruf:
    python play.py [--episodes N] [--checkpoint pfad]
"""

import argparse
import time
from pathlib import Path

import config
from wrappers import create_env
from agent import MarioAgent


def play(episodes=5, checkpoint="checkpoints/mario_agent.pt"):
    print("=" * 70)
    print("  SUPER MARIO BROS - Wiedergabe (greedy, kein Training)")
    print("=" * 70)

    checkpoint_path = Path(checkpoint)
    if not checkpoint_path.exists():
        print(f"Kein Checkpoint gefunden unter: {checkpoint_path}")
        print("Trainiere zuerst mit 'python train.py'.")
        return

    env = create_env(world=config.WORLD, stage=config.STAGE, render=True)
    agent = MarioAgent(env.action_space.n)
    agent.load(str(checkpoint_path))

    for episode in range(1, episodes + 1):
        state = env.reset()
        total_reward = 0.0
        info = {}

        while True:
            env.render()
            action = agent.act(state)  # greedy, ohne Seiteneffekte
            state, reward, done, info = env.step(action)
            total_reward += reward
            # Spielgeschwindigkeit etwas bremsen, damit es ansehbar bleibt
            time.sleep(0.01)
            if done:
                break

        flag = "  *** GESCHAFFT! ***" if info.get("flag_get", False) else ""
        print(
            f"Episode {episode:3d} | "
            f"Belohnung: {total_reward:7.1f} | "
            f"Position: {info.get('x_pos', 0):5d}{flag}"
        )

    env.close()
    print("Wiedergabe beendet!")


def main():
    parser = argparse.ArgumentParser(description="Trainierten Mario-Agenten abspielen.")
    parser.add_argument(
        "--episodes", type=int, default=5, help="Anzahl der Episoden (Standard: 5)"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/mario_agent.pt",
        help="Pfad zum Checkpoint",
    )
    args = parser.parse_args()
    play(episodes=args.episodes, checkpoint=args.checkpoint)


if __name__ == "__main__":
    main()
