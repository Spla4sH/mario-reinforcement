"""Greedy-Eval eines PPO-Modells: Flaggen-Rate ueber N deterministische Episoden.

Ausführen im .venv-ppo:
    .venv-ppo/Scripts/python eval_ppo.py --model checkpoints_ppo/mario_ppo_2-2.zip --world 2 --stage 2
"""

import argparse
import sys

import numpy as np
from stable_baselines3 import PPO

from mario_ppo_env import make_mario_env

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass

parser = argparse.ArgumentParser()
parser.add_argument("--model", default="checkpoints_ppo/mario_ppo_tuned.zip")
parser.add_argument("--world", type=int, default=1)
parser.add_argument("--stage", type=int, default=2)
parser.add_argument("--episodes", type=int, default=20)
args = parser.parse_args()

# norm_obs=False im Training -> Policy sieht rohe uint8-Frames, Eval braucht
# daher keine VecNormalize-Statistik (Rewards beeinflussen Aktionen nicht).
model = PPO.load(args.model)
env = make_mario_env(args.world, args.stage)

flags = 0
xs, rewards = [], []
for ep in range(1, args.episodes + 1):
    obs, info = env.reset()
    total = 0.0
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, r, terminated, truncated, info = env.step(int(action))
        total += r
        done = terminated or truncated
    flag = bool(info.get("flag_get", False))
    flags += int(flag)
    xs.append(int(info.get("x_pos", 0)))
    rewards.append(total)
    print(f"Episode {ep:2d} | x: {xs[-1]:4d} | Reward: {total:7.1f} | {'FLAGGE' if flag else '-'}")

env.close()
print("=" * 50)
print(f"  Flaggen-Rate : {flags}/{args.episodes} ({100 * flags / args.episodes:.0f}%)")
print(f"  Ø x-Position : {np.mean(xs):.0f} | max x: {max(xs)}")
print(f"  Ø Reward     : {np.mean(rewards):.1f}")
print("=" * 50)
print(f"Erfolgskriterium (>=80%): {'ERFÜLLT' if flags / args.episodes >= 0.8 else 'noch nicht'}")
