"""Nimmt eine deterministische PPO-Episode als GIF auf (Original-NES-Bild).

Ausführen im .venv-ppo:
    .venv-ppo/Scripts/python gif_ppo.py --model checkpoints_ppo/mario_ppo_2-2.zip --world 2 --stage 2 --out ppo_2-2.gif
"""

import argparse
import os
import sys

import imageio.v2 as imageio
import numpy as np
from PIL import Image
from stable_baselines3 import PPO

from mario.mario_ppo_env import make_mario_env

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass

parser = argparse.ArgumentParser()
parser.add_argument("--model", default="checkpoints_ppo/mario_ppo_tuned.zip")
parser.add_argument("--world", type=int, default=1)
parser.add_argument("--stage", type=int, default=2)
parser.add_argument("--out", default="assets/ppo_1-2.gif")
args = parser.parse_args()

model = PPO.load(args.model)
env = make_mario_env(args.world, args.stage)

frames = []
obs, info = env.reset()
done = False
while not done:
    action, _ = model.predict(obs, deterministic=True)
    # Original-RGB-Bild vom inneren NES-Env holen (shimmy reicht mode nicht durch).
    frames.append(np.array(env.gym_env.render(mode="rgb_array"), copy=True))
    obs, r, terminated, truncated, info = env.step(int(action))
    done = terminated or truncated
env.close()
print(f"Episode: x={info.get('x_pos')} flag={info.get('flag_get')} | Roh-Frames: {len(frames)}")

# 2x hochskalieren (NEAREST = scharfe Pixel), Frames ausduennen, unter ~9.5 MB halten.
frames = [f for f in frames if f is not None]
big = [np.array(Image.fromarray(f).resize((512, 480), Image.NEAREST)) for f in frames]
for keep in (0.7, 0.5, 0.35, 0.25):
    sel = [f for i, f in enumerate(big) if (i * keep) % 1 < keep]
    imageio.mimsave(args.out, sel, fps=int(30 * keep), loop=0)
    size = os.path.getsize(args.out) / 1e6
    print(f"keep={keep}: {len(sel)} Frames, {size:.1f} MB")
    if size <= 9.5:
        break
print(f"GIF gespeichert: {args.out}")
