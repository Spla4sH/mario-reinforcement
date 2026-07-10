"""Erzeugt für alle Level je ein Demo-Video (MP4) mit und ohne Grad-CAM + results.json.

MP4 statt GIF: H.264 komprimiert die bunten Grad-CAM-Overlays ~10x besser als GIF,
daher sind die Läufe in voller Auflösung (512x480), voller Framerate und trotzdem
klein (~0.5-2 MB). Volle Läufe bis zur Flagge, auf lokaler (= trainierter) Hardware.
Ergebnis landet in hf_space/demos/ und wird zusammen mit deploy/app.py in den
Hugging-Face-Space hochgeladen (siehe DEPLOY.md) – der Space braucht dann keinen
ML-Stack mehr.

Ausführen im .venv-ppo (braucht torch für DQN *und* SB3 für PPO), aus dem Repo-Root:
    .venv-ppo/Scripts/python gen_demos.py
"""
import json
import os
import sys
import types

sys.stdout.reconfigure(encoding="utf-8")

# gradio wird nur von app.py beim Import verlangt, nicht fürs Rechnen – stubben,
# damit die Demo-Erzeugung ohne installiertes gradio läuft.
g = types.ModuleType("gradio")
class _C:
    def __enter__(self): return self
    def __exit__(self, *a): return False
class _B:
    def click(self, *a, **k): return None
g.Blocks = lambda *a, **k: _C()
g.Row = lambda *a, **k: _C()
g.Markdown = g.Dropdown = g.Image = g.Video = g.Textbox = g.Checkbox = lambda *a, **k: None
g.Button = lambda *a, **k: _B()
sys.modules["gradio"] = g

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import imageio.v2 as imageio  # noqa: E402

import app  # noqa: E402  (die lokale Live-App liefert LEVELS + Predictoren)
from visualize import make_overlay  # noqa: E402
from wrappers import create_env  # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hf_space", "demos")
os.makedirs(OUT, exist_ok=True)


def play(predictor, world, stage, show_cam):
    """Volle native Episode in voller Auflösung (scale=2, kein Anti-Hänger)."""
    env = create_env(world=world, stage=stage, render=False)
    frames, state, done, info = [], env.reset(), False, {}
    while not done:
        if show_cam:
            action, heat = predictor.act(state)
        else:
            action, heat = predictor.act_fast(state), None
        rgb = env.render(mode="rgb_array")
        frames.append(make_overlay(rgb, heat, scale=2) if show_cam else app._upscale(rgb, 2))
        state, _, done, info = env.step(action)
    env.close()
    return frames, int(info.get("x_pos", 0)), bool(info.get("flag_get"))


def save_mp4(frames, path, fps=30):
    imageio.mimsave(
        path, frames, fps=fps, codec="libx264", quality=8,
        macro_block_size=1, output_params=["-pix_fmt", "yuv420p"],
    )
    return len(frames), os.path.getsize(path) / 1e6


if __name__ == "__main__":
    results = {}
    for level_key, level in app.LEVELS.items():
        w, s = level["world"], level["stage"]
        predictor = (app._DqnPredictor if level["type"] == "dqn" else app._PpoPredictor)(level["path"])
        for cam in (False, True):
            frames, x, flag = play(predictor, w, s, cam)
            name = f"{w}-{s}_{'cam' if cam else 'plain'}.mp4"
            n, mb = save_mp4(frames, os.path.join(OUT, name))
            flag_s = "🏁 Flagge erreicht!" if flag else ""
            results.setdefault(level_key, {})[("cam" if cam else "plain")] = {
                "video": f"demos/{name}",
                "text": f"x-Position: {x}  {flag_s}".strip(),
            }
            print(f"{name}: {n} Frames, {mb:.2f} MB | x={x} flag={flag}")

    with open(os.path.join(OUT, "results.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=1)
    print(f"\nresults.json geschrieben ({len(results)} Level).")
