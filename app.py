"""Gradio-Live-Demo: trainierten Mario-Agenten im Browser zuschauen – mit Grad-CAM.

Lokal starten:
    pip install gradio
    python app.py

Deploy: als Hugging Face Space (SDK = gradio). Es wird ein Checkpoint
(`checkpoints/mario_best.pt`) benötigt; CPU-Inferenz reicht zum Zuschauen.
"""

from __future__ import annotations

import os

import config
from agent import MarioAgent


def _load_agent(checkpoint: str):
    from wrappers import create_env  # lazy: zieht cv2/gym nach sich

    env = create_env(world=config.WORLD, stage=config.STAGE, render=False)
    agent = MarioAgent(env.action_space.n)
    agent.load(checkpoint)
    agent.online_net.eval()
    return agent, env


def run_episode(checkpoint: str = "checkpoints/mario_best.pt"):
    """Spielt eine greedy Episode, gibt (GIF-Pfad, Ergebnis-Text) zurück."""
    import torch

    from record import save_gif
    from visualize import GradCAM, make_overlay

    if not os.path.exists(checkpoint):
        return None, f"Kein Checkpoint gefunden: {checkpoint} – zuerst trainieren."

    agent, env = _load_agent(checkpoint)
    cam = GradCAM(agent.online_net, agent.online_net.features[4])

    # Anti-Hänger-Impuls: Auf fremder Hardware (andere Float-Rundung als beim
    # Training) kann die greedy-Trajektorie divergieren und der Agent an einem
    # Hindernis "festlaufen". Verbessert sich x einige Schritte nicht, wird kurz
    # ein Sprung erzwungen – weiterhin deterministisch, aber er strampelt sich frei.
    frames = []
    state = env.reset()
    done = False
    info: dict = {}
    best_x, stuck, boost, nudges = 0, 0, 0, 0
    while not done:
        state_t = (
            torch.tensor(state, dtype=torch.uint8)
            .permute(2, 0, 1)
            .unsqueeze(0)
            .to(agent.device)
        )
        action, heat = cam(state_t)
        if boost > 0:
            action = 4  # ['right', 'A', 'B']: Anlauf-Sprung
            boost -= 1
        frames.append(make_overlay(env.render(mode="rgb_array"), heat, scale=2))
        state, _, done, info = env.step(action)
        x = int(info.get("x_pos", 0))
        if x > best_x:
            best_x, stuck = x, 0
        else:
            stuck += 1
            if stuck >= 15:
                boost, stuck = 6, 0
                nudges += 1
    env.close()

    os.makedirs("highlights", exist_ok=True)
    out_path = "highlights/demo.gif"
    save_gif(frames, out_path)
    flag = "🏁 Flagge erreicht!" if info.get("flag_get", False) else ""
    note = f" ({nudges} Anti-Hänger-Impulse)" if nudges else ""
    return out_path, f"x-Position: {info.get('x_pos', 0)}  {flag}{note}"


def build():
    import gradio as gr

    with gr.Blocks(title="Mario RL – KI-Vision") as demo:
        gr.Markdown(
            "# 🍄 Super Mario Bros – KI-Vision Demo\n"
            "Ein Double-DQN-Agent spielt **nur aus den Pixeln**. Die Heatmap (Grad-CAM) "
            "zeigt, **worauf das neuronale Netz** bei seiner Entscheidung achtet."
        )
        out_img = gr.Image(label="Lauf mit Grad-CAM-Overlay", type="filepath")
        out_txt = gr.Textbox(label="Ergebnis")
        gr.Button("▶ Neue Episode spielen").click(run_episode, outputs=[out_img, out_txt])
    return demo


# Auf Modulebene: Hugging Face (Gradio-SDK) importiert app.py und erwartet ein
# globales `demo` – der __main__-Block läuft dort nie.
demo = build()

if __name__ == "__main__":
    demo.launch()
