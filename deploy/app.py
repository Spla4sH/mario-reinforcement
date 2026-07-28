"""Gradio-Demo: trainierte Mario-Agenten spielen 20 Level – mit optionalem Grad-CAM.

Rein statisch: Die Läufe wurden vorab lokal berechnet (volle Episoden bis zur
Flagge, auf trainierter Hardware) und liegen als MP4 in ``demos/`` – der Space
zeigt sie sofort, ohne ML-Stack und ohne Live-Rechnung. Das eigentliche RL-Können
steckt in den Modellen; hier geht es ums Zeigen. Erzeugt mit ``gen_demos.py``.

Deploy: diese Datei als ``app.py`` in den Hugging Face Space (SDK = gradio)
hochladen, zusammen mit ``demos/`` (MP4s + results.json). Braucht nur Gradio –
kein torch, kein Stable-Baselines3, kein Emulator (schneller Build, sofortiger
Start). Anleitung: DEPLOY.md.
"""

from __future__ import annotations

import json
import os

_HERE = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(_HERE, "demos", "results.json"), encoding="utf-8") as _f:
    RESULTS: dict = json.load(_f)


def show_demo(level_key: str, show_cam: bool):
    """Gibt (Video-Pfad, Ergebnistext) des gewählten Levels/Modus zurück."""
    entry = RESULTS.get(level_key) or next(iter(RESULTS.values()))
    variant = entry["cam" if show_cam else "plain"]
    return os.path.join(_HERE, variant["video"]), variant["text"]


def build():
    import gradio as gr

    first = next(iter(RESULTS))
    with gr.Blocks(title="Mario RL – KI-Vision") as demo:
        gr.Markdown(
            "# 🍄 Super Mario Bros – KI-Vision Demo\n"
            "Trainierte Agenten spielen **nur aus den Pixeln** – hier **20 gelöste Level** "
            "(1-1 = selbst implementiertes **Double DQN**, Rest = **PPO**, Stable-Baselines3).\n\n"
            "Wähle ein Level und schau dem Lauf bis zur Flagge zu. Das optionale "
            "**Grad-CAM-Overlay** (rot = wichtig) zeigt, worauf das neuronale Netz bei "
            "jeder Entscheidung achtet – ein Blick 'in den Kopf' des Agenten."
        )
        with gr.Row():
            level = gr.Dropdown(choices=list(RESULTS.keys()), value=first, label="Level / Agent")
            show_cam = gr.Checkbox(value=False, label="Grad-CAM-Overlay anzeigen")
        # MP4: volle Auflösung + Framerate, klein dank H.264. autoplay + loop
        # lassen den Lauf sofort und endlos laufen (wie ein GIF, nur scharf).
        # height begrenzt die Anzeigegröße (sonst füllt das Video den Container).
        out_vid = gr.Video(label="Lauf", autoplay=True, loop=True, height=480)
        out_txt = gr.Textbox(label="Ergebnis")
        btn = gr.Button("▶ Lauf anzeigen")
        btn.click(show_demo, inputs=[level, show_cam], outputs=[out_vid, out_txt])
        # Beim Laden gleich den ersten Lauf zeigen.
        demo.load(show_demo, inputs=[level, show_cam], outputs=[out_vid, out_txt])
    return demo


# Auf Modulebene: Hugging Face (Gradio-SDK) importiert app.py und erwartet `demo`.
demo = build()

if __name__ == "__main__":
    demo.launch()
