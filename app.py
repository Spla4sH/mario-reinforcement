"""Gradio-Live-Demo: trainierte Mario-Agenten im Browser zuschauen – mit Grad-CAM.

Ein Dropdown wählt das Level: den 1-1-Klassiker per Double DQN sowie alle mit PPO
gelösten Level (Welt 1–6). Eine Checkbox blendet das Grad-CAM-Overlay ein oder aus
(mit = "worauf achtet das Netz?", ohne = sauberes Original – und deutlich schneller,
weil dann nur ein Forward-Pass statt eines Grad-CAM-Backward-Passes nötig ist).

Lokal starten (PPO-Level brauchen das .venv-ppo mit Stable-Baselines3):
    python app.py

Deploy: als Hugging Face Space (SDK = gradio). CPU-Inferenz genügt.
"""

from __future__ import annotations

import gc
import os

_ACTIONS = 7  # SIMPLE_MOVEMENT: NOOP, right, right+A, right+B, right+A+B, A, left
# Obergrenze pro Episode. Auf der langsamen Free-CPU divergiert die greedy-
# Trajektorie oft (andere Float-Rundung als beim Training) und der Agent
# erreicht die Flagge nicht – dann würde er bis zum Level-Timeout weiterlaufen.
# 250 Schritte zeigen einen aussagekräftigen Ausschnitt und halten die Demo flott.
_MAX_STEPS = 250

# Verfügbare Level: Anzeigename -> Modelltyp, Checkpoint, World/Stage.
# 1-1 = selbst implementiertes Double DQN, alle übrigen = PPO (Stable-Baselines3).
_PPO = [
    ("1-2 · PPO", "mario_ppo_tuned.zip", 1, 2),
    ("1-3 · PPO", "mario_ppo_1-3_ent03.zip", 1, 3),
    ("1-4 · PPO (Schloss)", "mario_ppo_1-4.zip", 1, 4),
    ("2-1 · PPO (Trampolin-Superbounce)", "mario_ppo_2-1_tower.zip", 2, 1),
    ("2-2 · PPO (Wasser)", "mario_ppo_2-2.zip", 2, 2),
    ("2-3 · PPO (Brücken)", "mario_ppo_2-3.zip", 2, 3),
    ("2-4 · PPO (Bowser-Schloss)", "mario_ppo_2-4.zip", 2, 4),
    ("3-1 · PPO (Nacht)", "mario_ppo_3-1.zip", 3, 1),
    ("3-2 · PPO", "mario_ppo_3-2.zip", 3, 2),
    ("3-3 · PPO", "mario_ppo_3-3.zip", 3, 3),
    ("3-4 · PPO (Schloss)", "mario_ppo_3-4.zip", 3, 4),
    ("4-1 · PPO", "mario_ppo_4-1.zip", 4, 1),
    ("4-2 · PPO", "mario_ppo_4-2.zip", 4, 2),
    ("4-3 · PPO", "mario_ppo_4-3.zip", 4, 3),
    ("4-4 · PPO (Labyrinth, Go-Explore)", "mario_ppo_4-4_maze.zip", 4, 4),
    ("5-1 · PPO", "mario_ppo_5-1.zip", 5, 1),
    ("5-2 · PPO", "mario_ppo_5-2.zip", 5, 2),
    ("5-3 · PPO (Baumwipfel, Transfer von 1-3)", "mario_ppo_5-3.zip", 5, 3),
    ("5-4 · PPO (Schloss)", "mario_ppo_5-4.zip", 5, 4),
    ("6-1 · PPO", "mario_ppo_6-1.zip", 6, 1),
    ("6-2 · PPO", "mario_ppo_6-2.zip", 6, 2),
    ("6-3 · PPO (Baumwipfel)", "mario_ppo_6-3.zip", 6, 3),
    ("6-4 · PPO (Schloss)", "mario_ppo_6-4.zip", 6, 4),
]

LEVELS: dict[str, dict] = {
    "1-1 · Double DQN (der Klassiker)": {
        "type": "dqn", "path": "checkpoints/mario_best.pt", "world": 1, "stage": 1,
    },
}
for _label, _file, _w, _s in _PPO:
    LEVELS[_label] = {
        "type": "ppo", "path": f"checkpoints_ppo/{_file}", "world": _w, "stage": _s,
    }

# Geladene Modelle zwischenspeichern: ein Klick auf dasselbe Level lädt dann
# nichts neu (schneller + stabiler Speicher, kein Neuaufbau bei jedem Aufruf).
_CACHE: dict[str, object] = {}


class _DqnPredictor:
    """Double-DQN-Agent: greedy Aktion (schnell) + Grad-CAM (mit Backward)."""

    def __init__(self, path: str):
        import torch

        from agent import MarioAgent
        from visualize import GradCAM

        self._torch = torch
        self.agent = MarioAgent(_ACTIONS)
        self.agent.load(path)
        self.agent.online_net.eval()
        self.cam = GradCAM(self.agent.online_net, self.agent.online_net.features[4])

    def _tensor(self, state):
        return (
            self._torch.tensor(state, dtype=self._torch.uint8)
            .permute(2, 0, 1)
            .unsqueeze(0)
            .to(self.agent.device)
        )

    def act(self, state):  # mit Grad-CAM (Backward-Pass) -> (action, heat)
        return self.cam(self._tensor(state))

    def act_fast(self, state):  # nur Aktion, kein Backward -> viel schneller
        with self._torch.no_grad():
            return int(self.agent.online_net(self._tensor(state)).argmax(dim=1).item())


class _PpoPredictor:
    """SB3-PPO-Policy: greedy Aktion (schnell) + Grad-CAM auf dem Actor-Logit."""

    def __init__(self, path: str):
        import torch

        from stable_baselines3 import PPO

        from visualize import GradCAM
        from visualize_ppo import PolicyLogits

        self._torch = torch
        self.model = PPO.load(path)
        self.policy = self.model.policy
        self.policy.set_training_mode(False)
        self.cam = GradCAM(PolicyLogits(self.policy), self.policy.features_extractor.cnn[4])

    def act(self, state):  # mit Grad-CAM (Backward-Pass) -> (action, heat)
        obs_t, _ = self.policy.obs_to_tensor(state)
        action, heat = self.cam(obs_t)
        # PPO-Logit-Gradienten haben höheren Grundpegel als DQN-Q-Werte ->
        # Min abziehen und quadrieren, sonst liegt ein Schleier über dem Bild.
        heat = ((heat - heat.min()) / (heat.max() - heat.min() + 1e-8)) ** 2
        return action, heat

    def act_fast(self, state):  # nur Aktion, kein Backward -> viel schneller
        obs_t, _ = self.policy.obs_to_tensor(state)
        with self._torch.no_grad():
            probs = self.policy.get_distribution(obs_t).distribution.probs
            return int(probs.argmax(dim=1).item())


def _get_predictor(level_key: str, level: dict):
    predictor = _CACHE.get(level_key)
    if predictor is None:
        predictor = (_DqnPredictor if level["type"] == "dqn" else _PpoPredictor)(level["path"])
        _CACHE[level_key] = predictor
    return predictor


def _upscale(rgb, scale: int = 2):
    import cv2

    h, w = rgb.shape[:2]
    return cv2.resize(rgb, (w * scale, h * scale), interpolation=cv2.INTER_NEAREST)


def run_episode(level_key: str, show_cam: bool = True):
    """Spielt eine greedy Episode des gewählten Levels; gibt (GIF-Pfad, Text) zurück."""
    from visualize import make_overlay
    from record import save_gif
    from wrappers import create_env

    level = LEVELS.get(level_key) or next(iter(LEVELS.values()))
    if not os.path.exists(level["path"]):
        return None, f"Modell nicht gefunden: {level['path']}"

    predictor = _get_predictor(level_key, level)
    env = create_env(world=level["world"], stage=level["stage"], render=False)

    # Anti-Hänger-Impuls: Auf fremder Hardware (andere Float-Rundung als beim
    # Training) kann die greedy-Trajektorie divergieren und der Agent an einem
    # Hindernis "festlaufen". Verbessert sich x einige Schritte nicht, wird kurz
    # ein Sprung erzwungen – weiterhin deterministisch, aber er strampelt sich frei.
    frames = []
    state = env.reset()
    done = False
    info: dict = {}
    best_x, stuck, boost, nudges, steps = 0, 0, 0, 0, 0
    while not done and steps < _MAX_STEPS:
        if show_cam:
            action, heat = predictor.act(state)
        else:
            action, heat = predictor.act_fast(state), None
        if boost > 0:
            action = 4  # ['right', 'A', 'B']: Anlauf-Sprung
            boost -= 1
        rgb = env.render(mode="rgb_array")
        frames.append(make_overlay(rgb, heat, scale=2) if show_cam else _upscale(rgb, 2))
        state, _, done, info = env.step(action)
        steps += 1
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
    del frames
    gc.collect()  # GIF-Frames (mehrere hundert MB bei langen Levels) sofort freigeben

    flag = "🏁 Flagge erreicht!" if info.get("flag_get", False) else ""
    note = f" ({nudges} Anti-Hänger-Impulse)" if nudges else ""
    return out_path, f"x-Position: {info.get('x_pos', 0)}  {flag}{note}"


def build():
    import gradio as gr

    with gr.Blocks(title="Mario RL – KI-Vision") as demo:
        gr.Markdown(
            "# 🍄 Super Mario Bros – KI-Vision Demo\n"
            "Trainierte Agenten spielen **nur aus den Pixeln** – hier **24 gelöste Level** "
            "(1-1 = selbst implementiertes **Double DQN**, Rest = **PPO**).\n\n"
            "Das optionale **Grad-CAM-Overlay** (rot = wichtig) zeigt, worauf das neuronale "
            "Netz achtet – **ohne Häkchen läuft es deutlich schneller** (nur ein Forward-Pass "
            "statt eines Grad-CAM-Backward-Passes). Gezeigt wird ein Ausschnitt der Episode."
        )
        with gr.Row():
            level = gr.Dropdown(
                choices=list(LEVELS.keys()),
                value=next(iter(LEVELS.keys())),
                label="Level / Agent",
            )
            show_cam = gr.Checkbox(value=False, label="Grad-CAM-Overlay anzeigen (langsamer)")
        out_img = gr.Image(label="Lauf", type="filepath")
        out_txt = gr.Textbox(label="Ergebnis")
        gr.Button("▶ Episode spielen").click(
            run_episode, inputs=[level, show_cam], outputs=[out_img, out_txt]
        )
    return demo


# Kein Modulebenen-Build mehr: Der HF-Space nutzt seine eigene statische Kopie
# (deploy/app.py); hier würde ein Import sonst gradio erzwingen – andere Skripte
# (gen_demos.py, human_vs_ki.py) importieren nur LEVELS/Predictoren.
if __name__ == "__main__":
    build().launch()
