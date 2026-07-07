"""Hyperparameter und Konfiguration.

Ausgewählte Werte lassen sich per Umgebungsvariable überschreiben – praktisch
für Hyperparameter-Sweeps (z. B. mehrere Kubernetes-Jobs mit unterschiedlichen
Lernraten), ohne den Code zu ändern:

    LEARNING_RATE=0.0001 EPSILON_DECAY=500000 python train.py --no-render
"""

import os


def _env_float(name: str, default: float) -> float:
    value = os.environ.get(name)
    return float(value) if value is not None else default


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    return int(value) if value is not None else default


def _env_float_or_none(name: str, default: float | None) -> float | None:
    value = os.environ.get(name)
    return float(value) if value not in (None, "") else default


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.lower() in ("1", "true", "yes", "on")


# Umgebung
WORLD = _env_int("WORLD", 1)
STAGE = _env_int("STAGE", 1)
RENDER = True  # Live-Fenster anzeigen

# Netzwerk
LEARNING_RATE = _env_float("LEARNING_RATE", 0.00025)
GAMMA = _env_float("GAMMA", 0.99)  # Discount factor
BATCH_SIZE = _env_int("BATCH_SIZE", 32)
MEMORY_SIZE = _env_int("MEMORY_SIZE", 100_000)
MIN_MEMORY = _env_int("MIN_MEMORY", 1_000)  # Mindesterfahrungen vor dem Training

# Exploration
EPSILON_START = _env_float("EPSILON_START", 1.0)
EPSILON_END = _env_float("EPSILON_END", 0.02)
EPSILON_DECAY = _env_int("EPSILON_DECAY", 100_000)  # Schritte bis Epsilon minimal

# Reward-Shaping (siehe reward.py / wrappers.RewardWrapper)
REWARD_SCALE = _env_float("REWARD_SCALE", 1.0)  # 1.0 = unverändert
REWARD_CLIP = _env_float_or_none("REWARD_CLIP", None)  # z. B. 1.0 für [-1, 1]

# Training
TARGET_UPDATE = _env_int("TARGET_UPDATE", 10_000)  # Schritte zwischen Target-Updates
SAVE_INTERVAL = _env_int("SAVE_INTERVAL", 50)  # Episoden zwischen Checkpoints
MAX_EPISODES = _env_int("MAX_EPISODES", 50_000)
SEED = _env_int("SEED", 42)  # Reproduzierbarkeit (für Sweeps pro Job variieren)

# Ausgabeorte – für parallele Läufe / Sweeps isolierbar, damit sich Checkpoints
# und Highlight-GIFs nicht gegenseitig überschreiben.
CHECKPOINT_DIR = os.environ.get("CHECKPOINT_DIR", "checkpoints")
HIGHLIGHT_DIR = os.environ.get("HIGHLIGHT_DIR", "highlights")

# Warm-Start / Transfer fürs Curriculum: Pfad zu einem Checkpoint, dessen
# Netzgewichte beim Start übernommen werden (Epsilon/Step bleiben frisch).
# Greift nur, wenn im CHECKPOINT_DIR noch kein eigener Checkpoint liegt.
INIT_FROM = os.environ.get("INIT_FROM", "")

# Experiment-Tracking (optional, siehe tracking.py) – aktivieren mit USE_WANDB=1
USE_WANDB = _env_bool("USE_WANDB", False)
WANDB_PROJECT = os.environ.get("WANDB_PROJECT", "mario-rl")

# Greedy-Evaluation (echter Fortschritt ohne Exploration)
EVAL_INTERVAL = _env_int("EVAL_INTERVAL", 50)  # Episoden zwischen Evaluationen
EVAL_EPISODES = _env_int("EVAL_EPISODES", 5)  # Greedy-Episoden pro Evaluation

# Bildverarbeitung (für das neuronale Netz)
FRAME_STACK = _env_int("FRAME_STACK", 4)  # Anzahl gestapelter Frames
FRAME_SIZE = _env_int("FRAME_SIZE", 84)  # Bildgröße für das Netz (84x84)
# Aktions-Taktung: 1 Agent-Entscheidung pro FRAME_SKIP NES-Frames. 4 = Standard;
# 2 für Stellen, die feineres Timing brauchen (z. B. 2-1-Trampolinturm).
# ACHTUNG: Modelle sind an ihren Skip gebunden – bei Eval/Play denselben Wert setzen!
FRAME_SKIP = _env_int("FRAME_SKIP", 4)
