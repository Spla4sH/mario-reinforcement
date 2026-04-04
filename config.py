"""Hyperparameter und Konfiguration."""

# Umgebung
WORLD = 1
STAGE = 1
RENDER = True  # Live-Fenster anzeigen

# Netzwerk
LEARNING_RATE = 0.00025
GAMMA = 0.99  # Discount factor
BATCH_SIZE = 32
MEMORY_SIZE = 100_000
MIN_MEMORY = 1_000  # Mindestanzahl Erfahrungen vor dem Training

# Exploration
EPSILON_START = 1.0
EPSILON_END = 0.02
EPSILON_DECAY = 100_000  # Schritte bis Epsilon minimal ist

# Training
TARGET_UPDATE = 10_000  # Schritte zwischen Target-Network-Updates
SAVE_INTERVAL = 50  # Episoden zwischen Checkpoint-Speicherungen
MAX_EPISODES = 50_000

# Bildverarbeitung (für das neuronale Netz)
FRAME_STACK = 4  # Anzahl gestapelter Frames
FRAME_SIZE = 84  # Bildgröße für das Netz (84x84)
