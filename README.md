# Super Mario Bros - Reinforcement Learning

> **WIP** - Dieses Projekt befindet sich in aktiver Entwicklung.

Eine KI lernt Super Mario Bros zu spielen - mit Live-Fenster zum Zuschauen!

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red)
![Status](https://img.shields.io/badge/Status-Work%20in%20Progress-yellow)

https://github.com/user-attachments/assets/placeholder

## Was passiert hier?

Ein neuronales Netz lernt durch **Double DQN** (Deep Q-Network), Super Mario Bros Level 1-1 zu meistern. Die KI sieht nur die Pixel des Spielbildschirms und lernt selbstständig zu laufen, zu springen und Gegnern auszuweichen.

**Du kannst live zuschauen**, wie Mario anfangs planlos herumläuft und nach und nach besser wird.

## Wie funktioniert es?

| Komponente | Beschreibung |
|---|---|
| **Algorithmus** | Double DQN mit Experience Replay |
| **Input** | 4 gestapelte Graustufen-Frames (84x84) |
| **Output** | 7 Aktionen (laufen, springen, Kombos) |
| **Belohnung** | Fortschritt nach rechts, Münzen, Überleben |

```
Spielbild (240x256 RGB)
    |
    v
Vorverarbeitung (Graustufen, 84x84, 4 Frames gestapelt)
    |
    v
CNN (3x Conv + 2x FC) --> Q-Werte pro Aktion
    |
    v
Epsilon-Greedy Aktionsauswahl --> Mario bewegt sich
    |
    v
Belohnung + neuer Zustand --> Replay Memory --> Training
```

## Setup

```bash
# Repository klonen
git clone https://github.com/Spla4sH/mario-reinforcement.git
cd mario-reinforcement

# Abhängigkeiten installieren
pip install -r requirements.txt

# Training starten (öffnet Live-Spielfenster)
python train.py
```

**Voraussetzungen:**
- Python 3.10+
- NVIDIA GPU mit CUDA empfohlen (CPU funktioniert, aber deutlich langsamer)

## Projektstruktur

```
mario-reinforcement/
├── train.py          # Hauptskript - Training mit Live-Anzeige
├── agent.py          # Double DQN Agent + Replay Memory
├── model.py          # CNN-Architektur
├── wrappers.py       # Bildvorverarbeitung & Environment-Wrapper
├── config.py         # Hyperparameter
└── requirements.txt
```

## Konfiguration

In `config.py` lassen sich alle Hyperparameter anpassen:

| Parameter | Default | Beschreibung |
|---|---|---|
| `RENDER` | `True` | Live-Spielfenster anzeigen |
| `LEARNING_RATE` | `0.00025` | Lernrate |
| `EPSILON_DECAY` | `100.000` | Schritte bis Exploration minimal |
| `GAMMA` | `0.99` | Discount Factor |
| `MEMORY_SIZE` | `100.000` | Größe des Replay Buffers |

## Roadmap

- [x] Double DQN Agent
- [x] Live-Rendering des Original-Spiels
- [x] Checkpoint-System (Pause & Fortsetzen)
- [ ] Level 1-1 konsistent abschließen
- [ ] Alle Welten durchspielen
- [ ] Trainingsmetriken & Graphen
- [ ] Vortrainiertes Modell bereitstellen

## Technologien

- [gym-super-mario-bros](https://github.com/Kautenja/gym-super-mario-bros) - NES-Emulator als Gym-Umgebung
- [PyTorch](https://pytorch.org/) - Neuronales Netz & Training
- [OpenCV](https://opencv.org/) - Bildvorverarbeitung
