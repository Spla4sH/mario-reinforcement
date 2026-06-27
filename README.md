# Super Mario Bros - Reinforcement Learning

> **WIP** - Dieses Projekt befindet sich in aktiver Entwicklung.

Eine KI lernt Super Mario Bros zu spielen - mit Live-Fenster zum Zuschauen!

![CI](https://github.com/Spla4sH/mario-reinforcement/actions/workflows/ci.yml/badge.svg)
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

# Kurzer Probelauf (z. B. zum Testen), optional ohne Fenster
python train.py --episodes 300
python train.py --episodes 300 --no-render
```

**Voraussetzungen:**
- Python 3.10+
- NVIDIA GPU mit CUDA empfohlen (CPU funktioniert, aber deutlich langsamer)

> Erster Probelauf? Folge der Checkliste in [TESTLAUF.md](TESTLAUF.md).

### Mit Docker (headless Training)

```bash
# Image bauen
docker build -t mario-rl .

# Training im Container starten (GPU durchreichen)
docker run --gpus all mario-rl python train.py --no-render --episodes 2000
```

Für den Cluster liegt unter [k8s/](k8s/) ein Kubernetes-`Job` inkl. `PersistentVolumeClaim`
(GPU-Request, Checkpoints/Logs persistent). Details & Sweep-Ideen in
[NAECHSTE_SCHRITTE.md](NAECHSTE_SCHRITTE.md) (Phase D).

### Tests & Linting

```bash
pip install -r requirements-dev.txt
ruff check .   # Linting
pytest         # Tests
```

Die Tests laufen ohne Emulator/GPU (sie prüfen Replay-Memory, Metriken, Plots,
Reward-Shaping, Greedy-Eval, Grad-CAM-Logik und die Config) und werden bei jedem
Push automatisch per [GitHub Actions](.github/workflows/ci.yml) ausgeführt.

## Projektstruktur

```
mario-reinforcement/
├── train.py          # Hauptskript - Training mit Live-Anzeige
├── play.py           # Trainierten Agenten greedy zuschauen (Originalbild)
├── visualize.py      # KI-Vision-Overlay (Grad-CAM) - was sieht die KI?
├── agent.py          # Double DQN Agent + Replay Memory
├── model.py          # CNN-Architektur
├── wrappers.py       # Bildvorverarbeitung & Environment-Wrapper
├── reward.py         # Reward-Shaping (Skalierung/Clipping)
├── evaluate.py       # Greedy-Evaluation (flag_get-Rate, x-Position)
├── metrics.py        # CSV-Logging der Trainingsmetriken
├── plot.py           # Graphen aus den Metriken erzeugen
├── config.py         # Hyperparameter (per Env-Variablen überschreibbar)
├── tests/            # pytest-Suite
├── Dockerfile        # Headless-Training als Container
├── k8s/              # Kubernetes Job + PersistentVolume
├── .github/          # GitHub-Actions-CI (Lint + Tests)
└── requirements.txt
```

## Zuschauen & Auswerten

```bash
# Einem trainierten Agenten live im Original-Spielbild zuschauen (kein Training)
python play.py --episodes 5

# Graphen aus der neuesten Trainings-CSV erzeugen (logs/ -> plots/)
python plot.py
```

### KI-Vision: Was sieht die KI? (Grad-CAM)

```bash
# Heatmap-Overlay live anzeigen – rot = wichtig für die gewählte Aktion
python visualize.py --episodes 2

# Zusätzlich als GIF speichern (z. B. für dieses README)
python visualize.py --episodes 1 --save vision.gif
```

Per **Grad-CAM** wird der letzte Convolution-Layer ausgewertet und als Heatmap über
das Original-Spielbild gelegt. So wird sichtbar, **welche Bildregionen** (Gegner, Lücken,
Plattformen) die Entscheidung des Netzes gerade antreiben – ein direkter Blick „in den Kopf"
des Agenten.

Während des Trainings werden alle Metriken pro Episode nach `logs/run_*.csv`
geschrieben und bei jedem Checkpoint automatisch als Graph nach `plots/` geplottet.
Die ROM ist im Paket `gym-super-mario-bros` enthalten – es muss nichts separat
heruntergeladen werden.

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
- [x] Trainingsmetriken & Graphen
- [x] Wiedergabe-Modus für trainierte Agenten (`play.py`)
- [ ] Level 1-1 konsistent abschließen
- [ ] Alle Welten durchspielen
- [ ] Vortrainiertes Modell bereitstellen

### Vision / Weiterentwicklung

- [x] **KI-Vision-Overlay** (Grad-CAM) – sichtbar machen, worauf das CNN achtet
- [x] Containerisierung: Docker-Image + Kubernetes-Job für headless Training
- [x] Tests + CI (pytest, ruff, GitHub Actions)
- [x] Reward-Shaping + Greedy-Evaluation (Basis für „1-1 konsistent")
- [ ] Algorithmus-Upgrade: Double DQN → **PPO** (Stable-Baselines3) für stabileres Training
- [ ] Generalisierung: zweite Umgebung (Atari) über dieselbe Pixel-Pipeline
- [ ] MLOps-Ausbau: Hyperparameter-Sweeps als parallele K8s-Jobs

Detaillierter Fahrplan: [NAECHSTE_SCHRITTE.md](NAECHSTE_SCHRITTE.md).

## Technologien

- [gym-super-mario-bros](https://github.com/Kautenja/gym-super-mario-bros) - NES-Emulator als Gym-Umgebung
- [PyTorch](https://pytorch.org/) - Neuronales Netz & Training
- [OpenCV](https://opencv.org/) - Bildvorverarbeitung

## Entwicklung

Die Architektur- und Design-Entscheidungen (Double DQN, Reward-Strategie, Roadmap)
stammen von mir. Die Umsetzung entstand AI-assistiert mit [Claude Code](https://claude.com/claude-code)
als Pair-Programmer – entsprechende Beiträge sind in der Git-History per
`Co-Authored-By` ausgewiesen.
