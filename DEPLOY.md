# Deploy: Gradio-Demo als Hugging Face Space

Ziel: ein **öffentlicher, klickbarer Link**, auf dem die trainierten Agenten im Browser
spielen (alle 16 gelösten Level, optional mit Grad-CAM-Overlay) – ohne Installation für
den Betrachter. Läuft live: <https://huggingface.co/spaces/Spl4sH/mario-reinforcement>

**Der Space ist rein statisch:** Die Läufe werden vorab lokal berechnet (volle Episoden
bis zur Flagge, auf der trainierten Hardware) und als MP4 hochgeladen. Der Space selbst
braucht **nur Gradio** – kein torch, kein Stable-Baselines3, kein Emulator.

> Warum nicht live rechnen? Zwei Gründe, beide auf der Gratis-CPU erprobt:
> (1) zu langsam (~90 s pro Lauf, OOM bei Mehrfachklicks); (2) auf fremder Hardware
> **divergiert die deterministische Trajektorie** (andere BLAS-Rundung → knapper argmax
> kippt → der Agent strandet statt die Flagge zu erreichen). Details im README.

## Schritt für Schritt

1. **Demos lokal erzeugen** (einmalig bzw. wenn neue Level dazukommen):
   ```bash
   .venv-ppo/Scripts/python gen_demos.py
   ```
   Erzeugt pro Level zwei MP4s (mit/ohne Grad-CAM, H.264, 512×480) plus `results.json`
   nach `hf_space/demos/` – zusammen ~55 MB für 16 Level.

2. **Space anlegen** auf <https://huggingface.co>
   → *New Space* → **SDK: Gradio**, Hardware: *CPU basic* (gratis).

3. **Diese Dateien in den Space-Root hochladen:**
   | Quelle im Repo | Im Space ablegen als |
   |---|---|
   | `deploy/app.py` | `app.py` |
   | `hf_space/demos/` (30 MP4s + `results.json`) | `demos/` |

   Eine `requirements.txt`/`packages.txt` ist **nicht nötig** – Gradio kommt über das
   Space-SDK, und mehr braucht die statische App nicht.

4. **Space-`README.md`** mit diesem YAML-Kopf anlegen (steuert den Space):
   ```yaml
   ---
   title: Mario RL - KI-Vision
   emoji: 🍄
   colorFrom: red
   colorTo: green
   sdk: gradio
   sdk_version: 5.35.0
   app_file: app.py
   pinned: false
   ---
   ```
   > `sdk_version` **pinnen** – ohne Pin zieht HF die neueste Gradio-Version, die die
   > App schon einmal gebrochen hat.

5. **Fertig.** HF baut in unter einer Minute (keine Dependencies) und gibt dir eine
   öffentliche URL (`https://huggingface.co/spaces/<user>/<space>`). In CV/GitHub verlinken.

## Level-Namen im Dropdown ändern

Die Dropdown-Beschriftungen sind die **Keys in `demos/results.json`** (z. B.
`"2-1 · PPO (Trampolin-Superbounce)"`). Zum Umbenennen die Keys in der JSON-Datei
ändern (lokal in `hf_space/demos/results.json` oder direkt im Space-Editor) – `app.py`
liest die Namen von dort.

## Hinweise & gelernte Stolperfallen (aus der Live-Variante)

Der erste Anlauf rechnete die Episoden **live im Space** (torch + SB3 + Emulator).
Die dabei gesammelten Erkenntnisse, falls jemand den Weg erneut versucht:

- **`packages.txt` darf keine Kommentare enthalten** – jede Zeile wird als apt-Paketname
  geparst. Für den nes-py-Build braucht es `build-essential`.
- **`demo` muss auf Modulebene existieren** (`demo = build()`), sonst hängt der Space in
  „Application Startup".
- **gym-Version:** Der Code nutzt die alte gym-API (4er-Tupel) → `gym==0.25.2` pinnen.
- **`nes-py==8.2.1` pinnen** – 9.x liefert Gymnasium-Spaces und crasht die alte Pipeline.
- **OpenCV headless** (`opencv-python-headless`) vermeidet die `libGL`-Abhängigkeit.
- **„Scheduling failure"** beim Start ist ein HF-Kapazitätsproblem – warten und den
  Space neu starten, kein Fehler im eigenen Code.
- Die alte Live-`requirements.txt` liegt noch als Referenz in
  [deploy/requirements-live.txt](deploy/requirements-live.txt).

## Alternative ohne Hosting

Wenn dir der Space zu viel ist: lokal `python visualize.py --save vision.gif` (oder das
automatische `highlights/best_run.gif`) erzeugen und als GIF ins README einbetten – null
Hosting, funktioniert immer.
