# Deploy: Gradio-Demo als Hugging Face Space

Ziel: ein **öffentlicher, klickbarer Link**, auf dem der trainierte Agent im Browser spielt
(mit Grad-CAM-Overlay) – ohne Installation für den Betrachter. CPU genügt.

> Voraussetzung: ein trainiertes Modell (`checkpoints/mario_best.pt`). Entsteht automatisch
> beim Training (neues Bestmodell). Ohne Training gibt es nichts Sinnvolles zu zeigen.

## Schritt für Schritt

1. **Account & Space anlegen** auf <https://huggingface.co>
   → *New Space* → **SDK: Gradio**, Hardware: *CPU basic* (gratis).

2. **Diese Dateien in den Space-Root hochladen:**
   | Quelle im Repo | Im Space ablegen als |
   |---|---|
   | `app.py` | `app.py` |
   | `agent.py`, `model.py`, `wrappers.py`, `reward.py`, `config.py`, `visualize.py`, `record.py` | gleiche Namen |
   | `deploy/requirements.txt` | `requirements.txt` |
   | `deploy/packages.txt` | `packages.txt` |
   | `checkpoints/mario_best.pt` | `checkpoints/mario_best.pt` |

   > `checkpoints/` ist im Haupt-Repo per `.gitignore` ausgeschlossen – für den Space lädst du
   > die `.pt`-Datei **bewusst mit hoch** (oder per Git LFS, falls > 10 MB).

3. **Space-`README.md`** mit diesem YAML-Kopf anlegen (steuert den Space):
   ```yaml
   ---
   title: Mario RL - KI-Vision
   emoji: 🍄
   colorFrom: red
   colorTo: green
   sdk: gradio
   app_file: app.py
   pinned: false
   ---
   ```

4. **Fertig.** HF baut automatisch und gibt dir eine öffentliche URL
   (`https://huggingface.co/spaces/<user>/<space>`). Diese in CV/GitHub verlinken.

## Hinweise & Troubleshooting

- **gym-Version:** Der Code nutzt die alte gym-API (4er-Tupel). Deshalb ist `gym==0.25.2`
  gepinnt. Ohne Pin zieht ein frischer Build evtl. gym ≥0.26 → Crash.
- **OpenCV headless:** `opencv-python-headless` vermeidet die `libGL`-Abhängigkeit auf dem
  Server. Falls du stattdessen `opencv-python` nimmst, ergänze `libgl1` in `packages.txt`.
- **Image zu groß / Build langsam:** Optional CPU-Torch-Wheel erzwingen, indem du in der
  `requirements.txt` `torch` ersetzt durch:
  ```
  --extra-index-url https://download.pytorch.org/whl/cpu
  torch
  ```
- **`nes-py` baut nicht:** dafür ist `build-essential` in `packages.txt` (C-Extension-Compiler).
- **Kaltstart:** Der erste Klick lädt das Modell + spielt eine Episode – das dauert ein paar
  Sekunden. Danach ist es flott.

## Alternative ohne Hosting

Wenn dir der Space zu viel ist: lokal `python visualize.py --save vision.gif` (oder das
automatische `highlights/best_run.gif`) erzeugen und als GIF ins README einbetten – null Hosting,
funktioniert immer.
