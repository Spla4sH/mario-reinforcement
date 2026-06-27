# Probelauf-Checkliste (Smoke Test)

Ziel: bestätigen, dass das Fundament trägt – **lernt der Agent auf 1-1 überhaupt?**
Erst wenn das klappt, lohnt sich Reward-Shaping/Tuning oder das Hinzunehmen weiterer Welten.

> Hinweis: Erst nächste Woche laufen lassen (GPU heizt das Zimmer auf). Alles ist
> vorbereitet – du musst nur noch starten und abhaken.

## 0. Vorbereitung (einmalig)
```bash
pip install -r requirements.txt   # falls noch nicht geschehen
python -c "import torch; print('CUDA:', torch.cuda.is_available())"
```
- [ ] `CUDA: True` → läuft auf der GPU (bei `False` trainiert es auf der CPU, viel langsamer)

### ⚠️ Bekannte Stolperfalle beim ersten Lauf (gym-Version)
Der Code nutzt die **alte gym-API** (4er-Tupel `obs, reward, done, info`; `reset()` gibt nur `obs`).
Ein frischer Install (besonders im **Docker-Build**) zieht evtl. ein neueres gym (≥0.26) und crasht
dann z. B. mit „too many values to unpack" oder „tuple has no attribute ...".

**Falls das passiert:** eine alte-API-Version pinnen und neu installieren:
```bash
pip install "gym==0.25.2"
```
Wenn dein lokales Env schon läuft, ist nichts zu tun – relevant v. a. für frische Installs/Container.
Bei Bedarf den funktionierenden Versionsstand danach mit `pip freeze > requirements.lock` festhalten.

## 1. Kurzer Probelauf starten
```bash
python train.py --episodes 300
```
- Öffnet das Original-NES-Fenster und trainiert 300 Episoden.
- Schreibt Metriken nach `logs/run_*.csv`, plottet alle 50 Episoden nach `plots/`.
- Mit **Strg+C** kannst du jederzeit sauber abbrechen (Checkpoint & Graphen werden gespeichert).

Optional schneller/ohne Fenster:
```bash
python train.py --episodes 300 --no-render
```

## 2. Während des Laufs in der Konsole beobachten
Jede Zeile zeigt u. a. `Position`, `Best X`, `Epsilon`, `Loss`.
- [ ] **Best X** steigt im Verlauf an (Mario kommt weiter nach rechts)
- [ ] **Epsilon** sinkt von 1.0 langsam Richtung 0.02
- [ ] **Loss** wird angezeigt (nicht dauerhaft `---`) und explodiert nicht (kein NaN/Riesenwerte)

## 3. Graphen prüfen
```bash
python plot.py
```
Öffne das PNG in `plots/`. Erfolgskriterien:
- [ ] **Level-Fortschritt (x_pos):** Trend (Ø-Linie) zeigt nach oben
- [ ] **Belohnung:** Trend nach oben
- [ ] **Loss:** pendelt sich ein / sinkt, läuft nicht weg
- [ ] **Epsilon:** fällt sauber ab

## 4. Trainierten Agenten ansehen
```bash
python play.py --episodes 3
```
- [ ] Mario spielt im Original-Bild **greedy** (keine Zufallszüge)
- [ ] Er kommt sichtbar weiter als ein zufälliger Anfänger (vergleiche mit erster Episode)

## Bewertung
- **Alle Haken gesetzt** → Fundament trägt. Nächster Schritt: **1-1 konsistent lösen**
  (Reward-Shaping / Hyperparameter-Tuning), Effekt direkt an den Graphen ablesbar.
- **x_pos/Reward bleiben flach** → erst Lernproblem untersuchen (Lernrate, Reward-Skala,
  Replay-Buffer, Epsilon-Decay), **bevor** weitere Welten dazukommen.

## Referenz: nützliche Befehle
| Zweck | Befehl |
|---|---|
| Kurzer Probelauf | `python train.py --episodes 300` |
| Volles Training | `python train.py` |
| Ohne Fenster (schneller) | `python train.py --no-render` |
| Graphen erzeugen | `python plot.py` |
| Agent zuschauen | `python play.py --episodes 3` |
