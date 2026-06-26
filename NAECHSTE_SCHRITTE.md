# Nächste Schritte (ab nächster Woche)

Reihenfolge bewusst: **erst Smoke Test, dann 1-1 konsistent lösen, dann erst Ausbau.**
Nicht skalieren, bevor das Fundament bewiesen ist.

---

## Phase A — Smoke Test (Tag 1)
Komplett in [TESTLAUF.md](TESTLAUF.md) beschrieben. Kurz:
```bash
python train.py --episodes 300
python plot.py
python play.py --episodes 3
```
**Gate:** Steigt `x_pos`/Reward im Trend? Sinkt der Loss kontrolliert?
- ✅ Ja → weiter zu Phase B.
- ❌ Flach/instabil → zuerst Phase B-Punkt 1+2 (Reward & Decay) angehen, dann erneut testen.

---

## Phase B — Level 1-1 konsistent lösen
Ziel-Definition **vorab festlegen**, damit „konsistent" messbar ist:

> **Erfolgskriterium:** In einer Greedy-Evaluation (epsilon=0) erreicht Mario in
> **≥ 80 % von 20 Episoden** die Flagge (`flag_get`).

Aufgaben, nach Hebelwirkung sortiert:

### 1. Greedy-Evaluation + Erfolgsmetrik (zuerst – ohne Messung kein Tuning)
- Periodisch (z. B. alle 50 Episoden) N Greedy-Episoden ohne Exploration laufen lassen
  und **flag_get-Rate** + mittlere `x_pos` loggen. Wiederverwendet `agent.act()` und die
  `MetricsLogger`-Struktur; ggf. zweite CSV-Spaltengruppe oder separates `eval_*.csv`.
- **Bestes Modell separat speichern** (`mario_best.pt`), wenn die Eval-`x_pos`/flag_get-Rate
  steigt – nicht nur den letzten Stand wie aktuell in `agent.save()`.
- *Warum:* Trainings-Reward schwankt durch Exploration; die Greedy-Eval zeigt den echten Fortschritt.

### 2. Reward-Skalierung / Clipping
- Der Default-Reward von `gym-super-mario-bros` (x-Fortschritt + Zeitstrafe + Todesstrafe)
  kann betragsmäßig groß schwanken → instabiles Training. Reward in `wrappers.py`
  (neuer `RewardWrapper`) **skalieren** (z. B. /10) oder auf `[-1, 1]` **clippen**.
- *Warum:* Stabilere Q-Targets, gängige Praxis bei DQN.

### 3. Exploration länger strecken
- `EPSILON_DECAY = 100_000` Agent-Schritte = ~400k Frames (skip=4). Für Mario oft zu kurz.
  Testweise auf **500k–1M** erhöhen (`config.py`).
- *Warum:* Mario braucht mehr Exploration, um Sprung-Kombinationen zu finden.

### 4. Replay-Buffer / Speicher (nur falls RAM limitiert)
- Aktuell werden `state` UND `next_state` als volle (84×84×4)-uint8-Stacks gespeichert
  → ~5–6 GB RAM bei `MEMORY_SIZE=100_000`. Falls das knapp wird: nur Einzelframes speichern
  und Stacks erst beim Sampling bauen (LazyFrames-Prinzip in `agent.py`/`wrappers.py`).
- *Warum:* Mehr Kapazität bei gleichem RAM, optional. Erst anfassen, wenn es klemmt.

### 5. Tuning-Schleife
- Eine Änderung pro Lauf, Effekt an den `plot.py`-Graphen ablesen, beste Konfig behalten.
- Reihenfolge: erst (1)+(2) zusammen, dann (3), dann ggf. Feintuning `LEARNING_RATE`/`TARGET_UPDATE`.

**Abschluss Phase B:** Erfolgskriterium erreicht → kurzes GIF aus `play.py` fürs README
(Roadmap-Punkt „Vortrainiertes Modell"), `mario_best.pt` als Release anbieten.

---

## Phase C — Ausbau (danach)
1. **Weitere Welten/Stages** – mit funktionierendem 1-1 als Basis. Curriculum (1-1 → 1-2 …)
   oder zufällige Stage-Auswahl pro Episode, um Generalisierung zu erzwingen.
2. **Stärkerer Algorithmus statt anderes Spiel** (siehe unten) – der größere Lerneffekt.

---

## Frage: später ein anderes / komplexeres Spiel?

**Kurzantwort:** Für dein Portfolio bringt ein *stärkerer Algorithmus* mehr als ein neues Spiel.
Tiefe schlägt Breite – ein halb fertiges zweites Spiel wirkt schwächer als ein gemeistertes Mario
plus saubere algorithmische Weiterentwicklung.

**Empfohlene Progression (portfolio-stark, erzählbar im Interview):**
1. **DQN → Double DQN → PPO.** Auf **PPO** (via [Stable-Baselines3](https://stable-baselines3.readthedocs.io/))
   umstellen ist *der* Schritt: PPO ist sample-effizienter, stabiler und Industriestandard –
   löst Mario zuverlässiger und zeigt, dass du moderne Policy-Gradient-Methoden beherrschst.
   Die bestehende `wrappers.py`-Pipeline lässt sich weitgehend wiederverwenden.
2. **Optional ein zweites Environment zur Demo von Generalisierung** – am günstigsten
   **Atari** (über `gymnasium[atari]`), weil dieselbe CNN-Pixel-Pipeline transferiert. Zeigt Breite
   mit minimalem Mehraufwand.

**Wenn es bewusst „komplexer" sein soll** (eher *nach* dem PPO-Schritt):
- **Continuous Control** (MuJoCo/PyBullet) mit PPO/SAC – anderer Aktionsraum, zeigt zusätzliches Können.
- **VizDoom / 3D** – partielle Beobachtbarkeit, deutlich schwerer.
- **Procgen** – explizit auf Generalisierung ausgelegt.

**Meine Empfehlung in einem Satz:** Mario fertig machen (1-1 → Welten), dann auf **PPO** umstellen,
und *eine* Generalisierungs-Demo (Atari) anhängen – das ergibt eine klare, beeindruckende Story,
statt vieler angefangener Spiele.
