# Nächste Schritte

Reihenfolge bewusst: **erst Smoke Test, dann 1-1 konsistent lösen, dann erst Ausbau.**
Nicht skalieren, bevor das Fundament bewiesen ist.

> **Stand (03.07.2026):** Phase A ✅ und Phase B ✅ **abgeschlossen**. Der Hauptlauf löste 1-1
> im Training 93× und erreichte ab Ep. ~5.250 die greedy-Konsistenz. **Erfolgskriterium erfüllt:**
> ein separater 20-Episoden-Greedy-Test gegen `mario_best.pt` ergab **20/20 Flagge** (x 3161).
> Hinweis: Die Env ist bei greedy-Spiel deterministisch → „deterministisch gelöst", nicht
> stochastische Robustheit. Beim ersten Lauf gefixt: `view`→`reshape`-Crash in `model.py`
> (non-contiguous States nach `permute`) + UTF-8-Logausgabe, siehe TESTLAUF.md.
> **Nächster Schritt:** Phase C (weitere Welten) bzw. Algorithmus-Upgrade auf PPO.

---

## Phase A — Smoke Test (Tag 1) ✅ bestanden
Komplett in [TESTLAUF.md](TESTLAUF.md) beschrieben. Kurz:
```bash
python train.py --episodes 300
python plot.py
python play.py --episodes 3
```
**Gate:** Steigt `x_pos`/Reward im Trend? Sinkt der Loss kontrolliert?
- ✅ Ja → weiter zu Phase B.
- ❌ Flach/instabil → zuerst Phase B-Punkt 1+2 (Reward & Decay) angehen, dann erneut testen.

**Vision-GIF fürs README aufnehmen** (sobald ein halbwegs brauchbarer Checkpoint da ist –
auch ein teiltrainierter Agent gibt schon ein cooles GIF):
```bash
python visualize.py --episodes 1 --save vision.gif
```
Dann ins README einbinden (`![KI-Vision](vision.gif)`). Nach Phase B nochmal mit dem
finalen `mario_best.pt` neu aufnehmen für die polierte Version.

---

## Phase B — Level 1-1 konsistent lösen ✅ erreicht
Ziel-Definition **vorab festlegen**, damit „konsistent" messbar ist:

> **Erfolgskriterium:** In einer Greedy-Evaluation (epsilon=0) erreicht Mario in
> **≥ 80 % von 20 Episoden** die Flagge (`flag_get`).
> → **Erreicht (03.07.2026): 20/20 Flagge** gegen `mario_best.pt`.

> **Bereits im Code vorbereitet** (nur noch nutzen/tunen): Greedy-Eval + Best-Checkpoint
> (`evaluate.py`, läuft alle `EVAL_INTERVAL` Episoden, speichert `mario_best.pt`),
> Reward-Shaping (`reward.py` / `RewardWrapper`, via `REWARD_SCALE`/`REWARD_CLIP`),
> Hyperparameter per Env-Variablen (`config.py`). Nächste Woche bleibt v. a. **Tuning**.

> **Beobachtung 1. Lauf → konkreter Konsistenz-Fokus:** Training löst 1-1 (38×), greedy-Eval
> aber 0 %. Zwei Ansatzpunkte, in dieser Reihenfolge testen:
> 1. **Exploration verlängern** (Punkt 3 unten): Beim Fortsetzen fiel ε durch den Einstieg bei
>    Step 72.573 quasi sofort auf 0.02 – der Agent hatte kaum noch Explorationsbudget. Für einen
>    sauberen Vergleich einen **frischen** Lauf mit größerem `EPSILON_DECAY` (500k–1M) fahren.
> 2. **Mehr Training bei Ziel-ε**: Die greedy-Konsistenz kommt oft erst deutlich nach den ersten
>    Flaggen – Rate steigt bereits (18→13 pro 1.000 Ep.). Eval-Kurve (`grep '\[Eval\]'`) beobachten,
>    ob greedy > 0 % zieht, bevor an Hyperparametern gedreht wird.

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

## Phase D — Containerisierung & MLOps (Docker / Kubernetes)

Verbindet das RL-Projekt mit deiner aktuellen Docker/K8s-Weiterbildung – **RL + MLOps ist
eine seltene, gefragte Kombination** und macht das Portfolio rund.

**Ehrliche Einordnung:**
- **Docker** ist hier *echter* Mehrwert: reproduzierbare Umgebung (CUDA, Deps, Code), das
  Training läuft headless überall gleich (`train.py --no-render`). Klares Engineering-Signal.
- **Kubernetes** ist für eine einzelne Hobby-GPU technisch Overkill – aber als **Lern-Showcase
  legitim und beeindruckend**, wenn richtig gerahmt: Training als **Job**, Checkpoints/Logs auf
  einem **PersistentVolume**, später **Hyperparameter-Sweeps als parallele Jobs**. Genau diese
  Sweeps zahlen direkt auf Phase B (Tuning) ein.

**Starter-Artefakte sind bereits im Repo** (zum Studieren angelegt):
- `Dockerfile` – PyTorch-CUDA-Basis, OpenCV-System-Deps, Standard-CMD = headless Training.
- `.dockerignore` – hält Checkpoints/Logs/Git aus dem Image.
- `k8s/train-job.yaml` – kommentierter Job + PVC (GPU-Request, Checkpoints/Logs via `subPath`).

**Schritte:**
1. Image lokal bauen & testen: `docker build -t mario-rl .` → `docker run --gpus all mario-rl`
   (kurz mit `--episodes 50` testen).
2. Image in eine Registry pushen (z. B. GHCR), Image-Name in `k8s/train-job.yaml` anpassen.
3. Im Cluster: `kubectl apply -f k8s/train-job.yaml`, Logs mit `kubectl logs -f job/mario-train`.
4. **Ausbaustufe (zahlt auf Phase B ein):** Hyperparameter über Umgebungsvariablen in `config.py`
   überschreibbar machen (z. B. `LEARNING_RATE`, `EPSILON_DECAY`), dann mehrere Jobs mit
   unterschiedlichen Werten als **Sweep** starten und die `logs/`-CSVs vergleichen.

> Hinweis: GUI-Features (`play.py`, `visualize.py` mit Live-Fenster) laufen **nicht** im Container –
> die sind fürs lokale Zuschauen. Im Cluster läuft nur das headless Training. (Headless-GIF-Export
> wäre möglich, bräuchte aber eine kleine Anpassung in `visualize.py`: das `cv2.imshow` überspringen.)

---

## Phase E — Umstieg auf PPO ✅ (Migration + erstes Ziel erreicht)

> **Stand (05.07.2026): WELT 1 KOMPLETT — alle vier Level 20/20 greedy gelöst.**
> 1-1 (DQN), 1-2/1-3/1-4 (PPO, je 3M Steps, 16 Envs). Jedes Level hatte seine eigene Hürde:
> **1-2** Instabilität (Kollaps-Oszillation) → Fix VecNormalize + LR-Decay; **1-3** vorzeitige
> Konvergenz (1.4M Steps flach bei ~710, starb an derselben Lücke) → Fix ent_coef 0.01→0.03;
> **1-4** (Schloss) fiel mit dem Rezept direkt. Modelle: `checkpoints_ppo/mario_ppo_tuned.zip`
> (1-2), `mario_ppo_1-3_ent03.zip`, `mario_ppo_1-4.zip` (+ je `_vecnormalize.pkl`).
> Gleiche Determinismus-Fußnote wie bei 1-1. **Nächstes Ziel: Generalist-Modell**
> (ein PPO-Modell, zufällige Level pro Episode).

**Motivation:** DQN löst 1-1 zuverlässig, wird auf schwereren Leveln (1-2) aber zäh/instabil.
Genau da setzt **PPO** an: on-policy Policy-Gradient, sample-effizienter und stabiler,
Industriestandard für Pixel-RL. **Wichtig:** Das ist *kein* Modelltausch – die Lern-Engine
(`agent.py`: Replay, ε-Greedy, Target-Netz) wird ersetzt, nicht nur `model.py`.

**Weg: Stable-Baselines3 (SB3), nicht selbst implementieren.** To-dos:
1. **Env-API-Brücke (Hauptarbeit):** SB3 v2 will **Gymnasium** (5er-Tupel), unser Mario-Env ist
   altes **gym** (4er-Tupel). `nes-py`/`gym-super-mario-bros` sind auf altem gym gepinnt und
   **unmaintained** → es gibt keine Gymnasium-native Version. Lösung: **`shimmy`** (offizieller
   Farama-Adapter, ~2 Zeilen) hüllt das Legacy-Env in eine echte Gymnasium-Schnittstelle.
   Das ist ein dünner Adapter, **kein Port** – der reibungsloseste Weg.
   *Alternative B (größer):* Env-Quelle auf **Stable-Retro** (Gymnasium-nativ, Farama) wechseln –
   nur sinnvoll, wenn wir bewusst einen komplett modernen Stack / mehrere Spiele wollen
   (ROM-Import + Pipeline neu verifizieren).
2. **Vektorisierte Envs** (`SubprocVecEnv`): PPO lebt von parallelen Rollouts.
3. **Preprocessing:** `wrappers.py` weiternutzen oder SB3s `VecFrameStack` (84×84, 4 Frames).
4. **Policy:** SB3s `CnnPolicy` (≈ unser Nature-CNN) – kein eigenes Netz nötig.
5. **`train_ppo.py`** mit `PPO("CnnPolicy", env).learn(...)` + Callbacks (Eval/Checkpoints/Logging).
6. **Hyperparameter:** `n_steps`, `n_epochs`, `clip_range`, `gae_lambda`, `ent_coef` (nicht ε/Decay/Replay).
7. **Neue Deps:** `stable-baselines3`, `shimmy`, `gymnasium`.

**Wiederverwendbar:** Env-Erzeugung (+Adapter), Bildvorverarbeitung, Eval-Konzept (flag_get/x_pos),
Grad-CAM (auf PPO-CNN adaptierbar), Docker/K8s/CI. **Gradio-Demo** müsste ein SB3-Modell laden.

**Erstes PPO-Ziel:** 1-2 (das für DQN zu harte Level) – schlägt zwei Fliegen: härteres Level *und*
stärkerer Algorithmus.

---

## Phase F — Welt 2 & Hard Exploration ✅ (KOMPLETT, 07.07.2026)

> **Stand: Welt 2 komplett — alle vier Level 20/20 greedy.** Modelle:
> `mario_ppo_2-1_tower.zip` (+ Lösungssequenz `tower_seq_2-1.npz`, Seed 0),
> `mario_ppo_2-2.zip` (6M), `mario_ppo_2-3.zip` (3M), `mario_ppo_2-4.zip` (3M).

Der Trampolin-Turm am Levelende war ein echtes **Hard-Exploration-Problem**: Die Policy kam
zuverlässig hin (x 3026), fand den Superbounce aber in 12M Steps nie – Entropie-Boosts,
menschliche Demo + Behavior Cloning und ein Frame-Skip-Wechsel (4→2) halfen alle nicht.

**Was es gelöst hat** (neue, wiederverwendbare Pipeline für alle künftigen Hängestellen):
1. `goexplore.py` — Policy-Anlauf bis zur Hängestelle, **NES-Savestate** sichern, tausende
   zufällige Aktionssequenzen ab dem Savestate testen (`_restore` statt Neuanlauf).
   Fand die Flagge nach 81 Kandidaten.
2. `imitate.py bc-seq` — die gefundene Sequenz per Behavior Cloning in die Policy klonen
   (kein Distribution Shift, da der Anlauf ihr eigenes greedy-Verhalten ist), mit
   Greedy-Verifikation je Epoche.

Lehrstück nebenbei: Die 45-Varianten-Skriptprobe hatte „Skip-4 kann es physikalisch nicht"
nahegelegt — die breite Zufallssuche widerlegte das (zu enger Suchraum). `FRAME_SKIP` ist
seitdem trotzdem konfigurierbar (nützlich für spätere Präzisionsstellen).

**2-2 (Wasser-Level) ✅:** Standard-Rezept reichte – nach 3M Steps 0/20 (Gegner-Tod bei
x 2563, volle Restzeit), aber die Reward-Kurve stieg noch → Resume +3M statt Methodenwechsel
→ 20/20. Schwimm-Physik brauchte keinerlei Anpassung. Merkregel bestätigt: *Kurve steigt →
weitertrainieren; Kurve friert bei festem x ein → goexplore.*

**2-3 (Brücken) ✅** und **2-4 (Schloss) ✅:** Standard-Rezept, je 3M Steps, jeweils
erster Anlauf → 20/20 (x 3593 bzw. Axt hinter Bowser bei x 2267).

Bilanz Welt 2: **1× Spezialwerkzeug (2-1), 1× Geduld (2-2), 2× Rezept (2-3/2-4).**
Faustregel etabliert: *Kurve steigt → weitertrainieren; Kurve friert bei festem x ein →
goexplore.*

## Phase G — Welt 3 & Deployment (08.07.2026)

**WELT 3 KOMPLETT ✅:** alle vier Level 20/20, **jeweils im ersten Anlauf** (Standard-Rezept,
je 3M Steps) — die Pipeline ist eingespielt, ein neues Level kostet ~40 Min Training.
Zwischenstand: **12/12 angegangene Level gelöst.**
**Grad-CAM für PPO ✅:** `visualize_ppo.py` + DQN-vs-PPO-Vergleichs-GIF im README.
**HF-Space:** Build grün, App lief an; Fehlerkette komplett durchgestochen und dokumentiert
(packages.txt-Kommentare → demo-Global → sdk_version-Pin → HF-Scheduling → nes-py==8.2.1).
Letzter Stand: Episode-Button-Test nach nes-py-Pin steht aus.

---

## Phase H — Welt 4, Reward-Hacking & Multi-Level-Demo ✅ (KOMPLETT, 10.07.2026)

> **Stand: WELT 4 KOMPLETT — 4-4 gelöst (20/20 greedy, Axt bei x 2772).** Damit sind
> **alle 16 angegangenen Level gelöst (Welt 1–4)**. Modell:
> `checkpoints_ppo/mario_ppo_4-4_maze.zip`, Lösungssequenz `maze_seq_4-4.npz` (reproduzierbar).
> Der Weg: Reward-Fix (2 Exploits) → Explorationsdiagnose → **mehrstufiges Go-Explore**
> (`--head-seq`) → `bc-seq` mit Greedy-Verify je Epoche. Drei Lehren: (1) der 16-Bit-x-Glitch
> täuschte auch das *Erfolgskriterium der Suche* (→ Plausibilitäts-Deckel überall, wo x als
> Signal dient); (2) der erste Savestate lag 200 px hinter der Labyrinth-Weiche – „Suchraum
> zu eng" kann auch *räumlich* sein; (3) bei langen Zufalls-Tails zählt nicht die
> BC-Trefferquote, sondern der Greedy-Lauf (Flagge bei 97,2 % in Epoche 65).

**Welt 4: 4-1/4-2/4-3 je 20/20 im ersten Anlauf** (Standard-Rezept). **4-4 = Reward-Hacking-
Saga (läuft noch):** Der Standard-Δx-Reward lädt in dem Labyrinth zum **Farming** ein — der
Agent lief im Kreis und kassierte 13.824 Reward bei 0/20 Flagge (das Env setzt große
x-Rücksprünge auf 0, also kostet Im-Kreis-Laufen nichts). Fix: **`ProgressReward`-Wrapper**
(`wrappers.py`, opt-in via `PROGRESS_REWARD`), belohnt nur ein *neues* x-Maximum. Zweiter,
subtilerer Exploit: der Agent farmte einen **16-Bit-Overflow der x-Position** (`x_pos`=65535,
Emulator-Glitch → 57k Reward). Fix: nur plausible Schritte (`0 < Δx ≤ 64` px/Frame) zählen.
**Beide Fixes vorab am Exploit-Modell gemessen** (13.824→1.574 bzw. 64.735→221), bevor
Rechenzeit investiert wurde. **Eval-Ergebnis (10.07.):** Der 5M-Lauf (+ `ent_coef 0.03`)
lernte mit sauberem Reward ehrlich, fror aber ein — Greedy-Eval des 4.9M-Checkpoints
(`ppo_4-4_4900000_steps.zip`): **0/5, konstant x 1433**. Diagnose nach Faustregel eindeutig:
4-4 ist (nach dem Reward-Fix) ein **reines Explorationsproblem** → nächster Schritt
`goexplore.py` ab Savestate x ≈ 1400, wie bei 2-1 (Achtung Labyrinth: `--success-x` allein
reicht evtl. nicht, ggf. mehrstufig suchen).

**Live-Demo → Multi-Level, statisch (MP4):** Der HF-Space zeigt jetzt **alle 15 Level**
(Dropdown 1-1 DQN + 14 PPO, Grad-CAM-Checkbox). Weg dahin: erst als Live-Rechnung (torch/SB3
im Space) — aber Gratis-CPU zu langsam (~90 s/Lauf), OOM bei Mehrfachklicks, und die
Trajektorie **divergiert auf fremder Hardware** (Float-Rundung → argmax kippt). Lösung:
**vorberechnete Videos** (`gen_demos.py` erzeugt 30 MP4s + `results.json`; Space braucht nur
noch gradio). **GIF → MP4**, weil H.264 die bunten CAM-Overlays ~10× besser komprimiert
(groß, scharf, flüssig; 49 MB statt 75 MB GIF).

**INTERVIEW.md/PDF** um Einsteiger-Grundlagen + Reward-Hacking-Detailsektion erweitert.

**Nächste Schritte:**
1. ~~4-4 per `goexplore` knacken~~ ✅ **erledigt (10.07.)** – Welt 4 komplett, README-Story drin.
2. **K8s-Sweep real ausführen** (Docker/K8s-Lernthema, interview-stark) + Story-Write-up.
3. ~~Mensch-vs-KI-Levelwahl~~ ✅ **erledigt (11.07.)** – `record`/`compose` ohne Argumente
   zeigen ein Terminal-Menü (1-1…4-4) und wählen Modell + Dateinamen automatisch
   (Quelle: `app.py LEVELS`; `app.py` baut die Gradio-UI dafür nicht mehr auf Modulebene).
4. ~~HF-Space um 4-4 erweitern~~ ✅ (Sebastian hat MP4s+results.json hochgeladen) & Welt 5 optional.
5. Generalist nachschärfen / Curiosity-ICM bleibt Option für *flächiges* Explorations-Scheitern.
6. Geparkt: decay500k-Lauf + 100k-vs-500k-Plot.

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
