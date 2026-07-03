"""Hauptskript: Trainiert den Mario-Agenten mit Live-Anzeige."""

import argparse
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

import config
from agent import MarioAgent
from evaluate import evaluate
from metrics import MetricsLogger
from plot import generate_plots
from record import record_frames, save_gif
from tracking import Tracker
from wrappers import create_env

# Ausgabe auf UTF-8 zwingen: Windows schreibt beim Umleiten in eine Datei sonst
# in cp1252 – Umlaute/ß/Ø erscheinen dann in UTF-8-Editoren als �. Best effort.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass


def set_seed(seed):
    """Setzt alle Zufallsquellen für reproduzierbare Läufe."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# Hyperparameter, die ausgegeben und (falls aktiv) an W&B übergeben werden
_CONFIG_KEYS = [
    "WORLD", "STAGE", "SEED", "LEARNING_RATE", "GAMMA", "BATCH_SIZE",
    "MEMORY_SIZE", "EPSILON_DECAY", "TARGET_UPDATE",
    "REWARD_SCALE", "REWARD_CLIP", "EVAL_INTERVAL", "EVAL_EPISODES",
]


def print_config():
    """Gibt die effektiven Hyperparameter aus (landet so auch im Log/stdout)."""
    print("Konfiguration:")
    for key in _CONFIG_KEYS:
        print(f"  {key:14s} = {getattr(config, key)}")
    print()


def print_stats(episode, reward, x_pos, best_reward, best_x, epsilon, loss, fps):
    """Gibt Trainingsstatistiken aus."""
    loss_str = f"{loss:.4f}" if loss else "---"
    print(
        f"Episode {episode:5d} | "
        f"Belohnung: {reward:7.1f} | "
        f"Position: {x_pos:5d} | "
        f"Best Reward: {best_reward:7.1f} | "
        f"Best X: {best_x:5d} | "
        f"Epsilon: {epsilon:.3f} | "
        f"Loss: {loss_str} | "
        f"FPS: {fps:.0f}"
    )


def train():
    """Trainingsschleife mit Live-Rendering."""
    print("=" * 70)
    print("  SUPER MARIO BROS - Reinforcement Learning")
    print("  Double DQN mit Live-Anzeige")
    print("=" * 70)
    print()

    set_seed(config.SEED)
    print_config()

    # Umgebung erstellen
    print(f"Erstelle Umgebung: World {config.WORLD}-{config.STAGE}...")
    env = create_env(
        world=config.WORLD,
        stage=config.STAGE,
        render=config.RENDER,
    )
    # Env-Zufall ebenfalls seeden (best effort, API je nach gym-Version)
    try:
        env.seed(config.SEED)
    except (AttributeError, TypeError):
        pass
    try:
        env.action_space.seed(config.SEED)
    except (AttributeError, TypeError):
        pass
    num_actions = env.action_space.n
    print(f"Aktionen: {num_actions}")
    print()

    # Agent erstellen
    agent = MarioAgent(num_actions)

    # Checkpoint laden falls vorhanden – sonst optional Warm-Start (Curriculum)
    checkpoint_path = Path(config.CHECKPOINT_DIR) / "mario_agent.pt"
    if checkpoint_path.exists():
        agent.load(str(checkpoint_path))
        print("Vorheriger Checkpoint geladen!")
    elif config.INIT_FROM:
        agent.load_weights(config.INIT_FROM)
        print(f"Warm-Start aus {config.INIT_FROM} – frische Exploration auf neuem Level.")
    print()

    # Tracking
    best_reward = -float("inf")
    best_x_pos = 0
    best_eval = (0.0, 0.0)  # (flag_rate, mean_x) des bisher besten Modells
    recent_rewards = []
    logger = MetricsLogger()
    print(f"Metriken werden geloggt nach: {logger.csv_path}")

    # Optionales W&B-Tracking (ausfallsicher; aktivieren mit USE_WANDB=1)
    tracker = Tracker(
        config.USE_WANDB,
        config_dict={k: getattr(config, k) for k in _CONFIG_KEYS},
        project=config.WANDB_PROJECT,
        run_name=logger.run_name,
    )

    print("Training startet... Schließe das Spielfenster NICHT!")
    print("(Mit Strg+C abbrechen – Checkpoint & Graphen werden noch gespeichert.)")
    print("-" * 70)

    try:
        for episode in range(1, config.MAX_EPISODES + 1):
            state = env.reset()
            total_reward = 0.0
            loss_sum = 0.0
            loss_count = 0
            steps = 0
            start_time = time.time()

            while True:
                # Live-Rendering des Original-Spiels
                if config.RENDER:
                    env.render()

                # Aktion wählen und ausführen
                action = agent.select_action(state)
                next_state, reward, done, info = env.step(action)

                # Erfahrung speichern
                agent.memory.push(state, action, reward, next_state, done)

                # Lernen
                loss = agent.learn()
                if loss is not None:
                    loss_sum += loss
                    loss_count += 1

                state = next_state
                total_reward += reward
                steps += 1

                if done:
                    break

            # Statistiken
            elapsed = time.time() - start_time
            fps = (steps * 4) / elapsed if elapsed > 0 else 0  # *4 wegen SkipFrame
            x_pos = info.get("x_pos", 0)
            avg_loss = loss_sum / loss_count if loss_count > 0 else None
            flag_get = info.get("flag_get", False)

            if total_reward > best_reward:
                best_reward = total_reward
            if x_pos > best_x_pos:
                best_x_pos = x_pos

            recent_rewards.append(total_reward)
            if len(recent_rewards) > 100:
                recent_rewards.pop(0)

            print_stats(
                episode, total_reward, x_pos, best_reward, best_x_pos,
                agent.epsilon, avg_loss, fps
            )
            logger.log_episode(
                episode, total_reward, x_pos, steps,
                agent.epsilon, avg_loss, fps, flag_get
            )
            ep_metrics = {
                "reward": total_reward,
                "x_pos": x_pos,
                "steps": steps,
                "epsilon": agent.epsilon,
                "fps": fps,
                "flag_get": int(bool(flag_get)),
            }
            if avg_loss is not None:
                ep_metrics["loss"] = avg_loss
            tracker.log(ep_metrics, step=episode)

            # Checkpoint speichern + Graphen aktualisieren
            if episode % config.SAVE_INTERVAL == 0:
                agent.save(path=config.CHECKPOINT_DIR)
                generate_plots(logger.csv_path)
                avg = np.mean(recent_rewards)
                print(f"  -> Checkpoint & Graphen gespeichert! Durchschnitt letzte {len(recent_rewards)} Episoden: {avg:.1f}")

            # Greedy-Evaluation: echter Fortschritt ohne Exploration -> bestes Modell sichern
            if episode % config.EVAL_INTERVAL == 0:
                ev = evaluate(agent, env, config.EVAL_EPISODES)
                print(
                    f"  [Eval] Flagge: {ev['flag_rate'] * 100:.0f}% | "
                    f"Ø x: {ev['mean_x']:.0f} | max x: {ev['max_x']} | "
                    f"Ø Reward: {ev['mean_reward']:.1f}"
                )
                tracker.log(
                    {
                        "eval/flag_rate": ev["flag_rate"],
                        "eval/mean_x": ev["mean_x"],
                        "eval/max_x": ev["max_x"],
                        "eval/mean_reward": ev["mean_reward"],
                    },
                    step=episode,
                )
                if (ev["flag_rate"], ev["mean_x"]) > best_eval:
                    best_eval = (ev["flag_rate"], ev["mean_x"])
                    agent.save(path=config.CHECKPOINT_DIR, name="mario_best.pt")
                    print("  -> Neues bestes Modell gespeichert (mario_best.pt)")
                    # Auto-Highlight: beste Episode als GIF festhalten (ausfallsicher)
                    try:
                        frames, rec_info = record_frames(agent, env)
                        Path(config.HIGHLIGHT_DIR).mkdir(exist_ok=True)
                        if save_gif(frames, f"{config.HIGHLIGHT_DIR}/best_run.gif"):
                            print(f"  -> Highlight-GIF aktualisiert (x_pos {rec_info.get('x_pos', 0)})")
                    except Exception as exc:
                        print(f"  [Highlight] Aufnahme übersprungen ({exc})")

            # Level geschafft?
            if flag_get:
                print(f"\n  *** LEVEL GESCHAFFT in Episode {episode}! ***\n")
                agent.save(path=config.CHECKPOINT_DIR)
    except KeyboardInterrupt:
        print("\nTraining abgebrochen – speichere Checkpoint & Graphen...")
    finally:
        agent.save(path=config.CHECKPOINT_DIR)
        generate_plots(logger.csv_path)
        logger.close()
        tracker.finish()
        env.close()

    print("Training beendet!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Mario-Agent trainieren.")
    parser.add_argument(
        "--episodes",
        type=int,
        default=None,
        help="Anzahl Episoden (überschreibt config.MAX_EPISODES, z. B. für einen kurzen Probelauf)",
    )
    parser.add_argument(
        "--no-render",
        action="store_true",
        help="Ohne Live-Fenster trainieren (schneller, weniger Ablenkung)",
    )
    args = parser.parse_args()

    if args.episodes is not None:
        config.MAX_EPISODES = args.episodes
    if args.no_render:
        config.RENDER = False

    train()
