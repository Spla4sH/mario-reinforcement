"""Hauptskript: Trainiert den Mario-Agenten mit Live-Anzeige."""

import argparse
import time
from pathlib import Path

import numpy as np

import config
from agent import MarioAgent
from evaluate import evaluate
from metrics import MetricsLogger
from plot import generate_plots
from wrappers import create_env


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

    # Umgebung erstellen
    print(f"Erstelle Umgebung: World {config.WORLD}-{config.STAGE}...")
    env = create_env(
        world=config.WORLD,
        stage=config.STAGE,
        render=config.RENDER,
    )
    num_actions = env.action_space.n
    print(f"Aktionen: {num_actions}")
    print()

    # Agent erstellen
    agent = MarioAgent(num_actions)

    # Checkpoint laden falls vorhanden
    checkpoint_path = Path("checkpoints/mario_agent.pt")
    if checkpoint_path.exists():
        agent.load(str(checkpoint_path))
        print("Vorheriger Checkpoint geladen!")
    print()

    # Tracking
    best_reward = -float("inf")
    best_x_pos = 0
    best_eval = (0.0, 0.0)  # (flag_rate, mean_x) des bisher besten Modells
    recent_rewards = []
    logger = MetricsLogger()
    print(f"Metriken werden geloggt nach: {logger.csv_path}")

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

            # Checkpoint speichern + Graphen aktualisieren
            if episode % config.SAVE_INTERVAL == 0:
                agent.save()
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
                if (ev["flag_rate"], ev["mean_x"]) > best_eval:
                    best_eval = (ev["flag_rate"], ev["mean_x"])
                    agent.save(name="mario_best.pt")
                    print("  -> Neues bestes Modell gespeichert (mario_best.pt)")

            # Level geschafft?
            if flag_get:
                print(f"\n  *** LEVEL GESCHAFFT in Episode {episode}! ***\n")
                agent.save()
    except KeyboardInterrupt:
        print("\nTraining abgebrochen – speichere Checkpoint & Graphen...")
    finally:
        agent.save()
        generate_plots(logger.csv_path)
        logger.close()
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
