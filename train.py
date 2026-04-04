"""Hauptskript: Trainiert den Mario-Agenten mit Live-Anzeige."""

import sys
import time
import numpy as np
from pathlib import Path

import config
from wrappers import create_env
from agent import MarioAgent


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
    recent_rewards = []

    print("Training startet... Schließe das Spielfenster NICHT!")
    print("-" * 70)

    for episode in range(1, config.MAX_EPISODES + 1):
        state = env.reset()
        total_reward = 0.0
        last_loss = None
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
                last_loss = loss

            state = next_state
            total_reward += reward
            steps += 1

            if done:
                break

        # Statistiken
        elapsed = time.time() - start_time
        fps = (steps * 4) / elapsed if elapsed > 0 else 0  # *4 wegen SkipFrame
        x_pos = info.get("x_pos", 0)

        if total_reward > best_reward:
            best_reward = total_reward
        if x_pos > best_x_pos:
            best_x_pos = x_pos

        recent_rewards.append(total_reward)
        if len(recent_rewards) > 100:
            recent_rewards.pop(0)

        print_stats(
            episode, total_reward, x_pos, best_reward, best_x_pos,
            agent.epsilon, last_loss, fps
        )

        # Checkpoint speichern
        if episode % config.SAVE_INTERVAL == 0:
            agent.save()
            avg = np.mean(recent_rewards)
            print(f"  -> Checkpoint gespeichert! Durchschnitt letzte {len(recent_rewards)} Episoden: {avg:.1f}")

        # Level geschafft?
        if info.get("flag_get", False):
            print(f"\n  *** LEVEL GESCHAFFT in Episode {episode}! ***\n")
            agent.save()

    env.close()
    print("Training beendet!")


if __name__ == "__main__":
    train()
