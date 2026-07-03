"""Double DQN Agent mit Experience Replay."""

import random
import numpy as np
import torch
import torch.nn.functional as F
from collections import deque
from pathlib import Path

from model import MarioNet
import config


class ReplayMemory:
    """Speichert Erfahrungen (state, action, reward, next_state, done)."""

    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states),
            np.array(actions),
            np.array(rewards, dtype=np.float32),
            np.array(next_states),
            np.array(dones, dtype=np.float32),
        )

    def __len__(self):
        return len(self.buffer)


class MarioAgent:
    """Double DQN Agent der Super Mario Bros spielen lernt."""

    def __init__(self, num_actions, device=None):
        self.num_actions = num_actions
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        print(f"Verwende: {self.device}")

        # Online- und Target-Netzwerk
        self.online_net = MarioNet(config.FRAME_STACK, num_actions).to(self.device)
        self.target_net = MarioNet(config.FRAME_STACK, num_actions).to(self.device)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()

        self.optimizer = torch.optim.Adam(
            self.online_net.parameters(), lr=config.LEARNING_RATE
        )
        self.memory = ReplayMemory(config.MEMORY_SIZE)

        # Exploration
        self.epsilon = config.EPSILON_START
        self.step_count = 0

    def act(self, state):
        """Greedy-Aktion: bestes Q-Wert-Argmax ohne Exploration/Seiteneffekte.

        Geeignet für Evaluation/Wiedergabe (siehe play.py) – verändert weder
        Epsilon noch step_count.
        """
        # State vorbereiten: (H, W, C) -> (1, C, H, W)
        state_t = (
            torch.tensor(state, dtype=torch.uint8)
            .permute(2, 0, 1)
            .unsqueeze(0)
            .to(self.device)
        )
        with torch.no_grad():
            q_values = self.online_net(state_t)
        return q_values.argmax(dim=1).item()

    def select_action(self, state):
        """Wählt eine Aktion mit Epsilon-Greedy-Strategie (fürs Training)."""
        # Epsilon linear abbauen
        self.epsilon = max(
            config.EPSILON_END,
            config.EPSILON_START
            - (config.EPSILON_START - config.EPSILON_END)
            * self.step_count
            / config.EPSILON_DECAY,
        )
        self.step_count += 1

        if random.random() < self.epsilon:
            return random.randrange(self.num_actions)

        return self.act(state)

    def learn(self):
        """Führt einen Trainingsschritt mit Double DQN durch."""
        if len(self.memory) < config.MIN_MEMORY:
            return None

        states, actions, rewards, next_states, dones = self.memory.sample(
            config.BATCH_SIZE
        )

        # Tensoren erstellen: (batch, H, W, C) -> (batch, C, H, W)
        states_t = (
            torch.tensor(states, dtype=torch.uint8)
            .permute(0, 3, 1, 2)
            .to(self.device)
        )
        next_states_t = (
            torch.tensor(next_states, dtype=torch.uint8)
            .permute(0, 3, 1, 2)
            .to(self.device)
        )
        actions_t = torch.tensor(actions, dtype=torch.long).to(self.device)
        rewards_t = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        dones_t = torch.tensor(dones, dtype=torch.float32).to(self.device)

        # Q-Werte für gewählte Aktionen
        q_values = self.online_net(states_t).gather(1, actions_t.unsqueeze(1)).squeeze()

        # Double DQN: Online-Netz wählt Aktion, Target-Netz bewertet sie
        with torch.no_grad():
            best_actions = self.online_net(next_states_t).argmax(dim=1)
            next_q_values = (
                self.target_net(next_states_t)
                .gather(1, best_actions.unsqueeze(1))
                .squeeze()
            )
            targets = rewards_t + config.GAMMA * next_q_values * (1 - dones_t)

        loss = F.smooth_l1_loss(q_values, targets)

        self.optimizer.zero_grad()
        loss.backward()
        # Gradient Clipping für Stabilität
        torch.nn.utils.clip_grad_norm_(self.online_net.parameters(), 10.0)
        self.optimizer.step()

        # Target-Netz aktualisieren
        if self.step_count % config.TARGET_UPDATE == 0:
            self.target_net.load_state_dict(self.online_net.state_dict())

        return loss.item()

    def save(self, path="checkpoints", name="mario_agent.pt"):
        """Speichert den Agenten unter ``path/name``."""
        save_dir = Path(path)
        save_dir.mkdir(exist_ok=True)
        torch.save(
            {
                "online_net": self.online_net.state_dict(),
                "target_net": self.target_net.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "epsilon": self.epsilon,
                "step_count": self.step_count,
            },
            save_dir / name,
        )

    def load(self, path="checkpoints/mario_agent.pt"):
        """Lädt einen gespeicherten Agenten."""
        checkpoint = torch.load(path, map_location=self.device)
        self.online_net.load_state_dict(checkpoint["online_net"])
        self.target_net.load_state_dict(checkpoint["target_net"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.epsilon = checkpoint["epsilon"]
        self.step_count = checkpoint["step_count"]
        print(f"Agent geladen (Step {self.step_count}, Epsilon {self.epsilon:.3f})")

    def load_weights(self, path):
        """Übernimmt nur die Netzgewichte (Warm-Start / Transfer fürs Curriculum).

        Anders als ``load`` bleiben Epsilon, Step-Count und Optimizer unangetastet –
        der Agent startet also mit dem gelernten Feature-Extraktor eines Levels,
        aber **frischer Exploration** auf dem neuen Level.
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.online_net.load_state_dict(checkpoint["online_net"])
        self.target_net.load_state_dict(checkpoint["target_net"])
        print(
            f"Warm-Start: Netzgewichte aus {path} übernommen "
            f"(Epsilon {self.epsilon:.3f}, Step {self.step_count})"
        )
