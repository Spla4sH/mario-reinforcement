"""Environment-Wrapper für Bildvorverarbeitung."""

import cv2
import numpy as np
import gym
from gym.spaces import Box
from collections import deque


class SkipFrame(gym.Wrapper):
    """Wiederholt eine Aktion für mehrere Frames und summiert die Belohnung."""

    def __init__(self, env, skip=4):
        super().__init__(env)
        self._skip = skip

    def step(self, action):
        total_reward = 0.0
        for _ in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            total_reward += reward
            if done:
                break
        return obs, total_reward, done, info


class GrayScaleResize(gym.ObservationWrapper):
    """Konvertiert zu Graustufen und skaliert auf 84x84 für das Netz."""

    def __init__(self, env, size=84):
        super().__init__(env)
        self._size = size
        self.observation_space = Box(
            low=0, high=255, shape=(size, size, 1), dtype=np.uint8
        )

    def observation(self, obs):
        gray = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        resized = cv2.resize(gray, (self._size, self._size), interpolation=cv2.INTER_AREA)
        return resized[:, :, np.newaxis]


class FrameStack(gym.Wrapper):
    """Stapelt die letzten N Frames als Eingabe für das Netz."""

    def __init__(self, env, num_stack=4):
        super().__init__(env)
        self._num_stack = num_stack
        self._frames = deque(maxlen=num_stack)
        low = np.repeat(env.observation_space.low, num_stack, axis=-1)
        high = np.repeat(env.observation_space.high, num_stack, axis=-1)
        self.observation_space = Box(low=low, high=high, dtype=np.uint8)

    def reset(self):
        obs = self.env.reset()
        for _ in range(self._num_stack):
            self._frames.append(obs)
        return self._get_obs()

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self._frames.append(obs)
        return self._get_obs(), reward, done, info

    def _get_obs(self):
        return np.concatenate(list(self._frames), axis=-1)


def create_env(world=1, stage=1, render=True):
    """Erstellt die Super Mario Umgebung mit allen Wrappern.

    Das Original-Spielbild wird im Fenster angezeigt (render=True),
    während das Netz intern verkleinerte Graustufen-Frames bekommt.
    """
    import gym_super_mario_bros
    from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
    from nes_py.wrappers import JoypadSpace

    # Suffix -v0 = Originalgrafik des NES-Spiels (ROM ist im Paket enthalten).
    env_id = f"SuperMarioBros-{world}-{stage}-v0"
    env = gym_super_mario_bros.make(env_id, apply_api_compatibility=False)

    # Vereinfachte Aktionen: SIMPLE_MOVEMENT hat 7 Aktionen
    # [['NOOP'], ['right'], ['right', 'A'], ['right', 'B'], ['right', 'A', 'B'], ['A'], ['left']]
    env = JoypadSpace(env, SIMPLE_MOVEMENT)

    # Wrapper anwenden
    env = SkipFrame(env, skip=4)
    env = GrayScaleResize(env, size=84)
    env = FrameStack(env, num_stack=4)

    return env
