from collections import deque

import gym
import numpy as np
import torch
import torch.nn as nn

class SELayer(nn.Module):
    """Блок Squeeze-and-Excitation для внимания по каналам."""
    def __init__(self, channel, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # Глобальное усреднение по пространству
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class AugmentedObservationWrapperCNN(gym.ObservationWrapper):
    """
    Обёртка для формирования наблюдения в виде трёхканального изображения:
      - Канал 0: One-hot представление позиции агента
      - Канал 1: Кодированная карта (S → 0.0, F → 1.0, H → -1.0, G → 2.0)
      - Канал 2: Нормированная BFS-дистанция до цели (по реальным проходимым ячейкам)
    """
    def __init__(self, env):
        super().__init__(env)
        self.map = env.unwrapped.desc.astype(str)
        self.nrow, self.ncol = self.map.shape
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(3, self.nrow, self.ncol), dtype=np.float32
        )
        # Предвычисление BFS-расстояния от каждой ячейки до цели (по проходимым клеткам).
        # Дыры получают значение 1.0 (максимальное), свободные — нормированное число шагов BFS.
        max_dist = float(self.nrow + self.ncol - 2)
        self.bfs_map = np.ones((self.nrow, self.ncol), dtype=np.float32)
        goal_r, goal_c = self.nrow - 1, self.ncol - 1
        queue = deque([(goal_r, goal_c, 0)])
        visited_bfs = {(goal_r, goal_c)}
        while queue:
            r, c, d = queue.popleft()
            self.bfs_map[r, c] = d / max_dist
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < self.nrow and 0 <= nc < self.ncol and (nr, nc) not in visited_bfs:
                    if self.map[nr][nc] in ('S', 'F', 'G'):
                        visited_bfs.add((nr, nc))
                        queue.append((nr, nc, d + 1))

    def observation(self, obs):
        one_hot = np.zeros((self.nrow, self.ncol), dtype=np.float32)
        pos = np.unravel_index(obs, (self.nrow, self.ncol))
        one_hot[pos] = 1.0
        mapping = {'S': 0.0, 'F': 1.0, 'H': -1.0, 'G': 2.0}
        encoded_map = np.array([[mapping.get(cell, 0.0) for cell in row] for row in self.map], dtype=np.float32)
        observation = np.stack([one_hot, encoded_map, self.bfs_map], axis=0)
        return observation

class DQNCNN(nn.Module):
    """
    CNN-модель для DQN с механизмом внимания (SE-layer).
    Состоит из двух свёрточных слоёв, SE-блока, полносвязного слоя и выходного слоя.
    """
    def __init__(self, map_size, n_actions):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=2)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=2)
        self.se = SELayer(channel=64, reduction=16)
        self.fc1 = nn.Linear(64 * 2 * 2, 256)
        self.fc2 = nn.Linear(256, n_actions)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = self.se(x)
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        return self.fc2(x)