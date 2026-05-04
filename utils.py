# utils.py
import gym
import torch
import numpy as np

def is_valid_map(desc: list[str]) -> bool:
    """
    Проверяет, существует ли путь от стартовой до целевой клетки на карте.
    """
    nrow = len(desc)
    ncol = len(desc[0])
    start = (0, 0)
    goal = (nrow - 1, ncol - 1)
    queue = [start]
    visited = {start}
    while queue:
        r, c = queue.pop(0)
        if (r, c) == goal:
            return True
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < nrow and 0 <= nc < ncol and (nr, nc) not in visited:
                if desc[nr][nc] in ['F', 'G']:
                    visited.add((nr, nc))
                    queue.append((nr, nc))
    return False

def generate_all_valid_maps(map_size: int = 4) -> list[list[str]]:
    """
    Генерирует все валидные карты для FrozenLake размером map_size x map_size.
    """
    valid_maps = []
    positions = [(r, c) for r in range(map_size) for c in range(map_size)
                 if not ((r == 0 and c == 0) or (r == map_size - 1 and c == map_size - 1))]
    total_combinations = 2 ** len(positions)
    for bits in range(total_combinations):
        grid = [['F' for _ in range(map_size)] for _ in range(map_size)]
        grid[0][0] = 'S'
        grid[map_size - 1][map_size - 1] = 'G'
        for i, (r, c) in enumerate(positions):
            grid[r][c] = 'F' if (bits >> i) & 1 else 'H'
        map_str = [''.join(row) for row in grid]
        if is_valid_map(map_str):
            valid_maps.append(map_str)
    return valid_maps

def test_on_map(model: torch.nn.Module, map_desc: list[str], device: torch.device, observation_wrapper) -> float:
    """
    Проверяет работу модели на одной карте и возвращает суммарную награду.
    Параметр observation_wrapper позволяет использовать любую обёртку наблюдений.
    """
    env = gym.make("FrozenLake-v1", desc=map_desc, is_slippery=False)
    env = observation_wrapper(env)
    obs, _ = env.reset()
    state = torch.from_numpy(obs).float().unsqueeze(0).to(device)
    total_reward = 0.0
    done = False
    while not done:
        with torch.no_grad():
            action = model(state).max(1)[1].view(1, 1)
        next_obs, reward, terminated, truncated, _ = env.step(action.item())
        done = terminated or truncated
        total_reward += reward
        if not done:
            state = torch.from_numpy(next_obs).float().unsqueeze(0).to(device)
    env.close()
    return total_reward