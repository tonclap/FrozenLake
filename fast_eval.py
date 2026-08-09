# fast_eval.py
"""
Быстрая точная оценка greedy-политики на всех валидных картах.

Зачем: `utils.test_on_map` создаёт отдельное gym-окружение на каждую карту, то есть
3828 вызовов `gym.make` + пошаговый питоновский цикл. Это десятки секунд, из-за чего
полную оценку в `trainer.py` приходилось звать раз в 3000 эпизодов. Здесь все карты
прогоняются одновременно: наблюдения собираются векторно, сеть вызывается один раз
на шаг (батч = число ещё живых карт). Результат совпадает с gym-версией побитово
(проверяется `--verify`), но на два порядка быстрее — это и позволяет вести
приоритизированный curriculum по реально проваленным картам.

Про MAX_STEPS. Политика детерминирована, а состояние среды исчерпывается парой
(карта, позиция). Позиций всего map_size**2, поэтому если агент не дошёл до цели за
map_size**2 шагов, он обязан был повторить позицию — а значит зациклился навсегда.
Следовательно map_size**2 + 1 шагов достаточно, и ни один успех не отсекается.
Это строгая граница, а не эвристика вроде gym-ного max_episode_steps=100.
"""
from __future__ import annotations

import argparse
import random
import time

import numpy as np
import torch

from utils import generate_all_valid_maps

# Кодировка тайлов должна совпадать с AugmentedObservationWrapperCNN.observation
TILE_ENCODING = {"S": 0.0, "F": 1.0, "H": -1.0, "G": 2.0}
PASSABLE = ("S", "F", "G")
# Порядок действий gym FrozenLake: 0=LEFT, 1=DOWN, 2=RIGHT, 3=UP
ACTION_DELTA = np.array([[0, -1], [1, 0], [0, 1], [-1, 0]], dtype=np.int64)

_INF = np.float32(1e9)


def maps_to_grid(maps: list[list[str]]) -> np.ndarray:
    """Список карт -> массив символов формы (n, nrow, ncol)."""
    return np.array([[list(row) for row in m] for m in maps])


def bfs_distance_channel(grid: np.ndarray, map_size: int) -> np.ndarray:
    """
    Канал 2 наблюдения: нормированная BFS-дистанция до цели, векторно по всем картам.

    Повторяет семантику AugmentedObservationWrapperCNN: достижимые от цели клетки
    получают d / (nrow + ncol - 2) (значение может превышать 1.0 при длинном обходе),
    недостижимые и дыры остаются 1.0.
    """
    n = grid.shape[0]
    passable = np.isin(grid, PASSABLE)
    goal_r = goal_c = map_size - 1

    dist = np.full((n, map_size, map_size), _INF, dtype=np.float32)
    dist[:, goal_r, goal_c] = 0.0
    # Раскатка Беллмана-Форда: за один проход волна распространяется на 1 клетку,
    # максимальная дистанция на сетке < map_size**2, поэтому столько итераций и хватает.
    for _ in range(map_size * map_size):
        cand = dist.copy()
        cand[:, 1:, :] = np.minimum(cand[:, 1:, :], dist[:, :-1, :] + 1.0)
        cand[:, :-1, :] = np.minimum(cand[:, :-1, :], dist[:, 1:, :] + 1.0)
        cand[:, :, 1:] = np.minimum(cand[:, :, 1:], dist[:, :, :-1] + 1.0)
        cand[:, :, :-1] = np.minimum(cand[:, :, :-1], dist[:, :, 1:] + 1.0)
        cand[~passable] = _INF
        cand[:, goal_r, goal_c] = 0.0
        if np.array_equal(cand, dist):
            break
        dist = cand

    max_dist = float(2 * map_size - 2)
    return np.where(dist < _INF / 2, dist / max_dist, 1.0).astype(np.float32)


def encoded_map_channel(grid: np.ndarray) -> np.ndarray:
    """Канал 1 наблюдения: кодировка тайлов."""
    encoded = np.zeros(grid.shape, dtype=np.float32)
    for char, value in TILE_ENCODING.items():
        encoded[grid == char] = value
    return encoded


class MapBatch:
    """Предпосчитанные статические каналы наблюдений для набора карт."""

    def __init__(self, maps: list[list[str]], map_size: int = 4):
        self.maps = maps
        self.map_size = map_size
        grid = maps_to_grid(maps)
        self.grid = grid
        self.is_hole = grid == "H"
        self.is_goal = grid == "G"
        self.encoded = encoded_map_channel(grid)
        self.bfs = bfs_distance_channel(grid, map_size)
        self._device: torch.device | None = None
        self._encoded_t: torch.Tensor | None = None
        self._bfs_t: torch.Tensor | None = None

    def __len__(self) -> int:
        return len(self.maps)

    def tensors(self, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        """Ленивый перенос статических каналов на устройство (делается один раз)."""
        if self._device != device:
            self._encoded_t = torch.from_numpy(self.encoded).to(device)
            self._bfs_t = torch.from_numpy(self.bfs).to(device)
            self._device = device
        return self._encoded_t, self._bfs_t


@torch.no_grad()
def evaluate(model: torch.nn.Module, batch: MapBatch, device: torch.device) -> np.ndarray:
    """
    Прогоняет greedy-политику по всем картам сразу.

    Возвращает булев массив длины len(batch): True = агент дошёл до цели.
    """
    n = len(batch)
    size = batch.map_size
    encoded_t, bfs_t = batch.tensors(device)

    rows = np.zeros(n, dtype=np.int64)
    cols = np.zeros(n, dtype=np.int64)
    active = np.ones(n, dtype=bool)
    success = np.zeros(n, dtype=bool)

    was_training = model.training
    model.eval()
    for _ in range(size * size + 1):
        idx = np.flatnonzero(active)
        if idx.size == 0:
            break
        idx_t = torch.from_numpy(idx).to(device)
        ar = torch.arange(idx.size, device=device)

        one_hot = torch.zeros(idx.size, size, size, device=device)
        one_hot[ar, torch.from_numpy(rows[idx]).to(device), torch.from_numpy(cols[idx]).to(device)] = 1.0
        obs = torch.stack([one_hot, encoded_t[idx_t], bfs_t[idx_t]], dim=1)

        actions = model(obs).argmax(dim=1).cpu().numpy()
        new_r = np.clip(rows[idx] + ACTION_DELTA[actions, 0], 0, size - 1)
        new_c = np.clip(cols[idx] + ACTION_DELTA[actions, 1], 0, size - 1)
        rows[idx], cols[idx] = new_r, new_c

        hit_hole = batch.is_hole[idx, new_r, new_c]
        hit_goal = batch.is_goal[idx, new_r, new_c]
        success[idx[hit_goal]] = True
        active[idx[hit_hole | hit_goal]] = False

    if was_training:
        model.train()
    return success


_ALL_MAPS_CACHE: dict[int, MapBatch] = {}


def all_maps_batch(map_size: int = 4) -> MapBatch:
    """Кешированный MapBatch по всем валидным картам заданного размера."""
    if map_size not in _ALL_MAPS_CACHE:
        _ALL_MAPS_CACHE[map_size] = MapBatch(generate_all_valid_maps(map_size), map_size)
    return _ALL_MAPS_CACHE[map_size]


def evaluate_all(model: torch.nn.Module, device: torch.device, map_size: int = 4):
    """Оценка на всех валидных картах. Возвращает (успехов, всего, %, маска успехов)."""
    batch = all_maps_batch(map_size)
    success = evaluate(model, batch, device)
    n_ok = int(success.sum())
    total = len(batch)
    return n_ok, total, n_ok / total * 100.0, success


def _verify_against_gym(model, device, map_size, sample_size, seed=0):
    """Сверка с эталонной gym-реализацией на случайной выборке карт."""
    from dqn_model import AugmentedObservationWrapperCNN
    from utils import test_on_map

    maps = generate_all_valid_maps(map_size)
    rng = random.Random(seed)
    sample = maps if sample_size >= len(maps) else rng.sample(maps, sample_size)

    fast = evaluate(model, MapBatch(sample, map_size), device)
    mismatches = []
    t0 = time.perf_counter()
    for i, desc in enumerate(sample):
        gym_ok = test_on_map(model, desc, device, AugmentedObservationWrapperCNN) >= 1.0
        if gym_ok != bool(fast[i]):
            mismatches.append((i, desc, gym_ok, bool(fast[i])))
    gym_time = time.perf_counter() - t0

    print(f"Сверка с gym на {len(sample)} картах: gym занял {gym_time:.1f} c")
    if mismatches:
        print(f"РАСХОЖДЕНИЙ: {len(mismatches)}")
        for i, desc, gym_ok, fast_ok in mismatches[:5]:
            print(f"  карта #{i}: gym={gym_ok} fast={fast_ok} {desc}")
        return False
    print("Расхождений нет — быстрый оценщик эквивалентен gym-версии.")
    return True


def main():
    parser = argparse.ArgumentParser(description="Быстрая оценка чекпоинта на всех валидных картах")
    parser.add_argument("checkpoint", help="Путь к .pth")
    parser.add_argument("--map-size", type=int, default=4)
    parser.add_argument("--verify", type=int, default=0,
                        help="Сверить с gym-реализацией на N случайных картах (0 = не сверять)")
    parser.add_argument("--failed-out", default=None, help="Куда сохранить JSON проваленных карт")
    args = parser.parse_args()

    from dqn_model import DQNCNN

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DQNCNN(args.map_size, 4).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    state = ckpt["policy_net_state_dict"] if "policy_net_state_dict" in ckpt else ckpt
    model.load_state_dict(state)
    model.eval()

    t0 = time.perf_counter()
    n_ok, total, rate, success = evaluate_all(model, device, args.map_size)
    elapsed = time.perf_counter() - t0
    print(f"{args.checkpoint}: {n_ok}/{total} ({rate:.2f}%) за {elapsed:.2f} c")

    if args.failed_out:
        import json
        maps = generate_all_valid_maps(args.map_size)
        failed = [maps[i] for i in np.flatnonzero(~success)]
        with open(args.failed_out, "w", encoding="utf-8") as f:
            json.dump(failed, f, ensure_ascii=False, indent=4)
        print(f"Проваленные карты ({len(failed)}) сохранены в {args.failed_out}")

    if args.verify:
        _verify_against_gym(model, device, args.map_size, args.verify)


if __name__ == "__main__":
    main()
