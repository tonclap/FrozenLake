# curriculum_trainer.py
"""
Дообучение DQN до 100% через приоритизированный curriculum по перечисленным картам.

Диагноз, из которого выросла эта реализация
-------------------------------------------
Базовый `trainer.py` сэмплирует карты через `generate_random_map(size, p)`, где p —
вероятность клетки быть замёрзшей. Оценка же идёт по всем 3828 валидным картам
равномерно. Провальные карты (плотность дыр 28-50%, кратчайший путь ровно 6) при
p из [0.78, 0.82] выпадают с околонулевой вероятностью — то есть модель почти не
видит при обучении ровно тот класс карт, на котором её потом проверяют. Это
рассогласование train/eval-распределений, а не предел ёмкости сети.

Здесь распределение обучения совмещено с распределением оценки: карты берутся из
того же перечисления `generate_all_valid_maps`, а внутри него приоритет отдаётся
тем, что реально проваливаются на полной оценке.

Три отличия от прошлой (провалившейся) попытки fine-tune
--------------------------------------------------------
1. Прошлый `finetune.py` учился ТОЛЬКО на 8 провальных картах и обрушил общий
   результат до 73% (см. results/finetune.log). Здесь трудные карты всегда идут
   в смеси с равномерной выборкой из всех карт — доля задаётся HARD_FRACTION.
2. Пул трудных карт держится по EMA частоты провала, а не по последнему замеру.
   Карта, которую только что починили, ещё какое-то время остаётся в пуле и не
   успевает забыться.
3. Полная оценка (все 3828 карт) идёт часто и по ней ведётся лучший снапшот.
   При катастрофическом падении веса откатываются к лучшему снапшоту.

Reward shaping переведён на BFS-дистанцию. В `trainer.py` наблюдение уже содержит
BFS-канал, но shaping считался по манхэттенскому расстоянию — на картах с
единственным узким проходом это прямо вводило в заблуждение (шаг «ближе по
Манхэттену» часто означает шаг в тупик или в дыру).
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import random
import time

import numpy as np
import torch
import torch.optim as optim

import fast_eval
from dqn_model import DQNCNN
from fast_eval import MapBatch, all_maps_batch, evaluate
from trainer import ReplayMemory, optimize_model
from utils import generate_all_valid_maps, set_seed

ACTION_DELTA = fast_eval.ACTION_DELTA


# --------------------------------------------------------------------------------------
# Среда без gym: те же правила, но без gym.make на каждый эпизод
# --------------------------------------------------------------------------------------
class FastFrozenLake:
    """
    Детерминированный FrozenLake поверх предпосчитанных каналов MapBatch.

    Семантика совпадает с gym.make("FrozenLake-v1", is_slippery=False) + обёрткой
    AugmentedObservationWrapperCNN: награда 1.0 при достижении цели, эпизод
    завершается на дыре или цели, выход за границу оставляет агента на месте.
    Создание gym-окружения на каждый эпизод было заметной долей времени шага,
    здесь его нет.
    """

    def __init__(self, batch: MapBatch, max_steps: int = 100):
        self.batch = batch
        self.size = batch.map_size
        self.max_steps = max_steps
        self.map_idx = 0
        self.row = self.col = 0
        self.steps = 0

    def reset(self, map_idx: int) -> np.ndarray:
        self.map_idx = map_idx
        self.row = self.col = 0
        self.steps = 0
        return self._obs()

    def _obs(self) -> np.ndarray:
        one_hot = np.zeros((self.size, self.size), dtype=np.float32)
        one_hot[self.row, self.col] = 1.0
        return np.stack(
            [one_hot, self.batch.encoded[self.map_idx], self.batch.bfs[self.map_idx]], axis=0
        )

    def bfs_distance(self) -> float:
        """Настоящая длина пути до цели из текущей клетки (в шагах)."""
        return float(self.batch.bfs[self.map_idx, self.row, self.col]) * (2 * self.size - 2)

    def manhattan_distance(self) -> float:
        """Геометрическое расстояние до цели — шейпинг, под который обучались старые чекпоинты."""
        return float((self.size - 1 - self.row) + (self.size - 1 - self.col))

    def step(self, action: int):
        dr, dc = ACTION_DELTA[action]
        self.row = int(np.clip(self.row + dr, 0, self.size - 1))
        self.col = int(np.clip(self.col + dc, 0, self.size - 1))
        self.steps += 1

        terminated = False
        reward = 0.0
        if self.batch.is_hole[self.map_idx, self.row, self.col]:
            terminated = True
        elif self.batch.is_goal[self.map_idx, self.row, self.col]:
            terminated = True
            reward = 1.0
        truncated = (not terminated) and self.steps >= self.max_steps
        return self._obs(), reward, terminated, truncated


# --------------------------------------------------------------------------------------
# Приоритизированный сэмплер карт
# --------------------------------------------------------------------------------------
class MapSampler:
    """
    Смешанная выборка: часть эпизодов из «трудного» пула, остальное равномерно.

    Трудный пул определяется по EMA частоты провала каждой карты, а не по последней
    оценке. Это и удерживает недавно починенные карты под наблюдением, и не даёт
    пулу схлопнуться в ноль сразу после успешной оценки.
    """

    def __init__(self, n_maps: int, hard_fraction: float, ema_decay: float, hard_threshold: float):
        self.n_maps = n_maps
        self.hard_fraction = hard_fraction
        self.ema_decay = ema_decay
        self.hard_threshold = hard_threshold
        self.fail_ema = np.zeros(n_maps, dtype=np.float32)
        self.hard_pool = np.array([], dtype=np.int64)

    def update(self, success_mask: np.ndarray) -> None:
        failed = (~success_mask).astype(np.float32)
        self.fail_ema = self.ema_decay * self.fail_ema + (1.0 - self.ema_decay) * failed
        self.hard_pool = np.flatnonzero(self.fail_ema > self.hard_threshold)

    def sample(self) -> int:
        if self.hard_pool.size and random.random() < self.hard_fraction:
            return int(self.hard_pool[random.randrange(self.hard_pool.size)])
        return random.randrange(self.n_maps)

    def state_dict(self) -> dict:
        return {"fail_ema": self.fail_ema.tolist()}

    def load_state_dict(self, state: dict) -> None:
        self.fail_ema = np.array(state["fail_ema"], dtype=np.float32)
        self.hard_pool = np.flatnonzero(self.fail_ema > self.hard_threshold)


# --------------------------------------------------------------------------------------
# Конфигурация
# --------------------------------------------------------------------------------------
# За основу взяты гиперпараметры лучшего прогона (эксп. 1, 99.79%) из results/results.txt
DEFAULT_CONFIG = {
    "MAP_SIZE": 4,
    "BATCH_SIZE": 256,
    "REPLAY_CAPACITY": 55000,
    "N_STEP": 3,
    "GAMMA": 0.95,
    "LR": 1.5e-4,
    "LR_DECAY": 0.99995,
    "SOFT_TAU": 0.00813,
    "PLANNING_STEPS": 10,
    "EPS_START": 1.0,
    "EPS_END": 0.05,
    "EPS_DECAY": 0.99988,
    # shaping: "bfs" — реальная длина пути (лучше на узких коридорах),
    # "manhattan" — геометрия, под которую обучены существующие чекпоинты
    "SHAPING": "bfs",
    "STEP_PENALTY": 0.01115,
    "IMPROVEMENT_FACTOR": 0.05,
    "REVISIT_PENALTY": 0.053,
    # curriculum
    "HARD_FRACTION": 0.45,
    "EMA_DECAY": 0.85,
    "HARD_THRESHOLD": 0.02,
    "MAX_EPISODE_STEPS": 100,
}


def build_logger(log_path: str) -> None:
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_path, encoding="utf-8"), logging.StreamHandler()],
        force=True,
    )


def run_episode(env, sampler, policy_net, target_net, memory, optimizer, config, device, eps,
                do_optimize=True):
    """Один эпизод: сбор переходов, n-step, планирование. Возвращает (успех, награда)."""
    # Шейпинг должен совпадать с тем, под который обучался чекпоинт: Q-значения кодируют
    # возвраты именно этой функции награды, и её смена обесценивает выученное.
    distance = env.bfs_distance if config["SHAPING"] == "bfs" else env.manhattan_distance

    map_idx = sampler.sample()
    obs = env.reset(map_idx)
    state = torch.from_numpy(obs).unsqueeze(0).to(device)

    n_step_buffer = []
    planning_states = []
    visited = {(env.row, env.col)}
    prev_distance = distance()
    total_reward = 0.0
    success = 0

    while True:
        planning_states.append(state)
        if random.random() < eps:
            action = random.randrange(4)
        else:
            with torch.no_grad():
                action = int(policy_net(state).argmax(dim=1).item())

        next_obs, reward, terminated, truncated = env.step(action)
        done_flag = terminated or truncated
        if reward == 1.0:
            success = 1

        curr_pos = (env.row, env.col)
        curr_distance = distance()
        # shaping по BFS: «улучшение» = реальное сокращение пути, а не геометрия
        improvement = config["IMPROVEMENT_FACTOR"] * max(prev_distance - curr_distance, 0.0)
        revisit = config["REVISIT_PENALTY"] if curr_pos in visited else 0.0
        step_reward = reward + improvement - revisit - config["STEP_PENALTY"]

        visited.add(curr_pos)
        prev_distance = curr_distance
        total_reward += step_reward

        next_state = (
            torch.zeros_like(state)
            if done_flag
            else torch.from_numpy(next_obs).unsqueeze(0).to(device)
        )
        action_t = torch.tensor([[action]], device=device, dtype=torch.long)
        reward_t = torch.tensor([step_reward], device=device, dtype=torch.float32)
        n_step_buffer.append((state, action_t, reward_t, next_state, done_flag))

        if len(n_step_buffer) >= config["N_STEP"]:
            memory.push(_n_step_transition(n_step_buffer[: config["N_STEP"]], config["GAMMA"], device))
            n_step_buffer.pop(0)

        if do_optimize:
            optimize_model(policy_net, target_net, memory, optimizer, config)

        if done_flag:
            break
        state = next_state

    while n_step_buffer:
        memory.push(_n_step_transition(n_step_buffer, config["GAMMA"], device))
        n_step_buffer.pop(0)

    _planning_updates(env, planning_states, policy_net, memory, config, device)
    return success, total_reward


def _n_step_transition(buffer, gamma, device):
    R = 0.0
    for i, tr in enumerate(buffer):
        R += (gamma ** i) * float(tr[2].item())
        if tr[4]:
            break
    return (
        buffer[0][0],
        buffer[0][1],
        torch.tensor([R], device=device, dtype=torch.float32),
        buffer[-1][3],
        buffer[-1][4],
    )


def _planning_updates(env, planning_states, policy_net, memory, config, device):
    """«Воображаемые» переходы из детерминированного симулятора той же карты."""
    if not planning_states:
        return
    size = env.size
    batch = env.batch
    idx = env.map_idx
    for _ in range(config["PLANNING_STEPS"]):
        state = random.choice(planning_states)
        with torch.no_grad():
            action = int(policy_net(state).argmax(dim=1).item())
        one_hot = state[0, 0].cpu().numpy()
        r, c = np.unravel_index(int(one_hot.argmax()), (size, size))
        dr, dc = ACTION_DELTA[action]
        nr = int(np.clip(r + dr, 0, size - 1))
        nc = int(np.clip(c + dc, 0, size - 1))

        reward, done = 0.0, False
        if batch.is_hole[idx, nr, nc]:
            done = True
        elif batch.is_goal[idx, nr, nc]:
            reward, done = 1.0, True

        new_one_hot = np.zeros((size, size), dtype=np.float32)
        new_one_hot[nr, nc] = 1.0
        next_obs = np.stack([new_one_hot, batch.encoded[idx], batch.bfs[idx]], axis=0)
        next_state = (
            torch.zeros_like(state) if done else torch.from_numpy(next_obs).unsqueeze(0).to(device)
        )
        memory.push(
            (
                state,
                torch.tensor([[action]], device=device, dtype=torch.long),
                torch.tensor([reward], device=device, dtype=torch.float32),
                next_state,
                done,
            )
        )


def train(args):
    config = dict(DEFAULT_CONFIG)
    if args.eps_start is not None:
        config["EPS_START"] = args.eps_start
    if args.lr is not None:
        config["LR"] = args.lr
    # По умолчанию: дообучение наследует шейпинг чекпоинта, обучение с нуля берёт BFS
    config["SHAPING"] = args.shaping if args.shaping != "auto" else (
        "manhattan" if args.init_from else "bfs")

    device = torch.device(args.device if args.device != "auto"
                          else ("cuda" if torch.cuda.is_available() else "cpu"))
    torch.set_num_threads(args.threads)
    if args.seed is not None:
        set_seed(args.seed)

    build_logger(os.path.join("results", "curriculum.log"))
    logging.info("Устройство: %s, потоков torch: %d", device, args.threads)
    logging.info("Конфиг: %s", config)

    batch = all_maps_batch(config["MAP_SIZE"])
    n_maps = len(batch)
    logging.info("Карт в перечислении: %d", n_maps)

    policy_net = DQNCNN(config["MAP_SIZE"], 4).to(device)
    target_net = DQNCNN(config["MAP_SIZE"], 4).to(device)

    start_episode = 1
    sampler = MapSampler(n_maps, config["HARD_FRACTION"], config["EMA_DECAY"], config["HARD_THRESHOLD"])
    optimizer = optim.Adam(policy_net.parameters(), lr=config["LR"])

    loaded_target = False
    if args.init_from:
        ckpt = torch.load(args.init_from, map_location=device, weights_only=False)
        policy_net.load_state_dict(ckpt["policy_net_state_dict"])
        logging.info("Веса инициализированы из %s", args.init_from)

        # Гиперпараметры чекпоинта задают функцию награды и правило бутстрэпа, под
        # которые выучены Q-значения. Дообучать с другими — значит менять задачу под
        # моделью; именно это обрушило смоук-прогон с 99.53% до 78.68%.
        if args.use_ckpt_config and "config" in ckpt:
            inherit = ("MAP_SIZE", "BATCH_SIZE", "REPLAY_CAPACITY", "N_STEP", "GAMMA",
                       "LR_DECAY", "SOFT_TAU", "PLANNING_STEPS", "STEP_PENALTY",
                       "IMPROVEMENT_FACTOR", "REVISIT_PENALTY")
            adopted = {k: ckpt["config"][k] for k in inherit if k in ckpt["config"]}
            config.update(adopted)
            logging.info("Приняты гиперпараметры чекпоинта: %s", adopted)

        # Состояние Adam и target-сеть тоже переносим: свежие моменты Adam дают
        # непропорционально большие первые шаги по уже сошедшимся весам.
        if args.inherit_optimizer and "optimizer_state_dict" in ckpt:
            optimizer = optim.Adam(policy_net.parameters(), lr=config["LR"])
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            for g in optimizer.param_groups:
                g["lr"] = config["LR"]
            logging.info("Состояние оптимизатора перенесено из чекпоинта")
        if "target_net_state_dict" in ckpt:
            target_net.load_state_dict(ckpt["target_net_state_dict"])
            loaded_target = True

        if args.resume and "sampler" in ckpt:
            sampler.load_state_dict(ckpt["sampler"])
            start_episode = ckpt.get("episode", 0) + 1
            logging.info("Возобновление с эпизода %d", start_episode)

    if not loaded_target:
        target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=config["LR_DECAY"])
    memory = ReplayMemory(config["REPLAY_CAPACITY"])
    env = FastFrozenLake(batch, config["MAX_EPISODE_STEPS"])

    # стартовая оценка — она же задаёт первый трудный пул
    n_ok, total, rate, success_mask = _full_eval(policy_net, batch, device)
    sampler.update(success_mask)
    logging.info("Старт: %d/%d (%.2f%%), трудный пул: %d карт", n_ok, total, rate, sampler.hard_pool.size)

    best_rate = rate
    best_state = {k: v.clone() for k, v in policy_net.state_dict().items()}
    os.makedirs("models", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    eval_csv_path = os.path.join("results", "curriculum_eval.csv")
    write_header = not os.path.exists(eval_csv_path) or os.path.getsize(eval_csv_path) == 0
    eval_csv = open(eval_csv_path, "a", newline="", encoding="utf-8")
    eval_writer = csv.writer(eval_csv)
    if write_header:
        eval_writer.writerow(["episode", "solved", "total", "rate", "hard_pool", "eps", "lr", "seconds"])

    # Прогрев: набрать переходы БЕЗ обновления весов. Иначе первые шаги обучения идут
    # по почти пустому буферу из сильно скоррелированных переходов одной-двух карт.
    if args.warmup_episodes:
        for _ in range(args.warmup_episodes):
            run_episode(env, sampler, policy_net, target_net, memory, optimizer, config,
                        device, config["EPS_END"], do_optimize=False)
        logging.info("Прогрев: %d эпизодов, в буфере %d переходов",
                     args.warmup_episodes, len(memory))

    steps_done = 0
    recent_success = []
    t_start = time.perf_counter()
    reached_100 = False

    try:
        for episode in range(start_episode, args.episodes + 1):
            eps = max(config["EPS_END"], config["EPS_START"] * (config["EPS_DECAY"] ** steps_done))
            success, _ = run_episode(
                env, sampler, policy_net, target_net, memory, optimizer, config, device, eps
            )
            steps_done += 1
            scheduler.step()
            recent_success.append(success)

            if episode % 250 == 0:
                last = recent_success[-250:]
                logging.info(
                    "Эпизод %d: успех(250)=%.1f%%, eps=%.4f, lr=%.2e, трудный пул=%d",
                    episode, 100.0 * sum(last) / len(last), eps,
                    optimizer.param_groups[0]["lr"], sampler.hard_pool.size,
                )

            if episode % args.eval_every == 0:
                n_ok, total, rate, success_mask = _full_eval(policy_net, batch, device)
                sampler.update(success_mask)
                elapsed = time.perf_counter() - t_start
                logging.info(
                    "ОЦЕНКА эпизод %d: %d/%d (%.2f%%) | лучший %.2f%% | трудный пул %d | %.0f c",
                    episode, n_ok, total, rate, best_rate, sampler.hard_pool.size, elapsed,
                )
                eval_writer.writerow([episode, n_ok, total, f"{rate:.4f}", sampler.hard_pool.size,
                                      f"{eps:.5f}", f"{optimizer.param_groups[0]['lr']:.3e}", f"{elapsed:.0f}"])
                eval_csv.flush()

                if rate > best_rate:
                    best_rate = rate
                    best_state = {k: v.clone() for k, v in policy_net.state_dict().items()}
                    torch.save({"policy_net_state_dict": best_state, "config": config,
                                "episode": episode, "rate": rate, "seed": args.seed},
                               os.path.join("models", args.best_name))
                    logging.info("Новый лучший результат %.2f%% -> models/%s", rate, args.best_name)
                elif rate < best_rate - args.catastrophe_drop:
                    # ровно тот сценарий, что обрушил прошлый fine-tune до 73%
                    logging.warning("Падение %.2f%% ниже лучшего %.2f%% — откат к лучшим весам",
                                    rate, best_rate)
                    policy_net.load_state_dict(best_state)
                    target_net.load_state_dict(best_state)
                    for g in optimizer.param_groups:
                        g["lr"] *= 0.5

                _save_resume(policy_net, target_net, optimizer, sampler, config, episode, rate, args)

                if n_ok == total:
                    logging.info("Достигнуто 100%% (%d/%d) на эпизоде %d", n_ok, total, episode)
                    reached_100 = True
                    break

    except KeyboardInterrupt:
        logging.info("Прервано пользователем на эпизоде %d", episode)
    finally:
        eval_csv.close()

    # финальная сверка лучшего снапшота
    policy_net.load_state_dict(best_state)
    n_ok, total, rate, success_mask = _full_eval(policy_net, batch, device)
    logging.info("ИТОГ (лучший снапшот): %d/%d (%.2f%%)", n_ok, total, rate)

    failed = [generate_all_valid_maps(config["MAP_SIZE"])[i] for i in np.flatnonzero(~success_mask)]
    with open(os.path.join("results", "curriculum_failed_maps.json"), "w", encoding="utf-8") as f:
        json.dump(failed, f, ensure_ascii=False, indent=4)
    logging.info("Оставшиеся провалы: %d (results/curriculum_failed_maps.json)", len(failed))
    return reached_100


def _full_eval(model, batch, device):
    success = evaluate(model, batch, device)
    n_ok = int(success.sum())
    total = len(batch)
    return n_ok, total, n_ok / total * 100.0, success


def _save_resume(policy_net, target_net, optimizer, sampler, config, episode, rate, args):
    torch.save(
        {
            "episode": episode,
            "rate": rate,
            "policy_net_state_dict": policy_net.state_dict(),
            "target_net_state_dict": target_net.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "sampler": sampler.state_dict(),
            "config": config,
            "seed": args.seed,
        },
        os.path.join("checkpoints", args.resume_name),
    )


def main():
    p = argparse.ArgumentParser(description="Приоритизированный curriculum до 100% на всех валидных картах")
    p.add_argument("--episodes", type=int, default=40000)
    p.add_argument("--eval-every", type=int, default=1000)
    p.add_argument("--init-from", default=None, help="Чекпоинт для инициализации весов")
    p.add_argument("--resume", action="store_true", help="Вместе с --init-from: продолжить состояние (sampler, эпизод)")
    p.add_argument("--use-ckpt-config", action="store_true",
                   help="Взять гиперпараметры (GAMMA, N_STEP, shaping-коэффициенты) из чекпоинта")
    p.add_argument("--inherit-optimizer", action="store_true",
                   help="Перенести состояние Adam из чекпоинта")
    p.add_argument("--shaping", default="auto", choices=["auto", "bfs", "manhattan"])
    p.add_argument("--warmup-episodes", type=int, default=0,
                   help="Эпизодов сбора переходов без обновления весов перед обучением")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda", "auto"])
    p.add_argument("--threads", type=int, default=1,
                   help="Потоков torch. Для сети такого размера 1 обычно быстрее многопоточности")
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--eps-start", type=float, default=None)
    p.add_argument("--catastrophe-drop", type=float, default=3.0,
                   help="Падение в п.п. ниже лучшего, при котором откатываемся к лучшим весам")
    p.add_argument("--best-name", default="best_curriculum.pth")
    p.add_argument("--resume-name", default="curriculum_resume.pth")
    args = p.parse_args()
    train(args)


if __name__ == "__main__":
    main()
