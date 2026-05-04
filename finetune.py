#!/usr/bin/env python
import copy
import os
import sys
import json
import random
import time
import gym
import torch
import torch.optim as optim
import logging

from trainer import (
    select_action, get_n_step_info, optimize_model, ReplayMemory,
    soft_update, get_agent_position, simulate_transition, test_current_model,
)
from dqn_model import AugmentedObservationWrapperCNN, DQNCNN
from utils import set_seed


def fine_tune_on_failed_maps(
    model, failed_maps, config, device,
    num_episodes=10000, max_episode_steps=200,
):
    """
    Дообучает модель на наборе «трудных» карт с полным reward shaping,
    убывающим epsilon и контролем катастрофического забывания.

    Каждые 500 эпизодов тестируются все 3828 валидных карт; если общий
    success rate падает ниже 99.3% после того, как модель уже набрала ≥99.5%,
    восстанавливается лучший сохранённый checkpoint.

    Возвращает модель с наилучшим зафиксированным весом по всем картам.
    """
    logging.info("Начинаем дообучение на %d трудных картах...", len(failed_maps))

    # target_model обновляется через soft_update внутри optimize_model
    target_model = copy.deepcopy(model)
    target_model.train()

    fine_tune_lr = config["LR"] * 0.05
    optimizer = optim.Adam(model.parameters(), lr=fine_tune_lr)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=config["LR_DECAY"])

    memory = ReplayMemory(capacity=20000)
    n_actions = 4

    # Убывающий epsilon: 0.20 → 0.03
    eps_start = 0.20
    eps_end = 0.03
    eps_per_episode_decay = 0.9995

    overall_start_time = time.time()
    last_log_time = overall_start_time
    success_list = []

    best_all_rate = 0.0
    best_state_dict = copy.deepcopy(model.state_dict())

    for episode in range(1, num_episodes + 1):
        eps_threshold = max(eps_end, eps_start * (eps_per_episode_decay ** episode))

        map_desc = random.choice(failed_maps)
        env = gym.make("FrozenLake-v1", desc=map_desc, is_slippery=False)
        env = AugmentedObservationWrapperCNN(env)
        env = gym.wrappers.TimeLimit(env, max_episode_steps=max_episode_steps)

        obs, _ = env.reset()
        state = torch.from_numpy(obs).unsqueeze(0).to(device)

        nrow, ncol = obs.shape[1], obs.shape[2]
        goal = (nrow - 1, ncol - 1)
        pos = get_agent_position(state)
        prev_distance = abs(goal[0] - pos[0]) + abs(goal[1] - pos[1])
        visited = {pos}

        n_step_buffer = []
        planning_buffer = []
        done = False
        episode_success = 0
        total_reward = 0.0

        while not done:
            planning_buffer.append(state)
            action = select_action(state, model, steps_done=0, n_actions=n_actions,
                                   eps_threshold=eps_threshold)
            next_obs, reward, terminated, truncated, _ = env.step(action.item())
            done_flag = terminated or truncated

            if reward == 1.0:
                episode_success = 1

            next_state = (torch.from_numpy(next_obs).unsqueeze(0).to(device)
                          if not done_flag else torch.zeros_like(state))
            curr_pos = get_agent_position(next_state) if not done_flag else pos
            curr_distance = abs(goal[0] - curr_pos[0]) + abs(goal[1] - curr_pos[1])

            improvement_reward = config["IMPROVEMENT_FACTOR"] * max(prev_distance - curr_distance, 0)
            revisit_penalty = config["REVISIT_PENALTY"] if curr_pos in visited else 0.0
            shaping_reward = improvement_reward - revisit_penalty - config["STEP_PENALTY"]
            total_step_reward = reward + shaping_reward

            visited.add(curr_pos)
            prev_distance = curr_distance
            total_reward += total_step_reward

            reward_tensor = torch.tensor([total_step_reward], device=device, dtype=torch.float32)
            n_step_buffer.append((state, action, reward_tensor, next_state, done_flag))
            if len(n_step_buffer) >= config["N_STEP"]:
                transition = get_n_step_info(n_step_buffer[:config["N_STEP"]], config["GAMMA"])
                memory.push(transition)
                n_step_buffer.pop(0)

            state = next_state if not done_flag else None
            if done_flag:
                break

        env.close()

        while n_step_buffer:
            transition = get_n_step_info(n_step_buffer, config["GAMMA"])
            memory.push(transition)
            n_step_buffer.pop(0)

        # Dyna-style planning — те же шаги, что и в основном обучении
        for _ in range(config.get("PLANNING_STEPS", 10)):
            if not planning_buffer:
                break
            sampled_state = random.choice(planning_buffer)
            with torch.no_grad():
                sim_action = model(sampled_state).max(1)[1].view(1, 1)
            sim_next_state, sim_reward, sim_done = simulate_transition(sampled_state, sim_action.item())
            sim_reward_tensor = torch.tensor([sim_reward], device=device, dtype=torch.float32)
            memory.push((sampled_state, sim_action, sim_reward_tensor, sim_next_state, sim_done))

        optimize_model(model, target_model, memory, optimizer, config)
        scheduler.step()

        success_list.append(episode_success)

        # Контроль катастрофического забывания: тест на всех картах каждые 500 эпизодов
        if episode % 500 == 0:
            successful, total_maps, all_rate = test_current_model(model, 4, device)
            logging.info(
                "Fine-tune ep %d: all maps %d/%d (%.2f%%), eps=%.4f",
                episode, successful, total_maps, all_rate, eps_threshold,
            )
            print(
                f"Fine-tune ep {episode}: all maps {successful}/{total_maps} "
                f"({all_rate:.2f}%), eps={eps_threshold:.4f}"
            )
            if all_rate > best_all_rate:
                best_all_rate = all_rate
                best_state_dict = copy.deepcopy(model.state_dict())
                logging.info("New best all-maps rate: %.2f%% — checkpoint saved", best_all_rate)

            # Откат при катастрофическом забывании
            if all_rate < 99.3 and best_all_rate >= 99.5:
                logging.warning(
                    "Catastrophic forgetting detected (%.2f%%)! Restoring best checkpoint (%.2f%%)",
                    all_rate, best_all_rate,
                )
                print(
                    f"Catastrophic forgetting: {all_rate:.2f}% < 99.3%. "
                    f"Restoring checkpoint ({best_all_rate:.2f}%)"
                )
                model.load_state_dict(best_state_dict)
                break

        if episode % 100 == 0:
            last_100 = (sum(success_list[-100:]) / 100.0) * 100.0
            logging.info(
                "Fine-tune ep %d: failed-maps success=%.2f%%, eps=%.4f",
                episode, last_100, eps_threshold,
            )
            print(f"Fine-tune ep {episode}: failed-maps success={last_100:.2f}%, eps={eps_threshold:.4f}")

        current_time = time.time()
        if current_time - last_log_time >= 60:
            elapsed = current_time - overall_start_time
            logging.info("Fine-tune: %d episodes in %.1f seconds", episode, elapsed)
            print(f"Fine-tune: {episode} episodes in {elapsed:.1f}s")
            last_log_time = current_time

    # Гарантируем возврат лучших весов
    model.load_state_dict(best_state_dict)
    logging.info("Fine-tuning complete. Best all-maps rate: %.2f%%", best_all_rate)
    print(f"Fine-tuning complete. Best all-maps rate: {best_all_rate:.2f}%")
    return model


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Дообучение модели DQN на проблемных картах")
    parser.add_argument("checkpoint", help="Путь к файлу чекпоинта (.pth)")
    parser.add_argument("failed_maps", help="Путь к JSON с проваленными картами")
    parser.add_argument("--seed", type=int, default=None,
                        help="Seed для воспроизводимости (по умолчанию из чекпоинта или случайный)")
    parser.add_argument("--num-episodes", type=int, default=10000,
                        help="Число эпизодов дообучения (по умолчанию 10000)")
    parser.add_argument("--max-steps", type=int, default=200,
                        help="Максимум шагов в эпизоде при дообучении (по умолчанию 200)")
    args = parser.parse_args()

    logging.basicConfig(
        filename="finetune.log",
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        encoding="utf-8",
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    map_size = 4
    n_actions = 4

    model = DQNCNN(map_size, n_actions).to(device)
    if not os.path.exists(args.checkpoint):
        print(f"Файл чекпойнта {args.checkpoint} не найден.")
        logging.error("Файл чекпойнта %s не найден.", args.checkpoint)
        sys.exit(1)

    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["policy_net_state_dict"])
    model.eval()
    logging.info("Модель загружена из %s", args.checkpoint)
    print(f"Модель загружена из {args.checkpoint}")

    effective_seed = args.seed if args.seed is not None else checkpoint.get("seed")
    if effective_seed is not None:
        set_seed(effective_seed)
        logging.info("Seed установлен: %d", effective_seed)
        print(f"Seed: {effective_seed}")

    if "config" in checkpoint:
        config = checkpoint["config"]
    else:
        from config import generate_experiment_config
        config = generate_experiment_config()
        logging.info("Конфигурация сгенерирована заново (не найдена в чекпойнте).")

    try:
        with open(args.failed_maps, "r", encoding="utf-8") as f:
            failed_maps = json.load(f)
    except Exception as e:
        print(f"Ошибка при загрузке {args.failed_maps}: {e}")
        logging.error("Ошибка при загрузке %s: %s", args.failed_maps, e)
        sys.exit(1)

    logging.info("Загружено %d проваленных карт.", len(failed_maps))
    print(f"Загружено {len(failed_maps)} проваленных карт.")

    fine_tuned_model = fine_tune_on_failed_maps(
        model, failed_maps, config, device,
        num_episodes=args.num_episodes,
        max_episode_steps=args.max_steps,
    )

    fine_tuned_path = args.checkpoint.replace(".pth", "_finetuned.pth")
    fine_tuned_checkpoint = {
        "policy_net_state_dict": fine_tuned_model.state_dict(),
        "config": config,
        "seed": effective_seed,
    }
    torch.save(fine_tuned_checkpoint, fine_tuned_path)
    logging.info("Дообученная модель сохранена в %s", fine_tuned_path)
    print(f"Дообученная модель сохранена в {fine_tuned_path}")


if __name__ == "__main__":
    main()
