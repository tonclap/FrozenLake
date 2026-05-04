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

from trainer import select_action, get_n_step_info, optimize_model, ReplayMemory
from dqn_model import AugmentedObservationWrapperCNN, DQNCNN
from utils import set_seed, test_on_map

def fine_tune_on_failed_maps(model, failed_maps, config, device, num_episodes=2000, max_episode_steps=100):
    """
    Дообучает модель на наборе "трудных" карт.
    Используется сниженная скорость обучения (10% от исходной) и низкий epsilon (0.05).
    Логируются:
      - количество эпизодов за каждую минуту
      - средний процент успеха за последние 100 эпизодов (вычисляется каждые 100 эпизодов)
    """
    logging.info("Начинаем дообучение на трудных картах...")

    target_model = copy.deepcopy(model)
    target_model.eval()

    fine_tune_lr = config["LR"] * 0.1
    optimizer = optim.Adam(model.parameters(), lr=fine_tune_lr)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=config["LR_DECAY"])

    memory = ReplayMemory(capacity=1000)
    n_actions = 4

    overall_start_time = time.time()
    last_global_log_time = overall_start_time
    success_list = []

    for episode in range(1, num_episodes + 1):
        steps_in_episode = 0
        total_reward = 0.0
        episode_success = 0

        # Создаем окружение с нужными обёртками
        env = gym.make("FrozenLake-v1", desc=random.choice(failed_maps), is_slippery=False)
        env = AugmentedObservationWrapperCNN(env)
        env = gym.wrappers.TimeLimit(env, max_episode_steps=max_episode_steps)

        obs, _ = env.reset()
        state = torch.from_numpy(obs).unsqueeze(0).to(device)
        n_step_buffer = []
        done = False
        
        eps_threshold = 0.05
        while not done:
            action = select_action(state, model, steps_done=0, n_actions=n_actions, eps_threshold=eps_threshold)
            next_obs, reward, terminated, truncated, _ = env.step(action.item())
            done_flag = terminated or truncated
            next_state = torch.from_numpy(next_obs).unsqueeze(0).to(device) if not done_flag else torch.zeros_like(state)
            n_step_buffer.append((state, action, torch.tensor([reward], device=device, dtype=torch.float32), next_state, done_flag))
            state = next_state
            total_reward += reward
            steps_in_episode += 1
            if done_flag:
                break
        
        env.close()
        
        # Если за эпизод получена награда 1.0, считаем его успешным
        if total_reward >= 1.0:
            episode_success = 1
        success_list.append(episode_success)
        
        # Обработка n-шаговых переходов
        while n_step_buffer:
            transition = get_n_step_info(n_step_buffer, config["GAMMA"])
            memory.push(transition)
            n_step_buffer.pop(0)
        
        for _ in range(10):
            optimize_model(model, target_model, memory, optimizer, config)
        scheduler.step()
        
        # Лог каждые 60 секунд – вывод количества пройденных эпизодов за время
        current_time = time.time()
        if current_time - last_global_log_time >= 60:
            elapsed_overall = current_time - overall_start_time
            logging.info(f"Дообучение: пройдено {episode} эпизодов за {elapsed_overall:.1f} секунд")
            print(f"Дообучение: пройдено {episode} эпизодов за {elapsed_overall:.1f} секунд")
            last_global_log_time = current_time
        
        # Каждые 100 эпизодов вычисляем средний процент успеха за последние 100 эпизодов
        if episode % 100 == 0:
            last_100_success = (sum(success_list[-100:]) / 100.0) * 100.0
            logging.info(f"Дообучение, эпизод {episode}: средний процент успеха за последние 100 эпизодов = {last_100_success:.2f}%")
            print(f"Дообучение, эпизод {episode}: средний процент успеха за последние 100 эпизодов = {last_100_success:.2f}%")
            
    return model

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Дообучение модели DQN на проблемных картах")
    parser.add_argument("checkpoint", help="Путь к файлу чекпоинта (.pth)")
    parser.add_argument("failed_maps", help="Путь к JSON с проваленными картами")
    parser.add_argument("--seed", type=int, default=None,
                        help="Seed для воспроизводимости (по умолчанию из чекпоинта или случайный)")
    parser.add_argument("--num-episodes", type=int, default=2000,
                        help="Число эпизодов дообучения (по умолчанию 2000)")
    args = parser.parse_args()

    logging.basicConfig(filename="finetune.log",
                        level=logging.INFO,
                        format="%(asctime)s - %(levelname)s - %(message)s",
                        encoding="utf-8")

    checkpoint_path = args.checkpoint
    failed_maps_path = args.failed_maps
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    map_size = 4
    n_actions = 4
    
    model = DQNCNN(map_size, n_actions).to(device)
    if not os.path.exists(checkpoint_path):
        print(f"Файл чекпойнта {checkpoint_path} не найден.")
        logging.error(f"Файл чекпойнта {checkpoint_path} не найден.")
        return
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["policy_net_state_dict"])
    model.eval()
    logging.info(f"Модель загружена из {checkpoint_path}")
    print(f"Модель загружена из {checkpoint_path}")

    effective_seed = args.seed if args.seed is not None else checkpoint.get("seed")
    if effective_seed is not None:
        set_seed(effective_seed)
        logging.info(f"Seed установлен: {effective_seed}")
        print(f"Seed: {effective_seed}")
    
    if "config" in checkpoint:
        config = checkpoint["config"]
    else:
        from config import generate_experiment_config
        config = generate_experiment_config()
        logging.info("Конфигурация сгенерирована заново, так как не найдена в чекпойнте.")
    
    try:
        with open(failed_maps_path, "r", encoding="utf-8") as f:
            failed_maps = json.load(f)
    except Exception as e:
        print(f"Ошибка при загрузке {failed_maps_path}: {e}")
        logging.error(f"Ошибка при загрузке {failed_maps_path}: {e}")
        return
    
    logging.info(f"Загружено {len(failed_maps)} карт, на которых модель ошибалась.")
    print(f"Загружено {len(failed_maps)} карт, на которых модель ошибалась.")
    
    fine_tuned_model = fine_tune_on_failed_maps(model, failed_maps, config, device,
                                                num_episodes=args.num_episodes,
                                                max_episode_steps=100)
    
    fine_tuned_checkpoint = {
        "policy_net_state_dict": fine_tuned_model.state_dict(),
        "config": config
    }
    fine_tuned_path = checkpoint_path.replace(".pth", "_finetuned.pth")
    torch.save(fine_tuned_checkpoint, fine_tuned_path)
    logging.info(f"Дообученная модель сохранена в {fine_tuned_path}")
    print(f"Дообученная модель сохранена в {fine_tuned_path}")

if __name__ == "__main__":
    main()