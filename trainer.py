# trainer.py
import os
import csv
import time
import random
import logging
import warnings
from collections import deque

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from dqn_model import AugmentedObservationWrapperCNN, DQNCNN
from utils import generate_all_valid_maps, test_on_map, set_seed

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_agent_position(state):
    state_np = state.squeeze(0).cpu().numpy()
    one_hot = state_np[0]
    nrow, ncol = one_hot.shape
    agent_idx = np.argmax(one_hot.flatten())
    return np.unravel_index(agent_idx, (nrow, ncol))

def simulate_transition(state, action):
    state_np = state.squeeze(0).cpu().numpy()
    nrow, ncol = state_np.shape[1], state_np.shape[2]
    agent_idx = np.argmax(state_np[0].flatten())
    row, col = np.unravel_index(agent_idx, (nrow, ncol))
    
    action_delta = {0: (0, -1), 1: (1, 0), 2: (0, 1), 3: (-1, 0)}
    dr, dc = action_delta.get(action, (0, 0))
    new_row, new_col = row + dr, col + dc
    if not (0 <= new_row < nrow and 0 <= new_col < ncol):
        new_row, new_col = row, col

    cell_value = state_np[1, new_row, new_col]
    done = False
    reward = 0.0
    if cell_value == -1.0:
        done = True
    elif cell_value == 2.0:
        reward = 1.0
        done = True

    one_hot = np.zeros((nrow, ncol), dtype=np.float32)
    one_hot[new_row, new_col] = 1.0
    new_state_np = np.stack([one_hot, state_np[1], state_np[2]], axis=0)[np.newaxis, ...]
    new_state = torch.from_numpy(new_state_np).to(device)
    return new_state, reward, done

def create_env(map_size, p):
    from gym.envs.toy_text.frozen_lake import generate_random_map
    desc = generate_random_map(size=map_size, p=p)
    env = gym.make("FrozenLake-v1", desc=desc, is_slippery=False)
    return AugmentedObservationWrapperCNN(env)

def select_action(state, policy_net, steps_done, n_actions, eps_threshold):
    if random.random() < eps_threshold:
        return torch.tensor([[random.randrange(n_actions)]], device=device, dtype=torch.long)
    else:
        with torch.no_grad():
            return policy_net(state).max(1)[1].view(1, 1)

def soft_update(target_net, policy_net, tau):
    for target_param, policy_param in zip(target_net.parameters(), policy_net.parameters()):
        target_param.data.copy_(tau * policy_param.data + (1.0 - tau) * target_param.data)

def optimize_model(policy_net, target_net, memory, optimizer, config):
    if len(memory) < config["BATCH_SIZE"]:
        return None
    transitions = memory.sample(config["BATCH_SIZE"])
    batch = list(zip(*transitions))
    state_batch = torch.cat(batch[0])
    action_batch = torch.cat(batch[1])
    reward_batch = torch.cat(batch[2])
    next_state_batch = torch.cat(batch[3])
    done_batch = torch.tensor(batch[4], device=device, dtype=torch.float32)

    state_action_values = policy_net(state_batch).gather(1, action_batch)
    next_actions = policy_net(next_state_batch).max(1)[1].unsqueeze(1)
    next_state_values = target_net(next_state_batch).gather(1, next_actions).squeeze().detach()
    gamma_n = config["GAMMA"] ** config["N_STEP"]
    expected_values = (next_state_values * gamma_n * (1 - done_batch)) + reward_batch

    loss = nn.MSELoss()(state_action_values.squeeze(), expected_values)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    soft_update(target_net, policy_net, config["SOFT_TAU"])
    return loss.item()

def get_n_step_info(n_step_buffer, gamma):
    R = 0.0
    for idx, transition in enumerate(n_step_buffer):
        R += (gamma ** idx) * transition[2].item()
        if transition[4]:
            break
    next_state = n_step_buffer[-1][3]
    done = n_step_buffer[-1][4]
    return n_step_buffer[0][0], n_step_buffer[0][1], torch.tensor([R], device=device, dtype=torch.float32), next_state, done

class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)
    def push(self, transition):
        self.memory.append(transition)
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
    def __len__(self):
        return len(self.memory)

def test_current_model(model, map_size, device):
    valid_maps = generate_all_valid_maps(map_size)
    total_maps = len(valid_maps)
    successful = 0
    for map_desc in valid_maps:
        reward = test_on_map(model, map_desc, device, AugmentedObservationWrapperCNN)
        if reward >= 1.0:
            successful += 1
    success_rate = (successful / total_maps) * 100
    return successful, total_maps, success_rate

def train_model(config, experiment_number, test_writer, seed=None):
    if seed is not None:
        set_seed(seed)
        logging.info("Seed set to %d for experiment %d", seed, experiment_number)
    logging.info("Starting training with config: %s", config)
    writer = SummaryWriter(log_dir=f"runs/exp_{experiment_number}_{int(time.time())}")
    n_actions = 4

    policy_net = DQNCNN(config["MAP_SIZE"], n_actions).to(device)
    target_net = DQNCNN(config["MAP_SIZE"], n_actions).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters(), lr=config["LR"])
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=config["LR_DECAY"])
    memory = ReplayMemory(config["REPLAY_CAPACITY"])

    steps_done = 0
    episode_rewards = []
    episode_success_list = []
    best_overall_success = 0.0

    checkpoint_dir = "checkpoints"
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    if not os.path.exists("models"):
        os.makedirs("models")

    special_phase_started = False
    test_interval = 3000  # Тестирование каждые 3000 эпизодов

    try:
        for episode in range(1, config["NUM_EPISODES"] + 1):
            if episode <= 5:
                logging.info("Episode %d starting", episode)
                
            if episode <= config["CURRICULUM_EPISODES"]:
                current_p = config["START_P"] - (config["START_P"] - config["END_P"]) * (episode / config["CURRICULUM_EPISODES"])
            elif random.random() < config.get("HARD_INJECTION_PROB", 0.0):
                # Инъекция экстремально трудных карт: плотность дыр 35-50%, аналогично проваленным картам
                current_p = random.uniform(
                    config.get("HARD_INJECTION_P_MIN", 0.50),
                    config.get("HARD_INJECTION_P_MAX", 0.65),
                )
            else:
                current_p = random.uniform(config["POST_CURRICULUM_P_MIN"], config["POST_CURRICULUM_P_MAX"])

            env = create_env(config["MAP_SIZE"], current_p)
            obs, _ = env.reset()
            state = torch.from_numpy(obs).unsqueeze(0).to(device)
            total_reward = 0.0
            done = False
            n_step_buffer = []
            planning_buffer = []
            visited = set()

            nrow, ncol = obs.shape[1], obs.shape[2]
            goal = (nrow - 1, ncol - 1)
            pos = get_agent_position(state)
            visited.add(pos)
            prev_distance = abs(goal[0] - pos[0]) + abs(goal[1] - pos[1])
            episode_success = 0

            if episode == config["CURRICULUM_EPISODES"] + 1 and not special_phase_started:
                logging.info("Transition to special phase: updating hyperparameters")
                for param_group in optimizer.param_groups:
                    param_group['lr'] = config["NEW_LR"]
                scheduler.gamma = config["NEW_LR_DECAY"]
                config["EPS_DECAY"] = config["NEW_EPS_DECAY"]
                config["EPS_END"] = config["NEW_EPS_END"]
                config["GAMMA"] = config["NEW_GAMMA"]
                config["PLANNING_STEPS"] = config["NEW_PLANNING_STEPS"]
                special_phase_started = True

            eps_threshold = max(config["EPS_END"], config["EPS_START"] * (config["EPS_DECAY"] ** steps_done))

            while not done:
                planning_buffer.append(state)
                action = select_action(state, policy_net, steps_done, n_actions, eps_threshold)
                steps_done += 1

                next_obs, reward, terminated, truncated, _ = env.step(action.item())
                done_flag = terminated or truncated
                if reward == 1.0:
                    episode_success = 1

                next_state = torch.from_numpy(next_obs).unsqueeze(0).to(device) if not done_flag else torch.zeros_like(state)
                curr_pos = get_agent_position(next_state) if not done_flag else pos
                curr_distance = abs(goal[0] - curr_pos[0]) + abs(goal[1] - curr_pos[1])

                improvement_reward = config["IMPROVEMENT_FACTOR"] * max(prev_distance - curr_distance, 0)
                revisit_penalty = config["REVISIT_PENALTY"] if curr_pos in visited else 0.0
                shaping_reward = improvement_reward - revisit_penalty - config["STEP_PENALTY"]

                visited.add(curr_pos)
                prev_distance = curr_distance

                total_step_reward = reward + shaping_reward
                total_reward += total_step_reward

                writer.add_scalar("Reward/step", total_step_reward, steps_done)
                reward_tensor = torch.tensor([total_step_reward], device=device, dtype=torch.float32)
                n_step_buffer.append((state, action, reward_tensor, next_state, done_flag))
                if len(n_step_buffer) >= config["N_STEP"]:
                    transition = get_n_step_info(n_step_buffer[:config["N_STEP"]], config["GAMMA"])
                    memory.push(transition)
                    n_step_buffer.pop(0)
                state = next_state if not done_flag else None

                loss_val = optimize_model(policy_net, target_net, memory, optimizer, config)
                if done_flag:
                    break

            while n_step_buffer:
                transition = get_n_step_info(n_step_buffer, config["GAMMA"])
                memory.push(transition)
                n_step_buffer.pop(0)

            for _ in range(config["PLANNING_STEPS"]):
                if not planning_buffer:
                    break
                sampled_state = random.choice(planning_buffer)
                with torch.no_grad():
                    simulated_action = policy_net(sampled_state).max(1)[1].view(1, 1)
                simulated_next_state, sim_reward, sim_done = simulate_transition(sampled_state, simulated_action.item())
                sim_reward_tensor = torch.tensor([sim_reward], device=device, dtype=torch.float32)
                memory.push((sampled_state, simulated_action, sim_reward_tensor, simulated_next_state, sim_done))

            episode_rewards.append(total_reward)
            episode_success_list.append(episode_success)
            env.close()
            scheduler.step()

            writer.add_scalar("Reward/episode", total_reward, episode)
            writer.add_scalar("Epsilon/episode", eps_threshold, episode)
            writer.add_scalar("LearningRate/episode", optimizer.param_groups[0]['lr'], episode)
            writer.add_scalar("Map_difficulty_p", current_p, episode)
            
            if len(episode_success_list) >= 100:
                avg_success_last_100 = (sum(episode_success_list[-100:]) / 100.0) * 100.0
            else:
                avg_success_last_100 = (sum(episode_success_list) / len(episode_success_list)) * 100.0
            avg_success_all = (sum(episode_success_list) / len(episode_success_list)) * 100.0

            writer.add_scalar("Success/avg_percentage", avg_success_last_100, episode)
            writer.add_scalar("Success/avg_percentage_all", avg_success_all, episode)
            
            if episode % 100 == 0:
                avg_reward = np.mean(episode_rewards[-100:])
                success_rate_last100 = (sum(episode_success_list[-100:]) / 100.0) * 100.0
                logging.info("Episode %d, avg reward: %.2f, success (last 100): %.2f%%", episode, avg_reward, success_rate_last100)
            
            # Сохранение чекпойнта каждые 3000 эпизодов (без изменений)
            if episode % 3000 == 0:
                checkpoint = {
                    "episode": episode,
                    "policy_net_state_dict": policy_net.state_dict(),
                    "target_net_state_dict": target_net.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "config": config,
                    "seed": seed,
                }
                checkpoint_path = os.path.join(checkpoint_dir, f"exp_{experiment_number}_checkpoint_ep{episode}.pth")
                torch.save(checkpoint, checkpoint_path)
                logging.info("Checkpoint saved at episode %d in %s", episode, checkpoint_path)
            
            # Тестирование каждые 3000 эпизодов (без повторного сохранения чекпойнта)
            if episode % test_interval == 0:
                successful, total_maps, success_rate = test_current_model(policy_net, config["MAP_SIZE"], device)
                logging.info("Test at episode %d: %d/%d maps, success rate: %.2f%%", episode, successful, total_maps, success_rate)
                writer.add_scalar("Test/success_rate", success_rate, episode)
                test_writer.writerow([experiment_number, seed, episode, successful, total_maps, success_rate])
                
            if avg_success_all > best_overall_success:
                best_overall_success = avg_success_all

    except KeyboardInterrupt:
        logging.info("Experiment %d interrupted by user", experiment_number)
    finally:
        writer.close()
    
    logging.info("Training finished. Best overall success rate: %.2f%%", best_overall_success)
    return best_overall_success