import torch
import gym_super_mario_bros
from gym_super_mario_bros.actions import RIGHT_ONLY
from agent import Agent
from nes_py.wrappers import JoypadSpace
from wrappers import apply_wrappers
from save_trajectory import save_episode_data
import os
import re
from utils import save_episode_counter, load_episode_counter

# Параметры
model_dir = "models"
os.makedirs(model_dir, exist_ok=True)

# Параметры для обучения
ENV_NAME = 'SuperMarioBros-1-1-v0'
SHOULD_TRAIN = True
DISPLAY = True
CKPT_SAVE_INTERVAL = 5000
SAVE_EPISODE_COUNT = 5000
NUM_OF_EPISODES = 50_000

# Загружаем среду
env = gym_super_mario_bros.make(ENV_NAME, render_mode='human' if DISPLAY else 'rgb', apply_api_compatibility=True)
env = JoypadSpace(env, RIGHT_ONLY)
env = apply_wrappers(env)

# Создаём агента
agent = Agent(input_dims=env.observation_space.shape, num_actions=env.action_space.n)

# Функция для извлечения номера итерации из имени файла
def extract_number(model_name):
    match = re.search(r'model_(\d+)_iter\.pt', model_name)
    return int(match.group(1)) if match else 0

# Загружаем последнюю модель
model_files = [f for f in os.listdir(model_dir) if f.endswith('.pt')]
model_files.sort(key=extract_number)

if model_files:
    last_model = model_files[-1]
    print(f"Загружаем модель: {last_model}")
    agent.load_model(os.path.join(model_dir, last_model))

# Загружаем модель и продолжаем обучение
env.reset()
next_state, reward, done, trunc, info = env.step(action=0)

start_episode = load_episode_counter()
all_episodes_data = []

for i in range(start_episode, start_episode + NUM_OF_EPISODES):
    print(f"Episode: {i}")
    done = False
    state, _ = env.reset()
    total_reward = 0
    episode_data = []

    while not done:
        a = agent.choose_action(state)
        new_state, reward, done, truncated, info = env.step(a)
        total_reward += reward

        step_data = {
            "x_pos": info.get("x_pos"),
            "y_pos": info.get("y_pos"),
            "time": info.get("time"),
            "coins": info.get("coins"),
            "score": info.get("score"),
            "stage": info.get("stage"),
            "world": info.get("world"),
            "flag_get": info.get("flag_get"),
            "life": info.get("life"),
            "episode": i,
            "step": len(episode_data)
        }
        episode_data.append(step_data)

        if SHOULD_TRAIN:
            agent.store_in_memory(state, a, reward, new_state, done)
            agent.learn()

        state = new_state

    all_episodes_data.extend(episode_data)

    if (i + 1) % SAVE_EPISODE_COUNT == 0:
        save_episode_data(all_episodes_data, i)
        all_episodes_data = []

    print(f"Total reward: {total_reward}, Epsilon: {agent.epsilon}, Size of replay buffer: {len(agent.replay_buffer)}")

    if SHOULD_TRAIN and (i + 1) % CKPT_SAVE_INTERVAL == 0:
        ckpt_name = f"model_{i + 1}_iter.pt"
        checkpoint_path = os.path.join(model_dir, ckpt_name)
        agent.save_model(checkpoint_path)

    print(f"Total reward for episode {i}: {total_reward}")
    save_episode_counter(i + 1)

env.close()
