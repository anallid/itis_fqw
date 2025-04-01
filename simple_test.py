import os
import pandas as pd
import matplotlib.pyplot as plt
import torch
from agent import Agent
from gym_super_mario_bros import make
from gym_super_mario_bros.actions import RIGHT_ONLY
from nes_py.wrappers import JoypadSpace
from wrappers import apply_wrappers


# Функция для чтения данных из CSV
def read_csv_data(csv_path):
    return pd.read_csv(csv_path)


# Функция для извлечения среднего total_reward за эпизод
def calculate_avg_reward(csv_data):
    return csv_data.groupby('episode')['score'].sum().mean()


# Функция для построения графиков
def plot_rewards_comparison(before_rewards, after_rewards, label_before, label_after):
    plt.figure(figsize=(10, 6))
    plt.plot(before_rewards, label=label_before, color='blue')
    plt.plot(after_rewards, label=label_after, color='red')
    plt.xlabel('Episodes')
    plt.ylabel('Total Reward')
    plt.title('Agent Performance Comparison')
    plt.legend()
    plt.show()


# Функция для загрузки модели и тестирования агента
def test_agent(agent, env, num_episodes=10):
    total_reward = 0
    for _ in range(num_episodes):
        state, _ = env.reset()
        done = False
        episode_reward = 0
        while not done:
            action = agent.choose_action(state)
            new_state, reward, done, _, _ = env.step(action)
            episode_reward += reward
            state = new_state
        total_reward += episode_reward
    return total_reward / num_episodes


# Основная логика для сравнения моделей
def compare_agents(model_1_path, model_2_path, csv_path_1, csv_path_2):
    # Чтение данных CSV
    csv_data_1 = read_csv_data(csv_path_1)
    csv_data_2 = read_csv_data(csv_path_2)

    # Расчет средней награды для каждой модели
    avg_reward_before = calculate_avg_reward(csv_data_1)
    avg_reward_after = calculate_avg_reward(csv_data_2)

    print(f"Average Reward before (model_1): {avg_reward_before}")
    print(f"Average Reward after (model_2): {avg_reward_after}")

    # Построение графика сравнения награды
    plot_rewards_comparison(
        csv_data_1.groupby('episode')['score'].sum(),
        csv_data_2.groupby('episode')['score'].sum(),
        label_before=f'Model before: {model_1_path}',
        label_after=f'Model after: {model_2_path}'
    )


# Пример использования
if __name__ == "__main__":
    # Пути к сохраненным моделям и CSV
    model_1_path = "models/model_15000_iter.pt"
    model_2_path = "models/model_55000_iter.pt"
    csv_path_1 = "datasets/episode_14999.csv"
    csv_path_2 = "datasets/episode_54999.csv"

    # Сравнение моделей
    compare_agents(model_1_path, model_2_path, csv_path_1, csv_path_2)
