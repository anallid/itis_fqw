import gym
import gym_super_mario_bros
from gym.wrappers import GrayScaleObservation
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
import random

# Инициализация среды
env = gym.make("SuperMarioBros-v0", apply_api_compatibility=True, render_mode="human")
env = JoypadSpace(env, SIMPLE_MOVEMENT)

# Параметры epsilon-greedy
epsilon = 0.1  # вероятность случайного действия


# Функция для выбора оптимального действия (для обучения с подкреплением)
def choose_optimal_action(state):
    # В данном примере просто случайный выбор из возможных действий
    return env.action_space.sample()


# Игровой цикл
done = False
state, info = env.reset()
total_reward = 0

# Количество эпизодов
for episode in range(1000):
    state, info = env.reset()
    done = False
    episode_reward = 0

    while not done:
        # Выбор действия с использованием epsilon-greedy
        if random.random() < epsilon:
            action = env.action_space.sample()  # случайное действие
        else:
            action = choose_optimal_action(state)  # выбираем оптимальное действие (здесь случайное)

        # Выполнение действия и получение следующего состояния
        next_state, reward, done, truncated, info = env.step(action)
        episode_reward += reward

        state = next_state

    total_reward += episode_reward
    print(f"Episode {episode} finished with reward {episode_reward}")

env.close()
