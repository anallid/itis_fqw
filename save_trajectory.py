import os
import pandas as pd

# Создадим папку для хранения датасетов
DATASET_DIR = "datasets"
os.makedirs(DATASET_DIR, exist_ok=True)

def save_episode_data(episode_data, episode_num):
    """Сохраняет данные одного эпизода в CSV."""
    filename = f"episode_{episode_num}.csv"
    filepath = os.path.join(DATASET_DIR, filename)

    df = pd.DataFrame(episode_data)
    df.to_csv(filepath, index=False)
    print(f"Сохранено: {filepath}")

