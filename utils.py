import time
import datetime
import os

def get_current_date_time_string():
    return datetime.datetime.now().strftime("%Y-%m-%d-%H_%M_%S")

def save_episode_counter(episode_num):
    with open("episode_counter.txt", "w") as f:
        f.write(str(episode_num))

def load_episode_counter():
    if os.path.exists("episode_counter.txt"):
        with open("episode_counter.txt", "r") as f:
            return int(f.read())
    return 0  # если файла нет, начинаем с 0


class Timer():
    def __init__(self):
        self.times = []

    def start(self):
        self.t = time.time()

    def print(self, msg=''):
        print(f"Time taken: {msg}", time.time() - self.t)

    def get(self):
        return time.time() - self.t
    
    def store(self):
        self.times.append(time.time() - self.t)

    def average(self):
        return sum(self.times) / len(self.times)
