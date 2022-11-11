from snake_env import SnakeEnv
from qtable import SnakeQLearningAgent
import random

SEED = 1337
GRID_SIZE = 8

env = SnakeEnv(render_mode="human", size=GRID_SIZE, seed=SEED)
terminated, truncated = False, False

random.seed(SEED)

agent = SnakeQLearningAgent(GRID_SIZE)

def main():
    observation = env.reset()
    i = 0
    reward = 0
    while True:
        action = agent.update(observation, reward)
        observation, reward, terminated, truncated, info = env.step(action)
        print((observation, reward, terminated, truncated, info))
        
        if terminated or truncated:
            env.reset()

    env.close()

if __name__ == "__main__":
    main()
