from snake_env import SnakeEnv
from dqn_agent import DQNAgent
import random

SEED = 70

env = SnakeEnv(render_mode='human', size=8, seed=SEED)
terminated, truncated = False, False

agent = DQNAgent()

while True:
    # Select action (random or from QTable)
    action = agent.act()

    # Execute action and recieve output
    observation, reward, terminated, truncated, info = env.step(action)

    print((observation, reward, terminated, truncated, info))
    if terminated or truncated:
        env.reset(seed=SEED)

    # agent.memorize(...)

    # agent.replay(...)


env.close()
