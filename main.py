from snake_env import SnakeEnv
from dqn_agent import DQNAgent
import random

env = SnakeEnv(render_mode='human', size=8)
terminated, truncated = False, False

agent = DQNAgent()

while True:
    # Select action (random or from QTable)
    action = agent.act()

    # Execute action and recieve output
    observation, reward, terminated, truncated, info = env.step(action)

    print((observation, reward, terminated, truncated, info))
    if terminated or truncated:
        print("**RESET**")
        env.reset()

    # agent.memorize(...)

    # agent.replay(...)


env.close()
