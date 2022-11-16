from snake_env import SnakeEnv
import random
from dqn_agent import DQNAgent

SEED = None
GRID_SIZE = 8

def main():
    env = SnakeEnv(render_mode="human", size=8, seed=SEED)
    agent = DQNAgent(env)#DQNAgent.from_file('dqn_model_100k.qbf')
    #agent.set_env(env)

env = SnakeEnv(render_mode='human', size=8, seed=SEED)
agent = DQNAgent(env)

agent.train()

# env = SnakeEnv(render_mode='human', size=8, seed=SEED)
# terminated, truncated = False, False

# while True:
#     action = random.choice([0,1,2])
#     observation, reward, terminated, truncated, info = env.step(action)
#     print((observation, reward, terminated, truncated, info))
#     if terminated or truncated:
#         env.reset(seed=SEED)

# env.close()
