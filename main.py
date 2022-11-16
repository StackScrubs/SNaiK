from snake_env import SnakeEnv
import random
from dqn_agent import DQNAgent

SEED = 1337
GRID_SIZE = 8

#learning_env = SnakeEnv(render_mode=None, size=GRID_SIZE, seed=SEED)
#render_env = SnakeEnv(render_mode="human", size=GRID_SIZE, seed=SEED)
#render_obs = render_env.reset()

#terminated, truncated = False, False

#agent = DQNAgent(learning_env)

# def try_render_once():
#     global render_obs
#     if render_env.can_render:
#         render_env.death_counter = learning_env.death_counter
#         action = agent.get_optimal_action(render_obs)
#         render_obs, _, terminated, truncated, info = render_env.step(action)
        
#         if terminated or truncated:
#             render_env.reset()

# def main():
#     agent.train()

def main():
    env = SnakeEnv(render_mode="human", size=GRID_SIZE, seed=SEED)
    agent = DQNAgent(env)#DQNAgent.from_file('dqn_model_1668613640.807945.qbf')
    agent.set_env(env)

env = SnakeEnv(render_mode='human', size=8, seed=SEED)
agent = DQNAgent(env)

    #agent.to_file()

# env = SnakeEnv(render_mode='human', size=8, seed=SEED)
# terminated, truncated = False, False

# while True:
#     action = random.choice([0,1,2])
#     observation, reward, terminated, truncated, info = env.step(action)
#     print((observation, reward, terminated, truncated, info))
#     if terminated or truncated:
#         env.reset(seed=SEED)

# env.close()
