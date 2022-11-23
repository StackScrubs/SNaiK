from snake_env import SnakeEnv
import random
from dqn_agent import DQNAgent
from tqdm import tqdm

SEED = 1337
GRID_SIZE = 8

learning_env = SnakeEnv(render_mode=None, size=GRID_SIZE, seed=SEED)
render_env = SnakeEnv(render_mode="human", size=GRID_SIZE, seed=SEED)
render_obs = render_env.reset()

terminated, truncated = False, False

agent = DQNAgent(learning_env.size)

def try_render_once():
    global render_obs
    if render_env.can_render:
        render_env.death_counter = learning_env.death_counter
        action = agent.get_optimal_action(render_obs)
        render_obs, _, terminated, truncated, info = render_env.step(action)
        
        if terminated or truncated:
            render_env.reset()

def main():
    train_period = 1

    agent.experience_initial(learning_env)
    i = 0
    while True:
        try_render_once()
        
        agent.experience_replay(learning_env)

        if i % train_period == 0:
            agent.train_q_network()

        if i % agent.T == 0:
            agent.copy_q_to_target()

        i += 1
        if i % 1000 == 0:
            print(i)

if __name__ == "__main__":
    main()
