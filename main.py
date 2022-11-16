from snake_env import SnakeEnv
from qtable import SnakeQLearningAgent

SEED = None
GRID_SIZE = 6

learning_env = SnakeEnv(render_mode=None, size=GRID_SIZE, seed=SEED)
render_env = SnakeEnv(render_mode="human", size=GRID_SIZE, seed=SEED)
render_obs = render_env.reset()

terminated, truncated = False, False

agent = SnakeQLearningAgent(GRID_SIZE)

def render_once():
    global render_obs
    if render_env.can_render:
        render_env.death_counter = learning_env.death_counter
        action = agent.get(render_obs)
        render_obs, _, terminated, truncated, info = render_env.step(action)
        
        if terminated or truncated:
            render_env.reset()

def main():
    observation = learning_env.reset()
    reward = 0
    while True:
        render_once()
        
        action = agent.update(observation, reward)
        observation, reward, terminated, truncated, info = learning_env.step(action)
        
        if terminated or truncated:
            learning_env.reset()

    learning_env.close()

if __name__ == "__main__":
    main()
