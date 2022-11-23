from snake_env import SnakeEnv
from qtable import SnakeQLearningAgent
from discretizer import FullDiscretizer, QuadDiscretizer, AngularDiscretizer

SEED = None
GRID_SIZE = 8

learning_env = SnakeEnv(render_mode=None, size=GRID_SIZE, seed=SEED)
render_env = SnakeEnv(render_mode="human", size=GRID_SIZE, seed=SEED)
render_obs = render_env.reset()

terminated, truncated = False, False

#agent = SnakeQLearningAgent(FullDiscretizer(GRID_SIZE))
agent = SnakeQLearningAgent(QuadDiscretizer(GRID_SIZE, 1))
#agent = SnakeQLearningAgent(AngularDiscretizer(GRID_SIZE, 16))

def try_render_once():
    global render_obs
    if render_env.can_render:
        render_env.death_counter = learning_env.death_counter
        action = agent.get_optimal_action(render_obs)
        render_obs, _, terminated, truncated, info = render_env.step(action)
        
        if terminated or truncated:
            render_env.reset()

def main():
    observation = learning_env.reset()
    reward = 0
    
    while True:
        try_render_once()
        
        action = agent.update(observation, reward)
        observation, reward, terminated, truncated, info = learning_env.step(action)
        
        if terminated or truncated:
            learning_env.reset()

if __name__ == "__main__":
    main()
