from snake_env import SnakeEnv
import random

SEED = 1337

env = SnakeEnv(render_mode="human", size=8, seed=SEED)
terminated, truncated = False, False

random.seed(SEED)


def main():
    while True:
        action = random.randint(0, 2)
        observation, reward, terminated, truncated, info = env.step(action)
        print((observation, reward, terminated, truncated, info))
        
        if terminated or truncated:
            env.reset(seed=SEED)

    env.close()

if __name__ == "__main__":
    main()
