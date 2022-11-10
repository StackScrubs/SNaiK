from snake_env import SnakeEnv
import random

SEED = 1337

env = SnakeEnv(render_mode="human", size=8, seed=SEED)
terminated, truncated = False, False

random.seed(SEED)


def main():
    deds = 0
    best_reward = -100000000000
    while True:
        action = random.randint(0, 2)
        observation, reward, terminated, truncated, info = env.step(action)
        # print((observation, reward, terminated, truncated, info))
        
        if terminated or truncated:
            if reward > best_reward:
                print(f"{deds} REWARD {reward}")
                best_reward = reward
            deds += 1
            env.reset(seed=SEED)

    env.close()


if __name__ == "__main__":
    main()
