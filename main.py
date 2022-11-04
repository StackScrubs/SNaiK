from snake_env import SnakeEnv
import random

env = SnakeEnv(render_mode='human', size=8)
terminated, truncated = False, False

while True:
    action = random.choice([0,1,2])
    observation, reward, terminated, truncated, info = env.step(action)
    print((observation, reward, terminated, truncated, info))
    if terminated or truncated:
        print("**RESET**")
        env.reset()

env.close()