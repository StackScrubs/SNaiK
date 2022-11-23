from snake_env import SnakeEnv
from qtable import SnakeQLearningAgent
import click
from utils.ctx import CLIContext, AgentContext, EnvironmentContext
from utils.option_handlers import RequiredByWhenSetTo, OneOf
from pynput import keyboard
from discretizer import DISCRETIZER_TYPES, FullDiscretizer, QuadDiscretizer, AngularDiscretizer

@click.group()
@click.option("-a", "--alpha", type=click.FloatRange(0, 1, min_open=True, max_open=True), required=False, default=0.1)
@click.option("-g", "--gamma", type=click.FloatRange(0, 1, min_open=True, max_open=True), required=False, default=0.9)
@click.option("-sz","--size", type=int, required=False, default=4)
@click.option("-r", "--render", required=False, is_flag=True)
@click.option("-s", "--seed", type=int, required=False, default=None)
@click.pass_context
def entry(ctx, alpha, gamma, size, render, seed):
    print("SNaiK 1.0")
    print(f"Running with parameters:")
    print(f"ALPHA={alpha}")
    print(f"GAMMA={gamma}")
    print(f"SIZE={size}")
    print(f"RENDER={render}")
    print(f"SEED={seed}")
    
    agent_ctx = AgentContext(alpha, gamma, size)
    environment_ctx = EnvironmentContext(size, render, seed)
    
    ctx.obj = CLIContext(agent_ctx, environment_ctx)

@entry.command()
@click.option("-d", "--discretizer", type=click.Choice(DISCRETIZER_TYPES), required=False, cls=OneOf, other="file")
@click.option("-f", "--file", type=str, required=False, cls=OneOf, other="discretizer")
@click.option("-ns", "--n-sectors", type=int, required=False, cls=RequiredByWhenSetTo, required_by="discretizer", set_to=DISCRETIZER_TYPES.ANGULAR.value)
@click.option("-qs", "--quad-size", type=int, required=False, cls=RequiredByWhenSetTo, required_by="discretizer", set_to=DISCRETIZER_TYPES.QUAD.value)
@click.pass_obj
def qlearning(ctx, discretizer, file, n_sectors, quad_size):
    print("TYPE=Q LEARNING")
    agent = None
    
    if file:
        print(f"FROM FILE={file}")
        agent = SnakeQLearningAgent.from_file(file)
    
    elif discretizer:
        print(f"DISCRETIZER={discretizer}")
        discretizer_obj = FullDiscretizer(ctx.agent_ctx.size)

        if discretizer is DISCRETIZER_TYPES.ANGULAR:
            print(f"N SECTORS={n_sectors}")
            discretizer_obj = AngularDiscretizer(ctx.agent_ctx.size, n_sectors)
            
        elif discretizer is DISCRETIZER_TYPES.QUAD:
            print(f"QUAD SIZE={quad_size}")
            discretizer_obj = QuadDiscretizer(ctx.agent_ctx.size, quad_size)
            
        agent = SnakeQLearningAgent(discretizer_obj)

    main(agent, ctx.env_ctx)

@entry.command()
@click.option("-f", "--file", type=str, required=False)
@click.pass_obj
def dqn(ctx, file):
    print("TYPE=DQN")
    agent = None
    
    if file:
        #agent = DQNAgent.from_file(file)
        pass
    else:
        #agent = DQNAgent(ctx.agent_ctx)
        pass

    #main(agent, ctx.env_ctx)

class AgentRunner:
    def __init__(self, agent, env_ctx):
        self.agent = agent
        self.render = env_ctx.render

        self.learning_env = SnakeEnv(render_mode=None, size=env_ctx.size, seed=env_ctx.seed)
        if self.render:
            self.render_env = SnakeEnv(render_mode="human", size=env_ctx.size, seed=env_ctx.seed)
            self.render_obs = self.render_env.reset()

    def _try_render_once(self):
        if not self.render:
            return
            
        if self.render_env.can_render:
            self.render_env.death_counter = self.learning_env.death_counter
            action = self.agent.get_optimal_action(self.render_obs)
            self.render_obs, _, terminated, truncated, _ = self.render_env.step(action)
            
            if terminated or truncated:
                self.render_env.reset()

    def run(self):
        observation = self.learning_env.reset()
        reward = 0
        while True:
            if self.render:
                self._try_render_once()
            
            action = self.agent.update(observation, reward)
            observation, reward, terminated, truncated, _ = self.learning_env.step(action)
            
            if terminated or truncated:
                self.learning_env.reset()

def main(agent, env_ctx):
    
    def on_press(key):
        if not isinstance(key, keyboard.KeyCode):
            return

        if key.char == "s":
            print("Saving current model state...")
            file = agent.to_file()
            print(f"Saved model state as \"{file}\".")
            
        elif key.char == "g":
            print("Creating performance graph of current learning...")
            # ...
            print("Graph created and saved.")
        
        return
    
    listener = keyboard.Listener(
        on_press=on_press
    )
    listener.start()
    
    agent_runner = AgentRunner(agent, env_ctx)
    agent_runner.run()

if __name__ == "__main__":
    entry()
