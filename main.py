from __future__ import annotations
from typing_extensions import Self
from agent import Agent, QLearningAgent, DQNAgent, RenderingAgentDecorator, RandomAgent
from graphing import Grapher
from agent import Agent, QLearningAgent, RenderingAgentDecorator, RandomAgent
from utils.context import Context, AgentContext
from utils.option_handlers import RequiredByWhenSetTo, OneOf
from pickle import dumps, loads
from aioconsole import ainput, aprint
from dataclasses import dataclass
from discretizer import DiscretizerType, FullDiscretizer, QuadDiscretizer, AngularDiscretizer
from dqn import ModelType, LinearDQN, ConvolutionalDQN

import click
import asyncio

VERSION = "1.0"

def show_welcome(ctx: Context):
    print(f"SNaiK {VERSION}")
    human_seed = ctx.seed if ctx.seed is not None else "random"
    print(f"Parameters: {ctx.size}x{ctx.size}, α={ctx.alpha}, γ={ctx.gamma}, seed={human_seed}")

@click.group()
def entry():
    pass

@entry.command()
@click.argument("file", type=str, required=True)
@click.option("-e", "--max-episodes", type=int, required=False, default=-1)
@click.option("-r", "--render", required=False, is_flag=True)
def load(file: str, max_episodes: int, render: bool):
    ac: AgentWithContext = AgentWithContext.from_file(file)
    ac.ctx.render = render
    ac.ctx.max_episodes = max_episodes
    
    show_welcome(ac.ctx)
    asyncio.run(ac.run())

@entry.group()
@click.option("-a", "--alpha", type=click.FloatRange(0, 1, min_open=True, max_open=True), required=False, default=0.1)
@click.option("-g", "--gamma", type=click.FloatRange(0, 1, min_open=True, max_open=True), required=False, default=0.9)
@click.option("-sz","--size", type=int, required=False, default=4)
@click.option("-e", "--max-episodes", type=int, required=False, default=-1)
@click.option("-r", "--render", required=False, is_flag=True)
@click.option("-s", "--seed", type=int, required=False, default=None)
@click.pass_context
def new(ctx, alpha, gamma, size, max_episodes, render, seed):
    ctx.obj = Context(alpha=alpha, gamma=gamma, size=size, max_episodes=max_episodes, render=render, seed=seed)
    show_welcome(ctx.obj)
    
@new.command()
@click.argument("discretizer", type=click.Choice(DiscretizerType), required=True)
@click.option("-ns", "--n-sectors", type=int, required=False, cls=RequiredByWhenSetTo, required_by="discretizer", set_to=DiscretizerType.ANGULAR.value)
@click.option("-qs", "--quad-size", type=int, required=False, cls=RequiredByWhenSetTo, required_by="discretizer", set_to=DiscretizerType.QUAD.value)
@click.pass_obj
def qlearning(ctx: Context, discretizer: str | None, n_sectors: int, quad_size: int):
    disc = ({
        DiscretizerType.FULL: lambda: FullDiscretizer(ctx.size),
        DiscretizerType.ANGULAR: lambda: AngularDiscretizer(ctx.size, n_sectors),
        DiscretizerType.QUAD: lambda: QuadDiscretizer(ctx.size, quad_size)
    })[discretizer]()
    agent = QLearningAgent(ctx.agent_context, disc)
    asyncio.run(AgentWithContext(ctx, agent).run())

@new.command()
@click.pass_obj
def random(ctx: Context):
    agent = RandomAgent(ctx.agent_context)
    asyncio.run(AgentWithContext(ctx, agent).run())

@new.command()
@click.argument("model", type=click.Choice(ModelType), required=True)
@click.pass_obj
def dqn(ctx: Context, model: str | None):
    nn_model = ({
        ModelType.LINEAR: lambda: LinearDQN(ctx.size),
        ModelType.CONVOLUTIONAL: lambda: ConvolutionalDQN(ctx.size)
    })[model]()
    agent = DQNAgent(ctx.agent_context, nn_model)
    asyncio.run(AgentWithContext(ctx, agent).run())

@dataclass(frozen=True)
class AgentWithContext:
    ctx: Context
    agent: Agent
    grapher = Grapher()

    
    @staticmethod
    def from_file(file_path) -> Self:
        with open(file_path, "rb") as f:
            return loads(f.read())
        
    def to_file(self) -> str:
        from time import time
        base_path = "."
        file_name = f"{base_path}/{time()}.qbf"
        with open(file_name, "wb") as f:
            f.write(dumps(self))

        return file_name
    
    async def run(self):
        pretty_agent = self.agent
        if self.ctx.render:
            pretty_agent = RenderingAgentDecorator(self.ctx.render_env, self.agent)
        agent_runner = AgentRunner(pretty_agent, self.grapher, self.ctx)
        await asyncio.gather(
            self.__parse_cmd(agent_runner),
            agent_runner.run(),
        )
    
    async def __parse_cmd(self, agent_runner: AgentRunner):
        while True:
            cmd = await ainput("Write 'save', 'info' or 'exit': ")
            if cmd == "save":
                await self.__parse_save_cmd()
            elif cmd == "info":
                print(agent_runner.info)
            elif cmd == "exit":
                print("Exiting...")
                agent_runner.stop()
                return
            else:
                await aprint(f"Invalid command '{cmd}'.")

    async def __parse_save_cmd(self):
        cmd = await ainput("Write 'model', 'graph', 'stats', or 'abort' to go back: ")
        if cmd == "stats":
            stats_cmd = await ainput("Write 'avg', 'best' or 'abort' to go back: ")
            if stats_cmd == "abort":
                return
            elif stats_cmd in ["avg", "best"]:
                print("Gathering stats from current learning...")
                file = self.grapher.save_stats(stats_cmd, self.agent.info)
                print(f"Stats saved as \"{file}\".")
            else:
                await aprint(f"Invalid command '{stats_cmd}'.")
        elif cmd == "graph":
            graph_cmd = await ainput("Write 'avg', 'best' or 'abort' to go back: ")
            if graph_cmd == "abort":
                return
            elif graph_cmd in ["avg", "best"]:
                print("Creating performance graph of current learning...")
                file = self.grapher.get_score_graph(graph_cmd, ".", self.info)
                print(f"Graph created and saved as \"{file}\".")
            else:
                await aprint(f"Invalid command '{graph_cmd}'.")
        elif cmd == "model":
            print("Saving current model state...")
            file = self.to_file()
            print(f"Saved model state as \"{file}\".")
        elif cmd == "abort":
            return
        else:
            await aprint(f"Invalid command '{cmd}'.")

    @property
    def info(self) -> dict:
        return {
            **self.ctx.info,
            **self.agent.info,
        }

class AgentRunner:    
    def __init__(self, agent: Agent, grapher: Grapher, ctx: Context):
        self.agent = agent
        self.ctx = ctx
        self.grapher = grapher
        self.env = ctx.env
        self.exit = False
        self.current_episode = 0

    def stop(self):
        self.exit = True

    async def run(self):
        self.agent.initialize()
        while not self.exit and self.current_episode != self.ctx.max_episodes:
            self.current_episode += 1
            score = self.agent.run_episode()
            
            self.grapher.update(score)
            await asyncio.sleep(0)

        print(f"\nStopped running episodes. Ran a total of {self.current_episode} episodes")
        
    @property
    def info(self) -> dict:
        return {
            **self.ctx.info,
            **self.agent.info,
            "current_episode": self.current_episode
        }

if __name__ == "__main__":
    entry()
