from __future__ import annotations
from typing_extensions import Self
from graphing import Grapher
from agent import Agent, QLearningAgent, RenderingAgentDecorator, RandomAgent
import click
from utils.context import Context
from utils.option_handlers import RequiredByWhenSetTo, OneOf
import asyncio
from pickle import dumps, loads
from aioconsole import ainput, aprint
from dataclasses import dataclass
from discretizer import DiscretizerType, FullDiscretizer, QuadDiscretizer, AngularDiscretizer

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
@click.option("-r", "--render", required=False, is_flag=True)
def load(file: str, render: bool):
    ac: AgentWithContext = AgentWithContext.from_file(file)
    ac.ctx.render = render
    
    show_welcome(ac.ctx)
    asyncio.run(ac.run())

@entry.group()
@click.option("-a", "--alpha", type=click.FloatRange(0, 1, min_open=True, max_open=True), required=False, default=0.1)
@click.option("-g", "--gamma", type=click.FloatRange(0, 1, min_open=True, max_open=True), required=False, default=0.9)
@click.option("-sz","--size", type=int, required=False, default=4)
@click.option("-r", "--render", required=False, is_flag=True)
@click.option("-s", "--seed", type=int, required=False, default=None)
@click.pass_context
def new(ctx, alpha, gamma, size, render, seed):
    ctx.obj = Context(alpha=alpha, gamma=gamma, size=size, render=render, seed=seed)
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
@click.pass_obj
def dqn(ctx, file):
    print("TYPE=DQN")
    agent = None
    
    if file:
        #agent = DQNAgent.from_file(file)
        pass
    else:
        #agent = DQNAgent(ctx.agent_context)
        pass

    #main(agent, ctx.env_ctx)

@dataclass(frozen=True)
class AgentWithContext:
    ctx: Context
    agent: Agent
    grapher = Grapher()
    
    @staticmethod
    def from_file(file_path) -> Self:
        with open(file_path, "rb") as f:
            return loads(f.read())
        
    def to_file(self, file_name) -> str:
        base_path = "."
        file_name = f"{base_path}/{file_name}.qbf"
        with open(file_name, "wb") as f:
            f.write(dumps(self))

        return file_name
    
    async def run(self):    
        pretty_agent = self.agent
        if self.ctx.render:
            pretty_agent = RenderingAgentDecorator(self.ctx.render_env, self.agent)
        agent_runner = AgentRunner(pretty_agent, self.grapher, self.ctx)
        await asyncio.gather(
            self.__parse_cmd(),
            agent_runner.run(),
        )
    
    async def __parse_cmd(self):
        from time import time
        while True:
            cmd = await ainput("Write 'save', 'graph' or 'info': ")
            if cmd == "save":
                print("Saving current model state...")
                file = self.to_file(time())
                print(f"Saved model state as \"{file}\".")
            elif cmd == "graph":
                print("Creating performance graph of current learning...")
                file = self.grapher.avg_score_graph(".", time(), self.info)
                print(f"Graph created and saved as \"{file}\".")
            elif cmd == "info":
                print(self.info)
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
        self.grapher = grapher
        self.env = ctx.env

    async def run(self):
        episode, score = 0, 0
        self.agent.initialize()
        while True:
            episode += 1
            score = self.agent.run_episode()
            
            self.grapher.update(episode, score)
            await asyncio.sleep(0)

if __name__ == "__main__":
    entry()
