from snake_env import SnakeEnv
from qtable import SnakeQLearningAgent
import click
from typing import Optional
from enum import Enum

class DISCRETIZER_TYPES(str, Enum):
    FULL = "full"
    QUAD = "quad"
    ANGULAR = "ang"

class RequiredByDiscretizer(click.Option):
    def __init__(self, *args, **kwargs):
        self.required_by = kwargs.pop("required_by")
        assert self.required_by, "'required_by' parameter required"
        kwargs["help"] = (kwargs.get("help", "") +
            f" NOTE: This argument is required when discretizer is set to {self.required_by}"
        ).strip()
        super(RequiredByDiscretizer, self).__init__(*args, **kwargs)

    def handle_parse_result(self, ctx, opts, args):
        self_present = self.name in opts
        required_by_present = opts["discretizer"] == self.required_by

        if required_by_present:
            if not self_present:
                raise click.UsageError(
                    f"Illegal usage: {self.name} is required when discretizer is set to {self.required_by}"
                )

        return super(RequiredByDiscretizer, self).handle_parse_result(ctx, opts, args)

class CLIContext:
    def __init__(self, alpha, gamma, size, render):
        self.alpha = alpha
        self.gamma = gamma
        self.size = size
        self.render = render

@click.group()
@click.option("-a", "--alpha", type=click.FloatRange(0, 1, min_open=True, max_open=True), required=False, default=0.1)
@click.option("-g", "--gamma", type=click.FloatRange(0, 1, min_open=True, max_open=True), required=False, default=0.9)
@click.option("-s", "--size", type=int, required=False, default=4)
@click.option("-r", "--render", required=False, is_flag=True)
@click.pass_context
def main(ctx, alpha, gamma, size, render):
    # print("SNaiK 1.0")
    # print(f"Running with parameters:")
    # print(f"ALPHA={alpha}")
    # print(f"GAMMA={gamma}")
    # print(f"SIZE={size}")
    # print(f"RENDER={render}")
    ctx.obj = CLIContext(alpha, gamma, size, render)

@main.command()
@click.argument("discretizer", type=click.Choice(DISCRETIZER_TYPES), required=True)
@click.option("-ns", "--n-sectors", type=int, required=False, cls=RequiredByDiscretizer, required_by=DISCRETIZER_TYPES.ANGULAR.value)
@click.option("-qs", "--quad-size", type=int, required=False, cls=RequiredByDiscretizer, required_by=DISCRETIZER_TYPES.QUAD.value)
@click.pass_obj
def qlearning(ctx, discretizer, n_sectors, quad_size):
    print("TYPE=Q LEARNING")
    print(f"DISCRETIZER={discretizer}")
    if discretizer is DISCRETIZER_TYPES.ANGULAR:
        print(f"N SECTORS={n_sectors}")
    if discretizer is DISCRETIZER_TYPES.QUAD:
        print(f"QUAD SIZE={quad_size}")

    print(ctx.alpha)

@main.command()
def dqn():
    print("TYPE=DQN")

if __name__ == "__main__":
    main()
