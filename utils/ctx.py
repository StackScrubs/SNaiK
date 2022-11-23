class CLIContext:
    def __init__(self, agent_ctx, env_ctx):
        self.agent_ctx = agent_ctx
        self.env_ctx = env_ctx

class AgentContext:
    def __init__(self, alpha, gamma, size):
        self.alpha = alpha
        self.gamma = gamma
        self.size = size

class EnvironmentContext:
    def __init__(self, size, render, seed):
        self.size = size
        self.render = render
        self.seed = seed
