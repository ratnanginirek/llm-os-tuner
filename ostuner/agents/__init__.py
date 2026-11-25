from .base import Agent, Context, Trial
from .classic import DefaultAgent, HillClimbAgent, RandomAgent
from .llm import AnthropicClient, LLMAgent, OpenAICompatClient

__all__ = [
    "Agent",
    "AnthropicClient",
    "Context",
    "DefaultAgent",
    "HillClimbAgent",
    "LLMAgent",
    "OpenAICompatClient",
    "RandomAgent",
    "Trial",
    "make_agent",
]


def make_agent(name: str, seed: int = 0, model: str | None = None,
               provider: str = "anthropic", base_url: str | None = None,
               include_telemetry: bool = True, include_history: bool = True):
    if name == "default":
        return DefaultAgent()
    if name == "random":
        return RandomAgent(seed=seed)
    if name == "hillclimb":
        return HillClimbAgent(seed=seed)
    if name == "llm":
        if provider == "anthropic":
            client = AnthropicClient(model=model or "claude-sonnet-5-5")
        elif provider == "openai":
            client = OpenAICompatClient(model=model or "default", base_url=base_url)
        else:
            raise ValueError(f"unknown provider {provider}")
        return LLMAgent(client, include_telemetry=include_telemetry,
                        include_history=include_history, seed=seed)
    raise ValueError(f"unknown agent {name}")
