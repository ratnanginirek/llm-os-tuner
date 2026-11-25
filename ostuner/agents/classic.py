"""Non-LLM baselines. A reviewer will ask "does the LLM beat simple search
with the same trial budget?" -- these answer that."""

from __future__ import annotations

import random

from ..knobs import Knob
from ..safety import validate
from .base import Agent, Context


def _sample(knob: Knob, rng: random.Random):
    if knob.kind == "choice":
        return rng.choice(knob.choices)
    return rng.randint(knob.min, knob.max)


def _neighbor(knob: Knob, value, rng: random.Random, scale: float = 0.2):
    if knob.kind == "choice":
        others = [c for c in knob.choices if c != value] or list(knob.choices)
        return rng.choice(others)
    span = knob.max - knob.min
    if span <= 1:
        return knob.max if value == knob.min else knob.min
    step = max(1, int(span * scale))
    return min(knob.max, max(knob.min, value + rng.randint(-step, step)))


def _valid(prop: dict, ctx: Context) -> bool:
    return validate(prop, ctx.current, ctx.max_changes).ok


class DefaultAgent(Agent):
    """Re-measures the starting configuration. Gives the noise floor."""
    name = "default"

    def propose(self, ctx: Context):
        return dict(ctx.baseline), "baseline configuration"


class RandomAgent(Agent):
    name = "random"

    def __init__(self, seed: int = 0):
        self.rng = random.Random(seed)

    def propose(self, ctx: Context):
        # Resample until the proposal passes the safety checks, so the
        # baseline isn't handicapped by wasting trials on rejections.
        n = min(ctx.max_changes or len(ctx.knobs), len(ctx.knobs))
        prop = dict(ctx.current)
        for _ in range(50):
            chosen = self.rng.sample(ctx.knobs, k=n)
            prop = {**ctx.current, **{k.name: _sample(k, self.rng) for k in chosen}}
            if _valid(prop, ctx):
                break
        return prop, "random sample"


class HillClimbAgent(Agent):
    """Perturb a few knobs around the best config seen so far."""
    name = "hillclimb"

    def __init__(self, seed: int = 0, scale: float = 0.2):
        self.rng = random.Random(seed)
        self.scale = scale

    def propose(self, ctx: Context):
        base = dict(ctx.best.config if ctx.best else ctx.baseline)
        prop, n = base, 0
        for _ in range(50):
            prop = dict(base)
            n = self.rng.randint(1, min(ctx.max_changes or 2, len(ctx.knobs)))
            for k in self.rng.sample(ctx.knobs, k=n):
                prop[k.name] = _neighbor(k, prop[k.name], self.rng, self.scale)
            if _valid(prop, ctx):
                break
        return prop, f"perturb {n} knob(s) around best"
