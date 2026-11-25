"""LLM tuning agent.

The model sees the knob catalog (with bounds), the workload, host telemetry,
and the trial history, and replies with JSON: the next config plus a short
rationale. Everything it proposes still goes through `safety.validate`, so a
hallucinated knob or out-of-range value is rejected, never applied.

Clients:
  - AnthropicClient: Claude models via the `anthropic` SDK (pip install .[llm]).
  - OpenAICompatClient: any /v1/chat/completions endpoint (vLLM, Ollama,
    llama.cpp server, OpenAI) -- lets you compare open-weight models.
"""

from __future__ import annotations

import json
import os
import re
import urllib.request
from collections.abc import Callable

from .base import Agent, Context
from .classic import HillClimbAgent

LLMClient = Callable[[str, str], str]  # (system, user) -> text

SYSTEM_PROMPT = """You are an expert Linux performance engineer tuning kernel \
parameters for a specific workload. You run an iterative experiment: each turn \
you propose ONE configuration, it is applied and benchmarked, and you see the \
result next turn.

Rules:
- Only use knobs from the provided catalog, within their stated bounds.
- Change at most {max_changes} knobs relative to the CURRENT config per turn.
- Prefer changes with a clear mechanistic reason for this workload.
- Benchmarks are noisy; differences under ~2% may be noise.
- Reply with ONLY a JSON object, no prose outside it:
  {{"changes": {{"<knob>": <value>, ...}}, "rationale": "<one or two sentences>"}}
"""


def build_prompt(ctx: Context, include_telemetry: bool = True,
                 include_history: bool = True) -> str:
    lines = [
        f"WORKLOAD: {ctx.workload_name} -- {ctx.workload_description}",
        f"Metric unit: {ctx.unit}; {'higher' if ctx.direction == 'max' else 'lower'} is better.",
        f"Trials remaining: {ctx.budget_left}",
        "",
        "KNOB CATALOG:",
        *[f"- {k.describe()}" for k in ctx.knobs],
        "",
        "CURRENT CONFIG (= best so far; non-improving trials are reverted):",
        json.dumps(ctx.current, sort_keys=True),
    ]
    if include_telemetry and ctx.telemetry:
        lines += ["", "HOST TELEMETRY (latest):", json.dumps(ctx.telemetry, sort_keys=True)]
    if include_history and ctx.history:
        lines += ["", "HISTORY (most recent last):"]
        base_trial = next((t for t in ctx.history if t.status == "measured"), None)
        for t in ctx.history[-15:]:
            if t.status == "measured" and base_trial and base_trial.raw:
                delta = (t.raw / base_trial.raw - 1) * 100
                res = f"{t.raw:.1f} {ctx.unit} ({delta:+.1f}% vs first)"
            else:
                res = f"{t.status.upper()}: {'; '.join(t.errors)[:200]}"
            changed = {k: v for k, v in t.config.items() if ctx.baseline.get(k) != v}
            diff = json.dumps(changed, sort_keys=True)
            lines.append(f"#{t.index} diff-from-baseline={diff} -> {res}")
    if ctx.best:
        lines += ["", f"BEST SO FAR: trial #{ctx.best.index}, {ctx.best.raw:.1f} {ctx.unit}"]
    lines += ["", "Propose the next configuration as JSON."]
    return "\n".join(lines)


_JSON_OBJ = re.compile(r"\{.*\}", re.DOTALL)


def parse_reply(text: str) -> tuple[dict, str]:
    m = _JSON_OBJ.search(text)
    if not m:
        raise ValueError("no JSON object in reply")
    obj = json.loads(m.group(0))
    changes = obj.get("changes", obj.get("config"))
    if not isinstance(changes, dict):
        raise ValueError("reply JSON has no 'changes' object")
    return changes, str(obj.get("rationale", ""))


class LLMAgent(Agent):
    name = "llm"

    def __init__(self, client: LLMClient, include_telemetry: bool = True,
                 include_history: bool = True, retries: int = 1, seed: int = 0):
        self.client = client
        self.include_telemetry = include_telemetry
        self.include_history = include_history
        self.retries = retries
        self.fallback = HillClimbAgent(seed=seed)
        self.stats = {"calls": 0, "parse_failures": 0, "fallbacks": 0}

    def propose(self, ctx: Context):
        system = SYSTEM_PROMPT.format(max_changes=ctx.max_changes or "any number of")
        user = build_prompt(ctx, self.include_telemetry, self.include_history)
        last_err = ""
        for _ in range(self.retries + 1):
            prompt = user if not last_err else (
                f"{user}\n\nYour previous reply could not be parsed ({last_err}). "
                "Reply with only the JSON object.")
            self.stats["calls"] += 1
            try:
                changes, rationale = parse_reply(self.client(system, prompt))
                return {**ctx.current, **changes}, rationale
            except (ValueError, json.JSONDecodeError) as e:
                self.stats["parse_failures"] += 1
                last_err = str(e)
        self.stats["fallbacks"] += 1
        prop, _ = self.fallback.propose(ctx)
        return prop, f"[fallback to hillclimb after parse failure: {last_err}]"


class AnthropicClient:
    def __init__(self, model: str = "claude-sonnet-5-5", max_tokens: int = 1024,
                 temperature: float = 0.2):
        import anthropic  # optional dependency
        self._client = anthropic.Anthropic()
        self.model, self.max_tokens, self.temperature = model, max_tokens, temperature

    def __call__(self, system: str, user: str) -> str:
        msg = self._client.messages.create(
            model=self.model, max_tokens=self.max_tokens, temperature=self.temperature,
            system=system, messages=[{"role": "user", "content": user}])
        return "".join(b.text for b in msg.content if getattr(b, "type", "") == "text")


class OpenAICompatClient:
    def __init__(self, model: str, base_url: str | None = None,
                 api_key: str | None = None, temperature: float = 0.2):
        self.model = model
        self.base_url = (base_url or os.environ.get("OPENAI_BASE_URL",
                                                     "http://localhost:8000/v1")).rstrip("/")
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "none")
        self.temperature = temperature

    def __call__(self, system: str, user: str) -> str:
        body = json.dumps({
            "model": self.model, "temperature": self.temperature,
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": user}],
        }).encode()
        req = urllib.request.Request(
            f"{self.base_url}/chat/completions", data=body,
            headers={"Content-Type": "application/json",
                     "Authorization": f"Bearer {self.api_key}"})
        with urllib.request.urlopen(req, timeout=120) as resp:
            return json.loads(resp.read())["choices"][0]["message"]["content"]
