from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass
class Trial:
    index: int
    config: dict            # full effective config that was measured (or rejected)
    proposal: dict          # what the agent asked for
    status: str             # "measured" | "rejected" | "unhealthy" | "error"
    score: float | None = None          # higher is better (already sign-flipped)
    raw: float | None = None            # median raw measurement
    raw_runs: list[float] = field(default_factory=list)
    rationale: str = ""
    errors: list[str] = field(default_factory=list)
    rolled_back: bool = False
    seconds: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Context:
    """Everything an agent may look at when proposing the next config."""
    workload_name: str
    workload_description: str
    unit: str
    direction: str
    knobs: list            # list[Knob] available on this host
    current: dict          # config currently applied
    baseline: dict         # config at start
    best: Trial | None
    history: list[Trial]
    telemetry: dict
    max_changes: int | None
    budget_left: int


class Agent:
    name = "base"

    def propose(self, ctx: Context) -> tuple[dict, str]:
        """Return (proposal, rationale). The proposal maps knob name -> value
        and may contain only the knobs to change."""
        raise NotImplementedError
