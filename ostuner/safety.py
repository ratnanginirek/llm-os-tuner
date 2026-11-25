"""Safety layer: every proposal passes through here before touching the kernel.

Layers (each one is an ablation in the paper):
  1. Allowlist   -- only knobs in CATALOG, only ones present on this host.
  2. Bounds      -- per-knob ranges / choices.
  3. Cross-knob  -- relational constraints (e.g. dirty_background < dirty).
  4. Step limit  -- at most `max_changes` knobs changed per trial (blast radius).
  5. Health gate -- after applying, check the host is healthy before measuring.
  6. Rollback    -- revert to the best known-good config on regression/failure.
"""

from __future__ import annotations

from dataclasses import dataclass

from .knobs import CATALOG, cross_constraints


@dataclass
class Verdict:
    ok: bool
    errors: list[str]
    effective: dict  # full config after merging the proposal onto `current`


def validate(proposal: dict, current: dict, max_changes: int | None = 3) -> Verdict:
    errors: list[str] = []
    effective = dict(current)
    changed = 0
    for name, value in proposal.items():
        knob = CATALOG.get(name)
        if knob is None:
            errors.append(f"{name}: not in the allowlist")
            continue
        if name not in current:
            errors.append(f"{name}: not available on this host")
            continue
        err = knob.validate(value)
        if err:
            errors.append(err)
            continue
        if current[name] != value:
            changed += 1
        effective[name] = value
    errors += cross_constraints(effective)
    if max_changes is not None and changed > max_changes:
        errors.append(f"changes {changed} knobs; limit is {max_changes} per trial")
    return Verdict(ok=not errors, errors=errors, effective=effective)


@dataclass
class HealthPolicy:
    min_mem_available_frac: float = 0.05
    max_psi_memory_some_avg10: float = 50.0
    max_psi_io_some_avg10: float = 80.0


def health_check(telemetry: dict, policy: HealthPolicy | None = None) -> list[str]:
    """Return a list of problems; empty list means healthy. Missing metrics are
    skipped rather than treated as failures (not every kernel exposes PSI)."""
    p = policy or HealthPolicy()
    problems = []
    total, avail = telemetry.get("mem_total_kb"), telemetry.get("mem_available_kb")
    if total and avail is not None and avail / total < p.min_mem_available_frac:
        problems.append(f"MemAvailable {avail / total:.1%} < {p.min_mem_available_frac:.0%}")
    psi_mem = telemetry.get("psi_memory_some_avg10")
    if psi_mem is not None and psi_mem > p.max_psi_memory_some_avg10:
        problems.append(f"memory PSI avg10 {psi_mem} > {p.max_psi_memory_some_avg10}")
    psi_io = telemetry.get("psi_io_some_avg10")
    if psi_io is not None and psi_io > p.max_psi_io_some_avg10:
        problems.append(f"io PSI avg10 {psi_io} > {p.max_psi_io_some_avg10}")
    return problems
