"""The tuning loop: propose -> validate -> apply -> health-check -> measure ->
keep or roll back. The host is always restored to its original config at
the end (or to the best config with keep_best=True), including on Ctrl-C
or an exception."""

from __future__ import annotations

import json
import logging
import statistics
import time
from pathlib import Path

from . import safety, telemetry
from .agents.base import Agent, Context, Trial
from .knobs import CATALOG
from .system import SystemBackend
from .workloads import Workload

log = logging.getLogger(__name__)


class Tuner:
    def __init__(self, backend: SystemBackend, workload: Workload, agent: Agent,
                 budget: int = 20, reps: int = 3, warmup: int = 0,
                 max_changes: int | None = 3, regression_tolerance: float = 0.10,
                 keep_best: bool = False, out_dir: str | Path | None = None,
                 health_policy: safety.HealthPolicy | None = None,
                 health_gate: bool = True, settle_s: float = 0.0,
                 validate_reps: int = 5):
        self.backend, self.workload, self.agent = backend, workload, agent
        self.budget, self.reps, self.warmup = budget, reps, warmup
        self.max_changes = max_changes
        self.tol = regression_tolerance
        self.keep_best = keep_best
        self.out_dir = Path(out_dir) if out_dir else None
        self.health_policy = health_policy
        self.health_gate = health_gate
        self.settle_s = settle_s
        self.validate_reps = validate_reps
        self.validation: dict | None = None
        self.history: list[Trial] = []
        self.best: Trial | None = None

    # -- helpers -----------------------------------------------------------
    def _measure(self) -> tuple[float, list[float]]:
        for _ in range(self.warmup):
            self.workload.run_once(self.backend)
        runs = [self.workload.run_once(self.backend) for _ in range(self.reps)]
        return statistics.median(runs), runs

    def _is_regression(self, raw: float) -> bool:
        b = self.best.raw
        if self.workload.direction == "max":
            return raw < b * (1 - self.tol)
        return raw > b * (1 + self.tol)

    def _record(self, trial: Trial) -> None:
        self.history.append(trial)
        if self.out_dir:
            with open(self.out_dir / "trials.jsonl", "a") as f:
                f.write(json.dumps(trial.to_dict()) + "\n")
        log.info("trial %d %s raw=%s %s", trial.index, trial.status,
                 f"{trial.raw:.2f}" if trial.raw is not None else "-",
                 "(rolled back)" if trial.rolled_back else "")

    def _rollback_to_best(self) -> None:
        self.backend.restore(self.best.config)

    def _validate(self, original: dict) -> dict:
        """Re-measure baseline vs. best, interleaved (A B A B ...), so the
        reported gain is not just the luckiest draw out of `budget` noisy
        trials. Paper numbers should come from here, not from best_raw."""
        a_runs, b_runs = [], []
        for _ in range(self.validate_reps):
            self.backend.restore(original)
            a_runs.append(self.workload.run_once(self.backend))
            self.backend.restore(self.best.config)
            b_runs.append(self.workload.run_once(self.backend))
        a, b = statistics.median(a_runs), statistics.median(b_runs)
        gain = (b / a - 1) if self.workload.direction == "max" else (1 - b / a)
        return {"baseline_runs": a_runs, "best_runs": b_runs,
                "baseline_median": a, "best_median": b, "improvement": gain}

    # -- main loop ---------------------------------------------------------
    def run(self) -> dict:
        if self.out_dir:
            self.out_dir.mkdir(parents=True, exist_ok=True)
            (self.out_dir / "trials.jsonl").write_text("")
        original = self.backend.snapshot()
        if self.out_dir:
            # Written BEFORE any change, so `ostuner restore` can recover after a crash.
            SystemBackend.save_snapshot(original, self.out_dir / "original_snapshot.json")
        knobs = [CATALOG[n] for n in original]
        t_start = time.time()
        try:
            t0 = time.time()
            raw, runs = self._measure()
            base = Trial(0, dict(original), {}, "measured", self.workload.score(raw), raw,
                         runs, "baseline", seconds=time.time() - t0)
            self.best = base
            self._record(base)

            for i in range(1, self.budget + 1):
                t0 = time.time()
                current = self.backend.snapshot()
                ctx = Context(self.workload.name, self.workload.description,
                              self.workload.unit, self.workload.direction, knobs,
                              current, original, self.best, list(self.history),
                              telemetry.collect(self.backend.root), self.max_changes,
                              self.budget - i + 1)
                proposal, rationale = self.agent.propose(ctx)
                verdict = safety.validate(proposal, current, self.max_changes)
                if not verdict.ok:
                    self._record(Trial(i, verdict.effective, proposal, "rejected",
                                       rationale=rationale, errors=verdict.errors,
                                       seconds=time.time() - t0))
                    continue

                self.backend.apply(verdict.effective)
                if self.settle_s:
                    time.sleep(self.settle_s)
                if self.health_gate:
                    problems = safety.health_check(telemetry.collect(self.backend.root),
                                                   self.health_policy)
                    if problems:
                        self._rollback_to_best()
                        self._record(Trial(i, verdict.effective, proposal, "unhealthy",
                                           rationale=rationale, errors=problems,
                                           rolled_back=True, seconds=time.time() - t0))
                        continue
                try:
                    raw, runs = self._measure()
                except Exception as e:
                    self._rollback_to_best()
                    self._record(Trial(i, verdict.effective, proposal, "error",
                                       rationale=rationale, errors=[str(e)],
                                       rolled_back=True, seconds=time.time() - t0))
                    continue

                trial = Trial(i, verdict.effective, proposal, "measured",
                              self.workload.score(raw), raw, runs, rationale)
                if trial.score > self.best.score:
                    self.best = trial
                else:
                    # Greedy accept: the host only ever stays on the best known
                    # config. `rolled_back` marks regressions beyond tolerance,
                    # i.e. the cases where a live system would have been hurt.
                    self._rollback_to_best()
                    trial.rolled_back = self._is_regression(raw)
                trial.seconds = time.time() - t0
                self._record(trial)

            if self.validate_reps and self.best.index != 0:
                self.validation = self._validate(original)
        finally:
            final = self.best.config if (self.keep_best and self.best) else original
            self.backend.restore(final)
            log.info("restored %s config", "best" if self.keep_best else "original")

        summary = self.summary(time.time() - t_start)
        if self.out_dir:
            (self.out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
        return summary

    def summary(self, elapsed: float) -> dict:
        base = self.history[0]
        measured = [t for t in self.history if t.status == "measured"]
        if self.workload.direction == "max":
            improvement = self.best.raw / base.raw - 1
        else:
            improvement = 1 - self.best.raw / base.raw
        return {
            "workload": self.workload.name,
            "agent": self.agent.name,
            "agent_stats": getattr(self.agent, "stats", {}),
            "budget": self.budget,
            "reps": self.reps,
            "baseline_raw": base.raw,
            "best_raw": self.best.raw,
            "best_trial": self.best.index,
            "improvement_search": improvement,
            "improvement": (self.validation or {}).get("improvement", 0.0),
            "validation": self.validation,
            "best_changes": {k: v for k, v in self.best.config.items()
                             if base.config.get(k) != v},
            "counts": {s: sum(t.status == s for t in self.history)
                       for s in ("measured", "rejected", "unhealthy", "error")},
            "rollbacks": sum(t.rolled_back for t in self.history),
            "measured_trials": len(measured),
            "elapsed_s": elapsed,
        }
