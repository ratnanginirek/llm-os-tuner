"""Benchmarks the tuner optimizes for.

A workload returns one raw measurement per run. `direction` says whether
bigger is better ("max", e.g. throughput) or smaller is better ("min", e.g.
p99 latency). The tuner always maximizes `score()`, which flips the sign
for "min" workloads.
"""

from __future__ import annotations

import json
import random
import re
import shlex
import subprocess
import tempfile
from collections.abc import Callable
from dataclasses import dataclass, field

from .knobs import CATALOG


@dataclass
class Workload:
    name: str
    description: str
    unit: str
    direction: str = "max"  # "max" | "min"

    def run_once(self, backend) -> float:  # pragma: no cover - interface
        raise NotImplementedError

    def score(self, raw: float) -> float:
        return raw if self.direction == "max" else -raw


def _json_path(obj, path: str):
    for part in path.split("."):
        obj = obj[int(part)] if isinstance(obj, list) else obj[part]
    return float(obj)


@dataclass
class CommandWorkload(Workload):
    command: str = ""
    parser: Callable[[str], float] | None = None
    timeout_s: int = 600
    workdir: str | None = None

    def run_once(self, backend) -> float:
        workdir = self.workdir or tempfile.mkdtemp(prefix="ostuner-")
        cmd = self.command.format(workdir=shlex.quote(workdir))
        proc = subprocess.run(cmd, shell=True, capture_output=True, text=True,
                              timeout=self.timeout_s)
        if proc.returncode != 0:
            raise RuntimeError(f"{self.name} failed ({proc.returncode}): {proc.stderr[-500:]}")
        return self.parser(proc.stdout)


def regex_parser(pattern: str) -> Callable[[str], float]:
    rx = re.compile(pattern)

    def parse(out: str) -> float:
        m = rx.search(out)
        if not m:
            raise ValueError(f"metric pattern {pattern!r} not found in output")
        return float(m.group(1))
    return parse


def json_parser(path: str) -> Callable[[str], float]:
    def parse(out: str) -> float:
        return _json_path(json.loads(out), path)
    return parse


@dataclass
class SimulatedWorkload(Workload):
    """Analytic stand-in for a real benchmark. Lets you develop and test the
    tuning loop (and the LLM prompting) on any machine, without root.

    NOT for paper results -- every number in the paper must come from a real
    workload on real hardware.
    """
    optimum: dict = field(default_factory=dict)
    weights: dict = field(default_factory=dict)
    base: float = 1000.0
    noise: float = 0.01
    seed: int = 0

    def __post_init__(self):
        self._rng = random.Random(self.seed)

    def run_once(self, backend) -> float:
        penalty = 0.0
        for name, target in self.optimum.items():
            value = backend.read(name)
            w = self.weights.get(name, 1.0)
            if isinstance(target, str):
                penalty += w * (0.0 if value == target else 0.1)
            else:
                knob = CATALOG[name]
                d = (value - target) / max(knob.max - knob.min, 1)
                penalty += w * 0.1 * d ** 2
        raw = self.base * max(0.05, 1.0 - penalty)
        raw *= 1.0 + self._rng.gauss(0, self.noise)
        return raw if self.direction == "max" else 1e6 / raw


BUILTIN: dict[str, Callable[[], Workload]] = {
    "fio-seqwrite": lambda: CommandWorkload(
        name="fio-seqwrite",
        description="Buffered sequential 1M writes with final fsync (writeback-bound; "
                    "sensitive to dirty ratios).",
        unit="KiB/s", direction="max",
        command=("fio --name=seqwrite --directory={workdir} --rw=write --bs=1M --size=4G "
                 "--end_fsync=1 --output-format=json"),
        parser=json_parser("jobs.0.write.bw"),
    ),
    "fio-randread": lambda: CommandWorkload(
        name="fio-randread",
        description="Buffered 4k random reads, iodepth 32 (page-cache / reclaim sensitive).",
        unit="IOPS", direction="max",
        command=("fio --name=randread --directory={workdir} --rw=randread --bs=4k --size=2G "
                 "--ioengine=libaio --iodepth=32 --runtime=30 --time_based --output-format=json"),
        parser=json_parser("jobs.0.read.iops"),
    ),
    "sysbench-memory": lambda: CommandWorkload(
        name="sysbench-memory",
        description="Memory bandwidth, 1M blocks, 20G total (THP sensitive).",
        unit="MiB/s", direction="max",
        command="sysbench memory --memory-block-size=1M --memory-total-size=20G run",
        parser=regex_parser(r"\(([\d.]+) MiB/sec\)"),
    ),
    "sim-writeback": lambda: SimulatedWorkload(
        name="sim-writeback",
        description="SIMULATED writeback-heavy workload (development only).",
        unit="ops/s", direction="max",
        optimum={"vm.dirty_ratio": 40, "vm.dirty_background_ratio": 5,
                 "vm.swappiness": 10, "mm.transparent_hugepage": "madvise"},
        weights={"vm.dirty_ratio": 3.0, "vm.dirty_background_ratio": 2.0},
    ),
    "sim-memdb": lambda: SimulatedWorkload(
        name="sim-memdb",
        description="SIMULATED in-memory database, HANA-like (development only).",
        unit="tx/s", direction="max",
        optimum={"mm.transparent_hugepage": "never", "kernel.numa_balancing": 0,
                 "vm.swappiness": 1, "kernel.sched_autogroup_enabled": 0},
        weights={"mm.transparent_hugepage": 2.0, "kernel.numa_balancing": 1.5},
    ),
}


def from_config(cfg: dict) -> Workload:
    """Build a workload from a YAML/dict spec, or look up a builtin by name."""
    if "command" not in cfg:
        return BUILTIN[cfg["name"]]()
    if "json_path" in cfg:
        parser = json_parser(cfg["json_path"])
    else:
        parser = regex_parser(cfg["regex"])
    return CommandWorkload(
        name=cfg["name"], description=cfg.get("description", ""),
        unit=cfg.get("unit", ""), direction=cfg.get("direction", "max"),
        command=cfg["command"], parser=parser,
        timeout_s=cfg.get("timeout_s", 600), workdir=cfg.get("workdir"),
    )
