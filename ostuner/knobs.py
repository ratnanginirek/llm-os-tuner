"""Catalog of Linux kernel knobs the tuner is allowed to touch.

Every knob has explicit bounds. Anything not in this catalog cannot be
changed by any agent -- this is the first layer of the safety model.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Knob:
    name: str
    path: str  # relative to the filesystem root, e.g. "proc/sys/vm/swappiness"
    kind: str  # "int" or "choice"
    description: str
    min: int | None = None
    max: int | None = None
    choices: tuple[str, ...] = field(default_factory=tuple)
    risk: str = "low"  # "low" | "medium"; informs the LLM prompt and step limits

    def validate(self, value) -> str | None:
        """Return an error message, or None if the value is acceptable."""
        if self.kind == "int":
            if isinstance(value, bool) or not isinstance(value, int):
                return f"{self.name}: expected int, got {value!r}"
            if self.min is not None and value < self.min:
                return f"{self.name}: {value} < min {self.min}"
            if self.max is not None and value > self.max:
                return f"{self.name}: {value} > max {self.max}"
            return None
        if self.kind == "choice":
            if value not in self.choices:
                return f"{self.name}: {value!r} not in {list(self.choices)}"
            return None
        return f"{self.name}: unknown knob kind {self.kind}"

    def describe(self) -> str:
        if self.kind == "int":
            rng = f"int in [{self.min}, {self.max}]"
        else:
            rng = "one of " + ", ".join(self.choices)
        return f"{self.name} ({rng}; risk={self.risk}): {self.description}"


CATALOG: dict[str, Knob] = {
    k.name: k
    for k in [
        Knob("vm.swappiness", "proc/sys/vm/swappiness", "int",
             "Preference for swapping anonymous memory vs. dropping page cache.",
             min=0, max=100),
        Knob("vm.dirty_ratio", "proc/sys/vm/dirty_ratio", "int",
             "% of memory that may be dirty before writers are throttled.",
             min=5, max=60, risk="medium"),
        Knob("vm.dirty_background_ratio", "proc/sys/vm/dirty_background_ratio", "int",
             "% of memory dirty before background writeback starts. "
             "Must be < dirty_ratio.",
             min=1, max=30),
        Knob("vm.vfs_cache_pressure", "proc/sys/vm/vfs_cache_pressure", "int",
             "Tendency to reclaim dentry/inode caches relative to page cache.",
             min=10, max=500),
        Knob("kernel.numa_balancing", "proc/sys/kernel/numa_balancing", "int",
             "Automatic NUMA page migration (0=off, 1=on).",
             min=0, max=1, risk="medium"),
        Knob("kernel.sched_autogroup_enabled", "proc/sys/kernel/sched_autogroup_enabled", "int",
             "Group tasks by session for CFS/EEVDF fairness (0=off, 1=on).",
             min=0, max=1),
        Knob("net.core.somaxconn", "proc/sys/net/core/somaxconn", "int",
             "Max listen() backlog.",
             min=128, max=65535),
        Knob("net.core.netdev_max_backlog", "proc/sys/net/core/netdev_max_backlog", "int",
             "Max packets queued on input when the NIC receives faster than the "
             "kernel processes them.",
             min=1000, max=250000),
        Knob("mm.transparent_hugepage", "sys/kernel/mm/transparent_hugepage/enabled", "choice",
             "Transparent huge pages policy.",
             choices=("always", "madvise", "never"), risk="medium"),
        Knob("mm.transparent_hugepage_defrag",
             "sys/kernel/mm/transparent_hugepage/defrag", "choice",
             "THP defragmentation policy.",
             choices=("always", "defer", "defer+madvise", "madvise", "never"), risk="medium"),
    ]
}


def cross_constraints(config: dict) -> list[str]:
    """Constraints that involve more than one knob. `config` is the full
    effective configuration (current values merged with proposed changes)."""
    errs = []
    bg, fg = config.get("vm.dirty_background_ratio"), config.get("vm.dirty_ratio")
    if isinstance(bg, int) and isinstance(fg, int) and bg >= fg:
        errs.append(f"vm.dirty_background_ratio ({bg}) must be < vm.dirty_ratio ({fg})")
    return errs
