"""Read and write kernel knobs, with snapshot/restore.

`root` is "/" on a real host. Tests (and dry runs on a laptop) point it at a
temporary directory that mimics /proc/sys and /sys, so the whole tuning loop
can be exercised without root privileges.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path

from .knobs import CATALOG, Knob

log = logging.getLogger(__name__)

_BRACKET = re.compile(r"\[([^\]]+)\]")


def parse_value(knob: Knob, raw: str):
    raw = raw.strip()
    if knob.kind == "choice":
        # sysfs reports e.g. "always [madvise] never"; the bracketed one is active.
        m = _BRACKET.search(raw)
        return m.group(1) if m else raw
    return int(raw.split()[0])


class SystemBackend:
    def __init__(self, root: str | Path = "/", dry_run: bool = False):
        self.root = Path(root)
        self.dry_run = dry_run
        self._dry_state: dict = {}

    def _path(self, knob: Knob) -> Path:
        return self.root / knob.path

    def available(self) -> list[str]:
        return [n for n, k in CATALOG.items() if self._path(k).exists()]

    def read(self, name: str):
        if name in self._dry_state:
            return self._dry_state[name]
        knob = CATALOG[name]
        return parse_value(knob, self._path(knob).read_text())

    def write(self, name: str, value) -> None:
        knob = CATALOG[name]
        err = knob.validate(value)
        if err:
            raise ValueError(err)
        if self.dry_run:
            log.info("[dry-run] %s = %s", name, value)
            self._dry_state[name] = value
            return
        self._path(knob).write_text(f"{value}\n")

    def apply(self, config: dict) -> None:
        for name, value in config.items():
            if self.read(name) != value:
                self.write(name, value)

    def snapshot(self) -> dict:
        return {n: self.read(n) for n in self.available()}

    def restore(self, snap: dict) -> None:
        failures = []
        for name, value in snap.items():
            try:
                if self.read(name) != value:
                    self.write(name, value)
            except Exception as e:  # keep restoring the rest
                failures.append(f"{name}: {e}")
        if failures:
            raise RuntimeError("restore incomplete: " + "; ".join(failures))

    @staticmethod
    def save_snapshot(snap: dict, path: str | Path) -> None:
        Path(path).write_text(json.dumps(snap, indent=2))

    @staticmethod
    def load_snapshot(path: str | Path) -> dict:
        return json.loads(Path(path).read_text())


def make_fake_root(root: str | Path, values: dict | None = None) -> Path:
    """Create a directory tree that looks like /proc/sys + /sys for tests."""
    root = Path(root)
    defaults = {
        "vm.swappiness": 60,
        "vm.dirty_ratio": 20,
        "vm.dirty_background_ratio": 10,
        "vm.vfs_cache_pressure": 100,
        "kernel.numa_balancing": 1,
        "kernel.sched_autogroup_enabled": 1,
        "net.core.somaxconn": 4096,
        "net.core.netdev_max_backlog": 1000,
        "mm.transparent_hugepage": "madvise",
        "mm.transparent_hugepage_defrag": "madvise",
    }
    defaults.update(values or {})
    for name, value in defaults.items():
        knob = CATALOG[name]
        p = root / knob.path
        p.parent.mkdir(parents=True, exist_ok=True)
        if knob.kind == "choice":
            p.write_text(" ".join(f"[{c}]" if c == value else c for c in knob.choices) + "\n")
        else:
            p.write_text(f"{value}\n")
    proc = root / "proc"
    (proc / "meminfo").write_text("MemTotal: 16000000 kB\nMemAvailable: 8000000 kB\n")
    (proc / "loadavg").write_text("0.50 0.40 0.30 1/200 1234\n")
    return root
