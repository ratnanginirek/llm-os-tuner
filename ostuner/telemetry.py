"""Lightweight host telemetry from /proc. Used for health gating and as
context in the LLM prompt."""

from __future__ import annotations

from pathlib import Path

VMSTAT_FIELDS = ("pgmajfault", "pswpin", "pswpout", "thp_fault_alloc",
                 "nr_dirty", "nr_writeback", "pgscan_direct")
MEMINFO_FIELDS = {"MemTotal": "mem_total_kb", "MemAvailable": "mem_available_kb",
                  "Dirty": "dirty_kb", "SwapTotal": "swap_total_kb", "SwapFree": "swap_free_kb"}


def _read(path: Path) -> str | None:
    try:
        return path.read_text()
    except OSError:
        return None


def collect(root: str | Path = "/") -> dict:
    proc = Path(root) / "proc"
    t: dict = {}

    if (text := _read(proc / "meminfo")) is not None:
        for line in text.splitlines():
            key, _, rest = line.partition(":")
            if key in MEMINFO_FIELDS:
                t[MEMINFO_FIELDS[key]] = int(rest.split()[0])

    if (text := _read(proc / "loadavg")) is not None:
        parts = text.split()
        t["loadavg_1m"] = float(parts[0])

    for res in ("cpu", "memory", "io"):
        if (text := _read(proc / "pressure" / res)) is None:
            continue
        for line in text.splitlines():
            if line.startswith("some"):
                fields = dict(f.split("=") for f in line.split()[1:])
                t[f"psi_{res}_some_avg10"] = float(fields["avg10"])

    if (text := _read(proc / "vmstat")) is not None:
        for line in text.splitlines():
            k, _, v = line.partition(" ")
            if k in VMSTAT_FIELDS:
                t[f"vmstat_{k}"] = int(v)

    nodes = Path(root) / "sys/devices/system/node"
    if nodes.exists():
        t["numa_nodes"] = len([p for p in nodes.glob("node[0-9]*")])
    cpuinfo = _read(proc / "cpuinfo")
    if cpuinfo:
        t["cpus"] = cpuinfo.count("processor\t:")
    return t
