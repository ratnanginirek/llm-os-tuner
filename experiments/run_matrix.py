#!/usr/bin/env python3
"""Run every (workload x agent x seed) combination from a YAML config.

    sudo python3 experiments/run_matrix.py experiments/configs/main.yaml
    python3 experiments/run_matrix.py experiments/configs/smoke.yaml   # no root

Already-finished runs (summary.json exists) are skipped, so the script can be
re-run after an interruption. If a run crashes, the host is restored by the
tuner; if the whole machine dies, use `ostuner restore <run>/original_snapshot.json`.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

from ostuner.cli import main as ostuner_main


def agent_label(a: dict) -> str:
    label = a.get("label") or a["name"]
    return label.replace("/", "_")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--only-agent")
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    out_root = Path(cfg.get("out", "results")) / cfg["name"]
    common = cfg.get("common", {})

    failures = 0
    for wl in cfg["workloads"]:
        for agent in cfg["agents"]:
            if args.only_agent and agent_label(agent) != args.only_agent:
                continue
            for seed in cfg.get("seeds", [0]):
                out = out_root / wl / agent_label(agent) / f"seed{seed}"
                if (out / "summary.json").exists():
                    print(f"skip {out}")
                    continue
                argv = []
                if cfg.get("fake_root"):
                    argv.append("--fake-root")
                if cfg.get("dry_run"):
                    argv.append("--dry-run")
                argv += ["run", "--workload", wl, "--agent", agent["name"],
                         "--seed", str(seed), "--out", str(out)]
                opts = {**common, **{k: v for k, v in agent.items()
                                     if k not in ("name", "label")}}
                for k, v in opts.items():
                    flag = "--" + k.replace("_", "-")
                    if v is True:
                        argv.append(flag)
                    elif v is not False and v is not None:  # keep 0 (e.g. max_changes: 0)
                        argv += [flag, str(v)]
                print("ostuner", " ".join(argv), flush=True)
                try:
                    ostuner_main(argv)
                except Exception as e:  # keep going; record and continue
                    failures += 1
                    print(f"FAILED {out}: {e}", file=sys.stderr)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
