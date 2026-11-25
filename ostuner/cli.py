"""Command-line interface.

  ostuner knobs                       # list tunable knobs and current values
  ostuner run --workload sim-memdb --agent llm --budget 20 --out results/x
  ostuner restore results/x/original_snapshot.json
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import tempfile
from datetime import datetime

from . import telemetry
from .agents import make_agent
from .knobs import CATALOG
from .system import SystemBackend, make_fake_root
from .tuner import Tuner
from .workloads import BUILTIN, SimulatedWorkload


def _backend(args) -> SystemBackend:
    if args.fake_root:
        root = tempfile.mkdtemp(prefix="ostuner-fakeroot-")
        make_fake_root(root)
        logging.info("using fake root %s", root)
        return SystemBackend(root)
    return SystemBackend(args.root, dry_run=args.dry_run)


def cmd_knobs(args) -> int:
    be = _backend(args)
    snap = be.snapshot()
    for name, knob in CATALOG.items():
        cur = snap.get(name, "(not available)")
        print(f"{name:34} {cur!s:12} {knob.describe().split(': ', 1)[1]}")
    print("\ntelemetry:", json.dumps(telemetry.collect(be.root)))
    return 0


def cmd_run(args) -> int:
    be = _backend(args)
    if args.workload not in BUILTIN:
        print(f"unknown workload {args.workload}; choose from {sorted(BUILTIN)}",
              file=sys.stderr)
        return 2
    wl = BUILTIN[args.workload]()
    if isinstance(wl, SimulatedWorkload):
        wl.seed = args.seed
        wl.__post_init__()
    elif os.geteuid() != 0 and not (args.dry_run or args.fake_root):
        print("real workloads change kernel settings: run as root, or use --dry-run",
              file=sys.stderr)
        return 2
    agent = make_agent(args.agent, seed=args.seed, model=args.model,
                       provider=args.provider, base_url=args.base_url,
                       include_telemetry=not args.no_telemetry,
                       include_history=not args.no_history)
    out = args.out or f"results/{args.workload}/{args.agent}/" \
                      f"{datetime.now():%Y%m%d-%H%M%S}-s{args.seed}"
    tuner = Tuner(be, wl, agent, budget=args.budget, reps=args.reps, warmup=args.warmup,
                  max_changes=args.max_changes or None,
                  regression_tolerance=args.tolerance, keep_best=args.keep_best,
                  out_dir=out, health_gate=not args.no_health_gate,
                  settle_s=args.settle, validate_reps=args.validate_reps)
    summary = tuner.run()
    print(json.dumps(summary, indent=2))
    print(f"\nresults in {out}")
    return 0


def cmd_restore(args) -> int:
    be = SystemBackend(args.root)
    be.restore(SystemBackend.load_snapshot(args.snapshot))
    print("restored", args.snapshot)
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="ostuner", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("-v", "--verbose", action="store_true")
    p.add_argument("--root", default="/", help="filesystem root (default /)")
    p.add_argument("--fake-root", action="store_true",
                   help="use a throwaway fake /proc,/sys tree (no root needed)")
    p.add_argument("--dry-run", action="store_true",
                   help="read real values but never write them")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("knobs", help="list knobs and current values")

    r = sub.add_parser("run", help="run a tuning session")
    r.add_argument("--workload", required=True, help=f"one of {sorted(BUILTIN)}")
    r.add_argument("--agent", default="llm", choices=["default", "random", "hillclimb", "llm"])
    r.add_argument("--budget", type=int, default=20, help="trials after the baseline")
    r.add_argument("--reps", type=int, default=3, help="runs per trial (median taken)")
    r.add_argument("--warmup", type=int, default=0)
    r.add_argument("--validate-reps", type=int, default=5,
                   help="interleaved baseline-vs-best re-measurements at the end (0 = skip)")
    r.add_argument("--max-changes", type=int, default=3, help="0 = unlimited")
    r.add_argument("--tolerance", type=float, default=0.10,
                   help="flag a trial as a harmful regression if this much worse than best")
    r.add_argument("--keep-best", action="store_true",
                   help="leave the best config applied instead of restoring the original")
    r.add_argument("--seed", type=int, default=0)
    r.add_argument("--settle", type=float, default=0.0, help="seconds to wait after apply")
    r.add_argument("--out")
    r.add_argument("--provider", default="anthropic", choices=["anthropic", "openai"])
    r.add_argument("--model")
    r.add_argument("--base-url", help="for --provider openai (vLLM, Ollama, ...)")
    r.add_argument("--no-telemetry", action="store_true", help="ablation")
    r.add_argument("--no-history", action="store_true", help="ablation")
    r.add_argument("--no-health-gate", action="store_true", help="ablation")

    s = sub.add_parser("restore", help="restore a saved snapshot")
    s.add_argument("snapshot")

    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING,
                        format="%(asctime)s %(levelname)s %(message)s")
    return {"knobs": cmd_knobs, "run": cmd_run, "restore": cmd_restore}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
