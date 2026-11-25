#!/usr/bin/env python3
"""Turn results/<experiment>/ into paper figures and tables.

    python3 analysis/plot_results.py results/smoke

Writes into results/<experiment>/figures/:
  summary.csv        validated improvement (mean, std, n) per workload x agent
  safety.csv         rejected / unhealthy / harmful-regression counts per agent
  convergence_<workload>.png   best-so-far (normalized to baseline) vs. trial
  summary.tex        LaTeX table ready to \\input{} into the paper
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402


def load(exp_dir: Path):
    rows, curves = [], []
    for summ in exp_dir.glob("*/*/seed*/summary.json"):
        run = summ.parent
        workload, agent, seed = run.parent.parent.name, run.parent.name, run.name
        s = json.loads(summ.read_text())
        rows.append({"workload": workload, "agent": agent, "seed": seed,
                     "improvement": s["improvement"],
                     "improvement_search": s["improvement_search"],
                     "rejected": s["counts"]["rejected"],
                     "unhealthy": s["counts"]["unhealthy"],
                     "errors": s["counts"]["error"],
                     "harmful_regressions": s["rollbacks"],
                     "elapsed_s": s["elapsed_s"]})
        trials = [json.loads(line) for line in (run / "trials.jsonl").read_text().splitlines()]
        base = trials[0]["raw"]
        sign = 1 if trials[0]["score"] >= 0 else -1  # score = -raw for "min" workloads
        best = trials[0]["score"]
        for t in trials:
            if t["status"] == "measured":
                best = max(best, t["score"])
            curves.append({"workload": workload, "agent": agent, "seed": seed,
                           "trial": t["index"], "norm_best": sign * best / base})
    return pd.DataFrame(rows), pd.DataFrame(curves)


def main() -> int:
    exp_dir = Path(sys.argv[1] if len(sys.argv) > 1 else "results/smoke")
    df, curves = load(exp_dir)
    if df.empty:
        print(f"no runs found under {exp_dir}")
        return 1
    out = exp_dir / "figures"
    out.mkdir(exist_ok=True)

    summary = (df.groupby(["workload", "agent"])["improvement"]
                 .agg(["mean", "std", "count"]).reset_index())
    summary.to_csv(out / "summary.csv", index=False)
    safety = (df.groupby("agent")[["rejected", "unhealthy", "errors", "harmful_regressions"]]
                .sum().reset_index())
    safety.to_csv(out / "safety.csv", index=False)

    pivot = summary.assign(cell=lambda d: d.apply(
        lambda r: f"{100 * r['mean']:+.1f} $\\pm$ {100 * (r['std'] or 0):.1f}", axis=1))
    pivot = pivot.pivot(index="agent", columns="workload", values="cell")
    (out / "summary.tex").write_text(pivot.to_latex(escape=False))

    for wl, g in curves.groupby("workload"):
        fig, ax = plt.subplots(figsize=(4.5, 3))
        for agent, ga in g.groupby("agent"):
            m = ga.groupby("trial")["norm_best"].agg(["mean", "std"]).fillna(0)
            ax.plot(m.index, m["mean"], label=agent)
            ax.fill_between(m.index, m["mean"] - m["std"], m["mean"] + m["std"], alpha=0.15)
        ax.axhline(1.0, color="gray", lw=0.8, ls="--")
        ax.set_xlabel("trial")
        ax.set_ylabel("best so far / baseline")
        ax.set_title(wl)
        ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(out / f"convergence_{wl}.png", dpi=200)
        plt.close(fig)

    print(summary.to_string(index=False))
    print("\n", safety.to_string(index=False))
    print(f"\nfigures in {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
