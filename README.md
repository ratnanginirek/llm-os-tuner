# llm-os-tuner

**Safe, LLM-guided Linux kernel parameter tuning.** This is the research artifact for a paper on whether LLM agents can tune OS settings (sysctl, transparent huge pages, writeback, NUMA) for a given workload better and more sample-efficiently than classic search, *without ever putting the host in a bad state*.

The LLM proposes configurations, and a layered safety system decides what actually reaches the kernel:

```
agent.propose() ─► allowlist ─► bounds ─► cross-knob rules ─► step limit
                                                                   │
        rollback to best ◄── health gate (PSI, MemAvailable) ◄── apply
                │                                                  │
                └──────────── keep if better ◄──── benchmark (median of N)
```

On exit (normal, Ctrl-C or crash), the original configuration is restored. A snapshot is written to disk *before* any change, so `ostuner restore` can recover even if the process dies.

## Research questions

| RQ | Question | Experiment |
|----|----------|------------|
| RQ1 | Does an LLM agent find better configs than random search and hill climbing at the same trial budget? | `configs/main.yaml` |
| RQ2 | Is it more sample-efficient (trials to reach 90% of the best gain)? | convergence plots |
| RQ3 | How often does it propose unsafe or harmful configs, and does the safety layer catch them? | `safety.csv` |
| RQ4 | Which inputs matter: telemetry, trial history, step limit, health gate? | `configs/ablations.yaml` |
| RQ5 | Do open-weight models match proprietary ones? | `llm-open` agent in `main.yaml` |
| RQ6 | Are the LLM's rationales mechanistically correct? (manual expert grading) | `rationale` field in `trials.jsonl` |

## Quick start (no root, no API key)

```bash
pip install -e ".[dev,analysis]"
pytest                                                  # 22 tests
ostuner --fake-root run --workload sim-memdb --agent hillclimb --budget 20 --reps 1
python3 experiments/run_matrix.py experiments/configs/smoke.yaml
python3 analysis/plot_results.py results/smoke
```

`--fake-root` uses a throwaway directory that mimics `/proc/sys` and `/sys`. The `sim-*` workloads are **analytic simulators for development only**. Never report them as results.

## Real experiments

Use a **dedicated test machine or VM**, never production. Bare metal or a dedicated cloud instance gives the least noise.

```bash
sudo ./scripts/setup_host.sh                    # installs fio, sysbench; prints host info
pip install -e ".[llm,analysis]"
export ANTHROPIC_API_KEY=...
sudo -E ostuner --dry-run knobs                 # check what's tunable on this host
sudo -E ostuner run --workload fio-seqwrite --agent llm --budget 20 -v
sudo -E python3 experiments/run_matrix.py experiments/configs/main.yaml
python3 analysis/plot_results.py results/main
```

If anything goes wrong: `sudo ostuner restore results/<...>/original_snapshot.json`.

### Agents
- `default`: re-measures the starting config (noise floor)
- `random`: random valid configs
- `hillclimb`: perturbs the best config so far
- `llm`: Claude via `--provider anthropic`, or any OpenAI-compatible server (vLLM, Ollama) via `--provider openai --base-url ...`

### Workloads
- `fio-seqwrite`: buffered sequential writes + fsync (dirty ratios)
- `fio-randread`: buffered 4k random reads (page cache, reclaim)
- `sysbench-memory`: memory bandwidth (THP)
- Add your own (e.g. pgbench or HammerDB as a database proxy, redis-benchmark, an LLM-serving benchmark) via `workloads.from_config` or a new entry in `BUILTIN`.

## Methodology notes (what reviewers will check)

- **Selection bias.** Taking the best of 20 noisy trials overstates gains. Every run ends with an **interleaved A/B validation** (baseline, best, baseline, best...). `improvement` in `summary.json` is the validated number; `improvement_search` is the biased one. Report the former.
- **Equal budgets.** All agents get the same number of benchmark trials. Rejected proposals count against the budget.
- **Seeds and variance.** Use 5+ seeds and report mean ± std.
- **Cost.** Report LLM tokens and dollars per session alongside wall-clock time.
- **Hardware.** Record CPU, RAM, disk, kernel version and NUMA layout (`setup_host.sh` prints them).
- **Threats to validity.** Synthetic benchmarks vs. real applications; a single host type; LLM training-data contamination (sysctl advice is all over the web, so compare against a "folk wisdom" static config too).

## Layout

```
ostuner/            knobs.py (catalog + bounds) · system.py (read/write/snapshot)
                    safety.py · telemetry.py · workloads.py · tuner.py · cli.py
ostuner/agents/     classic.py (baselines) · llm.py (prompting, parsing, clients)
experiments/        run_matrix.py + configs/{smoke,main,ablations}.yaml
analysis/           plot_results.py → CSV, LaTeX table, convergence plots
paper/              LaTeX skeleton with section-by-section writing notes
tests/              runs fully on a fake /proc,/sys tree
```

## License

MIT
