import json

import pytest

from ostuner import safety
from ostuner.agents import DefaultAgent, HillClimbAgent, LLMAgent, RandomAgent
from ostuner.agents.llm import parse_reply
from ostuner.knobs import CATALOG
from ostuner.system import SystemBackend, make_fake_root
from ostuner.telemetry import collect
from ostuner.tuner import Tuner
from ostuner.workloads import BUILTIN, SimulatedWorkload, json_parser, regex_parser


@pytest.fixture
def backend(tmp_path):
    make_fake_root(tmp_path)
    return SystemBackend(tmp_path)


# --- backend ---------------------------------------------------------------

def test_read_parses_sysfs_brackets(backend):
    assert backend.read("mm.transparent_hugepage") == "madvise"
    assert backend.read("vm.swappiness") == 60


def test_snapshot_restore_roundtrip(backend):
    snap = backend.snapshot()
    backend.apply({"vm.swappiness": 5, "mm.transparent_hugepage": "never"})
    assert backend.read("vm.swappiness") == 5
    assert backend.read("mm.transparent_hugepage") == "never"
    backend.restore(snap)
    assert backend.snapshot() == snap


def test_write_rejects_out_of_bounds(backend):
    with pytest.raises(ValueError):
        backend.write("vm.swappiness", 500)


def test_dry_run_never_writes(tmp_path):
    make_fake_root(tmp_path)
    be = SystemBackend(tmp_path, dry_run=True)
    be.write("vm.swappiness", 1)
    assert be.read("vm.swappiness") == 1
    assert (tmp_path / CATALOG["vm.swappiness"].path).read_text().strip() == "60"


# --- safety ----------------------------------------------------------------

def test_validate_rejects_unknown_and_bounds(backend):
    cur = backend.snapshot()
    v = safety.validate({"kernel.panic": 1, "vm.swappiness": -1}, cur)
    assert not v.ok and len(v.errors) == 2


def test_validate_cross_constraint(backend):
    v = safety.validate({"vm.dirty_background_ratio": 25, "vm.dirty_ratio": 20},
                        backend.snapshot())
    assert not v.ok and "must be <" in v.errors[0]


def test_validate_step_limit(backend):
    prop = {"vm.swappiness": 1, "vm.vfs_cache_pressure": 50,
            "kernel.numa_balancing": 0, "mm.transparent_hugepage": "never"}
    assert not safety.validate(prop, backend.snapshot(), max_changes=3).ok
    assert safety.validate(prop, backend.snapshot(), max_changes=None).ok


def test_health_check():
    assert safety.health_check({"mem_total_kb": 100, "mem_available_kb": 50}) == []
    assert safety.health_check({"mem_total_kb": 100, "mem_available_kb": 1})
    assert safety.health_check({"psi_memory_some_avg10": 90.0})


def test_telemetry(tmp_path):
    make_fake_root(tmp_path)
    (tmp_path / "proc/pressure").mkdir()
    (tmp_path / "proc/pressure/memory").write_text(
        "some avg10=1.50 avg60=0.00 avg300=0.00 total=0\n"
        "full avg10=0.00 avg60=0.00 avg300=0.00 total=0\n")
    t = collect(tmp_path)
    assert t["mem_total_kb"] == 16000000 and t["psi_memory_some_avg10"] == 1.5


# --- parsers ---------------------------------------------------------------

def test_parsers():
    assert regex_parser(r"\(([\d.]+) MiB/sec\)")("x (1234.56 MiB/sec)") == 1234.56
    out = json.dumps({"jobs": [{"write": {"bw": 999}}]})
    assert json_parser("jobs.0.write.bw")(out) == 999.0


def test_parse_reply_tolerates_prose():
    ch, why = parse_reply('Sure!\n{"changes": {"vm.swappiness": 10}, "rationale": "less swap"}')
    assert ch == {"vm.swappiness": 10} and why == "less swap"


# --- tuning loop -----------------------------------------------------------

def _sim():
    return BUILTIN["sim-memdb"]()


@pytest.mark.parametrize("agent", [DefaultAgent(), RandomAgent(1), HillClimbAgent(1)])
def test_loop_restores_original(backend, tmp_path, agent):
    original = backend.snapshot()
    summary = Tuner(backend, _sim(), agent, budget=8, reps=1,
                    out_dir=tmp_path / "out").run()
    assert backend.snapshot() == original
    assert summary["measured_trials"] >= 1
    lines = (tmp_path / "out/trials.jsonl").read_text().splitlines()
    assert len(lines) == 9


def test_keep_best(backend):
    t = Tuner(backend, _sim(), HillClimbAgent(3), budget=15, reps=1, keep_best=True)
    t.run()
    assert backend.snapshot() == t.best.config


def test_restore_on_exception(backend):
    original = backend.snapshot()

    class Boom(HillClimbAgent):
        n = 0

        def propose(self, ctx):
            self.n += 1
            if self.n == 3:
                raise KeyboardInterrupt
            return super().propose(ctx)

    with pytest.raises(KeyboardInterrupt):
        Tuner(backend, _sim(), Boom(0), budget=10, reps=1).run()
    assert backend.snapshot() == original


def test_llm_agent_good_reply_finds_optimum(backend):
    replies = iter([
        ('{"changes": {"mm.transparent_hugepage": "never", "kernel.numa_balancing": 0},'
         ' "rationale": "in-memory DBs prefer no THP and no NUMA balancing"}'),
        '{"changes": {"vm.swappiness": 1}, "rationale": "avoid swap"}',
    ] + ['{"changes": {}, "rationale": "hold"}'] * 5)
    agent = LLMAgent(lambda s, u: next(replies))
    s = Tuner(backend, _sim(), agent, budget=4, reps=1).run()
    assert s["improvement"] > 0.2
    assert s["best_changes"]["mm.transparent_hugepage"] == "never"


def test_llm_hallucinations_are_rejected_not_applied(backend):
    original = backend.snapshot()
    agent = LLMAgent(lambda s, u: '{"changes": {"kernel.panic_on_oops": 1, '
                                  '"vm.swappiness": 9000}, "rationale": "yolo"}')
    s = Tuner(backend, _sim(), agent, budget=3, reps=1).run()
    assert s["counts"]["rejected"] == 3
    assert backend.snapshot() == original


def test_llm_parse_failure_falls_back(backend):
    agent = LLMAgent(lambda s, u: "I think you should lower swappiness.", retries=1)
    s = Tuner(backend, _sim(), agent, budget=2, reps=1).run()
    assert agent.stats["fallbacks"] == 2 and agent.stats["calls"] == 4
    assert s["counts"]["measured"] >= 1


def test_regression_rolls_back(backend):
    wl = SimulatedWorkload(name="w", description="", unit="u",
                           optimum={"vm.swappiness": 60}, noise=0.0)
    agent = LLMAgent(lambda s, u: '{"changes": {"vm.swappiness": 0}, "rationale": "x"}')
    t = Tuner(backend, wl, agent, budget=1, reps=1, regression_tolerance=0.01)
    t.run()
    assert t.history[1].rolled_back


def test_unhealthy_config_rolls_back(tmp_path):
    make_fake_root(tmp_path)
    be = SystemBackend(tmp_path)

    class Hostile(HillClimbAgent):
        def propose(self, ctx):
            (tmp_path / "proc/meminfo").write_text("MemTotal: 100 kB\nMemAvailable: 1 kB\n")
            return super().propose(ctx)

    s = Tuner(be, _sim(), Hostile(0), budget=2, reps=1).run()
    assert s["counts"]["unhealthy"] == 2


def test_validation_removes_selection_bias(backend):
    """The default agent re-measures the same config; any 'search' gain is
    noise, and the interleaved validation should show ~0."""
    wl = SimulatedWorkload(name="w", description="", unit="u", noise=0.03, seed=7)
    s = Tuner(backend, wl, DefaultAgent(), budget=20, reps=1, validate_reps=15).run()
    assert s["improvement_search"] > 0.0
    assert abs(s["improvement"]) < s["improvement_search"]
