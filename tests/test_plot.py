"""The run-history chart script renders every run, both line charts, the tier bars and the gate."""
import json

from scripts.plot_runs import render
from tests.conftest import ROOT


def sample():
    runs = [{"run": n, "change": c, "passed": p, "sessions": s, "cost_usd": 0.5, "replies": 50, "guard": g,
             "blocked": b, "fallbacks": f, "replies_by_tier": None}
            for n, c, p, s, g, b, f in [
                ("baseline", "Baseline", 14, 30, False, None, None), ("fix-1", "x", 25, 30, False, None, None),
                ("fix-2", "x", 29, 30, False, None, None), ("guard-1", "x", 24, 30, True, 15, 10),
                ("guard-2", "x", 27, 30, True, 3, 1), ("park-1", "x", 27, 30, True, 1, 0),
                ("uncited-1", "x", 27, 30, True, 6, 0), ("reg-1", "x", 38, 43, True, 9, 1),
                ("reg-2", "x", 42, 43, True, 25, 2), ("reg-3", "x", 42, 43, True, 15, 1),
                ("tier-1", "x", 38, 43, True, 18, 7), ("tier-5", "Final", 41, 43, True, 17, 4)]]
    return {"runs": runs, "latest_by_tier": {"run": "tier-5", "sessions": {
        "low": {"sessions": 11, "passed": 11}, "medium": {"sessions": 18, "passed": 18}, "high": {"sessions": 14, "passed": 12}}}}


def test_render_contains_every_run_and_section():
    html = render(sample())
    for name in ("baseline", "guard-1", "reg-2", "tier-5"):
        assert name in html
    for heading in ("Pass rate per run", "Guard activity per run", "Latest run by risk tier", "All runs"):
        assert heading in html
    assert "Blocked" in html and "2 unsafe" in html  # no split in the data: failures count as unsafe
    assert "12 of 14" in html


def test_runs_without_a_guard_show_a_dash_not_zero():
    html = render(sample())
    assert "<td class='n'>—</td>" in html


def test_gate_passes_when_no_high_risk_failures():
    d = sample()
    d["latest_by_tier"]["sessions"]["high"] = {"sessions": 14, "passed": 14}
    assert ">Pass</b>" in render(d)


def test_committed_history_renders():
    data = json.loads((ROOT / "docs" / "run_history.json").read_text())
    assert len(data["runs"]) >= 15
    assert "<svg" in render(data)


def test_gate_on_the_page_blocks_only_on_unsafe_high_risk_failures():
    d = sample()
    d["latest_by_tier"]["sessions"]["high"] = {"sessions": 14, "passed": 12, "unsafe": 0, "over_cautious": 2}
    html = render(d)
    assert ">Pass</b>" in html and "0 unsafe" in html and "2 over-cautious" in html
    d["latest_by_tier"]["sessions"]["high"] = {"sessions": 14, "passed": 12, "unsafe": 1, "over_cautious": 1}
    assert ">Blocked</b>" in render(d)


def test_billed_cost_replaces_the_estimate_when_known():
    d = sample()
    d["billed_usd"] = 2.34
    html = render(d)
    assert "$2.34" in html and "billed" in html and "estimated $6.00 at list price" in html


def test_call_runs_get_their_own_section():
    d = sample()
    d["call_runs"] = [{"run": "calls-baseline", "change": "Baseline", "turns": 75, "passed": 58,
                       "stages": {"1-5": 0.84, "6-10": 0.76, "11-15": 0.72}, "contradictions": 1,
                       "gate": False, "cost_usd": 0.29}]
    html = render(d)
    assert "Chapter 2: demo calls" in html and "calls-baseline" in html and "58/75" in html and "72%" in html


def test_no_call_section_without_call_runs():
    assert "Chapter 2" not in render(sample())


def test_release_readiness_is_shown_for_call_runs():
    d = sample()
    base = {"change": "x", "turns": 75, "passed": 66, "stages": {"1-5": .8, "6-10": .9, "11-15": .9},
            "contradictions": 0, "cost_usd": 0.3}
    d["call_runs"] = [{**base, "run": "g1", "gate": False}] + [{**base, "run": f"g{i}", "gate": True} for i in (2, 3, 4)]
    assert "Release ready" in render(d) and "g2, g3, g4" in render(d)
    d["call_runs"][-1]["gate"] = False
    assert "Not release ready" in render(d)
