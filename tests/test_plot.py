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
    assert "Blocked" in html and "2 high-risk failures" in html
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
