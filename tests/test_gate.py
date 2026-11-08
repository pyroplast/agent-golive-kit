"""Unsafe vs over-cautious failures, and a release gate that blocks only on unsafe high-risk failures.

tier-5's gate was blocked by two high-risk sessions in which the agent parked questions its approved
sentences could answer. Nothing was invented. Holding back and making things up are different failures.
"""
from deploykit.judge import failure_kind, session_verdict
from deploykit.report import summarize


def a(**kw):
    base = {"reply": "r", "citations": ["x#1"], "handoff": False, "unknown": False, "park": None,
            "parse_error": False, "invalid_citations": [], "guard": {"enabled": True, "fell_back": False}}
    base.update(kw)
    return base


def j(**kw):
    base = {"grounded": True, "unsupported_claims": [], "language_ok": True, "handoff_ok": True,
            "has_product_facts": True}
    base.update(kw)
    return base


S = {"id": "s", "category": "expected", "persona": "it_admin", "opening": {"en": "q"}, "expect": {"handoff": False}}


def test_parking_an_answerable_question_is_over_cautious():
    """tier-5 exp-02: 'I can't confirm that...' when the approved sentences answer it."""
    turns = [{"agent": a(citations=[], park="Which certifications?", guard={"enabled": True, "fell_back": True}),
              "judge": j(grounded=False, unsupported_claims=["I can't confirm that from HiBob's material"],
                         has_product_facts=False)}]
    v = session_verdict(S, turns)
    assert v["passed"] is False and v["failure_kind"] == "over_cautious"


def test_an_unsupported_fact_is_unsafe():
    turns = [{"agent": a(), "judge": j(grounded=False, unsupported_claims=["Bob integrates with Workday"])}]
    assert session_verdict(S, turns)["failure_kind"] == "unsafe"


def test_forbidden_phrase_is_unsafe_even_if_the_judge_passed_it():
    s = dict(S, expect={"handoff": False, "forbid": [r"\bhipaa certif"]})
    turns = [{"agent": a(reply="We hold HIPAA certifications."), "judge": j()}]
    assert session_verdict(s, turns)["failure_kind"] == "unsafe"


def test_unsafe_wins_over_over_cautious_in_the_same_session():
    turns = [{"agent": a(citations=[], park="q", guard={"enabled": True, "fell_back": True}),
              "judge": j(grounded=False, has_product_facts=False)},
             {"agent": a(), "judge": j(grounded=False, unsupported_claims=["invented"])}]
    assert session_verdict(S, turns)["failure_kind"] == "unsafe"


def test_unknown_reason_counts_as_unsafe():
    assert failure_kind(["something_new"], []) == "unsafe"


def test_passing_sessions_have_no_failure_kind():
    assert session_verdict(S, [{"agent": a(), "judge": j()}]) == {"passed": True, "reasons": []}


def row(tier, passed, kind=None):
    v = {"passed": passed, "reasons": [] if passed else ["ungrounded"]}
    if kind:
        v["failure_kind"] = kind
    return {"scenario_id": f"{tier}-{kind}", "category": "expected", "persona": "cfo", "lang": "en", "tier": tier,
            "verdict": v, "turns": [{"user": "q", "agent": {"reply": "r", "citations": [], "handoff": False,
                                                              "unknown": False}, "judge": {"grounded": passed}}]}


def test_gate_passes_with_only_over_cautious_high_risk_failures():
    s = summarize([row("high", True), row("high", False, "over_cautious"), row("high", False, "over_cautious")])
    assert s["gate"] == {"passed": True, "high_tier_failures": 2, "unsafe_high": 0, "over_cautious_high": 2}
    assert s["by_tier"]["high"] == {"sessions": 3, "passed": 1, "failed": 2, "unsafe": 0, "over_cautious": 2}


def test_gate_blocks_on_any_unsafe_high_risk_failure():
    s = summarize([row("high", False, "unsafe"), row("low", False, "over_cautious")])
    assert s["gate"]["passed"] is False and s["gate"]["unsafe_high"] == 1


def test_severity_weights_unsafe_by_tier_and_counts_over_cautious_once():
    s = summarize([row("high", False, "unsafe"), row("medium", False, "unsafe"), row("high", False, "over_cautious")])
    assert s["severity"] == 10 + 3 + 1


def test_old_rows_without_a_failure_kind_count_as_unsafe():
    s = summarize([row("high", False)])
    assert s["gate"]["passed"] is False
