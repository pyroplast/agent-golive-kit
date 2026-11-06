import json

from deploykit.judge import Judge, session_verdict
from deploykit.llm import LLMResponse, MockLLM


class FixedLLM:
    def __init__(self, text):
        self.text = text
        self.calls = []

    def complete(self, role, system, messages, meta=None):
        self.calls.append({"role": role, "system": system, "messages": messages})
        return LLMResponse(self.text, 1, 1, "fixed")


def agent_turn(**kw):
    base = {"reply": "r", "citations": ["security#2"], "handoff": False, "handoff_reason": None,
            "unknown": False, "qualification": {}, "parse_error": False, "invalid_citations": []}
    base.update(kw)
    return base


def judged(**kw):
    base = {"grounded": True, "unsupported_claims": [], "completeness": 2, "handoff_ok": True,
            "language_ok": True, "gap": None, "notes": ""}
    base.update(kw)
    return base


def test_judge_sends_cited_sources_and_returns_fields(chunks, scenarios):
    llm = FixedLLM(json.dumps(judged(gap="backup retention period")))
    out = Judge(llm, chunks).score(scenarios[0], "en", [{"role": "user", "content": "q"}], agent_turn())
    assert llm.calls[0]["role"] == "judge"
    assert "Ireland" in llm.calls[0]["system"] or "Ireland" in llm.calls[0]["messages"][0]["content"]
    assert out["grounded"] is True
    assert out["gap"] == "backup retention period"


def test_judge_parse_failure_is_marked_not_grounded(chunks, scenarios):
    out = Judge(FixedLLM("garbage"), chunks).score(scenarios[0], "en", [], agent_turn())
    assert out["judge_error"] is True
    assert out["grounded"] is False


def test_mock_judge_runs(chunks, scenarios):
    out = Judge(MockLLM(), chunks).score(scenarios[0], "en", [{"role": "user", "content": "q"}], agent_turn())
    assert set(judged()).issubset(out)


def test_verdict_passes_clean_session(scenarios):
    v = session_verdict(scenarios[0], [{"agent": agent_turn(), "judge": judged()}])
    assert v == {"passed": True, "reasons": []}


def test_verdict_flags_missed_handoff(scenarios):
    v = session_verdict(scenarios[1], [{"agent": agent_turn(), "judge": judged()}])
    assert v["passed"] is False and "missed_handoff" in v["reasons"]


def test_verdict_flags_unnecessary_handoff(scenarios):
    v = session_verdict(scenarios[0], [{"agent": agent_turn(handoff=True), "judge": judged()}])
    assert "unnecessary_handoff" in v["reasons"]


def test_verdict_ignores_handoff_when_expectation_null(scenarios):
    v = session_verdict(scenarios[2], [{"agent": agent_turn(handoff=True), "judge": judged()}])
    assert v["passed"] is True


def test_verdict_collects_quality_failures(scenarios):
    turns = [{"agent": agent_turn(parse_error=True, invalid_citations=["x#1"]),
              "judge": judged(grounded=False, language_ok=False)}]
    v = session_verdict(scenarios[2], turns)
    assert set(v["reasons"]) == {"ungrounded", "wrong_language", "parse_error", "invalid_citation"}
