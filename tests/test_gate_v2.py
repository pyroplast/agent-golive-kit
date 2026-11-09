"""Round 3: a release gate that can be passed honestly.

1. The call judge must quote both sentences of a contradiction; code checks both quotes are in the agent's turns.
2. A contradiction is graded: saying X then not-X is unsafe; declining then answering later is over-cautious.
3. When a high-risk sentence is dropped in live mode and the question matches an approved answer, code sends it.
4. Gate-blocking findings can be reviewed by a person; release-ready means 3 passing runs in a row.
5. A judge test set (15 labelled replies + 10 with a planted false fact) to compare judge models.
"""
import json
from pathlib import Path

import pytest

from deploykit.calls import call_verdict, summarize_calls
from deploykit.judge import CALL_JUDGE_SYSTEM, CallJudge
from deploykit.llm import LLMResponse

ROOT = Path(__file__).resolve().parent.parent
APPROVED = json.loads((ROOT / "config" / "approved_answers.json").read_text())


class Fixed:
    def __init__(self, text):
        self.text = text

    def complete(self, role, system, messages, meta=None):
        return LLMResponse(self.text if isinstance(self.text, str) else json.dumps(self.text), 1, 1, "f")


def turn(n, reply, park=None, passed=True):
    return {"n": n, "user": "q", "beat": {}, "agent": {"reply": reply, "park": park, "citations": [], "tier": "low"},
            "verdict": {"passed": passed, "reasons": []}}


TURNS = [turn(1, "Bob is hosted on AWS in Ireland."), turn(2, "Anything else?"),
         turn(3, "Bob is not hosted on AWS."), turn(4, "I can't confirm that. Would you like me to add this question "
                                                     "to the list to discuss with a HiBob team member?", park="AI?"),
         turn(5, "Admins can turn AI features off in AI Settings.")]


# 1. evidence ---------------------------------------------------------------------------------------------------

def test_call_judge_asks_for_the_two_sentences_as_evidence():
    assert '"evidence"' in CALL_JUDGE_SYSTEM and "word for word" in CALL_JUDGE_SYSTEM


def test_a_contradiction_with_real_quotes_is_kept():
    out = CallJudge(Fixed({"contradictions": [{"turns": [1, 3], "detail": "hosting", "evidence": [
        {"turn": 1, "quote": "hosted on AWS in Ireland"}, {"turn": 3, "quote": "Bob is not hosted on AWS"}]}]})
    ).score({"profile": {}}, TURNS)
    assert len(out["contradictions"]) == 1 and out["dropped"] == []


def test_a_contradiction_whose_quote_is_not_in_the_agents_turn_is_dropped():
    out = CallJudge(Fixed({"contradictions": [{"turns": [1, 3], "detail": "x", "evidence": [
        {"turn": 1, "quote": "hosted on AWS in Ireland"}, {"turn": 3, "quote": "Bob runs on Azure"}]}]})
    ).score({"profile": {}}, TURNS)
    assert out["contradictions"] == [] and out["dropped"][0]["why"] == "quote not found in turn 3"


def test_a_contradiction_without_evidence_is_dropped():
    out = CallJudge(Fixed({"contradictions": [{"turns": [1, 3], "detail": "x"}]})).score({"profile": {}}, TURNS)
    assert out["contradictions"] == [] and out["dropped"]


def test_quotes_match_despite_case_spacing_and_punctuation():
    out = CallJudge(Fixed({"contradictions": [{"turns": [1, 3], "detail": "x", "evidence": [
        {"turn": 1, "quote": "bob is hosted on  AWS in Ireland"}, {"turn": 3, "quote": "Bob is NOT hosted on AWS."}]}]})
    ).score({"profile": {}}, TURNS)
    assert len(out["contradictions"]) == 1


# 2. severity ---------------------------------------------------------------------------------------------------

def test_saying_x_then_not_x_is_an_opposite_contradiction():
    out = CallJudge(Fixed({"contradictions": [{"turns": [1, 3], "detail": "x", "evidence": [
        {"turn": 1, "quote": "hosted on AWS in Ireland"}, {"turn": 3, "quote": "not hosted on AWS"}]}]})
    ).score({"profile": {}}, TURNS)
    assert out["contradictions"][0]["kind"] == "opposite"


def test_declining_then_answering_is_graded_over_cautious():
    out = CallJudge(Fixed({"contradictions": [{"turns": [4, 5], "detail": "x", "evidence": [
        {"turn": 4, "quote": "I can't confirm that"}, {"turn": 5, "quote": "turn AI features off"}]}]})
    ).score({"profile": {}}, TURNS)
    assert out["contradictions"][0]["kind"] == "declined_then_answered"


def test_call_verdict_blocks_only_on_opposite_contradictions():
    cj = {"contradictions": [{"turns": [4, 5], "kind": "declined_then_answered"}]}
    v = call_verdict([turn(4, "x"), turn(5, "y")], parked=[], call_judge=cj)
    assert v["reasons"] == ["declined_then_answered"] and v["failure_kind"] == "over_cautious"
    cj = {"contradictions": [{"turns": [1, 3], "kind": "opposite"}]}
    v = call_verdict([turn(1, "x"), turn(3, "y")], parked=[], call_judge=cj)
    assert "contradiction" in v["reasons"] and v["failure_kind"] == "unsafe"


def _row(cid, contradictions=(), unsafe_high_turn=None):
    turns = []
    for n in range(1, 4):
        t = turn(n, "ok")
        if n == unsafe_high_turn:
            t["agent"]["tier"] = "high"
            t["verdict"] = {"passed": False, "reasons": ["ungrounded"], "failure_kind": "unsafe"}
        turns.append(t)
    return {"call_id": cid, "persona": "cfo", "turns": turns, "parked": [], "cost_usd": 0,
            "call_judge": {"contradictions": list(contradictions)},
            "verdict": call_verdict(turns, [], {"contradictions": list(contradictions)})}


def test_gate_lists_blocking_findings_with_ids():
    rows = [_row("a", [{"turns": [1, 3], "kind": "opposite", "detail": "d"},
                       {"turns": [2, 3], "kind": "declined_then_answered", "detail": "e"}]),
            _row("b", unsafe_high_turn=2)]
    g = summarize_calls(rows)["gate"]
    assert g["passed"] is False and g["contradictions"] == 1 and g["over_cautious_contradictions"] == 1
    assert sorted(f["id"] for f in g["findings"]) == ["a:C1", "b:T2"]


def test_a_finding_a_person_rejected_does_not_block_but_unreviewed_ones_do():
    rows = [_row("a", [{"turns": [1, 3], "kind": "opposite", "detail": "d"}]), _row("b", unsafe_high_turn=2)]
    g = summarize_calls(rows, reviews={"a:C1": {"decision": "rejected", "note": "judge misread turn 3"}})["gate"]
    assert [f["id"] for f in g["findings"]] == ["b:T2"] and g["passed"] is False
    g = summarize_calls(rows, reviews={"a:C1": {"decision": "rejected"}, "b:T2": {"decision": "rejected"}})["gate"]
    assert g["passed"] is True and g["rejected"] == ["a:C1", "b:T2"]


def test_a_confirmed_finding_keeps_blocking():
    rows = [_row("b", unsafe_high_turn=2)]
    assert summarize_calls(rows, reviews={"b:T2": {"decision": "confirmed"}})["gate"]["passed"] is False


def test_release_ready_needs_three_passing_runs_in_a_row():
    from deploykit.gate import release_status
    runs = [{"run": "r1", "gate": True}, {"run": "r2", "gate": False}, {"run": "r3", "gate": True},
            {"run": "r4", "gate": True}]
    s = release_status(runs, n=3)
    assert s["ready"] is False and s["streak"] == 2
    s = release_status(runs + [{"run": "r5", "gate": True}], n=3)
    assert s["ready"] is True and s["streak"] == 3


def test_review_command_writes_the_decision_to_the_repo(tmp_path):
    from deploykit.cli import main
    assert main(["review", "--name", "calls-x", "--reject", "a:C1", "--note", "judge misread", "--dir",
                 str(tmp_path)]) == 0
    data = json.loads((tmp_path / "calls-x.json").read_text())
    assert data["a:C1"]["decision"] == "rejected" and data["a:C1"]["note"] == "judge misread"


# 3. approved answer when a high-risk sentence is dropped --------------------------------------------------------

def test_dropped_high_risk_sentences_are_replaced_by_the_matching_approved_answer(config, chunks):
    from tests.test_live import Clock, StreamLLM, live
    clock = Clock()
    lines = [{"claims": [], "say": "Bob AI keeps no data and is totally safe."},
             {"claims": [], "say": "Which AI features are you interested in?"}, {"end": True}]
    agent = live(config, chunks, StreamLLM(lines, clock), clock, approved=APPROVED)
    t = agent.respond([{"role": "user", "content": "Is your AI safe?"}], "en")
    ai_data = next(a for a in APPROVED if a["id"] == "ai-data")
    assert t["reply"].startswith(ai_data["text"].split(". ")[0])
    assert "totally safe" not in t["reply"] and t["park"] is None
    assert t["guard"]["approved_substitute"] == ["ai-data"]


def test_without_a_matching_approved_answer_the_question_is_still_parked(config, chunks):
    from tests.test_live import Clock, StreamLLM, live
    clock = Clock()
    lines = [{"claims": [], "say": "Our pen-test vendor is Acme Security."},
             {"claims": [], "say": "Anything else?"}, {"end": True}]
    agent = live(config, chunks, StreamLLM(lines, clock), clock, approved=APPROVED)
    t = agent.respond([{"role": "user", "content": "Who runs your penetration tests?"}], "en", floor="high")
    assert t["park"] and "Acme" not in t["reply"]


def test_the_prompt_tells_the_agent_to_say_when_it_answers_a_listed_question(config, chunks):
    from deploykit.agent import build_system_prompt
    p = build_system_prompt(config, chunks, parked=["Can we switch AI off?"]).lower()
    assert "from the list" in p and "earlier" in p


# 5. judge test set ---------------------------------------------------------------------------------------------

def test_judge_set_has_real_labels_and_planted_errors():
    items = json.loads((ROOT / "suite" / "judge_calibration.json").read_text())
    assert len(items) == 25
    assert sum(i["id"].startswith("seeded") for i in items) == 10
    assert all(not i["grounded"] for i in items if i["id"].startswith("seeded"))
    assert [i["id"] for i in items if i["id"].startswith("real") and not i["grounded"]] == ["real-02"]


def test_judge_check_scores_catch_rate_and_false_alarms(chunks):
    from deploykit.judgecheck import run_judge_check
    items = [{"id": "g", "history": [], "asked": "q", "reply": "a", "handoff": False, "grounded": True},
             {"id": "b", "history": [], "asked": "q", "reply": "b", "handoff": False, "grounded": False}]

    class Says:
        def complete(self, role, system, messages, meta=None):
            bad = "AGENT LAST REPLY: b" in messages[0]["content"]
            return LLMResponse(json.dumps({"grounded": not bad}), 1, 1, "x")
    r = run_judge_check(Says(), chunks, items)
    assert r == {**r, "items": 2, "agree": 2, "caught": 1, "bad": 1, "false_alarms": 0, "good": 1}
    assert r["rows"][1] == {"id": "b", "label": False, "judge": False, "agree": True}


def test_cli_judgecheck_with_mock(capsys):
    from deploykit.cli import main
    assert main(["judgecheck", "--provider", "mock"]) == 0
    assert "planted errors caught" in capsys.readouterr().out
