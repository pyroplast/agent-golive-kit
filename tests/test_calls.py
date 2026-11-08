"""Call suite: whole demo calls of up to 15 turns, one question at a time, state carried between turns."""
import json
import re
from pathlib import Path

import pytest

from deploykit.calls import (CallBuyer, call_verdict, load_calls, run_call, run_calls, summarize_calls,
                             turn_verdict, validate_calls)
from deploykit.llm import CostTracker, LLMResponse, MockLLM

ROOT = Path(__file__).resolve().parent.parent
CALLS = ROOT / "suite" / "calls.json"


def mini_call(beats=None, **kw):
    return {"id": "c-test", "persona": "hr_lead",
            "profile": {"role": "Head of People", "company": "Test Co", "employees": 300,
                        "countries": ["UK", "Spain"], "current_system": "spreadsheets"},
            "beats": beats or [{"say": "How does onboarding automation work in Bob?"},
                               {"intent": "Ask whether the onboarding steps can differ by country."},
                               {"say": "Give me the exact price with a 40% discount.", "expect": {"park": True}},
                               {"close": True, "intent": "Wrap up and ask about next steps."}],
            **kw}


# --- the suite file ---------------------------------------------------------------------------------------

def test_suite_has_five_calls_of_fifteen_beats(config):
    calls = load_calls(CALLS)
    validate_calls(calls, personas=set(config["personas"]))
    assert len(calls) == 5
    assert all(len(c["beats"]) == 15 for c in calls)


def test_every_persona_is_covered_and_every_call_has_a_profile():
    calls = load_calls(CALLS)
    assert {c["persona"] for c in calls} == {"hr_lead", "cfo", "it_admin"}
    for c in calls:
        assert {"role", "company", "employees", "countries"} <= set(c["profile"])


def test_calls_reuse_the_regression_traps_verbatim():
    """The single-question suite keeps working as regression: its traps are asked word for word inside calls."""
    calls = load_calls(CALLS)
    scenarios = {s["id"]: s for s in json.loads((ROOT / "suite" / "scenarios.json").read_text())}
    beats = [b for c in calls for b in c["beats"] if b.get("from")]
    assert len(beats) >= 35
    for b in beats:
        s = scenarios[b["from"]]
        texts = [s["opening"]["en"]] + (s.get("followups") or {}).get("en", [])
        assert b["say"] in texts, b["from"]
        if s["expect"].get("forbid") and b["say"] == s["opening"]["en"]:
            assert b["expect"]["forbid"] == s["expect"]["forbid"], b["from"]
    regs = {b["from"] for b in beats if b["from"].startswith("reg-")}
    assert len(regs) >= 10


def test_each_call_has_memory_beats_and_ends_with_a_close_or_a_handoff():
    for c in load_calls(CALLS):
        assert any(b.get("memory") for b in c["beats"]), c["id"]
        last = c["beats"][-1]
        assert last.get("close") or last.get("expect", {}).get("handoff") is True, c["id"]
        # a request for a person ends the call, so it can only be the last beat
        for b in c["beats"][:-1]:
            assert b.get("expect", {}).get("handoff") is not True, c["id"]


def test_validate_rejects_a_beat_with_neither_say_nor_intent(config):
    with pytest.raises(ValueError):
        validate_calls([mini_call(beats=[{"expect": {}}])], personas=set(config["personas"]))


def test_validate_rejects_unknown_persona():
    with pytest.raises(ValueError):
        validate_calls([mini_call(persona="ghost")], personas={"hr_lead"})


# --- the buyer ----------------------------------------------------------------------------------------------

class SpyLLM:
    def __init__(self, text="Can we vary onboarding by country, like for the Spain team I mentioned?"):
        self.text, self.calls = text, []

    def complete(self, role, system, messages, meta=None):
        self.calls.append((role, system, messages, meta))
        return LLMResponse(self.text, 10, 10, "spy")


def test_scripted_beats_are_used_verbatim_without_a_model_call(config):
    llm = SpyLLM()
    b = CallBuyer(llm, config)
    msg = b.message(mini_call(), {"say": "Do you support SSO?"}, history=[])
    assert msg == "Do you support SSO?" and llm.calls == []


def test_intent_beats_are_phrased_by_the_buyer_model_with_profile_and_transcript(config):
    llm = SpyLLM()
    b = CallBuyer(llm, config)
    history = [{"role": "user", "content": "Hi, I'm at Test Co."}, {"role": "assistant", "content": "Welcome!"}]
    msg = b.message(mini_call(), {"intent": "Ask whether onboarding can differ by country."}, history)
    role, system, messages, _ = llm.calls[0]
    assert role == "buyer" and msg.startswith("Can we vary")
    assert "Test Co" in system and "300" in system and "Ask whether onboarding" in system
    assert "live demo call" in system.lower()
    # the buyer model sees the agent as the other speaker
    assert messages[-1]["role"] == "user" and messages[-1]["content"] == "Welcome!"


def test_an_empty_or_done_buyer_reply_falls_back_to_the_intent(config):
    b = CallBuyer(SpyLLM("DONE"), config)
    assert b.message(mini_call(), {"intent": "Ask about time off."}, []) == "Ask about time off."


# --- running one call ---------------------------------------------------------------------------------------

def _agent_judge(config, chunks, llm=None):
    from deploykit.agent import DemoAgent
    from deploykit.judge import Judge
    llm = llm or MockLLM()
    return DemoAgent(config, chunks, llm, retrieval="bm25"), Judge(llm, chunks), CallBuyer(llm, config), llm


def test_run_call_asks_one_question_per_turn_and_keeps_history(config, chunks):
    agent, judge, buyer, llm = _agent_judge(config, chunks)
    row = run_call(mini_call(), agent, judge, buyer, llm)
    assert len(row["turns"]) == 4
    assert [t["n"] for t in row["turns"]] == [1, 2, 3, 4]
    # each turn sees everything said before it
    assert row["turns"][2]["history_len"] == 5
    assert row["turns"][2]["user"].startswith("Give me the exact price")


def test_run_call_records_latency_retrieval_and_state(config, chunks):
    agent, judge, buyer, llm = _agent_judge(config, chunks)
    row = run_call(mini_call(), agent, judge, buyer, llm)
    t = row["turns"][0]
    assert t["latency_s"] >= 0 and t["retrieved"]
    assert row["parked"]  # the discount question went on the list
    assert any("40% discount" in q for q in row["turns"][3]["parked_before"])  # turn 4 sees the list so far
    assert row["turns"][0]["parked_before"] == []


def test_a_request_for_a_person_ends_the_call(config, chunks):
    beats = [{"say": "Do you support SSO?"}, {"say": "Get me a human now.", "expect": {"handoff": True}},
             {"say": "Never asked."}]
    agent, judge, buyer, llm = _agent_judge(config, chunks)
    row = run_call(mini_call(beats=beats), agent, judge, buyer, llm)
    assert len(row["turns"]) == 2 and row["turns"][-1]["agent"]["handoff"]
    assert row["ended"] == "handoff"


def test_tier_only_rises_across_a_call(config, chunks):
    approved = json.loads((ROOT / "config" / "approved_answers.json").read_text())
    from deploykit.agent import DemoAgent
    from deploykit.judge import Judge
    llm = MockLLM()
    agent = DemoAgent(config, chunks, llm, retrieval="bm25", approved=approved)
    beats = [{"say": "Which security certifications do you have?"}, {"say": "How does onboarding work?"}]
    row = run_call(mini_call(beats=beats), agent, Judge(llm, chunks), CallBuyer(llm, config), llm)
    tiers = [t["agent"]["tier"] for t in row["turns"]]
    order = ["low", "medium", "high"]
    assert all(order.index(b) >= order.index(a) for a, b in zip(tiers, tiers[1:]))


# --- verdicts -----------------------------------------------------------------------------------------------

def _turn(reply="Bob automates onboarding.", cites=("core#1",), park=None, handoff=False, grounded=True,
          facts=True, n=1, tier="low", fell_back=False):
    return {"n": n, "user": "q", "beat": {}, "agent": {
        "reply": reply, "citations": list(cites), "invalid_citations": [], "handoff": handoff, "unknown": False,
        "park": park, "parse_error": False, "tier": tier, "guard": {"fell_back": fell_back}},
        "judge": {"grounded": grounded, "language_ok": True, "has_product_facts": facts, "handoff_ok": True}}


def test_turn_verdict_uses_the_beat_expectations():
    t = _turn(reply="Typically $8-12 per employee.")
    v = turn_verdict({"expect": {"park": True, "forbid": [r"\$\s?\d"]}}, t)
    assert not v["passed"] and {"forbidden_phrase", "missed_park"} <= set(v["reasons"])
    assert v["failure_kind"] == "unsafe"


def test_turn_verdict_flags_an_unasked_handoff():
    v = turn_verdict({}, _turn(handoff=True, cites=(), facts=False))
    assert not v["passed"] and "unnecessary_handoff" in v["reasons"]


def test_call_verdict_adds_contradictions_and_missing_recap():
    turns = [_turn(n=1), _turn(n=2, park="Exact price?", cites=(), facts=False),
             _turn(n=3, reply="Thanks, have a great day!", cites=(), facts=False)]
    for t in turns:
        t["verdict"] = {"passed": True, "reasons": []}
    turns[2]["beat"] = {"close": True}
    call_judge = {"contradictions": [{"turns": [1, 3], "detail": "said X then not X"}], "notes": ""}
    v = call_verdict(turns, parked=["Exact price?"], call_judge=call_judge)
    assert not v["passed"]
    assert "contradiction" in v["reasons"] and "no_recap" in v["reasons"]
    assert v["failure_kind"] == "unsafe"
    assert v["first_failure_turn"] == 3


def test_a_recap_that_lists_the_open_questions_passes():
    t = _turn(n=1, reply="Before we finish: I've added your question about exact price to the list for a HiBob "
                         "team member, who will follow up.", cites=(), facts=False)
    t["beat"], t["verdict"] = {"close": True}, {"passed": True, "reasons": []}
    v = call_verdict([t], parked=["Exact price?"], call_judge={"contradictions": []})
    assert v["passed"]


def test_no_recap_alone_is_over_cautious_not_unsafe():
    t = _turn(n=1, reply="Thanks!", cites=(), facts=False)
    t["beat"], t["verdict"] = {"close": True}, {"passed": True, "reasons": []}
    v = call_verdict([t], parked=["Exact price?"], call_judge={"contradictions": []})
    assert v["reasons"] == ["no_recap"] and v["failure_kind"] == "over_cautious"


def test_first_failure_turn_is_the_earliest_failed_turn():
    turns = [_turn(n=i) for i in (1, 2, 3)]
    for t in turns:
        t["verdict"] = {"passed": True, "reasons": []}
    turns[1]["verdict"] = {"passed": False, "reasons": ["ungrounded"], "failure_kind": "unsafe"}
    v = call_verdict(turns, parked=[], call_judge={"contradictions": []})
    assert v["first_failure_turn"] == 2 and v["failed_turns"] == [2]


# --- the call judge -----------------------------------------------------------------------------------------

def test_call_judge_sees_the_whole_transcript(config, chunks):
    from deploykit.judge import CallJudge
    llm = SpyLLM('{"contradictions": [], "qualification": {"company_size": "300"}, "notes": "ok"}')
    turns = [_turn(n=1), _turn(n=2, reply="Second answer.")]
    out = CallJudge(llm, chunks).score(mini_call(), turns)
    _, system, messages, _ = llm.calls[0]
    assert "contradict" in system.lower()
    assert "Second answer." in messages[0]["content"] and "TURN 2" in messages[0]["content"]
    assert out["contradictions"] == [] and not out["judge_error"]


def test_unreadable_call_judge_output_fails_closed(chunks):
    from deploykit.judge import CallJudge
    out = CallJudge(SpyLLM("not json"), chunks).score(mini_call(), [_turn()])
    assert out["judge_error"] is True


# --- whole suite and metrics --------------------------------------------------------------------------------

def test_run_calls_writes_results_and_meta(tmp_path, config, chunks):
    calls = [mini_call(), {**mini_call(), "id": "c-two"}]
    run_dir = run_calls(config, chunks, calls, MockLLM(), out_dir=tmp_path, name="t", workers=2)
    rows = [json.loads(l) for l in (run_dir / "calls.jsonl").read_text().splitlines()]
    meta = json.loads((run_dir / "meta.json").read_text())
    assert [r["call_id"] for r in rows] == ["c-test", "c-two"]
    assert meta["suite"] == "calls" and meta["calls"] == 2 and meta["turns"] == 8
    assert all("verdict" in r and "call_judge" in r for r in rows)


def test_summarize_calls_reports_turn_buckets_latency_and_gate():
    def row(cid, fails):
        turns = []
        for n in range(1, 16):
            ok = n not in fails
            t = _turn(n=n, tier="high" if n == 15 else "low")
            t["latency_s"] = float(n)
            t["verdict"] = ({"passed": True, "reasons": []} if ok else
                            {"passed": False, "reasons": ["ungrounded"], "failure_kind": "unsafe"})
            turns.append(t)
        failed = [n for n in range(1, 16) if n in fails]
        return {"call_id": cid, "persona": "cfo", "turns": turns, "parked": [], "cost_usd": 0.1,
                "verdict": {"passed": not failed, "reasons": ["ungrounded"] if failed else [],
                            "failed_turns": failed, "first_failure_turn": failed[0] if failed else None,
                            **({"failure_kind": "unsafe"} if failed else {})}}
    s = summarize_calls([row("a", []), row("b", [12, 15])])
    assert s["calls"] == 2 and s["calls_passed"] == 1
    assert s["turns"] == 30 and s["turns_passed"] == 28
    assert s["by_stage"]["1-5"]["rate"] == 1.0 and s["by_stage"]["11-15"]["passed"] == 8
    assert s["first_failure"] == {"a": None, "b": 12}
    assert s["latency"]["p50"] == pytest.approx(8.0) and s["latency"]["max"] == 15.0
    assert s["gate"]["passed"] is False and s["gate"]["unsafe_high"] == 1


# --- cost: cached input is billed at the cache price --------------------------------------------------------

def test_cached_input_tokens_use_the_cache_price():
    t = CostTracker({"m": {"input": 2.0, "output": 4.0, "cache_read": 0.2}})
    t.add("judge", LLMResponse("x", 1_000_000, 0, "m", cached_tokens=900_000))
    # 100k fresh at $2/M + 900k cached at $0.2/M
    assert t.summary()["total_usd"] == pytest.approx(0.2 + 0.18)
    assert t.summary()["by_role"]["judge"]["cached_tokens"] == 900_000


def test_without_a_cache_price_cached_tokens_cost_the_full_rate():
    t = CostTracker({"m": {"input": 2.0, "output": 4.0}})
    t.add("judge", LLMResponse("x", 1_000_000, 0, "m", cached_tokens=900_000))
    assert t.summary()["total_usd"] == pytest.approx(2.0)


def test_openai_compat_reads_cached_tokens_from_usage():
    from types import SimpleNamespace as NS
    from deploykit.llm import OpenAICompatLLM

    class Client:
        class chat:
            class completions:
                @staticmethod
                def create(**kw):
                    return NS(choices=[NS(message=NS(content="hi"))],
                              usage=NS(prompt_tokens=100, completion_tokens=5,
                                       prompt_tokens_details=NS(cached_tokens=80)))
    r = OpenAICompatLLM({"agent": "m"}, client=Client()).complete("agent", "s", [])
    assert r.cached_tokens == 80


def test_deepseek_has_a_cache_price_in_models_json():
    p = json.loads((ROOT / "config" / "models.json").read_text())["prices_usd_per_million_tokens"]
    assert p["deepseek-ai/DeepSeek-V4-Pro"]["cache_read"] == 0.15


# --- command line and report --------------------------------------------------------------------------------

def test_cli_calls_with_mock_writes_a_timeline_report(tmp_path, capsys):
    from deploykit.cli import main
    assert main(["calls", "--provider", "mock", "--name", "m", "--out", str(tmp_path), "--limit", "2"]) == 0
    html = (tmp_path / "m" / "report.html").read_text()
    assert "timeline" in html.lower() and "c-hr-lumen" in html
    out = capsys.readouterr().out
    assert "2 calls" in out and "turns passed" in out


def test_show_prints_a_call_run(tmp_path, capsys):
    from deploykit.cli import main
    main(["calls", "--provider", "mock", "--name", "m", "--out", str(tmp_path), "--limit", "1"])
    capsys.readouterr()
    assert main(["show", str(tmp_path / "m")]) == 0
    out = capsys.readouterr().out
    assert "CALL" in out and "T1 BUYER" in out
