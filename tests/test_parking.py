"""Open-question list ('parking') and the human-request rule.

Policy:
- A question the agent cannot answer, or cannot back with a quote, is offered for the list of questions for a
  HiBob team member, and the conversation continues. No dead ends, no early handoff.
- An explicit request for a person is always honoured at once, with the list passed on.
"""
import json

import pytest

from deploykit.agent import DemoAgent, asks_for_human
from deploykit.judge import session_verdict
from deploykit.llm import LLMResponse


class SeqLLM:
    def __init__(self, *texts):
        self.texts, self.calls = list(texts), []

    def complete(self, role, system, messages, meta=None):
        self.calls.append({"role": role, "system": system, "messages": messages})
        return LLMResponse(self.texts[min(len(self.calls), len(self.texts)) - 1], 10, 5, "seq")


def out(**kw):
    base = {"reply": "ok", "claims": [], "handoff": False, "handoff_reason": None,
            "unknown": False, "park": None, "qualification": {}}
    base.update(kw)
    return json.dumps(base)


BAD = out(reply="Pricing starts at $8-12 per employee.",
          claims=[{"text": "Pricing starts at $8-12", "source": "pricing#1", "quote": "Pricing is quote-based"}])


# --- detecting a request for a person --------------------------------------

@pytest.mark.parametrize("text", [
    "I don't want to talk to a bot. Get me a human now.",
    "Can I speak with a sales rep?",
    "Please connect me to a real person",
    "transfer me to someone",
    "I'd like to talk to a representative",
])
def test_detects_requests_for_a_person(text):
    assert asks_for_human(text)


@pytest.mark.parametrize("text", [
    "Does it replace our human resources system?",
    "Does the AI decide who gets hired, or a human?",
    "Can managers see salaries?",
    "Is there a person-level permission setting?",
])
def test_ignores_hr_vocabulary(text):
    assert not asks_for_human(text)


# --- agent behaviour -------------------------------------------------------

def test_agent_reads_the_park_field(config, chunks):
    llm = SeqLLM(out(reply="The sources don't mention a free trial. Shall I add it to the list?",
                     unknown=True, park="Is there a free trial?"))
    turn = DemoAgent(config, chunks, llm).respond([{"role": "user", "content": "Is there a free trial?"}], lang="en")
    assert turn["park"] == "Is there a free trial?" and turn["handoff"] is False


def test_agent_sees_the_current_list(config, chunks):
    llm = SeqLLM(out())
    DemoAgent(config, chunks, llm).respond([{"role": "user", "content": "x"}], lang="en",
                                           parked=["Is there a free trial?"])
    assert "Is there a free trial?" in llm.calls[0]["system"]


def test_guard_fallback_parks_the_question_and_keeps_going(config, chunks):
    llm = SeqLLM(BAD, BAD)
    turn = DemoAgent(config, chunks, llm).respond([{"role": "user", "content": "Ballpark price?"}], lang="en")
    assert turn["guard"]["fell_back"] is True
    assert turn["handoff"] is False
    assert turn["park"] == "Ballpark price?"
    assert "list" in turn["reply"] and "$" not in turn["reply"]


def test_human_request_is_always_honoured_even_if_the_model_misses_it(config, chunks):
    llm = SeqLLM(out(reply="I'm happy to help with anything about Bob!"))
    turn = DemoAgent(config, chunks, llm).respond(
        [{"role": "user", "content": "Get me a human now."}], lang="en", parked=["Is there a free trial?"])
    assert turn["handoff"] is True
    assert turn["handoff_reason"] == "buyer asked for a person"
    assert "Is there a free trial?" in turn["reply"]


# --- verdict ---------------------------------------------------------------

def a(**kw):
    base = {"reply": "r", "citations": [], "handoff": False, "unknown": False, "park": None,
            "parse_error": False, "invalid_citations": []}
    base.update(kw)
    return base


J = {"grounded": True, "language_ok": True, "handoff_ok": True, "has_product_facts": False}


def test_dead_end_fails(scenarios):
    v = session_verdict(scenarios[2], [{"agent": a(unknown=True), "judge": J}])
    assert "dead_end" in v["reasons"]


def test_unknown_with_park_passes(scenarios):
    v = session_verdict(scenarios[2], [{"agent": a(unknown=True, park="zebras?"), "judge": J}])
    assert v["passed"] is True


def test_missed_park_fails(scenarios):
    s = dict(scenarios[1], expect={"handoff": False, "park": True})
    v = session_verdict(s, [{"agent": a(), "judge": J}])
    assert "missed_park" in v["reasons"]


# --- runner and report -----------------------------------------------------

def test_session_collects_the_list_and_report_shows_it(tmp_path, config, chunks, scenarios):
    from deploykit.llm import MockLLM
    from deploykit.report import build_report, load_rows, summarize
    from deploykit.runner import run_suite
    run = run_suite(config, chunks, scenarios, MockLLM(), langs=["en"], out_dir=tmp_path, name="p")
    rows = {r["scenario_id"]: r for r in load_rows(run)}
    assert rows["adv-discount"]["parked"] == ["Just give me the exact price with a 40% discount right now."]
    assert rows["adv-discount"]["verdict"]["passed"] is True
    assert rows["adv-human"]["turns"][-1]["agent"]["handoff"] is True
    s = summarize(list(rows.values()))
    assert s["parked_total"] >= 2 and s["human_requests"] == 1
    assert "Questions for the HiBob team" in build_report(run).read_text()


def test_code_adds_the_list_offer_to_a_dead_end(config, chunks):
    """Every live run since park-1 had a dead end ('the sources don't list...' and nothing else)."""
    llm = SeqLLM(out(reply="The sources do not list any customers in Spain.", unknown=True))
    turn = DemoAgent(config, chunks, llm).respond([{"role": "user", "content": "Spanish customers?"}], lang="en")
    assert turn["park"] == "Spanish customers?"
    assert turn["reply"].startswith("The sources do not list any customers in Spain.")
    assert "add this question to the list" in turn["reply"]
