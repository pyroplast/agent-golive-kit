import json

import pytest

from deploykit.agent import DemoAgent, parse_json
from deploykit.llm import LLMResponse, MockLLM


class FixedLLM:
    """Returns a canned text and records what it was sent."""

    def __init__(self, text):
        self.text = text
        self.calls = []

    def complete(self, role, system, messages, meta=None):
        self.calls.append({"role": role, "system": system, "messages": messages, "meta": meta})
        return LLMResponse(self.text, 10, 5, "fixed")


def test_parse_json_extracts_object_from_prose():
    assert parse_json('Sure!\n```json\n{"a": 1, "b": "x"}\n```') == {"a": 1, "b": "x"}


def test_parse_json_raises_without_object():
    with pytest.raises(ValueError):
        parse_json("no json here")


def test_agent_prompt_contains_rules_sources_and_persona_free_history(config, chunks):
    llm = FixedLLM(json.dumps({"reply": "Hi", "citations": [], "handoff": False,
                                "handoff_reason": None, "unknown": False, "qualification": {}}))
    agent = DemoAgent(config, chunks, llm)
    agent.respond([{"role": "user", "content": "hello"}], lang="en")
    call = llm.calls[0]
    assert call["role"] == "agent"
    assert "[security#2]" in call["system"]
    for rule in config["rules"]:
        assert rule in call["system"]
    assert call["messages"] == [{"role": "user", "content": "hello"}]


def test_agent_filters_unknown_citations(config, chunks):
    llm = FixedLLM(json.dumps({"reply": "Ireland.", "citations": ["security#2", "made-up#9"],
                                "handoff": False, "handoff_reason": None, "unknown": False,
                                "qualification": {}}))
    turn = DemoAgent(config, chunks, llm).respond([{"role": "user", "content": "region?"}], lang="en")
    assert turn["citations"] == ["security#2"]
    assert turn["invalid_citations"] == ["made-up#9"]
    assert turn["parse_error"] is False


def test_agent_survives_bad_json(config, chunks):
    turn = DemoAgent(config, chunks, FixedLLM("I am not JSON"), guard=False).respond(
        [{"role": "user", "content": "x"}], lang="en")
    assert turn["parse_error"] is True
    assert turn["reply"] == "I am not JSON"
    assert turn["handoff"] is False


def test_mock_agent_hands_off_on_discount(config, chunks):
    turn = DemoAgent(config, chunks, MockLLM()).respond(
        [{"role": "user", "content": "Give me a 40% discount"}], lang="en")
    assert turn["handoff"] is True


def test_mock_agent_cites_relevant_chunk(config, chunks):
    turn = DemoAgent(config, chunks, MockLLM()).respond(
        [{"role": "user", "content": "Which AWS region hosts the data?"}], lang="en")
    assert turn["citations"] == ["security#2"]
    assert turn["unknown"] is False


def test_agent_replays_its_own_turns_as_json(config, chunks):
    """Fix for baseline finding: plain-text history made the model drop the JSON format on turn 2."""
    llm = FixedLLM(json.dumps({"reply": "ok", "citations": [], "handoff": False,
                                "handoff_reason": None, "unknown": False, "qualification": {}}))
    history = [{"role": "user", "content": "q1"}, {"role": "assistant", "content": "plain answer"},
               {"role": "user", "content": "q2"}]
    DemoAgent(config, chunks, llm).respond(history, lang="en")
    sent = llm.calls[0]["messages"]
    assert json.loads(sent[1]["content"])["reply"] == "plain answer"
    assert sent[0] == history[0] and sent[2] == history[2]
    assert history[1]["content"] == "plain answer"  # caller's history is not mutated


def test_agent_accepts_bracketed_citation_ids(config, chunks):
    """Fix for baseline finding: the model sometimes cites '[pricing#1]' instead of 'pricing#1'."""
    llm = FixedLLM(json.dumps({"reply": "x", "citations": ["[security#2]", " pricing#1 "],
                                "handoff": False, "handoff_reason": None, "unknown": False,
                                "qualification": {}}))
    turn = DemoAgent(config, chunks, llm).respond([{"role": "user", "content": "q"}], lang="en")
    assert turn["citations"] == ["security#2", "pricing#1"]
    assert turn["invalid_citations"] == []


class SeqLLM:
    """Returns canned texts in order and records every call."""

    def __init__(self, *texts):
        self.texts, self.calls = list(texts), []

    def complete(self, role, system, messages, meta=None):
        self.calls.append({"role": role, "system": system, "messages": messages})
        return LLMResponse(self.texts[min(len(self.calls), len(self.texts)) - 1], 10, 5, "seq")


BAD = json.dumps({"reply": "Pricing starts at $8-12 per employee.", "claims": [
    {"text": "Pricing starts at $8-12", "source": "pricing#1", "quote": "Pricing is quote-based"}],
    "handoff": False, "handoff_reason": None, "unknown": False, "qualification": {}})
GOOD = json.dumps({"reply": "Pricing is quote-based and calculated per employee.", "claims": [
    {"text": "Pricing is quote-based", "source": "pricing#1", "quote": "Pricing is quote-based and calculated per employee"}],
    "handoff": False, "handoff_reason": None, "unknown": False, "qualification": {}})


def test_guard_retries_once_with_findings_then_sends_the_fixed_reply(config, chunks):
    llm = SeqLLM(BAD, GOOD)
    turn = DemoAgent(config, chunks, llm).respond([{"role": "user", "content": "ballpark?"}], lang="en")
    assert len(llm.calls) == 2
    assert "number_not_in_source" in llm.calls[1]["messages"][-1]["content"]
    assert turn["reply"].startswith("Pricing is quote-based")
    assert turn["guard"]["retried"] is True and turn["guard"]["fell_back"] is False
    assert turn["guard"]["first_problems"] and turn["guard"]["final_problems"] == []
    assert turn["citations"] == ["pricing#1"]


def test_guard_falls_back_to_a_safe_handoff_after_two_failures(config, chunks):
    llm = SeqLLM(BAD, BAD)
    turn = DemoAgent(config, chunks, llm).respond([{"role": "user", "content": "ballpark?"}], lang="en")
    assert len(llm.calls) == 2
    assert turn["handoff"] is True and turn["guard"]["fell_back"] is True
    assert "$" not in turn["reply"]
    assert turn["handoff_reason"] == "guard: could not verify the answer"


def test_guard_can_be_switched_off(config, chunks):
    llm = SeqLLM(BAD)
    turn = DemoAgent(config, chunks, llm, guard=False).respond([{"role": "user", "content": "x"}], lang="en")
    assert len(llm.calls) == 1 and "$8-12" in turn["reply"]
    assert turn["guard"] == {"enabled": False}


def test_agent_prompt_asks_for_quoted_claims(config, chunks):
    llm = SeqLLM(GOOD)
    DemoAgent(config, chunks, llm).respond([{"role": "user", "content": "x"}], lang="en")
    assert '"quote"' in llm.calls[0]["system"]


def test_guard_treats_malformed_output_as_a_failure(config, chunks):
    llm = SeqLLM("I am not JSON", "still not JSON")
    turn = DemoAgent(config, chunks, llm).respond([{"role": "user", "content": "x"}], lang="en")
    assert len(llm.calls) == 2 and turn["guard"]["fell_back"] is True
    assert turn["guard"]["first_problems"][0]["type"] == "parse_error"
    assert turn["handoff"] is True and turn["parse_error"] is False


def test_guard_keeps_the_blocked_draft_for_diagnosis(config, chunks):
    turn = DemoAgent(config, chunks, SeqLLM(BAD, GOOD)).respond([{"role": "user", "content": "x"}], lang="en")
    assert "$8-12" in turn["guard"]["blocked_draft"]
