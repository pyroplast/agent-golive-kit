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
    turn = DemoAgent(config, chunks, FixedLLM("I am not JSON")).respond(
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
