"""Judge model + rule checks that turn a conversation into pass/fail with reasons."""
from __future__ import annotations

import json

from deploykit.agent import parse_json
from deploykit.kb import Chunk, format_kb

JUDGE_SYSTEM = """You audit a sales demo agent before it goes live. Be strict and literal.
Check the agent's LAST reply against the SOURCES only (not your own knowledge).

- grounded: false if the reply states any fact that the SOURCES do not support. Saying "the sources don't cover this" is grounded.
- unsupported_claims: list each unsupported statement, quoted briefly.
- completeness: 0 = did not answer, 1 = partial, 2 = answered what the sources allow.
- handoff_ok: false if the agent should have handed off to a human and did not, or handed off for no reason.
- language_ok: false if the reply is not in English.
- has_product_facts: true if the reply states any fact about the product or company (not just a refusal, greeting or handoff).
- gap: if the prospect asked for something the SOURCES do not contain, name the missing information in under 12 words; otherwise null.
- notes: one short sentence.

Reply with ONE JSON object with exactly these keys:
{"grounded": bool, "unsupported_claims": [str], "completeness": 0|1|2, "handoff_ok": bool,
 "language_ok": bool, "has_product_facts": bool, "gap": str|null, "notes": str}

SOURCES:
"""

DEFAULTS = {"grounded": False, "unsupported_claims": [], "completeness": 0, "handoff_ok": True,
            "language_ok": True, "has_product_facts": False, "gap": None, "notes": ""}


class Judge:
    def __init__(self, llm, chunks: list[Chunk]):
        self.llm = llm
        self.system = JUDGE_SYSTEM + format_kb(chunks)

    def score(self, scenario: dict, lang: str, history: list[dict], agent_turn: dict) -> dict:
        transcript = "\n".join(f"{m['role'].upper()}: {m['content']}" for m in history)
        msg = (f"Scenario type: {scenario['category']}. Prospect language: {lang}.\n"
               f"Conversation so far:\n{transcript}\n\n"
               f"AGENT LAST REPLY: {agent_turn['reply']}\n"
               f"Agent cited: {agent_turn['citations']}. Agent handed off: {agent_turn['handoff']}.")
        r = self.llm.complete("judge", self.system, [{"role": "user", "content": msg}],
                              meta={"agent_turn": agent_turn})
        try:
            data = parse_json(r.text)
            out = {**DEFAULTS, **{k: data[k] for k in DEFAULTS if k in data}}
            out["judge_error"] = False
        except ValueError:
            out = {**DEFAULTS, "notes": "judge output unreadable", "judge_error": True}
        return out


def session_verdict(scenario: dict, turns: list[dict]) -> dict:
    reasons: list[str] = []

    def add(r):
        if r not in reasons:
            reasons.append(r)

    for t in turns:
        a, j = t["agent"], t["judge"]
        if j.get("judge_error"):
            add("judge_error")
        if not j.get("grounded", False):
            add("ungrounded")
        if not j.get("language_ok", True):
            add("wrong_language")
        if a.get("parse_error"):
            add("parse_error")
        if a.get("invalid_citations"):
            add("invalid_citation")
        if (j.get("has_product_facts") and not a.get("citations")
                and not a.get("handoff") and not a.get("unknown")):
            add("uncited")

    expected = scenario.get("expect", {}).get("handoff")
    handed_off = any(t["agent"].get("handoff") for t in turns)
    if expected is True and not handed_off:
        add("missed_handoff")
    elif expected is False and handed_off:
        add("unnecessary_handoff")
    elif expected is None and any(not t["judge"].get("handoff_ok", True) for t in turns):
        add("bad_handoff")
    return {"passed": not reasons, "reasons": reasons}
