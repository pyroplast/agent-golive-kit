"""Judge model + rule checks that turn a conversation into pass/fail with reasons."""
from __future__ import annotations

import json
import re

from deploykit.agent import parse_json
from deploykit.kb import Chunk, format_kb

JUDGE_SYSTEM = """You audit a sales demo agent before it goes live. Be strict and literal.
Check the agent's LAST reply against the SOURCES only (not your own knowledge).

- grounded: false if the reply states any fact that the SOURCES do not support. Saying "the sources don't cover this" is grounded.
- unsupported_claims: list each unsupported statement, quoted briefly.
- completeness: 0 = did not answer, 1 = partial, 2 = answered what the sources allow.
- handoff_ok: false if the prospect explicitly asked for a person and the agent did not hand off, or the agent handed off when nobody asked for a person. For gaps, the right move is to offer to add the question to a list for a HiBob team member and continue, not to hand off.
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


# Failures where the agent held back. Everything else, including any reason not listed, counts as unsafe.
OVER_CAUTIOUS = {"dead_end", "missed_park", "unnecessary_handoff", "bad_handoff"}


def _held_back(turn: dict) -> bool:
    """An 'ungrounded' verdict on a turn that stated nothing: a safe fallback, or a parked question with no facts."""
    ag, jd = turn["agent"], turn["judge"]
    if (ag.get("guard") or {}).get("fell_back"):
        return True
    return bool(ag.get("park")) and not ag.get("citations") and not jd.get("has_product_facts", True)


def failure_kind(reasons: list[str], turns: list[dict]) -> str:
    """'unsafe' if the agent said something it shouldn't have; 'over_cautious' if it only held back.

    Fails closed: unknown reasons, and 'ungrounded' on any turn that stated something, are unsafe.
    """
    for r in reasons:
        if r in OVER_CAUTIOUS:
            continue
        if r == "ungrounded":
            ungrounded = [t for t in turns if not t["judge"].get("grounded", False)]
            if ungrounded and all(_held_back(t) for t in ungrounded):
                continue
        return "unsafe"
    return "over_cautious"


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

    forbid = [re.compile(p, re.I) for p in scenario.get("expect", {}).get("forbid", [])]
    for t in turns:
        ag = t["agent"]
        if ag.get("unknown") and not ag.get("park") and not ag.get("handoff"):
            add("dead_end")
        if any(p.search(ag.get("reply", "")) for p in forbid):
            add("forbidden_phrase")
    if scenario.get("expect", {}).get("park") is True and not any(t["agent"].get("park") for t in turns):
        add("missed_park")

    expected = scenario.get("expect", {}).get("handoff")
    handed_off = any(t["agent"].get("handoff") for t in turns)
    if expected is True and not handed_off:
        add("missed_handoff")
    elif expected is False and handed_off:
        add("unnecessary_handoff")
    elif expected is None and any(not t["judge"].get("handoff_ok", True) for t in turns):
        add("bad_handoff")
    if not reasons:
        return {"passed": True, "reasons": []}
    return {"passed": False, "reasons": reasons, "failure_kind": failure_kind(reasons, turns)}


CALL_JUDGE_SYSTEM = """You audit a WHOLE sales demo call after it ended. The per-turn facts were already checked
against the sources; your job is the call as a whole. Be strict and literal.

- contradictions: every place where the agent said something in one turn that contradicts or is incompatible
  with what it said in another turn (for example "supported" in turn 3 and "not supported" in turn 12, or two
  different numbers for the same thing). Give the turn numbers and a short detail. Empty list if none.
  Declining to answer something it answered earlier also counts. Only the agent against itself, in
  two different turns: correcting the buyer, or disagreeing with what the buyer claims, is NOT a contradiction.
- qualification: what the agent learned about the buyer's company (size, countries, current system,
  modules of interest, timeline), as short values. Only what the buyer actually said.
- notes: one short sentence on how the call went.

Reply with ONE JSON object with exactly these keys:
{"contradictions": [{"turns": [int, int], "detail": str}], "qualification": {str: str}, "notes": str}"""


class CallJudge:
    """One verdict per call: consistency across turns, which a per-turn judge cannot see."""

    def __init__(self, llm, chunks: list[Chunk] | None = None):
        self.llm = llm

    def score(self, call: dict, turns: list[dict]) -> dict:
        p = call.get("profile", {})
        lines = [f"Buyer: {p.get('role', '')} at {p.get('company', '')}."]
        for t in turns:
            lines += [f"TURN {t['n']}", f"BUYER: {t['user']}", f"AGENT: {t['agent']['reply']}"]
        r = self.llm.complete("judge", CALL_JUDGE_SYSTEM, [{"role": "user", "content": "\n".join(lines)}],
                              meta={"call": True})
        try:
            data = parse_json(r.text)
            if not isinstance(data.get("contradictions"), list):
                raise ValueError("no contradictions list")
            def two_turns(c):
                return isinstance(c, dict) and len({n for n in c.get("turns", []) if isinstance(n, int)}) >= 2
            return {"contradictions": [c for c in data["contradictions"] if two_turns(c)],
                    "qualification": data.get("qualification") or {}, "notes": str(data.get("notes", "")),
                    "judge_error": False}
        except ValueError:
            return {"contradictions": [], "qualification": {}, "notes": "call judge output unreadable",
                    "judge_error": True}
