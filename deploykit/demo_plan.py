"""The demo plan: HiBob's core topics in the order a demo walks through them (config "demo_plan").

The agent presents one topic at a time, answers questions as they come, and ends every reply with a follow-up
question that offers the next topic. A simple "yes" means "go on with what you offered"; "no" skips it.
"""
from __future__ import annotations

import re

AFFIRM = {"yes", "yeah", "yep", "yup", "sure", "ok", "okay", "please", "go ahead", "go on", "sounds good", "lets do it",
          "lets see", "lets go", "absolutely", "of course", "definitely", "why not", "great", "perfect", "sure thing",
          "yes please", "please do", "show me", "yes sure", "ok sure", "sure go ahead", "yes go ahead", "ok lets do it",
          "yes lets do it", "sounds great", "that would be great", "id like that", "interested", "tell me more"}
NEGATE = {"no", "nope", "not now", "no thanks", "no thank you", "skip", "skip that", "skip it", "maybe later", "later",
          "not really", "not interested", "pass", "next", "something else"}
DONE = "Is there anything else you'd like to dig into, or shall I recap the open questions for the sales representative?"


def _clean(text: str) -> str:
    return " ".join(re.sub(r"[^a-z ]+", "", (text or "").lower().replace("’", "").replace("'", "")).split())


def is_affirmative(text: str) -> bool:
    return _clean(text) in AFFIRM


def is_negative(text: str) -> bool:
    return _clean(text) in NEGATE


class DemoPlan:
    def __init__(self, topics: list[dict]):
        self.topics = [dict(t) for t in topics]
        self.covered, self.declined, self.offered = set(), set(), None

    def topic(self, tid):
        return next((t for t in self.topics if t["id"] == tid), None)

    def upcoming(self, exclude=()):
        return next((t for t in self.topics if t["id"] not in self.covered | self.declined | set(exclude)), None)

    def next_question(self, exclude=()) -> str:
        t = self.upcoming(exclude)
        return t["question"] if t else DONE

    def before(self, text: str) -> dict:
        """What the agent is told for this turn: a topic to present (after a yes) and the follow-up to end with."""
        present = None
        if self.offered and is_affirmative(text):
            present = self.topic(self.offered)
        elif self.offered and is_negative(text):
            self.declined.add(self.offered)
        exclude = {present["id"]} if present else set()
        nxt = self.upcoming(exclude)
        return {"present": present, "next_id": nxt["id"] if nxt else None, "next_question": self.next_question(exclude),
                "covered": [t["title"] for t in self.topics if t["id"] in self.covered]}

    def after(self, turn: dict, guide: dict) -> None:
        docs = {c.split("#")[0] for c in turn.get("citations", [])}
        for t in self.topics:
            if set(t["sections"]) & docs:
                self.covered.add(t["id"])
        if guide.get("present"):
            self.covered.add(guide["present"]["id"])
        if not (turn.get("wrap_up") or turn.get("handoff")):
            self.offered = guide.get("next_id")

    def summary(self) -> dict:
        nxt = self.topic(self.offered)
        return {"topics": [{"title": t["title"], "status": "covered" if t["id"] in self.covered else
                            "skipped" if t["id"] in self.declined else "next" if t["id"] == self.offered else "to do"}
                           for t in self.topics],
                "covered": [t["title"] for t in self.topics if t["id"] in self.covered],
                "next": nxt["title"] if nxt else None}


def prompt_block(guide: dict) -> str:
    lines = ["DEMO PLAN. Your goal: walk the prospect through HiBob's core topics, one per reply, and answer their "
             "questions as they come.",
             "Covered so far: " + (", ".join(guide["covered"]) or "nothing yet") + ". Do not repeat those."]
    p = guide.get("present")
    if p:
        lines.append(f"The prospect agreed to hear about: {p['title']}. Present it now in two to four sentences, each "
                     f"backed by a quote from the sources (sections: {', '.join(p['sections'])}).")
    else:
        lines.append("Answer the prospect's message first.")
    lines.append(f'End your reply with this follow-up question, or a close variant: "{guide["next_question"]}"')
    return "\n".join(lines)
