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


_COUNTRIES = ["UK", "United Kingdom", "England", "Scotland", "Ireland", "Spain", "France", "Germany", "Portugal",
              "Italy", "Netherlands", "Poland", "US", "USA", "United States", "Canada", "Mexico", "Brazil", "India",
              "Australia", "Israel", "Sweden", "Denmark", "Belgium", "Switzerland", "Austria"]
_SYSTEMS = ["spreadsheets", "spreadsheet", "excel", "google sheets", "bamboohr", "personio", "workday", "sage",
            "hibob", "rippling", "deel", "factorial", "paper", "email"]
_SIZE = re.compile(r"\b(\d[\d,]*)\s*(?:\+\s*)?(?:people|employees|staff|person|ppl|fte)", re.I)
_REFUSE = re.compile(r"\b(?:rather not|prefer not|skip|not now|no time|just (?:show|see|start|demo)|straight to|"
                     r"let'?s (?:just )?(?:see|start|get to|go to|jump)|show me the (?:demo|product)|"
                     r"don'?t want to (?:answer|share|say)|not sharing|none of your business)\b", re.I)


def learn_facts(text: str) -> dict:
    """Facts a prospect states about themselves, read by code (the panel and discovery use them)."""
    out = {}
    m = _SIZE.search(text)
    if m:
        out["company_size"] = m.group(1).replace(",", "")
    found = [c for c in _COUNTRIES if re.search(rf"\b{re.escape(c)}\b", text)]
    if found:
        out["countries"] = ", ".join(dict.fromkeys(found))
    low = text.lower()
    sys_ = next((x for x in _SYSTEMS if re.search(rf"\b{re.escape(x)}\b", low)), None)
    if sys_:
        out["current_system"] = "spreadsheets" if sys_ == "spreadsheet" else sys_
    return out


class Discovery:
    """Before the demo: a few questions about the client (Pavlo, 9 Nov). Asks only what isn't known yet, each
    question once, at most max_questions; stops for good if the client would rather skip it."""

    def __init__(self, fields: list[dict], max_questions: int = 3):
        self.fields = [dict(f) for f in fields]
        self.max_questions = max_questions
        self.asked, self.pending, self.active, self.declined = set(), None, True, False

    def question(self, field: str) -> str:
        f = next(x for x in self.fields if x["field"] == field)
        return f.get("ask") or f["question"]

    def next_field(self, known: dict):
        if not self.active or len(self.asked) >= self.max_questions:
            return None
        return next((f["field"] for f in self.fields if f["field"] not in known and f["field"] not in self.asked), None)

    @staticmethod
    def is_refusal(text: str) -> bool:
        return is_negative(text) or bool(_REFUSE.search(text))

    def stop(self, declined: bool = False):
        self.active, self.pending, self.declined = False, None, declined


def agenda(tour: str, discovery: bool) -> str:
    first = "a few quick questions about your team, then " if discovery else ""
    return f"Here's the plan for today: {first}a walkthrough of Bob ({tour}), and I'll answer your questions along the way."


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

    def before(self, text: str, intro: bool = False) -> dict:
        """What the agent is told for this turn: a topic to present (after a yes), two topics it may offer next,
        and whether this is the buyer's introduction."""
        present = None
        if self.offered and is_affirmative(text):
            present = self.topic(self.offered)
        elif self.offered and is_negative(text):
            self.declined.add(self.offered)
        exclude = {present["id"]} if present else set()
        nxt = self.upcoming(exclude)
        nxt2 = self.upcoming(exclude | ({nxt["id"]} if nxt else set()))
        return {"present": present, "intro": intro, "next_id": nxt["id"] if nxt else None,
                "next2_id": nxt2["id"] if nxt2 else None, "next_title": nxt["title"] if nxt else None,
                "next2_title": nxt2["title"] if nxt2 else None, "next_question": self.next_question(exclude),
                "next2_question": nxt2["question"] if nxt2 else DONE,
                "covered": [t["title"] for t in self.topics if t["id"] in self.covered]}

    @staticmethod
    def _mentions(topic: dict, text: str) -> int:
        low = text.lower()
        return sum(k in low for k in topic.get("keywords", [])) + (topic["title"].lower() in low)

    def after(self, turn: dict, guide: dict) -> None:
        reply = turn.get("reply", "")
        sentences = re.split(r"(?<=[.!?])\s+", reply.strip())
        questions = [x for x in sentences if x.rstrip().endswith("?")]
        statements = " ".join(x for x in sentences if not x.rstrip().endswith("?"))
        docs = {c.split("#")[0] for c in turn.get("citations", [])}
        for t in self.topics:
            # covered: a cited section, or the reply itself talked about it (Pavlo, 9 Nov: Core HR was described,
            # then offered again)
            if set(t["sections"]) & docs or (not guide.get("intro") and self._mentions(t, statements) >= 2):
                self.covered.add(t["id"])
        if guide.get("present"):
            self.covered.add(guide["present"]["id"])
        if turn.get("wrap_up") or turn.get("handoff"):
            return
        # offered next: the topic the closing question is about (the model words it itself)
        last_q = questions[-1] if questions else ""
        best = max((t for t in self.topics if t["id"] not in self.covered | self.declined),
                   key=lambda t: self._mentions(t, last_q), default=None)
        if best is not None and self._mentions(best, last_q) > 0:
            self.offered = best["id"]
        else:
            nxt = self.upcoming()
            self.offered = nxt["id"] if nxt else None

    def summary(self) -> dict:
        nxt = self.topic(self.offered)
        return {"topics": [{"title": t["title"], "status": "covered" if t["id"] in self.covered else
                            "skipped" if t["id"] in self.declined else "next" if t["id"] == self.offered else "to do"}
                           for t in self.topics],
                "covered": [t["title"] for t in self.topics if t["id"] in self.covered],
                "next": nxt["title"] if nxt else None}


def prompt_block(guide: dict) -> str:
    lines = ["DEMO PLAN. Your goal: walk the prospect through HiBob's core topics, one per reply, answer their "
             "questions as they come, and keep them engaged.",
             "Covered so far: " + (", ".join(guide["covered"]) or "nothing yet") + ". Do not repeat those."]
    p = guide.get("present")
    disc = guide.get("discovery")
    if guide.get("intro"):
        lines.append("The prospect just introduced themselves. Greet them by name if they gave it and show in one short "
                     "sentence that you understood their situation (size, countries, current tools). Do not present "
                     "product facts yet. Do not ask a question: the call agenda and the next question are added after "
                     "your reply.")
        return "\n".join(lines)
    if disc == "answer":
        lines.append("Before the demo you are getting to know the prospect, and they just answered your question. "
                     "Acknowledge their answer in a few words, without product facts. Do not ask a question: the next "
                     "one is added after your reply.")
        return "\n".join(lines)
    if disc == "question":
        lines.append("Answer the prospect's question in two to four sentences. Do not ask a question: the next one is "
                     "added after your reply.")
        return "\n".join(lines)
    elif p:
        lines.append(f"The prospect agreed to hear about: {p['title']}. Present it now in two to four sentences, each "
                     f"backed by a quote from the sources (sections: {', '.join(p['sections'])}).")
    else:
        lines.append("Answer the prospect's message first, in two to four sentences.")
    if guide.get("next_title"):
        alt = f" If your reply already covered it, offer {guide['next2_title']} instead." if guide.get("next2_title") else ""
        lines.append(f"End with ONE short, natural follow-up question in your own words that offers {guide['next_title']} "
                     f"next.{alt} Tie it to what they told you when you can.")
    else:
        lines.append(f'End by asking: "{guide["next_question"]}"')
    return "\n".join(lines)
