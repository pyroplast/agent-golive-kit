"""The demo agent under test: answers prospects from the knowledge base only."""
from __future__ import annotations

import json
import re

from deploykit.guard import check_reply
from deploykit.kb import BM25Index, Chunk, format_kb

OUTPUT_SPEC = """Reply with ONE JSON object and nothing else:
{"reply": "<what you say to the prospect>",
 "claims": [{"text": "<one fact you state in the reply>",
             "source": "<chunk id it comes from>",
             "quote": "<the exact words from that chunk that support it, copied verbatim>"}],
 "handoff": <true if a human Account Executive should take over>,
 "handoff_reason": "<short reason or null>",
 "unknown": <true if the sources do not answer the question>,
 "park": "<the prospect's question, if you offer to add it to the list for a HiBob team member; otherwise null>",
 "qualification": {"<field>": "<value the prospect revealed>"}}
Every fact in "reply" must appear as a claim with a verbatim quote. A fact you cannot quote must not be in the reply.
Before sending, an automatic check compares every number and words like "certified", "guaranteed" or "annual"
against your quotes; anything not in a quote is blocked."""

GUARD_RETRY = """AUTOMATIC CHECK: your draft was blocked before it reached the prospect.
Problems: {problems}
Rewrite the reply. Keep only facts you can quote word for word from the sources, quote them exactly,
and remove any number or wording that is not in a quote. If you cannot support the answer, say so and hand off.
Reply with the JSON object only."""

FALLBACK_REPLY = ("I can't confirm that from HiBob's material, and I don't want to guess. Would you like me to add "
                  "this question to the list to discuss with a HiBob team member? We can keep going in the meantime.")

HANDOFF_REPLY = "Of course. I'm handing you over to a HiBob team member now."

_HUMAN = [re.compile(p, re.I) for p in (
    r"\b(talk|speak|chat)\s+(to|with)\s+(a\s+|an\s+|some\s*one|somebody|your\s+)?(real\s+|actual\s+)?"
    r"(human|person|someone|somebody|rep|representative|sales|agent|team)\b",
    r"\b(get|give|connect|transfer|put|pass)\s+me\s+(to\s+|with\s+|through\s+to\s+)?(a\s+|an\s+)?(real\s+)?"
    r"(human|person|someone|somebody|rep|representative|sales)\b",
    r"\b(real|actual|live)\s+(person|human)\b",
    r"\bhuman\s+(now|please)\b",
)]


def asks_for_human(text: str) -> bool:
    """Explicit request for a person. Deliberately ignores HR vocabulary like 'human resources'."""
    return any(p.search(text) for p in _HUMAN)


def parse_json(text: str) -> dict:
    dec = json.JSONDecoder()
    for i, ch in enumerate(text):
        if ch == "{":
            try:
                obj, _ = dec.raw_decode(text[i:])
                if isinstance(obj, dict):
                    return obj
            except json.JSONDecodeError:
                continue
    raise ValueError("no JSON object found")


def build_system_prompt(config: dict, sources: list[Chunk], parked: list[str] | None = None) -> str:
    rules = "\n".join(f"- {r}" for r in config["rules"])
    handoff = "\n".join(f"- {r}" for r in config["handoff_rules"])
    park = "\n".join(f"- {r}" for r in config.get("park_rules", []))
    current = "\n".join(f"- {q}" for q in parked or []) or "- (empty)"
    qual = "\n".join(f"- {q['field']}: {q['question']}" for q in config["qualification"])
    return f"""You are the {config['agent_name']} for {config['client']}'s product {config['product']}.
Goal: {config['goal']}

Rules:
{rules}

Hand off to a human only when:
{handoff}

Do not hand off for anything else. Instead, offer to add the question to the list for a HiBob team member
("{config.get('park_offer', '')}"), set "park" to the question, and keep the conversation going when:
{park}

Questions already on the list:
{current}

Qualification fields to learn naturally during the conversation (ask one at a time, only when it fits):
{qual}

{OUTPUT_SPEC}

SOURCES (cite by the id in square brackets):
{format_kb(sources)}"""


class DemoAgent:
    def __init__(self, config: dict, chunks: list[Chunk], llm, retrieval: str = "full", k: int = 6,
                 guard: bool = True):
        self.config, self.chunks, self.llm, self.guard = config, chunks, llm, guard
        self.retrieval, self.k = retrieval, k
        self.ids = {c.id for c in chunks}
        self.index = BM25Index(chunks) if retrieval == "bm25" else None

    def sources_for(self, history: list[dict]) -> list[Chunk]:
        if self.index is None:
            return self.chunks
        query = " ".join(m["content"] for m in history if m["role"] == "user")
        return [c for c, _ in self.index.search(query, self.k)] or self.chunks[: self.k]

    def _parse(self, text: str) -> tuple[dict, bool]:
        try:
            return parse_json(text), False
        except ValueError:
            return {"reply": text.strip()}, True

    def _to_turn(self, data: dict, parse_error: bool) -> dict:
        def clean(c):
            return c.strip().strip("[]").strip()
        claims = [c for c in data.get("claims") or [] if isinstance(c, dict)]
        cites = [clean(c) for c in data.get("citations") or [] if isinstance(c, str)]
        cites += [clean(str(c.get("source", ""))) for c in claims if c.get("source")]
        seen, ordered = set(), []
        for c in cites:
            if c and c not in seen:
                seen.add(c)
                ordered.append(c)
        return {
            "reply": str(data.get("reply", "")),
            "claims": claims,
            "citations": [c for c in ordered if c in self.ids],
            "invalid_citations": [c for c in ordered if c not in self.ids],
            "handoff": bool(data.get("handoff", False)),
            "handoff_reason": data.get("handoff_reason"),
            "unknown": bool(data.get("unknown", False)),
            "park": (str(data["park"]).strip() or None) if data.get("park") else None,
            "qualification": data.get("qualification") or {},
            "parse_error": parse_error,
        }

    def _check(self, turn: dict, user_text: str) -> list[dict]:
        if turn["parse_error"]:
            return [{"type": "parse_error", "detail": "output was not the JSON object"}]
        return check_reply(turn, self.chunks, user_text)["problems"]

    def respond(self, history: list[dict], lang: str, parked: list[str] | None = None) -> dict:
        turn = self._respond(history, lang, parked or [])
        last = history[-1]["content"] if history and history[-1]["role"] == "user" else ""
        if asks_for_human(last):
            # A person always gets a person, whatever the model said. Pass the open questions on.
            items = list(parked or []) + ([turn["park"]] if turn.get("park") else [])
            listing = (" I'll pass on the questions we collected: " + "; ".join(items) + ".") if items else ""
            turn.update(reply=HANDOFF_REPLY + listing, handoff=True, handoff_reason="buyer asked for a person",
                        park=None)
        return turn

    def _respond(self, history: list[dict], lang: str, parked: list[str]) -> dict:
        sources = self.sources_for(history)
        system = build_system_prompt(self.config, sources, parked)
        # Replay the agent's own earlier turns in the JSON shape it must keep producing.
        # With plain-text history the model drifted out of the format on follow-up turns.
        replay = [{"role": m["role"], "content": json.dumps({"reply": m["content"]}, ensure_ascii=False)}
                  if m["role"] == "assistant" else m for m in history]
        meta = {"chunks": sources, "lang": lang}
        r = self.llm.complete("agent", system, replay, meta=meta)
        turn = self._to_turn(*self._parse(r.text))
        if not self.guard:
            turn["guard"] = {"enabled": False}
            return turn

        user_text = " ".join(m["content"] for m in history if m["role"] == "user")
        first = self._check(turn, user_text)
        final, retried, fell_back, blocked_draft = first, False, False, None
        if first:
            retried = True
            blocked_draft = turn["reply"]
            feedback = "; ".join(f"{p['type']}: {p['detail']}" for p in first)
            retry_msgs = replay + [{"role": "assistant", "content": r.text},
                                   {"role": "user", "content": GUARD_RETRY.format(problems=feedback)}]
            r2 = self.llm.complete("agent", system, retry_msgs, meta=meta)
            turn = self._to_turn(*self._parse(r2.text))
            final = self._check(turn, user_text)
            if final:
                fell_back = True
                last = history[-1]["content"] if history and history[-1]["role"] == "user" else None
                turn = {**self._to_turn({}, False), "reply": FALLBACK_REPLY, "park": last}
        turn["guard"] = {"enabled": True, "retried": retried, "fell_back": fell_back,
                         "first_problems": first, "final_problems": final,
                         "blocked_draft": blocked_draft}
        return turn
