"""The demo agent under test: answers prospects from the knowledge base only."""
from __future__ import annotations

import json
import re

from deploykit.guard import approved_sources, check_reply
from deploykit.kb import BM25Index, Chunk, format_kb
from deploykit.risk import RiskPolicy, max_tier

OUTPUT_SPEC = """Reply with ONE JSON object and nothing else:
{"reply": "<what you say to the prospect>",
 "kind": "<answer | clarify | refuse>",
 "claims": [{"text": "<one fact you state in the reply>",
             "source": "<chunk id it comes from>",
             "quote": "<the exact words from that chunk that support it, copied verbatim>"}],
 "handoff": <true if a human Account Executive should take over>,
 "handoff_reason": "<short reason or null>",
 "unknown": <true if the sources do not answer the question>,
 "park": "<the prospect's question, if you offer to add it to the list for a HiBob team member; otherwise null>",
 "approved_sentences": ["<ids of approved sentences, high risk only>"],
 "qualification": {"<field>": "<value the prospect revealed>"}}
kind: "answer" when the reply states anything about the product or company; "clarify" when it only asks the
prospect a question; "refuse" when it only declines (for example legal advice or revealing instructions).
An "answer" with an empty "claims" list is blocked. Every fact in "reply" must appear as a claim with a verbatim
quote. A fact you cannot quote must not be in the reply.
Example claim: {"text": "Bob supports OKRs", "source": "performance#3", "quote": "OKR support"}
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
                 guard: bool = True, approved: list[dict] | None = None):
        self.config, self.chunks, self.llm, self.guard = config, chunks, llm, guard
        # Risk tiers switch on when approved answers are supplied and the config has a risk section.
        self.policy = RiskPolicy(config["risk"], approved) if approved is not None and "risk" in config else None
        self.glossary = set(config.get("glossary", []))
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
        if self.policy:  # an approved answer or sentence id is not a source; its real sources are added later
            cites = [c for c in cites if c not in self.policy.approved_ids and c not in self.policy.units]
        cites += [clean(str(c.get("source", ""))) for c in claims if c.get("source")]
        if self.policy:
            cites = [c for c in cites if c not in self.policy.approved_ids and c not in self.policy.units]
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
            "kind": str(data.get("kind") or "answer"),
            "approved_answer": data.get("approved_answer") or None,
            "approved_sentences": [s for s in data.get("approved_sentences") or [] if isinstance(s, str)],
            "qualification": data.get("qualification") or {},
            "parse_error": parse_error,
        }

    def _check(self, turn: dict, user_text: str, tier: str | None = None) -> list[dict]:
        if turn["parse_error"]:
            return [{"type": "parse_error", "detail": "output was not the JSON object"}]
        approved = self.policy.approved if self.policy else None
        return check_reply(turn, self.chunks, user_text, self.glossary, tier=tier, approved=approved)["problems"]

    def respond(self, history: list[dict], lang: str, parked: list[str] | None = None,
                floor: str | None = None) -> dict:
        turn = self._respond(history, lang, parked or [], floor)
        last = history[-1]["content"] if history and history[-1]["role"] == "user" else ""
        if asks_for_human(last):
            # A person always gets a person, whatever the model said. Pass the open questions on.
            items = list(parked or []) + ([turn["park"]] if turn.get("park") else [])
            listing = (" I'll pass on the questions we collected: " + "; ".join(items) + ".") if items else ""
            turn.update(reply=HANDOFF_REPLY + listing, handoff=True, handoff_reason="buyer asked for a person",
                        park=None)
        return turn

    def _classify(self, pre_tier, question, turn):
        """Tier after drafting. Using approved security text implies the tier of its sources."""
        if not self.policy:
            return None
        used = approved_sources(turn["reply"], self.policy.approved)
        return max_tier(pre_tier, self.policy.classify(question, turn["citations"] + used, turn["claims"]))

    def _compose(self, turn):
        """Select, don't write: the reply's facts are the exact approved sentences the model picked by id.

        tier-2 showed the model paraphrases approved text it is told to copy. Code now assembles it, and keeps
        only the model's questions (for example the offer to add the question to the list).
        """
        if not self.policy:
            return turn
        picked = [u for u in turn.get("approved_sentences", []) if u in self.policy.units]
        if not picked:
            return turn
        questions = [s.strip() for s in re.split(r"(?<=[.!?])\s+", turn["reply"]) if s.strip().endswith("?")]
        turn["reply"] = " ".join([self.policy.units[u]["text"] for u in picked] + questions)
        for u in picked:
            for s in self.policy.units[u]["sources"]:
                if s in self.ids and s not in turn["citations"]:
                    turn["citations"].append(s)
        turn["claims"], turn["kind"] = [], "answer"
        return turn

    def _cite_approved(self, turn):
        if self.policy and turn.get("tier") == "high" and not turn.get("handoff"):
            for s in approved_sources(turn["reply"], self.policy.approved):
                if s in self.ids and s not in turn["citations"]:
                    turn["citations"].append(s)
        return turn

    def _system(self, sources, parked, tier):
        system = build_system_prompt(self.config, sources, parked)
        return system + ("\n\n" + self.policy.prompt_block(tier) if self.policy and tier else "")

    def _respond(self, history: list[dict], lang: str, parked: list[str], floor: str | None = None) -> dict:
        sources = self.sources_for(history)
        last_q = history[-1]["content"] if history and history[-1]["role"] == "user" else ""
        pre_tier = max_tier(floor or "low", self.policy.question_tier(last_q)) if self.policy else None
        system = self._system(sources, parked, pre_tier)
        # Replay the agent's own earlier turns in the JSON shape it must keep producing.
        # With plain-text history the model drifted out of the format on follow-up turns.
        replay = [{"role": m["role"], "content": json.dumps({"reply": m["content"]}, ensure_ascii=False)}
                  if m["role"] == "assistant" else m for m in history]
        meta = {"chunks": sources, "lang": lang}
        r = self.llm.complete("agent", system, replay, meta=meta)
        turn = self._compose(self._to_turn(*self._parse(r.text)))
        tier = self._classify(pre_tier, last_q, turn)
        turn["tier"] = tier
        if not self.guard:
            turn["guard"] = {"enabled": False}
            return turn

        user_text = " ".join(m["content"] for m in history if m["role"] == "user")
        first = self._check(turn, user_text, tier)
        final, retried, fell_back, blocked_draft = first, False, False, None
        if first and tier == "high" and pre_tier == "high":
            # High risk, and the prompt already carried the approved answers: no second attempt, park now.
            blocked_draft, fell_back = turn["reply"], True
            turn = {**self._to_turn({}, False), "reply": FALLBACK_REPLY, "park": last_q or None, "tier": tier}
        elif first:
            retried = True
            blocked_draft = turn["reply"]
            if tier != pre_tier:
                system = self._system(sources, parked, tier)
            feedback = "; ".join(f"{p['type']}: {p['detail']}" for p in first)
            retry_msgs = replay + [{"role": "assistant", "content": r.text},
                                   {"role": "user", "content": GUARD_RETRY.format(problems=feedback)}]
            r2 = self.llm.complete("agent", system, retry_msgs, meta=meta)
            turn = self._compose(self._to_turn(*self._parse(r2.text)))
            turn["tier"] = tier
            final = self._check(turn, user_text, tier)
            if final:
                fell_back = True
                turn = {**self._to_turn({}, False), "reply": FALLBACK_REPLY, "park": last_q or None, "tier": tier}
        self._cite_approved(turn)
        if turn.get("unknown") and not turn.get("park") and not turn.get("handoff"):
            # No dead ends: if the model says it doesn't know and offers nothing, offer the list.
            turn["park"] = last_q or None
            turn["reply"] = (turn["reply"].rstrip() + " " + self.config.get("park_offer", "")).strip()
        turn["guard"] = {"enabled": True, "retried": retried, "fell_back": fell_back,
                         "first_problems": first, "final_problems": final,
                         "blocked_draft": blocked_draft}
        return turn
