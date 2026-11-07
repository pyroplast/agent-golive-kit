"""The demo agent under test: answers prospects from the knowledge base only."""
from __future__ import annotations

import json

from deploykit.kb import BM25Index, Chunk, format_kb

OUTPUT_SPEC = """Reply with ONE JSON object and nothing else:
{"reply": "<what you say to the prospect>",
 "citations": ["<chunk id you relied on>", ...],
 "handoff": <true if a human Account Executive should take over>,
 "handoff_reason": "<short reason or null>",
 "unknown": <true if the sources do not answer the question>,
 "qualification": {"<field>": "<value the prospect revealed>"}}"""


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


def build_system_prompt(config: dict, sources: list[Chunk]) -> str:
    rules = "\n".join(f"- {r}" for r in config["rules"])
    handoff = "\n".join(f"- {r}" for r in config["handoff_rules"])
    qual = "\n".join(f"- {q['field']}: {q['question']}" for q in config["qualification"])
    return f"""You are the {config['agent_name']} for {config['client']}'s product {config['product']}.
Goal: {config['goal']}

Rules:
{rules}

Hand off to a human when:
{handoff}

Qualification fields to learn naturally during the conversation (ask one at a time, only when it fits):
{qual}

{OUTPUT_SPEC}

SOURCES (cite by the id in square brackets):
{format_kb(sources)}"""


class DemoAgent:
    def __init__(self, config: dict, chunks: list[Chunk], llm, retrieval: str = "full", k: int = 6):
        self.config, self.chunks, self.llm = config, chunks, llm
        self.retrieval, self.k = retrieval, k
        self.ids = {c.id for c in chunks}
        self.index = BM25Index(chunks) if retrieval == "bm25" else None

    def sources_for(self, history: list[dict]) -> list[Chunk]:
        if self.index is None:
            return self.chunks
        query = " ".join(m["content"] for m in history if m["role"] == "user")
        return [c for c, _ in self.index.search(query, self.k)] or self.chunks[: self.k]

    def respond(self, history: list[dict], lang: str) -> dict:
        sources = self.sources_for(history)
        system = build_system_prompt(self.config, sources)
        # Replay the agent's own earlier turns in the JSON shape it must keep producing.
        # With plain-text history the model drifted out of the format on follow-up turns.
        replay = [{"role": m["role"], "content": json.dumps({"reply": m["content"]}, ensure_ascii=False)}
                  if m["role"] == "assistant" else m for m in history]
        r = self.llm.complete("agent", system, replay, meta={"chunks": sources, "lang": lang})
        try:
            data = parse_json(r.text)
            parse_error = False
        except ValueError:
            data, parse_error = {"reply": r.text.strip()}, True
        cites = [c.strip().strip("[]").strip() for c in data.get("citations") or [] if isinstance(c, str)]
        return {
            "reply": str(data.get("reply", "")),
            "citations": [c for c in cites if c in self.ids],
            "invalid_citations": [c for c in cites if c not in self.ids],
            "handoff": bool(data.get("handoff", False)),
            "handoff_reason": data.get("handoff_reason"),
            "unknown": bool(data.get("unknown", False)),
            "qualification": data.get("qualification") or {},
            "parse_error": parse_error,
        }
