"""Model providers. Every provider exposes complete(role, system, messages, meta=None).

`role` is one of: agent, judge, buyer. Each role can use a different model.
`meta` is only read by MockLLM, so tests run without network or keys.
"""
from __future__ import annotations

import json
import threading
from dataclasses import dataclass

MAX_TOKENS = {"agent": 800, "judge": 900, "buyer": 200}

HANDOFF_TRIGGERS = ("discount", "exact price", "contract", "legal", "lawyer",
                    "speak to", "talk to a", "human", "nda")


@dataclass
class LLMResponse:
    text: str
    input_tokens: int
    output_tokens: int
    model: str


class MockLLM:
    """Deterministic stand-in for a model. Good enough to test the plumbing."""

    def complete(self, role, system, messages, meta=None):
        meta = meta or {}
        if role == "agent":
            text = self._agent(messages, meta)
        elif role == "judge":
            text = self._judge(meta)
        else:
            text = "DONE"
        return LLMResponse(text, len(system) // 4, len(text) // 4, "mock")

    @staticmethod
    def _agent(messages, meta):
        from deploykit.kb import BM25Index  # local import avoids a cycle
        last = messages[-1]["content"]
        low = last.lower()
        if any(t in low for t in HANDOFF_TRIGGERS):
            out = {"reply": "I'll connect you with an Account Executive for that.",
                   "citations": [], "handoff": True, "handoff_reason": "pricing or terms",
                   "unknown": False, "qualification": {}}
        else:
            hits = BM25Index(meta.get("chunks", [])).search(last, k=1)
            if hits:
                c = hits[0][0]
                out = {"reply": c.text.split(". ")[0].strip(), "citations": [c.id],
                       "handoff": False, "handoff_reason": None, "unknown": False,
                       "qualification": {}}
            else:
                out = {"reply": "Our public material doesn't cover that. Want me to connect you with the team?",
                       "citations": [], "handoff": True, "handoff_reason": "not in sources",
                       "unknown": True, "qualification": {}}
        return json.dumps(out, ensure_ascii=False)

    @staticmethod
    def _judge(meta):
        t = meta.get("agent_turn", {})
        grounded = bool(t.get("citations")) or t.get("handoff") or t.get("unknown")
        return json.dumps({"grounded": bool(grounded), "unsupported_claims": [],
                           "completeness": 2 if t.get("citations") else 1,
                           "handoff_ok": True, "language_ok": True,
                           "gap": "not covered by sources" if t.get("unknown") else None,
                           "notes": "mock judge"})


class AnthropicLLM:
    """Claude through the Anthropic API. Needs ANTHROPIC_API_KEY."""

    def __init__(self, models: dict, client=None):
        self.models = models
        if client is None:
            import anthropic
            client = anthropic.Anthropic()
        self.client = client

    def complete(self, role, system, messages, meta=None):
        model = self.models[role]
        r = self.client.messages.create(
            model=model, max_tokens=MAX_TOKENS.get(role, 800),
            system=[{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
            messages=messages,
        )
        text = "".join(b.text for b in r.content if getattr(b, "type", "") == "text")
        return LLMResponse(text, r.usage.input_tokens, r.usage.output_tokens, model)


class BedrockLLM:
    """Claude through Amazon Bedrock (Converse API). Uses normal AWS credentials."""

    def __init__(self, models: dict, client=None, region: str = "eu-west-1"):
        self.models = models
        if client is None:
            import boto3
            client = boto3.client("bedrock-runtime", region_name=region)
        self.client = client

    def complete(self, role, system, messages, meta=None):
        model = self.models[role]
        r = self.client.converse(
            modelId=model,
            system=[{"text": system}],
            messages=[{"role": m["role"], "content": [{"text": m["content"]}]} for m in messages],
            inferenceConfig={"maxTokens": MAX_TOKENS.get(role, 800)},
        )
        text = "".join(p.get("text", "") for p in r["output"]["message"]["content"])
        u = r.get("usage", {})
        return LLMResponse(text, u.get("inputTokens", 0), u.get("outputTokens", 0), model)


NEBIUS_BASE_URL = "https://api.tokenfactory.nebius.com/v1/"


class OpenAICompatLLM:
    """Any OpenAI-compatible endpoint. Default: Nebius Token Factory (open-weight models)."""

    def __init__(self, models: dict, client=None, base_url: str = NEBIUS_BASE_URL,
                 api_key_env: str = "NEBIUS_API_KEY", token_multiplier: int = 3):
        self.models = models
        # Reasoning models spend part of max_tokens thinking, so give them headroom.
        self.token_multiplier = token_multiplier
        if client is None:
            import os

            from openai import OpenAI
            client = OpenAI(base_url=base_url, api_key=os.environ[api_key_env])
        self.client = client

    def complete(self, role, system, messages, meta=None):
        model = self.models[role]
        r = self.client.chat.completions.create(
            model=model, temperature=0,
            max_tokens=MAX_TOKENS.get(role, 800) * self.token_multiplier,
            messages=[{"role": "system", "content": system}, *messages],
        )
        text = r.choices[0].message.content or ""
        u = r.usage
        return LLMResponse(text, getattr(u, "prompt_tokens", 0) or 0,
                           getattr(u, "completion_tokens", 0) or 0, model)

    def list_models(self) -> list[str]:
        return sorted(m.id for m in self.client.models.list().data)


class CostTracker:
    """Adds up tokens per role. Prices are USD per million tokens, per model."""

    def __init__(self, prices: dict | None = None):
        self.prices = prices or {}
        self.by_role: dict = {}
        self.unpriced: set = set()
        self._lock = threading.Lock()

    def add(self, role: str, r: LLMResponse):
        with self._lock:
            self._add(role, r)

    def _add(self, role: str, r: LLMResponse):
        row =self.by_role.setdefault(role, {"input_tokens": 0, "output_tokens": 0, "usd": 0.0, "calls": 0})
        row["input_tokens"] += r.input_tokens
        row["output_tokens"] += r.output_tokens
        row["calls"] += 1
        p = self.prices.get(r.model)
        if p:
            row["usd"] += r.input_tokens / 1e6 * p["input"] + r.output_tokens / 1e6 * p["output"]
        elif r.model != "mock":
            self.unpriced.add(r.model)

    def summary(self) -> dict:
        return {"by_role": self.by_role,
                "total_usd": round(sum(v["usd"] for v in self.by_role.values()), 4),
                "unpriced_models": sorted(self.unpriced)}


class TrackedLLM:
    """Wraps any provider and records cost."""

    def __init__(self, llm, tracker: CostTracker):
        self.llm, self.tracker = llm, tracker

    def complete(self, role, system, messages, meta=None):
        r = self.llm.complete(role, system, messages, meta)
        self.tracker.add(role, r)
        return r
