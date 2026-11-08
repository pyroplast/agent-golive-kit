"""Model providers. Every provider exposes complete(role, system, messages, meta=None).

`role` is one of: agent, judge, buyer. Each role can use a different model.
`meta` is only read by MockLLM, so tests run without network or keys.
"""
from __future__ import annotations

import json
import threading
from dataclasses import dataclass

MAX_TOKENS = {"agent": 800, "judge": 900, "buyer": 200}

PARK_TRIGGERS = ("discount", "exact price", "price", "contract", "legal", "lawyer", "nda")


@dataclass
class LLMResponse:
    text: str
    input_tokens: int
    output_tokens: int
    model: str
    cached_tokens: int = 0  # part of input_tokens served from the provider's prompt cache


class MockLLM:
    """Deterministic stand-in for a model. Good enough to test the plumbing."""

    def stream(self, role, system, messages, meta=None, usage_out=None):
        """Streams the same reply as complete(), as JSON lines when the live format is asked for."""
        r = self.complete(role, system, messages, meta)
        text = r.text
        if "ONE LINE PER SENTENCE" in system:
            obj = json.loads(text)
            lines = [{"say": obj.get("reply", ""), "claims": obj.get("claims", [])},
                     {"end": True, **{k: obj.get(k) for k in ("kind", "park", "handoff", "unknown")}}]
            text = "".join(json.dumps(l) + "\n" for l in lines)
        if usage_out is not None:
            usage_out.update(input_tokens=r.input_tokens, output_tokens=r.output_tokens, model=r.model)
        for i in range(0, len(text), 40):
            yield text[i:i + 40]

    def complete(self, role, system, messages, meta=None):
        meta = meta or {}
        if role in ("agent", "agent_fast"):
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
        if any(t in low for t in PARK_TRIGGERS):
            out = {"reply": "I can't give that here. Shall I add it to the list for a HiBob team member?",
                   "citations": [], "handoff": False, "handoff_reason": None, "park": last,
                   "unknown": False, "qualification": {}}
        else:
            hits = BM25Index(meta.get("chunks", [])).search(last, k=1)
            if hits:
                c = hits[0][0]
                first = c.text.split(". ")[0].strip()
                out = {"reply": first, "claims": [{"text": first, "source": c.id, "quote": first}],
                       "handoff": False, "handoff_reason": None, "unknown": False,
                       "qualification": {}}
            else:
                out = {"reply": "Our public material doesn't cover that. Shall I add it to the list for a HiBob team member?",
                       "citations": [], "handoff": False, "handoff_reason": None, "park": last,
                       "unknown": True, "qualification": {}}
        return json.dumps(out, ensure_ascii=False)

    @staticmethod
    def _judge(meta):
        if meta.get("call"):
            return json.dumps({"contradictions": [], "qualification": {}, "notes": "mock call judge"})
        t = meta.get("agent_turn", {})
        grounded = bool(t.get("citations")) or t.get("handoff") or t.get("unknown") or t.get("park")
        return json.dumps({"grounded": bool(grounded), "unsupported_claims": [],
                           "completeness": 2 if t.get("citations") else 1,
                           "handoff_ok": True, "language_ok": True,
                           "has_product_facts": bool(t.get("citations")),
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

    def stream(self, role, system, messages, meta=None, usage_out=None):
        model = self.models.get(role) or self.models["agent"]
        resp = self.client.chat.completions.create(
            model=model, temperature=0, stream=True, stream_options={"include_usage": True},
            max_tokens=MAX_TOKENS.get(role, 800) * self.token_multiplier,
            messages=[{"role": "system", "content": system}, *messages],
        )
        for ch in resp:
            if getattr(ch, "choices", None):
                piece = getattr(ch.choices[0].delta, "content", None)
                if piece:
                    yield piece
            u = getattr(ch, "usage", None)
            if u is not None and usage_out is not None:
                usage_out.update(input_tokens=getattr(u, "prompt_tokens", 0) or 0,
                                 output_tokens=getattr(u, "completion_tokens", 0) or 0, model=model,
                                 cached_tokens=getattr(getattr(u, "prompt_tokens_details", None), "cached_tokens", 0) or 0)
        if usage_out is not None:
            usage_out.setdefault("model", model)

    def complete(self, role, system, messages, meta=None):
        model = self.models.get(role) or self.models["agent"]
        r = self.client.chat.completions.create(
            model=model, temperature=0,
            max_tokens=MAX_TOKENS.get(role, 800) * self.token_multiplier,
            messages=[{"role": "system", "content": system}, *messages],
        )
        text = r.choices[0].message.content or ""
        u = r.usage
        cached = getattr(getattr(u, "prompt_tokens_details", None), "cached_tokens", 0) or 0
        return LLMResponse(text, getattr(u, "prompt_tokens", 0) or 0,
                           getattr(u, "completion_tokens", 0) or 0, model, cached_tokens=cached)

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
        row = self.by_role.setdefault(role, {"input_tokens": 0, "cached_tokens": 0, "output_tokens": 0,
                                             "usd": 0.0, "calls": 0})
        row["input_tokens"] += r.input_tokens
        row["cached_tokens"] = row.get("cached_tokens", 0) + r.cached_tokens
        row["output_tokens"] += r.output_tokens
        row["calls"] += 1
        p = self.prices.get(r.model)
        if p:
            # Cached input is billed at the cache price when the provider has one (the judge resends the whole
            # knowledge base every call, so most of its input is cached). Without one, it costs the full rate.
            cached = min(r.cached_tokens, r.input_tokens) if "cache_read" in p else 0
            row["usd"] += ((r.input_tokens - cached) / 1e6 * p["input"] + cached / 1e6 * p["cache_read"] if cached
                           else r.input_tokens / 1e6 * p["input"]) + r.output_tokens / 1e6 * p["output"]
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

    def stream(self, role, system, messages, meta=None):
        if not hasattr(self.llm, "stream"):  # providers without streaming: one chunk
            r = self.complete(role, system, messages, meta)
            yield r.text
            return
        usage: dict = {}
        yield from self.llm.stream(role, system, messages, meta, usage_out=usage)
        self.tracker.add(role, LLMResponse("", usage.get("input_tokens", 0), usage.get("output_tokens", 0),
                                           usage.get("model", "unknown"), cached_tokens=usage.get("cached_tokens", 0)))
