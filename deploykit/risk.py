"""Risk tiers. The cost of being wrong decides how strict the checks are.

The tier of a reply is the HIGHEST of three signals, so a misclassification fails toward strictness:
1. the buyer's question (keyword lists per tier, from config),
2. the knowledge-base sections the reply cites (section tags, from config),
3. the claims themselves (a number makes a claim at least medium; a high-risk keyword makes it high).
A conversation's tier only goes up (the runner passes the running maximum as a floor).
"""
from __future__ import annotations

import re

TIERS = ("low", "medium", "high")
WEIGHTS = {"low": 1, "medium": 3, "high": 10}

TIER_RULES = {
    "low": ("RISK TIER: low. Paraphrase and combine facts from the sources. Do not add names, numbers or promises "
            "that the sources do not state."),
    "medium": ("RISK TIER: medium (commercial). Every number and name you state must be inside the words you quote. "
               "If you cannot quote it, offer to add the question to the list."),
    "high": ("RISK TIER: high (legal, security, compliance, money). Answer ONLY with one approved answer below, copied "
             "word for word, and set \"approved_answer\" to its id. You may add a short greeting or one question. "
             "If no approved answer fits, do not answer: offer to add the question to the list. Never give legal "
             "or tax advice."),
}


def max_tier(*tiers) -> str:
    known = [t for t in tiers if t in TIERS]
    return max(known, key=TIERS.index) if known else "low"


class RiskPolicy:
    def __init__(self, cfg: dict, approved: list[dict] | None = None):
        self.keywords = {t: [k.lower() for k in cfg.get("keywords", {}).get(t, [])] for t in ("high", "medium")}
        self.section_tiers = cfg.get("section_tiers", {})
        self.default = cfg.get("default", "low")
        self.approved = list(approved or [])

    def _keyword_tier(self, text: str) -> str:
        low = text.lower()
        for tier in ("high", "medium"):
            if any(re.search(r"\b" + re.escape(k), low) for k in self.keywords[tier]):
                return tier
        return self.default

    def question_tier(self, question: str) -> str:
        return self._keyword_tier(question)

    def section_tier(self, chunk_id: str) -> str:
        if chunk_id in self.section_tiers:
            return self.section_tiers[chunk_id]
        return self.section_tiers.get(chunk_id.split("#")[0], self.default)

    def claim_tier(self, claim: dict) -> str:
        text = str(claim.get("text", ""))
        tier = self._keyword_tier(text)
        if re.search(r"\d", text):
            tier = max_tier(tier, "medium")
        return tier

    def classify(self, question: str, citations: list[str], claims: list[dict]) -> str:
        return max_tier(self.question_tier(question),
                        *(self.section_tier(c) for c in citations),
                        *(self.claim_tier(c) for c in claims))

    def prompt_block(self, tier: str) -> str:
        block = TIER_RULES[tier]
        if tier == "high":
            items = "\n".join(f"- [{a['id']}] {a['text']}" for a in self.approved) or "- (none)"
            block += "\nApproved answers:\n" + items
        return block
