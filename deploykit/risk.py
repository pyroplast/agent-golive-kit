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
    "high": ("RISK TIER: high (legal, security, compliance, money). Do not write facts yourself. Put the ids of the "
             "approved sentences that answer the question in \"approved_sentences\" (one or several, from any answer); "
             "the reply is assembled from their exact text. In \"reply\" write only an optional question, such as the "
             "offer to add the question to the list. If the sentences do not fully answer the question, also set "
             "\"park\" and offer to add it to the list. Never give legal or tax advice."),
}


def max_tier(*tiers) -> str:
    known = [t for t in tiers if t in TIERS]
    return max(known, key=TIERS.index) if known else "low"


class RiskPolicy:
    def __init__(self, cfg: dict, approved: list[dict] | None = None):
        from deploykit.guard import _units
        self.units = {f"{a['id']}.{i}": {"text": u, "sources": a.get("sources", [])}
                      for a in approved or [] for i, u in enumerate(_units(a["text"]), 1)}
        self.approved_ids = {a["id"] for a in approved or []}
        self.keywords = {t: [k.lower() for k in cfg.get("keywords", {}).get(t, [])] for t in ("high", "medium")}
        self.section_tiers = cfg.get("section_tiers", {})
        self.default = cfg.get("default", "low")
        self.approved = list(approved or [])
        # Reviewed trigger phrases per approved answer: the only way code itself picks an answer to send.
        self.triggers = {a["id"]: [t.lower() for t in a.get("triggers", [])] for a in self.approved}

    def matching(self, question: str) -> list[str]:
        """Approved answers whose trigger phrases appear in the question, in file order."""
        low = (question or "").lower()
        return [aid for aid, ts in self.triggers.items()
                if any(re.search(r"(?<![a-z])" + re.escape(t) + r"(?![a-z])", low) for t in ts)]

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

    def prompt_block(self, tier: str, question: str | None = None) -> str:
        block = TIER_RULES[tier]
        if tier == "high":
            items = "\n".join(f"- [{uid}] {u['text']}" for uid, u in self.units.items()) or "- (none)"
            block += "\nApproved sentences (use their ids in approved_sentences):\n" + items
            match = self.matching(question or "")
            if match:
                ids = [uid for uid in self.units if uid.split(".")[0] in match]
                block += ("\nAPPROVED SENTENCES MATCHING THIS QUESTION (start from these): " + ", ".join(ids))
        return block
