"""Buyer side of the conversation.

Scripted follow-ups are used when present, so runs stay comparable for regression.
Scenarios with "simulate": true let a model play the buyer for open-ended turns.
"""
from __future__ import annotations

BUYER_SYSTEM = """You are role-playing a B2B software buyer in a live product demo.
Persona: {persona}. You care about: {cares}.
Your goal in this conversation: {goal}
Write the buyer's next message only, in {language}, under 40 words.
If your goal is met or the agent handed you to a human, reply with exactly DONE."""

LANG_NAMES = {"en": "English"}


class Buyer:
    def __init__(self, llm, config: dict):
        self.llm, self.config = llm, config

    def next_message(self, scenario: dict, lang: str, history: list[dict], turn_index: int):
        scripted = (scenario.get("followups") or {}).get(lang) or []
        if turn_index - 1 < len(scripted):
            return scripted[turn_index - 1]
        if not scenario.get("simulate"):
            return None
        p = self.config["personas"][scenario["persona"]]
        system = BUYER_SYSTEM.format(persona=p["label"], cares=", ".join(p["cares_about"]),
                                     goal=scenario.get("goal", "get your questions answered"),
                                     language=LANG_NAMES.get(lang, lang))
        # The model plays the buyer, so roles are flipped.
        flipped = [{"role": "assistant" if m["role"] == "user" else "user", "content": m["content"]}
                   for m in history]
        if not flipped or flipped[0]["role"] != "user":
            flipped.insert(0, {"role": "user", "content": "(The demo call starts.)"})
        r =self.llm.complete("buyer", system, flipped)
        text = r.text.strip()
        return None if not text or text.upper().startswith("DONE") else text
