"""Pavlo, 10 Nov (chat screenshot): the agent offered Core HR, the buyer said "bet", the agent presented Core HR
and then asked "Shall we start with Core HR?" again. Two causes: "bet" was not read as a yes, and the closing
question was chosen before the reply was written, so it could offer the topic the reply had just covered."""
import pytest

from deploykit.demo_plan import DemoPlan, is_affirmative
from tests.test_demo_plan import CONFIG, session
from tests.test_live import Clock, StreamLLM, live

CORE_Q = CONFIG["demo_plan"][0]["question"]
NEXT_Q = CONFIG["demo_plan"][1]["question"]
CORE = ("Core HR is the foundation of Bob, where all your people data, org structure and HR operations live in "
        "one system.")


@pytest.mark.parametrize("text", ["bet", "Bet!", "yea", "ya", "kk", "alright", "aight", "for sure", "go for it",
                                  "do it", "sgtm", "works for me", "fine", "cool", "yes let's start", "let's start"])
def test_casual_ways_of_saying_yes(text):
    assert is_affirmative(text), text


def test_a_question_or_a_longer_message_is_still_not_a_yes():
    for text in ("I bet it costs a lot", "fine, but what about payroll?", "cool, how does it work?"):
        assert not is_affirmative(text), text


def test_the_plan_knows_which_topic_a_question_offers_and_what_a_reply_covered():
    plan = DemoPlan(CONFIG["demo_plan"])
    assert plan.offers(CORE_Q) == "core" and plan.offers(NEXT_Q) == "onboarding"
    assert plan.offers("How many people are on your team?") is None
    assert "core" in plan.covered_by({"reply": CORE, "citations": []}, {})
    assert plan.is_stale_offer(CORE_Q, {"reply": CORE, "citations": []}, {})
    assert not plan.is_stale_offer(NEXT_Q, {"reply": CORE, "citations": []}, {})


def test_bet_presents_the_offered_topic_and_then_offers_the_next_one(chunks):
    s, llm = session(chunks, [{"reply": CORE + " " + CORE_Q, "kind": "answer"}], guard=False)
    assert s.ask("What are we doing here?")["reply"].endswith(CORE_Q)  # written by code; offers Core HR
    t = s.ask("bet")
    assert "The prospect agreed to hear about: Core HR" in llm.systems[-1]
    assert t["reply"].startswith(CORE) and CORE_Q not in t["reply"] and t["reply"].endswith("?")
    assert s.plan.offered != "core" and "core" in s.plan.covered


def test_a_reply_that_covers_the_topic_never_offers_it_again_whatever_the_buyer_said(chunks):
    # an unrecognised "yes": the model presents Core HR anyway and repeats the offer it was given
    s, _ = session(chunks, [{"reply": CORE + " " + CORE_Q, "kind": "answer"}], guard=False)
    s.ask("What are we doing here?")
    t = s.ask("mm go on then I guess")
    assert t["reply"] == CORE + " " + NEXT_Q


def test_live_agent_holds_back_a_closing_question_the_filter_rejects(config, chunks):
    clock = Clock()
    lines = [{"say": "Bob's pricing is quote-based.", "claims": [
                {"text": "Pricing is quote-based", "source": "pricing#1", "quote": "Pricing is quote-based"}]},
             {"say": "Shall we look at pricing?"}, {"end": True}]
    agent = live(config, chunks, StreamLLM(lines, clock), clock, fast=False)
    seen, said = [], []
    agent.question_filter = lambda turn, q: seen.append((turn["citations"], q)) or False
    agent.on_event = lambda e: said.append(e["text"]) if e["type"] == "say" else None
    t = agent.respond([{"role": "user", "content": "Tell me about how Bob is sold."}], "en")
    assert seen == [(["pricing#1"], "Shall we look at pricing?")]
    assert "Shall we look at pricing?" not in t["reply"] and "Shall we look at pricing?" not in said
