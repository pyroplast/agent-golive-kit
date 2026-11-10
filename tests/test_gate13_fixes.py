"""calls-gate-13 was blocked by three findings. Two had one cause: the rule that drops a repeated fact also dropped
the direct answer to a direct question ("Does it support webhooks?" got only a question back). The third: a decline
that carries a fact ("I can't share the SOC 2 report as it's available under NDA") passed the guard with no quote."""
import json
from pathlib import Path

from deploykit.guard import factual_sentences
from deploykit.kb import load_kb
from tests.test_live import Clock, StreamLLM, live

ROOT = Path(__file__).resolve().parent.parent
CONFIG = json.loads((ROOT / "config" / "client.json").read_text())
APPROVED = json.loads((ROOT / "config" / "approved_answers.json").read_text())
KB = load_kb(ROOT / "kb")

WEBHOOKS = {"claims": [{"source": "integrations#2", "quote": "Public HR API, which supports webhooks"}],
            "say": "Yes, the Public HR API supports webhooks."}
FIRST = "Yes, Bob supports custom integrations via the Public HR API, which includes webhook support."
LEDGER = [{"turn": 1, "text": "custom integrations via the Public HR API", "source": "integrations#2",
           "quote": "Custom integrations can be built on the Public HR API, which supports webhooks"}]


def _turn(question, lines, ledger=LEDGER, earlier=FIRST, **kw):
    clock = Clock()
    history = [{"role": "user", "content": "Do you offer an API for custom integrations?"},
               {"role": "assistant", "content": earlier}, {"role": "user", "content": question}]
    agent = live(CONFIG, KB, StreamLLM(lines + [{"end": True}], clock), clock, **kw)
    return agent.respond(history, "en", ledger=ledger)


# --- 1. a direct question gets its answer, even if the fact was given before -----------------------------------

def test_a_direct_question_about_a_fact_already_given_is_still_answered():
    t = _turn("Does it support webhooks?", [WEBHOOKS, {"claims": [], "say": "What would you like to trigger?"}])
    assert t["reply"].startswith("Yes, the Public HR API supports webhooks.")
    assert "integrations#2" in t["citations"]


def test_a_repeat_the_buyer_did_not_ask_about_is_still_dropped():
    t = _turn("ok, we mostly care about onboarding", [{"claims": [], "say": "Got it, onboarding first."}, WEBHOOKS])
    assert t["reply"] == "Got it, onboarding first." and t["guard"]["repeated"] == 1
    t = _turn("why do you care?", [{"claims": [], "say": "Good question, fair enough."}, WEBHOOKS])
    assert "webhooks" not in t["reply"] and t["guard"]["repeated"] == 1


# --- 2. a decline that carries a fact needs a quote like any other fact -----------------------------------------

def test_a_decline_with_a_reason_is_a_statement_of_fact():
    s = "I can't share the SOC 2 report directly as it's available under NDA."
    assert factual_sentences(s) == [s]
    s = "I'm unable to send it because HiBob only shares it with customers."
    assert factual_sentences(s) == [s]


def test_a_plain_decline_is_still_not_a_fact():
    for s in ("I can't confirm that from HiBob's material, and I don't want to guess.",
              "I can't share that because the material doesn't cover it.",
              "I can't give legal advice.",
              "I cannot send documents in this chat."):
        assert factual_sentences(s) == [], s


def test_asking_for_the_soc2_report_gets_the_approved_answer_not_an_uncited_decline():
    lines = [{"claims": [], "say": "I can't share the SOC 2 report directly as it's available under NDA."}]
    t = _turn("Great, can you send me the SOC 2 report now?", lines, ledger=[], approved=APPROVED)
    assert "The SOC 2 report is available on request under NDA" in t["reply"]
    assert t["citations"] and "I can't share the SOC 2 report directly" not in t["reply"]
