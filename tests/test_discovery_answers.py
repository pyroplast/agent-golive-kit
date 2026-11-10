"""Pavlo, 10 Nov (chat screenshot): during discovery "hello" was saved as the buyer's main interest and "wdym" as
their timeline, and a greeting in the middle of the call got "Hi! Nice to meet you." as if the call had just begun."""
import pytest

from deploykit.agent import is_clarification, small_talk_fallback
from tests.test_demo_plan import CONFIG, session

INTRO = "Hi, I'm Alex, Head of People at Lumen Studios. We're 300 people in the UK and Spain and use spreadsheets today."


@pytest.mark.parametrize("text", ["wdym", "wdym?", "What do you mean?", "what do u mean", "Sorry?", "huh?", "?",
                                  "I don't understand", "Not sure what you mean", "Can you rephrase that?",
                                  "come again?", "meaning?", "in what sense?", "I don't get it"])
def test_asking_what_the_agent_meant_is_a_clarification(text):
    assert is_clarification(text)


@pytest.mark.parametrize("text", ["Mostly onboarding, it's all manual", "By the end of Q1", "What do you mean by Core?",
                                  "What does Payroll Hub mean for Spain?", "hello", "yes", "We don't understand our headcount"])
def test_answers_and_real_questions_are_not_clarifications(text):
    assert not is_clarification(text)


def test_a_greeting_is_not_an_answer_to_a_discovery_question(chunks):
    s, _ = session(chunks, [{"reply": "Hi Alex.", "kind": "clarify"}, {"reply": "I'm here."}], discovery=True)
    s.ask(INTRO)
    assert s.discovery.pending == "modules"
    t = s.ask("hello")
    assert "modules" not in s.buyer
    assert s.discovery.pending == "modules" and t["reply"].endswith(s.discovery.question("modules"))


def test_wdym_gets_the_question_put_another_way_and_is_not_an_answer(chunks):
    s, llm = session(chunks, [{"reply": "Hi Alex.", "kind": "clarify"}, {"reply": "Onboarding, got it.", "kind": "clarify"}],
                     discovery=True)
    s.ask(INTRO)
    calls = len(llm.systems)
    t = s.ask("wdym")
    field = next(f for f in CONFIG["qualification"] if f["field"] == "modules")
    assert "modules" not in s.buyer and s.discovery.pending == "modules"
    assert t["reply"].endswith(field["clarify"]) and field["ask"] not in t["reply"]
    assert len(llm.systems) == calls                      # written by code, no model call
    assert t["park"] is None and not s.parked             # and it is not a question for the sales representative
    t = s.ask("Mostly onboarding, it's all manual")       # the real answer still lands, and discovery moves on
    assert s.buyer["modules"] == "Mostly onboarding, it's all manual"
    assert t["reply"].endswith(s.discovery.question("timeline"))


def test_what_do_you_mean_with_a_question_mark_is_handled_the_same_way(chunks):
    s, llm = session(chunks, [{"reply": "Hi Alex.", "kind": "clarify"}], discovery=True)
    s.ask(INTRO)
    calls = len(llm.systems)
    t = s.ask("What do you mean?")
    assert "modules" not in s.buyer and len(llm.systems) == calls and not s.parked
    assert t["reply"].endswith(next(f for f in CONFIG["qualification"] if f["field"] == "modules")["clarify"])


def test_a_greeting_mid_call_is_not_answered_like_a_first_meeting(chunks):
    # the model greets as if the call had just started: code replaces it
    s, _ = session(chunks, [{"reply": "Bob supports OKRs.", "kind": "answer", "claims": []},
                            {"reply": "Hi! Nice to meet you. What would you like to see today?"}], guard=False)
    s.ask("Does Bob support OKRs?")
    t = s.ask("hello")
    assert t["reply"].startswith("I'm here.")
    assert not t["reply"].lower().startswith(("hi", "hello", "hey")) and "nice to meet you" not in t["reply"].lower()


def test_who_are_you_mid_call_does_not_start_with_a_greeting(chunks):
    s, llm = session(chunks, [{"reply": "Bob supports OKRs.", "kind": "answer", "claims": []},
                              {"reply": "Hi there! I'm an AI assistant, nice to meet you."}], guard=False)
    s.ask("Does Bob support OKRs?")
    t = s.ask("who are you?")
    assert t["reply"].startswith("I'm HiBob's AI sales assistant")
    assert "already under way" in llm.systems[-1]


def test_the_first_greeting_of_a_call_is_still_a_greeting():
    assert small_talk_fallback("hello").startswith("Hi! Nice to meet you.")
    assert small_talk_fallback("hello", mid_call=True) == "I'm here. What would you like to look at next?"
    assert not small_talk_fallback("thanks", mid_call=True).lower().startswith("hi")
