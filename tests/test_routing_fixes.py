"""Routing review (10 Nov): three code routes fired on the wrong questions.

1. Wrap-up: "We need to run payroll in the UK" ended the call ("need to run").
2. Fixed price reply: "Can we track cost centres?" got the sales-quote line; a two-part question lost its first half.
3. Handover: "Can I get a call with an account executive?" was not a request for a person.
"""
import json
from pathlib import Path

import pytest

from deploykit.agent import DemoAgent, asks_for_human, fixed_answer, is_wrap_up, unfixed_part
from deploykit.llm import LLMResponse

ROOT = Path(__file__).resolve().parent.parent
CONFIG = json.loads((ROOT / "config" / "client.json").read_text())
PRICE = "A sales representative will provide an exact quote based on your specific project requirements."


class SeqLLM:
    def __init__(self, replies):
        self.replies, self.calls = list(replies), []

    def complete(self, role, system, messages, meta=None):
        self.calls.append({"role": role, "messages": messages})
        return LLMResponse(self.replies.pop(0), 10, 10, "test")


# --- 1. wrap-up ------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("text", [
    "We need to run payroll in the UK and Spain. Does Bob cover both?",
    "Do employees have to go through SSO every time?",
    "What happens next after a new hire signs the contract?",
    "Can managers say goodbye to spreadsheets for reviews?",
    "Do we have to run reviews twice a year?",
    "Is that all included in Core?",
    "We've got to drop our old HRIS by March. How long does a migration take?",
])
def test_product_questions_are_not_a_wrap_up(text):
    assert not is_wrap_up(text)


@pytest.mark.parametrize("text", [
    "Thanks, I have to jump. What happens next?",
    "Thanks for the info. I need to wrap up — what’s the next step?",
    "I'm out of time, thanks for your time.",
    "Sorry, we need to run. Thanks!",
    "I've got to go now, bye!",
    "OK, that's all for today.",
    "That's all from me, thanks.",
    "What happens next?",
    "Goodbye!",
    "Great, see you soon.",
])
def test_real_endings_are_still_a_wrap_up(text):
    assert is_wrap_up(text)


# --- 2. fixed price reply ----------------------------------------------------------------------------------------

@pytest.mark.parametrize("text", [
    "How does Bob help us reduce hiring costs?",
    "Can we track cost centres in Bob?",
    "How many review cycles per month can we run?",
    "Does UK payroll handle pension fees and deductions?",
    "Can I quote you on the ISO certification?",
    "How much time does implementation take?",
    "What reporting would we get for a board pack on headcount and people costs?",
    "How much of the onboarding can be automated?",
    "Who is in charge of approving time off?",
])
def test_questions_that_mention_money_words_are_not_price_questions(text):
    assert fixed_answer(CONFIG, text) is None, text


@pytest.mark.parametrize("text", [
    "Give me the exact price with a 40% discount.",
    "How is pricing calculated?",
    "Roughly what would this cost us? We're 150 people. Just a ballpark.",
    "How much is Bob per employee?",
    "Are you cheaper than BambooHR, yes or no?",
    "What's the price?",
    "How much does it cost?",
    "What do you charge per employee per month?",
    "Can you send me a quote?",
    "Are there implementation fees?",
    "What does the Payroll add-on cost?",
])
def test_price_questions_still_get_the_fixed_reply(text):
    f = fixed_answer(CONFIG, text)
    assert f and f["id"] == "price", text


def test_unfixed_part_finds_the_other_half_of_a_two_part_question():
    assert unfixed_part(CONFIG, "Does payroll cover the UK, and how much is it?") == "Does payroll cover the UK"
    assert unfixed_part(CONFIG, "Where is our data hosted? And what's the price?") == "Where is our data hosted?"
    assert unfixed_part(CONFIG, "Roughly what would this cost us? We're 150 people. Just a ballpark.") is None
    assert unfixed_part(CONFIG, "Give me the exact per-employee price with a 40% discount, right now.") is None
    assert unfixed_part(CONFIG, "Are you cheaper than BambooHR, yes or no?") is None


def test_a_two_part_question_gets_its_answer_and_the_fixed_price_reply(config, chunks):
    answer = json.dumps({"reply": "Bob is hosted on AWS.", "kind": "answer", "claims": [
        {"text": "Bob is hosted on AWS", "source": "security#2", "quote": "AWS"}]})
    llm = SeqLLM([answer])
    turn = DemoAgent(config, chunks, llm).respond(
        [{"role": "user", "content": "Is Bob on AWS, and how much is it?"}], "en")
    assert turn["reply"].startswith("Bob is hosted on AWS.") and turn["reply"].endswith(PRICE)
    assert turn["fixed"] == "price" and "security#2" in turn["citations"]
    # the model is asked only the part that code does not answer, so it has no price question to improvise on
    assert llm.calls[0]["messages"][-1]["content"] == "Is Bob on AWS"


def test_a_plain_price_question_still_needs_no_model_call(config, chunks):
    llm = SeqLLM([])
    turn = DemoAgent(config, chunks, llm).respond([{"role": "user", "content": "What's the price?"}], "en")
    assert turn["reply"] == PRICE and not llm.calls


def test_a_two_part_question_is_not_a_repeat_of_an_earlier_price_question():
    from deploykit.calls import RepeatTracker
    r = RepeatTracker(CONFIG)
    assert r.key("What's the price?") == ("fixed", "price")
    assert r.key("Does payroll cover the UK, and how much is it?") != ("fixed", "price")


# --- 3. handover -------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("text", [
    "Can I get a call with an account executive?",
    "Let me talk to your manager",
    "Can you book me a meeting with sales?",
    "I'd like to speak to an account executive please",
    "Could we set up a call with someone from your team?",
])
def test_more_ways_of_asking_for_a_person(text):
    assert asks_for_human(text)


@pytest.mark.parametrize("text", [
    "Can our team talk to the API directly?",
    "Can an employee book a meeting with their manager in Bob?",
    "Can employees speak with a manager through the app?",
    "Do managers get a call with HR before a review?",
    "Does it replace our human resources system?",
])
def test_product_questions_about_people_are_not_a_request_for_one(text):
    assert not asks_for_human(text)
