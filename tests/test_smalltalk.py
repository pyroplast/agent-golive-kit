"""Pavlo, 9 Nov: 'yes' twice looped; [automation#2] showed in the reply; greetings and small talk sounded stiff."""
import json
from pathlib import Path

from deploykit.agent import DemoAgent, is_small_talk
from deploykit.chat import ChatSession
from deploykit.llm import LLMResponse, MockLLM

ROOT = Path(__file__).resolve().parent.parent
CONFIG = json.loads((ROOT / "config" / "client.json").read_text())


class Script(MockLLM):
    """Agent replies from a list; counts calls by role."""

    def __init__(self, replies):
        super().__init__()
        self.replies, self.roles = list(replies), []

    def complete(self, role, system, messages, meta=None):
        if role.startswith("agent"):
            self.roles.append(role)
            r = self.replies.pop(0) if self.replies else {"reply": "OK.", "kind": "clarify"}
            return LLMResponse(json.dumps(r), 1, 1, "s")
        return super().complete(role, system, messages, meta)


# 1. "yes" is an answer, not a repeated question -----------------------------------------------------------------

def test_yes_twice_gets_two_fresh_replies(chunks):
    llm = Script([{"reply": "Would you like to see onboarding?", "kind": "clarify"},
                  {"reply": "Great, which team should we start with?", "kind": "clarify"},
                  {"reply": "Perfect, shall we look at the task list next?", "kind": "clarify"}])
    s = ChatSession(CONFIG, chunks, llm)
    s.agent = DemoAgent(CONFIG, chunks, llm)
    s.ask("How does onboarding work?")
    a, b = s.ask("yes"), s.ask("yes")
    assert a["reply"] != b["reply"] and not b.get("repeat") and s.parked == []


def test_prompt_says_to_follow_through_on_yes(chunks):
    from deploykit.agent import build_system_prompt
    assert "says yes" in build_system_prompt(CONFIG, chunks).lower()


# 2. citation markers stay out of the reply text -----------------------------------------------------------------

def test_inline_markers_are_removed_from_the_reply_but_kept_as_citations(chunks):
    llm = Script([{"reply": "Pricing is quote-based [pricing#1]. Anything else?", "kind": "answer",
                   "claims": [{"text": "Pricing is quote-based", "source": "pricing#1", "quote": "Pricing is quote-based"}]}])
    t = DemoAgent(CONFIG, chunks, llm, guard=False).respond(
        [{"role": "user", "content": "How do plans work?"}], "en")
    assert t["reply"] == "Pricing is quote-based. Anything else?" and "pricing#1" in t["citations"]


def test_live_sentences_have_no_markers(config, chunks):
    from tests.test_live import Clock, StreamLLM, live
    clock = Clock()
    line = {"claims": [{"source": "pricing#1", "quote": "Pricing is quote-based"}],
            "say": "Pricing is quote-based [pricing#1]."}
    events = []
    agent = live(CONFIG, chunks, StreamLLM([line, {"end": True}], clock), clock)
    agent.on_event = events.append
    t = agent.respond([{"role": "user", "content": "How do plans work?"}], "en")
    assert t["reply"] == "Pricing is quote-based." and "[" not in "".join(e["text"] for e in events)


# 3. small talk --------------------------------------------------------------------------------------------------

def test_small_talk_is_recognised():
    for q in ("Hi", "hello!", "Hey there", "Good morning", "Hi, how are you?", "how are you doing today",
              "Nice to meet you", "Thanks!", "who are you?", "Are you a bot?", "Are you human?", "Hi, I'm Alex"):
        assert is_small_talk(q), q
    for q in ("Hi, I'm Alex from Lumen Studios, 300 people in the UK and Spain.", "Does Bob support OKRs?",
              "Can I talk to a human?", "yes", "Hi, which certifications do you have?"):
        assert not is_small_talk(q), q


def test_small_talk_gets_a_friendly_reply_from_the_fast_model(chunks):
    llm = Script([{"reply": "I'm doing well, thanks for asking! What would you like to see in Bob today?"}])
    t = DemoAgent(CONFIG, chunks, llm).respond([{"role": "user", "content": "Hi, how are you?"}], "en")
    assert t["reply"].startswith("I'm doing well") and llm.roles == ["agent_fast"]
    assert t["small_talk"] is True and t["park"] is None and not t["handoff"] and t["citations"] == []


def test_a_small_talk_reply_with_product_claims_is_replaced(chunks):
    llm = Script([{"reply": "Hi! Bob is used by 4,000 companies and costs less than BambooHR."}])
    t = DemoAgent(CONFIG, chunks, llm).respond([{"role": "user", "content": "Hi"}], "en")
    assert "4,000" not in t["reply"] and t["reply"].startswith("Hi") and t["guard"]["fell_back"] is True


def test_are_you_human_is_small_talk_not_a_handoff(chunks):
    llm = Script([{"reply": "I'm HiBob's AI assistant. What would you like to see first?"}])
    t = DemoAgent(CONFIG, chunks, llm).respond([{"role": "user", "content": "Are you human?"}], "en")
    assert not t["handoff"] and "AI" in t["reply"]


def test_small_talk_keeps_what_the_buyer_said(chunks):
    llm = Script([{"reply": "Nice to meet you, Alex! What brings you here today?"}])
    s = ChatSession(CONFIG, chunks, llm)
    s.ask("Hi, I'm Alex")
    assert s.buyer.get("name") == "Alex" and len(s.history) == 2


def test_bye_ends_the_call_with_the_recap(chunks):
    t = DemoAgent(CONFIG, chunks, Script([])).respond([{"role": "user", "content": "Bye!"}], "en",
                                                       parked=["Do you integrate with Workday?"])
    assert t["wrap_up"] and "Workday" in t["reply"]
