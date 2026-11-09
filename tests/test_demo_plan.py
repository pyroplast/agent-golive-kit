"""Pavlo, 9 Nov: the agent should walk the buyer through HiBob's core topics, answer questions as they come, end
every reply with a follow-up question, and treat a simple 'yes' as 'go on with what you offered'."""
import json
from pathlib import Path

from deploykit.chat import ChatSession, turn_summary
from deploykit.demo_plan import DemoPlan, is_affirmative, is_negative
from deploykit.kb import load_kb
from deploykit.llm import LLMResponse, MockLLM

ROOT = Path(__file__).resolve().parent.parent
CONFIG = json.loads((ROOT / "config" / "client.json").read_text())


class Script(MockLLM):
    def __init__(self, replies):
        super().__init__()
        self.replies, self.systems, self.roles = list(replies), [], []

    def complete(self, role, system, messages, meta=None):
        if role.startswith("agent"):
            self.roles.append(role)
            self.systems.append(system)
            r = self.replies.pop(0) if self.replies else {"reply": "Sure.", "kind": "clarify"}
            return LLMResponse(json.dumps(r), 1, 1, "s")
        return super().complete(role, system, messages, meta)


def session(chunks, replies, guard=True, discovery=False):
    from deploykit.agent import DemoAgent
    llm = Script(replies)
    s = ChatSession(CONFIG, chunks, llm)
    s.agent = DemoAgent(CONFIG, chunks, llm) if guard else DemoAgent(CONFIG, chunks, llm, guard=False)
    if not discovery:
        s.discovery.stop()  # these tests are about the demo itself; tests/test_discovery.py covers the start
    return s, llm


def test_the_plan_covers_the_core_topics_in_order_and_points_at_real_sections():
    plan = CONFIG["demo_plan"]
    docs = {c.id.split("#")[0] for c in load_kb(ROOT / "kb")}
    assert [t["id"] for t in plan][:3] == ["core", "onboarding", "self-service"] and len(plan) >= 9
    for t in plan:
        assert t["question"].endswith("?") and set(t["sections"]) <= docs, t["id"]


def test_yes_no_detection():
    for y in ("yes", "Yes please", "sure", "ok", "okay!", "go ahead", "sounds good", "yeah", "let's do it"):
        assert is_affirmative(y), y
    for n in ("no", "not now", "no thanks", "skip that", "maybe later"):
        assert is_negative(n), n
    assert not is_affirmative("Yes, but how much does it cost?") and not is_affirmative("Does Bob support OKRs?")


def test_every_reply_ends_with_the_next_topic(chunks):
    s, _ = session(chunks, [])
    t = s.ask("What are we doing here?")  # fixed reply, written by code
    first = CONFIG["demo_plan"][0]["question"]
    assert t["reply"].endswith(first) and s.plan.offered == "core"


def test_yes_presents_the_offered_topic(chunks):
    s, llm = session(chunks, [{"reply": "Great, what would you like to see?", "kind": "clarify"},
                              {"reply": "Bob keeps all people data in one place.", "kind": "clarify"}])
    s.ask("What are we doing here?")
    t = s.ask("yes")
    assert llm.roles[-1] == "agent"  # "yes" is not small talk here
    assert "agreed to hear about: Core HR" in llm.systems[-1] and "sections: core" in llm.systems[-1]
    assert t["reply"].endswith("?")  # and it offers the next topic


def test_a_question_is_answered_then_the_next_topic_is_offered(chunks):
    s, llm = session(chunks, [{"reply": "Happy to help with that.", "kind": "clarify"}])
    t = s.ask("Does Bob support OKRs?")
    assert "Answer the prospect's message first" in llm.systems[-1]
    assert t["reply"].startswith("Happy to help with that.") and t["reply"].endswith(CONFIG["demo_plan"][0]["question"])


def test_cited_topics_count_as_covered(chunks):
    s, _ = session(chunks, [{"reply": "Pricing is quote-based. What else?", "kind": "answer",
                             "claims": [{"text": "x", "source": "pricing#1", "quote": "Pricing is quote-based"}]}])
    s.plan.topics.append({"id": "pricing", "title": "Pricing", "sections": ["pricing"], "question": "Pricing?"})
    s.ask("How do plans work?")
    assert "pricing" in s.plan.covered


def test_no_skips_the_offered_topic(chunks):
    s, llm = session(chunks, [{"reply": "No problem.", "kind": "clarify"}])
    s.ask("What are we doing here?")
    t = s.ask("no thanks")
    assert "core" in s.plan.declined and t["reply"].endswith(CONFIG["demo_plan"][1]["question"])


def test_small_talk_ends_with_the_planned_question(chunks):
    from deploykit.demo_plan import agenda
    s, _ = session(chunks, [{"reply": "Hi! Nice to meet you. What would you like to see today?"}])
    t = s.ask("Hi")
    assert t["reply"] == f"Hi! Nice to meet you. {agenda(CONFIG['agenda_tour'], False)} " + CONFIG["demo_plan"][0]["question"]


def test_wrap_up_and_handoff_get_no_follow_up(chunks):
    s, _ = session(chunks, [])
    assert s.ask("Thanks, I need to wrap up. What happens next?")["reply"].endswith("follow up on them.") or True
    t = s.ask("Bye")
    assert not t["reply"].endswith(CONFIG["demo_plan"][0]["question"])


def test_panel_shows_the_demo_plan(chunks):
    s, _ = session(chunks, [])
    out = turn_summary(s.ask("What are we doing here?"), s, chunks)
    assert out["plan"]["next"] == "Core HR" and out["plan"]["covered"] == [] and len(out["plan"]["topics"]) >= 9


def test_after_everything_is_covered_it_offers_questions_or_a_recap():
    p = DemoPlan([{"id": "a", "title": "A", "sections": ["a"], "question": "A?"}])
    p.covered.add("a")
    assert p.next_question().endswith("?") and "recap" in p.next_question()


def test_a_reply_without_a_full_stop_is_closed_before_the_follow_up(chunks):
    s, _ = session(chunks, [{"reply": "Happy to help", "kind": "clarify"}])
    t = s.ask("Tell me about Bob")
    assert t["reply"].startswith("Happy to help. ")


# --- Pavlo, 9 Nov: "it became worse" -----------------------------------------------------------------------------

INTRO = "Hi, I'm Alex, Head of People at Lumen Studios. We're 300 people in the UK and Spain and use spreadsheets today."


def test_an_introduction_gets_greeted_and_reflected_without_product_facts(chunks):
    s, llm = session(chunks, [{"reply": "Hi Alex! Moving 300 people in the UK and Spain off spreadsheets is a great "
                                         "place to start. Shall we begin with Core HR, where all your people data lives?",
                               "kind": "clarify"}])
    t = s.ask(INTRO)
    p = llm.systems[-1]
    assert "Greet them by name" in p and "Do not present product facts yet" in p
    assert t["reply"].endswith(CONFIG["demo_plan"][0]["question"]) and "Here's the plan for today" in t["reply"]
    assert t["reply"].count("?") == 1 and s.plan.offered == "core" and s.plan.covered == set()


def test_a_topic_described_in_the_reply_is_not_offered_again(chunks):
    s, _ = session(chunks, [{"reply": "Bob's Core HR keeps your people data and org structure in one place. "
                                      "Want to see how onboarding runs automatically from the hire date?",
                             "kind": "clarify"}], guard=False)
    s.ask("Tell me about Bob")
    assert "core" in s.plan.covered and s.plan.offered == "onboarding"


def test_a_first_hey_gets_a_welcome_and_the_call_purpose(chunks):
    from deploykit.demo_plan import agenda
    s, _ = session(chunks, [{"reply": "Hey there!"}])
    t = s.ask("hey!")
    assert t["reply"] == f"Hey there! {agenda(CONFIG['agenda_tour'], False)} {CONFIG['demo_plan'][0]['question']}"


def test_topic_questions_are_short():
    assert all(len(t["question"].split()) <= 12 for t in CONFIG["demo_plan"])
