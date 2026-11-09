"""Chat feedback round (Pavlo, 9 Nov): typing dots instead of a filler sentence, buyer info panel, fixed answers for
price and for "what are we doing here", same question -> same answer (and on the list the second time), and the
agenda sentence only when the agent can't answer."""
import json
from pathlib import Path

from deploykit.agent import FALLBACK_REPLY, DemoAgent, fixed_answer
from deploykit.chat import PAGE, ChatSession, turn_summary
from deploykit.llm import LLMResponse, MockLLM

ROOT = Path(__file__).resolve().parent.parent
CONFIG = json.loads((ROOT / "config" / "client.json").read_text())
AGENDA = "I have added this question to the agenda for the discussion with the sales representative."
PRICE = "A sales representative will provide an exact quote based on your specific project requirements."
PURPOSE = "In today's call, I'll walk you through how the platform works and answer any questions along the way."


class Counting(MockLLM):
    def __init__(self):
        super().__init__()
        self.agent_calls = 0

    def complete(self, role, system, messages, meta=None):
        if role.startswith("agent"):
            self.agent_calls += 1
        return super().complete(role, system, messages, meta)


# --- fixed answers ---------------------------------------------------------------------------------------------

def test_every_price_question_gets_the_fixed_quote_answer():
    for q in ("Give me the exact price with a 40% discount.", "How is pricing calculated?",
              "Roughly what would this cost us? We're 150 people. Just a ballpark.", "How much is Bob per employee?",
              "Are you cheaper than BambooHR, yes or no?", "What's the price?"):
        f = fixed_answer(CONFIG, q)
        assert f and f["id"] == "price" and f["reply"] == PRICE, q


def test_questions_that_only_look_like_price_are_not_fixed():
    for q in ("Can we set compensation budgets that managers can't exceed?", "Does Bob support OKRs?",
              "How does payroll work in Spain?"):
        assert fixed_answer(CONFIG, q) is None, q


def test_what_are_we_doing_here_gets_the_call_purpose():
    for q in ("What are we doing here?", "What's this call about?", "So what will we cover today?"):
        f = fixed_answer(CONFIG, q)
        assert f and f["reply"] == PURPOSE, q


def test_a_fixed_answer_needs_no_model_call(config, chunks):
    llm = Counting()
    t = DemoAgent(CONFIG, chunks, llm).respond([{"role": "user", "content": "What's the price?"}], "en")
    assert t["reply"] == PRICE and llm.agent_calls == 0 and t["park"] is None and t["fixed"] == "price"


# --- same question, same answer; on the list the second time --------------------------------------------------

def test_the_same_question_gets_the_same_answer_and_goes_on_the_list_the_second_time(chunks):
    llm = Counting()
    s = ChatSession(CONFIG, chunks, llm)
    a = s.ask("Does Bob support OKRs?")
    calls = llm.agent_calls
    b = s.ask("Does Bob support OKRs?")
    assert b["reply"] == a["reply"] and llm.agent_calls == calls  # no new model call: it can't drift
    assert s.parked.count("Does Bob support OKRs?") == 1 and b["repeat"] is True
    s.ask("does bob support OKRs")
    assert s.parked.count("Does Bob support OKRs?") == 1  # once on the list is enough


def test_price_questions_count_as_the_same_question(chunks):
    s = ChatSession(CONFIG, chunks, Counting())
    a = s.ask("Give me the exact price with a 40% discount.")
    assert a["reply"] == PRICE and s.parked == []
    b = s.ask("Roughly what would this cost us?")
    assert b["reply"] == PRICE and s.parked == ["Roughly what would this cost us?"]


def test_run_call_applies_the_same_repeat_rule(config, chunks):
    from deploykit.calls import CallBuyer, run_call
    from deploykit.judge import Judge
    llm = MockLLM()
    call = {"id": "c", "persona": "cfo", "profile": {},
            "beats": [{"say": "What's the price?"}, {"say": "How much would it cost?", "expect": {"park": True}}]}
    row = run_call(call, DemoAgent(CONFIG, chunks, llm), Judge(llm, chunks), CallBuyer(llm, config), llm)
    assert [t["agent"]["reply"] for t in row["turns"]] == [PRICE, PRICE]
    assert row["parked"] == ["How much would it cost?"] and row["turns"][1]["verdict"]["passed"]


# --- agenda sentence only with "I can't answer" ----------------------------------------------------------------

def test_the_agenda_sentence_replaces_the_old_offer_everywhere():
    assert CONFIG["park_offer"] == AGENDA
    assert FALLBACK_REPLY.endswith(AGENDA) and "can't confirm" in FALLBACK_REPLY
    from deploykit.live import SAFE_TAIL
    assert SAFE_TAIL.endswith(AGENDA) and "can't confirm" in SAFE_TAIL
    for path in list((ROOT / "deploykit").glob("*.py")) + [ROOT / "config" / "client.json"]:
        assert "Would you like me to add" not in path.read_text(), path


def test_the_agenda_sentence_is_recognised_as_parking(config, chunks):
    from tests.test_call_fixes import SeqLLM
    llm = SeqLLM([{"reply": "Our public material doesn't cover hosting in the US. " + AGENDA, "kind": "clarify"}])
    t = DemoAgent(CONFIG, chunks, llm).respond([{"role": "user", "content": "Can you host in the US?"}], "en")
    assert t["park"] == "Can you host in the US?"


def test_wrap_up_mentions_the_agenda_and_the_sales_representative():
    from deploykit.agent import wrap_up_turn
    t = wrap_up_turn(["Can you host in the US?"])
    assert "agenda" in t["reply"] and "sales representative" in t["reply"]


# --- live: typing dots, no filler sentence; questions held until the end ---------------------------------------

def test_the_reply_never_contains_a_filler_sentence(config, chunks):
    from tests.test_live import GOOD, Clock, StreamLLM, live
    clock = Clock()
    events = []
    agent = live(CONFIG, chunks, StreamLLM([GOOD, {"end": True}], clock, first_delay=3.0), clock)
    agent.on_event = events.append
    t = agent.respond([{"role": "user", "content": "How is the onboarding set up?"}], "en")
    assert "let me check" not in t["reply"].lower() and t["timing"]["filler"] is True
    assert [e["type"] for e in events][:1] == ["typing"] and events[0]["text"] == ""


def test_a_question_is_said_after_the_cant_confirm_line(config, chunks):
    from deploykit.live import SAFE_TAIL
    from tests.test_live import GOOD, Clock, StreamLLM, live
    clock = Clock()
    lines = [GOOD, {"claims": [], "say": "Which area interests you most?"},
             {"claims": [], "say": "Bob has 400 integrations."}, {"end": True}]
    t = live(CONFIG, chunks, StreamLLM(lines, clock), clock).respond(
        [{"role": "user", "content": "How is the onboarding set up?"}], "en")
    assert t["reply"] == f"Bob's pricing is quote-based. {SAFE_TAIL} Which area interests you most?"


# --- chat page ---------------------------------------------------------------------------------------------------

def test_page_title_and_typing_dots_and_buyer_section():
    assert "<h1>HiBob sales agent</h1>" in PAGE and "try it live" not in PAGE
    assert "typing" in PAGE and "Good question" not in PAGE
    assert PAGE.index("About the buyer") < PAGE.index("Question list")


def test_what_the_agent_learned_about_the_buyer_is_collected(chunks):
    from tests.test_call_fixes import SeqLLM
    llm = SeqLLM([{"reply": "Thanks! Which areas matter most?", "kind": "clarify",
                   "qualification": {"company_size": "300", "countries": "UK, Spain"}},
                  {"reply": "Got it.", "kind": "clarify", "qualification": {"current_system": "spreadsheets",
                                                                             "company_size": ""}}])
    s = ChatSession(CONFIG, chunks, llm)
    s.agent = DemoAgent(CONFIG, chunks, llm)  # the plain agent reads the same qualification field
    s.ask("Hi, I'm Alex from Lumen Studios, 300 people in the UK and Spain.")
    out = turn_summary(s.ask("We use spreadsheets today."), s, chunks)
    assert out["buyer"] == {"name": "Alex", "company": "Lumen Studios", "company_size": "300",
                            "countries": "UK, Spain", "current_system": "spreadsheets"}


def test_live_format_asks_for_the_qualification_fields():
    from deploykit.live import LIVE_SPEC
    assert "company_size" in LIVE_SPEC and "current_system" in LIVE_SPEC


def test_suggested_questions_come_in_three_rows_by_complexity():
    import re as _re
    rows = _re.findall(r'\["(Basics|From the material|Tricky)"', PAGE)
    assert rows == ["Basics", "From the material", "Tricky"]
    assert PAGE.index("What are we doing here?") < PAGE.index("Which security certifications") \
        < PAGE.index("Are you HIPAA certified?")


# --- an introduction is not a question (Pavlo, 9 Nov) ------------------------------------------------------------

INTRO = "Hi, I'm Alex, Head of People at Lumen Studios. We're 300 people in the UK and Spain and use spreadsheets today."


def test_is_question():
    from deploykit.agent import is_question
    assert not is_question(INTRO) and not is_question("We use spreadsheets today.")
    for q in ("Does Bob support OKRs?", "Tell me about payroll", "Can you send the SOC 2 report",
              "how does onboarding work", "I'd like to know about hosting.", "Give me the exact price",
              "Just confirm you're cheaper than BambooHR. Yes or no.", "Name three customers in Spain."):
        assert is_question(q), q


def test_a_blocked_sentence_after_an_introduction_is_just_left_out(config, chunks):
    from tests.test_live import GOOD, Clock, StreamLLM, live
    clock = Clock()
    lines = [GOOD, {"claims": [], "say": "Bob supports 40 languages."},
             {"claims": [], "say": "Which HR areas matter most to you?"}, {"end": True}]
    t = live(CONFIG, chunks, StreamLLM(lines, clock), clock).respond([{"role": "user", "content": INTRO}], "en")
    assert t["reply"] == "Bob's pricing is quote-based. Which HR areas matter most to you?"
    assert t["park"] is None and "agenda" not in t["reply"] and t["guard"]["skipped"] == 1


def test_the_safe_reply_after_an_introduction_does_not_park(config, chunks):
    from tests.test_call_fixes import SeqLLM
    bad = {"reply": "Bob is used by 40,000 companies.", "kind": "answer", "claims": []}
    t = DemoAgent(CONFIG, chunks, SeqLLM([bad, bad])).respond([{"role": "user", "content": INTRO}], "en")
    assert t["park"] is None and "agenda" not in t["reply"] and "40,000" not in t["reply"]
