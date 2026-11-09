"""Pavlo, 9 Nov: present the call agenda at the start, then collect key facts about the client before the demo.
If the client doesn't want to answer, suggest going straight to the demo."""
from deploykit.chat import ChatSession
from deploykit.demo_plan import Discovery, learn_facts
from tests.test_demo_plan import CONFIG, Script, session

INTRO = "Hi, I'm Alex, Head of People at Lumen Studios. We're 300 people in the UK and Spain and use spreadsheets today."
CORE_Q = CONFIG["demo_plan"][0]["question"]


def test_facts_are_read_from_an_introduction():
    f = learn_facts(INTRO)
    assert f["company_size"] == "300" and "UK" in f["countries"] and "Spain" in f["countries"]
    assert f["current_system"] == "spreadsheets"


def test_discovery_asks_only_what_it_does_not_know_and_at_most_three():
    d = Discovery(CONFIG["qualification"])
    assert d.next_field(learn_facts(INTRO)) == "modules"
    d = Discovery(CONFIG["qualification"])
    asked = []
    for _ in range(5):
        f = d.next_field({})
        if f:
            d.asked.add(f)
            asked.append(f)
    assert len(asked) == 3


def test_the_first_reply_greets_presents_the_agenda_and_asks_a_discovery_question(chunks):
    s, llm = session(chunks, [{"reply": "Hi Alex, thanks for introducing yourself.", "kind": "clarify"}], discovery=True)
    t = s.ask(INTRO)
    assert t["reply"].startswith("Hi Alex, thanks for introducing yourself.")
    assert "Here's the plan for today" in t["reply"] and "Core HR" in t["reply"]
    assert t["reply"].endswith(s.discovery.question("modules"))
    assert "Do not ask a question" in llm.systems[-1]
    assert s.plan.offered is None
    assert s.buyer["company_size"] == "300" and s.buyer["current_system"] == "spreadsheets"


def test_a_first_hey_also_gets_the_agenda_and_a_discovery_question(chunks):
    s, _ = session(chunks, [{"reply": "Hey there!"}], discovery=True)
    t = s.ask("hey!")
    assert t["reply"].startswith("Hey there! ") and "Here's the plan for today" in t["reply"]
    assert t["reply"].endswith(s.discovery.question("company_size"))


def test_an_answer_is_recorded_and_the_next_question_follows(chunks):
    s, llm = session(chunks, [{"reply": "Hi Alex.", "kind": "clarify"},
                              {"reply": "Onboarding, got it.", "kind": "clarify"}], discovery=True)
    s.ask(INTRO)
    t = s.ask("Mostly onboarding, it's all manual")
    assert s.buyer["modules"] == "Mostly onboarding, it's all manual"
    assert t["reply"] == "Onboarding, got it. " + s.discovery.question("timeline")
    assert "acknowledge" in llm.systems[-1].lower() and "Do not ask a question" in llm.systems[-1]


def test_when_discovery_is_done_the_demo_starts(chunks):
    s, _ = session(chunks, [{"reply": "Hi Alex.", "kind": "clarify"}, {"reply": "Got it.", "kind": "clarify"},
                            {"reply": "Great, end of Q1.", "kind": "clarify"}], discovery=True)
    s.ask(INTRO)
    s.ask("Onboarding")
    t = s.ask("End of Q1")
    assert t["reply"].endswith(CORE_Q) and s.plan.offered == "core" and not s.discovery.active


def test_a_client_who_declines_goes_straight_to_the_demo(chunks):
    s, llm = session(chunks, [{"reply": "Hi Alex.", "kind": "clarify"}], discovery=True)
    s.ask(INTRO)
    calls = len(llm.systems)
    t = s.ask("I'd rather not, just show me the product")
    assert t["reply"] == f"No problem, let's go straight to the demo. {CORE_Q}"
    assert len(llm.systems) == calls and not s.discovery.active and s.plan.offered == "core"
    assert s.ask("yes") and s.plan.covered == {"core"}


def test_a_question_during_discovery_is_answered_and_discovery_continues(chunks):
    s, _ = session(chunks, [{"reply": "Hi Alex.", "kind": "clarify"},
                            {"reply": "Good question, happy to cover that.", "kind": "clarify"}], discovery=True)
    s.ask(INTRO)
    t = s.ask("Does Bob have a mobile app?")
    assert t["reply"].endswith(s.discovery.question("modules")) and "modules" not in s.buyer


def test_discovery_is_shown_in_the_panel(chunks):
    from deploykit.chat import turn_summary
    s, _ = session(chunks, [{"reply": "Hi Alex.", "kind": "clarify"}], discovery=True)
    t = s.ask(INTRO)
    assert turn_summary(t, s, chunks)["plan"]["phase"] == "discovery"


def test_asking_for_a_person_first_still_hands_off(chunks):
    s, _ = session(chunks, [], discovery=True)
    s.agent = s.agent.__class__(CONFIG, s.chunks, s.llm)
    t = s.ask("Get me a human now.")
    assert t["handoff"] and "Here's the plan" not in t["reply"]


def test_a_greeting_that_mentions_the_product_falls_back_to_a_greeting_with_their_name(chunks):
    s, _ = session(chunks, [{"reply": "Hi Alex, Bob is built for exactly this."}], discovery=True)
    assert s.ask(INTRO)["reply"].startswith("Hi Alex, great to meet you. Here's the plan for today")
