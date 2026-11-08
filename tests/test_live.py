"""Live mode: reply sentence by sentence, check each sentence as it streams, fast model for easy turns,
a short filler line when the slow path runs, and fewer retries from false name alarms."""
import json
from pathlib import Path

import pytest

from deploykit.guard import check_reply
from deploykit.live import FILLER, LIVE_SPEC_MARK, SAFE_TAIL, LiveAgent
from deploykit.llm import LLMResponse, MockLLM

ROOT = Path(__file__).resolve().parent.parent
APPROVED = json.loads((ROOT / "config" / "approved_answers.json").read_text())

GOOD = {"say": "Bob's pricing is quote-based.", "claims": [
    {"text": "Pricing is quote-based", "source": "pricing#1", "quote": "Pricing is quote-based"}]}


class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


class StreamLLM:
    """Streams scripted lines in small chunks; each chunk advances the fake clock. complete() is the slow path."""

    def __init__(self, lines, clock, step=0.1, slow=None, first_delay=0.0):
        self.lines, self.clock, self.step, self.first_delay = lines, clock, step, first_delay
        self.slow = slow or {"reply": "Slow-path answer.", "kind": "clarify"}
        self.roles, self.completes = [], 0

    def stream(self, role, system, messages, meta=None):
        self.roles.append(role)
        assert LIVE_SPEC_MARK in system
        text = "".join(json.dumps(l) + "\n" for l in self.lines)
        self.clock.t += self.first_delay
        for i in range(0, len(text), 25):
            self.clock.t += self.step
            yield text[i:i + 25]

    def complete(self, role, system, messages, meta=None):
        self.completes += 1
        self.clock.t += 2.0
        return LLMResponse(json.dumps(self.slow), 10, 10, "slow")


def live(config, chunks, llm, clock, **kw):
    return LiveAgent(config, chunks, llm, clock=clock, **kw)


def ask(agent, text, **kw):
    return agent.respond([{"role": "user", "content": text}], "en", **kw)


@pytest.fixture
def pricing_chunks(chunks):
    # make sure the fixture text holds the quote used above
    assert any("quote-based" in c.text.lower() for c in chunks if c.id == "pricing#1"), \
        [c.text for c in chunks if c.id == "pricing#1"]
    return chunks


def test_sentences_stream_and_are_checked_one_by_one(config, pricing_chunks):
    clock = Clock()
    lines = [GOOD, {"say": "Which countries do you employ people in?", "claims": []},
             {"end": True, "kind": "answer"}]
    t = ask(live(config, pricing_chunks, StreamLLM(lines, clock), clock), "How is pricing calculated?")
    assert t["reply"] == "Bob's pricing is quote-based. Which countries do you employ people in?"
    assert t["citations"] == ["pricing#1"]
    tm = t["timing"]
    assert tm["streamed"] and 0 < tm["first_sentence_s"] < tm["total_s"]
    assert tm["first_words_s"] == tm["first_sentence_s"] and not tm["filler"]


def test_a_failing_later_sentence_is_skipped_and_the_question_parked(config, pricing_chunks):
    # live-1: stopping at the first bad sentence sent most turns down the slow path. Now a bad sentence is
    # dropped, the rest still streams, and the question goes on the list.
    clock = Clock()
    bad = {"say": "It costs $9 per employee.", "claims": []}
    lines = [GOOD, bad, {"say": "Which countries are you in?", "claims": []}, {"end": True}]
    llm = StreamLLM(lines, clock)
    t = ask(live(config, pricing_chunks, llm, clock), "How is pricing calculated?")
    assert t["reply"].startswith("Bob's pricing is quote-based.") and t["reply"].endswith(SAFE_TAIL)
    assert "$9" not in t["reply"] and "Which countries are you in?" in t["reply"]
    assert t["park"] == "How is pricing calculated?" and t["guard"]["skipped"] == 1
    assert llm.completes == 0  # what was already said stays said: no rewrite


def test_a_failing_first_sentence_is_skipped_if_a_later_one_passes(config, pricing_chunks):
    clock = Clock()
    lines = [{"say": "It costs $9 per employee.", "claims": []}, GOOD, {"end": True}]
    llm = StreamLLM(lines, clock)
    t = ask(live(config, pricing_chunks, llm, clock), "How is pricing calculated?")
    assert llm.completes == 0 and "$9" not in t["reply"] and "quote-based" in t["reply"]


def test_the_agent_does_not_see_its_own_filler_lines(config, pricing_chunks):
    clock = Clock()
    seen = []

    class Spy(StreamLLM):
        def stream(self, role, system, messages, meta=None):
            seen.append(messages)
            return super().stream(role, system, messages, meta)
    history = [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": FILLER + " Welcome!"},
               {"role": "user", "content": "How is pricing calculated?"}]
    live(config, pricing_chunks, Spy([GOOD, {"end": True}], clock), clock).respond(history, "en")
    assert FILLER not in json.dumps(seen[0])


def test_a_failing_first_sentence_takes_the_slow_path_with_a_filler(config, pricing_chunks):
    clock = Clock()
    llm = StreamLLM([{"say": "It costs $9 per employee.", "claims": []}, {"end": True}], clock)
    t = ask(live(config, pricing_chunks, llm, clock), "How is pricing calculated?")  # nothing passes
    assert llm.completes >= 1 and t["guard"]["slow_path"] is True
    assert t["reply"].startswith(FILLER) and "$9" not in t["reply"]
    assert t["timing"]["filler"] is True and t["timing"]["first_words_s"] <= 1.5


def test_a_slow_first_sentence_gets_the_filler_at_the_threshold(config, pricing_chunks):
    clock = Clock()
    llm = StreamLLM([GOOD, {"end": True}], clock, first_delay=3.0)
    t = ask(live(config, pricing_chunks, llm, clock), "How is pricing calculated?")
    assert t["timing"]["filler"] is True and t["timing"]["first_words_s"] == 1.5
    assert t["timing"]["first_sentence_s"] > 3.0 and t["reply"].startswith(FILLER)


def test_easy_turns_use_the_fast_model_and_risky_ones_the_big_model(config, pricing_chunks):
    clock = Clock()
    llm = StreamLLM([{"say": "Happy to help.", "claims": []}, {"end": True}], clock)
    agent = live(config, pricing_chunks, llm, clock, approved=APPROVED, fast=True)
    agent.respond([{"role": "user", "content": "What do you do?"}], "en")
    agent.respond([{"role": "user", "content": "Where is our data hosted?"}], "en")
    assert llm.roles == ["agent_fast", "agent"]


def test_without_fast_every_turn_uses_the_big_model(config, pricing_chunks):
    clock = Clock()
    llm = StreamLLM([{"say": "Happy to help.", "claims": []}, {"end": True}], clock)
    live(config, pricing_chunks, llm, clock, fast=False).respond([{"role": "user", "content": "Hi"}], "en")
    assert llm.roles == ["agent"]


def test_approved_sentence_lines_are_composed_by_code(config, chunks):
    clock = Clock()
    llm = StreamLLM([{"approved": "sec-hosting.1"}, {"end": True, "kind": "answer"}], clock)
    t = ask(live(config, chunks, llm, clock, approved=APPROVED), "Where is our data hosted?")
    assert t["reply"].startswith("Bob is hosted on AWS.") and t["tier"] == "high"


def test_an_unreadable_stream_takes_the_slow_path(config, chunks):
    clock = Clock()

    class Junk(StreamLLM):
        def stream(self, role, system, messages, meta=None):
            self.roles.append(role)
            yield "Sure! Bob is great at everything."
    llm = Junk([], clock)
    t = ask(live(config, chunks, llm, clock), "Is Bob good?")
    assert llm.completes >= 1 and t["guard"]["slow_path"] is True


def test_mock_llm_streams_in_the_live_format(config, chunks):
    agent = LiveAgent(config, chunks, MockLLM(), retrieval="bm25")
    t = agent.respond([{"role": "user", "content": "Where is the data hosted?"}], "en")
    assert t["timing"]["streamed"] and t["reply"]


# --- fewer retries: the buyer's own names are not invented names ----------------------------------------------

def _draft(reply):
    return {"reply": reply, "claims": [{"text": reply, "source": "security#2", "quote": "Hosted on AWS"}],
            "citations": ["security#2"], "kind": "answer"}


def test_the_buyers_company_and_name_are_allowed(chunks):
    r = check_reply(_draft("For Lumen, Bob is hosted on AWS."), chunks, context_names={"Lumen"})
    assert not [p for p in r["problems"] if p["type"] == "unsupported_name"]


def test_country_and_language_names_are_allowed(chunks):
    r = check_reply(_draft("For your teams in Spain and Germany, Bob is hosted on AWS."), chunks)
    assert not [p for p in r["problems"] if p["type"] == "unsupported_name"]


def test_an_invented_integration_is_still_flagged(chunks):
    r = check_reply(_draft("Bob is hosted on AWS and syncs with Workday."), chunks, context_names={"Lumen"})
    assert any("Workday" in p["detail"] for p in r["problems"] if p["type"] == "unsupported_name")


def test_run_call_tells_the_agent_who_is_on_the_call(config, chunks):
    from deploykit.calls import CallBuyer, run_call
    from deploykit.judge import Judge
    from deploykit.agent import DemoAgent
    llm = MockLLM()
    agent = DemoAgent(config, chunks, llm, retrieval="bm25")
    call = {"id": "c", "persona": "hr_lead", "profile": {"company": "Lumen Studios", "role": "Head of People"},
            "beats": [{"say": "Hi, I’m Alex from Lumen Studios. Where is the data hosted?"}]}  # curly apostrophe
    run_call(call, agent, Judge(llm, chunks), CallBuyer(llm, config), llm)
    assert {"Lumen", "Studios", "Alex"} <= agent.context_names


# --- whole suite in live mode ---------------------------------------------------------------------------------

def test_run_calls_live_reports_first_words_and_fast_model_share(tmp_path, config, chunks):
    from deploykit.calls import load_call_rows, run_calls, summarize_calls
    calls = [{"id": "c1", "persona": "hr_lead", "profile": {"company": "Test Co"},
              "beats": [{"say": "Where is the data hosted?"}, {"say": "What add-on modules are there?"}]}]
    run_dir = run_calls(config, chunks, calls, MockLLM(), out_dir=tmp_path, name="l", live=True)
    s = summarize_calls(load_call_rows(run_dir))
    assert json.loads((run_dir / "meta.json").read_text())["live"] is True
    assert {"first_words_p50", "first_words_p90", "filler_turns", "fast_model_turns"} <= set(s["latency"])


def test_nebius_has_a_fast_agent_model():
    m = json.loads((ROOT / "config" / "models.json").read_text())
    assert m["nebius"]["agent_fast"] == "Qwen/Qwen3-30B-A3B-Instruct-2507"
    assert m["prices_usd_per_million_tokens"]["Qwen/Qwen3-30B-A3B-Instruct-2507"] == {"input": 0.1, "output": 0.3}


def test_openai_compat_streams_and_records_usage_and_first_token():
    from types import SimpleNamespace as NS
    from deploykit.llm import CostTracker, OpenAICompatLLM, TrackedLLM

    class Client:
        class chat:
            class completions:
                @staticmethod
                def create(**kw):
                    assert kw["stream"] is True
                    def gen():
                        yield NS(choices=[NS(delta=NS(content="Hel"))], usage=None)
                        yield NS(choices=[NS(delta=NS(content="lo"))], usage=None)
                        yield NS(choices=[], usage=NS(prompt_tokens=100, completion_tokens=2,
                                                      prompt_tokens_details=NS(cached_tokens=0)))
                    return gen()
    tracker = CostTracker({"m": {"input": 1.0, "output": 1.0}})
    llm = TrackedLLM(OpenAICompatLLM({"agent": "m"}, client=Client()), tracker)
    assert "".join(llm.stream("agent", "s", [])) == "Hello"
    s = tracker.summary()
    assert s["by_role"]["agent"]["input_tokens"] == 100 and s["by_role"]["agent"]["calls"] == 1


def test_live_format_asks_for_evidence_before_each_sentence():
    # calls-live-2: 29 sentences stated facts with no claim on their line; quote-first ordering and an example fix it
    from deploykit.live import LIVE_SPEC
    first_line = next(l for l in LIVE_SPEC.splitlines() if l.startswith('{"claims"'))
    assert first_line.index('"claims"') < first_line.index('"say"')
    assert "evidence FIRST" in LIVE_SPEC and "integrations#2" in LIVE_SPEC


def test_claims_first_lines_parse(config, pricing_chunks):
    clock = Clock()
    line = {"claims": [{"source": "pricing#1", "quote": "Pricing is quote-based"}], "say": "Pricing is quote-based."}
    t = ask(live(config, pricing_chunks, StreamLLM([line, {"end": True}], clock), clock), "How is pricing set?")
    assert t["reply"] == "Pricing is quote-based." and t["citations"] == ["pricing#1"]
