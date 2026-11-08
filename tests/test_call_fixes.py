"""Fixes from the first live call run (calls-baseline): problems that only show up in a 15-turn call."""
import json
from pathlib import Path

from deploykit.agent import DemoAgent, FALLBACK_REPLY, build_system_prompt
from deploykit.calls import CallBuyer, run_call
from deploykit.judge import CALL_JUDGE_SYSTEM, CallJudge, Judge
from deploykit.llm import LLMResponse, MockLLM

ROOT = Path(__file__).resolve().parent.parent
APPROVED = json.loads((ROOT / "config" / "approved_answers.json").read_text())


class SeqLLM:
    """Agent answers from a script, in order; judge and buyer from the mock."""

    def __init__(self, agent_outputs):
        self.outputs, self.mock, self.agent_calls = list(agent_outputs), MockLLM(), []

    def complete(self, role, system, messages, meta=None):
        if role == "agent":
            self.agent_calls.append(messages)
            out = self.outputs.pop(0) if self.outputs else {"reply": "OK.", "kind": "clarify"}
            return LLMResponse(json.dumps(out), 10, 10, "seq")
        return self.mock.complete(role, system, messages, meta)


def ask(agent, text, **kw):
    return agent.respond([{"role": "user", "content": text}], "en", **kw)


# 1. The reply offers the question list but the model left "park" empty (8 turns in calls-baseline).

def test_an_offer_in_the_reply_puts_the_question_on_the_list(config, chunks):
    llm = SeqLLM([{"reply": "Hosting outside the EU is not described in our public material. Would you like me to add "
                            "this question to the list to discuss with a HiBob team member?", "kind": "clarify"}])
    t = ask(DemoAgent(config, chunks, llm), "Can you host our data in the US only?")
    assert t["park"] == "Can you host our data in the US only?"


def test_a_reply_without_an_offer_does_not_park(config, chunks):
    llm = SeqLLM([{"reply": "Which countries do you employ people in?", "kind": "clarify"}])
    assert ask(DemoAgent(config, chunks, llm), "Is Bob a fit for us?")["park"] is None


# 2. Inline [chunk#n] markers in the reply count as citations (calls-baseline turn 11 of c-cfo-arcadia).

def test_inline_citation_markers_count_as_citations(config, chunks):
    llm = SeqLLM([{"reply": "Pricing is quote-based [pricing#1].", "kind": "answer",
                   "claims": [{"text": "Pricing is quote-based", "source": "pricing#1", "quote": "quote"}]}])
    t = ask(DemoAgent(config, chunks, llm, guard=False), "How is pricing calculated?")
    assert "pricing#1" in t["citations"]


def test_inline_marker_without_a_claim_is_still_recorded(config, chunks):
    llm = SeqLLM([{"reply": "I can't compare vendors, as pricing is quote-based [pricing#1]. Shall I add it to the "
                            "list for a HiBob team member?", "kind": "clarify"}])
    t = ask(DemoAgent(config, chunks, llm, guard=False), "Are you cheaper than BambooHR?")
    assert t["citations"] == ["pricing#1"]


# 3. Only an explicit request for a person ends in a handoff (c-cfo-arcadia ended early on "What happens next?").

def test_a_handoff_nobody_asked_for_is_cancelled(config, chunks):
    llm = SeqLLM([{"reply": "You'll be connected with a HiBob team member about your open questions.",
                   "kind": "clarify", "handoff": True, "handoff_reason": "wrap-up"}])
    t = ask(DemoAgent(config, chunks, llm), "Thanks, I've heard enough. What happens next?")
    assert t["handoff"] is False


def test_an_explicit_request_still_hands_over(config, chunks):
    llm = SeqLLM([{"reply": "Sure.", "kind": "clarify"}])
    assert ask(DemoAgent(config, chunks, llm), "Put me through to a sales rep please.")["handoff"] is True


def test_the_prompt_says_how_to_close_a_call(config, chunks):
    p = build_system_prompt(config, chunks).lower()
    assert "wraps up" in p and "recap" in p


# 4. The risk tier carries one turn, then each question stands on its own (c-it-helix: one security question at
#    turn 1 held SSO, Workday and a recap to approved-sentences-only for the rest of the call).

def test_tier_carries_to_the_next_turn_only(config, chunks):
    llm = MockLLM()
    agent = DemoAgent(config, chunks, llm, retrieval="bm25", approved=APPROVED)
    call = {"id": "c", "persona": "it_admin", "profile": {},
            "beats": [{"say": "Which security certifications do you have?"},
                      {"say": "Great, can you send me that report now?"},
                      {"say": "What add-on modules are there?"},
                      {"say": "What add-on modules are there?"}]}
    row = run_call(call, agent, Judge(llm, chunks), CallBuyer(llm, config), llm)
    tiers = [t["agent"]["tier"] for t in row["turns"]]
    assert tiers[0] == "high" and tiers[1] == "high"  # a direct follow-up keeps the stakes of the question before
    assert tiers[3] != "high"  # an unrelated question two turns later is judged on its own


def test_each_turn_records_its_own_tier(config, chunks):
    agent = DemoAgent(config, chunks, MockLLM(), retrieval="bm25", approved=APPROVED)
    t = agent.respond([{"role": "user", "content": "What add-on modules are there?"}], "en", floor="high")
    assert t["tier"] == "high" and t["own_tier"] != "high"


# 5. A high-risk draft that fails the guard gets one retry before the safe reply (c-it-helix turn 14: repeating
#    an earlier hosting answer fell straight to "I can't confirm that").

def test_high_risk_draft_gets_one_retry(config, chunks):
    bad = {"reply": "Data is hosted on AWS in Ireland with annual audits.", "kind": "answer", "claims": []}
    good = {"reply": "", "kind": "answer", "approved_sentences": ["sec-hosting.1"]}
    llm = SeqLLM([bad, good])
    t = ask(DemoAgent(config, chunks, llm, approved=APPROVED), "Where is our data hosted?")
    assert len(llm.agent_calls) == 2
    assert t["reply"] != FALLBACK_REPLY and t["guard"]["retried"] and not t["guard"]["fell_back"]


def test_high_risk_retry_that_fails_again_falls_back(config, chunks):
    bad = {"reply": "Data is hosted on AWS in Ireland with annual audits.", "kind": "answer", "claims": []}
    llm = SeqLLM([bad, bad])
    t = ask(DemoAgent(config, chunks, llm, approved=APPROVED), "Where is our data hosted?")
    assert t["reply"] == FALLBACK_REPLY and t["guard"]["fell_back"]


# 6. The call judge flags the agent contradicting itself, not the agent correcting the buyer.

def test_call_judge_prompt_excludes_disagreeing_with_the_buyer():
    p = CALL_JUDGE_SYSTEM.lower()
    assert "correcting the buyer" in p and "two different turns" in p


def test_a_contradiction_inside_one_turn_is_dropped(chunks):
    class L:
        def complete(self, role, system, messages, meta=None):
            return LLMResponse(json.dumps({"contradictions": [{"turns": [5, 5], "detail": "x"},
                                                              {"turns": [6, 14], "detail": "y"}]}), 1, 1, "l")
    turn = {"n": 1, "user": "q", "agent": {"reply": "a"}}
    out = CallJudge(L()).score({"profile": {}}, [turn])
    assert out["contradictions"] == [{"turns": [6, 14], "detail": "y"}]
