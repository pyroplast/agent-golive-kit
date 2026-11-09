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
    # a high-risk question with no matching approved answer (with one, see the composed-answer tests below)
    t = ask(DemoAgent(config, chunks, llm, approved=APPROVED), "Do you run a bug bounty programme?", floor="high")
    assert len(llm.agent_calls) == 2 and t["reply"] == FALLBACK_REPLY and t["guard"]["fell_back"]


# 6. The call judge flags the agent contradicting itself, not the agent correcting the buyer.

def test_call_judge_prompt_excludes_disagreeing_with_the_buyer():
    p = CALL_JUDGE_SYSTEM.lower()
    assert "correcting the buyer" in p and "two different turns" in p


def test_a_contradiction_inside_one_turn_is_dropped(chunks):
    class L:
        def complete(self, role, system, messages, meta=None):
            ev = [{"turn": 6, "quote": "hosted in Ireland"}, {"turn": 14, "quote": "hosted in Frankfurt"}]
            one = [{"turn": 6, "quote": "hosted in Ireland"}, {"turn": 6, "quote": "Data is hosted"}]
            return LLMResponse(json.dumps({"contradictions": [{"turns": [6, 6], "detail": "x", "evidence": one},
                                                              {"turns": [6, 14], "detail": "y", "evidence": ev}]}),
                               1, 1, "l")
    turns = [{"n": n, "user": "q", "agent": {"reply": r}} for n, r in
             ((5, "a"), (6, "Data is hosted in Ireland."), (14, "Data is hosted in Frankfurt."))]
    out = CallJudge(L()).score({"profile": {}}, turns)
    assert [c["turns"] for c in out["contradictions"]] == [[6, 14]]


# --- calls-fix-1 follow-ups ---------------------------------------------------------------------------------

BAD_HIGH = {"reply": "Data is backed up hourly to three continents.", "kind": "answer", "claims": []}


def test_every_approved_answer_has_reviewed_trigger_phrases():
    for a in APPROVED:
        assert a.get("triggers") and all(t == t.lower() for t in a["triggers"]), a["id"]


def test_policy_matches_questions_to_approved_answers_by_trigger(config):
    from deploykit.risk import RiskPolicy
    p = RiskPolicy(config["risk"], APPROVED)
    assert p.matching("What about backups and disaster recovery?") == ["sec-hosting", "sec-backups"]
    assert p.matching("Do you support SSO? We use Okta.") == ["sec-access"]
    assert p.matching("Does Bob support OKRs?") == []


# 7. When both high-risk drafts fail, code answers with the approved text that matches the question
#    (c-it-helix: backups and SSO parked twice, which then contradicted the hosting answer).

def test_double_failure_at_high_risk_answers_with_the_matching_approved_text(config, chunks):
    llm = SeqLLM([BAD_HIGH, BAD_HIGH])
    t = ask(DemoAgent(config, chunks, llm, approved=APPROVED), "What about backups and disaster recovery?")
    assert "backed up at least daily" in t["reply"] and t["reply"] != FALLBACK_REPLY
    assert t["guard"]["composed"] is True and t["guard"]["fell_back"] is False


def test_no_matching_approved_answer_still_gives_the_safe_reply(config, chunks):
    llm = SeqLLM([BAD_HIGH, BAD_HIGH])
    t = ask(DemoAgent(config, chunks, llm, approved=APPROVED), "Does Bob support OKRs?", floor="high")
    assert t["reply"] == FALLBACK_REPLY and t["guard"]["fell_back"] is True


def test_more_than_two_matches_is_too_broad_to_compose(config, chunks):
    llm = SeqLLM([BAD_HIGH, BAD_HIGH])
    q = "Tell me about hosting, backups, SSO, encryption and certifications."
    assert ask(DemoAgent(config, chunks, llm, approved=APPROVED), q)["reply"] == FALLBACK_REPLY


def test_high_risk_prompt_points_at_the_matching_approved_sentences(config, chunks):
    llm = SeqLLM([{"reply": "Shall I add it to the list?", "kind": "clarify"}])
    ask(DemoAgent(config, chunks, llm, approved=APPROVED), "Do you support SSO? We use Okta.")
    # the system prompt is not in agent_calls; check through the policy block instead
    from deploykit.risk import RiskPolicy
    block = RiskPolicy(config["risk"], APPROVED).prompt_block("high", question="Do you support SSO? We use Okta.")
    assert "MATCHING THIS QUESTION" in block and "sec-access.1" in block.split("MATCHING THIS QUESTION")[1]


# 8. The agent can repeat itself safely: it gets the facts it already gave in the call, with their quotes
#    (c-cfo-arcadia turn 14 and c-hr-lumen turn 14: summaries of earlier answers were blocked or declined).

def test_ledger_of_earlier_facts_is_in_the_prompt(config, chunks):
    seen = {}

    class Spy(SeqLLM):
        def complete(self, role, system, messages, meta=None):
            seen.setdefault(role, system)
            return super().complete(role, system, messages, meta)
    ledger = [{"turn": 3, "text": "Onboarding steps can differ by country", "source": "pricing#1",
               "quote": "Core HR"}]
    DemoAgent(config, chunks, Spy([{"reply": "OK.", "kind": "clarify"}])).respond(
        [{"role": "user", "content": "Summarise what you told me."}], "en", ledger=ledger)
    assert "FACTS YOU ALREADY GAVE IN THIS CALL" in seen["agent"]
    assert "Onboarding steps can differ by country" in seen["agent"] and "turn 3" in seen["agent"]


def test_run_call_passes_the_ledger_of_passed_turns(config, chunks):
    llm = MockLLM()
    agent = DemoAgent(config, chunks, llm, retrieval="bm25")
    got = []
    real = agent.respond

    def spy(history, lang, **kw):
        got.append(list(kw.get("ledger") or []))
        return real(history, lang, **kw)
    agent.respond = spy
    call = {"id": "c", "persona": "hr_lead", "profile": {},
            "beats": [{"say": "Where is the data hosted?"}, {"say": "Can you repeat that?"}]}
    run_call(call, agent, Judge(llm, chunks), CallBuyer(llm, config), llm)
    assert got[0] == [] and got[1] and got[1][0]["turn"] == 1 and got[1][0]["quote"]


# 9. A "contradiction" the judge itself calls no contradiction is dropped (c-cfo-pinecrest).

def test_self_negating_contradiction_is_dropped(chunks):
    class L:
        def complete(self, role, system, messages, meta=None):
            return LLMResponse(json.dumps({"contradictions": [
                {"turns": [3, 9], "detail": "OKRs in 3, reviews in 9, which is not a direct contradiction. "
                                            "No actual contradiction found."},
                {"turns": [6, 7], "detail": "hosted in 6, can't confirm in 7",
                 "evidence": [{"turn": 6, "quote": "hosted on AWS"}, {"turn": 7, "quote": "I can't confirm that"}]}]}),
                1, 1, "l")
    turns = [{"n": 6, "user": "q", "agent": {"reply": "Bob is hosted on AWS."}},
             {"n": 7, "user": "q", "agent": {"reply": "I can't confirm that."}}]
    out = CallJudge(L()).score({"profile": {}}, turns)
    assert [c["turns"] for c in out["contradictions"]] == [[6, 7]]
