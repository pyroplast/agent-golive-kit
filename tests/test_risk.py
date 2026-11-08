"""Risk tiers: classification, tier-specific guard rules, retry policy, severity and the release gate.

Design rules under test:
- The tier is the HIGHEST of three signals (question, cited sections, claims): misclassification fails strict.
- A conversation's tier only goes up.
- low: paraphrase of cited sections; medium: numbers and names must be in the quote;
  high: only pre-approved answers, no retry (park straight away).
"""
import json

import pytest

from deploykit.agent import DemoAgent
from deploykit.guard import check_reply
from deploykit.llm import LLMResponse
from deploykit.risk import RiskPolicy, TIERS, max_tier

RISK = {
    "keywords": {"high": ["gdpr", "hipaa", "certif", "legal", "tax", "residency", "tribunal", "soc 2"],
                 "medium": ["price", "pricing", "cost", "discount", "contract", "sla", "integrat", "timeline"]},
    "section_tiers": {"security": "high", "pricing": "medium"},
    "default": "low",
}
APPROVED = [{"id": "sec-certs", "text": "HiBob is ISO 27001 certified with SOC 2 Type II audits.",
             "sources": ["security#1"], "status": "draft"}]


@pytest.fixture
def policy():
    return RiskPolicy(RISK, APPROVED)


def claim(text, source, quote):
    return {"text": text, "source": source, "quote": quote}


def types(r):
    return {p["type"] for p in r["problems"]}


# --- classification ----------------------------------------------------------

def test_tiers_are_ordered():
    assert TIERS == ("low", "medium", "high")
    assert max_tier("low", "high", "medium") == "high"


@pytest.mark.parametrize("q,tier", [
    ("Does it have a mobile app?", "low"),
    ("What does it cost for 150 people?", "medium"),
    ("Are you GDPR compliant?", "high"),
    ("Does Bob replace our human resources system?", "low"),
])
def test_question_signal(policy, q, tier):
    assert policy.question_tier(q) == tier


def test_cited_section_raises_the_tier(policy):
    assert policy.classify("Tell me more", citations=["security#2"], claims=[]) == "high"
    assert policy.classify("Tell me more", citations=["pricing#1"], claims=[]) == "medium"


def test_a_claim_with_a_number_is_at_least_medium(policy):
    c = [claim("Mid-market companies go live in 4 weeks", "x#1", "q")]
    assert policy.classify("How long?", citations=[], claims=c) == "medium"


def test_highest_signal_wins(policy):
    assert policy.classify("What does it cost?", citations=["security#1"], claims=[]) == "high"


# --- tier-specific guard rules -------------------------------------------------

def test_medium_needs_numbers_in_the_quote_low_accepts_the_section(chunks):
    d = {"reply": "SOC 2 audits and ISO 27001.", "handoff": False, "unknown": False,
         "claims": [claim("SOC 2 audits and ISO 27001", "security#1", "SOC 2 Type II audits")]}
    assert check_reply(d, chunks, "", tier="low")["ok"]
    assert "number_not_in_source" in types(check_reply(d, chunks, "", tier="medium"))


def test_low_accepts_watch_words_from_the_section_medium_needs_the_quote(chunks):
    d = {"reply": "HiBob is certified.", "handoff": False, "unknown": False,
         "claims": [claim("HiBob is certified", "security#1", "SOC 2 Type II audits")]}
    assert check_reply(d, chunks, "", tier="low")["ok"]
    assert "watch_word" in types(check_reply(d, chunks, "", tier="medium"))


def test_high_requires_an_approved_answer(chunks):
    d = {"reply": "HiBob holds ISO 27001.", "handoff": False, "unknown": False,
         "claims": [claim("HiBob holds ISO 27001", "security#1", "ISO 27001 certified")]}
    r = check_reply(d, chunks, "", tier="high", approved=APPROVED)
    assert "high_risk_unapproved" in types(r)


def test_high_accepts_the_approved_text_verbatim(chunks):
    d = {"reply": "Good question. HiBob is ISO 27001 certified with SOC 2 Type II audits. Anything else?",
         "approved_answer": "sec-certs", "handoff": False, "unknown": False, "claims": []}
    assert check_reply(d, chunks, "", tier="high", approved=APPROVED)["ok"]


def test_high_rejects_an_edited_approved_answer(chunks):
    d = {"reply": "HiBob is ISO 27001 and HIPAA certified with SOC 2 Type II audits.",
         "approved_answer": "sec-certs", "handoff": False, "unknown": False, "claims": []}
    assert "approved_answer_altered" in types(check_reply(d, chunks, "", tier="high", approved=APPROVED))


def test_high_allows_park_refusal_and_handoff_without_approved_text(chunks):
    for d in ({"reply": "I can't advise on that. Shall I add it to the list for a HiBob team member?",
               "park": "Tribunal?", "handoff": False, "unknown": False, "claims": []},
              {"reply": "Of course. I'm handing you over now.", "handoff": True, "unknown": False, "claims": []}):
        assert check_reply(d, chunks, "", tier="high", approved=APPROVED)["ok"]


# --- agent: tier flow and retry policy -----------------------------------------

class SeqLLM:
    def __init__(self, *texts):
        self.texts, self.calls = list(texts), []

    def complete(self, role, system, messages, meta=None):
        self.calls.append({"system": system, "messages": messages})
        return LLMResponse(self.texts[min(len(self.calls), len(self.texts)) - 1], 10, 5, "seq")


def out(**kw):
    base = {"reply": "ok", "kind": "answer", "claims": [], "handoff": False, "handoff_reason": None,
            "unknown": False, "park": None, "approved_answer": None, "qualification": {}}
    base.update(kw)
    return json.dumps(base)


def agent(config, chunks, llm):
    cfg = dict(config, risk=RISK)
    return DemoAgent(cfg, chunks, llm, approved=APPROVED)


def test_high_tier_failure_parks_without_retrying(config, chunks):
    llm = SeqLLM(out(reply="HiBob holds HIPAA certifications.",
                     claims=[claim("HiBob holds HIPAA certifications", "security#1", "SOC 2 Type II audits")]))
    turn = agent(config, chunks, llm).respond([{"role": "user", "content": "Are you HIPAA certified?"}], lang="en")
    assert len(llm.calls) == 1
    assert turn["tier"] == "high" and turn["guard"]["fell_back"] is True and turn["guard"]["retried"] is False
    assert turn["park"] == "Are you HIPAA certified?"


def test_prompt_carries_the_tier_and_approved_answers_when_high(config, chunks):
    llm = SeqLLM(out(reply="Shall I add that to the list?", park="GDPR?"))
    agent(config, chunks, llm).respond([{"role": "user", "content": "Are you GDPR compliant?"}], lang="en")
    assert "RISK TIER: high" in llm.calls[0]["system"]
    assert "sec-certs" in llm.calls[0]["system"]


def test_conversation_tier_only_goes_up(config, chunks):
    llm = SeqLLM(out(reply="Hosted on AWS.", claims=[claim("Hosted on AWS", "pricing#2", "Add-ons include Hiring")]))
    a = agent(config, chunks, llm)
    t = a.respond([{"role": "user", "content": "Is there a mobile app?"}], lang="en", floor="high")
    assert t["tier"] == "high"


# --- severity and release gate -----------------------------------------------

def test_summary_weights_failures_by_tier_and_gates_on_high():
    from deploykit.report import summarize

    def row(sid, tier, passed):
        return {"scenario_id": sid, "category": "expected", "persona": "cfo", "lang": "en", "tier": tier,
                "verdict": {"passed": passed, "reasons": [] if passed else ["ungrounded"]},
                "turns": [{"user": "q", "agent": {"reply": "r", "citations": [], "handoff": False, "unknown": False},
                           "judge": {"grounded": passed}}]}

    s = summarize([row("a", "low", False), row("b", "medium", False), row("c", "high", True), row("d", "high", False)])
    assert s["by_tier"]["high"] == {"sessions": 2, "passed": 1, "failed": 1}
    assert s["severity"] == 1 + 3 + 10
    assert s["gate"] == {"passed": False, "high_tier_failures": 1}
    s2 = summarize([row("a", "low", False), row("c", "high", True)])
    assert s2["gate"]["passed"] is True


def test_tiered_run_records_tiers_and_reports_the_gate(tmp_path, config, chunks, scenarios):
    from deploykit.llm import MockLLM
    from deploykit.report import build_report, load_rows
    from deploykit.runner import run_suite
    run = run_suite(dict(config, risk=RISK), chunks, scenarios, MockLLM(), langs=["en"], out_dir=tmp_path,
                    name="t", approved=APPROVED)
    rows = {r["scenario_id"]: r for r in load_rows(run)}
    assert rows["exp-hosting"]["tier"] == "high"        # cites the security section
    assert rows["adv-discount"]["tier"] == "medium"     # pricing question
    assert all(r["tier"] in ("low", "medium", "high") for r in rows.values())
    html = build_report(run).read_text()
    assert "Release gate" in html and "Results by risk tier" in html
