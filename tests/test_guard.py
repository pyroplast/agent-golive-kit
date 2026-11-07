"""Pre-send guard: deterministic checks that run before a reply reaches the buyer.

Every case here comes from a real failure seen in the live runs.
"""
from deploykit.guard import check_reply, extract_numbers


def claim(text, source, quote):
    return {"text": text, "source": source, "quote": quote}


def draft(reply, claims=(), handoff=False, unknown=False):
    return {"reply": reply, "claims": list(claims), "handoff": handoff, "unknown": unknown}


def problem_types(result):
    return {p["type"] for p in result["problems"]}


# --- numbers -------------------------------------------------------------

def test_extract_numbers_normalises_formats():
    assert extract_numbers("$8–12 per employee, 99.9% uptime, 1,000+ staff, ISO 27001:2022") == \
        {"8", "12", "99.9", "1000", "27001", "2022"}


def test_invented_price_is_blocked(chunks):
    """Baseline: asked for a ballpark, the agent said '$8–12 per employee per month'."""
    d = draft("Pricing typically starts at $8–12 per employee per month.",
              [claim("Pricing starts at $8–12 per employee", "pricing#1",
                     "Pricing is quote-based and calculated per employee")])
    r = check_reply(d, chunks, user_text="Just a ballpark.")
    assert not r["ok"]
    assert "number_not_in_source" in problem_types(r)


def test_numbers_from_the_buyer_are_allowed(chunks):
    d = draft("For your 300 people, Bob covers core HR.",
              [claim("Bob covers core HR", "pricing#1", "Pricing is quote-based")])
    assert check_reply(d, chunks, user_text="We are 300 people.")["ok"]


def test_numbers_inside_the_quote_are_allowed(chunks):
    d = draft("HiBob is ISO 27001 certified.",
              [claim("HiBob is ISO 27001 certified", "security#1", "ISO 27001 certified")])
    assert check_reply(d, chunks, user_text="")["ok"]


# --- watch-list words ----------------------------------------------------

def test_certification_claim_needs_the_word_in_its_quote(chunks):
    """Judge check: 'HIPAA certifications' when the source says HIPAA compliance."""
    d = draft("We hold SOC 2 and HIPAA certifications.",
              [claim("HiBob holds HIPAA certifications", "security#1", "SOC 2 Type II audits")])
    r = check_reply(d, chunks, user_text="")
    assert "watch_word" in problem_types(r)
    assert any("certif" in p["detail"] for p in r["problems"])


def test_watch_word_used_by_buyer_is_allowed(chunks):
    d = draft("The sources don't mention a free trial.", unknown=True)
    assert check_reply(d, chunks, user_text="Is there a free trial?")["ok"]


def test_annual_is_a_watch_word(chunks):
    """Judge check: 'SOC 2 Type II audits conducted annually' (source: 'routine')."""
    d = draft("SOC 2 Type II audits are conducted annually.",
              [claim("SOC 2 Type II audits are annual", "security#1", "SOC 2 Type II audits")])
    assert "watch_word" in problem_types(check_reply(d, chunks, user_text=""))


# --- quotes --------------------------------------------------------------

def test_quote_must_appear_in_its_source(chunks):
    d = draft("Hosted in the US.",
              [claim("Hosted in the US", "security#2", "Hosted on AWS in the United States")])
    assert "quote_not_found" in problem_types(check_reply(d, chunks, user_text=""))


def test_quote_matching_ignores_case_spacing_and_punctuation(chunks):
    d = draft("Hosted on AWS, primary region Ireland.",
              [claim("Hosted on AWS with primary region Ireland", "security#2",
                     "hosted on AWS  with the primary region in Ireland")])
    assert check_reply(d, chunks, user_text="")["ok"]


def test_unknown_source_is_blocked(chunks):
    d = draft("Bob does X.", [claim("Bob does X", "made-up#3", "Bob does X")])
    assert "unknown_source" in problem_types(check_reply(d, chunks, user_text=""))


def test_reply_text_outside_claims_is_still_checked(chunks):
    """A number slipped into the reply but not into any claim must still be caught."""
    d = draft("Hosted on AWS. We guarantee 99.99% uptime.",
              [claim("Hosted on AWS", "security#2", "Hosted on AWS")])
    t = problem_types(check_reply(d, chunks, user_text=""))
    assert {"number_not_in_source", "watch_word"} <= t


def test_clean_handoff_passes(chunks):
    d = draft("I'll connect you with an Account Executive for that.", handoff=True)
    assert check_reply(d, chunks, user_text="Give me a 40% discount")["ok"]
