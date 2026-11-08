"""Pre-send guard. Deterministic, zero model tokens.

Runs on the agent's draft before the buyer sees it. Three checks, each from a real failure:

1. Numbers: every number in the reply must appear in a cited section or in the buyer's words.
   (Baseline: an invented "$8–12 per employee per month".)
2. Watch-list words: high-risk words such as "certified" or "annual" must appear in the quote that
   backs the claim, or in the buyer's words. (Judge check: "HIPAA certifications", "audits annually".)
3. Quotes: each claim carries a verbatim quote, and the quote must exist in the cited source.
4. Names: capitalised names in factual sentences must appear in the cited sections; the buyer's words do
   not count. (reg-1: "Yes, Bob integrates with Workday ... via Workato".)
   Turns "is this answer true?" into string matching.

Checks run per claim (claim text vs its own quote) and over the whole reply (reply vs all quotes),
so text outside the claims cannot slip through.
"""
from __future__ import annotations

import re

from deploykit.kb import Chunk

# Sentences that state no product fact, so need no claim: questions are dropped separately; these cover
# statements about the sources, refusals, list offers, handover lines and short pleasantries.
# The reply's own label (answer/clarify/refuse) is not trusted: uncited-1 sent a fact inside a "clarify" reply.
_NON_FACT = [re.compile(p, re.I) for p in (
    r"\b(sources?|material|information|documentation|website)\b.*\b(do(es)?\s+not|don't|doesn't|isn't|aren't|not)\b",
    r"(\bnot|n't)\s+(mention|cover|specif|list|describ|publish|state|includ|confirm|share|advise|provide|give)",
    r"\b(i|we)\s*('m|\s+am)?\s*(can't|cannot|can not|unable|won't|not able)\b",
    r"\badd\w*\b.*\blist\b",
    r"\b(hand\w*|connect\w*|pass\w*)\b.*\b(team member|account executive|person|someone|team)\b",
    r"\bkeep going\b",
    r"^(let me know|feel free|would you like|do you want|happy to|i'd be happy|i can help)\b",
    r"\bconsult\b[^.]*\b(qualified|lawyer|attorney|advis|accountant|professional)",
)]
_PLEASANTRY = re.compile(r"^(hi|hello|hey|thanks|thank you|of course|sure|great|happy to help|got it|understood)\b", re.I)


def factual_sentences(reply: str) -> list[str]:
    """Sentences that look like statements of fact (everything that is not a question or a non-fact line)."""
    out = []
    for s in re.split(r"(?<=[.!?])\s+", strip_markup(reply).strip()):
        s = s.strip()
        if not s or s.endswith("?"):
            continue
        if _PLEASANTRY.match(s) and len(s.split()) <= 6:
            continue
        if any(p.search(s) for p in _NON_FACT):
            continue
        out.append(s)
    return out

WATCH_STEMS = ("certif", "guarantee", "sla", "free trial", "annual", "unlimited", "refund", "lifetime")

_NUM = re.compile(r"\d[\d,]*(?:\.\d+)?")
# Citation markup the agent sometimes writes into prose: "security#2", "[faq#5]", URLs. Not content.
_MARKUP = re.compile(r"\[?\b[a-z][a-z0-9-]*#\d+\]?|https?://\S+", re.I)


def strip_markup(text: str) -> str:
    return _MARKUP.sub(" ", text)


def extract_numbers(text: str) -> set[str]:
    out = set()
    for raw in _NUM.findall(text):
        raw = raw.rstrip(",.")
        if re.fullmatch(r"\d{1,3}(,\d{3})+", raw):
            raw = raw.replace(",", "")
        else:
            raw = raw.replace(",", ".")
        out.add(raw)
    return out


def _norm(text: str) -> str:
    return " " + " ".join(re.findall(r"[a-z0-9]+", text.lower())) + " "


def _watch_words(text: str) -> set[str]:
    low = text.lower()
    found = set()
    for stem in WATCH_STEMS:
        pattern = r"\b" + re.escape(stem) + (r"\b" if stem == "sla" else "")
        if re.search(pattern, low):
            found.add(stem)
    return found


_NAME = re.compile(r"\b[A-Z][A-Za-z0-9&+-]{2,}\b")
_NAME_IGNORE = {"Bob", "HiBob", "Yes", "Our", "Your", "You", "The", "This", "That", "These", "With", "For", "And",
                "However", "Also", "Great", "Sure", "Hi", "Hello"}


def names(sentence: str) -> set[str]:
    """Capitalised names in a sentence, ignoring its first word (capitalised by grammar, not because it's a name)."""
    words = sentence.split()
    rest = " ".join(words[1:]) if words else ""
    return {n for n in _NAME.findall(rest) if n not in _NAME_IGNORE}


def _canon(word: str) -> str:
    """Comparable form of a name: lower case, no possessive or plural s, British/American -ise/-ize alike."""
    w = word.lower().replace("’", "'")
    w = re.sub(r"'s$", "", w)
    w = w.replace("iz", "is").replace("yz", "ys")
    if len(w) > 3 and w.endswith("s") and not w.endswith("ss"):
        w = w[:-1]
    return w


def _missing_names(text: str, allowed: str, glossary: set[str] = frozenset()) -> list[str]:
    allowed_canon = {_canon(w) for w in re.findall(r"[A-Za-z0-9'’]+", allowed)}
    allowed_canon |= {_canon(w) for term in glossary for w in term.split()}
    out = []
    for n in names(text):
        parts = [p for p in re.split(r"[-&+]", n) if p] or [n]
        if not all(_canon(p) in allowed_canon for p in parts):
            out.append(n)
    return sorted(out)


def check_reply(draft: dict, chunks: list[Chunk], user_text: str = "", glossary=frozenset()) -> dict:
    """glossary: the client's own product and module names, allowed in any sentence (config: glossary)."""
    by_id = {c.id: c for c in chunks}
    problems: list[dict] = []

    def add(kind, detail):
        p = {"type": kind, "detail": detail}
        if p not in problems:
            problems.append(p)

    if not str(draft.get("reply", "")).strip():
        add("empty_reply", "the reply has no text")
    if not draft.get("handoff") and not (draft.get("claims") or []):
        facts = factual_sentences(str(draft.get("reply", "")))
        if facts:
            add("no_claims", f"states facts with no quoted claim: {facts[0][:80]}")

    # Numbers are checked against the whole cited section (quotes are often shorter than the fact);
    # watch-list words against the quote itself, because a section can say "certified" about one thing
    # while the claim applies it to another (e.g. "HIPAA certifications").
    quotes, sections = [], []
    for cl in draft.get("claims") or []:
        text = strip_markup(str(cl.get("text", "")))
        src, quote = str(cl.get("source", "")).strip().strip("[]"), str(cl.get("quote", ""))
        chunk = by_id.get(src)
        if chunk is None:
            add("unknown_source", src or "(empty)")
            continue
        section = f"{chunk.title} {chunk.heading} {chunk.text}"
        if not quote.strip() or _norm(quote) not in _norm(section):
            add("quote_not_found", f"{src}: {quote[:80]}")
            continue
        quotes.append(quote)
        sections.append(section)
        for n in sorted(extract_numbers(text) - extract_numbers(section) - extract_numbers(user_text)):
            add("number_not_in_source", n)
        for w in sorted(_watch_words(text) - _watch_words(quote) - _watch_words(user_text)):
            add("watch_word", w)
        # Names must come from the cited section. The buyer naming a product does not make it a fact.
        for n in _missing_names(text, section, glossary):
            add("unsupported_name", f"{n} (claim cites {src})")

    # Whole-reply pass: nothing in the reply may go beyond the cited material plus the buyer's words.
    reply = strip_markup(str(draft.get("reply", "")))
    for n in sorted(extract_numbers(reply) - extract_numbers(" ".join(sections) + " " + user_text)):
        add("number_not_in_source", n)
    for w in sorted(_watch_words(reply) - _watch_words(" ".join(quotes) + " " + user_text)):
        add("watch_word", w)
    for sentence in factual_sentences(str(draft.get("reply", ""))):
        for n in _missing_names(sentence, " ".join(sections), glossary):
            add("unsupported_name", f"{n} (not in any cited section)")

    return {"ok": not problems, "problems": problems}
