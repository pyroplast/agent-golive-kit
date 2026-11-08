"""Pre-send guard. Deterministic, zero model tokens.

Runs on the agent's draft before the buyer sees it. Three checks, each from a real failure:

1. Numbers: every number in the reply must appear in a cited section or in the buyer's words.
   (Baseline: an invented "$8–12 per employee per month".)
2. Watch-list words: high-risk words such as "certified" or "annual" must appear in the quote that
   backs the claim, or in the buyer's words. (Judge check: "HIPAA certifications", "audits annually".)
3. Quotes: each claim carries a verbatim quote, and the quote must exist in the cited source.
   Turns "is this answer true?" into string matching.

Checks run per claim (claim text vs its own quote) and over the whole reply (reply vs all quotes),
so text outside the claims cannot slip through.
"""
from __future__ import annotations

import re

from deploykit.kb import Chunk

# Replies that state no product facts and so need no claims. A missing kind counts as "answer".
NO_CLAIM_KINDS = {"clarify", "refuse"}

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


def check_reply(draft: dict, chunks: list[Chunk], user_text: str = "") -> dict:
    by_id = {c.id: c for c in chunks}
    problems: list[dict] = []

    def add(kind, detail):
        p = {"type": kind, "detail": detail}
        if p not in problems:
            problems.append(p)

    if not str(draft.get("reply", "")).strip():
        add("empty_reply", "the reply has no text")
    exempt = (draft.get("kind") in NO_CLAIM_KINDS or draft.get("unknown") or draft.get("handoff")
              or draft.get("park"))
    if not exempt and not (draft.get("claims") or []) and str(draft.get("reply", "")).strip():
        add("no_claims", "an answer must list each fact as a claim with a verbatim quote")

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

    # Whole-reply pass: nothing in the reply may go beyond the cited material plus the buyer's words.
    reply = strip_markup(str(draft.get("reply", "")))
    for n in sorted(extract_numbers(reply) - extract_numbers(" ".join(sections) + " " + user_text)):
        add("number_not_in_source", n)
    for w in sorted(_watch_words(reply) - _watch_words(" ".join(quotes) + " " + user_text)):
        add("watch_word", w)

    return {"ok": not problems, "problems": problems}
