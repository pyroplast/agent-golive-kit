"""Live mode: what a real-time demo call needs on top of the demo agent.

- The reply streams as JSON lines, one sentence per line with its own claims, and each sentence is checked by the
  guard as soon as its line is complete. A passing sentence is "said" at once.
- A sentence that fails is dropped and the rest keeps streaming; the reply then ends with a safe line that parks
  the question. Nothing is rewritten after the buyer has heard it. Only if no sentence passes does the slow path
  run (the normal draft, guard, retry and approved-answer fallback) behind a short filler line.
  (calls-live-1 stopped at the first bad sentence instead: 53 of 75 turns went down the slow path.)
- Optionally (fast=True), easy low-risk turns go to a smaller, faster model. Off by default: in calls-live-1 the
  small model's first sentence failed the checks on 17 of 20 turns, which made those turns slower, not faster.
- If the first sentence takes longer than FILLER_AFTER_S, the filler line goes out first.

In this text simulation "said" means: the time the sentence was ready, measured from the buyer's message.
"""
from __future__ import annotations

import json
import threading
import time

import re

from deploykit.agent import OUTPUT_SPEC, DemoAgent, carried_tier, is_question

_REPEAT_ASK = re.compile(r"\b(?:repeat|again|remind|recap|summar|earlier you|you said|you mentioned)", re.I)


def _norm(s: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9]+", " ", str(s).lower()).split())


_STOP = set("a an the and or of to in on for with by is are be it its this that your you we our so as at from "
            "can will all also".split())


def _words(s: str) -> set[str]:
    return {w for w in _norm(s).split() if w not in _STOP}


def _same_quote(a: str, b: str) -> bool:
    """Two quotes from one source overlap: one contains the other, or they share most of their words."""
    a, b = _norm(a), _norm(b)
    if not a or not b:
        return False
    if a in b or b in a:
        return True
    wa, wb = set(a.split()), set(b.split())
    return len(wa & wb) / len(wa | wb) >= 0.5


def _restates(say: str, earlier: list[str]) -> bool:
    """A sentence mostly made of the words of something the agent already said (Pavlo, 9 Nov)."""
    w = _words(say)
    if len(w) < 5:
        return False
    return any(len(w & _words(e)) / len(w) >= 0.7 for e in earlier)
from deploykit.guard import approved_sources
from deploykit.risk import max_tier

LIVE_SPEC_MARK = "ONE LINE PER SENTENCE"
LIVE_SPEC = f"""Reply as JSON LINES, {LIVE_SPEC_MARK}, so the prospect hears each sentence as soon as it is ready.
On each line write the evidence FIRST, then the sentence that uses it:
{{"claims": [{{"source": "<chunk id>", "quote": "<exact words copied from that chunk>"}}], "say": "<one sentence>"}}
Example lines:
{{"claims": [{{"source": "integrations#2", "quote": "Jira"}}], "say": "Bob connects to Jira."}}
{{"claims": [], "say": "Which other tools does your team use?"}}
A sentence that states ANY fact about Bob or HiBob, including a name, number or feature, must have at least one
claim on its own line; if you cannot quote it, leave the sentence out. Only questions, greetings and the offer to
add a question to the list have "claims": [].
At HIGH risk, do not write facts yourself: send one line per approved sentence instead: {{"approved": "<approved sentence id>"}}
Put the most useful sentence first and keep it short. Two to four sentences in total.
Last line: {{"end": true, "kind": "<answer | clarify | refuse>", "handoff": <true|false>, "handoff_reason": <string or null>, "unknown": <true|false>, "park": <the question or null>, "qualification": {{"company_size": "", "countries": "", "current_system": "", "modules": "", "timeline": ""}}}}
In "qualification" fill only what the prospect has said in this call, in their words; leave the rest empty.
No other text and no code fences. Each sentence is checked automatically before it is sent: one with a number,
name or risky word that is not in its quote stops the reply."""

# No filler sentence: when the first words take longer than FILLER_AFTER_S, the chat shows typing dots instead
# (Pavlo, 9 Nov), and the dots disappear as soon as the answer starts.
FILLER = ""
SAFE_TAIL = ("I can't confirm the rest from HiBob's material. I have added this question to the agenda for the "
             "discussion with the sales representative.")
FILLER_AFTER_S = 1.5


class LiveAgent(DemoAgent):
    def __init__(self, *args, fast: bool = True, clock=time.perf_counter, **kwargs):
        super().__init__(*args, **kwargs)
        self.fast, self.clock = fast, clock
        self.on_event = None  # optional listener: gets each sentence the moment it may be said (chat UI)

    def _emit(self, kind, text, t0):
        if self.on_event:
            self.on_event({"type": kind, "text": text, "t": round(self.clock() - t0, 3)})

    def _respond(self, history, lang, parked, floor=None, ledger=None):
        t0 = self.clock()
        sources = self.sources_for(history)
        last_q = history[-1]["content"] if history and history[-1]["role"] == "user" else ""
        user_text = " ".join(m["content"] for m in history if m["role"] == "user")
        pre_tier = carried_tier(self.policy, last_q, floor) if self.policy else None
        role = "agent_fast" if self.fast and pre_tier in (None, "low") else "agent"
        system = self._system(sources, parked, pre_tier, last_q, ledger).replace(OUTPUT_SPEC, LIVE_SPEC)
        # The agent never sees its own filler lines, or it starts copying them (calls-live-1 turn 6 said it twice).
        history = [{**m, "content": m["content"].replace("Good question, let me check that properly.", "").strip()}
                   if m["role"] == "assistant" else m for m in history]
        replay = [{"role": m["role"], "content": json.dumps({"say": m["content"]}, ensure_ascii=False)}
                  if m["role"] == "assistant" else m for m in history]

        said, claims, approved_cites, end = [], [], [], {}
        held = []  # questions to the prospect, said after the answer (and after a "can't confirm" line)
        # Facts already given in this call (source + quote). A sentence that only repeats them is left out, unless
        # the prospect asked to hear it again (Pavlo, 9 Nov: the OKR facts came back after the buyer answered).
        given = [(f.get("source"), f.get("quote", "")) for f in ledger or []]
        earlier = [x for m in history if m["role"] == "assistant"
                   for x in re.split(r"(?<=[.!?])\s+", m["content"]) if x.strip()]
        allow_repeat = bool(_REPEAT_ASK.search(last_q))
        repeated = [0]
        tier, first_at, problems, failed, skipped = pre_tier, None, [], None, 0
        skip_at = None  # where the first dropped sentence would have been said
        buf, stop = "", False
        first_skip_s = [None]
        substitute = []
        filler_sent = threading.Event()
        timer = None
        if self.on_event:  # in a live chat the filler goes out at the threshold, while the model is still working
            def _filler():
                if first_at is None and not filler_sent.is_set():
                    filler_sent.set()
                    self._emit("typing", "", t0)
            timer = threading.Timer(FILLER_AFTER_S, _filler)
            timer.start()

        def handle(line):
            nonlocal tier, first_at, problems, failed, skipped, end, skip_at, substitute
            line = line.strip().strip("`")
            if not line:
                return False
            try:
                obj = json.loads(line)
            except ValueError:
                return False  # not a line of the format: skip it
            if not isinstance(obj, dict):
                return False
            if obj.get("end"):
                end = obj
                return False
            if obj.get("approved") and self.policy and obj["approved"] in self.policy.units:
                u = self.policy.units[obj["approved"]]
                said.append(u["text"])
                self._emit("say", u["text"], t0)
                approved_cites.extend(s for s in u["sources"] if s in self.ids)
                tier = max_tier(tier or "low", "high")
                first_at = first_at if first_at is not None else self.clock() - t0
                return False
            say = str(obj.get("say", "")).strip()
            if not say:
                return False
            c = [x for x in obj.get("claims") or [] if isinstance(x, dict)]
            is_ask = say.rstrip().endswith("?")
            old_fact = bool(c) and bool(given) and all(
                any(x.get("source") == s and _same_quote(x.get("quote", ""), q) for s, q in given) for x in c)
            if not allow_repeat and not is_ask and (old_fact or _restates(say, earlier)):
                repeated[0] += 1
                return False
            mini = self._to_turn({"reply": say, "claims": c, "kind": "answer" if c else "clarify"}, False)
            s_tier = (max_tier(pre_tier, self.policy.classify(last_q, mini["citations"], mini["claims"]))
                      if self.policy else None)
            found = self._check(mini, user_text, s_tier)
            if found:  # drop this sentence, keep streaming the rest
                problems, failed = problems + found, failed or say
                if skip_at is None:
                    skip_at, first_skip_s[0] = len(said), self.clock() - t0
                # The question matches a client-approved answer: say that instead, right now (calls-live-3: AI
                # questions were parked at turns 3-4, then answered at turn 5).
                match = self.policy.matching(last_q) if self.policy and not substitute else []
                if 1 <= len(match) <= 2:
                    substitute = match
                    texts = {a["id"]: a for a in self.policy.approved}
                    for m in match:
                        said.append(texts[m]["text"])
                        self._emit("say", texts[m]["text"], t0)
                        approved_cites.extend(s for s in texts[m].get("sources", []) if s in self.ids)
                    tier = max_tier(tier or "low", "high")
                    first_at = first_at if first_at is not None else self.clock() - t0
                elif not substitute:
                    skipped += 1
                return False
            if say.endswith("?") and not c:
                held.append(mini["reply"])
                return False
            if first_at is None and self.clock() - t0 > FILLER_AFTER_S and not filler_sent.is_set():
                filler_sent.set()
                self._emit("typing", "", t0)
            say = mini["reply"]  # without [chunk#n] markers
            for h in held:  # a question that came before an answer keeps its place
                said.append(h)
                self._emit("say", h, t0)
            held.clear()
            said.append(say)
            self._emit("say", say, t0)
            claims.extend(c)
            tier = max_tier(tier or "low", s_tier) if s_tier else tier
            first_at = first_at if first_at is not None else self.clock() - t0
            return False

        for piece in self.llm.stream(role, system, replay, meta={"chunks": sources, "lang": lang}):
            buf += piece
            while "\n" in buf and not stop:
                line, buf = buf.split("\n", 1)
                stop = handle(line)
            if stop:
                break
        if not stop and buf.strip():
            stop = handle(buf)

        if timer:
            timer.cancel()
        if held and not said and not skipped:  # only questions: say them now
            for h in held:
                said.append(h)
                self._emit("say", h, t0)
            held.clear()
            first_at = self.clock() - t0

        if not said and not held:  # the first sentence failed, or nothing usable came back: slow path
            filler_at = min(self.clock() - t0, FILLER_AFTER_S)
            if not filler_sent.is_set():
                filler_sent.set()
                self._emit("typing", "", t0)
            turn = DemoAgent._respond(self, history, lang, parked, floor, ledger)
            self._emit("say", turn["reply"], t0)
            turn["guard"] = {**turn.get("guard", {}), "streamed": True, "slow_path": True, "skipped": skipped,
                             "stream_problems": problems, "blocked_sentence": failed}
            turn["model_role"] = "agent"
            turn["timing"] = {"streamed": True, "filler": True, "first_words_s": round(self.clock() - t0, 3),
                              "first_sentence_s": round(self.clock() - t0, 3), "total_s": round(self.clock() - t0, 3)}
            return turn

        if first_at is None:  # only questions were left to say, after the "can't confirm" line
            first_at = self.clock() - t0
        reply = " ".join(said)
        park = end.get("park")
        if skipped and not is_question(last_q):
            skipped_silently, skipped = skipped, 0  # an introduction: the blocked sentence is simply left out
        else:
            skipped_silently = 0
        if skipped:
            reply = f"{reply} {SAFE_TAIL}".strip()
            park = last_q or None
            self._emit("say", SAFE_TAIL, t0)
        if held:
            reply = " ".join([reply, *held]).strip()
            for h in held:
                self._emit("say", h, t0)
        filler = first_at > FILLER_AFTER_S
        kind = end.get("kind") or ("answer" if claims or approved_cites else "clarify")
        turn = self._to_turn({"reply": reply, "claims": claims, "kind": kind, "park": park,
                              "handoff": end.get("handoff", False), "handoff_reason": end.get("handoff_reason"),
                              "unknown": end.get("unknown", False), "qualification": end.get("qualification") or {}},
                             False)
        for s in approved_cites + (approved_sources(reply, self.policy.approved) if self.policy else []):
            if s in self.ids and s not in turn["citations"]:
                turn["citations"].append(s)
        turn["tier"] = tier
        turn["own_tier"] = (self.policy.classify(last_q, turn["citations"], turn["claims"]) if self.policy else None)
        if turn.get("unknown") and not turn.get("park") and not turn.get("handoff") and is_question(last_q):
            turn["park"] = last_q or None
            turn["reply"] = (turn["reply"].rstrip() + " " + self.config.get("park_offer", "")).strip()
        total = self.clock() - t0
        turn["guard"] = {"enabled": True, "streamed": True, "retried": False, "fell_back": False, "composed": False,
                         "slow_path": False, "skipped": skipped + skipped_silently, "approved_substitute": substitute,
                         "repeated": repeated[0],
                         "first_problems": problems, "final_problems": [],
                         "blocked_draft": failed}
        turn["model_role"] = role
        turn["timing"] = {"streamed": True, "filler": filler,
                          "first_words_s": round(first_at, 3),
                          "first_sentence_s": round(first_at, 3), "total_s": round(total, 3)}
        return turn
