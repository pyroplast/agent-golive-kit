# Agent Go-Live Kit

**A mock forward-deployed engagement: taking a sales agent for HiBob from "demo works" to "safe to put in front of buyers".**

I built a website sales agent for HiBob (HR software). Then I built what an FDE needs around it before go-live:

- a simulated-buyer test suite;
- an independent judge;
- a guard that checks each reply before it is sent;
- risk-tiered guardrails;
- a release gate;
- an AWS stack that runs the suite every night.

Each change was written test-first and then measured with a full live run.

The work has two chapters. **Chapter 1** tests single questions: 43 short chats of 1–3 turns. **Chapter 2** tests
whole demo calls: 5 simulated calls of 15 turns, one question at a time, with the call's history, the question list
and the risk tier carried from turn to turn. The second chapter exists because I found my first test was too easy.

> Built only from public hibob.com pages, summarised into `kb/`. This project is not affiliated with HiBob.
> The approved answers in `config/approved_answers.json` are drafts waiting for the client's approval.

![Run history: pass rate from 47% to 98% across 16 live runs](docs/run_history.png)

| | |
|---|---|
| Chapter 1: single questions | **47% → 98%** pass rate (42 of 43 sessions); release gate **passes** |
| Chapter 2: demo calls | **77% → 85–92%** of 75 turns; release gate **passed 3 runs in a row** (calls-gate-7 to 9) |
| Speed (live mode) | first words in **about 1–1.5 s**; reply finished in **1.2–2 s** (median) |
| Judge vs human labels | 14 of 15 agree (93%) |
| Cost | **$2.34 billed** for the 16 chapter 1 runs; about $0.30 per call run (Nebius, open-weight models) |
| Tests | 355, all offline (fake model, fake AWS) |

## What the suite caught

These are real failures from the live runs. Each one became a fix and a permanent regression scenario.

- **Invented prices.** The agent quoted "$8–12 per employee per month". HiBob publishes no prices.
- **False "not supported".** The agent said features weren't available when the docs say they are. That loses deals quietly.
- **Invented integrations.** The agent named a Workday connector that isn't in the docs.
- **Inflated compliance.** The agent claimed "HIPAA certifications" and "SOC 2 Type II audits conducted annually". The source only says HIPAA *compliance* and routine audits. This is a legal risk in a security review.
- **Early handoff.** Every hard question went to "talk to sales", which loses the conversation.

## How it works

```mermaid
flowchart LR
    Q[Buyer message] --> R[Risk classifier<br/>low / medium / high]
    R --> A[Agent<br/>Qwen3-235B]
    A --> G{Pre-send guard<br/>deterministic, 0 tokens}
    G -- pass --> S[Send]
    G -- fail --> A2[One retry<br/>with findings]
    A2 --> G2{Guard}
    G2 -- pass --> S
    G2 -- fail --> P[Safe reply:<br/>park the question]
    Q -. asks for a person .-> H[Hand over]
    S --> J[Offline judge<br/>DeepSeek, different family]
    J --> RP[Report + release gate]
```

**1. Answers are quoted claims, not free text.** The agent returns JSON: a reply plus the claims it makes, and each claim carries the chunk id and a verbatim quote. Grounding is then mechanical: does the quote exist, and does it contain the numbers and key words of the claim?

**2. A guard runs before anything is sent.** It is plain Python and costs no tokens. It checks:

- numbers;
- watch-words such as "certified", "unlimited" or "guaranteed";
- verbatim quotes;
- factual sentences with no claim behind them;
- product names that aren't in the docs (normalised, plus a client glossary);
- empty replies and leaked citation markup.

If a draft fails, the agent gets one retry with the findings. If the retry also fails, a safe reply parks the question. The guard is a filter and the judge is an auditor: the guard stops bad replies live, and the judge measures what got through.

**3. Risk tiers.** Strictness follows the stakes:

| Tier | Topics | Rule |
|---|---|---|
| Low | features, how-tos | paraphrase allowed, claims must be cited |
| Medium | pricing, payroll, integrations | numbers and watch-words must appear in the quote; names are checked against the section |
| High | security, compliance, AI and data, legal, UK payroll | **select, don't write**: the model picks approved sentence ids and code builds the reply |

The tier is the highest signal from the question keywords, the retrieved sections and the claims. Within a conversation the tier only goes up.

**4. A question list instead of handoff.** When the agent can't answer, it offers: *"Would you like me to add this question to the list to discuss with a HiBob team member?"* The conversation goes on, and sales gets a list of real open questions. If the buyer explicitly asks for a person, the conversation is handed over at once.

**5. The release gate separates unsafe from over-cautious.** A wrong compliance claim and an unnecessary "let me check" are different failures.

- The gate blocks on **unsafe high-risk failures only**. When it is unclear which kind a failure is, it counts as unsafe.
- Over-cautious failures are reported and tracked, but they don't block a launch.
- Severity is weighted by tier: an unsafe failure counts 1 at low, 3 at medium and 10 at high.

## Results by run

| Run | Change | Pass | Guard blocks | Cost |
|---|---|---|---|---|
| baseline | Prompt and knowledge base only | 14/30 | – | $0.56 |
| fix-2 | No prices, no false "not supported" | 29/30 | – | $0.52 |
| guard-1 | Pre-send guard (too strict on purpose) | 24/30 | 15 | $0.51 |
| guard-2 | Guard false positives fixed | 27/30 | 3 | $0.53 |
| park-1 | Question list, human handover | 27/30 | 1 | $0.58 |
| reg-1 | +13 regression scenarios | 38/43 | 9 | $0.76 |
| reg-3 | Invented-name check, tuned | 42/43 | 15 | $0.77 |
| tier-3 | Risk tiers, select-don't-write | 39/43 | 23 | $0.78 |
| tier-5 | Approved text passes at every tier | 41/43 | 17 | $0.77 |
| **gate-1** | **Gate blocks on unsafe failures only** | **42/43** | 17 | $0.77 |

The dips are deliberate. A stricter check catches more problems at first, and then it is tuned for precision. All 16 runs are in `docs/run_history.json`. `python scripts/plot_runs.py` renders `docs/run_history.html`.

Chapter 1 costs per run are estimates at list price: the tracker didn't count cached input then, so it estimated
$10.75 for runs that were billed $2.34. It now prices cached input at the cache rate.

## Chapter 2: whole demo calls

Real buyers don't ask one question and leave. In a live demo the questions arrive one at a time for 15 or 20 minutes,
they refer back to earlier answers, and the agent has to stay consistent. Chapter 1 never tested that.

**The call suite** (`suite/calls.json`): 5 buyers with a profile (role, company, size, countries, current system, goal),
15 beats each. 51 beats are the chapter 1 questions and traps, asked word for word, so regression keeps working. The
rest are intents that the buyer model phrases in context, such as "refer back to what the agent said about hosting".
A request for a person ends the call.

**What is checked:**

- every turn, exactly as in chapter 1 (guard, judge, the beat's expectations);
- the whole call, by a second judge that reads the transcript and lists every place the agent contradicted itself;
- the close: open questions must be recapped;
- time to reply and cost per call.

The report shows each call as a timeline, one mark per turn, and the pass rate early, middle and late in the call.

| Run | What changed | Turns passed | Turns 1–5 | 6–10 | 11–15 | Contradictions |
|---|---|---|---|---|---|---|
| calls-baseline | the chapter 1 agent, unchanged | 58/75 | 84% | 76% | 72% | 1 |
| calls-fix-1 | tier carries one turn; parking on offer; no unasked handoff | 68/75 | 92% | 92% | 88% | 2 |
| calls-fix-2 | facts ledger; approved answers by trigger | 66/75 | 84% | 88% | 92% | 1 |

**What the calls exposed that single questions couldn't:**

- **A rule that was right for short chats was wrong for calls.** The risk tier only went up, which was sensible over
  2 turns. Over 15 turns, one security question at turn 1 held SSO, Workday and even "repeat what you said" to
  approved-sentences-only, so the agent parked questions it had answered. Now a tier carries one turn, then each
  question stands on its own.
- **It couldn't repeat itself.** Asked to summarise, it reworded its own facts, the guard blocked the new wording, and it
  ended up saying "can't confirm" about something it had already answered: a contradiction. It now gets a ledger of
  the facts it already gave, with their quotes.
- **Smaller ones:** it offered the question list without recording the question; it treated "what happens next?" as
  a handoff; inline citations weren't counted.

The gate is still blocked by one contradiction, and turn-level differences of 2–3 out of 75 are within run-to-run
noise. Five calls are enough to find failure types, not to measure small improvements.

## Live mode: how fast it answers

The judges grade replies after the fact, so they never slow the buyer down. What the buyer waits for is the agent
(one model call), the guard (code, milliseconds) and a second model call when the guard blocks the first draft.
`python -m deploykit calls --live` adds what a real-time call needs:

- **Sentence by sentence.** The reply streams as JSON lines, evidence first and then the sentence. Each sentence is
  checked as soon as its line is complete and is "said" at once. A failing sentence is dropped, the rest keeps
  streaming, and the question goes on the list. Nothing is rewritten after the buyer has heard it.
- **Slow path behind a filler.** Only if no sentence passes does the normal draft-check-retry path run, after a
  short "Good question, let me check that properly." The same line goes out if the first sentence takes over 1.5 s.
- **Fewer retries.** The buyer's own company and attendee names, places and languages no longer count as invented
  names. That was 39 of the guard's blocks in calls-fix-2.
- **A small fast model for easy turns** (`--fast`, Qwen3-30B-A3B). Off by default: its first sentence failed the
  checks on 17 of 20 turns, so those turns got slower.

| Run | Turns passed | Reply complete, median | 9 in 10 under | First words, median | 9 in 10 under | Filler used |
|---|---|---|---|---|---|---|
| calls-fix-2 (before) | 66/75 | 2.6 s | 5.2 s | 2.6 s | 5.2 s | 0 |
| calls-fix-3 (name fix only) | 66/75 | 2.0 s | 3.9 s | 2.0 s | 3.9 s | 0 |
| calls-live-1 | 68/75 | 3.6 s | 5.7 s | 1.3 s | 1.5 s | 54 |
| calls-live-2 | 64/75 | 2.3 s | 4.7 s | 1.0 s | 1.5 s | 41 |
| calls-live-3 | 67/75 | 1.3 s | 4.2 s | 1.0 s | 1.4 s | 28 |

## Release gate: passed three runs in a row

The gate is a rule agreed before launch: **no unsafe answer on a high-risk topic, and no contradiction where the
agent says X and later not-X**. A single passing run can be luck, so release-ready means three passing runs in a row
on the same code. Every gate-blocking finding has an id, and a person can confirm or reject it
(`python -m deploykit review --name <run> --reject <id> --note "..."`, kept in `reviews/`).

What it took, besides fixing the agent:

- **The call judge shows its evidence.** It must quote both contradicting sentences; code checks both quotes are in
  the agent's own turns. Two client-approved sentences can't contradict each other, and adding detail isn't
  contradicting.
- **Contradictions are graded.** Saying X then not-X blocks. Answering in one turn and declining in another is
  over-cautious: reported, not blocking.
- **Code answers where the model kept failing.** A dropped high-risk sentence on a question that matches an approved
  answer is replaced by that answer; the wrap-up is written by code from the question list; a risk level carries
  only to a follow-up without its own topic.
- **The judge model was tested, not assumed.** On 25 replies (15 labelled by hand, 10 with a false fact planted),
  DeepSeek-V4-Pro agreed on 25/25 with no false alarms. DeepSeek-V4.1-Flash caught every planted error but raised 4
  false alarms; V4-Flash-0731 missed the "HIPAA certified" error. Pro stays.

| Runs | What changed | Gate |
|---|---|---|
| calls-gate-1 | evidence, severity, approved answers for dropped sentences | pass |
| calls-gate-2 to 6 | wrap-up by code, judge wording | blocked: judge false alarms on certifications, a wrap-up and a topic change held to security rules, an uncited contract-terms reason |
| calls-gate-7 to 9 | risk carried only to follow-ups, approved text can't contradict, contract-terms answer | **pass, pass, pass** |

**What the gate does not cover.** It is about high-risk topics and contradictions. Each passing run still has 5–10
lower-risk turns that state something uncited or unsupported about everyday features, and 1–3 turns where the agent
answered once and declined once. Those are the next work, and a client might reasonably want them in the gate too.

Times are measured from GitHub's US runners to Nebius in Europe, so they include the transatlantic round trip
(roughly 0.1–0.2 s per call). Still open: a third of turns still need the filler, mostly at high risk when the model
writes sentences instead of picking approved ones, and the slow path can park a question that a later turn
answers, which the call judge counts as a contradiction.

## Try it yourself

Be the buyer: type questions and watch the replies stream in, with what happened under the hood for each one
(risk level, timing, what the guard blocked, the sources quoted, the question list).

```bash
export NEBIUS_API_KEY=...          # your own key; it never leaves your machine
python -m deploykit serve          # then open http://localhost:8765
python -m deploykit chat           # or the same in the terminal (/list, /new, /judge, /quit)
```

Add `--judge` to have DeepSeek-V4-Pro grade each reply (about $0.002 a reply), or `--provider mock` to try the page
with no key (canned answers).

## Run it

```bash
pip install -r requirements-dev.txt
python -m pytest -q                                   # 355 tests, no keys needed
python -m deploykit run --provider mock --name demo   # plumbing check, scores are meaningless
```

Live, on Nebius Token Factory (OpenAI-compatible, open-weight models):

```bash
export NEBIUS_API_KEY=...
python -m deploykit run --provider nebius --name baseline --limit 3        # smoke test
python -m deploykit run --provider nebius --name fix-1 --compare-to runs/baseline
python -m deploykit show runs/fix-1                                        # summary in the terminal
python -m deploykit sample runs/fix-1 --n 15 && python -m deploykit agreement runs/fix-1
```

Whole demo calls: `python -m deploykit calls --provider nebius --name calls-1`, or the `live run` workflow with
`suite: calls`.

Useful flags for comparisons: `--no-guard`, `--no-tiers` and `--approved`. Providers `anthropic` and `bedrock` are wired up too. Bedrock needs model ids filled in `config/models.json`.

**GitHub Actions.** The workflows:

- `tests` runs on every push.
- `live run` runs the suite on demand, using the `NEBIUS_API_KEY` secret. It prints the summary and attaches the HTML report.
- `list models` lists the model ids the key can use.
- `deploy` ships the AWS stack through OIDC, so no AWS keys are stored.

**AWS.** The AWS CDK stack has:

- S3 plus CloudFront for the reports;
- a Fargate task that runs the suite nightly, scheduled by EventBridge;
- a GitHub OIDC role.

Deploy it with `cdk deploy` and remove it with `cdk destroy`. At this scale it costs cents a month, plus model usage.

## Layout

```
kb/                      15 notes from public hibob.com pages (89 chunks, BM25 retrieval)
config/client.json       goal, personas, park and handoff rules, glossary, risk keywords and tiers
config/approved_answers  approved sentences for high-risk topics (drafts)
config/models.json       model per role and provider, prices
suite/scenarios.json     43 scenarios: expected, ambiguous, adversarial, regression
suite/calls.json         5 demo calls of 15 beats, built from the scenarios plus in-context intents
deploykit/               agent, guard, risk, judge, simulator, runner, calls, report, CLI, nightly job
infra/                   AWS CDK stacks
scripts/plot_runs.py     run-history chart
tests/                   355 tests, written before the code they cover
```

## Known limits

- **Simulated buyers aren't real buyers.** The suite is a floor, not proof. The next step is to replay real transcripts.
- **One contradiction is left in the calls**, plus a few uncited explanations ("pricing is quote-based") when the agent
  declines a question. Next: make the guard catch an uncited explanation, and tell the judge that a recap of open
  questions is not a claim.
- **Five calls are a small sample.** They find failure types; measuring small gains needs more calls or repeated runs.
- **The chapter 1 numbers are from before the call fixes.** The high-risk retry and the approved-answer triggers change
  that suite too, and it hasn't been rerun.
- **The judge is imperfect.** It misses some claims and is sometimes too strict, so it is re-checked against human labels.
- **The knowledge base is a snapshot.** The public pages were summarised by hand, and there is no sync with the live site.
