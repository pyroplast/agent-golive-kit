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

> Built only from public hibob.com pages, summarised into `kb/`. This project is not affiliated with HiBob.
> The approved answers in `config/approved_answers.json` are drafts waiting for the client's approval.

![Run history: pass rate from 47% to 98% across 16 live runs](docs/run_history.png)

| | |
|---|---|
| Pass rate | **47% → 98%** (42 of 43 simulated sessions) |
| Release gate | **Pass**: 0 unsafe high-risk failures |
| Judge vs human labels | 14 of 15 agree (93%) |
| Cost of all 16 live runs | **$10.75** on open-weight models (Nebius) |
| Tests | 178, all offline (fake model, fake AWS) |

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

## Run it

```bash
pip install -r requirements-dev.txt
python -m pytest -q                                   # 178 tests, no keys needed
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
deploykit/               agent, guard, risk, judge, simulator, runner, report, CLI, nightly job
infra/                   AWS CDK stacks
scripts/plot_runs.py     run-history chart
tests/                   178 tests, written before the code they cover
```

## Known limits

- **Simulated buyers aren't real buyers.** The suite is a floor, not proof. The next step is to replay real transcripts.
- **Backups and disaster recovery.** The model doesn't always pick the right approved sentence, which gives the one over-cautious failure. A planned fix: code pre-selects the candidate sentences for each question.
- **The judge is imperfect.** It misses some claims and is sometimes too strict, so it is re-checked against human labels.
- **The knowledge base is a snapshot.** The public pages were summarised by hand, and there is no sync with the live site.
