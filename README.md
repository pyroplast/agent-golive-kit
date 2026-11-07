# HiBob mock deployment: pre-launch simulation kit

A mock forward-deployed engagement: a sales demo agent for HiBob's product Bob, tested before go-live
with simulated buyers, a judge model, a go-live report and nightly regression runs on AWS.

Built from HiBob's public website only. The knowledge base (`kb/`) holds short factual notes
summarised from public hibob.com pages, each with its source link. No HiBob internal data is used.
This project is not affiliated with HiBob or Supersonik.

## What it does

1. **Agent** (`deploykit/agent.py`): answers prospects only from `kb/`, cites chunk ids, follows the
   qualification and handoff rules in `config/client.json`, replies in English.
2. **Simulation suite** (`suite/scenarios.json`): 30 scenarios (10 expected, 10 ambiguous,
   10 adversarial) for three buyer personas, in English = 30 sessions.
   Follow-ups are scripted so runs are comparable; a scenario can set `"simulate": true` to let a
   model play the buyer.
3. **Judge** (`deploykit/judge.py`): a stronger model checks every reply against the sources
   (grounded, completeness, handoff, language, knowledge gap). Rule checks add missed or unnecessary
   handoffs, invalid citations and parse errors.
4. **Report** (`report.html` per run): pass rate by scenario type, persona and language, failure
   reasons, knowledge gaps, risky answers, every transcript, and run cost.
5. **Regression**: `diff` lists sessions that were fixed or broke between two runs.
6. **Judge check**: `sample` picks turns for you to label by hand, `agreement` scores the judge
   against you.
7. **AWS** (`infra/`): private S3 bucket + CloudFront for reports, a Fargate task on a nightly
   schedule that runs the suite on Amazon Bedrock, diffs against the last run and publishes.

## Quick start (no keys needed)

```bash
pip install -r requirements-dev.txt
python -m pytest -q                      # 48 tests, fake model and fake AWS
python -m deploykit run --provider mock --name mock-baseline
open runs/mock-baseline/report.html
```

The mock provider is a keyword matcher. It tests the plumbing; its scores mean nothing.

## Live run

Nebius Token Factory (open-weight models, OpenAI-compatible API):

```bash
export NEBIUS_API_KEY=...
python -m deploykit models                       # list available model ids
export MODEL_AGENT=... MODEL_BUYER=... MODEL_JUDGE=...
python -m deploykit run --provider nebius --name baseline --limit 3   # smoke test
python -m deploykit run --provider nebius --name baseline
```

Anthropic API:

```bash
export ANTHROPIC_API_KEY=...             # a key with a spending limit
python -m deploykit run --provider anthropic --name baseline --limit 3   # smoke test
python -m deploykit run --provider anthropic --name baseline
```

Amazon Bedrock (uses your AWS credentials, no API key):

1. In the Bedrock console, enable access to the Claude models in your region.
2. Put the inference profile ids in `config/models.json` under `bedrock`.
3. `python -m deploykit run --provider bedrock --name baseline`

Fill `prices_usd_per_million_tokens` in `config/models.json` to get the run cost in USD.
Token counts are always recorded.

## Fix, rerun, compare

```bash
# edit kb/ or config/client.json based on the report, then:
python -m deploykit run --provider bedrock --name after-fix-1 --compare-to runs/baseline
python -m deploykit sample runs/after-fix-1 --n 15     # label human_grounded in labels.csv
python -m deploykit agreement runs/after-fix-1
```

## Deploy the nightly job to AWS

Needs the AWS CDK CLI (`npm install -g aws-cdk`) and Docker.

```bash
cdk bootstrap        # once per account and region
cdk deploy
```

Outputs: the bucket name, the public report URL (CloudFront), and the cluster and task definition
for a manual run. The schedule runs at 02:17 UTC. Costs at this scale: cents a month for S3,
CloudFront and Fargate, plus model usage on Bedrock. `cdk destroy` removes everything.

## Running from GitHub Actions (no AWS keys stored)

One-time, from a laptop with AWS credentials:

```bash
cdk bootstrap
cdk deploy GitHubAccess          # prints RoleArn
```

Then in the repo's Settings → Secrets and variables → Actions → Variables, add `AWS_ROLE_ARN`,
`MODEL_AGENT`, `MODEL_JUDGE`, `MODEL_BUYER`. Workflows:

- **tests**: runs on every push.
- **deploy**: deploys the report site and nightly job; add its bucket output as `REPORT_BUCKET`.
- **list models**: prints the Nebius model ids your key can use.
- **live run**: runs the suite on Nebius or Bedrock on demand; `compare_to` diffs against an earlier run; with `REPORT_BUCKET` set it publishes and
  diffs against the previous run. The report is attached to the workflow run.

## Layout

```
kb/                 knowledge base notes, one file per public page
config/client.json  agent goal, personas, qualification, handoff rules
config/models.json  model ids per role and provider, prices
suite/              simulation scenarios
deploykit/          agent, judge, simulator, runner, report, publish, nightly job, CLI
infra/              AWS CDK stack
tests/              48 tests, written before the code
```
