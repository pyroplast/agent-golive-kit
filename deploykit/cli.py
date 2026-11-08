"""Command line: run, report, diff, sample, agreement."""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
from pathlib import Path

from deploykit.kb import load_kb
from deploykit.llm import AnthropicLLM, BedrockLLM, MockLLM, OpenAICompatLLM
from deploykit.report import build_report, load_rows, write_diff_md
from deploykit.runner import run_suite
from deploykit.scenarios import load_scenarios, validate_scenarios

ROOT = Path(__file__).resolve().parent.parent


def _models(path, provider):
    data = json.loads(Path(path).read_text())
    models = dict(data.get(provider, {}))
    for role in ("agent", "judge", "buyer"):  # env vars win, e.g. from GitHub Actions variables
        if os.environ.get(f"MODEL_{role.upper()}"):
            models[role] = os.environ[f"MODEL_{role.upper()}"]
    bad =[r for r in ("agent", "judge", "buyer") if not models.get(r) or "REPLACE" in models[r]]
    if bad:
        sys.exit(f"Set real model ids for {provider} roles {bad} in {path}")
    prices = {k: v for k, v in data.get("prices_usd_per_million_tokens", {}).items()
              if not k.startswith("_") and isinstance(v, dict)}
    return models, prices, data.get("bedrock_region", "eu-west-1")


def _approved(a):
    if a.no_tiers:
        return None
    path = Path(a.approved)
    return json.loads(path.read_text()) if path.exists() else None


def cmd_run(a):
    config = json.loads(Path(a.config).read_text())
    chunks = load_kb(a.kb)
    scenarios = load_scenarios(a.scenarios)
    validate_scenarios(scenarios, personas=set(config["personas"]))
    if a.limit:
        scenarios = scenarios[: a.limit]
    prices = {}
    if a.provider == "mock":
        llm = MockLLM()
    else:
        models, prices, region = _models(a.models, a.provider)
        if a.provider == "anthropic":
            llm = AnthropicLLM(models)
        elif a.provider == "bedrock":
            llm = BedrockLLM(models, region=os.environ.get("AWS_REGION", region))
        else:
            if not os.environ.get("NEBIUS_API_KEY"):
                sys.exit("Set NEBIUS_API_KEY to use the nebius provider")
            llm = OpenAICompatLLM(models)
    run_dir = run_suite(config, chunks, scenarios, llm, langs=a.langs.split(","),
                        max_turns=a.max_turns, out_dir=a.out, name=a.name, prices=prices,
                        workers=a.workers, retrieval=a.retrieval, provider=a.provider,
                        guard=not a.no_guard, approved=_approved(a))
    report = build_report(run_dir)
    meta = json.loads((run_dir / "meta.json").read_text())
    rows = load_rows(run_dir)
    passed = sum(r["verdict"]["passed"] for r in rows)
    print(f"{run_dir.name}: {passed}/{len(rows)} sessions passed · cost ${meta['cost']['total_usd']:.2f}"
          + (f" · unpriced: {meta['cost']['unpriced_models']}" if meta["cost"]["unpriced_models"] else ""))
    print(f"report: {report}")
    if a.compare_to:
        cmd_diff(argparse.Namespace(a=a.compare_to, b=str(run_dir)))
    if a.publish_bucket:
        from deploykit.publish import publish_run
        keys = publish_run(run_dir, a.publish_bucket)
        print(f"published {len(keys)} files to s3://{a.publish_bucket}")
    return 0


def cmd_models(a):
    if not os.environ.get("NEBIUS_API_KEY"):
        sys.exit("Set NEBIUS_API_KEY")
    import urllib.request
    from deploykit.llm import NEBIUS_BASE_URL
    req = urllib.request.Request(NEBIUS_BASE_URL + "models?verbose=true",
                                 headers={"Authorization": f"Bearer {os.environ['NEBIUS_API_KEY']}"})
    data = json.loads(urllib.request.urlopen(req, timeout=30).read())["data"]
    for m in sorted(data, key=lambda x: x["id"]):
        p = m.get("pricing") or {}
        print(f"{m['id']}\tin={p.get('prompt')}\tout={p.get('completion')}")
    return 0


def cmd_show(a):
    """Print every conversation in plain text (for CI logs)."""
    meta = json.loads((Path(a.run) / "meta.json").read_text())
    print(f"run {meta['name']} · cost {json.dumps(meta['cost'])}")
    for r in load_rows(a.run):
        v = r["verdict"]
        print(f"\n== {r['scenario_id']} [{r['category']}/{r['persona']}] "
              f"{'PASS' if v['passed'] else 'FAIL'} {', '.join(v['reasons'])}")
        for t in r["turns"]:
            ag, j = t["agent"], t["judge"]
            print(f"  BUYER: {t['user']}  [tier {t['agent'].get('tier')}]")
            print(f"  AGENT: {ag['reply'][:a.width]}")
            print(f"    cites={ag['citations']} handoff={ag['handoff']} unknown={ag['unknown']} "
                  f"parse_error={ag['parse_error']} invalid={ag['invalid_citations']}")
            g = ag.get("guard") or {}
            if g.get("retried"):
                print(f"    GUARD: blocked {[p['type'] + ':' + p['detail'] for p in g['first_problems']]}"
                      f" -> {'fallback handoff' if g['fell_back'] else 'fixed on retry'}")
                print(f"    BLOCKED DRAFT: {str(g.get('blocked_draft'))[:a.width]}")
            print(f"  JUDGE: grounded={j.get('grounded')} complete={j.get('completeness')} "
                  f"handoff_ok={j.get('handoff_ok')} gap={j.get('gap')} claims={j.get('unsupported_claims')} "
                  f"notes={str(j.get('notes'))[:a.width]}")
    return 0


def cmd_report(a):
    print(build_report(a.run))
    return 0


def cmd_diff(a):
    d, out = write_diff_md(a.a, a.b)
    print(f"{d['run_a']} → {d['run_b']}: pass rate {d['pass_rate_a']:.0%} → {d['pass_rate_b']:.0%}, "
          f"fixed {len(d['fixed'])}, broke {len(d['broke'])}")
    print(f"diff: {out}")
    return 0


def cmd_sample(a):
    rows = load_rows(a.run)
    turns = [(r, i, t) for r in rows for i, t in enumerate(r["turns"])]
    random.Random(a.seed).shuffle(turns)
    path = Path(a.run) / "labels.csv"
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["scenario", "lang", "turn", "question", "reply", "citations",
                    "judge_grounded", "human_grounded"])
        for r, i, t in turns[: a.n]:
            w.writerow([r["scenario_id"], r["lang"], i, t["user"], t["agent"]["reply"],
                        " ".join(t["agent"]["citations"]),
                        str(bool(t["judge"].get("grounded"))).lower(), ""])
    print(f"label the human_grounded column (true/false): {path}")
    return 0


def cmd_agreement(a):
    path = Path(a.run) / "labels.csv"
    rows = [r for r in csv.DictReader(path.open(encoding="utf-8"))
            if r["human_grounded"].strip().lower() in ("true", "false")]
    same = sum(r["human_grounded"].strip().lower() == r["judge_grounded"] for r in rows)
    result = {"labelled": len(rows), "agree": same,
              "agreement": round(same / len(rows), 3) if rows else None}
    (Path(a.run) / "agreement.json").write_text(json.dumps(result, indent=2))
    print(f"judge agrees with you on {same}/{len(rows)} labelled turns")
    return 0


def build_parser():
    p = argparse.ArgumentParser(prog="deploykit", description="Pre-launch simulation for a demo agent")
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="run the suite")
    r.add_argument("--provider", choices=["mock", "anthropic", "bedrock", "nebius"], default="mock")
    r.add_argument("--name")
    r.add_argument("--langs", default="en")
    r.add_argument("--max-turns", type=int, default=3)
    r.add_argument("--limit", type=int)
    r.add_argument("--workers", type=int, default=4)
    r.add_argument("--retrieval", choices=["full", "bm25"], default="full")
    r.add_argument("--kb", default=str(ROOT / "kb"))
    r.add_argument("--scenarios", default=str(ROOT / "suite" / "scenarios.json"))
    r.add_argument("--config", default=str(ROOT / "config" / "client.json"))
    r.add_argument("--models", default=str(ROOT / "config" / "models.json"))
    r.add_argument("--out", default=str(ROOT / "runs"))
    r.add_argument("--compare-to", help="earlier run dir to diff against")
    r.add_argument("--no-guard", action="store_true", help="send drafts without the pre-send guard")
    r.add_argument("--no-tiers", action="store_true", help="one strictness for every reply (no risk tiers)")
    r.add_argument("--approved", default=str(ROOT / "config" / "approved_answers.json"))
    r.add_argument("--publish-bucket", help="S3 bucket to upload the run to")
    r.set_defaults(fn=cmd_run)

    ml = sub.add_parser("models", help="list models on Nebius Token Factory"); ml.set_defaults(fn=cmd_models)
    sh = sub.add_parser("show", help="print every conversation"); sh.add_argument("run")
    sh.add_argument("--width", type=int, default=600); sh.set_defaults(fn=cmd_show)
    s = sub.add_parser("report"); s.add_argument("run"); s.set_defaults(fn=cmd_report)
    d = sub.add_parser("diff"); d.add_argument("a"); d.add_argument("b"); d.set_defaults(fn=cmd_diff)
    sm = sub.add_parser("sample", help="pick turns for you to label by hand")
    sm.add_argument("run"); sm.add_argument("--n", type=int, default=15); sm.add_argument("--seed", type=int, default=7)
    sm.set_defaults(fn=cmd_sample)
    ag = sub.add_parser("agreement", help="judge vs your labels"); ag.add_argument("run")
    ag.set_defaults(fn=cmd_agreement)
    return p


def main(argv=None) -> int:
    a = build_parser().parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    raise SystemExit(main())
