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
from deploykit.llm import CostTracker, TrackedLLM, AnthropicLLM, BedrockLLM, MockLLM, OpenAICompatLLM
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


def _llm(a):
    if a.provider == "mock":
        return MockLLM(), {}
    models, prices, region = _models(a.models, a.provider)
    if a.provider == "anthropic":
        return AnthropicLLM(models), prices
    if a.provider == "bedrock":
        return BedrockLLM(models, region=os.environ.get("AWS_REGION", region)), prices
    if not os.environ.get("NEBIUS_API_KEY"):
        sys.exit("Set NEBIUS_API_KEY to use the nebius provider")
    return OpenAICompatLLM(models), prices


def cmd_calls(a):
    from deploykit.calls import load_calls, run_calls, summarize_calls, load_call_rows, validate_calls
    config = json.loads(Path(a.config).read_text())
    calls = load_calls(a.calls)
    validate_calls(calls, personas=set(config["personas"]))
    if a.limit:
        calls = calls[: a.limit]
    llm, prices = _llm(a)
    run_dir = run_calls(config, load_kb(a.kb), calls, llm, out_dir=a.out, name=a.name, prices=prices,
                        workers=a.workers, retrieval=a.retrieval, provider=a.provider,
                        guard=not a.no_guard, approved=_approved(a), live=a.live, fast=a.fast)
    report = build_report(run_dir)
    from deploykit.gate import load_reviews
    s = summarize_calls(load_call_rows(run_dir), reviews=load_reviews(run_dir.name, ROOT / "reviews"))
    stages = " · ".join(f"turns {k}: {v['rate']:.0%}" for k, v in s["by_stage"].items() if v["rate"] is not None)
    print(f"{run_dir.name}: {s['calls']} calls · {s['turns_passed']}/{s['turns']} turns passed · "
          f"{s['calls_passed']} clean calls · gate {'PASS' if s['gate']['passed'] else 'BLOCKED'} · "
          f"cost ${s['cost_usd']:.2f}")
    L = s["latency"]
    print(f"  {stages} · reply time p50 {L['p50']}s, p90 {L['p90']}s, max {L['max']}s · first words p50 "
          f"{L['first_words_p50']}s, p90 {L['first_words_p90']}s · filler {L['filler_turns']} · fast model "
          f"{L['fast_model_turns']} turns")
    for f in s["gate"]["findings"]:
        print(f"  GATE FINDING {f['id']} [{f['kind']}] {f['detail'][:300]}")
    if s["gate"]["over_cautious_contradictions"]:
        print(f"  over-cautious inconsistencies (answered in one turn, declined in another): {s['gate']['over_cautious_contradictions']}")
    print(f"report: {report}")
    return 0


def _chat_session(a):
    from deploykit.chat import ChatSession
    config = json.loads(Path(a.config).read_text())
    llm, _ = _llm(a)
    approved = json.loads(Path(a.approved).read_text()) if Path(a.approved).exists() else None
    return ChatSession(config, load_kb(a.kb), llm, approved=approved, judge=a.judge)


def cmd_chat(a):
    """Talk to the agent in the terminal. /list shows the question list, /new starts a new call, /quit ends."""
    s = _chat_session(a)
    print("You are the buyer on a demo call with HiBob's sales agent. /list, /new, /judge, /quit")
    while True:
        try:
            text = input("\nYOU: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if text in ("/quit", "/exit"):
            break
        if text == "/new":
            s.reset()
            print("(new call)")
            continue
        if text == "/judge":
            s.judge = not s.judge
            print(f"(judge {'on' if s.judge else 'off'})")
            continue
        if text == "/list":
            print("Question list for the HiBob team:" + ("".join(f"\n  {i}. {q}" for i, q in enumerate(s.parked, 1))
                                                       or " empty"))
            continue
        if not text:
            continue
        print("AGENT: ", end="", flush=True)

        def show(e):
            if e["type"] == "say":
                print(e["text"], end=" ", flush=True)
            elif e["type"] == "typing":
                print("…", end=" ", flush=True)
        t = s.ask(text, on_event=show)
        tm, g = t.get("timing") or {}, t.get("guard") or {}
        how = ("same question, same answer" if t.get("repeat") else f"fixed reply ({t['fixed']})" if t.get("fixed")
               else "wrap-up by code" if t.get("wrap_up") else "slow path" if g.get("slow_path") else
               f"approved answer {g['approved_substitute']}" if g.get("approved_substitute") else
               f"{g.get('skipped')} sentence(s) dropped" if g.get("skipped") else "all sentences passed")
        print(f"\n   [tier {t.get('tier')} · first words {tm.get('first_words_s', 0):.1f}s · done "
              f"{tm.get('total_s', 0):.1f}s · {how} · cites {', '.join(t.get('citations', [])) or 'none'}"
              f"{' · added to list' if t.get('park') else ''}{' · HANDED OFF' if t.get('handoff') else ''}]")
        if t.get("judge"):
            print(f"   [judge: grounded={t['judge'].get('grounded')} {t['judge'].get('unsupported_claims') or ''}]")
        if s.ended:
            print("(The call was handed to a person. /new to start again.)")
    return 0


def cmd_serve(a):
    from deploykit.chat import ChatApp, serve
    serve(ChatApp(_chat_session(a)), port=a.port)
    return 0


def cmd_review(a):
    """A person's decision on a gate-blocking finding, kept in the repo next to the code it judged."""
    folder = Path(a.dir)
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{a.name}.json"
    data = json.loads(path.read_text()) if path.exists() else {}
    for fid in a.reject or []:
        data[fid] = {"decision": "rejected", "note": a.note or ""}
    for fid in a.confirm or []:
        data[fid] = {"decision": "confirmed", "note": a.note or ""}
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
    print(f"{path}: {len(data)} reviewed findings")
    return 0


def cmd_judgecheck(a):
    from deploykit.judgecheck import run_judge_check
    items = json.loads(Path(a.items).read_text())
    chunks = load_kb(a.kb)
    if a.provider == "mock":
        llms = [("mock", MockLLM(), {})]
    else:
        models, prices, _ = _models(a.models, a.provider)
        names = [m.strip() for m in (a.judge_models or models["judge"]).split(",") if m.strip()]
        llms = [(m, OpenAICompatLLM({**models, "judge": m}), prices) for m in names]
    for name, llm, prices in llms:
        tracker = CostTracker(prices)
        r = run_judge_check(TrackedLLM(llm, tracker), chunks, items)
        print(f"{name}: agrees on {r['agree']}/{r['items']} · planted errors caught {r['caught']}/{r['bad']} · "
              f"false alarms {r['false_alarms']}/{r['good']} · cost ${tracker.summary()['total_usd']:.3f}")
        missed = [x["id"] for x in r["rows"] if not x["agree"]]
        if missed:
            print(f"  disagrees on: {', '.join(missed)}")
    return 0


def _show_calls(a):
    from deploykit.calls import load_call_rows
    for r in load_call_rows(a.run):
        v = r["verdict"]
        print(f"\n== CALL {r['call_id']} [{r['persona']}] {'PASS' if v['passed'] else 'FAIL'} "
              f"{', '.join(v['reasons'])} · first failure turn {v.get('first_failure_turn')} · ended {r.get('ended')}")
        for t in r["turns"]:
            ag, j, tv = t["agent"], t["judge"], t["verdict"]
            tm = ag.get("timing") or {}
            live_info = (f", first words {tm['first_words_s']:.1f}s{' +filler' if tm.get('filler') else ''}, "
                         f"{ag.get('model_role')}" if tm else "")
            print(f"  T{t['n']} BUYER: {t['user']}  [tier {ag.get('tier')}, {t.get('latency_s', 0):.1f}s{live_info}]")
            print(f"  T{t['n']} AGENT: {ag['reply'][:a.width]}")
            print(f"    {'ok' if tv['passed'] else 'FAIL ' + ','.join(tv['reasons'])} cites={ag['citations']} "
                  f"park={bool(ag.get('park'))} handoff={ag['handoff']} grounded={j.get('grounded')} "
                  f"claims={j.get('unsupported_claims')}")
            g = ag.get("guard") or {}
            if g.get("slow_path") or g.get("skipped"):
                print(f"    LIVE: {'slow path' if g.get('slow_path') else str(g.get('skipped')) + ' sentence(s) skipped'}: "
                      f"{[p['type'] + ':' + p['detail'] for p in (g.get('stream_problems') or g.get('first_problems') or [])]}")
            if g.get("retried") or g.get("fell_back"):
                print(f"    GUARD: {[p['type'] + ':' + p['detail'] for p in g.get('first_problems', [])]}"
                      f" -> {'fallback' if g['fell_back'] else 'fixed on retry'}")
        cj = r.get("call_judge") or {}
        print(f"  CALL JUDGE: contradictions={cj.get('contradictions')} notes={cj.get('notes')}")
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
    if meta.get("suite") == "calls":
        return _show_calls(a)
    for r in load_rows(a.run):
        v = r["verdict"]
        print(f"\n== {r['scenario_id']} [{r['category']}/{r['persona']}] "
              f"{'PASS' if v['passed'] else 'FAIL'} {', '.join(v['reasons'])}"
              + (f" ({v.get('failure_kind', 'unsafe')})" if not v["passed"] else ""))
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

    c = sub.add_parser("calls", help="run the demo-call suite (whole calls, one question at a time)")
    c.add_argument("--provider", choices=["mock", "anthropic", "bedrock", "nebius"], default="mock")
    c.add_argument("--name")
    c.add_argument("--limit", type=int, help="only the first N calls")
    c.add_argument("--workers", type=int, default=5)
    c.add_argument("--retrieval", choices=["full", "bm25"], default="full")
    c.add_argument("--kb", default=str(ROOT / "kb"))
    c.add_argument("--calls", default=str(ROOT / "suite" / "calls.json"))
    c.add_argument("--config", default=str(ROOT / "config" / "client.json"))
    c.add_argument("--models", default=str(ROOT / "config" / "models.json"))
    c.add_argument("--out", default=str(ROOT / "runs"))
    c.add_argument("--no-guard", action="store_true")
    c.add_argument("--live", action="store_true", help="stream sentence by sentence, fast model, filler line")
    c.add_argument("--fast", action="store_true", help="live mode: easy turns on the small fast model "
                   "(off by default: in calls-live-1 its first sentence failed the checks on 17 of 20 turns)")
    c.add_argument("--no-tiers", action="store_true")
    c.add_argument("--approved", default=str(ROOT / "config" / "approved_answers.json"))
    c.set_defaults(fn=cmd_calls)

    for name, fn, helptext in (("chat", cmd_chat, "talk to the agent in the terminal"),
                               ("serve", cmd_serve, "talk to the agent in a local web page")):
        cp = sub.add_parser(name, help=helptext)
        cp.add_argument("--provider", choices=["mock", "anthropic", "bedrock", "nebius"], default="nebius")
        cp.add_argument("--judge", action="store_true", help="also judge each reply (DeepSeek-V4-Pro, ~$0.002 each)")
        cp.add_argument("--port", type=int, default=8765)
        cp.add_argument("--kb", default=str(ROOT / "kb"))
        cp.add_argument("--config", default=str(ROOT / "config" / "client.json"))
        cp.add_argument("--models", default=str(ROOT / "config" / "models.json"))
        cp.add_argument("--approved", default=str(ROOT / "config" / "approved_answers.json"))
        cp.set_defaults(fn=fn)

    rv = sub.add_parser("review", help="record a person's decision on a gate-blocking finding")
    rv.add_argument("--name", required=True, help="run name")
    rv.add_argument("--reject", action="append", help="finding id the person found to be a false alarm")
    rv.add_argument("--confirm", action="append", help="finding id the person confirmed")
    rv.add_argument("--note")
    rv.add_argument("--dir", default=str(ROOT / "reviews"))
    rv.set_defaults(fn=cmd_review)

    jc = sub.add_parser("judgecheck", help="score judge models on the judge test set")
    jc.add_argument("--provider", choices=["mock", "nebius"], default="mock")
    jc.add_argument("--judge-models", help="comma-separated model ids (default: the configured judge)")
    jc.add_argument("--items", default=str(ROOT / "suite" / "judge_calibration.json"))
    jc.add_argument("--kb", default=str(ROOT / "kb"))
    jc.add_argument("--models", default=str(ROOT / "config" / "models.json"))
    jc.set_defaults(fn=cmd_judgecheck)

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
