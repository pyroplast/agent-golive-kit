"""Summaries, the HTML go-live report, and run-to-run diffs."""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from html import escape
from pathlib import Path


def load_rows(run_dir) -> list[dict]:
    text = (Path(run_dir) / "results.jsonl").read_text(encoding="utf-8")
    return [json.loads(l) for l in text.splitlines() if l.strip()]


def _rate(rows):
    return round(sum(r["verdict"]["passed"] for r in rows) / len(rows), 3) if rows else 0.0


def _group(rows, key):
    g = defaultdict(list)
    for r in rows:
        g[r[key]].append(r)
    return {k: {"sessions": len(v), "passed": sum(x["verdict"]["passed"] for x in v),
                "pass_rate": _rate(v)} for k, v in sorted(g.items())}


def summarize(rows: list[dict]) -> dict:
    turns = [(r, t) for r in rows for t in r["turns"]]
    gaps, risky = [], []
    for r, t in turns:
        if t["judge"].get("gap") or t["agent"].get("unknown"):
            gaps.append({"scenario": r["scenario_id"], "lang": r["lang"], "question": t["user"],
                         "gap": t["judge"].get("gap") or "agent said sources don't cover it"})
        if not t["judge"].get("grounded", False):
            risky.append({"scenario": r["scenario_id"], "lang": r["lang"], "question": t["user"],
                          "reply": t["agent"]["reply"],
                          "claims": t["judge"].get("unsupported_claims", [])})
    guarded = [t["agent"].get("guard") for _, t in turns if (t["agent"].get("guard") or {}).get("enabled")]
    caught = Counter(kind for g in guarded for kind in {p["type"] for p in g.get("first_problems", [])})
    guard = {"turns_checked": len(guarded), "retries": sum(bool(g.get("retried")) for g in guarded),
             "fallbacks": sum(bool(g.get("fell_back")) for g in guarded), "caught": dict(caught)}
    parked = [{"scenario": r["scenario_id"], "question": q} for r in rows for q in r.get("parked", [])]
    human = sum(any(t["agent"].get("handoff_reason") == "buyer asked for a person" for t in r["turns"]) for r in rows)
    return {
        "parked": parked,
        "parked_total": len(parked),
        "human_requests": human,
        "guard": guard,
        "sessions": len(rows),
        "passed": sum(r["verdict"]["passed"] for r in rows),
        "pass_rate": _rate(rows),
        "turns": len(turns),
        "grounded_rate": round(sum(t["judge"].get("grounded", False) for _, t in turns) / len(turns), 3) if turns else 0.0,
        "handoff_sessions": sum(any(t["agent"]["handoff"] for t in r["turns"]) for r in rows),
        "by_category": _group(rows, "category"),
        "by_persona": _group(rows, "persona"),
        "by_lang": _group(rows, "lang"),
        "reasons": dict(Counter(x for r in rows for x in r["verdict"]["reasons"]).most_common()),
        "gaps": gaps,
        "risky": risky,
    }


def diff_runs(a_dir, b_dir) -> dict:
    a = {f'{r["scenario_id"]}:{r["lang"]}': r["verdict"]["passed"] for r in load_rows(a_dir)}
    b = {f'{r["scenario_id"]}:{r["lang"]}': r["verdict"]["passed"] for r in load_rows(b_dir)}
    common = sorted(set(a) & set(b))
    return {"run_a": Path(a_dir).name, "run_b": Path(b_dir).name,
            "pass_rate_a": round(sum(a[k] for k in common) / len(common), 3) if common else 0.0,
            "pass_rate_b": round(sum(b[k] for k in common) / len(common), 3) if common else 0.0,
            "fixed": [k for k in common if not a[k] and b[k]],
            "broke": [k for k in common if a[k] and not b[k]],
            "only_in_a": sorted(set(a) - set(b)), "only_in_b": sorted(set(b) - set(a))}


def write_diff_md(a_dir, b_dir) -> tuple[dict, Path]:
    d = diff_runs(a_dir, b_dir)
    lines = [f"# Regression: {d['run_a']} → {d['run_b']}", "",
             f"Pass rate: {d['pass_rate_a']:.0%} → {d['pass_rate_b']:.0%}", "",
             f"## Fixed ({len(d['fixed'])})", *[f"- {k}" for k in d["fixed"]], "",
             f"## Broke ({len(d['broke'])})", *[f"- {k}" for k in d["broke"]]]
    out = Path(b_dir) / f"diff_vs_{d['run_a']}.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return d, out


FONTS = ("<link rel='preconnect' href='https://fonts.googleapis.com'>"
         "<link href='https://fonts.googleapis.com/css2?family=Geist:wght@400;500;600&family=Geist+Mono&"
         "family=Newsreader:opsz,wght@6..72,300;6..72,400&display=swap' rel='stylesheet'>")

CSS = """
:root{--ink:#0f2124;--grey:#667275;--line:#dcdcdc;--lime:#d7ffa4;--soft:#f6f7f5;--fail:#f6cfc4}
*{box-sizing:border-box}
body{margin:0;font:15px/1.55 Geist,'Geist Fallback',system-ui,sans-serif;color:var(--ink);background:#fff}
.serif{font-family:Newsreader,Georgia,serif;font-weight:300;letter-spacing:-.03em}
.mono{font-family:'Geist Mono',ui-monospace,monospace;font-size:12.5px}
header{background:var(--ink);color:#fff;padding:56px 0 48px}
.wrap{max-width:1080px;margin:0 auto;padding:0 32px}
.eyebrow{display:inline-block;background:var(--lime);color:var(--ink);font-size:11px;font-weight:600;
  letter-spacing:.12em;text-transform:uppercase;padding:4px 8px;margin-bottom:22px}
header h1{font-size:56px;line-height:1.02;margin:0 0 14px}
header .sub{color:rgba(255,255,255,.72);font-size:17px;margin:0 0 26px;max-width:640px}
header .meta{color:rgba(255,255,255,.5);display:flex;flex-wrap:wrap;gap:6px 22px}
.kpis{display:grid;grid-template-columns:repeat(4,1fr);border:1px solid var(--line);border-top:0;margin-bottom:64px}
.kpi{padding:26px 24px 22px;border-right:1px solid var(--line)}.kpi:last-child{border-right:0}
.kpi .label{font-size:11px;font-weight:600;letter-spacing:.12em;text-transform:uppercase;color:var(--grey)}
.kpi .value{font-size:52px;line-height:1.05;margin:10px 0 4px}
.kpi .note{color:var(--grey);font-size:13px}
.kpi.hero{background:var(--lime)}.kpi.hero .label,.kpi.hero .note{color:var(--ink)}
section{margin:0 0 72px}
.sec-head{display:grid;grid-template-columns:120px 1fr;gap:24px;border-top:1px solid var(--ink);padding-top:18px;margin-bottom:22px}
.sec-head .num{font-size:12px;letter-spacing:.12em;color:var(--grey)}
.sec-head h2{font-size:38px;line-height:1.05;margin:0}
.sec-head p{color:var(--grey);margin:8px 0 0;max-width:620px}
.body{margin-left:144px}
h3{font-size:12px;font-weight:600;letter-spacing:.12em;text-transform:uppercase;color:var(--grey);margin:28px 0 10px}
table{border-collapse:collapse;width:100%;border-top:1px solid var(--line)}
th,td{border-bottom:1px solid var(--line);padding:11px 12px 11px 0;text-align:left;vertical-align:top}
th{font-size:11px;font-weight:600;letter-spacing:.1em;text-transform:uppercase;color:var(--grey)}
td.n{font-variant-numeric:tabular-nums;width:90px}
.bar{display:flex;align-items:center;gap:12px;min-width:220px}
.track{flex:1;height:10px;background:var(--soft);border:1px solid var(--line)}
.fill{height:100%;background:var(--ink)}
.chip{display:inline-block;font-size:11px;font-weight:600;letter-spacing:.08em;padding:3px 7px;text-transform:uppercase}
.chip.pass{background:var(--lime)}.chip.fail{background:var(--fail)}
.tag{display:inline-block;border:1px solid var(--line);font-size:12px;padding:1px 6px;margin:0 4px 4px 0;color:var(--grey)}
details{border-bottom:1px solid var(--line)}details:first-of-type{border-top:1px solid var(--line)}
summary{cursor:pointer;list-style:none;display:grid;grid-template-columns:70px 1.4fr 1fr 1fr 2fr;gap:12px;padding:12px 0;align-items:center}
summary::-webkit-details-marker{display:none}
summary .why{color:var(--grey);font-size:13px}
.turn{display:grid;grid-template-columns:70px 1fr;gap:4px 12px;padding:4px 0 14px 82px;font-size:14px}
.turn .who{font-size:11px;font-weight:600;letter-spacing:.1em;text-transform:uppercase;color:var(--grey);padding-top:2px}
.turn .facts{grid-column:2;color:var(--grey);font-size:12.5px}
.empty{color:var(--grey);font-style:italic}
footer{border-top:1px solid var(--line);color:var(--grey);font-size:13px;padding:24px 0 48px}
@media(max-width:800px){.kpis{grid-template-columns:1fr 1fr}.body{margin-left:0}.sec-head{grid-template-columns:1fr}
  summary{grid-template-columns:70px 1fr}summary .hide-sm{display:none}.turn{padding-left:0}header h1{font-size:40px}}
"""


def _table(headers, rows, numeric=()):
    head = "".join(f"<th>{escape(h)}</th>" for h in headers)
    body = "".join("<tr>" + "".join(
        f"<td class='n'>{c}</td>" if i in numeric else f"<td>{c}</td>" for i, c in enumerate(row)) + "</tr>"
        for row in rows)
    return f"<table><tr>{head}</tr>{body}</table>"


def _bar(rate):
    return (f"<div class='bar'><div class='track'><div class='fill' style='width:{rate:.0%}'></div></div>"
            f"<span class='mono'>{rate:.0%}</span></div>")


LABELS = {"hr_lead": "HR lead", "cfo": "CFO", "it_admin": "IT admin",
          "expected": "Expected", "ambiguous": "Ambiguous", "adversarial": "Adversarial",
          "ungrounded": "Not backed by sources", "missed_handoff": "Missed handoff",
          "unnecessary_handoff": "Unnecessary handoff", "wrong_language": "Wrong language",
          "parse_error": "Malformed agent output", "invalid_citation": "Cited a source that doesn't exist",
          "judge_error": "Judge output unreadable", "dead_end": "Said it didn't know and offered nothing",
          "missed_park": "Didn't offer to add the question to the list", "number_not_in_source": "Number not in a quoted source",
          "watch_word": "Risky wording not in a quote", "quote_not_found": "Quote not found in its source",
          "unknown_source": "Cited a source that doesn't exist", "empty_reply": "Empty reply", "uncited": "Facts without a cited source", "bad_handoff": "Questionable handoff"}


def _label(k):
    return escape(LABELS.get(k, str(k)))


def _group_table(title, groups):
    return f"<h3>{escape(title)}</h3>" + _table(
        ["Group", "Sessions", "Passed", "Pass rate"],
        [[_label(k), v["sessions"], v["passed"], _bar(v["pass_rate"])] for k, v in groups.items()],
        numeric=(1, 2))


def _section(num, title, intro, body):
    intro_html = f"<p>{escape(intro)}</p>" if intro else ""
    return (f"<section><div class='sec-head'><div class='num mono'>{num}</div>"
            f"<div><h2 class='serif'>{escape(title)}</h2>{intro_html}</div></div>"
            f"<div class='body'>{body}</div></section>")


def build_report(run_dir) -> Path:
    run_dir = Path(run_dir)
    rows = load_rows(run_dir)
    meta = json.loads((run_dir / "meta.json").read_text(encoding="utf-8"))
    s = summarize(rows)
    cost = meta.get("cost", {})
    usd = cost.get("total_usd", 0)
    cost_value = f"${usd:.2f}"
    cost_note = "some models unpriced" if cost.get("unpriced_models") else "model usage this run"
    multi_lang = len(s["by_lang"]) > 1

    groups = _group_table("By scenario type", s["by_category"]) + _group_table("By buyer persona", s["by_persona"])
    if multi_lang:
        groups += _group_table("By language", s["by_lang"])

    reasons = _table(["Reason", "Sessions"], [[_label(k), v] for k, v in s["reasons"].items()], numeric=(1,)) \
        if s["reasons"] else "<p class='empty'>No failures.</p>"
    gaps = _table(["Scenario", "Buyer asked", "Missing from sources"],
                  [[f"<span class='mono'>{escape(g['scenario'])}</span>", escape(g["question"]), escape(g["gap"])]
                   for g in s["gaps"]]) if s["gaps"] else "<p class='empty'>None found.</p>"
    risky = _table(["Scenario", "Buyer asked", "Agent said", "Unsupported claims"],
                   [[f"<span class='mono'>{escape(x['scenario'])}</span>", escape(x["question"]), escape(x["reply"]),
                     "".join(f"<span class='tag'>{escape(c)}</span>" for c in x["claims"]) or "—"]
                    for x in s["risky"]]) if s["risky"] else "<p class='empty'>None found.</p>"

    sessions = []
    for r in rows:
        ok = r["verdict"]["passed"]
        chip = "<span class='chip pass'>Pass</span>" if ok else "<span class='chip fail'>Fail</span>"
        why = ", ".join(LABELS.get(x, x) for x in r["verdict"]["reasons"])
        sessions.append(f"<details><summary>{chip}<span class='mono'>{escape(r['scenario_id'])}</span>"
                        f"<span class='hide-sm'>{_label(r['category'])}</span><span class='hide-sm'>{_label(r['persona'])}</span>"
                        f"<span class='why'>{escape(why)}</span></summary>")
        for t in r["turns"]:
            a, j = t["agent"], t["judge"]
            facts = (f"cites {escape(', '.join(a['citations']) or 'nothing')} · handoff {'yes' if a['handoff'] else 'no'}"
                     f" · grounded {'yes' if j.get('grounded') else 'no'}"
                     + (f" · {escape(str(j.get('notes')))}" if j.get("notes") else ""))
            sessions.append(f"<div class='turn'><div class='who'>Buyer</div><div>{escape(t['user'])}</div>"
                            f"<div class='who'>Agent</div><div>{escape(a['reply'])}</div>"
                            f"<div class='facts mono'>{facts}</div></div>")
        sessions.append("</details>")

    parts = [
        f"<!doctype html><html lang='en'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>"
        f"<title>Go-live readiness · {escape(meta['name'])}</title>{FONTS}<style>{CSS}</style></head><body>",
        "<header><div class='wrap'><span class='eyebrow'>Pre-launch simulation</span>",
        "<h1 class='serif'>Go-live readiness report</h1>",
        "<p class='sub'>Bob demo agent for HiBob, tested against simulated buyers before launch. "
        "Every answer is checked against HiBob's public documentation.</p>",
        f"<div class='meta mono'><span>run {escape(meta['name'])}</span><span>{escape(meta.get('created', ''))}</span>"
        f"<span>provider {escape(meta.get('provider', ''))}</span><span>{meta.get('kb_chunks', '?')} source chunks</span>"
        f"<span>config {escape(meta.get('config_fingerprint', ''))}</span></div></div></header>",
        "<main class='wrap'><div class='kpis'>",
        f"<div class='kpi hero'><div class='label'>Pass rate</div><div class='value serif'>{s['pass_rate']:.0%}</div>"
        f"<div class='note'>{s['passed']} of {s['sessions']} sessions</div></div>",
        f"<div class='kpi'><div class='label'>Grounded answers</div><div class='value serif'>{s['grounded_rate']:.0%}</div>"
        f"<div class='note'>across {s['turns']} turns</div></div>",
        f"<div class='kpi'><div class='label'>Handoffs</div><div class='value serif'>{s['handoff_sessions']}</div>"
        f"<div class='note'>sessions passed to a human</div></div>",
        f"<div class='kpi'><div class='label'>Run cost</div><div class='value serif'>{cost_value}</div>"
        f"<div class='note'>{escape(cost_note)}</div></div></div>",
        _section("01", "Results by group", "Where the agent is ready and where it isn't.", groups),
        _section("01b", "Pre-send guard", "Drafts blocked before the buyer saw them, by deterministic checks (no model tokens).",
                 (f"<p>{s['guard']['turns_checked']} replies checked · {s['guard']['retries']} blocked and retried · "
                  f"{s['guard']['fallbacks']} replaced by a safe handoff.</p>"
                  + (_table(["Caught", "Replies"], [[_label(k), v] for k, v in s['guard']['caught'].items()], numeric=(1,))
                     if s['guard']['caught'] else "<p class='empty'>Nothing blocked.</p>"))
                 if s["guard"]["turns_checked"] else "<p class='empty'>Guard was off for this run.</p>"),
        _section("01c", "Questions for the HiBob team",
                 f"Questions the agent could not answer and added to the list instead of ending the conversation. "
                 f"{s['human_requests']} buyer(s) asked for a person and were handed over at once.",
                 _table(["Scenario", "Question"], [[f"<span class='mono'>{escape(p['scenario'])}</span>", escape(p["question"])]
                                                    for p in s["parked"]]) if s["parked"] else "<p class='empty'>None.</p>"),
        _section("02", "Why sessions failed", "Each failed session can have several reasons.", reasons),
        _section("03", "Knowledge gaps", "Questions the sources could not answer. Each one is a doc to write or a handoff rule to add.", gaps),
        _section("04", "Risky answers", "Replies the judge could not back with the sources. Fix these before launch.", risky),
        _section("05", "All sessions", "Open a session to read the full conversation.", "".join(sessions)),
        "</main><footer><div class='wrap'>Built from public hibob.com pages. Not affiliated with HiBob.</div></footer></body></html>",
    ]
    out = run_dir / "report.html"
    out.write_text("".join(parts), encoding="utf-8")
    return out
