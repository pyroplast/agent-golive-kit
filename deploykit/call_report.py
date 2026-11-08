"""HTML report for a call run: one timeline row per call, one mark per turn."""
from __future__ import annotations

import json
from html import escape
from pathlib import Path

from deploykit.calls import load_call_rows, summarize_calls
from deploykit.report import CSS, FONTS, _bar, _label, _section, _table

# One mark per turn. Shape and letter carry the state too, so it never depends on colour alone.
STATE = {
    "pass": ("#2a78d6", "", "Passed"),
    "fixed": ("#7fb0ea", "G", "Passed after the guard blocked a draft"),
    "parked": ("#ffffff", "?", "Question added to the list"),
    "handoff": ("#0f2124", "H", "Handed over to a person"),
    "over_cautious": ("#eb6834", "C", "Failed: held back when it could answer"),
    "unsafe": ("#b3261e", "!", "Failed: said something it shouldn't"),
}

CALL_CSS = """
.tl{display:grid;grid-template-columns:170px repeat(15,1fr);gap:6px;align-items:center;margin:0 0 10px}
.tl .who{font-size:13px}.tl .who b{display:block;font-weight:600}
.mk{height:34px;border:2px solid #0f2124;border-radius:6px;display:flex;align-items:center;justify-content:center;
  font:600 13px 'Geist Mono',monospace;color:#fff}
.mk.parked{color:#0f2124;border-style:dashed}.mk.none{border:1px dashed var(--line)}
.tl.head div{font-size:11px;color:var(--grey);text-align:center;letter-spacing:.06em}
.legend{display:flex;flex-wrap:wrap;gap:8px 18px;font-size:13px;color:var(--grey);margin:14px 0 0}
.legend span{display:flex;align-items:center;gap:6px}.legend i{display:inline-flex;width:22px;height:18px;border:2px solid #0f2124;
  border-radius:4px;font:600 11px 'Geist Mono',monospace;color:#fff;align-items:center;justify-content:center;font-style:normal}
"""


def turn_state(t: dict) -> str:
    v, a = t["verdict"], t["agent"]
    if not v["passed"]:
        return v.get("failure_kind", "unsafe")
    if a.get("handoff"):
        return "handoff"
    if a.get("park"):
        return "parked"
    g = a.get("guard") or {}
    return "fixed" if g.get("retried") and not g.get("fell_back") else "pass"


def _mark(t):
    colour, letter, label = STATE[turn_state(t)]
    cls = "mk parked" if turn_state(t) == "parked" else "mk"
    tip = f"Turn {t['n']}: {label}. Buyer: {t['user'][:140]}"
    return f"<div class='{cls}' style='background:{colour}' title='{escape(tip)}'>{letter}</div>"


def build_call_report(run_dir) -> Path:
    run_dir = Path(run_dir)
    rows = load_call_rows(run_dir)
    meta = json.loads((run_dir / "meta.json").read_text(encoding="utf-8"))
    s = summarize_calls(rows)

    tl = ["<div class='tl head'><div></div>" + "".join(f"<div>{n}</div>" for n in range(1, 16)) + "</div>"]
    for r in rows:
        marks = [_mark(t) for t in r["turns"]] + ["<div class='mk none'></div>"] * (15 - len(r["turns"]))
        p = r.get("profile", {})
        tl.append(f"<div class='tl'><div class='who'><b class='mono'>{escape(r['call_id'])}</b>"
                  f"{escape(p.get('role', ''))}, {escape(p.get('company', ''))}</div>{''.join(marks)}</div>")
    legend = "<div class='legend'>" + "".join(
        f"<span><i style='background:{c};{'color:#0f2124;border-style:dashed' if k == 'parked' else ''}'>{l}</i>"
        f"{escape(lbl)}</span>" for k, (c, l, lbl) in STATE.items()) + "</div>"

    stages = _table(["Turns", "Replies", "Passed", "Pass rate"],
                    [[k, v["turns"], v["passed"], _bar(v["rate"] or 0)] for k, v in s["by_stage"].items()],
                    numeric=(1, 2))
    calls_tbl = _table(["Call", "Turns", "Passed turns", "First failure", "Ended", "Cost"],
                       [[f"<span class='mono'>{escape(r['call_id'])}</span>", len(r["turns"]),
                         sum(t["verdict"]["passed"] for t in r["turns"]),
                         f"turn {r['verdict']['first_failure_turn']}" if r["verdict"].get("first_failure_turn") else "none",
                         escape(r.get("ended", "")), f"${r.get('cost_usd', 0):.2f}"] for r in rows], numeric=(1, 2))
    contra = [(r["call_id"], c) for r in rows for c in (r.get("call_judge") or {}).get("contradictions", [])]
    contra_html = (_table(["Call", "Turns", "What contradicted"],
                          [[f"<span class='mono'>{escape(cid)}</span>", escape(", ".join(map(str, c.get("turns", [])))),
                            escape(str(c.get("detail", "")))] for cid, c in contra])
                   if contra else "<p class='empty'>No contradictions found across turns.</p>")
    reasons = (_table(["Reason", "Turns"], [[_label(k), v] for k, v in s["turn_reasons"].items()], numeric=(1,))
               if s["turn_reasons"] else "<p class='empty'>No failed turns.</p>")

    transcripts = []
    for r in rows:
        v = r["verdict"]
        chip = "<span class='chip pass'>pass</span>" if v["passed"] else "<span class='chip fail'>fail</span>"
        transcripts.append(f"<details><summary>{chip}<span class='mono'>{escape(r['call_id'])}</span>"
                           f"<span class='hide-sm'>{_label(r['persona'])}</span><span class='hide-sm'>"
                           f"{len(r['turns'])} turns</span><span class='why'>{escape(', '.join(v['reasons']))}</span></summary>")
        for t in r["turns"]:
            a, tv = t["agent"], t["verdict"]
            facts = (f"turn {t['n']} · {STATE[turn_state(t)][2]} · tier {a.get('tier') or '-'} · cites "
                     f"{', '.join(a.get('citations', [])) or 'nothing'} · {t.get('latency_s', 0):.1f}s"
                     + (f" · {', '.join(tv['reasons'])}" if tv["reasons"] else ""))
            transcripts.append(f"<div class='turn'><div class='who'>Buyer</div><div>{escape(t['user'])}</div>"
                               f"<div class='who'>Agent</div><div>{escape(a['reply'])}</div>"
                               f"<div class='facts mono'>{escape(facts)}</div></div>")
        transcripts.append("</details>")

    usd = meta.get("cost", {}).get("total_usd", 0)
    html = "".join([
        f"<!doctype html><html lang='en'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>"
        f"<title>Demo-call readiness · {escape(meta['name'])}</title>{FONTS}<style>{CSS}{CALL_CSS}</style></head><body>",
        "<header><div class='wrap'><span class='eyebrow'>Simulated demo calls</span>",
        "<h1 class='serif'>Demo-call readiness report</h1>",
        f"<p class='sub'>{s['calls']} simulated demo calls with HiBob's sales agent, up to 15 turns each. "
        "The buyer asks one question at a time and the agent keeps the whole call in mind.</p>",
        f"<div class='meta mono'><span>run {escape(meta['name'])}</span><span>{escape(meta.get('created', ''))}</span>"
        f"<span>provider {escape(meta.get('provider', ''))}</span><span>config {escape(meta.get('config_fingerprint', ''))}</span>"
        "</div></div></header><main class='wrap'><div class='kpis'>",
        f"<div class='kpi hero'><div class='label'>Turns passed</div><div class='value serif'>{s['turn_pass_rate']:.0%}</div>"
        f"<div class='note'>{s['turns_passed']} of {s['turns']} replies</div></div>",
        f"<div class='kpi'><div class='label'>Clean calls</div><div class='value serif'>{s['calls_passed']}/{s['calls']}</div>"
        f"<div class='note'>calls with no failed turn</div></div>",
        f"<div class='kpi'><div class='label'>Release gate</div><div class='value serif'>{'Pass' if s['gate']['passed'] else 'Blocked'}</div>"
        f"<div class='note'>{s['gate']['unsafe_high']} unsafe high-risk turns, {s['gate']['contradictions']} contradictions</div></div>",
        f"<div class='kpi'><div class='label'>Reply time</div><div class='value serif'>{s['latency']['p50']:.1f}s</div>"
        f"<div class='note'>typical; slowest {s['latency']['max']:.1f}s</div></div>",
        f"<div class='kpi'><div class='label'>Run cost</div><div class='value serif'>${usd:.2f}</div>"
        f"<div class='note'>model usage, cached input priced</div></div></div>",
        _section("01", "Call timeline", "Each row is one call, each mark one turn. Hover a mark to see the question.",
                 "".join(tl) + legend),
        _section("02", "Early, middle and late in the call", "Does the agent get worse as the call goes on?", stages),
        _section("03", "Calls", "Where each call first went wrong.", calls_tbl),
        _section("04", "Consistency across turns", "Found by a judge that reads the whole call at once.", contra_html),
        _section("05", "Why turns failed", "", reasons),
        _section("06", "Transcripts", "Open a call to read it turn by turn.", "".join(transcripts)),
        "</main><footer><div class='wrap'>Built from public hibob.com pages. Not affiliated with HiBob.</div></footer></body></html>",
    ])
    out = run_dir / "report.html"
    out.write_text(html, encoding="utf-8")
    return out
