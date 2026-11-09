"""Render the live-run history as one self-contained HTML page.

    python scripts/plot_runs.py                      # docs/run_history.json -> docs/run_history.html
    python scripts/plot_runs.py in.json out.html

Three charts, one measure each (no dual axes):
1. Pass rate per run, with the phases of work and the point where the suite grew from 30 to 43 scenarios.
2. Guard activity per run: share of replies blocked, and share replaced by a safe fallback.
3. Latest run by risk tier: pass rate per tier, plus the release gate.
Hover any point or bar for details. A data table follows the charts.
"""
from __future__ import annotations

import json
import sys
from html import escape
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PHASES = [("Prompt fixes", "baseline", "fix-2"), ("Pre-send guard", "guard-1", "uncited-1"),
          ("Regression suite", "reg-1", "reg-3"), ("Risk tiers", "tier-1", "tier-5")]

W, H = 920, 300
L, R, T, B = 56, 24, 40, 64  # plot margins

CSS = """
.viz-root{color-scheme:light;--surface:#fcfcfb;--text-primary:#0b0b0b;--text-secondary:#52514e;--text-muted:#8a8984;
--grid:#e6e5e0;--band:#f3f2ee;--s1:#2a78d6;--s2:#eb6834;--ok:#1a7f37;--bad:#b42318;
font:14px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif;color:var(--text-primary);background:var(--surface);
max-width:980px;margin:0 auto;padding:28px 24px}
@media (prefers-color-scheme:dark){:root:where(:not([data-theme="light"])) .viz-root{color-scheme:dark;--surface:#1a1a19;
--text-primary:#f0efec;--text-secondary:#c3c2b7;--text-muted:#8f8e86;--grid:#33332f;--band:#232321;--s1:#3987e5;--s2:#d95926;
--ok:#4ac26b;--bad:#ff7b72}}
:root[data-theme="dark"] .viz-root{color-scheme:dark;--surface:#1a1a19;--text-primary:#f0efec;--text-secondary:#c3c2b7;
--text-muted:#8f8e86;--grid:#33332f;--band:#232321;--s1:#3987e5;--s2:#d95926;--ok:#4ac26b;--bad:#ff7b72}
h1{font-size:22px;font-weight:600;margin:0 0 4px}h2{font-size:16px;font-weight:600;margin:36px 0 2px}
.sub{color:var(--text-secondary);margin:0 0 8px}.note{color:var(--text-muted);font-size:12px}
svg text{fill:var(--text-secondary);font-size:11px}svg .lbl{fill:var(--text-primary);font-size:12px}
.grid{stroke:var(--grid);stroke-width:1}.band{fill:var(--band)}
.legend{display:flex;gap:18px;font-size:13px;color:var(--text-secondary);margin:6px 0}
.legend i{display:inline-block;width:14px;height:3px;border-radius:2px;vertical-align:middle;margin-right:6px}
.tiles{display:flex;gap:12px;flex-wrap:wrap;margin:16px 0 4px}
.tile{border:1px solid var(--grid);border-radius:8px;padding:12px 16px;min-width:170px}
.tile b{display:block;font-size:24px;font-weight:600;color:var(--text-primary)}.tile span{color:var(--text-secondary);font-size:12px}
.hit{fill:transparent;cursor:default}
#tip{position:absolute;pointer-events:none;background:var(--surface);border:1px solid var(--grid);border-radius:6px;
padding:8px 10px;font-size:12px;color:var(--text-primary);box-shadow:0 2px 8px rgba(0,0,0,.12);display:none;max-width:280px}
table{border-collapse:collapse;width:100%;margin-top:8px;font-size:13px}th,td{text-align:left;padding:6px 8px;border-bottom:1px solid var(--grid)}
th{color:var(--text-secondary);font-weight:600}td.n{text-align:right;font-variant-numeric:tabular-nums}
"""

JS = """
const tip=document.getElementById('tip');
document.querySelectorAll('[data-tip]').forEach(el=>{
  el.addEventListener('mousemove',e=>{tip.innerHTML=el.dataset.tip;tip.style.display='block';
    const x=e.pageX+14, y=e.pageY+14; tip.style.left=Math.min(x, window.scrollX+innerWidth-300)+'px'; tip.style.top=y+'px';});
  el.addEventListener('mouseleave',()=>{tip.style.display='none';});
});
"""


def pct(a, b):
    return 100.0 * a / b if b else 0.0


def x_of(i, n):
    return L + (W - L - R) * (i + 0.5) / n


def y_of(v, top=100.0):
    return T + (H - T - B) * (1 - v / top)


def axes(n, runs, top=100.0, ticks=(0, 25, 50, 75, 100)):
    parts = []
    names = [r["run"] for r in runs]
    for p_name, a, b in PHASES:  # phase bands behind the data
        i0, i1 = names.index(a), names.index(b)
        x0 = x_of(i0, n) - (W - L - R) / n / 2
        x1 = x_of(i1, n) + (W - L - R) / n / 2
        if PHASES.index((p_name, a, b)) % 2 == 0:
            parts.append(f"<rect class='band' x='{x0:.1f}' y='{T}' width='{x1 - x0:.1f}' height='{H - T - B}'/>")
        parts.append(f"<text class='lbl' x='{(x0 + x1) / 2:.1f}' y='{T - 12}' text-anchor='middle'>{escape(p_name)}</text>")
    for t in ticks:
        y = y_of(t, top)
        parts.append(f"<line class='grid' x1='{L}' x2='{W - R}' y1='{y:.1f}' y2='{y:.1f}'/>")
        parts.append(f"<text x='{L - 8}' y='{y + 4:.1f}' text-anchor='end'>{t}%</text>")
    for i, r in enumerate(runs):  # staggered run names so they never collide
        y = H - B + 18 + (14 if i % 2 else 0)
        parts.append(f"<text x='{x_of(i, n):.1f}' y='{y}' text-anchor='middle'>{escape(r['run'])}</text>")
    return parts


def line(points, color_var, n, top=100.0, dashed=False):
    """points: list of (index, value, tooltip_html) with gaps where the measure did not exist."""
    segs, cur = [], []
    for i, v, _ in points:
        if v is None:
            if cur:
                segs.append(cur)
            cur = []
        else:
            cur.append((x_of(i, n), y_of(v, top)))
    if cur:
        segs.append(cur)
    dash = " stroke-dasharray='5 4'" if dashed else ""
    out = [f"<polyline fill='none' stroke='var({color_var})' stroke-width='2' stroke-linejoin='round'{dash} "
           f"points='{' '.join(f'{x:.1f},{y:.1f}' for x, y in s)}'/>" for s in segs]
    for i, v, tip in points:
        if v is None:
            continue
        x, y = x_of(i, n), y_of(v, top)
        out.append(f"<circle cx='{x:.1f}' cy='{y:.1f}' r='4' fill='var({color_var})' stroke='var(--surface)' stroke-width='2'/>")
        out.append(f"<circle class='hit' cx='{x:.1f}' cy='{y:.1f}' r='14' data-tip=\"{escape(tip, quote=True)}\"/>")
    return out


def chart_pass_rate(runs):
    n = len(runs)
    parts = axes(n, runs)
    names = [r["run"] for r in runs]
    grow = names.index("reg-1")
    xg = x_of(grow, n) - (W - L - R) / n / 2
    parts.append(f"<line x1='{xg:.1f}' x2='{xg:.1f}' y1='{T}' y2='{H - B}' stroke='var(--text-muted)' stroke-dasharray='3 3'/>")
    parts.append(f"<text x='{xg + 6:.1f}' y='{H - B - 8}'>suite grows 30 → 43 scenarios</text>")
    pts = [(i, pct(r["passed"], r["sessions"]),
            f"<b>{escape(r['run'])}</b><br>{r['passed']} of {r['sessions']} sessions passed "
            f"({pct(r['passed'], r['sessions']):.0f}%)<br>{escape(r['change'])}<br>cost ${r['cost_usd']:.2f}")
           for i, r in enumerate(runs)]
    parts += line(pts, "--s1", n)
    last = pts[-1]
    parts.append(f"<text class='lbl' x='{x_of(last[0], n) - 8:.1f}' y='{y_of(last[1]) - 12:.1f}' text-anchor='end'>"
                 f"{last[1]:.0f}%</text>")
    first = pts[0]
    parts.append(f"<text class='lbl' x='{x_of(0, n) + 8:.1f}' y='{y_of(first[1]) + 18:.1f}'>{first[1]:.0f}%</text>")
    return svg(parts, "Pass rate per live run")


def chart_guard(runs):
    n = len(runs)
    parts = axes(n, runs, top=50.0, ticks=(0, 10, 20, 30, 40, 50))
    blocked = [(i, pct(r["blocked"], r["replies"]) if r["guard"] else None,
                f"<b>{escape(r['run'])}</b><br>{r['blocked']} of {r['replies']} replies blocked before sending "
                f"({pct(r['blocked'] or 0, r['replies']):.0f}%)") for i, r in enumerate(runs)]
    fall = [(i, pct(r["fallbacks"], r["replies"]) if r["guard"] else None,
             f"<b>{escape(r['run'])}</b><br>{r['fallbacks']} of {r['replies']} replies replaced by a safe fallback "
             f"({pct(r['fallbacks'] or 0, r['replies']):.0f}%)") for i, r in enumerate(runs)]
    parts += line(blocked, "--s1", n, top=50.0)
    parts += line(fall, "--s2", n, top=50.0, dashed=True)
    parts.append(f"<text x='{x_of(1, n):.1f}' y='{y_of(4, 50.0):.1f}' text-anchor='middle'>no guard yet</text>")
    return svg(parts, "Share of replies blocked by the guard, and replaced by a fallback")


def chart_tiers(by_tier):
    order = [t for t in ("low", "medium", "high") if t in by_tier]
    w, h, left, bar_h, gap = 920, 40 + 44 * len(order), 90, 24, 20
    parts = []
    scale = lambda v: left + (w - left - 120) * v / 100.0  # noqa: E731
    for t in (0, 50, 100):
        parts.append(f"<line class='grid' x1='{scale(t):.1f}' x2='{scale(t):.1f}' y1='16' y2='{h - 20}'/>")
        parts.append(f"<text x='{scale(t):.1f}' y='{h - 4}' text-anchor='middle'>{t}%</text>")
    for k, t in enumerate(order):
        d = by_tier[t]
        rate = pct(d["passed"], d["sessions"])
        y = 20 + k * (bar_h + gap)
        tip = (f"<b>{t} risk</b><br>{d['passed']} of {d['sessions']} sessions passed ({rate:.0f}%)<br>"
               + {"low": "Paraphrase of cited sections allowed",
                  "medium": "Numbers and risky words must be inside the quote",
                  "high": "Only approved sentences, assembled by code"}[t])
        parts.append(f"<text class='lbl' x='{left - 10}' y='{y + 16}' text-anchor='end'>{t.capitalize()}</text>")
        parts.append(f"<rect x='{left}' y='{y}' width='{scale(rate) - left:.1f}' height='{bar_h}' rx='4' fill='var(--s1)'/>")
        parts.append(f"<text class='lbl' x='{scale(rate) + 8:.1f}' y='{y + 16}'>{d['passed']} of {d['sessions']}</text>")
        parts.append(f"<rect class='hit' x='{left}' y='{y - 4}' width='{w - left - 40}' height='{bar_h + 8}' "
                     f"data-tip=\"{escape(tip, quote=True)}\"/>")
    return (f"<svg viewBox='0 0 {w} {h}' width='100%' role='img' aria-label='Pass rate by risk tier, latest run'>"
            + "".join(parts) + "</svg>")


def svg(parts, label):
    return f"<svg viewBox='0 0 {W} {H}' width='100%' role='img' aria-label='{escape(label)}'>" + "".join(parts) + "</svg>"


def render(data: dict) -> str:
    runs = data["runs"]
    latest = runs[-1]
    by_tier = data["latest_by_tier"]["sessions"]
    high = by_tier.get("high", {"sessions": 0, "passed": 0})
    high_fail = high["sessions"] - high["passed"]
    # Gate: zero UNSAFE high-risk failures. Without the split, every failure counts as unsafe (fail closed).
    unsafe = high.get("unsafe", high_fail)
    cautious = high.get("over_cautious", 0)
    total_cost = sum(r["cost_usd"] for r in runs)
    def row(r):
        blocked = f"{r['blocked']}/{r['replies']}" if r["guard"] else "—"
        fallbacks = r["fallbacks"] if r["guard"] else "—"
        return (f"<tr><td>{escape(r['run'])}</td><td>{escape(r['change'])}</td>"
                f"<td class='n'>{r['passed']}/{r['sessions']}</td><td class='n'>{pct(r['passed'], r['sessions']):.0f}%</td>"
                f"<td class='n'>{blocked}</td><td class='n'>{fallbacks}</td><td class='n'>${r['cost_usd']:.2f}</td></tr>")

    rows = "".join(row(r) for r in runs)
    billed = data.get("billed_usd")
    cost_tile = (f"<b>${billed:.2f}</b><span>billed for {len(runs)} runs; estimated ${total_cost:.2f} at list price</span>"
                 if billed is not None else f"<b>${total_cost:.2f}</b><span>model usage, {len(runs)} runs</span>")
    calls = data.get("call_runs") or []
    calls_html = ""
    if calls:
        def crow(c):
            st = "".join(f"<td class='n'>{c['stages'].get(k, 0) * 100:.0f}%</td>" for k in ("1-5", "6-10", "11-15"))
            return (f"<tr><td>{escape(c['run'])}</td><td>{escape(c['change'])}</td>"
                    f"<td class='n'>{c['passed']}/{c['turns']}</td>{st}<td class='n'>{c['contradictions']}</td>"
                    f"<td>{'Pass' if c['gate'] else 'Blocked'}</td><td class='n'>${c['cost_usd']:.2f}</td></tr>")
        import sys as _sys
        _sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        from deploykit.gate import release_status
        rs = release_status(calls, n=3)
        last = ", ".join(c["run"] for c in calls[-rs["streak"]:]) if rs["streak"] else ""
        ready = (f"<p class='sub'><b>Release ready:</b> the gate passed {rs['streak']} runs in a row ({escape(last)})."
                 "</p>" if rs["ready"] else
                 f"<p class='sub'><b>Not release ready:</b> {rs['streak']} of {rs['needed']} passing runs in a row.</p>")
        calls_html = ("<h2>Chapter 2: demo calls</h2><p class='sub'>5 simulated calls of 15 turns, one question at a "
                      "time, with the whole call in context. Turn pass rate early, middle and late in the call.</p>"
                      "<table><tr><th>Run</th><th>What changed</th><th>Turns passed</th><th>Turns 1-5</th>"
                      "<th>6-10</th><th>11-15</th><th>Contradictions</th><th>Gate</th><th>Cost</th></tr>"
                      + "".join(crow(c) for c in calls) + "</table>" + ready)
    gate = ("<b style='color:var(--ok)'>Pass</b>" if unsafe == 0 else "<b style='color:var(--bad)'>Blocked</b>")
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Agent go-live kit · run history</title>
<style>{CSS}</style></head><body class="viz-root">
<h1>From 47% to {pct(latest['passed'], latest['sessions']):.0f}%: {len(runs)} live runs of a HiBob demo agent</h1>
<p class="sub">Each run is the full simulated-buyer suite on open models (Nebius). Every change between runs was test-first.</p>
<div class="tiles">
<div class="tile"><span>Latest pass rate</span><b>{pct(latest['passed'], latest['sessions']):.0f}%</b><span>{latest['passed']} of {latest['sessions']} sessions · {escape(latest['run'])}</span></div>
<div class="tile"><span>Release gate</span>{gate}<span>{unsafe} unsafe, {cautious} over-cautious high-risk failures</span></div>
<div class="tile"><span>Guard blocks, latest</span><b>{pct(latest['blocked'], latest['replies']):.0f}%</b><span>{latest['blocked']} of {latest['replies']} replies, {latest['fallbacks']} fallbacks</span></div>
<div class="tile"><span>Cost of all runs</span>{cost_tile}</div>
</div>
<h2>Pass rate per run</h2>
<p class="sub">Dips are deliberate: stricter checks first catch more, then get tuned for precision.</p>
{chart_pass_rate(runs)}
<h2>Guard activity per run</h2>
<p class="sub">Blocked drafts are fixed on retry or replaced by a safe reply before the buyer sees them.</p>
<div class="legend"><span><i style="background:var(--s1)"></i>Blocked before sending</span>
<span><i style="background:var(--s2)"></i>Replaced by a safe fallback (dashed)</span></div>
{chart_guard(runs)}
<h2>Latest run by risk tier</h2>
<p class="sub">Low: paraphrase allowed. Medium: numbers and risky words quoted. High: approved sentences only.
High-risk failures: {unsafe} unsafe (said something it shouldn't), {cautious} over-cautious (held back when it could have
answered). The gate blocks only on unsafe ones.</p>
{chart_tiers(by_tier)}
<h2>All runs</h2>
<table><tr><th>Run</th><th>What changed</th><th>Passed</th><th>Rate</th><th>Blocked</th><th>Fallbacks</th><th>Cost</th></tr>{rows}</table>
{calls_html}
<p class="note">{escape(data.get('_note', ''))}</p>
<div id="tip"></div><script>{JS}</script></body></html>"""


def main(argv=None):
    argv = argv if argv is not None else sys.argv[1:]
    src = Path(argv[0]) if argv else ROOT / "docs" / "run_history.json"
    dst = Path(argv[1]) if len(argv) > 1 else ROOT / "docs" / "run_history.html"
    dst.write_text(render(json.loads(src.read_text())), encoding="utf-8")
    print(dst)


if __name__ == "__main__":
    main()
