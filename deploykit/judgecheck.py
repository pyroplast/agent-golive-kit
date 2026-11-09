"""Judge test set: real replies labelled by hand, plus copies with one false fact planted. A judge model is scored
on agreement, on catching the planted and real errors, and on false alarms."""
from __future__ import annotations

from deploykit.judge import Judge


def run_judge_check(llm, chunks, items: list[dict]) -> dict:
    judge, rows = Judge(llm, chunks), []
    for it in items:
        history = list(it.get("history") or []) + [{"role": "user", "content": it["asked"]}]
        out = judge.score({"category": "judge-check"}, "en", history,
                          {"reply": it["reply"], "citations": it.get("citations", []), "handoff": it.get("handoff", False)})
        verdict = bool(out.get("grounded")) and not out.get("judge_error")
        rows.append({"id": it["id"], "label": bool(it["grounded"]), "judge": verdict, "agree": verdict == bool(it["grounded"])})
    bad = [r for r in rows if not r["label"]]
    good = [r for r in rows if r["label"]]
    return {"items": len(rows), "agree": sum(r["agree"] for r in rows),
            "caught": sum(not r["judge"] for r in bad), "bad": len(bad),
            "false_alarms": sum(not r["judge"] for r in good), "good": len(good), "rows": rows}
