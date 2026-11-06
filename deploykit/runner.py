"""Run the whole suite: every scenario in every language, scored turn by turn."""
from __future__ import annotations

import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

from deploykit.agent import DemoAgent
from deploykit.judge import Judge, session_verdict
from deploykit.llm import CostTracker, TrackedLLM
from deploykit.simulator import Buyer


def run_session(scenario, lang, agent, judge, buyer, max_turns):
    history = [{"role": "user", "content": scenario["opening"][lang]}]
    turns = []
    for i in range(max_turns):
        a = agent.respond(history, lang)
        j = judge.score(scenario, lang, history, a)
        turns.append({"user": history[-1]["content"], "agent": a, "judge": j})
        history.append({"role": "assistant", "content": a["reply"]})
        if a["handoff"] or i == max_turns - 1:
            break
        nxt = buyer.next_message(scenario, lang, history, i + 1)
        if not nxt:
            break
        history.append({"role": "user", "content": nxt})
    return {"scenario_id": scenario["id"], "category": scenario["category"],
            "persona": scenario["persona"], "lang": lang, "turns": turns,
            "verdict": session_verdict(scenario, turns)}


def run_suite(config, chunks, scenarios, llm, langs=("en",), max_turns=3,
              out_dir="runs", name=None, prices=None, workers=1, retrieval="full",
              provider="mock") -> Path:
    tracker = CostTracker(prices)
    tracked = TrackedLLM(llm, tracker)
    agent = DemoAgent(config, chunks, tracked, retrieval=retrieval)
    judge = Judge(tracked, chunks)
    buyer = Buyer(tracked, config)

    jobs = [(s, lang) for s in scenarios for lang in langs]
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        rows = list(pool.map(lambda job: run_session(job[0], job[1], agent, judge, buyer, max_turns), jobs))

    name = name or datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    run_dir = Path(out_dir) / name
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "results.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    fingerprint = hashlib.sha256(
        (json.dumps(config, sort_keys=True) + "".join(c.id + c.text for c in chunks)).encode()
    ).hexdigest()[:12]
    meta = {"name": name, "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "provider": provider, "sessions": len(rows), "langs": list(langs),
            "max_turns": max_turns, "retrieval": retrieval, "kb_chunks": len(chunks),
            "config_fingerprint": fingerprint, "cost": tracker.summary()}
    (run_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return run_dir
