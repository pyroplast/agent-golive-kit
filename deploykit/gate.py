"""Release readiness across runs: one passing run can be luck, so the gate must pass several runs in a row."""
from __future__ import annotations

import json
from pathlib import Path


def release_status(runs: list[dict], n: int = 3) -> dict:
    """runs: oldest first, each with a boolean "gate". Ready when the last n runs all passed."""
    streak = 0
    for r in reversed(runs):
        if not r.get("gate"):
            break
        streak += 1
    return {"ready": streak >= n, "streak": streak, "needed": n}


def load_reviews(name: str, folder) -> dict:
    p = Path(folder) / f"{name}.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}
