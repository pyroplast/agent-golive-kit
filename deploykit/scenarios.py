"""Load and validate the simulation suite."""
from __future__ import annotations

import json
from pathlib import Path

CATEGORIES = {"expected", "ambiguous", "adversarial"}
LANGS = ("en",)


def load_scenarios(path) -> list[dict]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    validate_scenarios(data)
    return data


def validate_scenarios(items: list[dict], personas: set | None = None) -> None:
    seen = set()
    for i, s in enumerate(items):
        where = f"scenario #{i} ({s.get('id', '?')})"
        for key in ("id", "category", "persona", "opening", "expect"):
            if key not in s:
                raise ValueError(f"{where}: missing '{key}'")
        if s["id"] in seen:
            raise ValueError(f"{where}: duplicate id")
        seen.add(s["id"])
        if s["category"] not in CATEGORIES:
            raise ValueError(f"{where}: category must be one of {sorted(CATEGORIES)}")
        if personas is not None and s["persona"] not in personas:
            raise ValueError(f"{where}: unknown persona '{s['persona']}'")
        for lang in LANGS:
            if not s["opening"].get(lang):
                raise ValueError(f"{where}: opening missing language '{lang}'")
        if s["expect"].get("handoff") not in (True, False, None):
            raise ValueError(f"{where}: expect.handoff must be true, false or null")
