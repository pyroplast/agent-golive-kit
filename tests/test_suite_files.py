"""The real project files (not fixtures) must stay valid."""
import json

from deploykit.kb import load_kb
from deploykit.scenarios import load_scenarios, validate_scenarios
from tests.conftest import ROOT


def test_real_suite_is_valid_and_balanced():
    config = json.loads((ROOT / "config" / "client.json").read_text())
    items = load_scenarios(ROOT / "suite" / "scenarios.json")
    validate_scenarios(items, personas=set(config["personas"]))
    assert len(items) == 30
    for cat in ("expected", "ambiguous", "adversarial"):
        assert sum(s["category"] == cat for s in items) == 10
    for s in items:
        assert set(s["opening"]) == {"en"} and set(s["followups"]) == {"en"}


def test_real_kb_loads_with_sources():
    chunks = load_kb(ROOT / "kb")
    assert len(chunks) >= 50
    assert all(c.url.startswith("https://www.hibob.com/") for c in chunks)
    assert len({c.id for c in chunks}) == len(chunks)
