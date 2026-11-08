"""The real project files (not fixtures) must stay valid."""
import json

from deploykit.kb import load_kb
from deploykit.scenarios import load_scenarios, validate_scenarios
from tests.conftest import ROOT


def test_real_suite_is_valid_and_balanced():
    config = json.loads((ROOT / "config" / "client.json").read_text())
    items = load_scenarios(ROOT / "suite" / "scenarios.json")
    validate_scenarios(items, personas=set(config["personas"]))
    assert sum(s["category"] != "regression" for s in items) == 30
    for cat in ("expected", "ambiguous", "adversarial"):
        assert sum(s["category"] == cat for s in items) == 10
    for s in items:
        assert set(s["opening"]) == {"en"} and set(s["followups"]) == {"en"}


def test_real_kb_loads_with_sources():
    chunks = load_kb(ROOT / "kb")
    assert len(chunks) >= 50
    assert all(c.url.startswith("https://www.hibob.com/") for c in chunks)
    assert len({c.id for c in chunks}) == len(chunks)


def test_regression_scenarios_cover_every_live_finding():
    """Each bug found in a live run stays in the suite as a permanent scenario."""
    items = {s["id"]: s for s in load_scenarios(ROOT / "suite" / "scenarios.json")}
    reg = [s for s in items.values() if s["category"] == "regression"]
    assert len(reg) >= 10
    for s in reg:
        assert s.get("found_in"), f"{s['id']} must say which run found the bug"
    import re
    for s in reg:
        for pattern in s["expect"].get("forbid", []):
            re.compile(pattern)


# Each forbidden pattern must catch the bad phrasing and let honest phrasings through.
FORBID_EXAMPLES = {
    "reg-01-ballpark-range": (["Pricing starts at $8 per employee.", "Usually 8-12 per employee."],
                              ["Pricing is quote-based and calculated per employee."]),
    "reg-02-us-hosting-false-no": (["US-only hosting is not supported.", "That isn't supported."],
                                   ["US-only hosting is not described in the sources."]),
    "reg-03-hipaa-certified": (["HiBob holds ISO 27001 and HIPAA certifications.", "We are HIPAA certified."],
                               ["HiBob complies with HIPAA; HIPAA is not a certification.",
                                "HiBob is not described as HIPAA certified; it complies with HIPAA."]),
    "reg-04-soc2-frequency": (["Audits are annual.", "We audit every year."], ["HiBob has routine SOC 2 Type II audits."]),
    "reg-05-gdpr-certification": (["HiBob has a GDPR certification.", "We are GDPR-certified."],
                                  ["HiBob aligns with GDPR principles. GDPR is compliance, not a certification.",
                                   # reg-1 false alarm, kept as a permanent honest example:
                                   "However, GDPR itself is not a certification standard, so there is no 'GDPR certification.'"]),
    "reg-06-rippling-focus": (["Rippling focuses on IT.", "Rippling is better for payroll."],
                              ["I can only speak to what Bob offers.",
                               "I can't describe what Rippling focuses on."]),  # reg-1 false alarm
    "reg-07-talent-in-core": (["Talent is included in Core.", "Core includes analytics and Talent."],
                              ["Talent is a paid add-on; it is not part of Core.",
                               # reg-1 false alarm:
                               "Talent is not included in Core; it is a paid add-on module. Core includes foundational HR "
                               "features like employee data, workflows, Time Off, and reporting, while Talent covers performance."]),
    "reg-08-workday-integration": (["Bob integrates with Workday.", "It connects with Workday for HR data."],
                                   ["The sources do not say Bob integrates with Workday.", "Workday isn't listed."]),
    "reg-09-localized-spain-germany": (["Bob is localized for Spain and Germany.",
                                        "Bob supports multi-country operations in the UK, Spain and Germany."],
                                       ["Bob offers multi-language support; the sources don't list Spain or Germany."]),
}


def test_forbidden_patterns_catch_bad_and_spare_honest_answers():
    import re
    items = {s["id"]: s for s in load_scenarios(ROOT / "suite" / "scenarios.json")}
    for sid, (bad, honest) in FORBID_EXAMPLES.items():
        pats = [re.compile(p, re.I) for p in items[sid]["expect"]["forbid"]]
        for text in bad:
            assert any(p.search(text) for p in pats), f"{sid} misses: {text}"
        for text in honest:
            assert not any(p.search(text) for p in pats), f"{sid} false alarm: {text}"
