import json

import pytest

from deploykit.llm import CostTracker, MockLLM
from deploykit.report import build_report, diff_runs, summarize
from deploykit.runner import run_suite
from deploykit.scenarios import validate_scenarios
from deploykit.simulator import Buyer


def test_validate_accepts_fixture(scenarios):
    validate_scenarios(scenarios)


@pytest.mark.parametrize("breaker", [
    lambda s: s[0].pop("opening"),
    lambda s: s[0].update(category="weird"),
    lambda s: s[0]["opening"].pop("en"),
    lambda s: s.append(dict(s[0])),  # duplicate id
])
def test_validate_rejects_bad_scenarios(scenarios, breaker):
    breaker(scenarios)
    with pytest.raises(ValueError):
        validate_scenarios(scenarios)


def test_buyer_mock_uses_scripted_followups(config, scenarios):
    buyer = Buyer(MockLLM(), config)
    assert buyer.next_message(scenarios[0], "en", [], 1) == "And where are backups stored?"
    assert buyer.next_message(scenarios[0], "en", [], 2) is None


def test_cost_tracker_sums_tokens_and_prices():
    t = CostTracker({"m": {"input": 1.0, "output": 2.0}})
    from deploykit.llm import LLMResponse
    t.add("agent", LLMResponse("x", 1_000_000, 500_000, "m"))
    t.add("judge", LLMResponse("x", 0, 0, "unpriced"))
    s = t.summary()
    assert s["by_role"]["agent"]["input_tokens"] == 1_000_000
    assert s["total_usd"] == pytest.approx(2.0)
    assert s["unpriced_models"] == ["unpriced"]


@pytest.fixture
def run_dir(tmp_path, config, chunks, scenarios):
    return run_suite(config, chunks, scenarios, MockLLM(), langs=["en"],
                     max_turns=3, out_dir=tmp_path, name="base")


def test_run_suite_writes_one_result_per_session_and_lang(run_dir, scenarios):
    rows = [json.loads(l) for l in (run_dir / "results.jsonl").read_text().splitlines()]
    assert len(rows) == len(scenarios)
    hosting_en = next(r for r in rows if r["scenario_id"] == "exp-hosting" and r["lang"] == "en")
    assert len(hosting_en["turns"]) == 2  # opening + one scripted follow-up
    meta = json.loads((run_dir / "meta.json").read_text())
    assert meta["name"] == "base" and meta["sessions"] == 3 and "cost" in meta


def test_mock_run_outcomes(run_dir):
    rows = {(r["scenario_id"], r["lang"]): r for r in
            (json.loads(l) for l in (run_dir / "results.jsonl").read_text().splitlines())}
    assert rows[("adv-discount", "en")]["verdict"]["passed"] is True
    assert rows[("exp-hosting", "en")]["verdict"]["passed"] is True


def test_summarize_and_report(run_dir):
    rows = [json.loads(l) for l in (run_dir / "results.jsonl").read_text().splitlines()]
    s = summarize(rows)
    assert s["sessions"] == 3
    assert set(s["by_category"]) == {"expected", "ambiguous", "adversarial"}
    assert 0.0 <= s["pass_rate"] <= 1.0
    html = build_report(run_dir).read_text()
    assert "<html" in html and "Pass rate" in html and "Knowledge gaps" in html


def test_diff_runs_finds_flips(tmp_path, run_dir):
    other = tmp_path / "other"
    other.mkdir()
    rows = [json.loads(l) for l in (run_dir / "results.jsonl").read_text().splitlines()]
    rows[0]["verdict"] = {"passed": not rows[0]["verdict"]["passed"], "reasons": ["ungrounded"]}
    (other / "results.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
    (other / "meta.json").write_text((run_dir / "meta.json").read_text())
    d = diff_runs(run_dir, other)
    key = f'{rows[0]["scenario_id"]}:{rows[0]["lang"]}'
    assert key in d["fixed"] + d["broke"]
    assert len(d["fixed"]) + len(d["broke"]) == 1


def test_summary_counts_guard_activity():
    rows = [{"scenario_id": "a", "category": "expected", "persona": "cfo", "lang": "en",
             "verdict": {"passed": True, "reasons": []},
             "turns": [{"user": "q", "agent": {"reply": "r", "citations": [], "handoff": False, "unknown": False,
                                                "guard": {"enabled": True, "retried": True, "fell_back": False,
                                                          "first_problems": [{"type": "watch_word", "detail": "certif"}],
                                                          "final_problems": []}},
                        "judge": {"grounded": True}},
                       {"user": "q2", "agent": {"reply": "r", "citations": [], "handoff": True, "unknown": False,
                                                 "guard": {"enabled": True, "retried": True, "fell_back": True,
                                                           "first_problems": [{"type": "number_not_in_source", "detail": "8"}],
                                                           "final_problems": [{"type": "number_not_in_source", "detail": "8"}]}},
                        "judge": {"grounded": True}}]}]
    s = summarize(rows)
    assert s["guard"] == {"turns_checked": 2, "retries": 2, "fallbacks": 1,
                          "caught": {"watch_word": 1, "number_not_in_source": 1}}
