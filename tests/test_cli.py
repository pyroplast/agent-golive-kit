import csv
import json

import pytest

from deploykit.cli import main
from tests.conftest import FIXTURES, ROOT


def run_args(tmp_path, name="base", extra=()):
    return ["run", "--provider", "mock", "--name", name, "--out", str(tmp_path),
            "--kb", str(FIXTURES / "kb"), "--scenarios", str(FIXTURES / "scenarios.json"),
            "--config", str(ROOT / "config" / "client.json"), *extra]


def test_cli_run_creates_report(tmp_path):
    assert main(run_args(tmp_path)) == 0
    assert (tmp_path / "base" / "report.html").exists()


def test_cli_limit_and_langs(tmp_path):
    main(run_args(tmp_path, extra=["--limit", "1", "--langs", "en"]))
    rows = (tmp_path / "base" / "results.jsonl").read_text().splitlines()
    assert len(rows) == 1


def test_cli_diff_writes_markdown(tmp_path):
    main(run_args(tmp_path, "a"))
    main(run_args(tmp_path, "b"))
    assert main(["diff", str(tmp_path / "a"), str(tmp_path / "b")]) == 0
    md = (tmp_path / "b" / "diff_vs_a.md").read_text()
    assert "Pass rate" in md


def test_cli_rejects_placeholder_model_ids(tmp_path):
    models = tmp_path / "models.json"
    models.write_text(json.dumps({"bedrock": {"agent": "REPLACE_ME", "judge": "x", "buyer": "x"}}))
    with pytest.raises(SystemExit):
        main(run_args(tmp_path, extra=["--provider", "bedrock", "--models", str(models)]))


def test_sample_and_agreement(tmp_path):
    main(run_args(tmp_path))
    run = tmp_path / "base"
    assert main(["sample", str(run), "--n", "3", "--seed", "1"]) == 0
    path = run / "labels.csv"
    rows = list(csv.DictReader(path.open()))
    assert len(rows) == 3 and rows[0]["human_grounded"] == ""
    for r in rows:
        r["human_grounded"] = r["judge_grounded"]
    rows[0]["human_grounded"] = "false" if rows[0]["judge_grounded"] == "true" else "true"
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    assert main(["agreement", str(run)]) == 0
    result = json.loads((run / "agreement.json").read_text())
    assert result["labelled"] == 3
    assert result["agreement"] == pytest.approx(2 / 3, abs=1e-3)


def test_model_ids_can_come_from_env(tmp_path, monkeypatch):
    from deploykit.cli import _models
    models = tmp_path / "models.json"
    models.write_text(json.dumps({"bedrock": {"agent": "REPLACE_ME", "judge": "REPLACE_ME", "buyer": "REPLACE_ME"}}))
    for role in ("AGENT", "JUDGE", "BUYER"):
        monkeypatch.setenv(f"MODEL_{role}", f"eu.anthropic.{role.lower()}")
    m, _, _ = _models(models, "bedrock")
    assert m == {"agent": "eu.anthropic.agent", "judge": "eu.anthropic.judge", "buyer": "eu.anthropic.buyer"}
