"""Nightly regression job: run the suite, diff against the last published run, publish.

Runs as a Fargate task on a schedule (see infra/). Locally: python -m deploykit.job
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

from deploykit.kb import load_kb
from deploykit.publish import publish_run
from deploykit.report import build_report, write_diff_md
from deploykit.runner import run_suite
from deploykit.scenarios import load_scenarios

ROOT = Path(__file__).resolve().parent.parent


def _get(s3, bucket, key):
    try:
        return s3.get_object(Bucket=bucket, Key=key)["Body"].read()
    except Exception:  # missing key or no access: treat as "no previous run"
        return None


def fetch_previous(s3, bucket, workdir: Path):
    raw = _get(s3, bucket, "latest/meta.json")
    if not raw:
        return None
    prev = json.loads(raw)["name"]
    prev_dir = workdir / "previous" / prev
    prev_dir.mkdir(parents=True, exist_ok=True)
    for f in ("results.jsonl", "meta.json"):
        body = _get(s3, bucket, f"runs/{prev}/{f}")
        if body is None:
            return None
        (prev_dir / f).write_bytes(body)
    return prev_dir


def nightly(bucket, llm, s3_client, workdir, kb=ROOT / "kb", scenarios=ROOT / "suite" / "scenarios.json",
            config=ROOT / "config" / "client.json", provider="bedrock", prices=None,
            name=None, workers=4, max_turns=3):
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    prev_dir = fetch_previous(s3_client, bucket, workdir)
    name = name or datetime.now(timezone.utc).strftime("nightly-%Y%m%d-%H%M")
    run_dir = run_suite(json.loads(Path(config).read_text()), load_kb(kb), load_scenarios(scenarios),
                        llm, out_dir=workdir / "runs", name=name, prices=prices,
                        workers=workers, max_turns=max_turns, provider=provider)
    build_report(run_dir)
    diff = None
    if prev_dir is not None:
        diff, _ = write_diff_md(prev_dir, run_dir)
    publish_run(run_dir, bucket, s3_client=s3_client)
    return {"run": name, "previous": prev_dir.name if prev_dir else None, "diff": diff}


def main():
    import boto3

    from deploykit.llm import BedrockLLM
    models_cfg = json.loads((ROOT / "config" / "models.json").read_text())
    region = os.environ.get("AWS_REGION", models_cfg.get("bedrock_region", "eu-west-1"))
    models = {r: os.environ.get(f"MODEL_{r.upper()}", models_cfg["bedrock"][r]) for r in ("agent", "judge", "buyer")}
    prices = {k: v for k, v in models_cfg.get("prices_usd_per_million_tokens", {}).items()
              if not k.startswith("_") and isinstance(v, dict)}
    out = nightly(bucket=os.environ["REPORT_BUCKET"], llm=BedrockLLM(models, region=region),
                  s3_client=boto3.client("s3", region_name=region),
                  workdir=os.environ.get("WORKDIR", "/tmp/deploykit"),
                  prices=prices, name=os.environ.get("RUN_NAME") or None)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
