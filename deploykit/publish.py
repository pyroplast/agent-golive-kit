"""Upload a run to S3 and refresh the public 'latest' report."""
from __future__ import annotations

from pathlib import Path

CONTENT_TYPES = {".html": "text/html; charset=utf-8", ".json": "application/json",
                 ".jsonl": "application/x-ndjson", ".md": "text/markdown; charset=utf-8"}


def publish_run(run_dir, bucket: str, prefix: str = "runs", s3_client=None) -> list[str]:
    if s3_client is None:
        import boto3
        s3_client = boto3.client("s3")
    run_dir = Path(run_dir)
    keys = []
    for f in sorted(run_dir.iterdir()):
        if f.suffix not in CONTENT_TYPES:
            continue
        key = f"{prefix}/{run_dir.name}/{f.name}"
        s3_client.put_object(Bucket=bucket, Key=key, Body=f.read_bytes(),
                             ContentType=CONTENT_TYPES[f.suffix])
        keys.append(key)
    for name in ("report.html", "meta.json"):
        f = run_dir / name
        if f.exists():
            key = f"latest/{name}"
            s3_client.put_object(Bucket=bucket, Key=key, Body=f.read_bytes(),
                                 ContentType=CONTENT_TYPES[f.suffix], CacheControl="no-cache")
            keys.append(key)
    return keys
