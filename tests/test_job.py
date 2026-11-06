"""Nightly regression job, tested with a fake S3 and the mock model."""
import io
import json

from tests.botocore_stub import NoSuchKey  # noqa: F401  (see fake below)
from deploykit.job import nightly
from deploykit.llm import MockLLM
from tests.conftest import FIXTURES, ROOT


class FakeS3:
    def __init__(self, objects=None):
        self.objects = dict(objects or {})

    def put_object(self, Bucket, Key, Body, ContentType=None, CacheControl=None):
        self.objects[Key] = Body if isinstance(Body, bytes) else Body.encode()

    def get_object(self, Bucket, Key):
        if Key not in self.objects:
            raise NoSuchKey(Key)
        return {"Body": io.BytesIO(self.objects[Key])}


def job_kwargs(tmp_path, s3):
    return dict(bucket="b", llm=MockLLM(), s3_client=s3, workdir=tmp_path,
                kb=FIXTURES / "kb", scenarios=FIXTURES / "scenarios.json",
                config=ROOT / "config" / "client.json", provider="mock")


def test_first_night_publishes_without_diff(tmp_path):
    s3 = FakeS3()
    out = nightly(name="n1", **job_kwargs(tmp_path, s3))
    assert out["previous"] is None
    assert "latest/report.html" in s3.objects
    assert json.loads(s3.objects["latest/meta.json"])["name"] == "n1"


def test_second_night_diffs_against_latest(tmp_path):
    s3 = FakeS3()
    nightly(name="n1", **job_kwargs(tmp_path / "a", s3))
    out = nightly(name="n2", **job_kwargs(tmp_path / "b", s3))
    assert out["previous"] == "n1"
    assert out["diff"]["fixed"] == [] and out["diff"]["broke"] == []
    assert "runs/n2/diff_vs_n1.md" in s3.objects
