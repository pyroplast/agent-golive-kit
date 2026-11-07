"""Model providers and AWS publishing, tested with fake clients (no network)."""
import json

from deploykit.llm import AnthropicLLM, BedrockLLM
from deploykit.publish import publish_run


class FakeAnthropic:
    def __init__(self):
        self.kwargs = None

        class Msgs:
            def create(inner, **kw):
                self.kwargs = kw

                class R:
                    content = [type("B", (), {"type": "text", "text": "hello"})()]
                    usage = type("U", (), {"input_tokens": 12, "output_tokens": 3})()
                return R()
        self.messages = Msgs()


def test_anthropic_llm_maps_role_to_model_and_caches_system():
    fake = FakeAnthropic()
    llm = AnthropicLLM({"agent": "model-a", "judge": "model-j"}, client=fake)
    r = llm.complete("agent", "SYS", [{"role": "user", "content": "hi"}])
    assert r.text == "hello" and r.input_tokens == 12 and r.model == "model-a"
    assert fake.kwargs["model"] == "model-a"
    assert fake.kwargs["system"][0]["text"] == "SYS"
    assert fake.kwargs["system"][0]["cache_control"] == {"type": "ephemeral"}


class FakeBedrock:
    def __init__(self):
        self.kwargs = None

    def converse(self, **kw):
        self.kwargs = kw
        return {"output": {"message": {"content": [{"text": "hola"}]}},
                "usage": {"inputTokens": 7, "outputTokens": 2}}


def test_bedrock_llm_uses_converse_format():
    fake = FakeBedrock()
    llm = BedrockLLM({"agent": "eu.anthropic.model"}, client=fake)
    r = llm.complete("agent", "SYS", [{"role": "user", "content": "hi"}])
    assert r.text == "hola" and r.output_tokens == 2
    assert fake.kwargs["modelId"] == "eu.anthropic.model"
    assert fake.kwargs["system"] == [{"text": "SYS"}]
    assert fake.kwargs["messages"] == [{"role": "user", "content": [{"text": "hi"}]}]


class FakeS3:
    def __init__(self):
        self.puts = []

    def put_object(self, **kw):
        self.puts.append(kw)


def test_publish_run_uploads_files_and_latest(tmp_path):
    run = tmp_path / "base"
    run.mkdir()
    (run / "results.jsonl").write_text("{}")
    (run / "meta.json").write_text(json.dumps({"name": "base"}))
    (run / "report.html").write_text("<html></html>")
    s3 = FakeS3()
    keys = publish_run(run, "bucket", "runs", s3_client=s3)
    assert "runs/base/report.html" in keys
    assert "latest/report.html" in keys
    html_put = next(p for p in s3.puts if p["Key"] == "latest/report.html")
    assert html_put["ContentType"] == "text/html; charset=utf-8"
    assert all(p["Bucket"] == "bucket" for p in s3.puts)


class FakeOpenAI:
    """Mimics the openai SDK client shape used by OpenAICompatLLM."""

    def __init__(self, text="ok"):
        self.kwargs = None
        outer = self

        class Completions:
            def create(self, **kw):
                outer.kwargs = kw
                msg = type("M", (), {"content": text})()
                choice = type("C", (), {"message": msg})()
                usage = type("U", (), {"prompt_tokens": 20, "completion_tokens": 4})()
                return type("R", (), {"choices": [choice], "usage": usage})()

        class Chat:
            completions = Completions()

        class Models:
            def list(self):
                return type("L", (), {"data": [type("D", (), {"id": "qwen/x"})(), type("D", (), {"id": "openai/gpt-oss-120b"})()]})()

        self.chat = Chat()
        self.models = Models()


def test_openai_compat_puts_system_first_and_reads_usage():
    from deploykit.llm import OpenAICompatLLM
    fake = FakeOpenAI("hello")
    llm = OpenAICompatLLM({"agent": "openai/gpt-oss-120b"}, client=fake)
    r = llm.complete("agent", "SYS", [{"role": "user", "content": "hi"}])
    assert r.text == "hello" and r.input_tokens == 20 and r.output_tokens == 4
    assert r.model == "openai/gpt-oss-120b"
    assert fake.kwargs["messages"][0] == {"role": "system", "content": "SYS"}
    assert fake.kwargs["messages"][1] == {"role": "user", "content": "hi"}
    assert fake.kwargs["temperature"] == 0


def test_openai_compat_handles_empty_content():
    from deploykit.llm import OpenAICompatLLM
    r = OpenAICompatLLM({"agent": "m"}, client=FakeOpenAI(None)).complete("agent", "S", [{"role": "user", "content": "x"}])
    assert r.text == ""


def test_list_models():
    from deploykit.llm import OpenAICompatLLM
    assert OpenAICompatLLM({}, client=FakeOpenAI()).list_models() == ["openai/gpt-oss-120b", "qwen/x"]
