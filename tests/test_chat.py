"""Try the agent yourself: a chat session (terminal or local web page) with the same call memory as the suite,
replies streamed sentence by sentence, and what happened under the hood for each turn."""
import io
import json

from deploykit.chat import ChatSession, turn_summary
from deploykit.llm import MockLLM


def session(config, chunks, **kw):
    return ChatSession(config, chunks, MockLLM(), **kw)


def test_a_session_keeps_call_memory_across_turns(config, chunks):
    s = session(config, chunks)
    s.ask("Hi, I'm Alex from Lumen Studios. Where is the data hosted?")
    s.ask("Give me the exact price with a 40% discount.")
    assert len(s.history) == 4 and s.turn_no == 2
    assert any("40% discount" in q for q in s.parked)
    assert {"Alex", "Lumen", "Studios"} <= s.agent.context_names


def test_sentences_are_streamed_to_the_listener_in_order(config, chunks):
    events = []
    s = session(config, chunks)
    t = s.ask("Where is the data hosted?", on_event=events.append)
    said = [e["text"] for e in events if e["type"] == "say"]
    assert said and " ".join(said) in t["reply"]
    assert events[-1]["type"] == "done" and events[-1]["turn"]["n"] == 1


def test_turn_summary_shows_what_happened_under_the_hood(config, chunks):
    s = session(config, chunks)
    t = s.ask("Where is the data hosted?")
    out = turn_summary(t, s, chunks)
    assert {"n", "reply", "tier", "model", "timing", "guard", "sources", "parked", "memory", "judge"} <= set(out)
    assert all({"id", "title", "url"} <= set(src) for src in out["sources"])
    assert out["memory"]["messages"] == 2 and out["judge"] is None


def test_the_judge_runs_only_when_asked(config, chunks):
    s = session(config, chunks, judge=True)
    out = turn_summary(s.ask("Where is the data hosted?"), s, chunks)
    assert out["judge"] is not None and "grounded" in out["judge"]


def test_a_request_for_a_person_ends_the_call(config, chunks):
    s = session(config, chunks)
    t = s.ask("Get me a human now.")
    assert t["handoff"] and s.ended


def test_new_call_resets_everything(config, chunks):
    s = session(config, chunks)
    s.ask("Give me the exact price with a 40% discount.")
    s.reset()
    assert s.history == [] and s.parked == [] and s.turn_no == 0 and not s.ended


def test_web_page_has_the_chat_and_the_under_the_hood_panel():
    from deploykit.chat import PAGE
    assert "Under the hood" in PAGE and "/api/turn" in PAGE and "New call" in PAGE


def test_web_turn_endpoint_streams_json_lines(config, chunks):
    from deploykit.chat import ChatApp
    app = ChatApp(session(config, chunks))
    buf = io.BytesIO()
    app.turn({"text": "Where is the data hosted?"}, buf.write)
    lines = [json.loads(l) for l in buf.getvalue().decode().splitlines()]
    assert lines[0]["type"] in ("say", "filler") and lines[-1]["type"] == "done"
    assert lines[-1]["turn"]["memory"]["messages"] == 2


def test_web_reset_endpoint(config, chunks):
    from deploykit.chat import ChatApp
    app = ChatApp(session(config, chunks))
    app.turn({"text": "Hi"}, lambda b: None)
    assert app.reset() == {"ok": True} and app.session.turn_no == 0


def test_terminal_chat_with_mock(monkeypatch, capsys):
    from deploykit.cli import main
    answers = iter(["Where is the data hosted?", "/list", "/quit"])
    monkeypatch.setattr("builtins.input", lambda prompt="": next(answers))
    assert main(["chat", "--provider", "mock"]) == 0
    out = capsys.readouterr().out
    assert "AGENT" in out and "tier" in out and "Question list" in out
