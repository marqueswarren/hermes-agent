"""Regression tests for Argus 2026-11-14 findings.

1. Stale-stream recovery on the Anthropic Messages path tried to rebuild an
   OpenAI client (empty ``_client_kwargs``) and failed with "api_key client
   option must be set" every time, so a stalled first call simply waited out
   the HTTP read timeout (170-600 s per tick).
2. The stale-stream warning counted only ``messages`` (~438 tokens) and
   ignored ``system`` + ``tools`` (~31K real input).
3. Sonnet 5 / Opus 5 reject ``temperature`` with "is deprecated for this model".
"""
import socket
import threading
import time

import pytest

import run_agent
from agent.anthropic_adapter import build_anthropic_client, _forbids_sampling_params
from agent.auxiliary_client import _is_unsupported_temperature_error


def _stalled_sse_server():
    """Return (port, stop_event): sends SSE headers + pings, never real events."""
    srv = socket.socket()
    srv.bind(("127.0.0.1", 0))
    srv.listen(1)
    stop = threading.Event()

    def serve():
        conn, _ = srv.accept()
        conn.recv(65536)
        conn.sendall(b"HTTP/1.1 200 OK\r\ncontent-type: text/event-stream\r\n"
                     b"transfer-encoding: chunked\r\n\r\n")
        body = b"event: ping\ndata: {\"type\": \"ping\"}\n\n"
        while not stop.is_set():
            try:
                conn.sendall(b"%x\r\n%s\r\n" % (len(body), body))
            except OSError:
                return
            time.sleep(0.5)

    threading.Thread(target=serve, daemon=True).start()
    return srv.getsockname()[1], stop


def _bare_anthropic_agent(base_url):
    a = run_agent.AIAgent.__new__(run_agent.AIAgent)
    a.api_mode = "anthropic_messages"
    a.provider = "anthropic"
    a.model = "claude-sonnet-5-5"
    a.base_url = base_url
    a._anthropic_api_key = "sk-ant-api03-test"
    a._anthropic_base_url = base_url
    a._client_kwargs = {}
    a.client = None
    a._anthropic_client = build_anthropic_client(a._anthropic_api_key, base_url)
    return a


def test_replace_primary_client_on_anthropic_aborts_stalled_stream_and_rebuilds():
    port, stop = _stalled_sse_server()
    try:
        agent = _bare_anthropic_agent(f"http://127.0.0.1:{port}")
        old = agent._anthropic_client
        out = {}

        def worker():
            try:
                with old.messages.stream(model="claude-sonnet-5-5", max_tokens=8,
                                         messages=[{"role": "user", "content": "hi"}]) as s:
                    for _ in s:
                        pass
                out["result"] = "completed"
            except Exception as e:  # noqa: BLE001
                out["result"] = type(e).__name__

        t = threading.Thread(target=worker, daemon=True)
        t.start()
        time.sleep(1.5)

        assert agent._replace_primary_openai_client(reason="stale_stream_pool_cleanup") is True
        t.join(timeout=5)

        assert not t.is_alive(), "stalled stream was not aborted"
        assert out["result"] in {"ReadError", "RemoteProtocolError"}
        assert agent._anthropic_client is not old
    finally:
        stop.set()


def test_estimate_includes_system_and_tools():
    kwargs = {
        "system": "s" * 100_000,
        "messages": [{"role": "user", "content": "u" * 1_700}],
        "tools": [{"name": "t" * 20_000}],
    }
    est = run_agent.AIAgent._estimate_api_kwargs_tokens(kwargs)
    messages_only = sum(len(str(m)) for m in kwargs["messages"]) // 4
    assert messages_only < 500
    assert est > 30_000


@pytest.mark.parametrize("model,expected", [
    ("claude-sonnet-5-5", True),
    ("claude-opus-5-5", True),
    ("claude-opus-4-7", True),
    ("claude-sonnet-4-6", False),
    ("claude-haiku-4-5", False),
])
def test_forbids_sampling_params_covers_5_family(model, expected):
    assert _forbids_sampling_params(model) is expected


def test_deprecated_temperature_400_is_retryable():
    err = Exception("Error code: 400 - {'type': 'error', 'error': {'type': "
                    "'invalid_request_error', 'message': '`temperature` is "
                    "deprecated for this model.'}}")
    assert _is_unsupported_temperature_error(err) is True


def test_title_generator_does_not_send_temperature(monkeypatch):
    from agent import title_generator

    seen = {}

    class _Msg:
        content = "A Title"

    class _Choice:
        message = _Msg()

    class _Resp:
        choices = [_Choice()]

    def fake_call_llm(**kwargs):
        seen.update(kwargs)
        return _Resp()

    monkeypatch.setattr(title_generator, "call_llm", fake_call_llm)
    assert title_generator.generate_title("hello", "world") == "A Title"
    assert "temperature" not in seen
