"""Adapter tests; run with .venv/bin/python -m unittest discover -p 'test_*.py'."""

import os
import tempfile
import unittest
from datetime import datetime, timezone
from unittest.mock import patch

import mlflow

_tracking = tempfile.TemporaryDirectory()
os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
mlflow.set_tracking_uri(_tracking.name)
import mlflow_adapter as adapter  # noqa: E402 - set the test tracking store before import
from session_resolver import SessionResolver


class AdapterTests(unittest.TestCase):
    def test_chat_sse_reassembles_reasoning_and_tool_calls(self):
        events = [
            {"choices": [{"delta": {"reasoning_content": "think", "tool_calls": [
                {"index": 0, "id": "call_1", "type": "function", "function": {"name": "look", "arguments": '{"q":'}}
            ]}}]},
            {"choices": [{"delta": {"content": "done", "tool_calls": [
                {"index": 0, "function": {"name": "up", "arguments": '"hi"}'}}
            ]}, "finish_reason": "tool_calls"}]},
            {"usage": {"prompt_tokens": 3, "completion_tokens": 4, "total_tokens": 7}, "choices": []},
        ]
        stream = "\n\n".join(f"data: {adapter.json.dumps(event)}" for event in events) + "\n\ndata: [DONE]\n"
        result = adapter.to_display(stream)
        message = result["choices"][0]["message"]
        self.assertEqual(message["content"], "done")
        self.assertEqual(message["reasoning_content"], "think")
        self.assertEqual(message["tool_calls"], [{
            "id": "call_1", "type": "function", "function": {"name": "lookup", "arguments": '{"q":"hi"}'}
        }])
        self.assertEqual(result["usage"]["total_tokens"], 7)

    def test_responses_sse_and_token_fallback(self):
        stream = '\n'.join([
            'data: {"type":"response.output_text.delta","delta":"hi"}',
            'data: {"type":"response.completed","response":{"usage":{"input_tokens":2,"output_tokens":3}}}',
        ])
        result = adapter.to_display(stream)
        self.assertEqual(result["output"][0]["content"][0]["text"], "hi")
        self.assertEqual(adapter.token_usage({}, result), {
            "input_tokens": 2, "output_tokens": 3, "total_tokens": 5
        })

    def test_responses_tool_call_uses_openai_chat_format(self):
        completion = {
            "id": "resp_1", "object": "response", "model": "gpt-4o", "status": "completed",
            "output": [
                {"type": "message", "role": "assistant", "content": [
                    {"type": "output_text", "text": "Looking up the answer"}
                ]},
                {"type": "function_call", "call_id": "call_1", "name": "lookup", "arguments": '{"q":"hi"}'},
            ],
            "usage": {"input_tokens": 2, "output_tokens": 3, "total_tokens": 5},
        }
        stream = "event: response.completed\ndata: " + adapter.json.dumps({
            "type": "response.completed", "response": completion
        }) + "\n\n"
        self.assertEqual(adapter.to_display(stream), completion)
        request = {
            "model": "gpt-4o", "prompt_cache_key": "session-tools", "input": [
                {"role": "user", "content": [{"type": "input_text", "text": "Find this"}]}
            ],
            "tools": [{"type": "function", "name": "lookup", "parameters": {"type": "object"}}],
        }
        envelope = {
            "schema": "praxis.traffic_mirror.v1", "timestamp": "2026-10-16T12:00:00.123Z",
            "duration_ms": 10, "method": "POST", "path": "/v1/responses", "status": 200,
            "request": {"body": adapter.json.dumps(request)}, "response": {"body": stream},
        }
        response = adapter.app.test_client().post("/ingest", json=envelope)
        self.assertEqual(response.status_code, 200, response.json)
        mlflow.flush_trace_async_logging()
        trace = adapter.client.get_trace(response.json["trace_id"])
        root = trace.data.spans[0]
        self.assertEqual(root.inputs, {"messages": [{"role": "user", "content": "Find this"}]})
        self.assertEqual(trace.info.request_preview, "Find this")
        self.assertEqual(trace.info.response_preview, "Looking up the answer")
        self.assertEqual(root.outputs["output"][1], completion["output"][1])
        self.assertEqual(root.attributes[adapter.SpanAttributeKey.MESSAGE_FORMAT], "openai")
        self.assertEqual(root.attributes[adapter.SpanAttributeKey.CHAT_TOOLS], [{
            "type": "function", "function": {"name": "lookup", "parameters": {"type": "object"}}
        }])

    def test_tool_result_in_responses_request_is_chat_message(self):
        request = {"input": [{"type": "function_call_output", "call_id": "call_1", "output": "found"}]}
        self.assertEqual(adapter.trace_inputs(request), {
            "messages": [{"role": "tool", "tool_call_id": "call_1", "content": "found"}]
        })

    def test_multimodal_input_preserves_image_and_normalizes_text(self):
        request = {"input": [{"role": "user", "content": [
            {"type": "input_text", "text": "Describe this"},
            {"type": "input_image", "image_url": "data:image/png;base64,aGVsbG8="},
        ]}]}
        content = adapter.trace_inputs(request)["messages"][0]["content"]
        self.assertEqual(content[0], {"type": "text", "text": "Describe this"})
        self.assertEqual(content[1], request["input"][0]["content"][1])

    def test_session_uses_client_cache_key_not_message_prefix(self):
        resolver = SessionResolver()
        first = [{"role": "system", "content": "help"}, {"role": "user", "content": "hello"}]
        later = first + [{"role": "assistant", "content": "hi"}, {"role": "user", "content": "again"}]
        self.assertEqual(resolver.resolve({"messages": first, "prompt_cache_key": "session-a"}, {})[0], "session-a")
        self.assertEqual(resolver.resolve({"messages": later, "prompt_cache_key": "session-a"}, {})[0], "session-a")
        self.assertEqual(resolver.resolve({"messages": first, "prompt_cache_key": "session-b"}, {})[0], "session-b")
        for invalid in (None, "  ", 42):
            generated, _, _ = resolver.resolve({"messages": first, "prompt_cache_key": invalid}, {})
            self.assertTrue(generated.startswith("praxis-"))

    def test_session_diagnostics_do_not_log_request_values(self):
        request = {
            "model": "gpt-4o", "input": "private prompt",
            "metadata": {"session_id": "private session id", "extra": "private metadata"},
            "previous_response_id": "private previous id",
            "prompt_cache_key": "private cache key",
        }
        result = adapter.session_field_diagnostics(request)
        rendered = repr(result)
        for secret in (
            "private prompt", "private session id", "private metadata", "private previous id", "private cache key",
        ):
            self.assertNotIn(secret, rendered)
        self.assertEqual(result["metadata_keys"], ["extra", "session_id"])
        self.assertEqual(len(result["candidate_hashes"]["metadata.session_id"]), 12)
        self.assertEqual(len(result["candidate_hashes"]["request.prompt_cache_key"]), 12)
        self.assertEqual(result["candidate_hashes"], adapter.session_field_diagnostics(request)["candidate_hashes"])

    def test_trace_input_contains_only_latest_user_turn(self):
        prefix = [{"role": "system", "content": "help"}, {"role": "user", "content": "hello"}]
        later = prefix + [
            {"role": "assistant", "content": "hi"},
            {"role": "user", "content": "again"},
            {"role": "tool", "content": "result"},
        ]
        request = {"model": "gpt-4o", "messages": later, "prompt_cache_key": "session-a"}
        self.assertEqual(adapter.trace_inputs(request), {
            "messages": later[-2:]
        })
        self.assertEqual(request["messages"], later, "do not mutate the original request")
        response_request = {"model": "gpt-4o", "input": later}
        self.assertEqual(adapter.trace_inputs(response_request)["messages"], later[-2:])
        self.assertEqual(adapter.trace_inputs({"input": "new message"}), {
            "messages": [{"role": "user", "content": "new message"}]
        })
        self.assertEqual(adapter.trace_inputs({"messages": prefix[:1]}), {"messages": prefix[:1]})

    def test_ingest_persists_timeline_metadata_and_span_attributes(self):
        envelope = {
            "schema": "praxis.traffic_mirror.v1", "timestamp": "2026-10-16T12:00:00.123Z",
            "duration_ms": 145, "request_id": "req-123", "method": "POST",
            "path": "/v1/chat/completions", "cluster": "backend", "status": 200,
            "token_input": 11, "token_output": 4, "token_total": 15,
            "request": {"body": '{"model":"gpt-4o","prompt_cache_key":"session-a","messages":[{"role":"user","content":"hi"}]}'},
            "response": {"body": '{"usage":{"prompt_tokens":1,"completion_tokens":2}}'},
        }
        response = adapter.app.test_client().post("/ingest", json=envelope)
        self.assertEqual(response.status_code, 200, response.json)
        mlflow.flush_trace_async_logging()
        trace = adapter.client.get_trace(response.json["trace_id"])
        self.assertEqual(trace.info.request_time, 1792152000123)
        self.assertEqual(trace.info.execution_duration, 145)
        self.assertEqual(trace.info.trace_metadata["praxis.request_id"], "req-123")
        self.assertEqual(trace.info.trace_metadata["mlflow.trace.session"], "session-a")
        root = trace.data.spans[0]
        self.assertEqual(root.span_type, "CHAT_MODEL")
        self.assertEqual(root.attributes[adapter.SpanAttributeKey.MODEL], "gpt-4o")
        self.assertEqual(root.attributes[adapter.SpanAttributeKey.MESSAGE_FORMAT], "openai")
        self.assertEqual(root.inputs, {"messages": [{"role": "user", "content": "hi"}]})
        self.assertEqual(trace.info.request_preview, "hi")
        self.assertEqual(root.attributes[adapter.SpanAttributeKey.CHAT_USAGE], {
            "input_tokens": 11, "output_tokens": 4, "total_tokens": 15
        })

    def test_identical_prompts_in_different_client_sessions_do_not_merge(self):
        envelope = {
            "schema": "praxis.traffic_mirror.v1", "timestamp": "2026-10-16T12:00:00.123Z",
            "duration_ms": 10, "method": "POST", "path": "/v1/responses", "status": 200,
            "response": {"body": "{}"},
        }
        inferred = None
        for cache_key in ("session-one", "session-two", None):
            body = {"model": "gpt-4o", "input": [{"role": "user", "content": "same opening prompt"}]}
            if cache_key is not None:
                body["prompt_cache_key"] = cache_key
            envelope["request"] = {"body": adapter.json.dumps(body)}
            response = adapter.app.test_client().post("/ingest", json=envelope)
            self.assertEqual(response.status_code, 200, response.json)
            mlflow.flush_trace_async_logging()
            trace = adapter.client.get_trace(response.json["trace_id"])
            session = trace.info.trace_metadata.get("mlflow.trace.session")
            if cache_key is None:
                self.assertTrue(session.startswith("praxis-"))
                inferred = session
            else:
                self.assertEqual(session, cache_key)
        self.assertNotIn(inferred, ("session-one", "session-two"))
        experiment = mlflow.get_experiment_by_name(adapter.EXPERIMENT_NAME)
        sessions = mlflow.search_sessions(locations=[experiment.experiment_id])
        self.assertTrue({"session-one", "session-two"}.issubset({session.id for session in sessions}))

    def test_invalid_envelope_does_not_create_trace(self):
        with patch.object(adapter.client, "start_trace") as start:
            response = adapter.app.test_client().post("/ingest", json={"schema": "praxis.traffic_mirror.v1"})
        self.assertEqual(response.status_code, 400)
        start.assert_not_called()

    def test_truncated_request_still_receives_a_session_id(self):
        envelope = {
            "schema": "praxis.traffic_mirror.v1", "timestamp": datetime.now(timezone.utc).isoformat(),
            "duration_ms": 1, "method": "POST", "path": "/v1/chat/completions", "status": 200,
            "request": {"body": '{"messages": [', "truncated": True},
            "response": {"body": '{}'},
        }
        posted = adapter.app.test_client().post("/ingest", json=envelope)
        self.assertEqual(posted.status_code, 200, posted.json)
        mlflow.flush_trace_async_logging()
        trace = adapter.client.get_trace(posted.json["trace_id"])
        self.assertTrue(trace.info.trace_metadata["mlflow.trace.session"].startswith("praxis-"))
        self.assertEqual(trace.info.tags["praxis.session.source"], "new")

    def test_terminal_sse_record_preserves_end_of_stream_indicator(self):
        envelope = {
            "schema": "praxis.traffic_mirror.v1", "method": "POST",
            "timestamp": "2026-10-16T12:00:00.123Z", "duration_ms": 145,
            "path": "/v1/responses", "status": 200,
            "request": {"body": '{"model":"gpt-4o","input":"hi"}'},
            "response": {
                "body": 'event: response.completed\ndata: {"type":"response.completed"}\n\n',
                "end_of_stream_observed": False,
            },
        }
        response = adapter.app.test_client().post("/ingest", json=envelope)
        self.assertEqual(response.status_code, 200, response.json)
        mlflow.flush_trace_async_logging()
        trace = adapter.client.get_trace(response.json["trace_id"])
        self.assertEqual(trace.info.tags["response_end_of_stream_observed"], "false")

    def test_restart_restores_session_from_mlflow_tags(self):
        envelope = {
            "schema": "praxis.traffic_mirror.v1", "timestamp": datetime.now(timezone.utc).isoformat(),
            "duration_ms": 10, "method": "POST", "path": "/v1/chat/completions", "status": 200,
            "request": {"body": adapter.json.dumps({
                "model": "model-a", "messages": [{"role": "user", "content": "restart example"}],
            })},
            "response": {"body": adapter.json.dumps({
                "choices": [{"message": {"role": "assistant", "content": "reply"}}],
            })},
        }
        posted = adapter.app.test_client().post("/ingest", json=envelope)
        self.assertEqual(posted.status_code, 200, posted.json)
        mlflow.flush_trace_async_logging()
        trace = adapter.client.get_trace(posted.json["trace_id"])
        recovered = SessionResolver()
        recovered.restore(adapter.client, adapter.experiment.experiment_id)
        followup = {"model": "different-model", "messages": [
            {"role": "user", "content": "restart example"},
            {"role": "assistant", "content": "reply"},
            {"role": "user", "content": "second turn"},
        ]}
        matched, source, _ = recovered.resolve(followup, {"choices": []})
        self.assertEqual((matched, source), (trace.info.trace_metadata["mlflow.trace.session"], "history"))

    def test_get_upgrade_probe_is_accepted_but_not_traced(self):
        with patch.object(adapter.client, "start_trace") as start:
            response = adapter.app.test_client().post("/ingest", json={
                "schema": "praxis.traffic_mirror.v1",
                "method": "GET",
                "path": "/v1/responses",
                "status": 400,
                "request": {"body": ""},
                "response": {"body": "error:connection upgrade not supported on this route"},
            })
        self.assertEqual(response.status_code, 204)
        start.assert_not_called()


if __name__ == "__main__":
    unittest.main()
