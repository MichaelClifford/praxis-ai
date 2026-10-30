# MLflow Traffic Mirror Adapter
#
# Translates praxis-ai `traffic_mirror` envelopes into MLflow traces so
# proxied LLM traffic can be explored, searched, and evaluated in the
# MLflow UI.
#
# Each mirrored request/response pair becomes one trace:
#   - request body  -> trace inputs (parsed JSON when possible)
#   - response body -> trace outputs (parsed JSON, or a reassembled
#     summary when the response is an OpenAI SSE stream, including
#     reasoning content and tool calls)
#   - model, status, cluster, duration, token usage -> trace tags
#   - HTTP status >= 400 marks the trace state as ERROR
#
# MLflow field mapping (3.x first-class trace fields):
#   - envelope `timestamp` / `duration_ms` -> trace request time and
#     execution duration, so the timeline reflects the proxied request
#     rather than adapter processing time
#   - envelope `token_*` (or usage in the response body) -> the root
#     span's `mlflow.chat.tokenUsage` attribute, which MLflow
#     aggregates into the trace-level token usage
#   - request `model` -> the root span's `mlflow.llm.model` attribute,
#     which MLflow prices via LiteLLM to populate trace cost (models
#     missing from the pricing catalog are left unpriced)
#   - the client's `prompt_cache_key` -> `mlflow.trace.session` metadata,
#     grouping traces per OpenCode session in the UI (no guessed fallback)
#   - envelope `request_id` -> trace metadata, for cross-referencing
#     proxy logs
#
# Run:
#   pip install -r requirements.txt
#   export MLFLOW_TRACKING_URI=http://localhost:5000   # your MLflow server
#   python mlflow_adapter.py                          # listens on :9090
#
# Point the proxy's traffic_mirror filter at this adapter:
#   filter: traffic_mirror
#   url: "http://127.0.0.1:9090/ingest"
#   allow_private_addresses: true
#
# Traces land in the experiment named by MLFLOW_EXPERIMENT_NAME
# (default: "praxis-ai-gateway"). Model names are extracted from the
# request body when it is OpenAI-shaped JSON.

import base64
import hashlib
import json
import logging
import os
from datetime import datetime, timezone

import mlflow
from flask import Flask, request
from mlflow import MlflowClient
from mlflow.entities import SpanType
from mlflow.tracing.constant import SpanAttributeKey
from session_resolver import (
    HISTORY_DIGEST_TAG, HISTORY_LENGTH_TAG, USER_DIGEST_TAG, USER_LENGTH_TAG, SessionResolver,
)

LISTEN_PORT = int(os.environ.get("MIRROR_ADAPTER_PORT", "9090"))
EXPERIMENT_NAME = os.environ.get("MLFLOW_EXPERIMENT_NAME", "praxis-ai-gateway")
LOG_SESSION_FIELDS = os.environ.get("MIRROR_ADAPTER_LOG_SESSION_FIELDS") == "1"

app = Flask(__name__)
app.logger.setLevel(logging.INFO)
client = MlflowClient()
experiment = mlflow.set_experiment(EXPERIMENT_NAME)
sessions = SessionResolver()
try:
    sessions.restore(client, experiment.experiment_id)
except Exception as error:
    app.logger.warning("Could not restore recent session matches from MLflow: %s", error)


def body_text(record) -> str:
    """Captured body as text, decoding base64 when needed."""
    if not isinstance(record, dict):
        return ""
    if isinstance(record.get("body"), str):
        return record["body"]
    if isinstance(record.get("body_b64"), str):
        return base64.b64decode(record["body_b64"]).decode("utf-8", errors="replace")
    return ""


def parse_json(text):
    """Parse JSON into a dict/list, or None when the text is not JSON."""
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return None


def reassemble_sse(text):
    """Reconstruct an OpenAI-shaped completion or Response from SSE events."""
    content_parts = []
    reasoning_parts = []
    tool_calls = {}
    response_items = {}
    completed_response = None
    finish_reason = None
    usage = None
    chunks = 0
    for line in text.splitlines():
        line = line.strip()
        if not line.startswith("data:"):
            continue
        payload = line[len("data:") :].strip()
        if not payload or payload == "[DONE]":
            continue
        event = parse_json(payload)
        if not isinstance(event, dict):
            continue
        chunks += 1
        if isinstance(event.get("usage"), dict) and event["usage"]:
            usage = event["usage"]
        # The Responses API emits named events rather than chat choices.
        if event.get("type") == "response.output_text.delta" and isinstance(event.get("delta"), str):
            content_parts.append(event["delta"])
        if event.get("type") == "response.output_item.done":
            if isinstance(event.get("output_index"), int) and isinstance(event.get("item"), dict):
                response_items[event["output_index"]] = event["item"]
        if event.get("type") == "response.completed":
            response = event.get("response")
            if isinstance(response, dict):
                completed_response = response
                if isinstance(response.get("usage"), dict):
                    usage = response["usage"]
        for choice in event.get("choices") or []:
            if not isinstance(choice, dict):
                continue
            if choice.get("finish_reason"):
                finish_reason = choice["finish_reason"]
            delta = choice.get("delta")
            if not isinstance(delta, dict):
                continue
            if isinstance(delta.get("content"), str):
                content_parts.append(delta["content"])
            if isinstance(delta.get("reasoning_content"), str):
                reasoning_parts.append(delta["reasoning_content"])
            for call in delta.get("tool_calls") or []:
                if not isinstance(call, dict) or not isinstance(call.get("index"), int):
                    continue
                current = tool_calls.setdefault(
                    call["index"], {"function": {"name": "", "arguments": ""}}
                )
                for field in ("id", "type"):
                    if isinstance(call.get(field), str):
                        current[field] = call[field]
                function = call.get("function")
                if isinstance(function, dict):
                    for field in ("name", "arguments"):
                        if isinstance(function.get(field), str):
                            current["function"][field] += function[field]
    if chunks == 0:
        return None
    if completed_response is not None or response_items:
        result = dict(completed_response or {})
        if not isinstance(result.get("output"), list) or not result["output"]:
            result["output"] = [response_items[index] for index in sorted(response_items)]
            if not result["output"] and content_parts:
                result["output"] = [{
                    "type": "message", "role": "assistant",
                    "content": [{"type": "output_text", "text": "".join(content_parts)}],
                }]
        if usage and "usage" not in result:
            result["usage"] = usage
        return result

    message = {"role": "assistant", "content": "".join(content_parts)}
    if reasoning_parts:
        message["reasoning_content"] = "".join(reasoning_parts)
    if tool_calls:
        message["tool_calls"] = [tool_calls[index] for index in sorted(tool_calls)]
    result = {"choices": [{"index": 0, "message": message, "finish_reason": finish_reason}]}
    if usage:
        result["usage"] = usage
    return result


def to_display(text):
    """Best shape for the MLflow trace UI: parsed JSON, assembled SSE, or raw text."""
    parsed = parse_json(text)
    if parsed is not None:
        return parsed
    return reassemble_sse(text) or text


def token_usage(envelope, response):
    """Map proxy or provider counts to MLflow's token usage attribute."""
    usage = response.get("usage") if isinstance(response, dict) else None
    usage = usage if isinstance(usage, dict) else {}
    fields = (
        ("input_tokens", "token_input", ("input_tokens", "prompt_tokens")),
        ("output_tokens", "token_output", ("output_tokens", "completion_tokens")),
        ("total_tokens", "token_total", ("total_tokens",)),
    )
    result = {}
    for target, proxy_key, provider_keys in fields:
        value = envelope.get(proxy_key)
        if value is None:
            value = next((usage[key] for key in provider_keys if key in usage), None)
        if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
            result[target] = value
    if "total_tokens" not in result and "input_tokens" in result and "output_tokens" in result:
        result["total_tokens"] = result["input_tokens"] + result["output_tokens"]
    return result


def session_field_diagnostics(request_body):
    """Report only field names and short hashes of possible client IDs.

    Do not log request values: bodies can contain prompts, tool outputs, and
    credentials. Hashes let us compare candidate IDs across turns without
    disclosing them in adapter logs.
    """
    if not isinstance(request_body, dict):
        return {"request_type": type(request_body).__name__}
    metadata = request_body.get("metadata")
    sources = {"request": request_body}
    if isinstance(metadata, dict):
        sources["metadata"] = metadata
    candidates = {}
    for location, fields in sources.items():
        for name in (
            "session_id", "conversation_id", "conversation", "opencode_session_id",
            "previous_response_id", "prompt_cache_key",
        ):
            value = fields.get(name)
            if isinstance(value, (str, int)) and not isinstance(value, bool) and value:
                candidates[f"{location}.{name}"] = hashlib.sha256(str(value).encode("utf-8")).hexdigest()[:12]
    return {
        "request_keys": sorted(request_body),
        "metadata_keys": sorted(metadata) if isinstance(metadata, dict) else [],
        "candidate_hashes": candidates,
    }


def trace_inputs(request_body):
    """Present the latest turn as chat messages, without request configuration."""
    if not isinstance(request_body, dict):
        return request_body
    messages = request_body.get("messages")
    if not isinstance(messages, list):
        messages = request_body.get("input")
    if isinstance(messages, str):
        return {"messages": [{"role": "user", "content": messages}]}
    if not isinstance(messages, list):
        return request_body
    last_user = next(
        (index for index in range(len(messages) - 1, -1, -1)
         if isinstance(messages[index], dict) and messages[index].get("role") == "user"),
        None,
    )
    turn = messages[last_user:] if last_user is not None else messages
    chat_messages = []
    for item in turn:
        if not isinstance(item, dict):
            continue
        if item.get("role") in ("system", "developer", "user", "assistant", "tool"):
            content = item.get("content")
            if isinstance(content, list):
                # Keep multimodal parts; collapse text-only Responses content
                # to make MLflow's request preview readable.
                parts = [
                    {**part, "type": "text"} if isinstance(part, dict) and part.get("type") == "input_text"
                    else part for part in content
                ]
                if all(isinstance(part, dict) and part.get("type") == "text"
                       and isinstance(part.get("text"), str) for part in parts):
                    content = "".join(part["text"] for part in parts)
                else:
                    content = parts
            message = {"role": item["role"], "content": content}
            for key in ("tool_calls", "tool_call_id"):
                if key in item:
                    message[key] = item[key]
            chat_messages.append(message)
        elif item.get("type") == "function_call_output":
            chat_messages.append({
                "role": "tool", "tool_call_id": item.get("call_id"), "content": item.get("output"),
            })
    return {"messages": chat_messages} if chat_messages else request_body


def chat_tools(request_body):
    """Normalize OpenAI Responses and Chat Completions tool definitions."""
    if not isinstance(request_body, dict) or not isinstance(request_body.get("tools"), list):
        return []
    tools = []
    for tool in request_body["tools"]:
        if not isinstance(tool, dict) or tool.get("type") != "function":
            continue
        if isinstance(tool.get("function"), dict):
            tools.append(tool)
        elif isinstance(tool.get("name"), str):
            function = {key: tool[key] for key in ("name", "description", "parameters") if key in tool}
            tools.append({"type": "function", "function": function})
    return tools


def start_time_ns(envelope):
    """Convert the proxy's RFC 3339 UTC timestamp to epoch nanoseconds."""
    value = envelope.get("timestamp")
    if not isinstance(value, str):
        raise ValueError("timestamp must be an RFC 3339 string")
    instant = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if instant.tzinfo is None:
        raise ValueError("timestamp must include a timezone")
    epoch = datetime(1970, 1, 1, tzinfo=timezone.utc)
    delta = instant - epoch
    return (delta.days * 86400 + delta.seconds) * 1_000_000_000 + delta.microseconds * 1000


@app.get("/healthz")
def healthz():
    return {"ok": True}


@app.post("/ingest")
def ingest():
    envelope = request.get_json(silent=True)
    if not isinstance(envelope, dict) or envelope.get("schema") != "praxis.traffic_mirror.v1":
        return {"error": "unsupported schema"}, 400
    if not isinstance(envelope.get("method"), str):
        return {"error": "invalid envelope: missing method"}, 400
    # The mirror observes every HTTP request. Clients may probe an inference
    # route with GET (including an attempted connection upgrade); those are
    # not LLM calls and should not appear as empty-input MLflow traces.
    if envelope.get("method") != "POST":
        app.logger.info(
            "Skipping non-inference mirror: method=%s path=%s status=%s request_id=%s",
            envelope["method"], envelope.get("path"), envelope.get("status"), envelope.get("request_id"),
        )
        return "", 204
    try:
        started = start_time_ns(envelope)
        duration = envelope["duration_ms"]
        status = envelope["status"]
        if (
            not isinstance(duration, int)
            or isinstance(duration, bool)
            or duration < 0
            or not isinstance(status, int)
            or isinstance(status, bool)
        ):
            raise ValueError("invalid duration or status")
        req_text = body_text(envelope.get("request"))
        resp_text = body_text(envelope.get("response"))
    except (ValueError, KeyError, TypeError, base64.binascii.Error) as error:
        return {"error": f"invalid envelope: {error}"}, 400

    req_parsed = parse_json(req_text)
    if LOG_SESSION_FIELDS:
        app.logger.info("Session field diagnostics: %s", session_field_diagnostics(req_parsed))
    output = to_display(resp_text)
    response_record = envelope.get("response") if isinstance(envelope.get("response"), dict) else {}
    model = req_parsed.get("model") if isinstance(req_parsed, dict) else None
    model = model if isinstance(model, str) and model else "unknown"
    tags = {
        "model": model,
        "http.method": str(envelope.get("method", "")),
        "http.path": str(envelope.get("path", "")),
        "http.status": str(status),
        "cluster": str(envelope.get("cluster") or ""),
        "duration_ms": str(duration),
        "truncated": str(
            any(
                isinstance(envelope.get(side), dict) and envelope[side].get("truncated")
                for side in ("request", "response")
            )
        ).lower(),
        "response_end_of_stream_observed": str(
            response_record.get("end_of_stream_observed", True)
        ).lower(),
    }
    counts = token_usage(envelope, output)
    for key, value in counts.items():
        proxy_key = {
            "input_tokens": "token_input",
            "output_tokens": "token_output",
            "total_tokens": "token_total",
        }[key]
        tags[proxy_key] = str(value)
    conversation, _, correlation_tags = sessions.resolve(req_parsed, output, resp_text)
    tags.update(correlation_tags)
    if LOG_SESSION_FIELDS:
        app.logger.info(
            "Session match: path=%s source=%s session_hash=%s history=%s/%s users=%s/%s",
            envelope.get("path"), correlation_tags["praxis.session.source"],
            hashlib.sha256(conversation.encode("utf-8")).hexdigest()[:12],
            correlation_tags.get(HISTORY_LENGTH_TAG, 0), correlation_tags.get(HISTORY_DIGEST_TAG, "")[:12],
            correlation_tags.get(USER_LENGTH_TAG, 0), correlation_tags.get(USER_DIGEST_TAG, "")[:12],
        )

    attributes = {SpanAttributeKey.MODEL: model}
    if counts:
        attributes[SpanAttributeKey.CHAT_USAGE] = counts
    chat_request = isinstance(req_parsed, dict) and ("input" in req_parsed or "messages" in req_parsed)
    if chat_request:
        attributes[SpanAttributeKey.MESSAGE_FORMAT] = "openai"
        if tools := chat_tools(req_parsed):
            attributes[SpanAttributeKey.CHAT_TOOLS] = tools
    metadata = {}
    if envelope.get("request_id") not in (None, "", "-"):
        metadata["praxis.request_id"] = str(envelope["request_id"])
    metadata["mlflow.trace.session"] = conversation

    # The client API creates a root span without requiring a live span context.
    # Trace context supplies immutable metadata when the trace is created.
    with mlflow.tracing.context(metadata=metadata):
        span = client.start_trace(
            name=f"{model} {envelope.get('path', '')}",
            span_type=SpanType.CHAT_MODEL if chat_request else SpanType.LLM,
            inputs=trace_inputs(req_parsed) if req_parsed is not None else req_text,
            attributes=attributes,
            tags=tags,
            start_time_ns=started,
        )
        client.end_trace(
            trace_id=span.trace_id,
            outputs=output,
            status="ERROR" if status >= 400 else "OK",
            end_time_ns=started + duration * 1_000_000,
        )
    return {"ok": True, "trace_id": span.trace_id}


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=LISTEN_PORT)
