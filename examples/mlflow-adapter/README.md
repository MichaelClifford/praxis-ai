# MLflow Traffic Mirror Adapter

Translates praxis-ai `traffic_mirror` envelopes into [MLflow] traces so
proxied LLM traffic can be explored, searched, and evaluated in the MLflow
UI. See `examples/configs/traffic-mirror.yaml` for the proxy side.

Each mirrored request becomes one trace: only the latest user turn (and any
following tool results) is shown as chat-message input, rather than repeating
the conversation history or request configuration. OpenAI Responses and Chat
Completions output is stored in MLflow's OpenAI message format, so assistant
text and model-requested function calls appear in the chat view. Model, status,
cluster, duration, and token usage are trace tags.
The proxy timestamp and duration set the trace timeline; token usage and
model are also MLflow LLM span attributes for cost estimates. Every trace gets
a session ID. The adapter prefers the client's `prompt_cache_key` or
`previous_response_id`, then uniquely matching conversation history across
models and providers. A new conversation receives a random ID: identical
opening prompts never automatically merge. Ambiguous or truncated histories
may split a real conversation rather than falsely merging two. Proxy request
IDs are trace metadata.

When a provider switch reformats assistant/tool messages, a unique prefix of
at least **two prior user turns** can also link the conversation across
providers. A single shared opening prompt is never enough evidence. Enable
`MIRROR_ADAPTER_LOG_SESSION_FIELDS=1` to log hashed match diagnostics (no
prompt text) when investigating an unexpected split.

Correlation is bounded to 256 recently active sessions (24-hour expiry) and
retains only hashes of message text, not an in-memory copy of prompts. Recent
fingerprints are recorded as MLflow trace tags so the adapter can try to
recover matches after a restart. Restoration is best-effort; if MLflow is
unavailable, or a trace was not yet persisted, new requests get new IDs.

The mirror observes provider HTTP traffic, not tool execution inside OpenCode.
It can display function calls requested by the model and tool results sent in
later requests, but it cannot create truthful tool-execution spans or measure
their execution times without instrumentation in the client.

## Run

```console
pip install -r requirements.txt
export MLFLOW_TRACKING_URI=http://localhost:5000   # your MLflow server
export MLFLOW_EXPERIMENT_NAME=praxis-ai-gateway    # optional
python mlflow_adapter.py                           # listens on 127.0.0.1:9090
```

## Proxy configuration

Add the `traffic_mirror` filter to your filter chain, declared before
`token_count` when you want token usage in the traces:

```yaml
- filter: traffic_mirror
  url: "http://127.0.0.1:9090/ingest"
  allow_private_addresses: true   # the adapter listens on loopback

- filter: token_count
  provider: openai
```

## Envelope

The adapter accepts the `praxis.traffic_mirror.v1` envelope POSTed by the
`traffic_mirror` filter. Bodies are UTF-8 strings in `body`, or base64 in
`body_b64` for binary payloads (e.g. multipart uploads).
Non-POST requests (such as OpenCode's GET upgrade probes) are acknowledged
without creating MLflow traces; the proxy may still mirror them to the adapter.
For SSE responses mirrored on a complete terminal event before transport EOF,
the trace tag `response_end_of_stream_observed=false` distinguishes them from
responses for which the proxy observed the final body callback.

Run adapter tests with `python -m unittest discover -p 'test_*.py'`.

To identify whether the client sends a stable session ID, temporarily run the
adapter with `MIRROR_ADAPTER_LOG_SESSION_FIELDS=1`. For each POST, it logs only
request/metadata field names and short hashes of candidate ID fields, not
field values or prompt content. Compare two turns in one session and a turn in
another session; disable the setting once the ID source is known.

[MLflow]: https://mlflow.org
