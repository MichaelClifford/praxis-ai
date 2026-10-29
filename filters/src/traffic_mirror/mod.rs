// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Traffic mirror filter: captures bounded, read-only copies of the request
//! and response bodies and delivers one JSON record per request to an
//! external collector over HTTP for analysis.
//!
//! The filter exists to feed traffic-analysis pipelines (for example an
//! [MLflow] adapter that turns proxied LLM calls into traces). It is a pure
//! observer: bodies stream through untouched, nothing is mutated, and the
//! delivery POST runs on a background task after the response completes, so
//! a slow or down collector never adds latency to, or fails, the proxied
//! request.
//!
//! [MLflow]: https://mlflow.org
//!
//! # Capture semantics
//!
//! * Bodies are observed chunk-by-chunk in `BodyMode::Stream` with [`BodyAccess::ReadOnly`]; the proxy never buffers a
//!   body it would not buffer otherwise. Each body is copied into a bounded buffer of `max_body_bytes` (default 1 MiB,
//!   applied separately to request and response). Bytes past the cap are counted but not captured, and the record
//!   carries `truncated: true`.
//! * Request and response headers are deliberately **not** captured: the proxy injects upstream credentials (see
//!   `credential_injection`), and mirroring headers would leak them to the collector.
//! * Bodies that are not valid UTF-8 (e.g. multipart file uploads) are base64-encoded into `body_b64` instead of
//!   `body`.
//! * Bodyless responses (204, 304, `HEAD`, 1xx) never reach a body phase, so the record is emitted from the response
//!   phase with an empty response body. SSE responses emit after the complete terminal event, even if the transport
//!   closes without a final body callback. Records emitted before that callback carry `response.end_of_stream_observed:
//!   false` (trailing bytes and `token_count` metadata may not be available yet).
//!
//! # Envelope
//!
//! Each delivery is a `POST` of one JSON document:
//!
//! ```json
//! {
//!   "schema": "praxis.traffic_mirror.v1",
//!   "timestamp": "2026-10-16T12:00:00.000Z",
//!   "request_id": "…",
//!   "method": "POST",
//!   "path": "/v1/chat/completions",
//!   "cluster": "backend",
//!   "status": 200,
//!   "duration_ms": 1234,
//!   "token_input": 100,
//!   "token_output": 50,
//!   "token_total": 150,
//!   "request":  { "body": "…", "body_bytes": 2048, "truncated": false },
//!   "response": { "body": "…", "body_bytes": 8192, "truncated": false }
//! }
//! ```
//!
//! The `token_*` fields appear only when a `token_count` filter ran in the
//! same chain: response hooks execute in reverse declaration order, so
//! declaring `traffic_mirror` **before** `token_count` lets this filter read
//! the usage metadata `token_count` publishes at end-of-stream. An SSE record
//! emitted on its terminal event before end-of-stream may omit `token_*`; the
//! response body still contains the provider's terminal usage event. The same reverse-order rule
//! makes `traffic_mirror` see the final response status when declared
//! before the `load_balancer`.
//!
//! # YAML
//!
//! ```yaml
//! filter: traffic_mirror
//! url: "http://collector:9090/ingest"
//! timeout_seconds: 5             # optional, default 5
//! max_body_bytes: 1048576        # optional, default 1 MiB per body
//! allow_private_addresses: false # optional, default false
//! ```

mod config;

#[cfg(test)]
#[expect(clippy::allow_attributes, reason = "blanket test suppressions")]
#[allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::indexing_slicing,
    clippy::panic,
    reason = "tests"
)]
mod tests;

use std::time::{Duration, Instant};

use async_trait::async_trait;
use base64::Engine as _;
use bytes::Bytes;
use http::header::CONTENT_TYPE;
use metrics::counter;
use praxis_ai_apis::{
    callout_target::AddressPolicy,
    subrequest::{self, SubRequest, SubRequestClient, SubRequestError, SubResponse},
};
use praxis_filter::{
    BodyAccess, BodyMode, FilterAction, FilterError, HttpFilter, HttpFilterContext, bodyless_response,
    parse_filter_config,
};
use tracing::{debug, warn};

use self::config::{TrafficMirrorConfig, validate_config};
use crate::token_usage::{META_TOKEN_INPUT, META_TOKEN_OUTPUT, META_TOKEN_TOTAL};

/// Envelope schema identifier. The collector branches on this field, so it
/// changes only when the envelope shape changes.
const SCHEMA: &str = "praxis.traffic_mirror.v1";

/// Counter incremented whenever a mirror record fails to reach the
/// collector (transport failure or non-2xx acknowledgement).
const METRIC_DELIVERY_FAILURES: &str = "praxis_ai_traffic_mirror_delivery_failures_total";

/// Upper bound on collector acknowledgement bodies. Acks are tiny JSON
/// documents or empty 2xx responses; anything larger indicates a
/// misbehaving collector and is cut off.
const MAX_ACK_BYTES: usize = 64 * 1024;

/// Keepalive pool size for the private per-filter sub-request connector
/// created by [`TrafficMirrorFilter::from_config`].
const PRIVATE_POOL_SIZE: usize = 4;

// -----------------------------------------------------------------------------
// TrafficMirrorFilter
// -----------------------------------------------------------------------------

/// Mirrors bounded request/response bodies to an external collector.
///
/// See the [module documentation](self) for capture semantics and the
/// envelope shape.
pub struct TrafficMirrorFilter {
    /// Connect-time policy for the collector endpoint. Rejects non-public
    /// addresses unless the operator sets `allow_private_addresses`.
    address_policy: AddressPolicy,

    /// Shared or private sub-request client used for deliveries.
    client: SubRequestClient,

    /// Per-body capture cap in bytes.
    max_body_bytes: usize,

    /// Absolute collector URL the envelopes are delivered to.
    target_url: String,

    /// One-budget delivery timeout (DNS + connect + I/O).
    timeout: Duration,
}

impl TrafficMirrorFilter {
    /// Create from parsed YAML config with a private sub-request client.
    ///
    /// # Errors
    ///
    /// Returns [`FilterError`] when the config is invalid; see
    /// [`validate_config`].
    pub fn from_config(config: &serde_yaml::Value) -> Result<Box<dyn HttpFilter>, FilterError> {
        let client = crate::isolated_subrequest_client(PRIVATE_POOL_SIZE);
        Ok(Box::new(Self::build(config, client)?))
    }

    /// Create from parsed YAML config using the shared [`SubRequestClient`].
    ///
    /// # Errors
    ///
    /// Returns [`FilterError`] when the config is invalid; see
    /// [`validate_config`].
    pub fn from_config_with_client(
        config: &serde_yaml::Value,
        client: SubRequestClient,
    ) -> Result<Box<dyn HttpFilter>, FilterError> {
        Ok(Box::new(Self::build(config, client)?))
    }

    /// Construct the filter from config and a sub-request client.
    fn build(config: &serde_yaml::Value, client: SubRequestClient) -> Result<Self, FilterError> {
        let cfg: TrafficMirrorConfig = parse_filter_config("traffic_mirror", config)?;
        validate_config(&cfg)?;
        Ok(Self {
            address_policy: AddressPolicy::from_allow_private(cfg.allow_private_addresses),
            client,
            max_body_bytes: cfg.max_body_bytes,
            target_url: cfg.url,
            timeout: Duration::from_secs(cfg.timeout_seconds),
        })
    }

    /// Build the envelope and hand delivery to a background task.
    ///
    /// Called at most once per request: from the response phase for
    /// bodyless responses and from the response body phase otherwise.
    fn emit(&self, ctx: &HttpFilterContext<'_>, state: &mut MirrorState, end_of_stream_observed: bool) {
        let Some(body) = serialize_envelope(ctx, state, end_of_stream_observed) else {
            return;
        };
        debug!(
            request_id = ctx.request_id().unwrap_or("-"),
            method = %state.method,
            path = %state.path,
            status = state.status.unwrap_or(0),
            request_body_bytes = ctx.request_body_bytes,
            response_body_bytes = ctx.response_body_bytes,
            end_of_stream_observed,
            "traffic_mirror: queuing record"
        );
        // Drop only reports responses that never reached an emission point.
        state.status = None;
        spawn_delivery(
            self.client.clone(),
            self.target_url.clone(),
            self.timeout,
            self.address_policy,
            body,
        );
    }
}

/// Serialize a record, marking terminal-event emission if transport EOF was
/// not observed. Fail soft if serialization unexpectedly fails.
fn serialize_envelope(
    ctx: &HttpFilterContext<'_>,
    state: &MirrorState,
    end_of_stream_observed: bool,
) -> Option<Vec<u8>> {
    let mut envelope = build_envelope(ctx, state);
    if !end_of_stream_observed
        && let Some(response) = envelope.get_mut("response").and_then(serde_json::Value::as_object_mut)
    {
        response.insert("end_of_stream_observed".into(), serde_json::json!(false));
    }
    match serde_json::to_vec(&envelope) {
        Ok(body) => Some(body),
        Err(error) => {
            warn!(%error, "traffic_mirror: envelope serialization failed");
            None
        },
    }
}

#[async_trait]
impl HttpFilter for TrafficMirrorFilter {
    fn name(&self) -> &'static str {
        "traffic_mirror"
    }

    async fn on_request(&self, ctx: &mut HttpFilterContext<'_>) -> Result<FilterAction, FilterError> {
        // State is keyed by filter id so multiple instances in one chain
        // (or the same filter in nested contexts) never collide.
        let key = filter_state_key(ctx.current_filter_id);
        debug!(
            request_id = ctx.request_id().unwrap_or("-"),
            method = %ctx.request.method,
            path = ctx.request.uri.path(),
            "traffic_mirror: observing request"
        );
        ctx.filter_state.insert(key, Box::new(MirrorState::new(ctx)));
        Ok(FilterAction::Continue)
    }

    fn request_body_access(&self) -> BodyAccess {
        BodyAccess::ReadOnly
    }

    fn request_body_mode(&self) -> BodyMode {
        BodyMode::Stream
    }

    async fn on_request_body(
        &self,
        ctx: &mut HttpFilterContext<'_>,
        body: &mut Option<Bytes>,
        _end_of_stream: bool,
    ) -> Result<FilterAction, FilterError> {
        if let (Some(chunk), Some(state)) = (body.as_ref(), state_mut(ctx)) {
            state.capture_request_chunk(chunk, self.max_body_bytes);
        }
        Ok(FilterAction::Continue)
    }

    async fn on_response(&self, ctx: &mut HttpFilterContext<'_>) -> Result<FilterAction, FilterError> {
        let Some(status) = ctx.response_header.as_ref().map(|r| r.status.as_u16()) else {
            return Ok(FilterAction::Continue);
        };
        let Some(mut state) = take_state(ctx) else {
            return Ok(FilterAction::Continue);
        };
        state.status = Some(status);
        state.sse_terminal = ctx
            .response_header
            .as_ref()
            .and_then(|r| r.headers.get(CONTENT_TYPE))
            .and_then(|value| value.to_str().ok())
            .and_then(|value| value.split(';').next())
            .is_some_and(|value| value.trim().eq_ignore_ascii_case("text/event-stream"))
            .then(SseTerminal::default);
        debug!(
            request_id = ctx.request_id().unwrap_or("-"),
            status, "traffic_mirror: received response headers"
        );

        // Bodyless responses never reach a body phase; emit from here so
        // they are still mirrored.
        let bodyless = ctx
            .response_header
            .as_ref()
            .is_some_and(|r| bodyless_response(r.status, &ctx.request.method));
        if bodyless {
            self.emit(ctx, &mut state, true);
        } else {
            ctx.filter_state
                .insert(filter_state_key(ctx.current_filter_id), Box::new(state));
        }
        Ok(FilterAction::Continue)
    }

    fn response_body_access(&self) -> BodyAccess {
        BodyAccess::ReadOnly
    }

    fn response_body_mode(&self) -> BodyMode {
        BodyMode::Stream
    }

    fn on_response_body(
        &self,
        ctx: &mut HttpFilterContext<'_>,
        body: &mut Option<Bytes>,
        end_of_stream: bool,
    ) -> Result<FilterAction, FilterError> {
        let Some(mut state) = take_state(ctx) else {
            return Ok(FilterAction::Continue);
        };
        if let Some(chunk) = body.as_ref() {
            state.capture_response_chunk(chunk, self.max_body_bytes);
            if let Some(terminal) = state.sse_terminal.as_mut() {
                terminal.observe(chunk);
            }
        }
        if !end_of_stream && !state.sse_terminal.as_ref().is_some_and(SseTerminal::is_complete) {
            // Re-insert: the body phase calls this hook per chunk.
            ctx.filter_state
                .insert(filter_state_key(ctx.current_filter_id), Box::new(state));
            return Ok(FilterAction::Continue);
        }
        self.emit(ctx, &mut state, end_of_stream);
        Ok(FilterAction::Continue)
    }
}

// -----------------------------------------------------------------------------
// Per-Request State
// -----------------------------------------------------------------------------

/// Request identity, timing, and captured bodies for one mirrored request.
struct MirrorState {
    /// Request method, captured at the request phase.
    method: String,

    /// Request path (no query), captured at the request phase.
    path: String,

    /// Cluster the router selected, when this filter runs after it.
    cluster: Option<String>,

    /// Response status, captured at the response phase. Cleared after
    /// emission so Drop can identify a response that never emitted.
    status: Option<u16>,

    /// When the request phase started, for `duration_ms`.
    started: Instant,

    /// Bounded copy of the request body.
    request_body: Vec<u8>,

    /// Whether the request body exceeded the capture cap.
    request_truncated: bool,

    /// Bounded copy of the response body.
    response_body: Vec<u8>,

    /// Whether the response body exceeded the capture cap.
    response_truncated: bool,

    /// Incremental SSE line scanner; allows emission on the complete
    /// terminal event when the body hook never receives end-of-stream.
    sse_terminal: Option<SseTerminal>,
}

impl MirrorState {
    /// Capture request identity and start the request timer.
    fn new(ctx: &HttpFilterContext<'_>) -> Self {
        Self {
            method: ctx.request.method.as_str().to_owned(),
            path: ctx.request.uri.path().to_owned(),
            cluster: ctx.cluster_name().map(str::to_owned),
            status: None,
            started: Instant::now(),
            request_body: Vec::new(),
            request_truncated: false,
            response_body: Vec::new(),
            response_truncated: false,
            sse_terminal: None,
        }
    }

    /// Copy one request body chunk into the bounded buffer.
    fn capture_request_chunk(&mut self, chunk: &[u8], cap: usize) {
        capture_into(&mut self.request_body, &mut self.request_truncated, chunk, cap);
    }

    /// Copy one response body chunk into the bounded buffer.
    fn capture_response_chunk(&mut self, chunk: &[u8], cap: usize) {
        capture_into(&mut self.response_body, &mut self.response_truncated, chunk, cap);
    }
}

/// Recognize a terminal SSE event only after its blank-line delimiter. A
/// small line buffer handles markers split across chunks without repeatedly
/// scanning the captured body or buffering an unbounded `data:` line.
struct SseTerminal {
    /// Partial SSE line across body chunks, capped to prevent unbounded data buffering.
    line: Vec<u8>,
    /// Ignore a line too long to contain a terminal marker.
    oversized_line: bool,
    /// Whether a terminal event has been identified and delimited.
    phase: SsePhase,
}

/// Scanner state for the current SSE event.
#[derive(Default, PartialEq, Eq)]
enum SsePhase {
    /// No terminal event has been seen.
    #[default]
    Reading,
    /// A terminal event line has been seen but its data may still be arriving.
    PendingTerminal,
    /// The terminal event's blank-line delimiter has been seen.
    Complete,
}

impl Default for SseTerminal {
    fn default() -> Self {
        Self {
            line: Vec::new(),
            oversized_line: false,
            phase: SsePhase::Reading,
        }
    }
}

impl SseTerminal {
    /// Inspect incoming bytes without buffering the full SSE event.
    fn observe(&mut self, chunk: &[u8]) {
        const MAX_LINE_BYTES: usize = 256;
        for &byte in chunk {
            if self.is_complete() {
                break;
            }
            if byte == b'\n' {
                if !self.oversized_line {
                    self.finish_line();
                }
                self.line.clear();
                self.oversized_line = false;
            } else if !self.oversized_line {
                if self.line.len() < MAX_LINE_BYTES {
                    self.line.push(byte);
                } else {
                    self.line.clear();
                    self.oversized_line = true;
                }
            }
        }
    }

    /// Interpret a completed SSE line, waiting for the blank line before emission.
    fn finish_line(&mut self) {
        let line = self.line.strip_suffix(b"\r").unwrap_or(&self.line);
        if line.is_empty() {
            self.phase = if self.phase == SsePhase::PendingTerminal {
                SsePhase::Complete
            } else {
                SsePhase::Reading
            };
        } else if matches!(
            line,
            b"event: response.completed" | b"event: response.failed" | b"event: response.incomplete" | b"data: [DONE]"
        ) {
            self.phase = SsePhase::PendingTerminal;
        }
    }

    /// Whether a complete terminal event was observed.
    fn is_complete(&self) -> bool {
        self.phase == SsePhase::Complete
    }
}

impl Drop for MirrorState {
    fn drop(&mut self) {
        if self.status.is_some() {
            debug!(
                method = %self.method,
                path = %self.path,
                status = self.status.unwrap_or(0),
                captured_response_bytes = self.response_body.len(),
                response_truncated = self.response_truncated,
                sse_terminal_marker = sse_terminal_marker(&self.response_body),
                "traffic_mirror: response ended without mirror emission"
            );
        }
    }
}

/// Diagnostic only: a terminal SSE event in the bounded capture suggests
/// the upstream finished even if the proxy did not run an EOS body hook.
/// Absence is inconclusive when the capture hit its size cap.
fn sse_terminal_marker(body: &[u8]) -> &'static str {
    let markers: &[(&[u8], &str)] = &[
        (b"event: response.completed", "response.completed"),
        (b"data: [DONE]", "[DONE]"),
        (b"event: response.failed", "response.failed"),
        (b"event: response.incomplete", "response.incomplete"),
    ];
    for &(marker, label) in markers {
        if body.windows(marker.len()).any(|part| part == marker) {
            return label;
        }
    }
    "none"
}

/// Append `chunk` to `buf` up to `cap` bytes, marking `truncated` when
/// bytes are dropped.
fn capture_into(buf: &mut Vec<u8>, truncated: &mut bool, chunk: &[u8], cap: usize) {
    let room = cap.saturating_sub(buf.len());
    let take = room.min(chunk.len());
    buf.extend_from_slice(chunk.get(..take).unwrap_or_default());
    if chunk.len() > room {
        *truncated = true;
    }
}

/// Key under which this filter instance stores its per-request state.
fn filter_state_key(id: Option<usize>) -> usize {
    id.unwrap_or(0)
}

/// Borrow this filter's per-request state, if present.
fn state_mut<'a>(ctx: &'a mut HttpFilterContext<'_>) -> Option<&'a mut MirrorState> {
    ctx.filter_state
        .get_mut(&filter_state_key(ctx.current_filter_id))
        .and_then(|state| state.downcast_mut::<MirrorState>())
}

/// Remove this filter's per-request state, if present.
fn take_state(ctx: &mut HttpFilterContext<'_>) -> Option<MirrorState> {
    ctx.filter_state
        .remove(&filter_state_key(ctx.current_filter_id))
        .and_then(|state| state.downcast::<MirrorState>().ok())
        .map(|state| *state)
}

// -----------------------------------------------------------------------------
// Envelope
// -----------------------------------------------------------------------------

/// Read an optional numeric `filter_metadata` value.
fn read_token_meta(ctx: &HttpFilterContext<'_>, key: &str) -> Option<u64> {
    ctx.filter_metadata.get(key).and_then(|v| v.parse().ok())
}

/// Build the mirror envelope for one completed request.
///
/// Pure with respect to the response pipeline: reads only the context
/// counters, metadata, and the captured state.
fn build_envelope(ctx: &HttpFilterContext<'_>, state: &MirrorState) -> serde_json::Value {
    let mut envelope = serde_json::Map::new();
    envelope.insert("schema".into(), serde_json::json!(SCHEMA));
    envelope.insert("timestamp".into(), serde_json::json!(timestamp()));
    envelope.insert("request_id".into(), serde_json::json!(ctx.request_id().unwrap_or("-")));
    envelope.insert("method".into(), serde_json::json!(state.method));
    envelope.insert("path".into(), serde_json::json!(state.path));
    envelope.insert("cluster".into(), serde_json::json!(state.cluster));
    envelope.insert("status".into(), serde_json::json!(state.status.unwrap_or(0)));
    envelope.insert("duration_ms".into(), serde_json::json!(elapsed_ms(state)));
    envelope.insert(
        "request".into(),
        body_record(&state.request_body, ctx.request_body_bytes, state.request_truncated),
    );
    envelope.insert(
        "response".into(),
        body_record(&state.response_body, ctx.response_body_bytes, state.response_truncated),
    );
    insert_token_usage(ctx, &mut envelope);
    serde_json::Value::Object(envelope)
}

/// RFC 3339 UTC timestamp with millisecond precision.
fn timestamp() -> String {
    chrono::Utc::now().to_rfc3339_opts(chrono::SecondsFormat::Millis, true)
}

/// Elapsed milliseconds since the request phase started.
fn elapsed_ms(state: &MirrorState) -> u64 {
    u64::try_from(state.started.elapsed().as_millis()).unwrap_or(u64::MAX)
}

/// Enrich the envelope with token usage when a `token_count` filter ran
/// in the same chain (declared after this filter; response hooks run in
/// reverse).
fn insert_token_usage(ctx: &HttpFilterContext<'_>, envelope: &mut serde_json::Map<String, serde_json::Value>) {
    if let Some(input) = read_token_meta(ctx, META_TOKEN_INPUT) {
        envelope.insert("token_input".into(), serde_json::json!(input));
    }
    if let Some(output) = read_token_meta(ctx, META_TOKEN_OUTPUT) {
        envelope.insert("token_output".into(), serde_json::json!(output));
    }
    if let Some(total) = read_token_meta(ctx, META_TOKEN_TOTAL) {
        envelope.insert("token_total".into(), serde_json::json!(total));
    }
}

/// Project one captured body into its envelope section.
///
/// `body_bytes` is the true byte count that flowed through the proxy;
/// `truncated` is true when the captured copy is shorter than that.
fn body_record(captured: &[u8], total_bytes: u64, truncated: bool) -> serde_json::Value {
    let mut record = serde_json::Map::new();
    record.insert("body_bytes".into(), serde_json::json!(total_bytes));
    record.insert("truncated".into(), serde_json::json!(truncated));
    match std::str::from_utf8(captured) {
        // LLM traffic is JSON or SSE text; pass it through verbatim so
        // collectors can parse without a decode step.
        Ok(text) => {
            record.insert("body".into(), serde_json::json!(text));
        },
        // Binary bodies (e.g. multipart file uploads) are base64-encoded
        // rather than dropped.
        Err(_) => {
            record.insert(
                "body_b64".into(),
                serde_json::json!(base64::engine::general_purpose::STANDARD.encode(captured)),
            );
        },
    }
    serde_json::Value::Object(record)
}

// -----------------------------------------------------------------------------
// Delivery
// -----------------------------------------------------------------------------

/// POST one envelope to the collector on a background task.
///
/// Fire-and-forget by design: the caller (a response-phase hook) returns
/// as soon as the task is spawned, so collector latency and failures never
/// touch the proxied request.
fn spawn_delivery(
    client: SubRequestClient,
    url: String,
    timeout: Duration,
    address_policy: AddressPolicy,
    body: Vec<u8>,
) {
    tokio::spawn(async move {
        let mut headers = http::HeaderMap::new();
        headers.insert(CONTENT_TYPE, http::HeaderValue::from_static("application/json"));
        let request = SubRequest {
            method: http::Method::POST,
            // The URL authority supplies Host and SNI; see SubRequest docs.
            uri: http::Uri::default(),
            headers,
            body: Bytes::from(body),
        };
        let result = client
            .execute_url(
                &url,
                request,
                subrequest::address_policy_hook(address_policy),
                subrequest::UrlResolutionPolicy::OperatorCached,
                timeout,
                MAX_ACK_BYTES,
                None,
            )
            .await
            .map_err(subrequest::map_url_error);
        report_delivery(&url, result);
    });
}

/// Log and count a failed delivery. Mirroring is observability: failures
/// warn and increment a counter, never reject traffic.
fn report_delivery(url: &str, result: Result<SubResponse, SubRequestError>) {
    match result {
        Ok(response) if (200..300).contains(&response.status) => {
            debug!(
                url,
                status = response.status,
                "traffic_mirror: collector accepted record"
            );
        },
        Ok(response) => {
            counter!(METRIC_DELIVERY_FAILURES).increment(1);
            warn!(
                url,
                status = response.status,
                "traffic_mirror: collector returned non-2xx"
            );
        },
        Err(error) => {
            counter!(METRIC_DELIVERY_FAILURES).increment(1);
            warn!(url, %error, "traffic_mirror: delivery failed");
        },
    }
}
