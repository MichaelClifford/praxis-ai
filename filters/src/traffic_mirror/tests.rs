// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

use super::*;
use crate::test_utils::{make_filter_context, make_request, make_response};

/// Build the concrete filter with a private test client.
fn build_filter(yaml: &serde_yaml::Value) -> Result<TrafficMirrorFilter, FilterError> {
    TrafficMirrorFilter::build(yaml, crate::isolated_subrequest_client(1))
}

/// YAML pointing at an unroutable loopback collector: deliveries fail fast
/// in the background without touching the proxied request.
const UNROUTABLE_YAML: &str = "\
url: \"http://127.0.0.1:1/ingest\"
allow_private_addresses: true
";

/// Borrow this filter's state for inspection.
fn state_ref<'a>(ctx: &'a HttpFilterContext<'_>) -> Option<&'a MirrorState> {
    ctx.filter_state
        .values()
        .find_map(|state| state.downcast_ref::<MirrorState>())
}

// -----------------------------------------------------------------------------
// Config Parsing
// -----------------------------------------------------------------------------

#[test]
fn valid_config_parses() {
    let yaml: serde_yaml::Value = serde_yaml::from_str("url: \"http://collector:9090/ingest\"").unwrap();
    let filter = build_filter(&yaml).unwrap();
    assert_eq!(filter.name(), "traffic_mirror");
    assert_eq!(filter.max_body_bytes, config::DEFAULT_MAX_BODY_BYTES);
}

#[test]
fn invalid_config_is_rejected() {
    let yaml: serde_yaml::Value = serde_yaml::from_str("url: \"http://127.0.0.1:1/ingest\"").unwrap();
    assert!(
        build_filter(&yaml).is_err(),
        "loopback collector requires the private opt-in"
    );
}

// -----------------------------------------------------------------------------
// Capture
// -----------------------------------------------------------------------------

#[tokio::test]
async fn request_body_chunks_accumulate_up_to_the_cap() {
    let yaml: serde_yaml::Value =
        serde_yaml::from_str("url: \"http://collector:9090/ingest\"\nmax_body_bytes: 10").unwrap();
    let filter = build_filter(&yaml).unwrap();

    let req = make_request(http::Method::POST, "/v1/chat/completions");
    let mut ctx = make_filter_context(&req);
    assert!(matches!(
        filter.on_request(&mut ctx).await.unwrap(),
        FilterAction::Continue
    ));

    let mut chunk = Some(Bytes::from_static(b"0123456789"));
    assert!(matches!(
        filter.on_request_body(&mut ctx, &mut chunk, false).await.unwrap(),
        FilterAction::Continue
    ));
    let mut overflow = Some(Bytes::from_static(b"abcdef"));
    assert!(matches!(
        filter.on_request_body(&mut ctx, &mut overflow, true).await.unwrap(),
        FilterAction::Continue
    ));

    let state = state_ref(&ctx).expect("state survives the request body phase");
    assert_eq!(state.request_body, b"0123456789".to_vec(), "cap holds the first chunk");
    assert!(state.request_truncated, "bytes past the cap are dropped and flagged");
}

#[tokio::test]
async fn response_body_chunks_accumulate_across_calls() {
    let yaml: serde_yaml::Value = serde_yaml::from_str(UNROUTABLE_YAML).unwrap();
    let filter = build_filter(&yaml).unwrap();

    let req = make_request(http::Method::POST, "/v1/responses");
    let mut ctx = make_filter_context(&req);
    assert!(matches!(
        filter.on_request(&mut ctx).await.unwrap(),
        FilterAction::Continue
    ));

    let mut first = Some(Bytes::from_static(b"{\"id\":"));
    let action = filter.on_response_body(&mut ctx, &mut first, false).unwrap();
    assert!(matches!(action, FilterAction::Continue));
    assert!(state_ref(&ctx).is_some(), "state is re-inserted between chunks");

    let mut second = Some(Bytes::from_static(b"\"resp_1\"}"));
    assert!(matches!(
        filter.on_response_body(&mut ctx, &mut second, true).unwrap(),
        FilterAction::Continue
    ));

    // End of stream emitted the record and removed the state; the captured
    // copy is no longer observable, but the emit must not have panicked and
    // must not have mutated the forwarded chunks.
    assert!(state_ref(&ctx).is_none(), "emit clears the state");
    assert_eq!(
        first.as_ref().map(Bytes::as_ref),
        Some(&b"{\"id\":"[..]),
        "chunk passes through untouched"
    );
}

#[tokio::test]
async fn missing_state_is_a_noop() {
    let yaml: serde_yaml::Value = serde_yaml::from_str(UNROUTABLE_YAML).unwrap();
    let filter = build_filter(&yaml).unwrap();

    let req = make_request(http::Method::GET, "/v1/models");
    let mut ctx = make_filter_context(&req);

    let mut chunk = Some(Bytes::from_static(b"{}"));
    let action = filter.on_request_body(&mut ctx, &mut chunk, true).await.unwrap();
    assert!(matches!(action, FilterAction::Continue));
    let action = filter.on_response_body(&mut ctx, &mut chunk, true).unwrap();
    assert!(matches!(action, FilterAction::Continue));
    assert!(ctx.filter_state.is_empty());
}

#[tokio::test]
async fn bodyless_response_emits_from_the_response_phase() {
    let yaml: serde_yaml::Value = serde_yaml::from_str(UNROUTABLE_YAML).unwrap();
    let filter = build_filter(&yaml).unwrap();

    let req = make_request(http::Method::DELETE, "/v1/files/file_1");
    let mut ctx = make_filter_context(&req);
    assert!(matches!(
        filter.on_request(&mut ctx).await.unwrap(),
        FilterAction::Continue
    ));

    let mut response = make_response();
    response.status = http::StatusCode::NO_CONTENT;
    ctx.response_header = Some(&mut response);
    assert!(matches!(
        filter.on_response(&mut ctx).await.unwrap(),
        FilterAction::Continue
    ));

    assert!(state_ref(&ctx).is_none(), "bodyless 204 emits without a body phase");

    // A late body hook (e.g. empty final chunk) must not double-emit.
    let mut empty: Option<Bytes> = None;
    let action = filter.on_response_body(&mut ctx, &mut empty, true).unwrap();
    assert!(matches!(action, FilterAction::Continue));
}

// -----------------------------------------------------------------------------
// Envelope
// -----------------------------------------------------------------------------

#[test]
fn utf8_body_is_passed_verbatim_and_binary_bodies_are_base64() {
    let text = body_record(b"{\"model\":\"gpt-4.1\"}", 18, false);
    assert_eq!(text["body"].as_str(), Some("{\"model\":\"gpt-4.1\"}"));
    assert!(text.get("body_b64").is_none());

    let binary = body_record(&[0xFF, 0x00, 0xFE], 3, false);
    assert!(binary.get("body").is_none(), "invalid UTF-8 must not be lossy");
    assert_eq!(
        binary["body_b64"].as_str(),
        Some(
            base64::engine::general_purpose::STANDARD
                .encode([0xFF, 0x00, 0xFE])
                .as_str()
        )
    );
}

#[test]
fn body_record_reports_totals_and_truncation() {
    let record = body_record(b"short", 1_048_576, true);
    assert_eq!(record["body_bytes"].as_u64(), Some(1_048_576));
    assert_eq!(record["truncated"].as_bool(), Some(true));
}

#[test]
fn terminal_sse_marker_is_detected_in_bounded_capture() {
    let mut body = b"event: response.com".to_vec();
    body.extend_from_slice(b"pleted\ndata: {}\n\n");
    assert_eq!(sse_terminal_marker(&body), "response.completed");
    assert_eq!(sse_terminal_marker(b"data: [DONE]\n\n"), "[DONE]");
    assert_eq!(sse_terminal_marker(b"data: {}\n\n"), "none");
}

#[test]
fn sse_terminal_requires_event_delimiter_across_chunks() {
    let mut terminal = SseTerminal::default();
    terminal.observe(b"event: response.com");
    terminal.observe(b"pleted\r\ndata: {\"type\":\"response.completed\"}\r\n");
    assert!(!terminal.is_complete(), "the event's data must be captured first");
    terminal.observe(b"\r\n");
    assert!(terminal.is_complete());

    let mut done = SseTerminal::default();
    done.observe(b"data: [DONE]\n");
    assert!(!done.is_complete());
    done.observe(b"\n");
    assert!(done.is_complete());

    let mut oversized = SseTerminal::default();
    oversized.observe(&vec![b'a'; 1024]);
    oversized.observe(b"\nevent: response.completed\n\n");
    assert!(
        oversized.is_complete(),
        "an oversized data line must not prevent a later terminal event"
    );
}

#[tokio::test]
async fn complete_sse_event_emits_once_even_if_eos_arrives_later() {
    let yaml: serde_yaml::Value = serde_yaml::from_str(UNROUTABLE_YAML).unwrap();
    let filter = build_filter(&yaml).unwrap();
    let req = make_request(http::Method::POST, "/v1/responses");
    let mut ctx = make_filter_context(&req);
    drop(filter.on_request(&mut ctx).await.unwrap());

    let mut response = make_response();
    response
        .headers
        .insert(CONTENT_TYPE, http::HeaderValue::from_static("text/event-stream"));
    ctx.response_header = Some(&mut response);
    drop(filter.on_response(&mut ctx).await.unwrap());

    let mut first = Some(Bytes::from_static(b"event: response.completed\ndata: {}\n"));
    drop(filter.on_response_body(&mut ctx, &mut first, false).unwrap());
    assert!(state_ref(&ctx).is_some(), "do not emit before the event delimiter");

    let mut last = Some(Bytes::from_static(b"\n"));
    drop(filter.on_response_body(&mut ctx, &mut last, false).unwrap());
    assert!(
        state_ref(&ctx).is_none(),
        "the terminal event emits without an EOS callback"
    );

    let mut eos = None;
    drop(filter.on_response_body(&mut ctx, &mut eos, true).unwrap());
    assert!(state_ref(&ctx).is_none(), "a later EOS cannot emit a second record");
}

/// Build a state carrying identity fields and captured bodies.
fn envelope_state(ctx: &HttpFilterContext<'_>) -> MirrorState {
    let mut state = MirrorState::new(ctx);
    state.method = "POST".into();
    state.path = "/v1/chat/completions".into();
    state.cluster = Some("backend".into());
    state.status = Some(200);
    state.request_body = b"{\"model\":\"gpt-4.1\"}".to_vec();
    state.response_body = b"{\"id\":\"resp_1\"}".to_vec();
    state
}

#[test]
fn envelope_carries_identity_and_timing() {
    let req = make_request(http::Method::POST, "/v1/chat/completions");
    let mut ctx = make_filter_context(&req);
    ctx.cluster = Some(std::sync::Arc::from("backend"));

    let envelope = build_envelope(&ctx, &envelope_state(&ctx));
    assert_eq!(envelope["schema"].as_str(), Some("praxis.traffic_mirror.v1"));
    assert_eq!(envelope["method"].as_str(), Some("POST"));
    assert_eq!(envelope["path"].as_str(), Some("/v1/chat/completions"));
    assert_eq!(envelope["cluster"].as_str(), Some("backend"));
    assert_eq!(envelope["status"].as_u64(), Some(200));
    assert!(envelope["timestamp"].as_str().is_some_and(|t| t.ends_with('Z')));
    assert!(envelope["duration_ms"].as_u64().is_some());
    assert_eq!(
        envelope["request_id"].as_str(),
        Some("-"),
        "no request id resolves to a placeholder"
    );
}

#[test]
fn envelope_carries_captured_bodies() {
    let req = make_request(http::Method::POST, "/v1/chat/completions");
    let mut ctx = make_filter_context(&req);
    ctx.request_body_bytes = 42;
    ctx.response_body_bytes = 4_242;

    let envelope = build_envelope(&ctx, &envelope_state(&ctx));
    assert_eq!(envelope["request"]["body"].as_str(), Some("{\"model\":\"gpt-4.1\"}"));
    assert_eq!(envelope["request"]["body_bytes"].as_u64(), Some(42));
    assert_eq!(envelope["request"]["truncated"].as_bool(), Some(false));
    assert_eq!(envelope["response"]["body"].as_str(), Some("{\"id\":\"resp_1\"}"));
    assert_eq!(envelope["response"]["body_bytes"].as_u64(), Some(4_242));
    assert!(
        envelope.get("token_input").is_none() && envelope.get("token_output").is_none(),
        "token fields are absent without token_count metadata"
    );
}

#[test]
fn envelope_is_enriched_with_token_metadata_when_present() {
    let req = make_request(http::Method::POST, "/v1/chat/completions");
    let mut ctx = make_filter_context(&req);
    ctx.filter_metadata
        .insert(META_TOKEN_INPUT.to_owned(), "100".to_owned());
    ctx.filter_metadata
        .insert(META_TOKEN_OUTPUT.to_owned(), "50".to_owned());
    ctx.filter_metadata
        .insert(META_TOKEN_TOTAL.to_owned(), "150".to_owned());

    let state = MirrorState::new(&ctx);
    let envelope = build_envelope(&ctx, &state);
    assert_eq!(envelope["token_input"].as_u64(), Some(100));
    assert_eq!(envelope["token_output"].as_u64(), Some(50));
    assert_eq!(envelope["token_total"].as_u64(), Some(150));
}

// -----------------------------------------------------------------------------
// Delivery
// -----------------------------------------------------------------------------

#[tokio::test]
async fn unroutable_collector_never_fails_the_proxied_request() {
    let yaml: serde_yaml::Value = serde_yaml::from_str(UNROUTABLE_YAML).unwrap();
    let filter = build_filter(&yaml).unwrap();

    let req = make_request(http::Method::POST, "/v1/chat/completions");
    let mut ctx = make_filter_context(&req);
    assert!(matches!(
        filter.on_request(&mut ctx).await.unwrap(),
        FilterAction::Continue
    ));

    let mut chunk = Some(Bytes::from_static(b"{\"model\":\"gpt-4.1\"}"));
    assert!(matches!(
        filter.on_request_body(&mut ctx, &mut chunk, true).await.unwrap(),
        FilterAction::Continue
    ));

    let mut response = make_response();
    ctx.response_header = Some(&mut response);
    assert!(matches!(
        filter.on_response(&mut ctx).await.unwrap(),
        FilterAction::Continue
    ));

    let mut body = Some(Bytes::from_static(b"{\"id\":\"resp_1\"}"));
    assert!(matches!(
        filter.on_response_body(&mut ctx, &mut body, true).unwrap(),
        FilterAction::Continue
    ));

    // Give the fire-and-forget delivery a moment to fail; the failure warns
    // and counts, it must never propagate.
    tokio::time::sleep(Duration::from_millis(50)).await;
    assert!(state_ref(&ctx).is_none());
}
