// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Tests for the traffic mirror example configuration.

use std::{
    collections::HashMap,
    io::{Read as _, Write as _},
    net::TcpListener,
    sync::mpsc,
    thread,
    time::Duration,
};

use praxis_test_utils::{
    Backend, StatefulCapturingBackend, StatefulCapturingGuard, free_port, http_send, parse_status,
};

// -----------------------------------------------------------------------------
// Config Parsing
// -----------------------------------------------------------------------------

#[test]
fn traffic_mirror_config_parses() {
    let collector_port = free_port();
    let config = super::load_example_config(
        "traffic-mirror.yaml",
        29900,
        HashMap::from([("127.0.0.1:3000", 29901_u16), ("127.0.0.1:9090", collector_port)]),
    );

    assert_eq!(config.listeners.len(), 1, "should have 1 listener");
}

// -----------------------------------------------------------------------------
// End-to-End Delivery
// -----------------------------------------------------------------------------

#[test]
fn traffic_mirror_delivers_request_response_and_token_usage() {
    // Upstream returns an OpenAI-shaped usage body so token_count writes
    // token.input / token.output / token.total for the mirror to read back.
    let response_body = r#"{"id":"chatcmpl-1","object":"chat.completion","usage":{"prompt_tokens":10,"completion_tokens":90,"total_tokens":100}}"#;
    let backend_guard = Backend::status(200, response_body)
        .header("Content-Type", "application/json")
        .start_with_shutdown();
    let backend_port = backend_guard.port();
    let proxy_port = free_port();

    let collector = StatefulCapturingBackend::new(vec![(204, String::new())]).start_with_shutdown();
    let collector_port = collector.port();

    let config = super::load_example_config(
        "traffic-mirror.yaml",
        proxy_port,
        HashMap::from([("127.0.0.1:3000", backend_port), ("127.0.0.1:9090", collector_port)]),
    );

    let proxy = praxis_test_utils::start_proxy(&config);
    let request_body = r#"{"model":"gpt-4","messages":[{"role":"user","content":"hi"}]}"#;
    let raw = http_send(
        proxy.addr(),
        &format!(
            "POST /v1/chat/completions HTTP/1.1\r\n\
             Host: localhost\r\n\
             Content-Type: application/json\r\n\
             Content-Length: {}\r\n\
             Connection: close\r\n\r\n{}",
            request_body.len(),
            request_body
        ),
    );

    assert_eq!(parse_status(&raw), 200, "the mirrored request should be proxied");

    let record = wait_for_mirror_record(&collector);
    let json: serde_json::Value = serde_json::from_str(&record).expect("mirror envelope should be valid JSON");

    assert_eq!(json["schema"], "praxis.traffic_mirror.v1");
    assert_eq!(json["method"], "POST");
    assert_eq!(json["path"], "/v1/chat/completions");
    assert_eq!(json["cluster"], "backend");
    assert_eq!(json["status"], 200, "the response status should be mirrored");
    assert_eq!(
        json["request"]["body"].as_str(),
        Some(request_body),
        "the request body should be captured verbatim"
    );
    assert_eq!(json["request"]["body_bytes"].as_u64(), Some(request_body.len() as u64));
    assert_eq!(json["request"]["truncated"], false);
    assert_eq!(
        json["response"]["body"].as_str(),
        Some(response_body),
        "the response body should be captured verbatim"
    );
    assert_eq!(
        json["response"]["body_bytes"].as_u64(),
        Some(response_body.len() as u64)
    );
    assert_eq!(
        json["token_input"], 10,
        "token usage published by token_count must reach the envelope"
    );
    assert_eq!(json["token_output"], 90);
    assert_eq!(json["token_total"], 100);
}

#[test]
fn traffic_mirror_down_collector_does_not_break_proxying() {
    let backend_guard = Backend::status(200, "ok").start_with_shutdown();
    let backend_port = backend_guard.port();
    let proxy_port = free_port();

    // Point the collector at a port with nothing listening: delivery is
    // fire-and-forget, so the proxied request must still succeed.
    let collector_port = free_port();

    let config = super::load_example_config(
        "traffic-mirror.yaml",
        proxy_port,
        HashMap::from([("127.0.0.1:3000", backend_port), ("127.0.0.1:9090", collector_port)]),
    );

    let proxy = praxis_test_utils::start_proxy(&config);
    let raw = http_send(
        proxy.addr(),
        "POST /v1/chat/completions HTTP/1.1\r\n\
         Host: localhost\r\n\
         Content-Type: application/json\r\n\
         Content-Length: 2\r\n\
         Connection: close\r\n\r\n{}",
    );

    assert_eq!(
        parse_status(&raw),
        200,
        "an unreachable collector must never fail the proxied request"
    );
}

#[test]
fn traffic_mirror_delivers_each_sequential_streamed_response() {
    let stream = "data: {\"type\":\"response.output_text.delta\",\"delta\":\"hello\"}\n\n\
                  event: response.completed\n\
                  data: {\"type\":\"response.completed\",\"response\":{\"usage\":{\"input_tokens\":2,\"output_tokens\":1}}}\n\n";
    let backend = StatefulCapturingBackend::new(vec![(200, stream.to_owned()); 4]).start_with_shutdown();
    let collector = StatefulCapturingBackend::new(vec![(204, String::new()); 4]).start_with_shutdown();
    let config = super::load_example_config(
        "traffic-mirror.yaml",
        free_port(),
        HashMap::from([("127.0.0.1:3000", backend.port()), ("127.0.0.1:9090", collector.port())]),
    );
    let proxy = praxis_test_utils::start_proxy(&config);

    for turn in 0..4 {
        let body = format!(r#"{{"model":"gpt-4","stream":true,"turn":{turn}}}"#);
        let raw = http_send(
            proxy.addr(),
            &format!(
                "POST /v1/chat/completions HTTP/1.1\r\n\
                 Host: localhost\r\n\
                 Content-Type: application/json\r\n\
                 Content-Length: {}\r\n\
                 Connection: close\r\n\r\n{}",
                body.len(),
                body
            ),
        );
        assert_eq!(parse_status(&raw), 200, "turn {turn} should be proxied");
    }

    for _ in 0..50 {
        if collector
            .requests()
            .iter()
            .filter(|r| r.uri.starts_with("/ingest"))
            .count()
            == 4
        {
            break;
        }
        thread::sleep(Duration::from_millis(100));
    }
    let records: Vec<_> = collector
        .requests()
        .into_iter()
        .filter(|r| r.uri.starts_with("/ingest"))
        .map(|r| serde_json::from_str::<serde_json::Value>(&r.body).expect("mirror record must be JSON"))
        .collect();
    assert_eq!(records.len(), 4, "each completed SSE response must be mirrored once");
    for turn in 0..4 {
        assert!(
            records.iter().any(|record| record["request"]["body"]
                .as_str()
                .is_some_and(|body| { body.contains(&format!("\"turn\":{turn}")) })),
            "missing mirror record for turn {turn}"
        );
    }
    assert!(
        records.iter().all(|record| record["response"]["body"]
            .as_str()
            .is_some_and(|body| body.contains("event: response.completed"))),
        "the complete SSE event must be captured before delivery"
    );
}

#[test]
fn traffic_mirror_emits_on_terminal_sse_event_before_transport_eof() {
    let listener = TcpListener::bind("127.0.0.1:0").expect("backend should bind");
    let backend_port = listener.local_addr().expect("backend address").port();
    let (release_tx, release_rx) = mpsc::channel();
    let backend = thread::spawn(move || {
        let (mut stream, _) = listener.accept().expect("backend should accept request");
        let mut request = [0_u8; 4096];
        assert!(stream.read(&mut request).expect("backend should read request") > 0);
        let event = "event: response.completed\ndata: {\"type\":\"response.completed\",\"response\":{\"usage\":{\"input_tokens\":2,\"output_tokens\":1}}\n\n";
        write!(
            stream,
            "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nTransfer-Encoding: chunked\r\nConnection: close\r\n\r\n{:x}\r\n{event}\r\n",
            event.len()
        )
        .expect("backend should write terminal event");
        stream.flush().expect("terminal event should flush");
        release_rx
            .recv_timeout(Duration::from_secs(5))
            .expect("collector must receive record before EOF");
        stream.write_all(b"0\r\n\r\n").expect("backend should finish stream");
    });
    let collector = StatefulCapturingBackend::new(vec![(204, String::new())]).start_with_shutdown();
    let config = super::load_example_config(
        "traffic-mirror.yaml",
        free_port(),
        HashMap::from([("127.0.0.1:3000", backend_port), ("127.0.0.1:9090", collector.port())]),
    );
    let proxy = praxis_test_utils::start_proxy(&config);
    let proxy_addr = proxy.addr().to_owned();
    let client = thread::spawn(move || {
        http_send(
            &proxy_addr,
            "POST /v1/responses HTTP/1.1\r\n\
             Host: localhost\r\n\
             Content-Type: application/json\r\n\
             Content-Length: 31\r\n\
             Connection: close\r\n\r\n\
             {\"model\":\"gpt-4\",\"stream\":true}",
        )
    });

    let record = wait_for_mirror_record(&collector);
    let json: serde_json::Value = serde_json::from_str(&record).expect("mirror record should be JSON");
    assert_eq!(json["response"]["end_of_stream_observed"], false);
    assert!(
        json["response"]["body"]
            .as_str()
            .is_some_and(|body| body.ends_with("\n\n"))
    );
    release_tx.send(()).expect("backend should finish");
    assert_eq!(parse_status(&client.join().expect("client should finish")), 200);
    backend.join().expect("backend should finish");
    thread::sleep(Duration::from_millis(50));
    assert_eq!(
        collector.requests().len(),
        1,
        "later transport EOF must not duplicate the record"
    );
}

// -----------------------------------------------------------------------------
// Truncation (inline config)
// -----------------------------------------------------------------------------

#[test]
fn traffic_mirror_truncates_bodies_past_the_cap() {
    let backend_guard = Backend::status(200, "0123456789abcdef").start_with_shutdown();
    let backend_port = backend_guard.port();
    let proxy_port = free_port();

    let collector = StatefulCapturingBackend::new(vec![(204, String::new())]).start_with_shutdown();
    let collector_port = collector.port();

    let yaml = format!(
        "\
listeners:
  - name: default
    address: \"127.0.0.1:{proxy_port}\"
    filter_chains: [main]
filter_chains:
  - name: main
    filters:
      - filter: router
        routes:
          - path_prefix: \"/\"
            cluster: backend
      - filter: traffic_mirror
        url: \"http://127.0.0.1:{collector_port}/ingest\"
        allow_private_addresses: true
        max_body_bytes: 8
      - filter: load_balancer
        clusters:
          - name: backend
            endpoints:
              - \"127.0.0.1:{backend_port}\"
insecure_options:
  allow_private_endpoints: true
"
    );
    let config = praxis_core::config::Config::from_yaml(&yaml).expect("inline config should parse");

    let proxy = praxis_test_utils::start_proxy(&config);
    let request_body = r#"{"model":"gpt-4"}"#;
    let raw = http_send(
        proxy.addr(),
        &format!(
            "POST /v1/chat/completions HTTP/1.1\r\n\
             Host: localhost\r\n\
             Content-Type: application/json\r\n\
             Content-Length: {}\r\n\
             Connection: close\r\n\r\n{}",
            request_body.len(),
            request_body
        ),
    );
    assert_eq!(parse_status(&raw), 200);

    let record = wait_for_mirror_record(&collector);
    let json: serde_json::Value = serde_json::from_str(&record).expect("mirror envelope should be valid JSON");

    assert_eq!(
        json["request"]["body"].as_str(),
        request_body.get(..8),
        "capture stops at the cap"
    );
    assert_eq!(json["request"]["body_bytes"].as_u64(), Some(request_body.len() as u64));
    assert_eq!(json["request"]["truncated"], true, "bytes past the cap are flagged");
    assert_eq!(
        json["response"]["truncated"], true,
        "the 16-byte response is also truncated at the cap"
    );
}

// -----------------------------------------------------------------------------
// Helpers
// -----------------------------------------------------------------------------

/// Poll the collector mock until the fire-and-forget mirror POST arrives,
/// returning its body. Panics if it does not arrive in time.
fn wait_for_mirror_record(collector: &StatefulCapturingGuard) -> String {
    for _ in 0..50 {
        if let Some(req) = collector
            .requests()
            .into_iter()
            .find(|r| r.method == "POST" && r.uri.starts_with("/ingest"))
        {
            return req.body;
        }
        thread::sleep(Duration::from_millis(100));
    }
    panic!("traffic mirror record POST was not received within 5s");
}
