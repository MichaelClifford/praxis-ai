// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Serde configuration types for the traffic mirror filter.

use praxis_ai_apis::callout_target::{AddressPolicy, validate_configured_http_target};
use praxis_filter::FilterError;
use serde::Deserialize;

/// Default per-body capture cap: 1 MiB, applied separately to the request
/// and the response body.
pub(crate) const DEFAULT_MAX_BODY_BYTES: usize = 1_048_576;

/// Default delivery timeout in seconds.
pub(crate) const DEFAULT_TIMEOUT_SECONDS: u64 = 5;

/// Deserialized YAML configuration for the traffic mirror filter.
///
/// ```yaml
/// filter: traffic_mirror
/// url: "http://collector:9090/ingest"
/// timeout_seconds: 5            # optional, default 5
/// max_body_bytes: 1048576       # optional, default 1 MiB per body
/// allow_private_addresses: false # optional, default false
/// ```
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct TrafficMirrorConfig {
    /// Absolute HTTP(S) URL of the collector endpoint that mirror
    /// records are delivered to. Validated at construction time; see
    /// [`validate_config`].
    pub url: String,

    /// Delivery timeout in seconds. One budget covers DNS, connect, and
    /// I/O for the POST. Delivery is fire-and-forget: the timeout bounds a
    /// background task, never the proxied request.
    #[serde(default = "default_timeout_seconds")]
    pub timeout_seconds: u64,

    /// Maximum bytes of each body captured into the mirror record. The
    /// cap applies separately to the request body and the response body;
    /// bytes past the cap are counted (`body_bytes`) but not included,
    /// and the record is marked `truncated: true`.
    #[serde(default = "default_max_body_bytes")]
    pub max_body_bytes: usize,

    /// Allow the collector to resolve to a private, loopback, or
    /// link-local address. Defaults to `false`; set to `true` explicitly
    /// when a trusted loopback collector (e.g. a local `MLflow` adapter)
    /// is the intended destination. Cloud-metadata, unspecified, and
    /// multicast addresses remain blocked even when enabled.
    #[serde(default)]
    pub allow_private_addresses: bool,
}

/// Serde default for [`TrafficMirrorConfig::timeout_seconds`].
const fn default_timeout_seconds() -> u64 {
    DEFAULT_TIMEOUT_SECONDS
}

/// Serde default for [`TrafficMirrorConfig::max_body_bytes`].
const fn default_max_body_bytes() -> usize {
    DEFAULT_MAX_BODY_BYTES
}

/// Validate the parsed configuration.
///
/// # Errors
///
/// Returns [`FilterError`] when the URL is empty, the timeout or body cap
/// is zero, or the collector URL is malformed or statically classifiable
/// as a private target without the opt-in.
pub(crate) fn validate_config(config: &TrafficMirrorConfig) -> Result<(), FilterError> {
    if config.timeout_seconds == 0 {
        return Err("traffic_mirror: timeout_seconds must be at least 1".into());
    }
    if config.max_body_bytes == 0 {
        return Err("traffic_mirror: max_body_bytes must be at least 1".into());
    }
    validate_configured_http_target(
        "traffic_mirror",
        &config.url,
        AddressPolicy::from_allow_private(config.allow_private_addresses),
    )?;
    Ok(())
}

#[cfg(test)]
#[expect(clippy::allow_attributes, reason = "blanket test suppressions")]
#[allow(clippy::unwrap_used, clippy::expect_used, reason = "tests")]
mod tests {
    use super::*;

    fn parse(yaml: &str) -> Result<TrafficMirrorConfig, serde_yaml::Error> {
        serde_yaml::from_str(yaml)
    }

    #[test]
    fn minimal_config_applies_defaults() {
        let cfg = parse("url: http://collector.example:9090/ingest").unwrap();
        assert_eq!(cfg.timeout_seconds, DEFAULT_TIMEOUT_SECONDS);
        assert_eq!(cfg.max_body_bytes, DEFAULT_MAX_BODY_BYTES);
        assert!(!cfg.allow_private_addresses);
        validate_config(&cfg).expect("public http target should validate");
    }

    #[test]
    fn full_config_parses() {
        let cfg = parse(
            "url: http://collector.example:9090/ingest\n\
             timeout_seconds: 2\n\
             max_body_bytes: 4096\n\
             allow_private_addresses: true\n",
        )
        .unwrap();
        assert_eq!(cfg.timeout_seconds, 2);
        assert_eq!(cfg.max_body_bytes, 4096);
        assert!(cfg.allow_private_addresses);
        validate_config(&cfg).expect("explicit opts should validate");
    }

    #[test]
    fn unknown_fields_are_rejected() {
        let err = parse("url: http://collector.example:9090/ingest\nnope: 1").expect_err("deny_unknown_fields");
        assert!(
            err.to_string().contains("unknown field"),
            "should name the unknown field: {err}"
        );
    }

    #[test]
    fn zero_timeout_is_rejected() {
        let cfg = parse("url: http://collector.example:9090/ingest\ntimeout_seconds: 0").unwrap();
        assert!(validate_config(&cfg).is_err());
    }

    #[test]
    fn zero_body_cap_is_rejected() {
        let cfg = parse("url: http://collector.example:9090/ingest\nmax_body_bytes: 0").unwrap();
        assert!(validate_config(&cfg).is_err());
    }

    #[test]
    fn loopback_target_requires_private_opt_in() {
        let cfg = parse("url: http://127.0.0.1:9090/ingest").unwrap();
        let err = validate_config(&cfg).expect_err("literal loopback IP should be rejected");
        assert!(
            err.to_string().contains("private-target opt-in"),
            "should explain the opt-in: {err}"
        );
    }

    #[test]
    fn loopback_target_allowed_with_private_opt_in() {
        let cfg = parse("url: http://127.0.0.1:9090/ingest\nallow_private_addresses: true").unwrap();
        validate_config(&cfg).expect("opted-in loopback target should validate");
    }

    #[test]
    fn non_http_scheme_is_rejected() {
        let cfg = parse("url: ftp://collector.example/ingest").unwrap();
        assert!(validate_config(&cfg).is_err());
    }
}
