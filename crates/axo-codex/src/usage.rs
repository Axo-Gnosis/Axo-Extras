//! Codex account usage: `GET {base}/wham/usage` on demand, and the `x-codex-*` rate-limit
//! headers of every Codex response.

use std::time::Duration;

use async_trait::async_trait;
use axo_ai::{AccountUsage, Error, ModelAuth, ProviderUsage, UsageLimit, UsageSpend, now_ms};
use reqwest::header::HeaderMap;
use serde_json::Value;

use crate::login::email;
use crate::models::account_request;

const TIMEOUT: Duration = Duration::from_secs(10);
/// The two windows every rate limit carries: the short one (5 hours) and the weekly one.
const WINDOWS: [&str; 2] = ["primary", "secondary"];

pub(crate) struct CodexUsage {
    pub(crate) http: reqwest::Client,
}

#[async_trait]
impl ProviderUsage for CodexUsage {
    async fn fetch(&self, auth: ModelAuth) -> Result<AccountUsage, Error> {
        let account = auth.api_key.as_deref().and_then(email);
        let (base, headers) = account_request(auth)?;
        let body: Value = self
            .http
            .get(format!("{base}/wham/usage"))
            .headers(headers)
            .timeout(TIMEOUT)
            .send()
            .await?
            .error_for_status()?
            .json()
            .await?;
        Ok(report(&body, account, now_ms()))
    }

    fn observe(&self, _status: u16, headers: &HeaderMap) -> Vec<UsageLimit> {
        let now = now_ms();
        WINDOWS
            .into_iter()
            .filter_map(|key| {
                let get = |field: &str| {
                    headers
                        .get(format!("x-codex-{key}-{field}"))
                        .and_then(|value| value.to_str().ok()?.trim().parse::<f64>().ok())
                };
                window(
                    key.to_owned(),
                    None,
                    get("used-percent")?,
                    get("window-minutes").map(|minutes| minutes * 60.0),
                    get("reset-at"),
                    None,
                    now,
                )
            })
            .collect()
    }
}

/// The `wham/usage` body as a report: the account's windows, each additional meter's (named
/// after the model it covers), and the spend cap.
fn report(body: &Value, account: Option<String>, now: i64) -> AccountUsage {
    let mut limits = windows(None, &body["rate_limit"], now);
    for extra in body["additional_rate_limits"]
        .as_array()
        .into_iter()
        .flatten()
    {
        if let Some(name) = extra["limit_name"]
            .as_str()
            .or_else(|| extra["metered_feature"].as_str())
            .filter(|name| !name.is_empty())
        {
            limits.extend(windows(Some(name), &extra["rate_limit"], now));
        }
    }
    let cap = &body["spend_control"]["individual_limit"];
    if let Some(used) = number(&cap["used"]) {
        let limit = number(&cap["limit"]).filter(|limit| *limit > 0.0);
        limits.push(UsageLimit {
            id: "spend".into(),
            label: "Spend cap".into(),
            used: number(&cap["used_percent"])
                .map(|percent| percent / 100.0)
                .or_else(|| limit.map(|limit| used / limit)),
            resets_at: reset(
                number(&cap["reset_at"]),
                number(&cap["reset_after_seconds"]),
                now,
            ),
            spend: Some(UsageSpend {
                used,
                limit,
                currency: cap["unit"].as_str().unwrap_or("credits").to_owned(),
            }),
            ..Default::default()
        });
    }
    AccountUsage {
        plan: body["plan_type"].as_str().map(str::to_owned),
        account,
        limits,
        ..Default::default()
    }
}

fn windows(model: Option<&str>, rate_limit: &Value, now: i64) -> Vec<UsageLimit> {
    WINDOWS
        .into_iter()
        .filter_map(|key| {
            let entry = &rate_limit[format!("{key}_window")];
            window(
                model.map_or_else(|| key.to_owned(), |model| format!("{model}:{key}")),
                model,
                number(&entry["used_percent"])?,
                number(&entry["limit_window_seconds"]),
                number(&entry["reset_at"]),
                number(&entry["reset_after_seconds"]),
                now,
            )
        })
        .collect()
}

fn window(
    id: String,
    model: Option<&str>,
    percent: f64,
    seconds: Option<f64>,
    reset_at: Option<f64>,
    reset_after: Option<f64>,
    now: i64,
) -> Option<UsageLimit> {
    let window_ms = seconds
        .filter(|seconds| *seconds > 0.0)
        .map(|s| (s * 1000.0) as u64);
    let span = window_ms.map_or_else(|| id.clone(), span);
    Some(UsageLimit {
        label: model.map_or_else(|| span.clone(), |model| format!("{model} · {span}")),
        id,
        used: Some(percent.clamp(0.0, 100.0) / 100.0),
        resets_at: reset(reset_at, reset_after, now),
        window_ms,
        model: model.map(str::to_owned),
        spend: None,
    })
}

/// A reset instant in milliseconds: `reset_at` in seconds (or already milliseconds), else
/// `reset_after` seconds from now.
fn reset(reset_at: Option<f64>, reset_after: Option<f64>, now: i64) -> Option<i64> {
    reset_at
        .filter(|at| *at > 0.0)
        .map(|at| if at > 1e12 { at } else { at * 1000.0 } as i64)
        .or_else(|| reset_after.map(|seconds| now + (seconds * 1000.0) as i64))
}

/// A window length to the nearest hour, in days from a day up (Codex reports a few seconds
/// short of 5 hours or 7 days).
fn span(ms: u64) -> String {
    let hours = ((ms as f64) / 3_600_000.0).round().max(1.0) as u64;
    match hours {
        1 => "1 hour".into(),
        h if h < 24 => format!("{h} hours"),
        h if (h + 12) / 24 == 1 => "1 day".into(),
        h => format!("{} days", (h + 12) / 24),
    }
}

/// A number, or a numeric string.
fn number(value: &Value) -> Option<f64> {
    value
        .as_f64()
        .or_else(|| value.as_str()?.trim().parse().ok())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn wham_usage_and_headers_normalize_to_the_same_windows() {
        let body = json!({
            "plan_type": "plus",
            "rate_limit": {
                "primary_window": {"used_percent": 42, "limit_window_seconds": 17940, "reset_at": 1_900_000_000},
                "secondary_window": {"used_percent": "7.5", "limit_window_seconds": 604740, "reset_after_seconds": 60}
            },
            "additional_rate_limits": [{"limit_name": "GPT-5.3-Codex-Spark", "rate_limit": {
                "primary_window": {"used_percent": 100, "limit_window_seconds": 18000}
            }}],
            "spend_control": {"individual_limit": {"used": 12.5, "limit": 50}}
        });
        let report = report(&body, Some("a@b.c".into()), 1_000);
        assert_eq!(report.plan.as_deref(), Some("plus"));
        let summary: Vec<_> = report
            .limits
            .iter()
            .map(|limit| {
                (
                    limit.id.as_str(),
                    limit.label.as_str(),
                    limit.used,
                    limit.resets_at,
                )
            })
            .collect();
        assert_eq!(
            summary,
            [
                ("primary", "5 hours", Some(0.42), Some(1_900_000_000_000)),
                ("secondary", "7 days", Some(0.075), Some(61_000)),
                (
                    "GPT-5.3-Codex-Spark:primary",
                    "GPT-5.3-Codex-Spark · 5 hours",
                    Some(1.0),
                    None
                ),
                ("spend", "Spend cap", Some(0.25), None),
            ]
        );

        let mut headers = HeaderMap::new();
        for (name, value) in [
            ("x-codex-primary-used-percent", "50"),
            ("x-codex-primary-window-minutes", "300"),
            ("x-codex-primary-reset-at", "1900000000"),
        ] {
            headers.insert(name, value.parse().unwrap());
        }
        let observed = CodexUsage {
            http: reqwest::Client::new(),
        }
        .observe(200, &headers);
        assert_eq!(
            observed,
            [UsageLimit {
                id: "primary".into(),
                label: "5 hours".into(),
                used: Some(0.5),
                resets_at: Some(1_900_000_000_000),
                window_ms: Some(18_000_000),
                ..Default::default()
            }]
        );
    }
}
