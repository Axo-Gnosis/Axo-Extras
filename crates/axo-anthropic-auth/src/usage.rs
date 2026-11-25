//! Subscription usage: `GET /api/oauth/usage` and `/api/oauth/profile` on demand, and the
//! `anthropic-ratelimit-unified-*` headers of every Messages response.

use async_trait::async_trait;
use axo_ai::{AccountUsage, Error, ModelAuth, ProviderUsage, UsageLimit, UsageSpend};
use reqwest::header::HeaderMap;
use serde_json::Value;

use crate::account_get;

const FIVE_HOURS_MS: u64 = 5 * 3_600_000;
const SEVEN_DAYS_MS: u64 = 7 * 86_400_000;

pub(crate) struct AnthropicAuthUsage {
    pub(crate) http: reqwest::Client,
}

#[async_trait]
impl ProviderUsage for AnthropicAuthUsage {
    async fn fetch(&self, auth: ModelAuth) -> Result<AccountUsage, Error> {
        let (usage, profile) = tokio::join!(
            account_get(&self.http, &auth, "/api/oauth/usage"),
            account_get(&self.http, &auth, "/api/oauth/profile"),
        );
        let profile = profile.unwrap_or_else(|error| {
            tracing::debug!("Anthropic profile unavailable: {error}");
            Value::Null
        });
        Ok(report(&usage?, &profile))
    }

    fn observe(&self, _status: u16, headers: &HeaderMap) -> Vec<UsageLimit> {
        let get = |name: &str| {
            headers
                .get(format!("anthropic-ratelimit-unified-{name}"))
                .and_then(|value| value.to_str().ok()?.trim().parse::<f64>().ok())
        };
        [("5h", "5h"), ("7d", "weekly")]
            .into_iter()
            .filter_map(|(header, id)| {
                let used = get(&format!("{header}-utilization"))?;
                let mut limit = window(id, None);
                limit.used = Some(used.max(0.0));
                limit.resets_at = get(&format!("{header}-reset"))
                    .filter(|at| *at > 0.0)
                    .map(|at| (at * 1000.0) as i64);
                Some(limit)
            })
            .collect()
    }
}

/// An account-wide window (`"5h"` or `"weekly"`), or a weekly one scoped to `model`.
fn window(id: &str, model: Option<&str>) -> UsageLimit {
    let (label, window_ms) = match id {
        "5h" => ("5 hours".to_owned(), FIVE_HOURS_MS),
        _ => match model {
            Some(model) => (format!("{model} · 7 days"), SEVEN_DAYS_MS),
            None => ("7 days".to_owned(), SEVEN_DAYS_MS),
        },
    };
    UsageLimit {
        id: model.map_or_else(
            || id.to_owned(),
            |model| format!("weekly:{}", model.to_lowercase()),
        ),
        label,
        window_ms: Some(window_ms),
        model: model.map(str::to_owned),
        ..Default::default()
    }
}

/// A `{ utilization | percent (0-100), resets_at (ISO) }` entry as `limit`'s amounts.
fn filled(mut limit: UsageLimit, entry: &Value, percent_key: &str) -> Option<UsageLimit> {
    let percent = entry[percent_key].as_f64()?;
    limit.used = Some(percent.max(0.0) / 100.0);
    limit.resets_at = entry["resets_at"]
        .as_str()
        .and_then(|at| at.parse::<jiff::Timestamp>().ok())
        .map(|at| at.as_millisecond());
    Some(limit)
}

/// The usage and profile bodies as a report. The named windows (`five_hour`, `seven_day`,
/// `seven_day_opus`, `seven_day_sonnet`) win over the same window in the newer `limits` list.
fn report(usage: &Value, profile: &Value) -> AccountUsage {
    let mut limits: Vec<UsageLimit> = [
        ("five_hour", "5h", None),
        ("seven_day", "weekly", None),
        ("seven_day_opus", "weekly", Some("Opus")),
        ("seven_day_sonnet", "weekly", Some("Sonnet")),
    ]
    .into_iter()
    .filter_map(|(key, id, model)| filled(window(id, model), &usage[key], "utilization"))
    .collect();
    for entry in usage["limits"].as_array().into_iter().flatten() {
        let model = entry
            .pointer("/scope/model/display_name")
            .and_then(Value::as_str);
        let limit = match (entry["kind"].as_str(), model) {
            (Some("session"), _) => window("5h", None),
            (Some("weekly_all"), _) => window("weekly", None),
            (Some("weekly_scoped"), Some(model)) => window("weekly", Some(model)),
            _ => continue,
        };
        if !limits.iter().any(|known| known.id == limit.id)
            && let Some(limit) = filled(limit, entry, "percent")
        {
            limits.push(limit);
        }
    }
    limits.extend(spend(usage));
    let organization = &profile["organization"];
    let plan = organization["rate_limit_tier"]
        .as_str()
        .and_then(|tier| tier.split("max_").nth(1))
        .map(|multiplier| format!("Max {multiplier}"))
        .or_else(|| {
            let kind = organization["organization_type"].as_str()?;
            let name = kind.strip_prefix("claude_").unwrap_or(kind);
            let mut chars = name.chars();
            let first = chars.next()?;
            Some(first.to_uppercase().chain(chars).collect())
        });
    AccountUsage {
        plan,
        account: profile
            .pointer("/account/email")
            .and_then(Value::as_str)
            .map(str::to_owned),
        limits,
        fetched_at: None,
    }
}

/// Extra usage beyond the plan: `spend` (minor units with an exponent), else the older
/// `extra_usage` (minor units with `decimal_places`). Disabled meters are omitted.
fn spend(usage: &Value) -> Option<UsageLimit> {
    let (used, limit, currency) = if usage["spend"]["enabled"].as_bool() == Some(true) {
        let amount = |value: &Value| {
            let minor = value["amount_minor"].as_f64()?;
            let exponent = value["exponent"].as_i64().unwrap_or(2) as i32;
            Some(minor / 10f64.powi(exponent))
        };
        let spend = &usage["spend"];
        (
            amount(&spend["used"])?,
            amount(&spend["limit"]),
            spend["used"]["currency"].as_str(),
        )
    } else if usage["extra_usage"]["is_enabled"].as_bool() == Some(true) {
        let extra = &usage["extra_usage"];
        let scale = 10f64.powi(extra["decimal_places"].as_i64().unwrap_or(2) as i32);
        (
            extra["used_credits"].as_f64()? / scale,
            extra["monthly_limit"].as_f64().map(|limit| limit / scale),
            extra["currency"].as_str(),
        )
    } else {
        return None;
    };
    let limit = limit.filter(|limit| *limit > 0.0);
    Some(UsageLimit {
        id: "spend".into(),
        label: "Extra usage".into(),
        used: limit.map(|limit| used / limit),
        spend: Some(UsageSpend {
            used,
            limit,
            currency: currency.unwrap_or("USD").to_owned(),
        }),
        ..Default::default()
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn usage_profile_and_headers_normalize() {
        let usage = json!({
            "five_hour": {"utilization": 30.0, "resets_at": "2026-01-01T05:00:00Z"},
            "seven_day": {"utilization": 12.5, "resets_at": null},
            "seven_day_opus": null,
            "limits": [
                {"kind": "weekly_all", "percent": 99},
                {"kind": "weekly_scoped", "percent": 80, "scope": {"model": {"display_name": "Fable"}}}
            ],
            "spend": {"enabled": true, "used": {"amount_minor": 1250, "currency": "USD", "exponent": 2},
                      "limit": {"amount_minor": 5000, "currency": "USD", "exponent": 2}}
        });
        let profile = json!({
            "account": {"email": "a@b.c"},
            "organization": {"organization_type": "claude_max", "rate_limit_tier": "default_claude_max_5x"}
        });
        let report = report(&usage, &profile);
        assert_eq!(report.plan.as_deref(), Some("Max 5x"));
        assert_eq!(report.account.as_deref(), Some("a@b.c"));
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
                ("5h", "5 hours", Some(0.3), Some(1_767_243_600_000)),
                ("weekly", "7 days", Some(0.125), None),
                ("weekly:fable", "Fable · 7 days", Some(0.8), None),
                ("spend", "Extra usage", Some(0.25), None),
            ]
        );

        let mut headers = HeaderMap::new();
        for (name, value) in [
            ("anthropic-ratelimit-unified-5h-utilization", "0.5"),
            ("anthropic-ratelimit-unified-5h-reset", "1767243600"),
            ("anthropic-ratelimit-unified-7d-reset", "1767243600"),
        ] {
            headers.insert(name, value.parse().unwrap());
        }
        let observed = AnthropicAuthUsage {
            http: reqwest::Client::new(),
        }
        .observe(200, &headers);
        assert_eq!(
            observed,
            [UsageLimit {
                id: "5h".into(),
                label: "5 hours".into(),
                used: Some(0.5),
                resets_at: Some(1_767_243_600_000),
                window_ms: Some(FIVE_HOURS_MS),
                ..Default::default()
            }]
        );
    }
}
