//! The subscription exchange: a request axo's Anthropic Messages adapter built, sent to the
//! fixed first-party endpoint with the OAuth token as a bearer and Claude Code's identity.

use async_trait::async_trait;
use axo_ai::{Error, Model, ProviderRequest, ProviderResponse, ProviderTransport, RequestContext};
use reqwest::header::{AUTHORIZATION, HeaderMap, HeaderValue};
use serde_json::json;

use crate::{BASE_URL, PROVIDER};

const SUBSCRIPTION_IDENTITY: &str = "You are Claude Code, Anthropic's official CLI for Claude.";
const OAUTH_BETA: &str = "oauth-2025-04-20";
/// The request headers the subscription endpoint receives; every other one is dropped.
const ALLOWED_HEADERS: [&str; 5] = [
    "anthropic-version",
    "anthropic-beta",
    "accept",
    "content-type",
    "anthropic-dangerous-direct-browser-access",
];

pub(crate) struct SubscriptionTransport {
    pub(crate) http: reqwest::Client,
}

#[async_trait]
impl ProviderTransport for SubscriptionTransport {
    async fn send(
        &self,
        _model: &Model,
        request: &ProviderRequest,
        _context: &RequestContext,
    ) -> Result<ProviderResponse, Error> {
        let request = outgoing(&self.http, request)?;
        Ok(self.http.execute(request).await?.into())
    }
}

/// The request as sent: fixed destination, allowlisted headers, bearer token, OAuth beta
/// and the identity system block.
fn outgoing(http: &reqwest::Client, request: &ProviderRequest) -> Result<reqwest::Request, Error> {
    let login_required = || Error::LoginRequired {
        provider: PROVIDER.to_owned(),
    };
    let token = request
        .headers
        .get("x-api-key")
        .and_then(|value| value.to_str().ok())
        .filter(|token| !token.trim().is_empty())
        .ok_or_else(login_required)?;
    let mut bearer = HeaderValue::from_str(&format!("Bearer {token}"))
        .map_err(|_| Error::Auth("Invalid Claude subscription access token".into()))?;
    bearer.set_sensitive(true);

    let mut headers = HeaderMap::new();
    for name in ALLOWED_HEADERS {
        for value in request.headers.get_all(name) {
            headers.append(name, value.clone());
        }
    }
    let mut betas: Vec<String> = headers
        .get_all("anthropic-beta")
        .iter()
        .filter_map(|value| value.to_str().ok())
        .flat_map(|value| value.split(','))
        .map(str::trim)
        .filter(|beta| !beta.is_empty())
        .map(str::to_owned)
        .collect();
    if !betas.iter().any(|beta| beta == OAUTH_BETA) {
        betas.push(OAUTH_BETA.to_owned());
    }
    headers.insert(
        "anthropic-beta",
        HeaderValue::from_str(&betas.join(","))
            .map_err(|_| Error::Config("invalid anthropic-beta header".into()))?,
    );
    headers.insert(AUTHORIZATION, bearer);

    let mut body = request.body.clone();
    if let Some(object) = body.as_object_mut() {
        let system = object.entry("system").or_insert_with(|| json!([]));
        if let Some(blocks) = system.as_array_mut() {
            blocks.insert(0, json!({ "type": "text", "text": SUBSCRIPTION_IDENTITY }));
        }
    }
    Ok(http
        .post(format!("{BASE_URL}/v1/messages?beta=true"))
        .headers(headers)
        .body(serde_json::to_vec(&body)?)
        .build()?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::Value;

    fn request(headers: &[(&str, &str)], body: Value) -> ProviderRequest {
        let mut map = HeaderMap::new();
        for (name, value) in headers {
            map.append(
                reqwest::header::HeaderName::from_bytes(name.as_bytes()).unwrap(),
                HeaderValue::from_str(value).unwrap(),
            );
        }
        ProviderRequest {
            url: "https://attacker.invalid/v1/messages?beta=true".into(),
            headers: map,
            body,
        }
    }

    fn sent_body(request: &reqwest::Request) -> Value {
        serde_json::from_slice(request.body().unwrap().as_bytes().unwrap()).unwrap()
    }

    #[test]
    fn subscription_requests_go_first_party_with_bearer_beta_and_identity() {
        let http = reqwest::Client::new();
        let sent = outgoing(
            &http,
            &request(
                &[
                    ("anthropic-version", "2023-06-01"),
                    ("anthropic-beta", "inline-tools-2026-09-15"),
                    ("accept", "application/json"),
                    ("content-type", "application/json"),
                    ("x-api-key", "synthetic-subscription-access"),
                    ("x-custom", "injected"),
                    ("host", "attacker.invalid"),
                ],
                json!({"system": [{"type": "text", "text": "Host instructions."}], "messages": []}),
            ),
        )
        .unwrap();
        assert_eq!(
            sent.url().as_str(),
            "https://api.anthropic.com/v1/messages?beta=true"
        );
        let headers = sent.headers();
        assert_eq!(
            headers[AUTHORIZATION],
            "Bearer synthetic-subscription-access"
        );
        assert!(headers[AUTHORIZATION].is_sensitive());
        for dropped in ["x-api-key", "x-custom", "host"] {
            assert!(!headers.contains_key(dropped), "kept {dropped}");
        }
        assert_eq!(headers["anthropic-version"], "2023-06-01");
        assert_eq!(
            headers["anthropic-beta"],
            "inline-tools-2026-09-15,oauth-2025-04-20"
        );
        assert_eq!(
            sent_body(&sent)["system"],
            json!([
                {"type": "text", "text": SUBSCRIPTION_IDENTITY},
                {"type": "text", "text": "Host instructions."}
            ])
        );

        // No system prompt and the beta already present: one identity block, no duplicate beta.
        let sent = outgoing(
            &http,
            &request(
                &[("anthropic-beta", OAUTH_BETA), ("x-api-key", "token")],
                json!({"messages": []}),
            ),
        )
        .unwrap();
        assert_eq!(sent.headers()["anthropic-beta"], OAUTH_BETA);
        assert_eq!(
            sent_body(&sent)["system"],
            json!([{"type": "text", "text": SUBSCRIPTION_IDENTITY}])
        );

        for token in [None, Some(""), Some(" \t")] {
            let headers: Vec<_> = token
                .map(|token| ("x-api-key", token))
                .into_iter()
                .collect();
            assert!(matches!(
                outgoing(&http, &request(&headers, json!({}))),
                Err(Error::LoginRequired { .. })
            ));
        }
    }
}
