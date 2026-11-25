//! The Anthropic subscription provider for axo: claude.ai OAuth, the account's Claude models,
//! its usage limits, and the subscription exchange over axo's Anthropic Messages adapter. A
//! host registers it with `Models::register(owner, axo_anthropic_auth::provider()?)`.

use std::sync::Arc;
use std::time::Duration;

use axo_ai::{Error, ModelAuth};
use reqwest::header::{AUTHORIZATION, HeaderValue};
use serde_json::Value;

mod login;
mod models;
mod transport;
mod usage;

/// The provider id.
pub const PROVIDER: &str = "anthropic-auth";
/// The only destination subscription requests go to.
const BASE_URL: &str = "https://api.anthropic.com";
const ACCOUNT_TIMEOUT: Duration = Duration::from_secs(10);

/// The provider to pass to `Models::register`. Login, inference and account requests share
/// a dedicated client that follows no redirects and never retries on its own.
pub fn provider() -> Result<axo_ai::Provider, Error> {
    let http = login::client()?;
    Ok(axo_ai::Provider {
        id: PROVIDER.to_owned(),
        models: Vec::new(),
        env: Vec::new(),
        requires_key: true,
        oauth_only: true,
        login: Some(Arc::new(login::AnthropicAuthLogin::new(http.clone()))),
        rewrite: None,
        transport: Some(Arc::new(transport::AnthropicAuthTransport {
            http: http.clone(),
        })),
        list_models: Some(Arc::new(models::AnthropicAuthModels { http: http.clone() })),
        usage: Some(Arc::new(usage::AnthropicAuthUsage { http })),
    })
}

/// `GET {BASE_URL}{path}` as the signed-in account: the OAuth token as a bearer, with the
/// OAuth beta.
async fn account_get(http: &reqwest::Client, auth: &ModelAuth, path: &str) -> Result<Value, Error> {
    let token = auth
        .api_key
        .as_deref()
        .filter(|token| !token.trim().is_empty())
        .ok_or_else(|| Error::LoginRequired {
            provider: PROVIDER.to_owned(),
        })?;
    let mut bearer = HeaderValue::from_str(&format!("Bearer {token}"))
        .map_err(|_| Error::Auth("Invalid Anthropic access token".into()))?;
    bearer.set_sensitive(true);
    Ok(http
        .get(format!("{BASE_URL}{path}"))
        .header(AUTHORIZATION, bearer)
        .header("anthropic-version", "2023-06-01")
        .header("anthropic-beta", transport::OAUTH_BETA)
        .header("accept", "application/json")
        .timeout(ACCOUNT_TIMEOUT)
        .send()
        .await?
        .error_for_status()?
        .json()
        .await?)
}
