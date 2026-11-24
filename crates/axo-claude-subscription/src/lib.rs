//! The Claude subscription provider for axo: claude.ai OAuth, the catalog's first-party
//! Claude models, and the subscription exchange over axo's Anthropic Messages adapter. A
//! host registers it with `Models::register(owner, axo_claude_subscription::provider()?)`.

use std::sync::Arc;

mod login;
mod models;
mod transport;

/// The provider id.
pub const PROVIDER: &str = "anthropic-subscription";
/// The only destination subscription requests go to.
const BASE_URL: &str = "https://api.anthropic.com";

/// The provider to pass to `Models::register`. Login and inference share a dedicated
/// client that follows no redirects and never retries on its own.
pub fn provider() -> Result<axo_ai::Provider, axo_ai::Error> {
    let http = login::client()?;
    Ok(axo_ai::Provider {
        id: PROVIDER.to_owned(),
        models: Vec::new(),
        env: Vec::new(),
        requires_key: true,
        oauth_only: true,
        login: Some(Arc::new(login::ClaudeSubscriptionLogin::new(http.clone()))),
        rewrite: None,
        transport: Some(Arc::new(transport::SubscriptionTransport { http })),
        list_models: Some(Arc::new(models::SubscriptionModels)),
    })
}
