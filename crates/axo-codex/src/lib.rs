//! The ChatGPT Codex provider for axo: ChatGPT OAuth, the account's model list, and the
//! Codex exchange over axo's OpenAI Responses adapter. A host registers it with
//! `Models::register(owner, axo_codex::provider(http))`.

use std::sync::Arc;

mod login;
mod models;
mod transport;

/// The provider id.
pub const PROVIDER: &str = "openai-codex";
/// The ChatGPT backend serving Codex models.
pub const BASE_URL: &str = "https://chatgpt.com/backend-api";

/// The provider to pass to `Models::register`. `http` serves login, model listing and
/// inference.
pub fn provider(http: reqwest::Client) -> axo_ai::Provider {
    axo_ai::Provider {
        id: PROVIDER.to_owned(),
        models: models::floor(),
        env: Vec::new(),
        requires_key: true,
        oauth_only: false,
        login: Some(Arc::new(login::CodexLogin::new(http.clone()))),
        rewrite: None,
        transport: Some(Arc::new(transport::CodexTransport { http: http.clone() })),
        list_models: Some(Arc::new(models::CodexModels { http })),
    }
}
