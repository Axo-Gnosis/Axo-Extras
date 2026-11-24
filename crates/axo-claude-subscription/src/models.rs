//! Subscription models: the catalog's first-party Anthropic Messages rows, served under
//! this provider without prices or service tiers.

use async_trait::async_trait;
use axo_ai::{ApiKind, Error, Model, ModelsRequest, ProviderModels};

use crate::{BASE_URL, PROVIDER};

/// The wire prefix that keeps tool names clear of Claude Code's own.
const TOOL_NAME_PREFIX: &str = "_";

pub(crate) struct SubscriptionModels;

#[async_trait]
impl ProviderModels for SubscriptionModels {
    async fn list(&self, request: ModelsRequest) -> Result<Vec<Model>, Error> {
        Ok(request
            .catalog
            .iter()
            .filter(|model| {
                model.provider == "anthropic"
                    && model.api == ApiKind::AnthropicMessages
                    && model.base_url == BASE_URL
            })
            .map(|model| {
                let mut model = model.as_ref().clone();
                // The subscription is not billed per token, but it keeps the window the API
                // prices as standard, whatever `extended_context` says.
                if let Some(threshold) = model
                    .cost
                    .as_ref()
                    .and_then(|cost| cost.tiers.iter().map(|tier| tier.input_tokens_above).min())
                {
                    model.context_window = model.context_window.min(threshold);
                }
                model.provider = PROVIDER.to_owned();
                model.cost = None;
                model.service_tiers.clear();
                model.compat.tool_name_prefix = Some(TOOL_NAME_PREFIX.to_owned());
                model
            })
            .collect())
    }
}
