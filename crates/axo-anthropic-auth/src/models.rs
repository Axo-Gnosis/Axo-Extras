//! The account's models: the ids `GET /v1/models` lists for the signed-in account, described
//! by the catalog's first-party Anthropic Messages rows and served under this provider
//! without prices or service tiers. Logged out, every first-party row.

use std::collections::HashSet;

use async_trait::async_trait;
use axo_ai::{ApiKind, Error, Model, ModelsRequest, ProviderModels};

use crate::{BASE_URL, PROVIDER, account_get};

/// The wire prefix that keeps tool names clear of Claude Code's own.
const TOOL_NAME_PREFIX: &str = "_";

pub(crate) struct AnthropicAuthModels {
    pub(crate) http: reqwest::Client,
}

#[async_trait]
impl ProviderModels for AnthropicAuthModels {
    async fn list(&self, request: ModelsRequest) -> Result<Vec<Model>, Error> {
        let account: Option<HashSet<String>> = match &request.auth {
            Some(auth) => {
                let list = account_get(&self.http, auth, "/v1/models?limit=1000").await?;
                Some(
                    list["data"]
                        .as_array()
                        .into_iter()
                        .flatten()
                        .filter_map(|row| row["id"].as_str().map(str::to_owned))
                        .collect(),
                )
            }
            None => None,
        };
        let models: Vec<Model> = request
            .catalog
            .iter()
            .filter(|model| {
                model.provider == "anthropic"
                    && model.api == ApiKind::AnthropicMessages
                    && model.base_url == BASE_URL
                    && account.as_ref().is_none_or(|ids| ids.contains(&model.id))
            })
            .map(|model| subscription_model(model))
            .collect();
        // ponytail: an account model the catalog does not describe yet waits for the catalog; build a default row if that lag matters
        if let Some(ids) = &account {
            for id in ids
                .iter()
                .filter(|id| !models.iter().any(|model| &model.id == *id))
            {
                tracing::debug!("{PROVIDER}: account model {id} is not in the catalog");
            }
            if models.is_empty() {
                return Err(Error::Config(format!(
                    "{PROVIDER}: the account lists no model the catalog describes"
                )));
            }
        }
        Ok(models)
    }
}

fn subscription_model(model: &Model) -> Model {
    let mut model = model.clone();
    // The subscription is not billed per token, but it keeps the window the API prices as
    // standard, whatever `extended_context` says.
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
}
