use std::sync::Arc;

use axo_ai::testing::fixture_model;
use axo_ai::{ApiKind, Cost, CostTier, Model, ModelsRequest, ServiceTier};

#[tokio::test]
async fn listing_keeps_first_party_messages_rows_unpriced_and_capped() {
    let row = |provider: &str, id: &str, api: ApiKind, base_url: &str| {
        let mut model = fixture_model(provider, id, id, api, base_url.into(), 1_000_000);
        model.cost = Some(Cost {
            input: 3.0,
            output: 15.0,
            tiers: vec![CostTier {
                input_tokens_above: 200_000,
                input: 6.0,
                output: 22.5,
                cache_read: 0.0,
                cache_write: 0.0,
            }],
            ..Default::default()
        });
        model.service_tiers.insert(ServiceTier::Fast, 6.0);
        Arc::new(model)
    };
    let catalog: Arc<[Arc<Model>]> = vec![
        row(
            "anthropic",
            "first-party",
            ApiKind::AnthropicMessages,
            "https://api.anthropic.com",
        ),
        row(
            "anthropic",
            "gateway",
            ApiKind::AnthropicMessages,
            "https://gw.example",
        ),
        row(
            "anthropic",
            "completions",
            ApiKind::OpenAiCompletions,
            "https://api.anthropic.com",
        ),
        row(
            "other",
            "first-party",
            ApiKind::AnthropicMessages,
            "https://api.anthropic.com",
        ),
    ]
    .into();
    let list = axo_anthropic_auth::provider().unwrap().list_models.unwrap();
    let listed = list
        .list(ModelsRequest {
            auth: None,
            catalog,
            extended_context: true,
        })
        .await
        .unwrap();
    let [model] = listed.as_slice() else {
        panic!("{listed:?}");
    };
    assert_eq!(model.id, "first-party");
    assert_eq!(model.provider, axo_anthropic_auth::PROVIDER);
    assert_eq!(model.cost, None);
    assert!(model.service_tiers.is_empty());
    assert_eq!(model.compat.tool_name_prefix.as_deref(), Some("_"));
    assert_eq!(model.context_window, 200_000);
}
