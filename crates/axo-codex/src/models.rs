//! Codex models: the ChatGPT account's list from `GET {base}/codex/models`, priced from the
//! catalog's `openai` rows, with a fixed floor standing in before login.

use std::str::FromStr;
use std::time::Duration;

use async_trait::async_trait;
use axo_ai::{
    ApiKind, Compat, Cost, CostTier, Error, InputKind, Model, ModelAuth, ModelsRequest,
    ProviderModels, ServiceTier, ThinkingLevel, ThinkingLevelMap,
};
use reqwest::header::{HeaderMap, HeaderName, HeaderValue};
use serde::Deserialize;
use serde_json::Value;

use crate::login::account_id;
use crate::{BASE_URL, PROVIDER};

const MAX_TOKENS: u64 = 128_000;
const OPENAI_LONG_CONTEXT_INPUT_THRESHOLD: u64 = 272_000;
/// Efforts of the floor rows, which have no account list to read them from.
const FLOOR_EFFORTS: [&str; 5] = ["low", "medium", "high", "xhigh", "max"];
const TIMEOUT: Duration = Duration::from_secs(15);
/// The server hides models whose `minimal_client_version` is above this; axo is not a Codex
/// CLI build, so it asks for everything.
const CLIENT_VERSION: &str = "999.0.0";

struct FloorRow {
    id: &'static str,
    name: &'static str,
    context_window: u64,
    // Input, output, cache read, cache write.
    prices: [f64; 4],
    // Whether reasoning effort "none" (thinking Off) is accepted.
    off: bool,
}

/// Offline floor only; never add rows. New models arrive through the account list.
const FLOOR: [FloorRow; 7] = [
    FloorRow {
        id: "gpt-6.1-sol",
        name: "GPT-6.1 Sol",
        context_window: 500_000,
        prices: [2.0, 10.0, 0.1, 2.5],
        off: false,
    },
    FloorRow {
        id: "gpt-6-astra",
        name: "GPT-6 Astra",
        context_window: 272_000,
        prices: [10.0, 50.0, 1.0, 12.5],
        off: false,
    },
    FloorRow {
        id: "gpt-6-sol",
        name: "GPT-6 Sol",
        context_window: 272_000,
        prices: [2.0, 10.0, 0.2, 2.5],
        off: true,
    },
    FloorRow {
        id: "gpt-6-luna",
        name: "GPT-6 Luna",
        context_window: 272_000,
        prices: [0.1, 0.5, 0.01, 0.125],
        off: true,
    },
    FloorRow {
        id: "gpt-5.6-luna",
        name: "GPT-5.6 Luna",
        context_window: 272_000,
        prices: [0.2, 1.2, 0.02, 0.25],
        off: true,
    },
    FloorRow {
        id: "gpt-5.6-sol",
        name: "GPT-5.6 Sol",
        context_window: 272_000,
        prices: [4.0, 20.0, 0.4, 5.0],
        off: true,
    },
    FloorRow {
        id: "gpt-5.6-terra",
        name: "GPT-5.6 Terra",
        context_window: 272_000,
        prices: [2.0, 12.0, 0.2, 2.5],
        off: true,
    },
];

/// One Codex model before pricing: a row of the account list, or a floor row.
struct CodexRow {
    id: String,
    name: String,
    context_window: u64,
    image: bool,
    /// Reasoning efforts as listed; ones axo has no thinking level for are ignored.
    efforts: Vec<String>,
    tiers: Vec<String>,
}

fn floor_rows() -> Vec<CodexRow> {
    FLOOR
        .iter()
        .map(|row| CodexRow {
            id: row.id.to_owned(),
            name: row.name.to_owned(),
            context_window: row.context_window,
            image: true,
            efforts: FLOOR_EFFORTS.map(str::to_owned).to_vec(),
            tiers: Vec::new(),
        })
        .collect()
}

/// The floor as the provider's static list: floor prices, full windows.
pub(crate) fn floor() -> Vec<Model> {
    floor_rows()
        .into_iter()
        .map(|row| codex_model(row, &[], true))
        .collect()
}

/// Lists the account's Codex models; the floor while logged out.
pub(crate) struct CodexModels {
    pub(crate) http: reqwest::Client,
}

#[async_trait]
impl ProviderModels for CodexModels {
    async fn list(&self, request: ModelsRequest) -> Result<Vec<Model>, Error> {
        let rows = match request.auth {
            Some(auth) => discover(&self.http, auth).await?,
            None => floor_rows(),
        };
        Ok(rows
            .into_iter()
            .map(|row| codex_model(row, &request.catalog, request.extended_context))
            .collect())
    }
}

#[derive(Deserialize)]
struct ModelList {
    #[serde(default)]
    models: Vec<Value>,
}

/// One row of the list; it carries much more (prompts, tool settings) that axo does not use.
#[derive(Deserialize)]
struct Row {
    slug: String,
    #[serde(default)]
    display_name: Option<String>,
    #[serde(default)]
    context_window: Option<u64>,
    #[serde(default)]
    input_modalities: Vec<String>,
    #[serde(default)]
    supported_reasoning_levels: Vec<Level>,
    #[serde(default)]
    service_tiers: Vec<RowTier>,
    #[serde(default)]
    additional_speed_tiers: Vec<String>,
    #[serde(default)]
    visibility: Option<String>,
}

#[derive(Deserialize)]
struct Level {
    effort: String,
}

#[derive(Deserialize)]
struct RowTier {
    id: String,
}

/// The account's listed models; hidden ones and rows without a context window are skipped.
async fn discover(http: &reqwest::Client, auth: ModelAuth) -> Result<Vec<CodexRow>, Error> {
    let base = auth.base_url.as_deref().unwrap_or(BASE_URL);
    let base = base.trim_end_matches('/');
    let base = base
        .strip_suffix("/codex/responses")
        .or_else(|| base.strip_suffix("/codex"))
        .unwrap_or(base);
    // Later layers win, as for inference requests.
    let token = auth.api_key.unwrap_or_default();
    let mut headers = HeaderMap::new();
    for (name, value) in [("originator".to_owned(), "axo".to_owned())]
        .into_iter()
        .chain(account_id(&token).map(|id| ("chatgpt-account-id".to_owned(), id)))
        .chain(auth.headers)
        .chain([("authorization".to_owned(), format!("Bearer {token}"))])
    {
        headers.insert(
            HeaderName::from_str(&name)
                .map_err(|_| Error::Config(format!("invalid header name: {name}")))?,
            HeaderValue::from_str(&value)
                .map_err(|_| Error::Config(format!("invalid value for header {name}")))?,
        );
    }
    let list: ModelList = http
        .get(format!(
            "{base}/codex/models?client_version={CLIENT_VERSION}"
        ))
        .headers(headers)
        .timeout(TIMEOUT)
        .send()
        .await?
        .error_for_status()?
        .json()
        .await?;
    let rows: Vec<CodexRow> = list
        .models
        .iter()
        .filter_map(|raw| match Row::deserialize(raw) {
            Ok(row) => Some(row),
            Err(error) => {
                tracing::debug!("skipping malformed Codex model row: {error}");
                None
            }
        })
        .filter(|row| row.visibility.as_deref() != Some("hide"))
        .filter_map(|row| {
            let context_window = row.context_window.filter(|&window| window > 0)?;
            let mut tiers: Vec<_> = row.service_tiers.into_iter().map(|tier| tier.id).collect();
            if row.additional_speed_tiers.iter().any(|tier| tier == "fast")
                && !tiers.iter().any(|tier| tier == "priority")
            {
                tiers.push("priority".to_owned());
            }
            Some(CodexRow {
                context_window,
                name: row.display_name.unwrap_or_else(|| row.slug.clone()),
                id: row.slug,
                image: row.input_modalities.iter().any(|kind| kind == "image"),
                efforts: row
                    .supported_reasoning_levels
                    .into_iter()
                    .map(|level| level.effort)
                    .collect(),
                tiers,
            })
        })
        .collect();
    if rows.is_empty() {
        return Err(Error::Config("Codex model list is empty".into()));
    }
    Ok(rows)
}

/// A Codex model. Prices, Off support and tier multipliers come from the catalog's `openai`
/// row with the same id, else the floor; limits never do (the catalog lists the API's
/// window, not the one Codex serves).
fn codex_model(row: CodexRow, catalog: &[std::sync::Arc<Model>], extended_context: bool) -> Model {
    let openai = catalog
        .iter()
        .find(|model| model.provider == "openai" && model.id == row.id);
    let floor = FLOOR.iter().find(|floor| floor.id == row.id);
    let (cost, off) = match (openai, floor) {
        (Some(openai), _) => (
            openai.cost.clone(),
            openai
                .thinking_level_map
                .as_ref()
                .and_then(|map| map.get(&ThinkingLevel::Off))
                .is_some_and(|off| off.as_deref() == Some("none")),
        ),
        (None, Some(floor)) => (Some(floor_cost(floor.prices)), floor.off),
        (None, None) => (None, false),
    };
    let mut thinking = thinking_map(&row.efforts);
    if off {
        thinking
            .entry(ThinkingLevel::Off)
            .or_default()
            .get_or_insert_with(|| "none".to_owned());
    }
    // A list with no effort axo knows and no Off is a plain model, not a reasoning one with
    // nothing to select.
    let reasoning = thinking.values().any(Option::is_some);
    let mut model = Model {
        id: row.id,
        name: row.name,
        api: ApiKind::OpenAiResponses,
        provider: PROVIDER.to_owned(),
        base_url: BASE_URL.to_owned(),
        reasoning,
        thinking_level_map: reasoning.then_some(thinking),
        input: if row.image {
            vec![InputKind::Text, InputKind::Image]
        } else {
            vec![InputKind::Text]
        },
        cost,
        service_tiers: row
            .tiers
            .iter()
            .filter_map(|id| tier_named(id))
            .map(|tier| {
                let factor = openai.and_then(|openai| openai.service_tiers.get(&tier));
                (tier, factor.copied().unwrap_or(1.0))
            })
            .collect(),
        context_window: row.context_window,
        max_tokens: MAX_TOKENS,
        compat: Compat::default(),
    };
    if !extended_context {
        cap_at_price_tier(&mut model);
    }
    model
}

/// The tier an account-list tier id names on the Responses wire.
fn tier_named(id: &str) -> Option<ServiceTier> {
    match id {
        "flex" => Some(ServiceTier::Flex),
        "priority" => Some(ServiceTier::Fast),
        "ultrafast" => Some(ServiceTier::Ultrafast),
        _ => None,
    }
}

/// Stop at the lowest context price tier's threshold.
fn cap_at_price_tier(model: &mut Model) {
    if let Some(threshold) = model
        .cost
        .as_ref()
        .and_then(|cost| cost.tiers.iter().map(|tier| tier.input_tokens_above).min())
        && threshold < model.context_window
    {
        model.context_window = threshold;
    }
}

/// Thinking levels from a list of reasoning efforts: each named level maps to itself and
/// the rest are unsupported. Off maps to `"none"` when listed and is unsupported otherwise.
/// Unknown effort strings are ignored.
fn thinking_map(efforts: &[String]) -> ThinkingLevelMap {
    let listed = |name: &str| efforts.iter().any(|effort| effort == name);
    let mut map: ThinkingLevelMap = ThinkingLevel::ALL[1..]
        .iter()
        .map(|&level| {
            let name = level.as_str();
            (level, listed(name).then(|| name.to_owned()))
        })
        .collect();
    map.insert(
        ThinkingLevel::Off,
        listed("none").then(|| "none".to_owned()),
    );
    map
}

/// A floor row's prices with OpenAI's long-context tier (double input and cache, 1.5×
/// output above 272K).
fn floor_cost([input, output, cache_read, cache_write]: [f64; 4]) -> Cost {
    let round = |value: f64| (value * 1e6).round() / 1e6;
    Cost {
        input,
        output,
        cache_read,
        cache_write,
        tiers: vec![CostTier {
            input_tokens_above: OPENAI_LONG_CONTEXT_INPUT_THRESHOLD,
            input: round(input * 2.0),
            output: round(output * 1.5),
            cache_read: round(cache_read * 2.0),
            cache_write: round(cache_write * 2.0),
        }],
    }
}
