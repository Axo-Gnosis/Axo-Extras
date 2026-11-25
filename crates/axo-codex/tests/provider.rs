use std::collections::BTreeMap;
use std::io::{Read, Write};
use std::net::TcpListener;
use std::sync::Arc;
use std::thread::JoinHandle;

use axo_ai::testing::{fixture_model, http_header_values, serve_http};
use axo_ai::{
    Api, ApiKind, AssistantEvent, AssistantMessage, Cost, CostTier, Message, Model, ModelAuth,
    ModelsRequest, ResponsesApi, ServiceTier, SimpleStreamOptions, StreamOptions, ThinkingLevel,
    Transcript, UserBlock, UserMessage,
};
use base64::Engine;
use futures::StreamExt;
use serde_json::{Value, json};
use tokio_util::sync::CancellationToken;

/// A ChatGPT-style access token whose account claim is `test`.
fn token() -> String {
    let payload = base64::engine::general_purpose::URL_SAFE_NO_PAD
        .encode(r#"{"https://api.openai.com/auth":{"chatgpt_account_id":"test"}}"#);
    format!("header.{payload}.signature")
}

async fn stream(
    model: &Model,
    session_id: &str,
    headers: BTreeMap<String, String>,
) -> AssistantMessage {
    let provider = axo_codex::provider(reqwest::Client::new());
    let events: Vec<_> = ResponsesApi::new(reqwest::Client::new())
        .stream(
            model,
            Transcript::new(vec![Message::User(UserMessage {
                content: vec![UserBlock::Text { text: "hi".into() }],
                timestamp: 0,
            })]),
            StreamOptions {
                simple: SimpleStreamOptions {
                    max_tokens: Some(42),
                    session_id: Some(session_id.into()),
                    ..Default::default()
                },
                auth: ModelAuth {
                    api_key: Some(token()),
                    headers,
                    ..Default::default()
                },
                rewrite: None,
                transport: provider.transport,
                observe: None,
            },
            CancellationToken::new(),
        )
        .collect()
        .await;
    match events.into_iter().last().unwrap() {
        AssistantEvent::Error { message, .. } | AssistantEvent::Done { message, .. } => message,
        event => panic!("not terminal: {event:?}"),
    }
}

#[tokio::test]
async fn transport_sends_codex_requests_and_explains_usage_limits() {
    let done = r#"data: {"type":"response.completed","response":{"status":"completed","usage":{"input_tokens":4}}}"#;
    let ok = format!(
        "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{done}\n\n",
        done.len() + 2
    );
    let limited = r#"{"error":{"code":"usage_limit_reached","message":"original provider diagnostic","plan_type":"PLUS"}}"#;
    let limited = format!(
        "HTTP/1.1 429 Too Many Requests\r\nContent-Type: application/json\r\nx-should-retry: false\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{limited}",
        limited.len()
    );
    let (url, server) = serve_http([ok, limited].into_iter());
    let model = |base_url: String| {
        fixture_model(
            axo_codex::PROVIDER,
            "gpt-test",
            "GPT Test",
            ApiKind::OpenAiResponses,
            base_url,
            272_000,
        )
    };

    let session = "s".repeat(80);
    let reply = stream(&model(url.clone()), &session, BTreeMap::new()).await;
    assert_eq!(reply.error_message, None);

    let reply = stream(
        &model(format!("{url}/codex")),
        "session",
        BTreeMap::from([("chatgpt-account-id".into(), "from-credential".into())]),
    )
    .await;
    let failure = reply.failure.as_ref().unwrap();
    assert_eq!(failure.status, Some(429));
    assert_eq!(failure.code.as_deref(), Some("usage_limit_reached"));
    let message = reply.error_message.as_deref().unwrap();
    assert!(
        message.contains("You have hit your ChatGPT usage limit (plus plan)."),
        "{message}"
    );
    assert!(
        message.contains("original provider diagnostic"),
        "{message}"
    );

    let requests = server.join().unwrap();
    let (head, body) = &requests[0];
    assert!(head.starts_with("POST /codex/responses "), "{head}");
    let header = |head, name| http_header_values(head, name).collect::<Vec<_>>();
    assert_eq!(header(head, "openai-beta"), ["responses=experimental"]);
    assert_eq!(header(head, "originator"), ["axo"]);
    assert_eq!(header(head, "chatgpt-account-id"), ["test"]);
    assert_eq!(header(head, "session-id"), ["s".repeat(64)]);
    assert_eq!(header(head, "x-client-request-id"), ["s".repeat(64)]);
    assert_eq!(body["text"], json!({"verbosity": "low"}));
    assert_eq!(body["instructions"], "You are a helpful assistant.");
    assert!(body.get("max_output_tokens").is_none());
    assert_eq!(body["model"], "gpt-test");

    let (head, _) = &requests[1];
    assert!(head.starts_with("POST /codex/responses "), "{head}");
    assert_eq!(header(head, "chatgpt-account-id"), ["from-credential"]);
    assert_eq!(header(head, "session-id"), ["session"]);
}

/// Serves one GET with `body` and returns the request head.
fn serve_get(body: Value) -> (String, JoinHandle<String>) {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let url = format!("http://{}", listener.local_addr().unwrap());
    let thread = std::thread::spawn(move || {
        let (mut stream, _) = listener.accept().unwrap();
        let mut head = Vec::new();
        let mut chunk = [0; 4096];
        while !head.windows(4).any(|window| window == b"\r\n\r\n") {
            let count = stream.read(&mut chunk).unwrap();
            assert_ne!(count, 0);
            head.extend_from_slice(&chunk[..count]);
        }
        let body = body.to_string();
        write!(
            stream,
            "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
            body.len()
        )
        .unwrap();
        String::from_utf8(head).unwrap()
    });
    (url, thread)
}

fn levels(model: &Model) -> Vec<ThinkingLevel> {
    model
        .thinking_level_map
        .iter()
        .flatten()
        .filter(|(_, value)| value.is_some())
        .map(|(level, _)| *level)
        .collect()
}

#[tokio::test]
async fn listing_prices_the_account_list_from_the_catalog_and_falls_back_to_the_floor() {
    let row = |slug: &str, visibility: &str| {
        json!({
            "slug": slug, "display_name": "GPT New", "context_window": 500000,
            "input_modalities": ["text", "image"],
            "supported_reasoning_levels": [{"effort": "low"}, {"effort": "high"}, {"effort": "ultra"}],
            "additional_speed_tiers": ["fast"],
            "visibility": visibility
        })
    };
    let mut bare = row("gpt-bare", "list");
    bare["supported_reasoning_levels"] = json!([{"effort": "ultra"}]);
    let mut windowless = row("gpt-windowless", "list");
    windowless["context_window"] = Value::Null;
    let (url, server) = serve_get(json!({"models": [
        row("gpt-new", "list"), row("gpt-hidden", "hide"), bare, windowless, {"display_name": "malformed"}
    ]}));

    let mut openai = fixture_model(
        "openai",
        "gpt-new",
        "GPT New",
        ApiKind::OpenAiResponses,
        "https://api.openai.com/v1".into(),
        1_000_000,
    );
    openai.cost = Some(Cost {
        input: 3.0,
        output: 9.0,
        tiers: vec![CostTier {
            input_tokens_above: 272_000,
            input: 6.0,
            output: 13.5,
            cache_read: 0.0,
            cache_write: 0.0,
        }],
        ..Default::default()
    });
    openai.thinking_level_map = Some(BTreeMap::from([(ThinkingLevel::Off, Some("none".into()))]));
    openai.service_tiers = BTreeMap::from([(ServiceTier::Fast, 2.0)]);
    let mut impostor = openai.clone();
    impostor.provider = "elsewhere".into();
    impostor.cost = None;
    let catalog: Arc<[Arc<Model>]> = vec![Arc::new(impostor), Arc::new(openai)].into();

    let list = axo_codex::provider(reqwest::Client::new())
        .list_models
        .unwrap();
    let listed = list
        .list(ModelsRequest {
            auth: Some(ModelAuth {
                api_key: Some(token()),
                base_url: Some(format!("{url}/codex")),
                ..Default::default()
            }),
            catalog: catalog.clone(),
            extended_context: false,
        })
        .await
        .unwrap();
    let head = server.join().unwrap();
    assert!(
        head.starts_with("GET /codex/models?client_version=999.0.0 "),
        "{head}"
    );
    assert_eq!(
        http_header_values(&head, "originator").collect::<Vec<_>>(),
        ["axo"]
    );
    assert_eq!(
        http_header_values(&head, "chatgpt-account-id").collect::<Vec<_>>(),
        ["test"]
    );
    assert_eq!(
        http_header_values(&head, "authorization").collect::<Vec<_>>(),
        [format!("Bearer {}", token())]
    );

    let ids: Vec<_> = listed.iter().map(|model| model.id.as_str()).collect();
    assert_eq!(ids, ["gpt-new", "gpt-bare"]);
    let new = &listed[0];
    assert_eq!(new.provider, axo_codex::PROVIDER);
    assert_eq!(new.api, ApiKind::OpenAiResponses);
    assert_eq!(new.base_url, axo_codex::BASE_URL);
    let cost = new.cost.as_ref().unwrap();
    assert_eq!((cost.input, cost.output), (3.0, 9.0));
    assert_eq!(
        levels(new),
        [ThinkingLevel::Off, ThinkingLevel::Low, ThinkingLevel::High]
    );
    assert_eq!(
        new.service_tiers,
        BTreeMap::from([(ServiceTier::Fast, 2.0)])
    );
    assert_eq!(new.context_window, 272_000, "capped at the price tier");
    let bare = &listed[1];
    assert!(!bare.reasoning);
    assert_eq!(bare.cost, None);
    assert_eq!(bare.context_window, 500_000);

    for (extended_context, window) in [(false, 272_000), (true, 500_000)] {
        let floor = list
            .list(ModelsRequest {
                auth: None,
                catalog: catalog.clone(),
                extended_context,
            })
            .await
            .unwrap();
        assert_eq!(floor.len(), 7);
        let sol = floor
            .iter()
            .find(|model| model.id == "gpt-6.1-sol")
            .unwrap();
        assert_eq!(sol.context_window, window);
        assert_eq!(sol.cost.as_ref().unwrap().input, 2.0);
        assert!(!levels(sol).contains(&ThinkingLevel::Off));
    }
}
