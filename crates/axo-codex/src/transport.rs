//! The Codex exchange: a plain Responses request, sent to the ChatGPT backend's Codex path
//! with Codex headers and body defaults. Port of pi `api/openai-codex-responses.ts`
//! (`buildRequestBody`, `parseErrorResponse` and the SSE headers).

use async_trait::async_trait;
use axo_ai::{
    Error, Model, ProviderRequest, ProviderResponse, ProviderTransport, RequestContext, now_ms,
};
use futures::StreamExt;
use reqwest::header::{AUTHORIZATION, CONTENT_LENGTH, HeaderMap, HeaderValue};
use serde_json::{Value, json};

use crate::login::account_id;

/// pi `OPENAI_PROMPT_CACHE_KEY_MAX_LENGTH`.
const SESSION_ID_MAX_CHARS: usize = 64;

pub(crate) struct CodexTransport {
    pub(crate) http: reqwest::Client,
}

#[async_trait]
impl ProviderTransport for CodexTransport {
    async fn send(
        &self,
        _model: &Model,
        request: &ProviderRequest,
        context: &RequestContext,
    ) -> Result<ProviderResponse, Error> {
        let mut headers = request.headers.clone();
        codex_headers(&mut headers, context)?;
        let response = self
            .http
            .post(codex_url(&request.url))
            .headers(headers)
            .json(&codex_body(request.body.clone()))
            .send()
            .await?;
        if response.status().is_success() {
            return Ok(response.into());
        }
        let status = response.status().as_u16();
        let mut headers = response.headers().clone();
        headers.remove(CONTENT_LENGTH);
        let raw = response.text().await?;
        let parsed: Option<Value> = serde_json::from_str(&raw).ok();
        let message = parse_error_response(status, &raw, parsed.as_ref());
        // Keep the original error object so axo still classifies its `code`.
        let mut body = parsed.filter(Value::is_object).unwrap_or_else(|| json!({}));
        if !body["error"].is_object() {
            body["error"] = json!({});
        }
        body["error"]["message"] = json!(message);
        let body = serde_json::to_vec(&body)?;
        Ok(ProviderResponse {
            status,
            headers,
            body: futures::stream::once(async { Ok(body.into()) }).boxed(),
        })
    }
}

/// The Codex endpoint for a Responses request URL (`{base}/responses`): bases ending in
/// `/codex` or `/codex/responses` are used as given, others gain `/codex/responses`.
fn codex_url(url: &str) -> String {
    let base = url.strip_suffix("/responses").unwrap_or(url);
    if base.ends_with("/codex/responses") {
        base.to_owned()
    } else if base.ends_with("/codex") {
        format!("{base}/responses")
    } else {
        format!("{base}/codex/responses")
    }
}

/// Codex protocol headers. The account id derived from a ChatGPT token is only a default: a
/// credential's `chatgpt-account-id` header wins, and other tokens send none.
fn codex_headers(headers: &mut HeaderMap, context: &RequestContext) -> Result<(), Error> {
    let value = |value: &str| {
        HeaderValue::from_str(value)
            .map_err(|_| Error::Config(format!("invalid Codex header value: {value}")))
    };
    headers.insert("openai-beta", value("responses=experimental")?);
    if !headers.contains_key("originator") {
        headers.insert("originator", value("axo")?);
    }
    if !headers.contains_key("chatgpt-account-id")
        && let Some(id) = headers
            .get(AUTHORIZATION)
            .and_then(|value| value.to_str().ok())
            .and_then(|value| value.strip_prefix("Bearer "))
            .and_then(account_id)
    {
        headers.insert("chatgpt-account-id", value(&id)?);
    }
    if let Some(session_id) = context.session_id.as_deref().filter(|id| !id.is_empty()) {
        let session_id: String = session_id.chars().take(SESSION_ID_MAX_CHARS).collect();
        headers.insert("session-id", value(&session_id)?);
        headers.insert("x-client-request-id", value(&session_id)?);
    }
    Ok(())
}

/// Codex body defaults on top of the plain Responses body.
fn codex_body(mut body: Value) -> Value {
    if let Some(object) = body.as_object_mut() {
        object.insert("text".into(), json!({ "verbosity": "low" }));
        object
            .entry("instructions")
            .or_insert_with(|| json!("You are a helpful assistant."));
        object.remove("max_output_tokens");
    }
    body
}

/// pi `parseErrorResponse`: the provider's error message, or a friendly usage-limit
/// message.
fn parse_error_response(status: u16, raw: &str, body: Option<&Value>) -> String {
    let mut message = if raw.is_empty() {
        reqwest::StatusCode::from_u16(status)
            .ok()
            .and_then(|status| status.canonical_reason())
            .unwrap_or("Request failed")
            .to_owned()
    } else {
        raw.to_owned()
    };
    let mut friendly = None;
    if let Some(error) = body
        .and_then(|body| body.get("error"))
        .filter(|error| error.is_object())
    {
        let field = |key: &str| {
            error
                .get(key)
                .and_then(Value::as_str)
                .filter(|text| !text.is_empty())
        };
        let code = field("code")
            .or_else(|| field("type"))
            .unwrap_or_default()
            .to_ascii_lowercase();
        if [
            "usage_limit_reached",
            "usage_not_included",
            "rate_limit_exceeded",
        ]
        .iter()
        .any(|pattern| code.contains(pattern))
            || status == 429
        {
            let plan = field("plan_type")
                .map(|plan| format!(" ({} plan)", plan.to_lowercase()))
                .unwrap_or_default();
            let when = error
                .get("resets_at")
                .and_then(Value::as_f64)
                .filter(|resets_at| *resets_at != 0.0)
                .map(|resets_at| {
                    // Math.round: halves round up.
                    let minutes = ((resets_at * 1000.0 - now_ms() as f64) / 60_000.0 + 0.5).floor();
                    format!(" Try again in ~{} min.", minutes.max(0.0) as i64)
                })
                .unwrap_or_default();
            friendly = Some(
                format!("You have hit your ChatGPT usage limit{plan}.{when}")
                    .trim()
                    .to_owned(),
            );
        }
        message = field("message")
            .map(str::to_owned)
            .or_else(|| friendly.clone())
            .unwrap_or(message);
    }
    match friendly {
        Some(friendly) => format!("{friendly}\n{raw}"),
        None => message,
    }
}
