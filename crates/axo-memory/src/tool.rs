//! `memory_search`, the one memory tool: everything else about notes is plain file
//! work on `memory://` (see `scheme.rs`).

use std::{borrow::Cow, sync::Arc};

use async_trait::async_trait;
use axo_agent::{ExecutionMode, Replay, Tool, ToolContext, ToolError, ToolOutput};
use axo_ai::{ToolCall, UserBlock};
use serde::Deserialize;
use serde_json::{Value, json};

use crate::{Banks, Guard, Memory, Result, SearchRequest, invalid, links::Direction};

const DESCRIPTION: &str = "Search long-term memory notes by keywords and meaning. Each result gives the note's memory:// path and the notes it links to (→) and from (←); read and edit notes through those paths.";

const GUIDELINES: &[&str] = &[
    "Long-term memory holds notes from past sessions: decisions and their reasons, project facts, and the user's preferences. Search it when a task depends on history that is not in the repository or this conversation.",
    "Memory records what was true when written. When it conflicts with the current code or the user, those win.",
];

pub struct SearchTool {
    memory: Arc<Memory>,
    banks: Banks,
    guard: Option<Guard>,
    description: String,
    guidelines: Vec<Cow<'static, str>>,
    parameters: Value,
}

impl SearchTool {
    pub fn new(memory: Arc<Memory>, banks: Banks) -> Self {
        Self {
            memory,
            banks,
            guard: None,
            description: DESCRIPTION.into(),
            guidelines: GUIDELINES.iter().map(|text| Cow::Borrowed(*text)).collect(),
            parameters: parameters(),
        }
    }

    pub fn description(mut self, text: impl Into<String>) -> Self {
        self.description = text.into();
        self
    }

    pub fn guidelines(mut self, bullets: Vec<Cow<'static, str>>) -> Self {
        self.guidelines = bullets;
        self
    }

    pub fn guard(mut self, guard: Guard) -> Self {
        self.guard = Some(guard);
        self
    }
}

fn parameters() -> Value {
    let string = json!({"type": "string"});
    let strings = json!({"type": "array", "items": {"type": "string"}});
    json!({
        "type": "object",
        "properties": {
            "query": string,
            "types": {"type": "array", "items": {"type": "string", "enum": ["entity", "observation", "relation"]}},
            "categories": strings,
            "tags": strings,
            "path": {"type": "string", "description": "Bank-relative folder or file prefix."},
            "after": {"type": "string", "description": "ISO date; notes updated after it."},
            "valid_at": {"type": "string", "description": "ISO date or instant; facts valid then."},
            "all_banks": {"type": "boolean"},
            "limit": {"type": "integer", "minimum": 1, "maximum": 50, "default": 10}
        },
        "required": ["query"],
        "additionalProperties": false
    })
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct SearchArgs {
    query: String,
    #[serde(default)]
    types: Vec<String>,
    #[serde(default)]
    categories: Vec<String>,
    #[serde(default)]
    tags: Vec<String>,
    path: Option<String>,
    after: Option<String>,
    valid_at: Option<String>,
    #[serde(default)]
    all_banks: bool,
    limit: Option<usize>,
}

#[async_trait]
impl Tool for SearchTool {
    fn name(&self) -> &str {
        "memory_search"
    }
    fn description(&self) -> &str {
        &self.description
    }
    fn parameters(&self) -> &Value {
        &self.parameters
    }
    fn execution_mode(&self) -> ExecutionMode {
        ExecutionMode::Parallel
    }
    fn replay(&self, _call: &ToolCall) -> Replay {
        Replay::Safe
    }
    fn prompt_guidelines(&self) -> &[Cow<'static, str>] {
        &self.guidelines
    }
    async fn execute(&self, cx: ToolContext) -> Result<ToolOutput, ToolError> {
        if let Some(guard) = &self.guard {
            guard(&self.banks).map_err(ToolError)?;
        }
        self.run(Value::Object(cx.call.arguments.clone()))
            .await
            .map_err(|error| ToolError(error.to_string()))
    }
}

impl SearchTool {
    async fn run(&self, value: Value) -> Result<ToolOutput> {
        let args: SearchArgs = serde_json::from_value(value)?;
        if !args
            .types
            .iter()
            .all(|t| matches!(t.as_str(), "entity" | "observation" | "relation"))
        {
            return Err(invalid("Invalid memory row type"));
        }
        let limit = args.limit.unwrap_or(10);
        if !(1..=50).contains(&limit) {
            return Err(invalid("limit must be between 1 and 50"));
        }
        let hits = self
            .memory
            .search(
                &self.banks,
                SearchRequest {
                    query: args.query,
                    types: args.types,
                    categories: args.categories,
                    tags: args.tags,
                    path: args.path,
                    after: args.after,
                    valid_at: args.valid_at,
                    all_banks: args.all_banks,
                    limit,
                },
            )
            .await?;
        let text = if hits.is_empty() {
            "No matching notes.".into()
        } else {
            hits.iter()
                .enumerate()
                .map(|(i, hit)| {
                    let mut entry = format!(
                        "{}. {} ({}) [{}{}]\n   {}",
                        i + 1,
                        hit.hit.title,
                        hit.url,
                        hit.hit.row_type,
                        hit.hit
                            .category
                            .as_ref()
                            .map(|c| format!("/{c}"))
                            .unwrap_or_default(),
                        hit.hit.snippet.trim(),
                    );
                    for link in &hit.links {
                        let arrow = match link.direction {
                            Direction::Outgoing => "→",
                            Direction::Incoming => "←",
                        };
                        entry += &format!("\n   {arrow} {} {}", link.relation_type, link.url);
                    }
                    entry
                })
                .collect::<Vec<_>>()
                .join("\n")
        };
        Ok(ToolOutput {
            content: vec![UserBlock::Text { text }],
            details: Some(json!({"hits": hits, "semantic": self.memory.has_embedder()})),
            ..ToolOutput::default()
        })
    }
}
