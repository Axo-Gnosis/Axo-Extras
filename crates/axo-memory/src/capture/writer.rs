//! The writer protocol: the extract and classify prompts, the grammar of their
//! replies, and the round trip that asks the writer agent each one.

use std::{sync::LazyLock, time::Duration};

use axo::{AgentSpec, PromptRequest, PromptResult, Session};
use axo_agent::{AgentMessage, RunOutcome};
use axo_ai::Message;
use regex::Regex;

use super::transcript::message_text;
use crate::{Result, index::LocatedObservation, invalid};

const EXTRACT_PROMPT: &str = r#"Project: {project}
Date: {date}

Session excerpt:
<session>
{transcript}
</session>

List the facts from this excerpt that will still matter in future sessions: decisions and their reasons, facts about how the project works, and preferences or corrections the user stated.
Leave out task progress, things being tried, tool output, and anything only the assistant claimed without the user accepting it.

One fact per line, at most 12 lines, in exactly this form:
<page> :: [<category>] <fact>

<page> is "{project}/<Topic>" for facts about the project, or "me/<Topic>" for facts about the user that hold across projects. Topics are short noun phrases; reuse the same page for related facts.
<category> is one of: decision, fact, preference, reason, problem.
<fact> is one self-contained sentence. Link another page by writing [[{project}/<Topic>]] inside it when it clearly relates.

If nothing qualifies, answer exactly: NONE"#;

const CLASSIFY_PROMPT: &str = r#"New fact:
[{category}] {fact}

Existing notes:
{numbered}

Compare the new fact with the existing notes. Answer with exactly one of:
NEW          - it says something none of them say
SAME <n>     - note <n> already says this
REPLACES <n> - it updates or contradicts note <n>, which is now out of date
SKIP         - it is not worth keeping after all"#;

static CANDIDATE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"^(?P<page>[^:\n]{1,120}?)\s*::\s*\[(?P<category>decision|fact|preference|reason|problem)\]\s+(?P<fact>.{8,400})$").unwrap()
});
static CLASSIFICATION: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"^(NEW|SKIP|(SAME|REPLACES)\s+(\d+))\s*$").unwrap());

/// The most candidates one extract reply yields; the prompt asks for the same.
const MAX_FACTS: usize = 12;

/// The bundled writer agent. Hosts may change its `model` or pass their own spec.
pub fn writer_agent() -> AgentSpec {
    let (spec, diagnostics) = AgentSpec::parse("memory-writer", include_str!("writer.md"));
    assert!(diagnostics.is_empty(), "writer.md: {diagnostics:?}");
    spec.expect("writer.md is a valid agent")
}

pub(super) struct Candidate {
    pub page: String,
    pub category: String,
    pub fact: String,
}

/// How a candidate relates to the observations it was compared with; `n` indexes them.
pub(super) enum Choice {
    New,
    Same(usize),
    Replace(usize),
    Skip,
}

/// Asks for the durable facts in `transcript`.
pub(super) async fn extract(
    session: &Session,
    project: &str,
    date: &str,
    transcript: &str,
) -> Result<Vec<Candidate>> {
    let prompt = EXTRACT_PROMPT
        .replace("{project}", project)
        .replace("{date}", date)
        .replace("{transcript}", transcript);
    let reply = ask(session, prompt).await?;
    Ok(reply
        .lines()
        .filter_map(|line| CANDIDATE.captures(line))
        .take(MAX_FACTS)
        .map(|found| Candidate {
            page: found["page"].trim().into(),
            category: found["category"].into(),
            fact: found["fact"].into(),
        })
        .collect())
}

/// Asks how `candidate` relates to `observations`; `New` without asking when there are none.
pub(super) async fn classify(
    session: &Session,
    candidate: &Candidate,
    observations: &[LocatedObservation],
) -> Result<Choice> {
    if observations.is_empty() {
        return Ok(Choice::New);
    }
    // Classify full observation text, not truncated/highlighted search snippets.
    let numbered = observations
        .iter()
        .enumerate()
        .map(|(n, located)| {
            format!(
                "{}. ({}) [{}] {}",
                n + 1,
                located.entity.title,
                located.observation.category,
                located.observation.content
            )
        })
        .collect::<Vec<_>>()
        .join("\n");
    let prompt = CLASSIFY_PROMPT
        .replace("{category}", &candidate.category)
        .replace("{fact}", &candidate.fact)
        .replace("{numbered}", &numbered);
    Ok(parse_choice(
        &ask(session, prompt).await?,
        observations.len(),
    ))
}

fn parse_choice(reply: &str, count: usize) -> Choice {
    let Some(line) = reply.lines().find(|line| !line.trim().is_empty()) else {
        return Choice::Skip;
    };
    let Some(captures) = CLASSIFICATION.captures(line) else {
        return Choice::Skip;
    };
    if &captures[1] == "NEW" {
        return Choice::New;
    }
    let n = captures
        .get(3)
        .and_then(|n| n.as_str().parse::<usize>().ok())
        .filter(|n| *n > 0 && *n <= count);
    match (captures.get(2).map(|kind| kind.as_str()), n) {
        (Some("REPLACES"), Some(n)) => Choice::Replace(n - 1),
        (Some("SAME"), Some(n)) => Choice::Same(n - 1),
        _ => Choice::Skip,
    }
}

async fn ask(session: &Session, text: String) -> Result<String> {
    let result = tokio::time::timeout(
        Duration::from_secs(600),
        session.prompt_with(PromptRequest {
            input: text.into(),
            request_id: None,
        }),
    )
    .await
    .map_err(|_| invalid("Memory writer timed out"))??;
    if !matches!(
        result,
        PromptResult::Settled {
            outcome: RunOutcome::Done,
            ..
        }
    ) {
        return Err(invalid("Memory writer did not finish successfully"));
    }
    let attach = session.attach(None).await;
    let message = attach
        .live
        .messages
        .iter()
        .rev()
        .find_map(|message| match message {
            AgentMessage::Llm(message @ Message::Assistant(_)) => Some(message),
            _ => None,
        })
        .ok_or_else(|| invalid("Memory writer returned no assistant reply"))?;
    Ok(message_text(message))
}
