//! A session's new user and assistant text, split into chunks the writer reads.

use std::collections::HashSet;

use axo::store::{Entry, Header, SeqEntry};
use axo_agent::AgentMessage;
use axo_ai::{AssistantBlock, Message, UserBlock};

// ponytail: fixed chunk size; make it part of `CaptureRequest` if a host needs another.
const CHUNK_CHARS: usize = 12_000;

pub(crate) struct Chunk {
    pub text: String,
    /// The session sequence number this chunk reads up to.
    pub seq: u64,
}

/// A message's text blocks, without thinking, tool calls or images.
pub(crate) fn message_text(message: &Message) -> impl Iterator<Item = &str> {
    let (user, assistant) = match message {
        Message::User(user) => (Some(user), None),
        Message::Assistant(assistant) => (None, Some(assistant)),
        Message::System(_) | Message::ToolResult(_) => (None, None),
    };
    user.into_iter()
        .flat_map(|user| &user.content)
        .filter_map(|block| match block {
            UserBlock::Text { text } => Some(text.as_str()),
            UserBlock::Image { .. } => None,
        })
        .chain(
            assistant
                .into_iter()
                .flat_map(|assistant| &assistant.content)
                .filter_map(|block| match block {
                    AssistantBlock::Text { text, .. } => Some(text.as_str()),
                    AssistantBlock::Thinking { .. } | AssistantBlock::ToolCall(_) => None,
                }),
        )
}

/// The chunks of `entries` after `cursor`, and the last sequence number when it moved
/// past `cursor`. No chunks when the new part holds no user text.
pub(crate) fn transcript(
    header: &Header,
    entries: &[SeqEntry],
    cursor: u64,
) -> (Vec<Chunk>, Option<u64>) {
    let notices: HashSet<_> = entries
        .iter()
        .filter_map(|seq| match &seq.entry {
            Entry::Submission {
                submission_id,
                kind: axo::SubmissionKind::Quiet | axo::SubmissionKind::Mail,
                ..
            } => Some(submission_id),
            _ => None,
        })
        .collect();
    // A fork copies its parent's entries with their original timestamps, all in one
    // fixed-width format: those at or before the fork's own were the parent's to capture.
    let inherited = |entry: &Entry| {
        header.parent_session.is_some()
            && entry
                .base()
                .is_some_and(|base| base.timestamp <= header.timestamp)
    };
    let projection = axo::store::project(entries, None);
    let mut chunks = Vec::new();
    let mut text = String::new();
    let mut length = 0;
    let mut seq = cursor;
    let mut has_user = false;
    let last_seq = entries
        .last()
        .map(|entry| entry.seq)
        .filter(|seq| *seq > cursor);
    for &index in projection.path() {
        let entry = &entries[index];
        if entry.seq <= cursor || inherited(&entry.entry) {
            continue;
        }
        let Entry::Message {
            message: AgentMessage::Llm(message),
            submission_id,
            ..
        } = &entry.entry
        else {
            continue;
        };
        let role = match message {
            Message::User(_)
                if !submission_id
                    .as_ref()
                    .is_some_and(|id| notices.contains(id)) =>
            {
                "user"
            }
            Message::Assistant(_) => "assistant",
            _ => continue,
        };
        let content = message_text(message).collect::<Vec<_>>().join("\n");
        if content.trim().is_empty() {
            continue;
        }
        has_user |= role == "user";
        let rendered: String = format!("[{role}]\n{content}")
            .chars()
            .take(CHUNK_CHARS)
            .collect();
        let count = rendered.chars().count();
        if !text.is_empty() && length + 2 + count > CHUNK_CHARS {
            chunks.push(Chunk {
                text: std::mem::take(&mut text),
                seq,
            });
            length = 0;
        }
        if !text.is_empty() {
            text.push_str("\n\n");
            length += 2;
        }
        text.push_str(&rendered);
        length += count;
        seq = entry.seq;
    }
    if !text.is_empty() {
        chunks.push(Chunk {
            text,
            seq: last_seq.unwrap_or(seq),
        });
    }
    if !has_user {
        chunks.clear();
    }
    (chunks, last_seq)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn header(parent: bool) -> Header {
        let parent = if parent {
            r#","parentSession":"/s/parent.jsonl""#
        } else {
            ""
        };
        serde_json::from_str(&format!(
            r#"{{"type":"session","version":1,"id":"s","timestamp":"2026-11-27T10:00:00.000Z","cwd":"/p"{parent}}}"#
        ))
        .unwrap()
    }

    /// `(role, text, timestamp)` messages at seq 1.., each the parent of the next.
    fn entries(messages: &[(&str, &str, &str)]) -> Vec<SeqEntry> {
        messages
            .iter()
            .enumerate()
            .map(|(seq, (role, text, timestamp))| {
                let parent = seq.checked_sub(1).map_or("null".into(), |p| format!(r#""e{p}""#));
                let extra = if *role == "assistant" {
                    r#","api":"test","provider":"test","model":"test","usage":{"input":0,"output":0,"cacheRead":0,"cacheWrite":0,"totalTokens":0,"cost":{"input":0,"output":0,"cacheRead":0,"cacheWrite":0,"total":0}},"stopReason":"stop""#
                } else {
                    ""
                };
                let line = format!(
                    r#"{{"type":"message","id":"e{seq}","parentId":{parent},"timestamp":"{timestamp}","message":{{"role":"{role}","content":[{{"type":"text","text":"{text}"}}],"timestamp":0{extra}}}}}"#
                );
                SeqEntry {
                    seq: seq as u64 + 1,
                    entry: serde_json::from_str(&line).unwrap(),
                }
            })
            .collect()
    }

    #[test]
    fn reads_only_new_text_of_its_own() {
        let before = "2026-11-27T09:00:00.000Z";
        let after = "2026-11-27T11:00:00.000Z";
        let log = entries(&[
            ("user", "old question", before),
            ("assistant", "old answer", before),
            ("user", "new question", after),
        ]);

        let (chunks, last) = transcript(&header(false), &log, 2);
        assert_eq!(last, Some(3));
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].text, "[user]\nnew question");
        assert_eq!(chunks[0].seq, 3);

        let (chunks, _) = transcript(&header(true), &log[..2], 0);
        assert!(
            chunks.is_empty(),
            "a fork's inherited entries are the parent's"
        );
        let (chunks, _) = transcript(&header(true), &log, 0);
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].text, "[user]\nnew question");

        let (chunks, last) = transcript(&header(false), &log[..2], 2);
        assert!(chunks.is_empty() && last.is_none());
        let answers = entries(&[("assistant", "only an answer", after)]);
        let (chunks, last) = transcript(&header(false), &answers, 0);
        assert!(chunks.is_empty(), "no user text, nothing to capture");
        assert_eq!(last, Some(1));
    }
}
