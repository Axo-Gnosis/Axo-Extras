//! Capture: reading what a session gained since its last capture, asking the writer
//! agent which facts are durable, and recording them in the default bank.

pub(crate) mod log;
mod transcript;
mod writer;

use std::{
    path::{Path, PathBuf},
    sync::Arc,
};

use axo::{AgentSpec, Axo, RecoveryPolicy, Session, SessionOptions};
use chrono::Utc;
use tokio::sync::OwnedMutexGuard;
use tokio_util::sync::CancellationToken;

use crate::{
    Bank, Banks, Error, Memory, Result, SearchRequest, bank::checked_path, edit, format,
    index::LocatedObservation, invalid, links,
};
use log::{ClosedLine, Log, NewFact, Undo};
use transcript::{Chunk, transcript};
pub use writer::writer_agent;
use writer::{Candidate, Choice};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FactOutcome {
    New,
    Updated,
    Known,
}

#[derive(Clone, Debug)]
pub struct CapturedFact {
    pub id: i64,
    pub outcome: FactOutcome,
    pub category: String,
    pub text: String,
    /// The observation an `Updated` fact replaced.
    pub previous: Option<String>,
    /// Bank-relative note name without `.md`.
    pub note: String,
    pub path: String,
    pub reverted: bool,
}

#[derive(Clone, Debug)]
pub struct CaptureRun {
    pub id: i64,
    pub session_id: String,
    pub cwd: String,
    pub bank: String,
    /// What the host attached to the request, returned unchanged.
    pub host: serde_json::Value,
    pub started_at: String,
    /// `None` while the capture runs.
    pub ended_at: Option<String>,
    pub error: Option<String>,
    /// The writer agent's model could not be resolved.
    pub no_model: bool,
    pub facts: Vec<CapturedFact>,
}

pub struct CaptureRequest {
    /// The session file to read.
    pub session: PathBuf,
    /// Facts go to the default bank; duplicates are looked up there.
    pub banks: Banks,
    /// Runs the extract and classify prompts. A user agent file of the same name overrides it,
    /// by axo's own precedence.
    pub agent: AgentSpec,
    /// Stored with the run and returned as `CaptureRun::host`.
    pub host: serde_json::Value,
    /// Called when the run starts and after each chunk.
    pub progress: Option<Box<dyn Fn(CaptureRun) + Send + Sync>>,
    pub cancel: CancellationToken,
}

impl Memory {
    /// Reads what the session gained since its last capture and records durable facts.
    /// `None` when there was nothing to read: a child session, a writer session, or no new
    /// user text. A failure after the run started is recorded in `CaptureRun::error`.
    pub async fn capture(
        self: &Arc<Self>,
        axo: &Arc<Axo>,
        request: CaptureRequest,
    ) -> Result<Option<CaptureRun>> {
        // ponytail: one capture at a time; per-session locks if a host needs parallel captures
        let _capturing = self.capturing.lock().await;
        let (header, entries) = axo.open_session_read(&request.session).await?;
        if header.spawn.is_some() || header.cwd == self.writer_dir() {
            return Ok(None);
        }
        let session_id = header.id.clone();
        let cursor = {
            let id = session_id.clone();
            self.log(move |log| log.cursor(&id)).await?.unwrap_or(0)
        };
        let (chunks, last_seq) = transcript(&header, &entries, cursor);
        if chunks.is_empty() {
            if let Some(seq) = last_seq {
                let id = session_id.clone();
                self.log(move |log| log.save_cursor(&id, seq)).await?;
            }
            return Ok(None);
        }
        let project = header
            .cwd
            .file_name()
            .map_or_else(|| "root".into(), |name| name.to_string_lossy().into_owned());
        let capture_id = {
            let (id, cwd) = (
                session_id.clone(),
                header.cwd.to_string_lossy().into_owned(),
            );
            let bank = request.banks.default_bank().name().to_owned();
            let host = request.host.clone();
            self.log(move |log| log.start(&id, &cwd, &bank, &host))
                .await?
        };
        self.report(&request, capture_id).await;
        let result = self
            .capture_chunks(axo, &request, &session_id, &project, chunks, capture_id)
            .await;
        let error = result.err();
        if let Some(error) = &error {
            tracing::warn!(%error, session_id, "Memory capture failed");
        }
        let no_model = matches!(
            error,
            Some(Error::Axo(
                axo::Error::UnknownModelRole(_)
                    | axo::Error::ModelNotFound(_)
                    | axo::Error::NoModel
            ))
        );
        let error = error.map(|error| error.to_string());
        let run = self
            .log(move |log| {
                log.finish(capture_id, error.as_deref(), no_model)?;
                log.get(capture_id)
            })
            .await?;
        Ok(Some(run))
    }

    /// Removes a captured fact's line and reopens the observation it replaced.
    /// Fails when the note changed since, or `banks` has no bank named like the run's.
    pub async fn revert_fact(self: &Arc<Self>, fact_id: i64, banks: &Banks) -> Result<CaptureRun> {
        let (fact, run) = self
            .log(move |log| {
                let fact = log.fact_edit(fact_id)?;
                let run = log.get(fact.capture_id)?;
                Ok((fact, run))
            })
            .await?;
        let undo = fact
            .undo
            .ok_or_else(|| invalid("A known fact has nothing to revert"))?;
        if fact.reverted {
            return Err(invalid("This fact was already reverted"));
        }
        let bank = banks
            .get(&run.bank)
            .ok_or_else(|| invalid(format!("Memory bank {} is no longer configured", run.bank)))?
            .clone();
        let path = checked_path(&bank, Path::new(&fact.path))?;
        let previous = match undo.closed {
            Some(closed) => Some((
                checked_path(&bank, Path::new(&closed.path))?,
                closed.before,
                closed.after,
            )),
            None => None,
        };
        let guards = lock_notes(
            [Some(path.clone()), previous.as_ref().map(|p| p.0.clone())]
                .into_iter()
                .flatten(),
        )
        .await;
        let memory = Arc::clone(self);
        tokio::task::spawn_blocking(move || -> Result<()> {
            let _guards = guards;
            let changed =
                || invalid("This note changed since the capture. Open the note to edit it.");
            let removed = edit::replace_line(&std::fs::read_to_string(&path)?, &undo.line, None)
                .ok_or_else(changed)?;
            // Both edits are checked before either is written.
            let (note, restored) = match previous {
                Some((previous_path, before, after)) if previous_path == path => (
                    edit::replace_line(&removed, &after, Some(&before)).ok_or_else(changed)?,
                    None,
                ),
                Some((previous_path, before, after)) => {
                    let text = std::fs::read_to_string(&previous_path)?;
                    let restored =
                        edit::replace_line(&text, &after, Some(&before)).ok_or_else(changed)?;
                    (removed, Some((previous_path, restored)))
                }
                None => (removed, None),
            };
            // The old fact is back first: a failed second write leaves both, never neither.
            if let Some((previous_path, restored)) = restored {
                atomic_write(&previous_path, &restored)?;
            }
            atomic_write(&path, &note)?;
            memory.log.mark_reverted(fact_id)
        })
        .await??;
        self.indexed(vec![bank], |_| Ok(())).await?;
        let id = run.id;
        self.log(move |log| log.get(id)).await
    }

    /// Newest first, optionally of one session or bank; `limit` is clamped to 1..=500.
    /// Blocks briefly.
    pub fn captures(
        &self,
        session_id: Option<&str>,
        bank: Option<&str>,
        limit: u32,
    ) -> Result<Vec<CaptureRun>> {
        self.log.list(session_id, bank, limit)
    }

    /// The run holding `fact_id`.
    pub fn capture_of_fact(&self, fact_id: i64) -> Result<CaptureRun> {
        self.log.get(self.log.fact_edit(fact_id)?.capture_id)
    }

    /// The session sequence number captured up to.
    pub fn cursor(&self, session_id: &str) -> Result<Option<u64>> {
        self.log.cursor(session_id)
    }

    /// Runs `f` against the capture log off the async runtime.
    async fn log<T: Send + 'static>(
        self: &Arc<Self>,
        f: impl FnOnce(&Log) -> Result<T> + Send + 'static,
    ) -> Result<T> {
        let memory = Arc::clone(self);
        tokio::task::spawn_blocking(move || f(&memory.log)).await?
    }

    async fn report(self: &Arc<Self>, request: &CaptureRequest, capture_id: i64) {
        let Some(progress) = &request.progress else {
            return;
        };
        match self.log(move |log| log.get(capture_id)).await {
            Ok(run) => progress(run),
            Err(error) => tracing::warn!(%error, "Could not read a memory capture"),
        }
    }

    async fn capture_chunks(
        self: &Arc<Self>,
        axo: &Arc<Axo>,
        request: &CaptureRequest,
        session_id: &str,
        project: &str,
        chunks: Vec<Chunk>,
        capture_id: i64,
    ) -> Result<()> {
        for chunk in chunks.into_iter().take(20) {
            let cwd = self.writer_dir();
            tokio::fs::create_dir_all(&cwd).await?;
            let session = axo
                .create_session(SessionOptions {
                    cwd,
                    agent: Some(request.agent.name.clone()),
                    host: axo::Registrations {
                        definitions: axo::Definitions {
                            agents: vec![request.agent.clone()],
                            ..Default::default()
                        },
                        ..Default::default()
                    },
                    ask_enabled: false,
                    keywords_enabled: false,
                    recovery: RecoveryPolicy::MarkInterrupted,
                    ..Default::default()
                })
                .await?;
            let result = tokio::select! {
                biased;
                () = request.cancel.cancelled() => Err(invalid("Capture was cancelled.")),
                result = self.capture_chunk(&request.banks, &session, &chunk.text, project, session_id, capture_id) => result,
            };
            let journal = session.path().to_owned();
            session.shutdown().await;
            drop(session);
            // The excerpt must not outlive its capture: remove the journal and its session directory.
            for result in [
                tokio::fs::remove_file(&journal).await,
                tokio::fs::remove_dir_all(journal.with_extension("")).await,
            ] {
                if let Err(error) = result
                    && error.kind() != std::io::ErrorKind::NotFound
                {
                    tracing::warn!(%error, "Could not remove a memory writer journal");
                }
            }
            result?;
            let (id, seq) = (session_id.to_owned(), chunk.seq);
            self.log(move |log| log.save_cursor(&id, seq)).await?;
            self.report(request, capture_id).await;
        }
        Ok(())
    }

    async fn capture_chunk(
        self: &Arc<Self>,
        banks: &Banks,
        session: &Session,
        transcript: &str,
        project: &str,
        session_id: &str,
        capture_id: i64,
    ) -> Result<()> {
        let bank = banks.default_bank();
        let date = Utc::now().format("%Y-%m-%d").to_string();
        for candidate in writer::extract(session, project, &date, transcript).await? {
            let hits = self
                .search(
                    banks,
                    SearchRequest {
                        query: candidate.fact.clone(),
                        types: vec!["observation".into()],
                        limit: Some(5),
                        ..Default::default()
                    },
                )
                .await?;
            let mut observations = self
                .indexed(vec![bank.clone()], move |index| {
                    hits.iter()
                        .map(|hit| {
                            index
                                .observation(hit.hit.row_id)?
                                .ok_or_else(|| invalid("Memory observation changed during capture"))
                        })
                        .collect::<Result<Vec<_>>>()
                })
                .await?;
            let fact = match writer::classify(session, &candidate, &observations).await? {
                Choice::Skip => continue,
                Choice::Same(n) => {
                    let known = observations.swap_remove(n);
                    let path = bank.dir().join(&known.entity.file_path);
                    NewFact {
                        outcome: FactOutcome::Known,
                        category: known.observation.category,
                        text: known.observation.content,
                        previous: None,
                        note: note_name(bank, &path),
                        path: path.to_string_lossy().into_owned(),
                        undo: None,
                    }
                }
                Choice::New => {
                    self.apply_candidate(bank, candidate, FactOutcome::New, None, &date, session_id)
                        .await?
                }
                Choice::Replace(n) => {
                    let old = observations.swap_remove(n);
                    self.apply_candidate(
                        bank,
                        candidate,
                        FactOutcome::Updated,
                        Some(old),
                        &date,
                        session_id,
                    )
                    .await?
                }
            };
            self.log(move |log| log.add_fact(capture_id, fact)).await?;
        }
        // Also sync chunks with no accepted candidates, without creating a bank directory.
        self.indexed(vec![bank.clone()], |_| Ok(())).await?;
        Ok(())
    }

    async fn apply_candidate(
        self: &Arc<Self>,
        bank: &Bank,
        candidate: Candidate,
        outcome: FactOutcome,
        old: Option<LocatedObservation>,
        date: &str,
        session_id: &str,
    ) -> Result<NewFact> {
        let mut relative = PathBuf::new();
        for segment in candidate.page.split('/') {
            let segment = segment.trim();
            if segment.is_empty() || segment == "." || segment == ".." {
                return Err(invalid("Invalid memory page"));
            }
            relative.push(segment.replace(['/', '\\', ':', '*', '?', '"', '<', '>', '|'], "-"));
        }
        let title = relative
            .file_name()
            .ok_or_else(|| invalid("Memory page has no title"))?
            .to_string_lossy()
            .into_owned();
        let page = candidate.page.clone();
        let path = self
            .indexed(vec![bank.clone()], {
                let bank = bank.clone();
                move |index| {
                    let bank_id = index.upsert_bank(bank.dir())?;
                    let path = match links::resolve(index, bank_id, &page)? {
                        Some(id) => bank.dir().join(
                            index
                                .entity(id)?
                                .ok_or_else(|| invalid("Memory note disappeared"))?
                                .file_path,
                        ),
                        None => bank.dir().join(format!("{}.md", relative.display())),
                    };
                    checked_path(&bank, &path)
                }
            })
            .await?;
        let old_path = old
            .as_ref()
            .map(|hit| checked_path(bank, &bank.dir().join(&hit.entity.file_path)))
            .transpose()?;
        let previous = old.as_ref().map(|old| old.observation.content.clone());
        let (category, text) = (candidate.category.clone(), candidate.fact.clone());
        let (date, session_id) = (date.to_owned(), session_id.to_owned());
        let guards = lock_notes([Some(path.clone()), old_path.clone()].into_iter().flatten()).await;
        let (path, undo) = tokio::task::spawn_blocking(move || -> Result<(PathBuf, Undo)> {
            let _guards = guards;
            let mut text = match std::fs::read_to_string(&path) {
                Ok(text) => text,
                Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                    edit::new_note(&title, "note", &[], "## Observations\n")
                }
                Err(error) => return Err(error.into()),
            };
            let mut closed_note = None;
            let mut closed_line = None;
            let temporal = if let (Some(hit), Some(old_path)) = (old, old_path) {
                let old_text = if old_path == path {
                    &text
                } else {
                    &std::fs::read_to_string(&old_path)?
                };
                let line = hit.observation.line;
                // The note may have changed since the search: refuse to close a different observation.
                if !format::parse(old_text).observations.iter().any(|o| {
                    o.line == line
                        && o.category == hit.observation.category
                        && o.content == hit.observation.content
                        && o.context == hit.observation.context
                }) {
                    return Err(invalid("Memory observation changed during capture"));
                }
                let closed = edit::close_observation(old_text, line, &date);
                if closed != *old_text {
                    // What a revert restores: the closed line exactly as it read before.
                    let before = old_text
                        .lines()
                        .nth(line - 1)
                        .unwrap_or_default()
                        .to_owned();
                    let after = closed.lines().nth(line - 1).unwrap_or_default().to_owned();
                    closed_line = Some(ClosedLine {
                        path: old_path.to_string_lossy().into_owned(),
                        before,
                        after,
                    });
                }
                if old_path == path {
                    text = closed;
                } else if closed != *old_text {
                    closed_note = Some((old_path, closed));
                }
                format!("@[{date},) ")
            } else {
                String::new()
            };
            let line = format!(
                "- [{}] {temporal}{} (session {session_id} {date})",
                candidate.category, candidate.fact
            );
            let appended = edit::append_observation(&text, &line);
            if appended == text {
                return Err(invalid("Cannot edit malformed frontmatter"));
            }
            atomic_write(&path, &appended)?;
            // The replacement is on disk first: a failed close leaves two valid facts, never none.
            if let Some((old_path, closed)) = closed_note {
                atomic_write(&old_path, &closed)?;
            }
            Ok((
                path,
                Undo {
                    line,
                    closed: closed_line,
                },
            ))
        })
        .await??;
        Ok(NewFact {
            outcome,
            category,
            text,
            previous,
            note: note_name(bank, &path),
            path: path.to_string_lossy().into_owned(),
            undo: Some(undo),
        })
    }
}

/// Takes the same process-wide per-file lock axo's `write` and `edit` hold, so the agent's
/// edits through memory:// and the writer's never interleave. Sorted: no lock-order cycles.
async fn lock_notes(paths: impl IntoIterator<Item = PathBuf>) -> Vec<OwnedMutexGuard<()>> {
    let mut paths: Vec<_> = paths.into_iter().collect();
    paths.sort();
    paths.dedup();
    let mut guards = Vec::with_capacity(paths.len());
    for path in &paths {
        guards.push(axo::tools::lock_file(path).await);
    }
    guards
}

fn atomic_write(path: &Path, text: &str) -> Result<()> {
    use std::io::Write;
    let parent = path
        .parent()
        .ok_or_else(|| invalid("Note has no parent directory"))?;
    std::fs::create_dir_all(parent)?;
    let mut file = tempfile::Builder::new()
        .prefix(".axo-memory-")
        .suffix(".tmp")
        .tempfile_in(parent)?;
    file.write_all(text.as_bytes())?;
    file.as_file().sync_all()?;
    file.persist(path).map_err(|error| error.error)?;
    Ok(())
}

/// `Axome/Pause menu` for `<bank>/Axome/Pause menu.md`.
fn note_name(bank: &Bank, path: &Path) -> String {
    let relative = path
        .strip_prefix(bank.dir())
        .unwrap_or(path)
        .to_string_lossy();
    relative.strip_suffix(".md").unwrap_or(&relative).to_owned()
}
