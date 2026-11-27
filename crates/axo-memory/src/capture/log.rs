//! `captures.db`: one row per capture run, one per fact it saved, updated or found
//! known, and each session's capture cursor.

use std::{
    path::PathBuf,
    sync::{Mutex, MutexGuard},
    time::Duration,
};

use chrono::Utc;
use rusqlite::{Connection, OptionalExtension, Row, params};

use super::{CaptureQuery, CaptureRun, CapturedFact, FactOutcome};
use crate::{Result, invalid, lock};

const SCHEMA: &str = r"
CREATE TABLE capture (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  session_id TEXT NOT NULL,
  cwd TEXT NOT NULL,
  bank TEXT NOT NULL,
  host TEXT NOT NULL,
  started_at TEXT NOT NULL,
  ended_at TEXT,
  error TEXT,
  no_model INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX capture_session ON capture (session_id, id);
CREATE TABLE fact (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  capture_id INTEGER NOT NULL REFERENCES capture(id) ON DELETE CASCADE,
  outcome TEXT NOT NULL,
  category TEXT NOT NULL,
  text TEXT NOT NULL,
  previous TEXT,
  note TEXT NOT NULL,
  path TEXT NOT NULL,
  line_text TEXT,
  previous_path TEXT,
  previous_before TEXT,
  previous_after TEXT,
  reverted INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX fact_capture ON fact (capture_id);
CREATE TABLE cursor (session_id TEXT PRIMARY KEY, seq INTEGER NOT NULL, updated_at TEXT NOT NULL);
";

/// What reverting a saved fact restores.
pub(crate) struct Undo {
    /// The exact line the capture appended.
    pub line: String,
    /// The observation an updated fact closed, when closing changed it.
    pub closed: Option<ClosedLine>,
}

/// A closed observation line: its note, and the line before and after closing.
pub(crate) struct ClosedLine {
    pub path: String,
    pub before: String,
    pub after: String,
}

pub(crate) struct NewFact {
    pub outcome: FactOutcome,
    pub category: String,
    pub text: String,
    pub previous: Option<String>,
    pub note: String,
    pub path: String,
    /// `None` for known facts.
    pub undo: Option<Undo>,
}

/// What a revert needs beyond the public fact.
pub(crate) struct FactEdit {
    pub capture_id: i64,
    pub reverted: bool,
    pub path: String,
    pub undo: Option<Undo>,
}

pub(crate) struct Log {
    path: PathBuf,
    conn: Mutex<Option<Connection>>,
}

fn now() -> String {
    Utc::now().to_rfc3339()
}

fn conversion(
    column: usize,
    error: impl std::error::Error + Send + Sync + 'static,
) -> rusqlite::Error {
    rusqlite::Error::FromSqlConversionFailure(column, rusqlite::types::Type::Text, Box::new(error))
}

fn outcome(row: &Row, column: usize) -> rusqlite::Result<FactOutcome> {
    match row.get_ref(column)?.as_str()? {
        "new" => Ok(FactOutcome::New),
        "updated" => Ok(FactOutcome::Updated),
        "known" => Ok(FactOutcome::Known),
        other => Err(rusqlite::Error::InvalidColumnType(
            column,
            other.into(),
            rusqlite::types::Type::Text,
        )),
    }
}

fn outcome_text(outcome: FactOutcome) -> &'static str {
    match outcome {
        FactOutcome::New => "new",
        FactOutcome::Updated => "updated",
        FactOutcome::Known => "known",
    }
}

const RUN_COLUMNS: &str = "id, session_id, cwd, bank, host, started_at, ended_at, error, no_model";

fn run(row: &Row) -> rusqlite::Result<CaptureRun> {
    let host = row.get_ref(4)?.as_str()?;
    Ok(CaptureRun {
        id: row.get(0)?,
        session_id: row.get(1)?,
        cwd: row.get(2)?,
        bank: row.get(3)?,
        host: serde_json::from_str(host).map_err(|error| conversion(4, error))?,
        started_at: row.get(5)?,
        ended_at: row.get(6)?,
        error: row.get(7)?,
        no_model: row.get(8)?,
        facts: Vec::new(),
    })
}

fn with_facts(conn: &Connection, mut run: CaptureRun) -> Result<CaptureRun> {
    let mut statement = conn.prepare_cached(
        "SELECT id, outcome, category, text, previous, note, path, reverted
         FROM fact WHERE capture_id = ?1 ORDER BY id",
    )?;
    run.facts = statement
        .query_map([run.id], |row| {
            Ok(CapturedFact {
                id: row.get(0)?,
                outcome: outcome(row, 1)?,
                category: row.get(2)?,
                text: row.get(3)?,
                previous: row.get(4)?,
                note: row.get(5)?,
                path: row.get(6)?,
                reverted: row.get(7)?,
            })
        })?
        .collect::<rusqlite::Result<_>>()?;
    Ok(run)
}

impl Log {
    pub fn new(path: PathBuf) -> Self {
        Self {
            path,
            conn: Mutex::new(None),
        }
    }

    /// The open connection; `None` when `create` is false and the file does not exist.
    fn conn(&self, create: bool) -> Result<Option<MutexGuard<'_, Option<Connection>>>> {
        let mut slot = lock(&self.conn);
        if slot.is_none() {
            if !create && !self.path.exists() {
                return Ok(None);
            }
            if let Some(parent) = self.path.parent() {
                std::fs::create_dir_all(parent)?;
            }
            let mut conn = Connection::open(&self.path)?;
            conn.busy_timeout(Duration::from_secs(5))?;
            conn.execute_batch("PRAGMA journal_mode=WAL; PRAGMA foreign_keys=ON;")?;
            let version: i32 = conn.query_row("PRAGMA user_version", [], |row| row.get(0))?;
            if version == 0 {
                let tx = conn.transaction()?;
                tx.execute_batch(SCHEMA)?;
                tx.pragma_update(None, "user_version", 1)?;
                tx.commit()?;
            }
            // Runs a previous process left open can never finish.
            conn.execute(
                "UPDATE capture SET ended_at = ?1, error = 'Interrupted before this capture finished.' WHERE ended_at IS NULL",
                [now()],
            )?;
            *slot = Some(conn);
        }
        Ok(Some(slot))
    }

    fn read<T>(
        &self,
        missing: impl FnOnce() -> Result<T>,
        f: impl FnOnce(&Connection) -> Result<T>,
    ) -> Result<T> {
        match self.conn(false)? {
            Some(guard) => f(guard.as_ref().expect("opened")),
            None => missing(),
        }
    }

    fn write<T>(&self, f: impl FnOnce(&Connection) -> Result<T>) -> Result<T> {
        let guard = self.conn(true)?.expect("created");
        f(guard.as_ref().expect("opened"))
    }

    pub fn start(
        &self,
        session_id: &str,
        cwd: &str,
        bank: &str,
        host: &serde_json::Value,
    ) -> Result<i64> {
        self.write(|conn| {
            conn.execute(
                "INSERT INTO capture (session_id, cwd, bank, host, started_at) VALUES (?1, ?2, ?3, ?4, ?5)",
                params![session_id, cwd, bank, serde_json::to_string(host)?, now()],
            )?;
            Ok(conn.last_insert_rowid())
        })
    }

    pub fn add_fact(&self, capture_id: i64, fact: NewFact) -> Result<()> {
        let (line, closed) = fact
            .undo
            .map_or((None, None), |undo| (Some(undo.line), undo.closed));
        let (closed_path, before, after) = closed.map_or((None, None, None), |closed| {
            (Some(closed.path), Some(closed.before), Some(closed.after))
        });
        self.write(|conn| {
            conn.execute(
                "INSERT INTO fact (capture_id, outcome, category, text, previous, note, path, line_text, previous_path, previous_before, previous_after)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11)",
                params![
                    capture_id,
                    outcome_text(fact.outcome),
                    fact.category,
                    fact.text,
                    fact.previous,
                    fact.note,
                    fact.path,
                    line,
                    closed_path,
                    before,
                    after
                ],
            )?;
            Ok(())
        })
    }

    pub fn finish(&self, capture_id: i64, error: Option<&str>, no_model: bool) -> Result<()> {
        self.write(|conn| {
            conn.execute(
                "UPDATE capture SET ended_at = ?2, error = ?3, no_model = ?4 WHERE id = ?1",
                params![capture_id, now(), error, no_model],
            )?;
            Ok(())
        })
    }

    pub fn get(&self, capture_id: i64) -> Result<CaptureRun> {
        let not_found = || invalid("Memory capture not found");
        self.read(
            || Err(not_found()),
            |conn| {
                let run = conn
                    .query_row(
                        &format!("SELECT {RUN_COLUMNS} FROM capture WHERE id = ?1"),
                        [capture_id],
                        run,
                    )
                    .optional()?
                    .ok_or_else(not_found)?;
                with_facts(conn, run)
            },
        )
    }

    pub fn list(&self, query: &CaptureQuery) -> Result<Vec<CaptureRun>> {
        self.read(
            || Ok(Vec::new()),
            |conn| {
                let runs = conn
                    .prepare(&format!(
                        "SELECT {RUN_COLUMNS} FROM capture
                         WHERE (?1 IS NULL OR session_id = ?1) AND (?2 IS NULL OR bank = ?2)
                         ORDER BY id DESC LIMIT ?3"
                    ))?
                    .query_map(
                        params![query.session_id, query.bank, query.limit.clamp(1, 500)],
                        run,
                    )?
                    .collect::<rusqlite::Result<Vec<_>>>()?;
                runs.into_iter().map(|run| with_facts(conn, run)).collect()
            },
        )
    }

    pub fn fact_edit(&self, fact_id: i64) -> Result<FactEdit> {
        type Columns = (
            i64,
            bool,
            String,
            Option<String>,
            Option<String>,
            Option<String>,
            Option<String>,
        );
        let not_found = || invalid("Memory fact not found");
        let (capture_id, reverted, path, line, closed_path, before, after): Columns = self.read(
            || Err(not_found()),
            |conn| {
                conn.query_row(
                    "SELECT capture_id, reverted, path, line_text, previous_path, previous_before, previous_after
                     FROM fact WHERE id = ?1",
                    [fact_id],
                    |row| {
                        Ok((
                            row.get(0)?,
                            row.get(1)?,
                            row.get(2)?,
                            row.get(3)?,
                            row.get(4)?,
                            row.get(5)?,
                            row.get(6)?,
                        ))
                    },
                )
                .optional()?
                .ok_or_else(not_found)
            },
        )?;
        let closed = match (closed_path, before, after) {
            (Some(path), Some(before), Some(after)) => Some(ClosedLine {
                path,
                before,
                after,
            }),
            (None, None, None) => None,
            _ => return Err(invalid("Memory fact undo data is incomplete")),
        };
        Ok(FactEdit {
            capture_id,
            reverted,
            path,
            undo: line.map(|line| Undo { line, closed }),
        })
    }

    pub fn mark_reverted(&self, fact_id: i64) -> Result<()> {
        self.write(|conn| {
            conn.execute("UPDATE fact SET reverted = 1 WHERE id = ?1", [fact_id])?;
            Ok(())
        })
    }

    pub fn save_cursor(&self, session_id: &str, seq: u64) -> Result<()> {
        self.write(|conn| {
            conn.execute(
                "INSERT INTO cursor (session_id, seq, updated_at) VALUES (?1, ?2, ?3)
                 ON CONFLICT(session_id) DO UPDATE SET seq = excluded.seq, updated_at = excluded.updated_at",
                params![session_id, seq, now()],
            )?;
            Ok(())
        })
    }

    pub fn cursor(&self, session_id: &str) -> Result<Option<u64>> {
        self.read(
            || Ok(None),
            |conn| {
                Ok(conn
                    .query_row(
                        "SELECT seq FROM cursor WHERE session_id = ?1",
                        [session_id],
                        |row| row.get::<_, u64>(0),
                    )
                    .optional()?)
            },
        )
    }
}
