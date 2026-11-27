//! Long-term memory for axo hosts: Markdown notes with typed facts and links in
//! banks of plain files, a derived SQLite index for hybrid keyword and vector
//! search, the `memory://` scheme and `memory_search` tool, and capture of durable
//! facts from finished sessions.

pub mod edit;
pub mod format;
pub mod index;
pub mod links;
pub mod permalink;
pub mod search;
pub mod sync;
pub mod temporal;

mod bank;
#[cfg(feature = "bge")]
mod bge;
mod capture;
mod embed;
mod memory;
mod scheme;
mod tool;

pub use bank::{Bank, Banks, valid_bank_name};
#[cfg(feature = "bge")]
pub use bge::{BGE_SMALL, BgeSmall};
pub use capture::{
    CaptureQuery, CaptureRequest, CaptureRun, CapturedFact, FactOutcome, writer_agent,
};
pub use embed::Embedder;
pub use index::Index;
pub use memory::{Hit, Memory, NoteLink, SearchRequest};
pub use scheme::{Guard, MemoryScheme};
pub use tool::SearchTool;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error(transparent)]
    Sql(#[from] rusqlite::Error),
    #[error(transparent)]
    Io(#[from] std::io::Error),
    #[error(transparent)]
    Json(#[from] serde_json::Error),
    #[error(transparent)]
    Axo(#[from] axo::Error),
    #[error(transparent)]
    Task(#[from] tokio::task::JoinError),
    /// An embedder failed.
    #[error("{0}")]
    Embed(String),
    #[error("{0}")]
    Invalid(String),
}

pub type Result<T, E = Error> = std::result::Result<T, E>;

pub(crate) fn invalid(message: impl Into<String>) -> Error {
    Error::Invalid(message.into())
}

/// Locks `mutex`, ignoring poison: every guarded value stays valid across a panic.
pub(crate) fn lock<T>(mutex: &std::sync::Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    mutex
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

#[cfg(test)]
mod tests;
