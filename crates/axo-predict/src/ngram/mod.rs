//! Personal word n-gram engine: the cheap, instantly available engine.
//!
//! A port of the text-prediction research winner (tracks `ngram` ×
//! `adaptive`): an interpolated absolute-discounting word
//! trigram model over the user's prose history, backed by an embedded Norvig
//! web unigram/bigram prior (`web.rs`), ranked under the typed prefix with a
//! finished-word-aware posterior, a prompt-local cache, a session cache, and
//! vocabulary hygiene.
//!
//! # Model
//! - **Counts** (`model.rs`): prose words only (the editor's gates,
//!   [`crate::prose`]), with their previous one or two prose words as context.
//!   Prompts that look typed count 1, pastes and logs count
//!   [`Params::paste_weight`].
//! - **Ranking** (`query.rs`): word ids are lexicographic ranks, so a prefix is
//!   one id range; a Fenwick tree gives its unigram mass, a max segment tree
//!   its unigram top-k, id-sorted follower lists and the web CSR its bigram
//!   evidence, and an open-addressing table the trigram counts.
//! - **Confidence**: posterior of the best word among *every* word under the
//!   prefix, including the prefix itself (`the|re` shows only when *there*
//!   clearly beats stopping at *the*), mixed with the prompt-local and session
//!   caches. Typing more letters of the shown word keeps it the pick: ghosts do
//!   not change under the user's fingers.
//! - **Hygiene**: words outside the web vocabulary and the dictionary need
//!   [`Params::min_count`] typed occurrences; rare misspellings fold into an
//!   edit-distance-1 real word at learn time (not across apostrophes, not at
//!   the first or last letter); habitual slips of a word the user also types
//!   are never offered; pastes never make a word suggestible.
//!
//! # Operating point
//! Prefixes may be a single letter (`figure o|ut`); the rest of the word is
//! ranked with the same model, where left context does most of the work.
//! One show threshold τ = 0.15 ([`Params::show_threshold`]) gates every
//! prefix length. The point favours recall: a wrong ghost costs a glance, a
//! missing right one costs the keystrokes.
//!
//! Replay of the last 400 single-line typed history.db prompts (trained on
//! the ~58k before them, learning online; a wrong ghost counts once per word
//! however long it stays up; Tab accepts with the trailing space). KSR %
//! (net keystrokes saved) for a typist who Tabs every right ghost / notices
//! one half the time, and wrong ghosts per 100 words:
//!
//! | τ    | KSR           | wrong / 100 w |
//! |------|---------------|---------------|
//! | 0    | 33.10 / 22.65 | 103.4         |
//! | 0.15 | 32.36 / 22.12 | 88.3          |
//! | 0.30 | 30.31 / 20.59 | 54.5          |
//! | 0.45 | 27.04 / 18.29 | 30.0          |
//!
//! Engine-side costs (omp research replay bench, M4 Max, 52,950
//! bootstrap rows): `complete` p50 3.6–4.0 µs, p99 14–17
//! µs with a two-letter gate; with single letters p50 4.7–5.1 µs, p99 23–39
//! µs overall and p50 5.8–7.6 µs, p99 28–59 µs at one letter;
//! `observe` p50 0.03 ms, p99 0.2–0.3 ms (a vocabulary compaction every
//! 4,096 new words takes ~20 ms); bootstrap 2.3 s; heap 52 MiB after bootstrap
//! (62 MiB peak), 38 MiB restored; snapshot 3.8 MiB, written in ~80 ms,
//! restored in ~65 ms. The embedded web prior is 2.6 MB compressed, ~9 MiB
//! decoded.
//!
//! The calibrated per-(prefix, word) feedback bias from the adaptive track
//! was measured and not adopted: +0.03 (online) and +0.06 (cold) balanced
//! KSR, within noise.
//!
//! # Persistence
//! [`NgramPredictor::persist`] writes a versioned zstd snapshot
//! (`snapshot.rs`) to `ngram.snapshot` in the state directory atomically
//! (temp file + rename); [`NgramPredictor::open`] restores it and rejects an
//! unknown version or a corrupt file so the caller can wipe the directory.

mod hygiene;
mod model;
mod query;
mod snapshot;
mod structures;
#[cfg(test)]
mod tests;
mod text;
mod vocab;
mod web;

use std::path::{Path, PathBuf};

use anyhow::Context;
use model::Params;

/// Snapshot file inside the state directory.
pub(crate) const SNAPSHOT_FILE: &str = "ngram.snapshot";

/// The n-gram engine.
pub(crate) struct NgramPredictor {
    model: model::Model,
    query: query::QueryState,
    state_dir: PathBuf,
    /// Something was learned since the last snapshot write.
    dirty: bool,
}

impl NgramPredictor {
    /// Open with the tuned [`Params`], restoring persisted state from
    /// `state_dir`.
    ///
    /// # Errors
    /// Returns an error when the snapshot exists but cannot be read, has an
    /// unknown version, or is corrupt.
    pub fn open(state_dir: &Path) -> anyhow::Result<Self> {
        let params = Params::default();
        let web = web::web_prior()?;
        let path = state_dir.join(SNAPSHOT_FILE);
        let model = match std::fs::read(&path) {
            Ok(bytes) => snapshot::decode(&bytes, params, web)
                .with_context(|| format!("restore {}", path.display()))?,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                model::Model::new(params, web)
            }
            Err(error) => return Err(error).with_context(|| format!("read {}", path.display())),
        };
        Ok(Self {
            model,
            query: query::QueryState::default(),
            state_dir: state_dir.to_path_buf(),
            dirty: false,
        })
    }

    /// Best completion of `prefix` after `before`: the suffix to show and its
    /// confidence, already above [`Params::show_threshold`].
    pub fn complete(&mut self, before: &str, prefix: &str) -> Option<(String, f32)> {
        let (suffix, confidence) = self.query.complete(&self.model, before, prefix)?;
        let confidence = confidence as f32;
        (confidence >= self.model.params.show_threshold).then_some((suffix, confidence))
    }

    /// Learn from one submitted prompt. The model owns its filtering of code,
    /// pastes, and misspellings.
    pub fn observe(&mut self, prompt: &str) {
        self.dirty |= self.model.learn(prompt);
    }

    /// Write the snapshot to the state directory if anything was learned
    /// since the last write.
    ///
    /// # Errors
    /// Returns an error when the state directory cannot be written.
    pub fn persist(&mut self) -> anyhow::Result<()> {
        if !self.dirty {
            return Ok(());
        }
        self.model.prune_trigrams();
        let bytes = snapshot::encode(&self.model)?;
        crate::write_atomic(&self.state_dir, SNAPSHOT_FILE, &bytes)?;
        self.dirty = false;
        Ok(())
    }
}
