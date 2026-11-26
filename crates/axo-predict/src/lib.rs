//! Local predictive text for a prompt editor.

mod blend;
mod gate;
mod lm;
mod ngram;
mod prose;

use std::{
    fmt,
    io::Write,
    path::{Path, PathBuf},
};

use anyhow::Context as _;
use gate::Caret;
use lm::{LanguageModel, NEXT_WORD_THRESHOLD};
use ngram::NgramPredictor;

pub use lm::SMOLLM2;

/// Where the predictor keeps what it learned.
pub struct Options {
    /// Private directory for the learned state; created on first persist.
    pub state_dir: PathBuf,
}

/// Text to show after the caret.
#[derive(Clone, Debug, PartialEq)]
pub struct Suggestion {
    /// The rest of the current word, and with a language model possibly following words.
    pub text: String,
    /// 0..=1; already above the predictor's show threshold.
    pub confidence: f32,
}

/// Opaque failure: unreadable state, corrupt snapshot, unloadable model.
#[derive(Debug)]
pub struct Error(anyhow::Error);

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(&self.0, f)
    }
}

impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        self.0.source()
    }
}

impl From<anyhow::Error> for Error {
    fn from(error: anyhow::Error) -> Self {
        Self(error)
    }
}

/// Word completion learned from what the user submits, and with a language
/// model loaded, next-word suggestion. `Send`; one caller at a time.
pub struct Predictor {
    ngram: NgramPredictor,
    lm: Option<LanguageModel>,
    next_words: u8,
    /// A language model failure was logged; later ones are not.
    lm_warned: bool,
    state_dir: PathBuf,
}

const _: () = {
    const fn assert_send<T: Send>() {}
    assert_send::<Predictor>();
};

impl Predictor {
    /// Restores state from `state_dir`, or starts from the built-in English prior.
    /// Fails on a corrupt or unknown-version snapshot; `reset_dir` then recovers.
    ///
    /// # Errors
    /// Returns an error when the saved state cannot be read or decoded.
    pub fn open(options: Options) -> Result<Self, Error> {
        Ok(Self {
            ngram: NgramPredictor::open(&options.state_dir)?,
            lm: None,
            next_words: 1,
            lm_warned: false,
            state_dir: options.state_dir,
        })
    }

    /// Loads the language model from `model_dir` (see `SMOLLM2`), or unloads it with `None`.
    /// Slow (reads 145 MB, consumes the prompt memory): call off the typing path.
    ///
    /// # Errors
    /// Returns an error when the current model's memory cannot be written, or
    /// the new model or its saved memory cannot be loaded; then no model is
    /// loaded.
    pub fn set_model(&mut self, model_dir: Option<&Path>) -> Result<(), Error> {
        if let Some(lm) = &mut self.lm {
            lm.persist()?;
        }
        self.lm = None;
        if let Some(dir) = model_dir {
            self.lm = Some(LanguageModel::open(dir, &self.state_dir)?);
        }
        Ok(())
    }

    pub fn has_model(&self) -> bool {
        self.lm.is_some()
    }

    /// Words to suggest beyond the current one, 0..=2 (default 1). Only a
    /// language model suggests them.
    pub fn set_next_words(&mut self, words: u8) {
        self.next_words = words.min(2);
    }

    /// The suggestion for a caret between `before` and `after`, both taken from
    /// the one run of typed text that holds the caret. `None` shows nothing.
    pub fn complete(&mut self, before: &str, after: &str) -> Option<Suggestion> {
        let caret = gate::caret(before, after)?;
        let Some(lm) = &mut self.lm else {
            return match caret {
                Caret::Word { before, prefix } => self
                    .ngram
                    .complete(before, prefix)
                    .map(|(text, confidence)| Suggestion { text, confidence }),
                Caret::Next { .. } => None,
            };
        };
        let warned = &mut self.lm_warned;
        match caret {
            Caret::Word { before, prefix } => {
                let ngram = self.ngram.complete(before, prefix);
                let model = logged(warned, lm.word(before, prefix)).flatten();
                let pick = blend::blend(ngram, model)?;
                let mut text = pick.suffix;
                // Chained confidence only falls, so a weak pick takes no words.
                if self.next_words > 0 && pick.raw >= NEXT_WORD_THRESHOLD {
                    let typed = format!("{prefix}{text}");
                    let words = logged(warned, lm.next(before, &typed, self.next_words));
                    chain(&mut text, pick.raw, words.unwrap_or_default());
                }
                Some(Suggestion {
                    text,
                    confidence: pick.score,
                })
            }
            Caret::Next { before } => {
                if self.next_words == 0 {
                    return None;
                }
                let words = logged(warned, lm.next(before, "", self.next_words))?;
                let mut text = String::new();
                let confidence = chain(&mut text, 1.0, words);
                (!text.is_empty()).then_some(Suggestion { text, confidence })
            }
        }
    }

    /// Learns one run of text the user typed and submitted. Code, pastes and
    /// one-off misspellings are filtered here. With a language model loaded
    /// this also consumes the prompt, so call it off the typing path.
    pub fn observe(&mut self, text: &str) {
        self.ngram.observe(text);
        if let Some(lm) = &mut self.lm {
            lm.observe(text);
        }
    }

    /// Writes learned state if it changed since the last write.
    ///
    /// # Errors
    /// Returns an error when the state directory cannot be written.
    pub fn persist(&mut self) -> Result<(), Error> {
        let ngram = self.ngram.persist();
        let lm = self.lm.as_mut().map_or(Ok(()), LanguageModel::persist);
        Ok(ngram.and(lm)?)
    }
}

/// The value of a language model call; its first failure is logged.
fn logged<T>(warned: &mut bool, result: anyhow::Result<T>) -> Option<T> {
    result
        .inspect_err(|error| {
            if !std::mem::replace(warned, true) {
                tracing::warn!("axo-predict: language model failed: {error:#}");
            }
        })
        .ok()
}

/// Appends `words` to `text`, space-separated, while their chained confidence
/// stays at [`NEXT_WORD_THRESHOLD`] or above; returns the confidence after
/// the last word appended.
fn chain(text: &mut String, mut confidence: f32, words: Vec<(String, f32)>) -> f32 {
    for (word, word_confidence) in words {
        if confidence * word_confidence < NEXT_WORD_THRESHOLD {
            break;
        }
        confidence *= word_confidence;
        if !text.is_empty() {
            text.push(' ');
        }
        text.push_str(&word);
    }
    confidence
}

/// Deletes everything learned under `state_dir`. For "forget" and for recovery
/// after `open` fails.
///
/// # Errors
/// Returns an error when a state file exists but cannot be removed.
pub fn reset_dir(state_dir: &Path) -> Result<(), Error> {
    for name in [ngram::SNAPSHOT_FILE, lm::STATE_FILE] {
        let path = state_dir.join(name);
        match std::fs::remove_file(&path) {
            Err(error) if error.kind() != std::io::ErrorKind::NotFound => {
                return Err(anyhow::Error::new(error)
                    .context(format!("remove {}", path.display()))
                    .into());
            }
            _ => {}
        }
    }
    Ok(())
}

/// Writes `bytes` to `dir/name` atomically: a temp file in the same directory,
/// then a rename over the target.
fn write_atomic(dir: &Path, name: &str, bytes: &[u8]) -> anyhow::Result<()> {
    std::fs::create_dir_all(dir).with_context(|| format!("create {}", dir.display()))?;
    let target = dir.join(name);
    let temp = dir.join(format!("{name}.{}.tmp", std::process::id()));
    let written = (|| {
        let mut file = std::fs::File::create(&temp)?;
        file.write_all(bytes)?;
        file.sync_all()?;
        drop(file);
        std::fs::rename(&temp, &target)
    })();
    if let Err(error) = written {
        let _ = std::fs::remove_file(&temp);
        return Err(error).with_context(|| format!("write {}", target.display()));
    }
    Ok(())
}
