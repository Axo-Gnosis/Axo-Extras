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
use lm::{
    DEFAULT_MEMORY_MAX_UNITS, DEFAULT_MEMORY_UNITS, DEFAULT_NEXT_WORD_THRESHOLD, LanguageModel,
};
use ngram::NgramPredictor;

pub use lm::{ModelFiles, SMOLLM2};

/// Host-configurable tuning for [`Predictor`]. [`Default`] holds the tuned
/// baselines; hosts can override any field at open time or live via
/// [`Predictor::set_config`].
#[derive(Clone, Debug, PartialEq)]
pub struct Config {
    /// Words to suggest beyond the current word, clamped to `0..=2`. Only a
    /// language model suggests them.
    pub next_words: u8,
    /// Minimum confidence (`0.0..=1.0`) to show an n-gram-only word completion.
    pub ngram_threshold: f32,
    /// Minimum blended score (`0.0..=1.0`) to show a word completion when a
    /// language model is loaded.
    pub blend_threshold: f32,
    /// Share of the n-gram in the blend (`0.0..=1.0`); the language model gets
    /// the rest.
    pub ngram_weight: f32,
    /// Minimum chained confidence (`0.0..=1.0`) each following word must keep.
    pub next_word_threshold: f32,
    /// Target recent-prompt memory in the language model head after a trim, in
    /// UTF-16 units.
    pub memory_units: usize,
    /// Ceiling in UTF-16 units that triggers a trim back to `memory_units`.
    pub memory_max_units: usize,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            next_words: 1,
            ngram_threshold: 0.15,
            blend_threshold: blend::DEFAULT_SHOW_THRESHOLD,
            ngram_weight: blend::DEFAULT_NGRAM_WEIGHT,
            next_word_threshold: DEFAULT_NEXT_WORD_THRESHOLD,
            memory_units: DEFAULT_MEMORY_UNITS,
            memory_max_units: DEFAULT_MEMORY_MAX_UNITS,
        }
    }
}

/// Deferred language-model head work produced by [`Predictor::observe_deferred`].
/// Run [`HeadUpdate::compute`] off the typing lock, then install the result with
/// [`Predictor::apply_head`].
pub struct HeadUpdate(lm::HeadUpdate);

/// Completed head state returned by [`HeadUpdate::compute`].
pub struct ComputedHead(lm::ComputedHead);

impl HeadUpdate {
    /// Runs the language model over the updated prompt memory.
    ///
    /// # Errors
    /// Returns an error when the model forward pass fails.
    pub fn compute(self) -> Result<ComputedHead, Error> {
        Ok(ComputedHead(self.0.compute()?))
    }
}

/// Where the predictor keeps what it learned and its initial configuration.
pub struct Options {
    /// Private directory for the learned state; created on first persist.
    pub state_dir: PathBuf,
    /// Initial tuning parameters.
    pub config: Config,
}

impl Options {
    /// Options for `state_dir` with [`Config::default`].
    pub fn new(state_dir: PathBuf) -> Self {
        Self {
            state_dir,
            config: Config::default(),
        }
    }
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
    config: Config,
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
        let mut ngram = NgramPredictor::open(&options.state_dir)?;
        let config = normalize_config(options.config);
        ngram.set_show_threshold(config.ngram_threshold);
        Ok(Self {
            ngram,
            lm: None,
            config,
            lm_warned: false,
            state_dir: options.state_dir,
        })
    }

    /// Loads a language model from `files` (see [`ModelFiles::smollm2`]), or
    /// unloads it with `None`.
    /// Slow (reads weights, consumes the prompt memory): call off the typing path.
    ///
    /// # Errors
    /// Returns an error when the current model's memory cannot be written, or
    /// the new model or its saved memory cannot be loaded; then no model is
    /// loaded.
    pub fn set_model(&mut self, files: Option<&ModelFiles>) -> Result<(), Error> {
        if let Some(lm) = &mut self.lm {
            lm.persist()?;
        }
        self.lm = None;
        if let Some(files) = files {
            self.lm = Some(LanguageModel::open(
                files,
                &self.state_dir,
                self.config.memory_units,
                self.config.memory_max_units,
            )?);
        }
        Ok(())
    }

    pub fn has_model(&self) -> bool {
        self.lm.is_some()
    }

    /// Updates tuning parameters live without reopening the predictor or model.
    pub fn set_config(&mut self, config: Config) {
        let config = normalize_config(config);
        self.ngram.set_show_threshold(config.ngram_threshold);
        if let Some(lm) = &mut self.lm {
            lm.set_memory_limits(config.memory_units, config.memory_max_units);
        }
        self.config = config;
    }

    /// Current configuration.
    pub fn config(&self) -> &Config {
        &self.config
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
        let next_words = self.config.next_words;
        let next_threshold = self.config.next_word_threshold;
        match caret {
            Caret::Word { before, prefix } => {
                let ngram = self.ngram.complete(before, prefix);
                let model = logged(warned, lm.word(before, prefix)).flatten();
                let pick = blend::blend(
                    ngram,
                    model,
                    self.config.ngram_weight,
                    self.config.blend_threshold,
                )?;
                let mut text = pick.suffix;
                // Chained confidence only falls, so a weak pick takes no words.
                if next_words > 0 && pick.raw >= next_threshold {
                    let typed = format!("{prefix}{text}");
                    let words = logged(warned, lm.next(before, &typed, next_words));
                    chain(
                        &mut text,
                        pick.raw,
                        next_threshold,
                        words.unwrap_or_default(),
                    );
                }
                Some(Suggestion {
                    text,
                    confidence: pick.score,
                })
            }
            Caret::Next { before } => {
                if next_words == 0 {
                    return None;
                }
                let words = logged(warned, lm.next(before, "", next_words))?;
                let mut text = String::new();
                let confidence = chain(&mut text, 1.0, next_threshold, words);
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

    /// Learns `texts` into the n-gram and language-model memory immediately,
    /// returning deferred head-prefill work to run off the typing lock via
    /// [`HeadUpdate::compute`] and [`Predictor::apply_head`].
    pub fn observe_deferred<S: AsRef<str>>(&mut self, texts: &[S]) -> Option<HeadUpdate> {
        for text in texts {
            self.ngram.observe(text.as_ref());
        }
        let lm = self.lm.as_mut()?;
        for text in texts {
            lm.remember(text.as_ref());
        }
        match lm.prepare_head() {
            Ok(update) => update.map(HeadUpdate),
            Err(error) => {
                tracing::warn!("axo-predict: language model head not prepared: {error:#}");
                None
            }
        }
    }

    /// Installs a head computed off-lock by [`HeadUpdate::compute`].
    pub fn apply_head(&mut self, computed: ComputedHead) {
        if let Some(lm) = &mut self.lm {
            lm.apply_head(computed.0);
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

fn normalize_config(mut config: Config) -> Config {
    config.next_words = config.next_words.min(2);
    config.ngram_threshold = config.ngram_threshold.clamp(0.0, 1.0);
    config.blend_threshold = config.blend_threshold.clamp(0.0, 1.0);
    config.ngram_weight = config.ngram_weight.clamp(0.0, 1.0);
    config.next_word_threshold = config.next_word_threshold.clamp(0.0, 1.0);
    config.memory_max_units = config.memory_max_units.max(config.memory_units);
    config
}

/// Appends `words` to `text`, space-separated, while their chained confidence
/// stays at `threshold` or above; returns the confidence after the last word
/// appended.
fn chain(text: &mut String, mut confidence: f32, threshold: f32, words: Vec<(String, f32)>) -> f32 {
    for (word, word_confidence) in words {
        if confidence * word_confidence < threshold {
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
