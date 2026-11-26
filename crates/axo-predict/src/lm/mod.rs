//! SmolLM2-135M through candle: a pretrained base LM used zero-shot with
//! token-healed, prefix-constrained decoding, adapted from omp's `smollm`
//! engine.
//!
//! - Head: [`PREAMBLE`] and the user's recent typed prompts ([`Memory`],
//!   1,500–2,500 UTF-16 units), consumed once per submitted prompt in
//!   [`LanguageModel::observe`], never on the typing path.
//! - Context: the last ~400 units of the draft ([`context_window`]) as a
//!   [`Branch`] off the head, extended while its token ids only grow.
//! - Word: `before + prefix` is token-healed ([`heal`]) and
//!   [`search::complete_nbest`] spells the prefix to a word boundary; a word
//!   offered at [`TYPED_PAST`] or more is not offered again while that word is
//!   typed.
//! - Next words: the same search with an empty prefix after a space, chained
//!   by extending the branch with each picked word.

mod branch;
mod search;

use std::{
    collections::{HashSet, VecDeque},
    path::{Path, PathBuf},
    sync::Arc,
};

use anyhow::Context as _;
use axo_models::{ModelFile, ModelSpec};
use serde::{Deserialize, Serialize};
use tokenizers::Tokenizer;

use self::{
    branch::{Branch, Lm},
    search::{SearchParams, TokenIndex},
};
use crate::prose::{is_js_space, is_typed_like, utf16_len};

/// SmolLM2-135M (Apache-2.0), 8-bit GGUF. 147 MB in total.
pub const SMOLLM2: ModelSpec = ModelSpec {
    dir: "smollm2-135m",
    files: &[
        ModelFile {
            name: TOKENIZER_FILE,
            size: 2_104_556,
            url: "https://huggingface.co/HuggingFaceTB/SmolLM2-135M/resolve/93efa2f097d58c2a74874c7e644dbc9b0cee75a2/tokenizer.json",
            sha256: "9ca9acddb6525a194ec8ac7a87f24fbba7232a9a15ffa1af0c1224fcd888e47c",
        },
        ModelFile {
            name: WEIGHTS_FILE,
            size: 144_810_464,
            url: "https://huggingface.co/QuantFactory/SmolLM2-135M-GGUF/resolve/d948db3614be18259a175aafd7689a70f1cb4e2f/SmolLM2-135M.Q8_0.gguf",
            sha256: "b761d9ccdfce67726e41ca2ef30e9dfbcf6a32ca2aaef47df5c049ec362d04cd",
        },
    ],
};

/// Files needed to load a Llama-architecture GGUF language model.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ModelFiles {
    /// Path to the quantized GGUF weights file.
    pub weights: PathBuf,
    /// Path to the Hugging Face `tokenizer.json`.
    pub tokenizer: PathBuf,
    /// Beginning-of-sequence token id.
    pub bos: u32,
}

impl ModelFiles {
    /// Standard file layout for [`SMOLLM2`] inside `model_dir`.
    pub fn smollm2(model_dir: &Path) -> Self {
        Self {
            weights: model_dir.join(WEIGHTS_FILE),
            tokenizer: model_dir.join(TOKENIZER_FILE),
            bos: SMOLLM2_BOS,
        }
    }
}

const TOKENIZER_FILE: &str = "tokenizer.json";
const WEIGHTS_FILE: &str = "SmolLM2-135M.Q8_0.gguf";
/// SmolLM2's beginning-of-sequence token.
const SMOLLM2_BOS: u32 = 0;
/// Head text before the memory: frames the context as a typed chat message.
const PREAMBLE: &str = include_str!("preamble.txt");
/// Context taken from `before`, in UTF-16 units.
const CONTEXT_UNITS: usize = 400;
/// How much further back [`context_window`] may start to reach an anchor.
const CONTEXT_SLACK: usize = 256;
/// One word in this many anchors context windows.
const ANCHOR_EVERY: u32 = 16;
/// Default recent-prompt memory in the head after a trim, in UTF-16 units.
pub(crate) const DEFAULT_MEMORY_UNITS: usize = 750;
/// Default memory size that triggers a trim back to [`DEFAULT_MEMORY_UNITS`].
pub(crate) const DEFAULT_MEMORY_MAX_UNITS: usize = 1250;
/// Typed-past exclusion threshold τ: a suggestion at or above it counts as
/// shown.
const TYPED_PAST: f32 = 0.15;
/// Extra search expansions allowed per already-shown word so typed-past
/// exclusion still has budget to find a fallback.
const SKIP_BUDGET_PER_SHOWN: usize = 4;
/// Default chained confidence each following word must keep to be shown.
pub(crate) const DEFAULT_NEXT_WORD_THRESHOLD: f32 = 0.25;
/// Persisted recent-prompt memory in the state directory.
pub(crate) const STATE_FILE: &str = "memory.json";
const STATE_VERSION: u32 = 1;

/// Last `units` UTF-16 units of `before`, starting after a whitespace so no
/// word is cut in half.
fn context_tail(before: &str, units: usize) -> &str {
    let mut count = 0;
    let mut start = before.len();
    for (at, c) in before.char_indices().rev() {
        count += c.len_utf16();
        if count > units {
            break;
        }
        start = at;
    }
    if start == 0 {
        return before;
    }
    let tail = &before[start..];
    match tail.char_indices().find(|&(_, c)| is_js_space(c)) {
        Some((at, c)) => &tail[at + c.len_utf8()..],
        None => tail,
    }
}

/// Whether the word starting `text` anchors context windows: 1 in
/// [`ANCHOR_EVERY`] words, by a stable hash of the word itself.
fn is_anchor(text: &str) -> bool {
    let word = text.split(is_js_space).next().unwrap_or_default();
    // FNV-1a: stable across runs and platforms.
    let hash = word.bytes().fold(0x811c_9dc5u32, |h, b| {
        (h ^ u32::from(b)).wrapping_mul(0x0100_0193)
    });
    hash.is_multiple_of(ANCHOR_EVERY)
}

/// The context `before` a word: the last [`CONTEXT_UNITS`] UTF-16 units,
/// extended back (by at most [`CONTEXT_SLACK`]) to the nearest word start
/// whose word [`is_anchor`]. The start then only moves when a new anchor
/// scrolls past the 400-unit mark (every ~16 words), so consecutive words
/// share their context prefix and its cached branch. Without an anchor in
/// range it falls back to [`context_tail`]. A pure function of `before`.
fn context_window(before: &str) -> &str {
    let total = utf16_len(before);
    if total <= CONTEXT_UNITS {
        return before;
    }
    let (lo, hi) = (
        total.saturating_sub(CONTEXT_UNITS + CONTEXT_SLACK),
        total - CONTEXT_UNITS,
    );
    let mut units = 0;
    let mut after_space = false;
    let mut anchor = None;
    for (at, c) in before.char_indices() {
        if units > hi {
            break;
        }
        let space = is_js_space(c);
        if units >= lo && after_space && !space && is_anchor(&before[at..]) {
            anchor = Some(at);
        }
        after_space = space;
        units += c.len_utf16();
    }
    anchor.map_or_else(|| context_tail(before, CONTEXT_UNITS), |at| &before[at..])
}

fn encode(tokenizer: &Tokenizer, text: &str) -> anyhow::Result<tokenizers::Encoding> {
    tokenizer.encode(text, false).map_err(anyhow::Error::msg)
}

/// Token-heal `context + prefix`: the context's tokens up to the one holding
/// the word start, and the text of that token before the word (the lead the
/// search must re-spell, usually a space).
fn heal(tokenizer: &Tokenizer, context: &str, prefix: &str) -> anyhow::Result<(Vec<u32>, String)> {
    let text = format!("{context}{prefix}");
    let encoding = encode(tokenizer, &text)?;
    let start = context.len();
    let offsets = encoding.get_offsets();
    let cut_token = offsets
        .iter()
        .position(|&(_, end)| end > start)
        .unwrap_or(offsets.len());
    let cut = offsets.get(cut_token).map_or(start, |&(from, _)| from);
    if !text.is_char_boundary(cut) {
        // The word's first token also holds part of a character before it.
        return Ok((
            encode(tokenizer, context)?.get_ids().to_vec(),
            String::new(),
        ));
    }
    Ok((
        encoding.get_ids()[..cut_token].to_vec(),
        text[cut..start].to_owned(),
    ))
}

/// Recent typed-like prompts in the head, oldest first. The window only
/// grows by appending, so a new prompt extends the head instead of
/// re-scoring it, until it passes [`MEMORY_MAX_UNITS`]; then the oldest
/// prompts drop until it fits [`MEMORY_UNITS`] again.
struct Memory {
    recent: VecDeque<String>,
    /// UTF-16 units of the remembered prompts plus separators.
    units: usize,
    /// Target UTF-16 units after a trim.
    target_units: usize,
    /// Ceiling that triggers a trim back to `target_units`.
    max_units: usize,
    /// The head no longer matches `recent`.
    dirty: bool,
}

impl Memory {
    fn new(target_units: usize, max_units: usize) -> Self {
        let max_units = max_units.max(target_units);
        Self {
            recent: VecDeque::new(),
            units: 0,
            target_units,
            max_units,
            dirty: false,
        }
    }

    fn set_limits(&mut self, target_units: usize, max_units: usize) {
        self.target_units = target_units;
        self.max_units = max_units.max(target_units);
        self.trim();
    }

    fn remember(&mut self, prompt: &str) {
        if !is_typed_like(prompt) {
            return;
        }
        self.units += utf16_len(prompt) + 2;
        self.recent.push_back(prompt.to_owned());
        self.trim();
        self.dirty = true;
    }

    fn trim(&mut self) {
        if self.units > self.max_units {
            let before = self.recent.len();
            while self.units > self.target_units && self.recent.len() > 1 {
                if let Some(old) = self.recent.pop_front() {
                    self.units -= utf16_len(&old) + 2;
                }
            }
            if self.recent.len() != before {
                self.dirty = true;
            }
        }
    }
}

#[derive(Serialize, Deserialize)]
struct State {
    version: u32,
    /// Recent typed prompts, oldest first.
    recent: Vec<String>,
}

/// Words offered (confidence ≥ τ) while the current word is typed.
#[derive(Default)]
struct Shown {
    before: String,
    prefix_units: usize,
    words: HashSet<String>,
}

impl Shown {
    /// A new word starts when `before` changes or the prefix stops growing.
    fn track(&mut self, before: &str, prefix: &str) {
        let units = utf16_len(prefix);
        if before != self.before || units <= self.prefix_units {
            self.words.clear();
            if before != self.before {
                before.clone_into(&mut self.before);
            }
        }
        self.prefix_units = units;
    }

    fn key(prefix: &str, suffix: &str) -> String {
        format!("{prefix}{suffix}").to_lowercase()
    }
}

/// The branch for the word being completed.
struct Context {
    /// [`context_window`] text it was built from.
    window: String,
    /// Healed token ids after the head.
    ids: Vec<u32>,
    /// Text the search re-spells before the prefix.
    lead: String,
    branch: Branch,
}

/// Deferred head-prefill work produced by [`LanguageModel::prepare_head`] so a
/// host can run the slow model forward pass without holding the predictor lock.
pub(crate) struct HeadUpdate {
    head: Lm,
    ids: Vec<u32>,
    from: usize,
}

/// Completed head state ready to be swapped into [`LanguageModel::apply_head`].
pub(crate) struct ComputedHead {
    head: Lm,
    ids: Vec<u32>,
    row: Arc<[f32]>,
}

impl HeadUpdate {
    pub(crate) fn compute(mut self) -> anyhow::Result<ComputedHead> {
        let row = self.head.feed(&self.ids[self.from..])?;
        Ok(ComputedHead {
            head: self.head,
            ids: self.ids,
            row,
        })
    }
}

pub(crate) struct LanguageModel {
    tokenizer: Tokenizer,
    index: TokenIndex,
    bos: u32,
    /// Nothing consumed.
    fresh: Lm,
    head: Lm,
    head_ids: Vec<u32>,
    head_row: Arc<[f32]>,
    context: Option<Context>,
    memory: Memory,
    /// `memory` changed since the last write.
    unsaved: bool,
    shown: Shown,
    state_dir: PathBuf,
}

impl LanguageModel {
    /// Loads the model files and the prompt memory from `state_dir`, then
    /// consumes the head.
    pub fn open(
        files: &ModelFiles,
        state_dir: &Path,
        memory_units: usize,
        memory_max_units: usize,
    ) -> anyhow::Result<Self> {
        let tokenizer = Tokenizer::from_file(&files.tokenizer)
            .map_err(anyhow::Error::msg)
            .with_context(|| format!("load tokenizer {}", files.tokenizer.display()))?;
        let fresh = Lm::load(&files.weights)?;
        let specials = tokenizer.get_added_tokens_decoder();
        let texts = (0..tokenizer.get_vocab_size(true) as u32)
            .map(|id| {
                let text = tokenizer.decode(&[id], false).ok()?;
                // A lone byte of a multi-byte character decodes to U+FFFD.
                (!specials.contains_key(&id) && !text.is_empty() && !text.contains('\u{FFFD}'))
                    .then_some(text)
            })
            .collect();
        let mut memory = Memory::new(memory_units, memory_max_units);
        let path = state_dir.join(STATE_FILE);
        match std::fs::read(&path) {
            Ok(bytes) => {
                let state: State = serde_json::from_slice(&bytes)
                    .with_context(|| format!("parse {}", path.display()))?;
                anyhow::ensure!(
                    state.version == STATE_VERSION,
                    "unknown {STATE_FILE} version {}",
                    state.version
                );
                for prompt in state.recent {
                    memory.remember(&prompt);
                }
            }
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
            Err(error) => return Err(error).with_context(|| format!("read {}", path.display())),
        }
        let mut lm = Self {
            index: TokenIndex::new(texts),
            bos: files.bos,
            head: fresh.clone(),
            fresh,
            head_ids: Vec::new(),
            head_row: Arc::from([]),
            tokenizer,
            context: None,
            memory,
            unsaved: false,
            shown: Shown::default(),
            state_dir: state_dir.to_owned(),
        };
        lm.update_head()?;
        Ok(lm)
    }

    pub fn set_memory_limits(&mut self, target_units: usize, max_units: usize) {
        self.memory.set_limits(target_units, max_units);
    }

    /// Encodes `[BOS] + PREAMBLE + memory` and clones the base `Lm` needed to
    /// advance the head, without running the model forward pass yet.
    ///
    /// The preamble and each prompt are encoded on their own: a trailing
    /// `"\n\n"` is one token at the end of the text but two before a letter,
    /// so encoding the whole text would change the old tail on every append
    /// and re-feed the head from scratch.
    pub fn prepare_head(&mut self) -> anyhow::Result<Option<HeadUpdate>> {
        let mut ids = vec![self.bos];
        ids.extend_from_slice(encode(&self.tokenizer, PREAMBLE)?.get_ids());
        for prompt in &self.memory.recent {
            ids.extend_from_slice(encode(&self.tokenizer, &format!("{prompt}\n\n"))?.get_ids());
        }
        let (head, from) = if ids.starts_with(&self.head_ids) {
            (self.head.clone(), self.head_ids.len())
        } else {
            (self.fresh.clone(), 0)
        };
        self.memory.dirty = false;
        if from >= ids.len() {
            return Ok(None);
        }
        Ok(Some(HeadUpdate { head, ids, from }))
    }

    /// Installs a head computed off-lock. Ignored if memory changed again while
    /// it was computing.
    pub fn apply_head(&mut self, computed: ComputedHead) {
        if self.memory.dirty {
            return;
        }
        self.head = computed.head;
        self.head_ids = computed.ids;
        self.head_row = computed.row;
        self.context = None;
    }

    fn update_head(&mut self) -> anyhow::Result<()> {
        if let Some(update) = self.prepare_head()? {
            let computed = update.compute()?;
            self.apply_head(computed);
        }
        Ok(())
    }

    /// Remembers a submitted prompt without running the model forward pass.
    /// Call [`Self::prepare_head`] afterwards to build deferred head work.
    pub fn remember(&mut self, text: &str) {
        self.memory.remember(text);
        if self.memory.dirty {
            self.unsaved = true;
        }
    }

    /// Remembers a submitted prompt and extends the head with it.
    pub fn observe(&mut self, text: &str) {
        self.memory.remember(text);
        if self.memory.dirty {
            self.unsaved = true;
            if let Err(error) = self.update_head() {
                tracing::warn!("axo-predict: language model head not updated: {error:#}");
            }
        }
    }

    /// The branch after the head plus `ids`: extends the word context's
    /// branch when its ids are a prefix of `ids`.
    fn branch_after(&self, ids: &[u32]) -> anyhow::Result<Branch> {
        match &self.context {
            Some(context) if ids.starts_with(&context.ids) => {
                context.branch.extend(&ids[context.ids.len()..])
            }
            _ => Branch::new(self.head.clone(), Arc::clone(&self.head_row)).extend(ids),
        }
    }

    /// Best completion of `prefix` after `before`: (suffix, confidence), unthresholded.
    pub fn word(&mut self, before: &str, prefix: &str) -> anyhow::Result<Option<(String, f32)>> {
        self.shown.track(before, prefix);
        let window = context_window(before);
        if self
            .context
            .as_ref()
            .is_none_or(|context| context.window != window)
        {
            let (ids, lead) = heal(&self.tokenizer, window, prefix)?;
            self.context = Some(Context {
                window: window.to_owned(),
                branch: self.branch_after(&ids)?,
                ids,
                lead,
            });
        }
        let Some(context) = &mut self.context else {
            return Ok(None);
        };
        let target = format!("{}{prefix}", context.lead);
        let mut params = SearchParams::default();
        params.max_expand += SKIP_BUDGET_PER_SHOWN * self.shown.words.len().min(2);
        let shown = &self.shown.words;
        let alternatives = search::complete_nbest_filtered(
            &self.index,
            params,
            &target,
            1,
            |suffix| shown.contains(&Shown::key(prefix, suffix)),
            &mut context.branch,
        );
        let alternatives = match alternatives {
            Ok(alternatives) => alternatives,
            Err(error) => {
                self.context = None;
                return Err(error);
            }
        };
        let Some((suffix, confidence)) = alternatives.into_iter().next() else {
            return Ok(None);
        };
        if confidence >= TYPED_PAST {
            self.shown.words.insert(Shown::key(prefix, &suffix));
        }
        Ok(Some((suffix, confidence)))
    }

    /// Up to `max` following words after `before + typed`, each with its own confidence.
    pub fn next(&self, before: &str, typed: &str, max: u8) -> anyhow::Result<Vec<(String, f32)>> {
        let text = format!("{before}{typed}");
        let ids = encode(&self.tokenizer, context_window(&text))?;
        let mut branch = self.branch_after(ids.get_ids())?;
        let mut words = Vec::new();
        while words.len() < usize::from(max) {
            let best = search::complete_nbest_filtered(
                &self.index,
                SearchParams::default(),
                " ",
                1,
                |_| false,
                &mut branch,
            )?;
            let Some((word, confidence)) = best.into_iter().next().filter(|(w, _)| !w.is_empty())
            else {
                break;
            };
            if words.len() + 1 < usize::from(max) {
                branch = branch.extend(encode(&self.tokenizer, &format!(" {word}"))?.get_ids())?;
            }
            words.push((word, confidence));
        }
        Ok(words)
    }

    /// Writes the prompt memory if it changed since the last write.
    pub fn persist(&mut self) -> anyhow::Result<()> {
        if !self.unsaved {
            return Ok(());
        }
        let state = State {
            version: STATE_VERSION,
            recent: self.memory.recent.iter().cloned().collect(),
        };
        crate::write_atomic(&self.state_dir, STATE_FILE, &serde_json::to_vec(&state)?)?;
        self.unsaved = false;
        Ok(())
    }
}
