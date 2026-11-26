//! The language model as cloneable continuations of consumed text.
//!
//! candle's quantized Llama keeps its KV cache inside the weights struct, and
//! cloning it shares every tensor, so a clone is an independent continuation
//! that costs microseconds. A [`Branch`] is one consumed context plus the
//! search rows explored from it.

use std::{collections::HashMap, path::Path, sync::Arc};

use anyhow::Context as _;
use candle_core::{Device, Tensor, quantized::gguf_file};
use candle_transformers::models::quantized_llama::ModelWeights;

use super::search;

/// The model plus the tokens it has consumed. Cloning is cheap (shared tensors) and
/// gives an independent continuation.
#[derive(Clone)]
pub(super) struct Lm {
    weights: ModelWeights,
    len: usize,
}

impl Lm {
    /// Reads the GGUF onto the CPU.
    pub fn load(gguf: &Path) -> anyhow::Result<Self> {
        let mut file =
            std::fs::File::open(gguf).with_context(|| format!("open {}", gguf.display()))?;
        let content = gguf_file::Content::read(&mut file)
            .with_context(|| format!("read {}", gguf.display()))?;
        let weights = ModelWeights::from_gguf(content, &mut file, &Device::Cpu)?;
        Ok(Self { weights, len: 0 })
    }

    /// Consumes `tokens` (non-empty); returns next-token log-probabilities after the last one.
    pub fn feed(&mut self, tokens: &[u32]) -> anyhow::Result<Arc<[f32]>> {
        let x = Tensor::new(tokens, &Device::Cpu)?.unsqueeze(0)?;
        let logits: Vec<f32> = self
            .weights
            .forward(&x, self.len)?
            .flatten_all()?
            .to_vec1()?;
        self.len += tokens.len();
        Ok(log_softmax(logits))
    }
}

fn log_softmax(mut logits: Vec<f32>) -> Arc<[f32]> {
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let log_sum = logits.iter().map(|x| (x - max).exp()).sum::<f32>().ln() + max;
    for x in &mut logits {
        *x -= log_sum;
    }
    logits.into()
}

/// A consumed context and the rows explored from it. Implements `search::Rows`.
pub(super) struct Branch {
    lm: Lm,
    root: Arc<[f32]>,
    /// Rows after token paths from `lm`, for the search.
    rows: HashMap<Vec<u32>, Arc<[f32]>>,
}

impl Branch {
    pub fn new(lm: Lm, root: Arc<[f32]>) -> Self {
        Self {
            lm,
            root,
            rows: HashMap::new(),
        }
    }

    /// A new branch after `tokens` more; `self` is unchanged.
    pub fn extend(&self, tokens: &[u32]) -> anyhow::Result<Self> {
        if tokens.is_empty() {
            return Ok(Self {
                lm: self.lm.clone(),
                root: Arc::clone(&self.root),
                rows: self.rows.clone(),
            });
        }
        let mut lm = self.lm.clone();
        let root = lm.feed(tokens)?;
        Ok(Self::new(lm, root))
    }
}

impl search::Rows for Branch {
    fn row(&mut self, path: &[u32]) -> anyhow::Result<Arc<[f32]>> {
        if path.is_empty() {
            return Ok(Arc::clone(&self.root));
        }
        if let Some(row) = self.rows.get(path) {
            return Ok(Arc::clone(row));
        }
        // ponytail: each uncached path re-feeds from the context clone; keep per-node clones if the bench shows it matters
        let row = self.lm.clone().feed(path)?;
        self.rows.insert(path.to_vec(), Arc::clone(&row));
        Ok(row)
    }
}
