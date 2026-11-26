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

/// A consumed context and the nodes explored from it. Implements `search::Rows`.
pub(super) struct Branch {
    lm: Lm,
    root: Arc<[f32]>,
    /// Cached `(lm, row)` after token paths from `lm`, so each child expansion
    /// is a single-token forward off its parent.
    nodes: HashMap<Vec<u32>, (Lm, Arc<[f32]>)>,
}

impl Branch {
    pub fn new(lm: Lm, root: Arc<[f32]>) -> Self {
        Self {
            lm,
            root,
            nodes: HashMap::new(),
        }
    }

    /// A new branch after `tokens` more; `self` is unchanged.
    pub fn extend(&self, tokens: &[u32]) -> anyhow::Result<Self> {
        if tokens.is_empty() {
            return Ok(Self {
                lm: self.lm.clone(),
                root: Arc::clone(&self.root),
                nodes: self.nodes.clone(),
            });
        }
        if let Some((lm, root)) = self.nodes.get(tokens) {
            return Ok(Self::new(lm.clone(), Arc::clone(root)));
        }
        let mut split = tokens.len() - 1;
        while split > 0 && !self.nodes.contains_key(&tokens[..split]) {
            split -= 1;
        }
        let mut lm = if split == 0 {
            self.lm.clone()
        } else {
            self.nodes[&tokens[..split]].0.clone()
        };
        let root = lm.feed(&tokens[split..])?;
        Ok(Self::new(lm, root))
    }
}

impl search::Rows for Branch {
    fn row(&mut self, path: &[u32]) -> anyhow::Result<Arc<[f32]>> {
        if path.is_empty() {
            return Ok(Arc::clone(&self.root));
        }
        if let Some((_, row)) = self.nodes.get(path) {
            return Ok(Arc::clone(row));
        }
        let mut split = path.len() - 1;
        while split > 0 && !self.nodes.contains_key(&path[..split]) {
            split -= 1;
        }
        let mut lm = if split == 0 {
            self.lm.clone()
        } else {
            self.nodes[&path[..split]].0.clone()
        };
        let row = lm.feed(&path[split..])?;
        self.nodes.insert(path.to_vec(), (lm, Arc::clone(&row)));
        Ok(row)
    }
}
