//! The bundled embedder: BAAI/bge-small-en-v1.5 on candle, CPU.

use std::path::Path;

use axo_models::{ModelFile, ModelSpec};
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::bert::{BertModel, Config};
use tokenizers::{PaddingParams, PaddingStrategy, Tokenizer, TruncationParams};

use crate::{Embedder, Error, Result};

const ID: &str = "BAAI/bge-small-en-v1.5@5c38ec7c405ec4b44b94cc5a9bb96e735b38267a";
const DIM: usize = 384;
const QUERY_PREFIX: &str = "Represent this sentence for searching relevant passages: ";

/// The files [`BgeSmall::load`] reads; fetch them with [`ModelSpec::download`].
pub const BGE_SMALL: ModelSpec = ModelSpec {
    dir: "bge-small-en-v1.5",
    files: &[
        ModelFile {
            name: "config.json",
            url: "https://huggingface.co/BAAI/bge-small-en-v1.5/resolve/5c38ec7c405ec4b44b94cc5a9bb96e735b38267a/config.json",
            size: 743,
            sha256: "094f8e891b932f2000c92cfc663bac4c62069f5d8af5b5278c4306aef3084750",
        },
        ModelFile {
            name: "tokenizer.json",
            url: "https://huggingface.co/BAAI/bge-small-en-v1.5/resolve/5c38ec7c405ec4b44b94cc5a9bb96e735b38267a/tokenizer.json",
            size: 711_396,
            sha256: "d241a60d5e8f04cc1b2b3e9ef7a4921b27bf526d9f6050ab90f9267a1f9e5c66",
        },
        ModelFile {
            name: "model.safetensors",
            url: "https://huggingface.co/BAAI/bge-small-en-v1.5/resolve/5c38ec7c405ec4b44b94cc5a9bb96e735b38267a/model.safetensors",
            size: 133_466_304,
            sha256: "3c9f31665447c8911517620762200d2245a2518d6e7208acc78cd9db317e21ad",
        },
    ],
};

pub struct BgeSmall {
    model: BertModel,
    tokenizer: Tokenizer,
}

fn embed_error(error: impl std::fmt::Display) -> Error {
    Error::Embed(error.to_string())
}

impl BgeSmall {
    /// Loads the files of [`BGE_SMALL`] from `model_dir` (`BGE_SMALL.path(models_root)`).
    /// Slow (reads 133 MB): call off the async runtime.
    pub fn load(model_dir: &Path) -> Result<Self> {
        let config: Config =
            serde_json::from_reader(std::fs::File::open(model_dir.join("config.json"))?)?;
        if config.hidden_size != DIM {
            return Err(Error::Embed("Unexpected embedding dimension".into()));
        }
        let mut tokenizer =
            Tokenizer::from_file(model_dir.join("tokenizer.json")).map_err(embed_error)?;
        tokenizer
            .with_truncation(Some(TruncationParams {
                max_length: 512,
                ..Default::default()
            }))
            .map_err(embed_error)?;
        tokenizer.with_padding(Some(PaddingParams {
            strategy: PaddingStrategy::BatchLongest,
            pad_id: config.pad_token_id as u32,
            ..Default::default()
        }));
        let weights = VarBuilder::from_buffered_safetensors(
            std::fs::read(model_dir.join("model.safetensors"))?,
            DType::F32,
            &Device::Cpu,
        )
        .map_err(embed_error)?;
        let model = BertModel::load(weights, &config).map_err(embed_error)?;
        Ok(Self { model, tokenizer })
    }

    fn embed_batch(&self, batch: &[String]) -> candle_core::Result<Vec<Vec<f32>>> {
        let encodings = self
            .tokenizer
            .encode_batch(batch.iter().map(String::as_str).collect(), true)
            .map_err(|error| candle_core::Error::Msg(error.to_string()))?;
        let length = encodings[0].len();
        let shape = (batch.len(), length);
        let tensor = |values: Vec<u32>| Tensor::from_vec(values, shape, &Device::Cpu);
        let ids = tensor(
            encodings
                .iter()
                .flat_map(|e| e.get_ids().iter().copied())
                .collect(),
        )?;
        let types = tensor(
            encodings
                .iter()
                .flat_map(|e| e.get_type_ids().iter().copied())
                .collect(),
        )?;
        let mask = tensor(
            encodings
                .iter()
                .flat_map(|e| e.get_attention_mask().iter().copied())
                .collect(),
        )?;
        let hidden = self.model.forward(&ids, &types, Some(&mask))?;
        hidden.narrow(1, 0, 1)?.squeeze(1)?.to_vec2::<f32>()
    }
}

impl Embedder for BgeSmall {
    fn id(&self) -> &str {
        ID
    }

    fn embed_passages(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        let mut output = Vec::with_capacity(texts.len());
        for batch in texts.chunks(16) {
            let mut vectors = self.embed_batch(batch).map_err(embed_error)?;
            for vector in &mut vectors {
                let norm = vector.iter().map(|value| value * value).sum::<f32>().sqrt();
                if !norm.is_finite() || norm <= 0.0 {
                    return Err(Error::Embed("Invalid embedding norm".into()));
                }
                for value in vector {
                    *value /= norm;
                }
            }
            output.extend(vectors);
        }
        Ok(output)
    }

    fn embed_query(&self, text: &str) -> Result<Vec<f32>> {
        let mut vectors = self.embed_passages(&[format!("{QUERY_PREFIX}{text}")])?;
        Ok(vectors.remove(0))
    }
}
