use crate::Result;

/// Turns text into unit-length vectors. Calls block; `Memory` makes them off the async runtime.
// ponytail: sync trait; a network embedder blocks its worker thread. Make it async if one needs it.
pub trait Embedder: Send + Sync {
    /// Identifies the model and revision. Stored with every vector; a different id re-embeds.
    fn id(&self) -> &str;
    fn embed_passages(&self, texts: &[String]) -> Result<Vec<Vec<f32>>>;
    fn embed_query(&self, text: &str) -> Result<Vec<f32>>;
}
