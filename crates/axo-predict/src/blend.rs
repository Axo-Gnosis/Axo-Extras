//! The word completion shown when both engines answer, ported from omp's
//! `packages/coding-agent/src/predict/blend.ts`.
//!
//! Each engine's word scores its weighted confidence, plus the other engine's
//! weighted confidence when both picked the same word; the higher score wins
//! and shows at or above [`SHOW_THRESHOLD`]. On omp's replay of 400 held-out
//! prompts the blend saved 39.6 % of keystrokes against SmolLM's 37.7 % and
//! the n-gram's 32.4 %, at a similar wrong-ghost rate.

/// Default n-gram share of the blend; the language model gets the rest.
pub(crate) const DEFAULT_NGRAM_WEIGHT: f32 = 0.3;
/// Default minimum blended score to show a completion.
pub(crate) const DEFAULT_SHOW_THRESHOLD: f32 = 0.05;

/// The completion to show.
pub(crate) struct Blended {
    pub suffix: String,
    /// Blended score, at least `show_threshold`.
    pub score: f32,
    /// Confidence of the engine that produced the pick; the larger when both agree.
    pub raw: f32,
}

/// Blends each engine's `(suffix, confidence)`; `None` shows nothing.
pub(crate) fn blend(
    ngram: Option<(String, f32)>,
    lm: Option<(String, f32)>,
    ngram_weight: f32,
    show_threshold: f32,
) -> Option<Blended> {
    let ngram_weight = ngram_weight.clamp(0.0, 1.0);
    let ngram_score = ngram.as_ref().map_or(0.0, |(_, c)| ngram_weight * c);
    let lm_score = lm.as_ref().map_or(0.0, |(_, c)| (1.0 - ngram_weight) * c);
    let blended = match (ngram, lm) {
        (Some((suffix, n)), Some((other, m))) if suffix.to_lowercase() == other.to_lowercase() => {
            Blended {
                suffix,
                score: ngram_score + lm_score,
                raw: n.max(m),
            }
        }
        (Some((suffix, raw)), _) if ngram_score >= lm_score => Blended {
            suffix,
            score: ngram_score,
            raw,
        },
        (_, Some((suffix, raw))) => Blended {
            suffix,
            score: lm_score,
            raw,
        },
        (_, None) => return None,
    };
    (blended.score >= show_threshold).then_some(blended)
}
