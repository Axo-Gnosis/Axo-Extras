//! Replays typed prompts through the predictor with the language model on and
//! prints latency and quality for `next_words` 0, 1 and 2.
//!
//! ```sh
//! cargo run --release -p axo-predict --example replay -- <prompts.json> <models_root> [held=60]
//! ```
//!
//! `prompts.json` is a JSON array of prompt strings, oldest first. SmolLM2 is
//! downloaded into `models_root` when missing. Each pass opens a fresh
//! predictor, learns every prompt but the last `held` (the newest
//! [`MEMORY_PROMPTS`] of them with the model loaded, so its prompt memory is
//! full as in steady use), then types each held-out prompt one character at a
//! time, accepting every ghost that matches what comes next, and learns it.

use std::{path::Path, time::Instant};

use anyhow::Context as _;
use axo_predict::{Options, Predictor, SMOLLM2};

/// Prompts learned with the model loaded before the replay.
const MEMORY_PROMPTS: usize = 30;

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let [prompts, root, rest @ ..] = args.as_slice() else {
        anyhow::bail!("usage: replay <prompts.json> <models_root> [held=60]");
    };
    let held: usize = rest.first().map_or(Ok(60), |n| n.parse())?;
    let prompts: Vec<String> = serde_json::from_slice(&std::fs::read(prompts)?)?;
    anyhow::ensure!(held < prompts.len(), "held must be below the prompt count");
    let root = Path::new(root);
    if !SMOLLM2.is_complete(root) {
        eprintln!("downloading SmolLM2 into {}", root.display());
        let (progress, _) = tokio::sync::watch::channel(axo_models::ModelState::Missing);
        tokio::runtime::Runtime::new()?.block_on(SMOLLM2.download(
            &reqwest::Client::new(),
            root,
            &progress,
        ))?;
    }
    let (train, test) = prompts.split_at(prompts.len() - held);
    println!(
        "{} prompts learned, {} replayed ({} characters)",
        train.len(),
        test.len(),
        test.iter().map(|p| p.chars().count()).sum::<usize>()
    );
    for next_words in 0..=2 {
        let stats = pass(train, test, &SMOLLM2.path(root), next_words)?;
        stats.print(next_words);
    }
    Ok(())
}

fn pass(train: &[String], test: &[String], model: &Path, next_words: u8) -> anyhow::Result<Stats> {
    let dir = tempfile::tempdir()?;
    let mut predictor = Predictor::open(Options {
        state_dir: dir.path().to_owned(),
    })?;
    let (early, recent) = train.split_at(train.len().saturating_sub(MEMORY_PROMPTS));
    early.iter().for_each(|prompt| predictor.observe(prompt));
    let started = Instant::now();
    predictor.set_model(Some(model)).context("load model")?;
    let load = started.elapsed().as_secs_f64();
    recent.iter().for_each(|prompt| predictor.observe(prompt));
    predictor.set_next_words(next_words);

    let mut stats = Stats {
        load,
        ..Stats::default()
    };
    for prompt in test {
        stats.replay(&mut predictor, prompt);
        let started = Instant::now();
        predictor.observe(prompt);
        stats.observe_ms.push(ms(started));
    }
    Ok(stats)
}

fn ms(started: Instant) -> f64 {
    started.elapsed().as_secs_f64() * 1e3
}

#[derive(Default)]
struct Stats {
    load: f64,
    latency_ms: Vec<f64>,
    observe_ms: Vec<f64>,
    characters: usize,
    keystrokes: usize,
    /// Distinct ghosts by words beyond the current one: shown, matched.
    ghosts: [(usize, usize); 3],
}

impl Stats {
    fn replay(&mut self, predictor: &mut Predictor, prompt: &str) {
        self.characters += prompt.chars().count();
        let mut pos = prompt.chars().next().map_or(0, char::len_utf8);
        self.keystrokes += 1;
        let mut last_ghost = String::new();
        while pos < prompt.len() {
            let typed = &prompt[..pos];
            let started = Instant::now();
            let suggestion = predictor.complete(typed, "");
            self.latency_ms.push(ms(started));
            self.keystrokes += 1;
            if let Some(suggestion) = suggestion {
                let text = suggestion.text;
                let rest = &prompt[pos..];
                let matched = rest.starts_with(&text)
                    && rest[text.len()..]
                        .chars()
                        .next()
                        .is_none_or(|c| !is_word_char(c));
                let ghost = format!("{typed}{text}");
                if ghost != last_ghost {
                    // A word being completed, or a next-word ghost after spaces.
                    let words = text.matches(' ').count() + usize::from(typed.ends_with(' '));
                    let category = &mut self.ghosts[words.min(2)];
                    category.0 += 1;
                    category.1 += usize::from(matched);
                    last_ghost = ghost;
                }
                if matched {
                    pos += text.len();
                    continue;
                }
            }
            pos += prompt[pos..].chars().next().map_or(1, char::len_utf8);
        }
    }

    fn print(&self, next_words: u8) {
        let mut latency = self.latency_ms.clone();
        latency.sort_by(f64::total_cmp);
        let at = |q: f64| latency[((latency.len() - 1) as f64 * q).round() as usize];
        let observe = self.observe_ms.iter().copied().fold(0.0, f64::max);
        let saved = 100.0 * (1.0 - self.keystrokes as f64 / self.characters as f64);
        println!("\nnext_words = {next_words}");
        println!(
            "  complete: {} calls, p50 {:.1} ms, p90 {:.1} ms, p99 {:.1} ms (model load {:.2} s, observe max {observe:.0} ms)",
            latency.len(),
            at(0.5),
            at(0.9),
            at(0.99),
            self.load,
        );
        println!("  keystrokes saved: {saved:.1} %");
        let total: usize = self.ghosts.iter().map(|g| g.0).sum();
        println!("  ghosts shown: {total}");
        for (name, (shown, matched)) in ["word", "+1 word", "+2 words"].iter().zip(self.ghosts) {
            let rate = if shown == 0 {
                0.0
            } else {
                100.0 * matched as f64 / shown as f64
            };
            println!("    {name:<9} {shown:>5} shown, {matched:>5} matched ({rate:.1} %)");
        }
    }
}

/// Letters, digits and apostrophes continue a word.
fn is_word_char(c: char) -> bool {
    c.is_alphanumeric() || c == '\''
}
