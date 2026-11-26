# axo-predict

Local predictive text for a prompt editor: completes the word being typed and, with the optional language model, suggests up to two following words.

- **N-gram engine** (always on): a personal model learned from the prompts the user submits, seeded with an English web prior. Ported from [oh-my-pi](https://github.com/can1357/oh-my-pi) (see `NOTICE`).
- **Language model** (opt-in): SmolLM2-135M, 8-bit GGUF, run on the CPU through candle. Fetch it with `axo_models` using `SMOLLM2`, then call `Predictor::set_model`. Its suggestions are blended with the n-gram's; only it suggests next words.

## Host contract

- `complete(before, after)`: pass the one run of typed text holding the caret, split at the caret. Exclude text belonging to file chips, folded pastes and other non-typed parts. The crate applies the caret gate itself (code tokens, paths, command lines, mid-word carets, empty drafts all return `None`).
- `observe(text)`: call once per typed run of a submitted prompt, off the typing path. `set_model` is also slow (loads 145 MB, consumes the prompt memory); call it off the typing path.
- `persist()`: writes only when something was learned; call it periodically and at shutdown.
- `open` fails on a corrupt or unknown-version snapshot; recover with `reset_dir` and open again.
- candle-core is patched to a fork in this workspace (see the root README); hosts must carry the same `[patch.crates-io]` line or the language model runs 3–4× slower.

## Privacy

Everything stays on the machine. Word counts (`ngram.snapshot`) and recent prompts (`memory.json`) are stored unencrypted under `state_dir`.

## Limits

English only. Next-word suggestions appear only at the end of a line after plain spaces.

## Bench

`cargo run --release -p axo-predict --example replay -- <prompts.json> <models_root> [held=60]` replays held-out prompts keystroke by keystroke and prints latency and match rates. Final gate numbers with the patched candle are still to be recorded here.

## Attribution

`src/ngram/`, `src/prose.rs`, `src/lm/search.rs` and `data/web-prior.bin.zst` derive from oh-my-pi (MIT) at commit `fbb8085ff432419132c6e3132f36ad857893b1a4`. SmolLM2 is Apache-2.0 by Hugging Face.
