# Axo-Extras

Optional capabilities for [axo](https://git.ryot.foo/toph/axo) hosts, kept out of axo's core. A host embeds only the crates it wants.

| Crate | What it adds |
|---|---|
| `axo-models` | Pinned local model files: verified download onto disk and a typed state (missing, downloading, ready, failed) |
| `axo-predict` | Local predictive text for prompt editors: word completion from a personal n-gram engine, plus next-word suggestion with an optional on-device language model ([README](crates/axo-predict/README.md)) |

## Subscription providers

axo speaks wire protocols; these crates sign in to a subscription and plug into axo's provider seam (`Provider::login`, `Provider::transport`, `Provider::list_models`, `Provider::usage`, `compat.toolNamePrefix`).

| Crate | Provider id | What it adds |
|---|---|---|
| `axo-codex` | `openai-codex` | ChatGPT OAuth (browser + loopback on `127.0.0.1:1455`), the account's Codex model list priced from the catalog's `openai` rows (a fixed offline floor until the first listing), usage limits (`wham/usage`: plan, 5-hour and weekly windows, per-model meters, spend cap; `x-codex-*` headers on every response), and the Codex exchange over axo's OpenAI Responses adapter |
| `axo-anthropic-auth` | `anthropic-auth` | claude.ai OAuth (pasted code), the account's Claude models (`/v1/models`, described by the catalog's first-party rows without API pricing; every first-party row while logged out), usage limits (`/api/oauth/usage` and `/api/oauth/profile`: plan, email, 5-hour, weekly and per-model weekly windows, extra usage; `anthropic-ratelimit-unified-*` headers on every response), and the subscription exchange over axo's Anthropic Messages adapter |

## Registering

```rust
models.register("axo-extras", axo_codex::provider(http.clone()))?;
models.register("axo-extras", axo_anthropic_auth::provider()?)?;
```

Credentials live in the `registered/axo-extras/<id>` slots of the host's credential store; axo lists and caches each provider's models per slot.

Usage limits are read through axo: `Models::usage(id)` returns what is known without a request, `Models::refresh_usage(id)` asks the account at most once every five minutes, and `Models::subscribe_usage()` signals changes.

## Versioning

`axo-ai` is pinned by git rev in `Cargo.toml`. It must equal the host's axo rev, or Cargo builds two `axo-ai` crates and `Provider` does not type-check. Bump both together. Crates without an `axo-ai` dependency (`axo-models`, `axo-predict`) are exempt from this rule.

`candle-core` is temporarily patched to a fork carrying an unreleased fix (see `[patch.crates-io]` in `Cargo.toml`). Patches do not propagate to dependents: a host that uses `axo-predict` must carry the same `[patch.crates-io]` line.

To develop against a local axo checkout, add an untracked `.cargo/config.toml`:

```toml
[patch."https://git.ryot.foo/toph/axo.git"]
axo-ai = { path = "../axo/crates/axo-ai" }
```

## Checks

```sh
nix develop -c cargo nextest run --workspace   # fast parallel runner (or `cargo test --workspace`)
nix develop -c cargo check --workspace --all-targets
cargo run -p axo-codex --example login -- --dir <dir> openai-codex            # login, then print usage
cargo run -p axo-codex --example login -- --dir <dir> anthropic-auth --usage   # usage only
```
