# Axo-Extras

Optional capabilities for [axo](https://git.ryot.foo/toph/axo) hosts, kept out of axo's core. These are features worth having that not every host wants, or wants done this way. Each crate is built only on axo's public API, to axo's engineering standards, and assumes nothing about its host ([Axome](https://github.com/tophc7/Axome) is one consumer). A host embeds only the crates it wants.

| Crate | What it adds |
|---|---|
| `axo-memory` | Long-term memory: linked Markdown notes in banks, hybrid keyword and vector search (bundled BGE-small embedder), a `memory://` scheme and `memory_search` tool, and capture of durable facts from sessions by a writer agent. AGPL-3.0-or-later ([README](crates/axo-memory/README.md)) |
| `axo-models` | Pinned local model files: verified download onto disk and a typed state (missing, downloading, ready, failed) |
| `axo-predict` | Local predictive text for prompt editors: word completion from a personal n-gram engine, plus next-word suggestion with an optional on-device language model ([README](crates/axo-predict/README.md)) |

## Licences

Every crate is MIT except `axo-memory`, which is AGPL-3.0-or-later because it ports [Basic Memory](https://github.com/basicmachines-co/basic-memory) (see its `NOTICE`).

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

`axo`, `axo-agent` and `axo-ai` are pinned by one git rev in `Cargo.toml`. It must equal the host's axo rev, or Cargo builds two copies of each and `Provider`, `Scheme` and `Tool` do not type-check. Bump both together. Crates without an axo dependency (`axo-models`, `axo-predict`) are exempt from this rule.

`axo-memory` depends on `rusqlite` with bundled SQLite; its version must equal the host's, because two bundled SQLite copies do not link.

`candle-core` is temporarily patched to a fork carrying an unreleased fix (see `[patch.crates-io]` in `Cargo.toml`). Patches do not propagate to dependents: a host that uses `axo-predict` or `axo-memory` must carry the same `[patch.crates-io]` line.

To develop against a local axo checkout, add an untracked `.cargo/config.toml`:

```toml
[patch."https://git.ryot.foo/toph/axo.git"]
axo = { path = "../axo/crates/axo" }
axo-agent = { path = "../axo/crates/axo-agent" }
axo-ai = { path = "../axo/crates/axo-ai" }
```

## Checks

```sh
nix develop -c cargo nextest run --workspace   # fast parallel runner (or `cargo test --workspace`)
nix develop -c cargo check --workspace --all-targets
cargo run -p axo-codex --example login -- --dir <dir> openai-codex            # login, then print usage
cargo run -p axo-codex --example login -- --dir <dir> anthropic-auth --usage   # usage only
```
