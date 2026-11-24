# Axo-Extras

Subscription providers for [axo](https://git.ryot.foo/toph/axo), kept out of axo's core. axo speaks wire protocols; these crates sign in to a subscription and plug into axo's provider seam (`Provider::login`, `Provider::transport`, `Provider::list_models`, `compat.toolNamePrefix`).

| Crate | Provider id | What it adds |
|---|---|---|
| `axo-codex` | `openai-codex` | ChatGPT OAuth (browser + loopback on `127.0.0.1:1455`), the account's Codex model list priced from the catalog's `openai` rows (a fixed offline floor until the first listing), and the Codex exchange over axo's OpenAI Responses adapter |
| `axo-claude-subscription` | `anthropic-subscription` | claude.ai OAuth (pasted code), the catalog's first-party Claude models without API pricing, and the subscription exchange over axo's Anthropic Messages adapter |

## Registering

```rust
models.register("axo-extras", axo_codex::provider(http.clone()))?;
models.register("axo-extras", axo_claude_subscription::provider()?)?;
```

Credentials live in the `registered/axo-extras/<id>` slots of the host's credential store; axo lists and caches each provider's models per slot.

## Versioning

`axo-ai` is pinned by git rev in `Cargo.toml`. It must equal the host's axo rev, or Cargo builds two `axo-ai` crates and `Provider` does not type-check. Bump both together.

To develop against a local axo checkout, add an untracked `.cargo/config.toml`:

```toml
[patch."https://git.ryot.foo/toph/axo.git"]
axo-ai = { path = "../axo/crates/axo-ai" }
```

## Checks

```sh
nix develop -c cargo test --workspace
cargo run -p axo-codex --example login -- --dir <dir> openai-codex
```
