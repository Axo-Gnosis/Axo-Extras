# Axo-Extras

Optional, host-agnostic capabilities for axo hosts that stay out of axo's core: subscription providers, local model files, and predictive text.

## The Axo Family

Shared by the AGENTS.md of axo, Axo-Extras, Axome, and Axophyte; change all four together.

- **axo**: the embeddable agent-harness library. It owns harness architecture and the public API every host builds on, designed for any host, never one.
- **Axo-Extras**: optional, opinionated capabilities core should not impose (subscription providers, local models, predictive text). Host-agnostic, built only on axo's public API to axo's standard; each host adopts, ignores, or replaces each crate.
- **Axome**: Toph's desktop host. It proves axo's surface and exposes its gaps; its product choices never define axo or Extras.
- **Axophyte**: Toph's personal persona, workflows, and extensions. It inspires ideas but has no bearing on the engineering of the others.

Dependencies point one way: Axophyte runs on hosts, Axome builds on axo and Extras, Extras builds on axo. A lower layer never names, special-cases, or assumes a higher one.

Where work belongs:

1. **axo**: every host needs it for correctness or efficiency, or hosts cannot build it themselves through the public surface (a missing seam, hook, typed datum, or operation).
2. **Axo-Extras**: a complete, reusable feature some hosts will want but core should not impose. When it needs something axo lacks, add the general seam to axo; never reach into internals.
3. **Axome**: Axome's product: its UI, settings, presentation, bundled modes and agents, and policy choices.
4. **Axophyte**: Toph's own workflow: persona text, prompts, personal workflows such as /cleanup and /decomplect, and personal tools.

"Axome needs X" or "Axophyte does X" is evidence of a need, not the design. Shape the change for every consumer of the layer it lands in, then adopt it upward.

Axophyte ("Axo") may also be the persona you are running as. Its instructions govern how you work, not these repositories' product or architecture.

### What this means in Axo-Extras

- Design each crate as if it shipped inside axo, under axo's embedder-first rule: the crate owns its capability's correctness and efficient defaults; the host owns policy, presentation, and interpretation. Expose anything a host needs as typed data and typed methods.
- Stay host-agnostic. No Axome paths, settings keys, UI assumptions, or Axophyte conventions; the host supplies directories, HTTP clients, credentials, and configuration.
- Each crate is a separate opt-in. A crate depends on another only when its job requires it (axo-predict uses axo-models for model files); a host embeds only what it wants.
- Build only on axo's public API. If a crate needs a seam axo lacks, add the general seam to axo first, never a workaround here.
- Hold axo's code invariants: no unsafe code (denied workspace-wide), never a std MutexGuard across an await, provider streams that report failures as a terminal error event rather than a Result, camelCase wire and config JSON, and offline deterministic tests (live smoke runs go in examples/).

## Living Documentation

- **README.md**: the crate table, provider registration, versioning, and checks. Update it with every crate, API, or behavior change.
- **crates/<crate>/README.md**: the reference for a crate that needs more than a table row (currently axo-predict).

## Versioning

The axo-ai git rev in Cargo.toml must equal the host's axo rev; bump both together. The candle-core patch must be mirrored by every host that uses axo-predict. See README.md for both and for developing against a local axo checkout.

## Development

Run inside `nix develop`:

```bash
cargo check --workspace --all-targets
cargo nextest run --workspace   # or `cargo test --workspace`
cargo clippy --workspace --all-targets -- -D warnings
cargo fmt --check
```
