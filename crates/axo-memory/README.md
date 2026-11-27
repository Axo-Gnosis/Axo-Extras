# axo-memory

Long-term memory for an axo host: banks of plain Markdown notes holding typed facts and links, a derived SQLite index with hybrid keyword and vector search, a `memory://` scheme so the agent reads and edits notes with its ordinary file tools, a `memory_search` tool, and capture of durable facts from finished sessions by a background writer agent.

Licensed **AGPL-3.0-or-later** (see `LICENSE` and `NOTICE`); the rest of Axo-Extras is MIT.

## Layout

One `Memory` owns everything under the root the host passes to `Memory::open(root)`:

| Path | What it holds |
|---|---|
| `index.db` | The search index. Derived: rebuilt from the notes, safe to delete. |
| `captures.db` | The capture log (runs and the facts each saved) and each session's capture cursor. |
| `banks/<name>/` | A bank's notes, unless the host puts the bank elsewhere (`default_bank_dir(name)`). |
| `writer/` | The working directory of writer sessions. Their journals are deleted after each chunk. |

Opening touches no files. `index.db` is created by the first search or count; `captures.db` when a capture first records a run or a cursor. Open each root once per process: opening the capture log marks runs it finds unfinished as interrupted.

## Host contract

- **Banks.** Build a `Bank` per bank from an absolute directory (`default_bank_dir` for the default place) and a `Banks` naming the session's default one. Settings, path expansion, and which folders get memory stay with the host.
- **Scheme and tool.** Register both per session:

  ```rust
  let scheme: Arc<dyn axo::Scheme> = Arc::new(MemoryScheme::new(banks.clone()));
  let tool: Arc<dyn axo_agent::Tool> = Arc::new(SearchTool::new(memory.clone(), banks));
  let tools = vec![axo::HostTool::direct(Arc::new(move |_| tool.clone()))];
  ```

  When banks or permission can change under a live session, either re-register with `Session::set_schemes` / `set_tools`, or attach a `Guard` (`.guard(…)` on both); its `Err` text is shown to the model. The scheme's prompt line (`.doc`), the tool's description and guidelines are overridable defaults. The tool's `details` are `{"hits": [Hit], "semantic": bool}`, camelCase.
- **Embeddings.** Without an embedder, search is keyword-only. For the bundled one (feature `bge`, on by default), fetch `BGE_SMALL` with `axo_models`, load it off the async runtime with `BgeSmall::load(&BGE_SMALL.path(models_root))`, then `memory.set_embedder(Some(Arc::new(embedder)))`. Or implement `Embedder`; its `id()` is stored with every vector, and a new id re-embeds.
- **Embedding backlog.** The crate spawns no tasks. Run one loop:

  ```rust
  loop {
      memory.embedding_wanted().await;
      let memory = memory.clone();
      tokio::task::spawn_blocking(move || {
          while memory.embed_pending()? {}
          Ok::<_, axo_memory::Error>(())
      })
      .await??;
  }
  ```

  Here `?` ends the loop on the first error; a host that wants to keep going logs it instead.

- **Capture.** Call `memory.capture(&axo, CaptureRequest { … })` whenever a session should be read (on close, after compaction). It reads only what the session gained since its last capture, so repeated calls are safe; forks skip the history they copied, and child sessions and writer sessions are refused. `writer_agent()` is the bundled writer; set its `model` or pass your own spec. Captures run one at a time per `Memory`. `host` is stored with the run and returned as `CaptureRun::host`. List runs with `captures`, undo one fact with `revert_fact`.

Not configurable: the capture prompts and limits (12 facts per chunk, 20 chunks of 12,000 characters per capture, 5 candidates compared per fact), and there is no public "apply one fact" API. The writer's system prompt and model come from its agent spec.

## Notes

A note is Markdown with optional YAML frontmatter (`title`, `type`, `tags`). Facts are list items `- [category] text`, with optional `#tags`, a trailing `(context)`, and a validity range `@[from,to)` before the text; links are `- relation_type [[Other Note]]` or inline `[[Other Note]]`, and `[[bank::Note]]` crosses banks. Capture appends facts under `## Observations` and closes a replaced fact's range rather than deleting it.

## Privacy

Everything stays on the machine. Notes, the index, the capture log and writer working files are unencrypted under `root` and the bank directories. Writer journals, which hold session excerpts, are deleted after each chunk.

## Attribution

Parts of the crate are ported from [Basic Memory](https://github.com/basicmachines-co/basic-memory) (AGPL-3.0-or-later); `NOTICE` lists the files and the upstream commit. BAAI/bge-small-en-v1.5 is MIT by BAAI.
