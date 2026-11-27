use std::{
    cell::RefCell,
    collections::{HashMap, HashSet},
    path::{Path, PathBuf},
    sync::{Arc, Mutex},
};

use tokio::sync::Notify;

use crate::{Bank, Banks, Embedder, Index, Result, capture::log::Log, links, lock, search, sync};

// ponytail: links shown per hit; a heavily linked note lists only its first ones.
const MAX_LINKS: usize = 10;

/// One memory store. Everything it keeps lives under `root`:
/// `index.db` (derived, rebuilt from the notes), `captures.db` (capture log and
/// cursors), `banks/<name>/` (default bank location), `writer/` (writer sessions' cwd).
pub struct Memory {
    root: PathBuf,
    index: Mutex<Option<Index>>,
    embedder: Mutex<Option<Arc<dyn Embedder>>>,
    wanted: Notify,
    pub(crate) log: Log,
    pub(crate) capturing: tokio::sync::Mutex<()>,
}

#[derive(Default)]
pub struct SearchRequest {
    pub query: String,
    pub types: Vec<String>,
    pub categories: Vec<String>,
    pub tags: Vec<String>,
    pub path: Option<String>,
    pub after: Option<String>,
    pub valid_at: Option<String>,
    /// Search every bank, not just the default one.
    pub all_banks: bool,
    pub limit: usize,
}

/// A search hit with its note's `memory://` URL and, on the first hit from each note,
/// the notes it links to and from.
#[derive(serde::Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Hit {
    #[serde(flatten)]
    pub hit: search::SearchHit,
    pub url: String,
    pub links: Vec<NoteLink>,
}

#[derive(serde::Serialize)]
#[serde(rename_all = "camelCase")]
pub struct NoteLink {
    pub relation_type: String,
    pub direction: links::Direction,
    pub url: String,
}

impl Memory {
    /// Touches no files; each file is created on first write. Open a root once per
    /// process: opening the capture log closes the runs it finds unfinished.
    pub fn open(root: PathBuf) -> Arc<Self> {
        Arc::new(Self {
            log: Log::new(root.join("captures.db")),
            root,
            index: Mutex::new(None),
            embedder: Mutex::new(None),
            wanted: Notify::new(),
            capturing: tokio::sync::Mutex::new(()),
        })
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    /// `<root>/banks/<name>`: where a bank lives unless the host puts it elsewhere.
    pub fn default_bank_dir(&self, name: &str) -> PathBuf {
        self.root.join("banks").join(name)
    }

    pub(crate) fn writer_dir(&self) -> PathBuf {
        self.root.join("writer")
    }

    /// Sets or removes the embedder. Without one, search is keyword-only.
    pub fn set_embedder(&self, embedder: Option<Arc<dyn Embedder>>) {
        let some = embedder.is_some();
        *lock(&self.embedder) = embedder;
        if some {
            self.wanted.notify_one();
        }
    }

    pub fn has_embedder(&self) -> bool {
        lock(&self.embedder).is_some()
    }

    fn embedder(&self) -> Option<Arc<dyn Embedder>> {
        lock(&self.embedder).clone()
    }

    fn with_index<T>(&self, f: impl FnOnce(&mut Index) -> Result<T>) -> Result<T> {
        let mut slot = lock(&self.index);
        let index = match &mut *slot {
            Some(index) => index,
            empty => {
                std::fs::create_dir_all(&self.root)?;
                empty.insert(Index::open(&self.root.join("index.db"))?)
            }
        };
        f(index)
    }

    fn sync(&self, index: &mut Index, banks: &[Bank]) -> Result<()> {
        // ponytail: scan on every call; add notify if the bank walk shows up in profiles.
        let mut changed = false;
        for bank in banks {
            changed |= sync::sync_bank(index, bank.dir())?.has_changes();
        }
        if changed {
            self.wanted.notify_one();
        }
        Ok(())
    }

    /// Runs `f` off the async runtime against the index, after syncing `banks`.
    pub(crate) async fn indexed<T: Send + 'static>(
        self: &Arc<Self>,
        banks: Vec<Bank>,
        f: impl FnOnce(&Index) -> Result<T> + Send + 'static,
    ) -> Result<T> {
        let memory = Arc::clone(self);
        tokio::task::spawn_blocking(move || {
            memory.with_index(|index| {
                memory.sync(index, &banks)?;
                f(index)
            })
        })
        .await?
    }

    pub async fn search(
        self: &Arc<Self>,
        banks: &Banks,
        request: SearchRequest,
    ) -> Result<Vec<Hit>> {
        let embedder = self.embedder();
        let model = embedder
            .as_ref()
            .map_or_else(String::new, |embedder| embedder.id().to_owned());
        let embedding = match embedder {
            Some(embedder) => {
                let query = request.query.clone();
                match tokio::task::spawn_blocking(move || embedder.embed_query(&query)).await? {
                    Ok(embedding) => Some(embedding),
                    Err(error) => {
                        tracing::warn!(%error, "Memory search is keyword-only");
                        None
                    }
                }
            }
            None => None,
        };
        let selected = if request.all_banks {
            banks.all().to_vec()
        } else {
            vec![banks.default_bank().clone()]
        };
        let banks = banks.clone();
        self.indexed(selected.clone(), move |index| {
            // Links cross banks, so every bank this session reaches names its URLs.
            let names = banks
                .all()
                .iter()
                .map(|bank| Ok((index.upsert_bank(bank.dir())?, bank.name().to_owned())))
                .collect::<Result<HashMap<_, _>>>()?;
            let ids = selected
                .iter()
                .map(|bank| index.upsert_bank(bank.dir()))
                .collect::<Result<Vec<_>>>()?;
            let types = request.types.iter().map(String::as_str).collect::<Vec<_>>();
            let hits = search::search(
                index,
                &search::SearchQuery {
                    bank_ids: &ids,
                    text: &request.query,
                    query_embedding: embedding.as_deref(),
                    embed_model: &model,
                    row_types: &types,
                    categories: &request.categories,
                    tags: &request.tags,
                    path_prefix: request.path.as_deref(),
                    after: request.after.as_deref(),
                    valid_at: request.valid_at.as_deref(),
                    limit: request.limit.clamp(1, 50),
                    ..Default::default()
                },
            )?;
            let url = |bank_id: i64, file_path: &str| {
                format!(
                    "memory://{}/{file_path}",
                    names.get(&bank_id).map_or("?", String::as_str)
                )
            };
            let resolvers = RefCell::new(HashMap::<String, links::Resolver>::new());
            let resolve_cross = |name: &str, target: &str| -> Result<Option<i64>> {
                let Some(bank) = banks.get(name) else {
                    return Ok(None);
                };
                let mut resolvers = resolvers.borrow_mut();
                if !resolvers.contains_key(name) {
                    let resolver = links::Resolver::load(index, index.upsert_bank(bank.dir())?)?;
                    resolvers.insert(name.to_owned(), resolver);
                }
                Ok(resolvers[name].resolve(target))
            };
            let mut listed = HashSet::new();
            hits.into_iter()
                .map(|hit| {
                    let mut note_links = Vec::new();
                    if listed.insert(hit.entity_id) {
                        for neighbor in links::neighbors(index, hit.entity_id, &resolve_cross)?
                            .into_iter()
                            .take(MAX_LINKS)
                        {
                            if let Some(entity) = index.entity(neighbor.entity_id)? {
                                note_links.push(NoteLink {
                                    relation_type: neighbor.relation_type,
                                    direction: neighbor.direction,
                                    url: url(entity.bank_id, &entity.file_path),
                                });
                            }
                        }
                    }
                    Ok(Hit {
                        url: url(hit.bank_id, &hit.file_path),
                        hit,
                        links: note_links,
                    })
                })
                .collect()
        })
        .await
    }

    /// Notes in `bank` after syncing it.
    pub async fn note_count(self: &Arc<Self>, bank: &Bank) -> Result<u64> {
        self.indexed(vec![bank.clone()], {
            let dir = bank.dir().to_owned();
            move |index| index.note_count(index.upsert_bank(&dir)?)
        })
        .await
    }

    /// Resolves when notes may be waiting for vectors: an embedder was set, or a sync changed notes.
    pub async fn embedding_wanted(&self) {
        self.wanted.notified().await;
    }

    /// Embeds up to 16 waiting chunks. `true` when it embedded any, so call again.
    /// Blocks; run it off the async runtime. Does nothing without an embedder or an index file.
    pub fn embed_pending(&self) -> Result<bool> {
        let Some(embedder) = self.embedder() else {
            return Ok(false);
        };
        // Do not create an index just because an embedder became available.
        if !self.root.join("index.db").exists() {
            return Ok(false);
        }
        let model = embedder.id();
        let chunks = self.with_index(|index| index.chunks_needing_embedding(model, 16))?;
        if chunks.is_empty() {
            return Ok(false);
        }
        let texts: Vec<String> = chunks.iter().map(|(_, text)| text.clone()).collect();
        // Inference runs unlocked; the text guard skips a chunk whose ID sync recycled meanwhile.
        let vectors = embedder.embed_passages(&texts)?;
        self.with_index(|index| {
            for ((id, text), vector) in chunks.iter().zip(vectors) {
                index.set_embedding(*id, model, text, &vector)?;
            }
            Ok(())
        })?;
        Ok(true)
    }
}
