// Ported from Basic Memory (https://github.com/basicmachines-co/basic-memory, 7a1ff5f), AGPL-3.0-or-later.
use crate::{
    Error, Result,
    format::Observation,
    temporal::{TimeKind, TimeRange},
};
use rusqlite::{Connection, OptionalExtension, params};
use std::{
    path::{Path, PathBuf},
    time::Duration,
};
const SCHEMA_VERSION: i32 = 1;
pub struct Index {
    pub(crate) conn: Connection,
}
#[derive(Debug, Clone)]
pub struct Entity {
    pub bank_id: i64,
    pub title: String,
    pub file_path: String,
}
#[derive(Debug, Clone)]
pub struct LocatedObservation {
    pub entity: Entity,
    pub observation: Observation,
}
pub(crate) fn bank_path(dir: &Path) -> Result<PathBuf> {
    match dir.canonicalize() {
        Ok(p) => Ok(p),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
            let absolute = if dir.is_absolute() {
                dir.to_owned()
            } else {
                std::env::current_dir()?.join(dir)
            };
            let mut ancestor = absolute.as_path();
            let mut rest = vec![];
            while !ancestor.exists() {
                rest.push(
                    ancestor
                        .file_name()
                        .ok_or_else(|| Error::Invalid("Invalid bank path".into()))?,
                );
                ancestor = ancestor
                    .parent()
                    .ok_or_else(|| Error::Invalid("Invalid bank path".into()))?;
            }
            let mut p = ancestor.canonicalize()?;
            for part in rest.into_iter().rev() {
                p.push(part);
            }
            Ok(p)
        }
        Err(e) => Err(e.into()),
    }
}
impl Index {
    pub fn open(path: &Path) -> Result<Self> {
        let mut conn = Connection::open(path)?;
        conn.busy_timeout(Duration::from_secs(5))?;
        conn.execute_batch("PRAGMA journal_mode=WAL; PRAGMA foreign_keys=ON;")?;
        let version: i32 = conn.query_row("PRAGMA user_version", [], |r| r.get(0))?;
        if version != SCHEMA_VERSION {
            let tx = conn.transaction()?;
            tx.execute_batch("DROP TABLE IF EXISTS chunk; DROP TABLE IF EXISTS search_index; DROP TABLE IF EXISTS relation; DROP TABLE IF EXISTS observation; DROP TABLE IF EXISTS entity; DROP TABLE IF EXISTS bank;")?;
            tx.execute_batch(SCHEMA)?;
            tx.pragma_update(None, "user_version", SCHEMA_VERSION)?;
            tx.commit()?;
        }
        Ok(Self { conn })
    }
    pub fn upsert_bank(&self, dir: &Path) -> Result<i64> {
        let p = bank_path(dir)?;
        self.conn.execute(
            "INSERT INTO bank(path) VALUES (?) ON CONFLICT(path) DO NOTHING",
            [p.to_string_lossy().as_ref()],
        )?;
        Ok(self.conn.query_row(
            "SELECT id FROM bank WHERE path=?",
            [p.to_string_lossy().as_ref()],
            |r| r.get(0),
        )?)
    }
    pub fn note_count(&self, bank_id: i64) -> Result<u64> {
        Ok(self.conn.query_row(
            "SELECT count(*) FROM entity WHERE bank_id=?",
            [bank_id],
            |r| r.get(0),
        )?)
    }
    pub fn entity(&self, id: i64) -> Result<Option<Entity>> {
        Ok(self
            .conn
            .query_row(
                "SELECT bank_id,title,file_path FROM entity WHERE id=?",
                [id],
                |r| {
                    Ok(Entity {
                        bank_id: r.get(0)?,
                        title: r.get(1)?,
                        file_path: r.get(2)?,
                    })
                },
            )
            .optional()?)
    }
    pub fn observation(&self, id: i64) -> Result<Option<LocatedObservation>> {
        match self.observation_with_entity(id)? {
            Some((observation, entity_id)) => {
                Ok(self.entity(entity_id)?.map(|entity| LocatedObservation {
                    entity,
                    observation,
                }))
            }
            None => Ok(None),
        }
    }
    pub(crate) fn observation_with_entity(&self, id: i64) -> Result<Option<(Observation, i64)>> {
        Ok(self.conn.query_row("SELECT category,content,tags,context,time_kind,time_lower,time_upper,line,entity_id FROM observation WHERE id=?",[id],|r|Ok((observation_row(r)?,r.get::<_,i64>(8)?))).optional()?)
    }
    pub fn chunks_needing_embedding(
        &self,
        model: &str,
        limit: usize,
    ) -> Result<Vec<(i64, String)>> {
        let mut stmt=self.conn.prepare("SELECT id,text FROM chunk WHERE embedding IS NULL OR model IS NOT ? ORDER BY id LIMIT ?")?;
        let rows = stmt.query_map(params![model, limit as i64], |r| Ok((r.get(0)?, r.get(1)?)))?;
        Ok(rows.collect::<rusqlite::Result<_>>()?)
    }
    pub fn set_embedding(
        &self,
        chunk_id: i64,
        model: &str,
        text: &str,
        embedding: &[f32],
    ) -> Result<()> {
        if embedding.is_empty() || embedding.iter().any(|v| !v.is_finite()) {
            return Err(Error::Invalid(
                "Embedding must contain finite floats".into(),
            ));
        }
        let bytes: Vec<u8> = embedding.iter().flat_map(|v| v.to_le_bytes()).collect();
        self.conn.execute(
            "UPDATE chunk SET model=?,embedding=? WHERE id=? AND text=?",
            params![model, bytes, chunk_id, text],
        )?;
        Ok(())
    }
}
fn observation_row(r: &rusqlite::Row<'_>) -> rusqlite::Result<Observation> {
    let kind: Option<String> = r.get(4)?;
    Ok(Observation {
        category: r.get(0)?,
        content: r.get(1)?,
        tags: serde_json::from_str(&r.get::<_, String>(2)?).unwrap_or_default(),
        context: r.get(3)?,
        time: kind
            .and_then(|k| TimeKind::parse(&k))
            .map(|kind| -> rusqlite::Result<_> {
                Ok(TimeRange {
                    kind,
                    lower: r.get(5)?,
                    upper: r.get(6)?,
                })
            })
            .transpose()?,
        line: r.get(7)?,
    })
}
const SCHEMA: &str = r"
CREATE TABLE bank(id INTEGER PRIMARY KEY,path TEXT NOT NULL UNIQUE);
CREATE TABLE entity(id INTEGER PRIMARY KEY,bank_id INTEGER NOT NULL REFERENCES bank(id) ON DELETE CASCADE,title TEXT NOT NULL,note_type TEXT NOT NULL,permalink TEXT NOT NULL,file_path TEXT NOT NULL,checksum TEXT NOT NULL,mtime REAL NOT NULL,size INTEGER NOT NULL,tags TEXT NOT NULL,created_at TEXT NOT NULL,updated_at TEXT NOT NULL,UNIQUE(bank_id,file_path));
CREATE INDEX entity_permalink ON entity(bank_id,permalink);
CREATE INDEX entity_title ON entity(bank_id,title);
CREATE TABLE observation(id INTEGER PRIMARY KEY,entity_id INTEGER NOT NULL REFERENCES entity(id) ON DELETE CASCADE,category TEXT NOT NULL,content TEXT NOT NULL,context TEXT,tags TEXT NOT NULL,time_kind TEXT,time_lower TEXT,time_upper TEXT,line INTEGER NOT NULL);
CREATE TABLE relation(id INTEGER PRIMARY KEY,from_id INTEGER NOT NULL REFERENCES entity(id) ON DELETE CASCADE,to_id INTEGER REFERENCES entity(id) ON DELETE SET NULL,to_name TEXT NOT NULL,relation_type TEXT NOT NULL,context TEXT);
CREATE INDEX relation_from ON relation(from_id);
CREATE INDEX relation_to ON relation(to_id);
CREATE VIRTUAL TABLE search_index USING fts5(title,content,permalink,row_type UNINDEXED,row_id UNINDEXED,entity_id UNINDEXED,bank_id UNINDEXED,category UNINDEXED,tokenize='unicode61 tokenchars ''/''',prefix='1,2,3,4');
CREATE TABLE chunk(id INTEGER PRIMARY KEY,entity_id INTEGER NOT NULL REFERENCES entity(id) ON DELETE CASCADE,row_type TEXT NOT NULL,row_id INTEGER NOT NULL,text TEXT NOT NULL,text_hash TEXT NOT NULL,model TEXT,embedding BLOB);
CREATE INDEX chunk_entity ON chunk(entity_id);
CREATE INDEX chunk_hash ON chunk(text_hash);
";
