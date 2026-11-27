//! `memory://<bank>/<path>`: memory banks as ordinary files, so axo's `read`, `write`,
//! `edit`, `grep`, `find` and `bash` work on notes, record their changes, and obey the
//! session's write scope.

use std::path::PathBuf;
use std::sync::Arc;

use async_trait::async_trait;
use axo::{Scheme, SchemeContext, SchemeUrl, Target};
use axo_agent::ToolError;

use crate::{Banks, Result, bank::checked_path, invalid};

/// Runs before every scheme resolve and tool call; `Err` is shown to the model.
/// For hosts whose banks or permission can change under a live session.
pub type Guard = Arc<dyn Fn(&Banks) -> std::result::Result<(), String> + Send + Sync>;

pub struct MemoryScheme {
    banks: Banks,
    doc: String,
    guard: Option<Guard>,
}

impl MemoryScheme {
    pub fn new(banks: Banks) -> Self {
        let names = banks
            .all()
            .iter()
            .map(|bank| format!("`{}`", bank.name()))
            .collect::<Vec<_>>()
            .join(", ");
        let doc = format!(
            "`memory://<bank>/<path>`: long-term memory, Markdown notes kept across sessions; \
             read, grep, find, write and edit them like files. Banks: {names}; this project's is \
             `{}`. Write to memory only when the user asks you to remember something or states a \
             durable preference or decision. Notes may start with YAML frontmatter (`title`, \
             `type`, `tags`); facts are list items `- [category] text`, links \
             `- relation_type [[Other Note]]`.",
            banks.default_bank().name()
        );
        Self {
            banks,
            doc,
            guard: None,
        }
    }

    /// Replaces the line shown in the system prompt's `schemes` section.
    pub fn doc(mut self, doc: impl Into<String>) -> Self {
        self.doc = doc.into();
        self
    }

    pub fn guard(mut self, guard: Guard) -> Self {
        self.guard = Some(guard);
        self
    }

    fn path(&self, raw: &str) -> Result<PathBuf> {
        let raw = raw.trim_end_matches('/');
        let (name, rest) = raw.split_once('/').unwrap_or((raw, ""));
        if name.is_empty() {
            return Err(invalid(format!(
                "memory:// needs a bank, such as memory://{}/",
                self.banks.default_bank().name()
            )));
        }
        let bank = self
            .banks
            .get(name)
            .ok_or_else(|| invalid(format!("Unknown memory bank `{name}`")))?;
        if rest.starts_with('/') || rest.split('/').any(|part| part == "..") {
            return Err(invalid("memory:// paths stay inside their bank"));
        }
        checked_path(bank, &bank.dir().join(rest))
    }
}

#[async_trait]
impl Scheme for MemoryScheme {
    fn name(&self) -> &str {
        "memory"
    }

    fn prompt_doc(&self) -> &str {
        &self.doc
    }

    async fn resolve(&self, url: &SchemeUrl, _cx: &SchemeContext) -> Result<Target, ToolError> {
        if let Some(guard) = &self.guard {
            guard(&self.banks).map_err(ToolError)?;
        }
        self.path(&url.path)
            .map(|path| Target::File {
                path,
                writable: true,
            })
            .map_err(|error| ToolError(error.to_string()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Bank;

    async fn resolve(
        scheme: &MemoryScheme,
        cx: &SchemeContext,
        path: &str,
    ) -> Result<Target, ToolError> {
        let url = SchemeUrl {
            scheme: "memory".into(),
            path: path.into(),
        };
        scheme.resolve(&url, cx).await
    }

    #[tokio::test]
    async fn paths_stay_inside_their_bank() {
        let root = tempfile::tempdir().unwrap();
        let root = root.path().canonicalize().unwrap();
        let main = root.join("banks/main");
        let banks = Banks::new(vec![Bank::new("main", main.clone()).unwrap()], "main").unwrap();
        let scheme = MemoryScheme::new(banks);
        let cx = SchemeContext {
            session_id: "s".into(),
            session_dir: root.clone(),
            cwd: root.join("project"),
        };
        assert_eq!(
            resolve(&scheme, &cx, "main/proj/Note.md").await.unwrap(),
            Target::File {
                path: main.join("proj/Note.md"),
                writable: true
            }
        );
        assert!(!main.exists());
        std::fs::create_dir_all(&main).unwrap();
        for bad in ["", "other/x.md", "main/../x.md", "main//etc/passwd"] {
            assert!(resolve(&scheme, &cx, bad).await.is_err(), "{bad}");
        }
        std::os::unix::fs::symlink(&root, main.join("escape")).unwrap();
        assert!(resolve(&scheme, &cx, "main/escape/x.md").await.is_err());
    }
}
