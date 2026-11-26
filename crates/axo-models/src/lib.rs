//! Pinned local model files: verified download and state.
//!
//! A [`ModelSpec`] names a directory and the files it must hold, each pinned by
//! size and SHA-256. Hosts choose the models root, when to download and how to
//! show progress; consumers load the files themselves.

use std::fmt;
use std::path::{Path, PathBuf};

use sha2::{Digest, Sha256};
use tokio::{io::AsyncWriteExt, sync::watch};

/// One file of a model, pinned by content.
pub struct ModelFile {
    /// File name inside the model's directory.
    pub name: &'static str,
    pub url: &'static str,
    pub size: u64,
    /// Lowercase hex.
    pub sha256: &'static str,
}

/// A model as a set of pinned files.
pub struct ModelSpec {
    /// Directory name under the host's models root.
    pub dir: &'static str,
    pub files: &'static [ModelFile],
}

#[derive(Clone, Debug, PartialEq)]
pub enum ModelState {
    Missing,
    Downloading { received: u64, total: u64 },
    Ready,
    Failed(String),
}

#[derive(Debug)]
pub enum Error {
    Io(std::io::Error),
    Http(reqwest::Error),
    Mismatch {
        file: &'static str,
        reason: &'static str,
    },
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(error) => write!(f, "{error}"),
            Self::Http(error) => write!(f, "{error}"),
            Self::Mismatch { file, reason } => write!(f, "{file}: {reason}"),
        }
    }
}

impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io(error) => Some(error),
            Self::Http(error) => Some(error),
            Self::Mismatch { .. } => None,
        }
    }
}

impl From<std::io::Error> for Error {
    fn from(error: std::io::Error) -> Self {
        Self::Io(error)
    }
}

impl From<reqwest::Error> for Error {
    fn from(error: reqwest::Error) -> Self {
        Self::Http(error)
    }
}

impl ModelFile {
    fn is_complete(&self, dir: &Path) -> bool {
        std::fs::metadata(dir.join(self.name))
            .is_ok_and(|metadata| metadata.is_file() && metadata.len() == self.size)
    }

    fn mismatch(&self, reason: &'static str) -> Error {
        Error::Mismatch {
            file: self.name,
            reason,
        }
    }
}

impl ModelSpec {
    pub fn path(&self, models_root: &Path) -> PathBuf {
        models_root.join(self.dir)
    }

    /// Every file exists with its pinned size. Final names appear only after verification.
    pub fn is_complete(&self, models_root: &Path) -> bool {
        let dir = self.path(models_root);
        self.files.iter().all(|file| file.is_complete(&dir))
    }

    /// Fetches missing files, reporting progress. Sends `Ready` or `Failed` last.
    pub async fn download(
        &self,
        client: &reqwest::Client,
        models_root: &Path,
        progress: &watch::Sender<ModelState>,
    ) -> Result<(), Error> {
        let result = self.fetch(client, &self.path(models_root), progress).await;
        progress.send_replace(match &result {
            Ok(()) => ModelState::Ready,
            Err(error) => ModelState::Failed(error.to_string()),
        });
        result
    }

    async fn fetch(
        &self,
        client: &reqwest::Client,
        dir: &Path,
        progress: &watch::Sender<ModelState>,
    ) -> Result<(), Error> {
        tokio::fs::create_dir_all(dir).await?;
        let total = self.files.iter().map(|file| file.size).sum();
        let mut received = self
            .files
            .iter()
            .filter(|file| file.is_complete(dir))
            .map(|file| file.size)
            .sum();
        progress.send_replace(ModelState::Downloading { received, total });
        for file in self.files {
            if file.is_complete(dir) {
                continue;
            }
            let part = dir.join(format!("{}.part", file.name));
            let result = async {
                let mut response = client.get(file.url).send().await?.error_for_status()?;
                let mut out = tokio::fs::File::create(&part).await?;
                let mut downloaded = 0;
                let mut hash = Sha256::new();
                while let Some(bytes) = response.chunk().await? {
                    downloaded += bytes.len() as u64;
                    if downloaded > file.size {
                        return Err(file.mismatch("exceeds expected size"));
                    }
                    out.write_all(&bytes).await?;
                    hash.update(&bytes);
                    received += bytes.len() as u64;
                    progress.send_replace(ModelState::Downloading { received, total });
                }
                if downloaded != file.size {
                    return Err(file.mismatch("size mismatch"));
                }
                let digest: String = hash
                    .finalize()
                    .iter()
                    .map(|byte| format!("{byte:02x}"))
                    .collect();
                if digest != file.sha256 {
                    return Err(file.mismatch("SHA-256 mismatch"));
                }
                out.flush().await?;
                out.sync_all().await?;
                drop(out);
                tokio::fs::rename(&part, dir.join(file.name)).await?;
                Ok(())
            }
            .await;
            if let Err(error) = result {
                let _ = tokio::fs::remove_file(&part).await;
                return Err(error);
            }
        }
        Ok(())
    }
}
