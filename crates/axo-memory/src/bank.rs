use std::path::{Component, Path, PathBuf};

use crate::{Result, invalid};

/// A folder of notes, named for `memory://<name>/` URLs and `[[name::Note]]` links.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Bank {
    name: String,
    dir: PathBuf,
}

impl Bank {
    /// Fails when `name` is not a valid bank name or `dir` is not absolute.
    pub fn new(name: impl Into<String>, dir: PathBuf) -> Result<Self> {
        let name = name.into();
        if !valid_bank_name(&name) {
            return Err(invalid(format!("Invalid memory bank name `{name}`")));
        }
        if !dir.is_absolute() {
            return Err(invalid(format!(
                "Memory bank `{name}` needs an absolute directory"
            )));
        }
        Ok(Self { name, dir })
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn dir(&self) -> &Path {
        &self.dir
    }
}

/// The banks one session reaches; `default` is the one it writes to.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Banks {
    banks: Vec<Bank>,
    default: usize,
}

impl Banks {
    /// Fails when names repeat or `default` names no bank in `banks`.
    pub fn new(banks: Vec<Bank>, default: &str) -> Result<Self> {
        for (i, bank) in banks.iter().enumerate() {
            if banks[..i].iter().any(|other| other.name == bank.name) {
                return Err(invalid(format!("Memory bank `{}` is repeated", bank.name)));
            }
        }
        let default = banks
            .iter()
            .position(|bank| bank.name == default)
            .ok_or_else(|| invalid(format!("Memory bank `{default}` is not configured")))?;
        Ok(Self { banks, default })
    }

    pub fn default_bank(&self) -> &Bank {
        &self.banks[self.default]
    }

    pub fn all(&self) -> &[Bank] {
        &self.banks
    }

    pub fn get(&self, name: &str) -> Option<&Bank> {
        self.banks.iter().find(|bank| bank.name == name)
    }
}

/// `^[a-z][a-z0-9-]*$`.
pub fn valid_bank_name(name: &str) -> bool {
    let mut chars = name.chars();
    chars.next().is_some_and(|c| c.is_ascii_lowercase())
        && chars.all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '-')
}

/// `path` when it lies inside `bank` through plain components and no symlinks.
pub(crate) fn checked_path(bank: &Bank, path: &Path) -> Result<PathBuf> {
    let relative = path
        .strip_prefix(&bank.dir)
        .map_err(|_| invalid("Note is outside its bank"))?;
    if !relative
        .components()
        .all(|c| matches!(c, Component::Normal(_) | Component::CurDir))
    {
        return Err(invalid("Invalid note path"));
    }
    let mut current = bank.dir.clone();
    for component in relative.components() {
        current.push(component);
        match std::fs::symlink_metadata(&current) {
            Ok(metadata) if metadata.file_type().is_symlink() => {
                return Err(invalid("Memory notes cannot traverse symlinks"));
            }
            Ok(_) => {}
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
            Err(error) => return Err(error.into()),
        }
    }
    Ok(current)
}
