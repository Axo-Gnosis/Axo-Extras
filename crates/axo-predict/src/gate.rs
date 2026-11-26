//! The caret gate: whether a caret position may show a suggestion, and what
//! kind. Ported from omp's editor gates (`prose-gate.ts`,
//! `word-completion.ts`) so no host reimplements them.

use crate::prose::{
    MAX_BUFFER, MAX_LINE, is_blank, is_code_token, is_command_line, is_js_space, is_word_char,
    mask_non_prose, utf16_len,
};

/// Left context handed to the engines, in `char`s.
const TAIL_CHARS: usize = 2_000;

/// What to suggest at the caret.
#[derive(Debug, PartialEq)]
pub(crate) enum Caret<'a> {
    /// Complete the word being typed.
    Word { before: &'a str, prefix: &'a str },
    /// Suggest the next word; `before` ends at the last word, spaces trimmed.
    Next { before: &'a str },
}

/// Classifies a caret between `before` and `after` (one run of typed text);
/// `None` when nothing may be suggested there.
pub(crate) fn caret<'a>(before: &'a str, after: &str) -> Option<Caret<'a>> {
    if utf16_len(before) + utf16_len(after) > MAX_BUFFER {
        return None;
    }
    let full = format!("{before}{after}");
    let line_start = before.rfind('\n').map_or(0, |at| at + 1);
    let line_end = before.len() + after.find('\n').unwrap_or(after.len());
    let line = &full[line_start..line_end];
    if utf16_len(line) > MAX_LINE
        || after.chars().next().is_some_and(is_word_char)
        || is_command_line(line)
    {
        return None;
    }

    let start = before.trim_end_matches(is_word_char).len();
    let prefix = &before[start..];
    let mask = mask_non_prose(&full);
    if !prefix.is_empty() {
        let token_start = before[..start].trim_end_matches(|c| !is_js_space(c)).len();
        let token_end = before.len() + after.find(is_js_space).unwrap_or(after.len());
        if is_code_token(&full[token_start..token_end]) || is_blank(&mask[start..before.len()]) {
            return None;
        }
        return Some(Caret::Word {
            before: tail(&before[..start]),
            prefix,
        });
    }

    if !before.ends_with(' ') || !(after.is_empty() || after.starts_with('\n')) {
        return None;
    }
    let trimmed = before.trim_end_matches(' ');
    let last = trimmed.chars().next_back()?;
    if !(is_word_char(last) || matches!(last, ',' | '.' | ';' | '!' | '?'))
        || mask.as_bytes()[trimmed.len() - 1] == b' '
    {
        return None;
    }
    Some(Caret::Next {
        before: tail(trimmed),
    })
}

/// The last [`TAIL_CHARS`] characters of `text`.
fn tail(text: &str) -> &str {
    text.char_indices()
        .rev()
        .nth(TAIL_CHARS - 1)
        .map_or(text, |(at, _)| &text[at..])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gate_rules() {
        assert_eq!(caret("", ""), None);
        assert_eq!(
            caret("th", ""),
            Some(Caret::Word {
                before: "",
                prefix: "th"
            })
        );
        assert!(matches!(caret("fix the ", ""), Some(Caret::Next { .. })));
        assert_eq!(caret(" ", ""), None);
        assert_eq!(caret("fix th", "is"), None);
        assert_eq!(caret("see src/ma", ""), None);
        assert_eq!(caret("/mod", ""), None);
        assert_eq!(caret("fix the ", "bug"), None);
    }
}
