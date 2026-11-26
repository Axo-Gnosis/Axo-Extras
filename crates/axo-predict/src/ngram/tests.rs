use std::path::Path;

use super::NgramPredictor;

fn open(dir: &Path) -> NgramPredictor {
    NgramPredictor::open(dir).unwrap()
}

fn suffix(engine: &mut NgramPredictor, before: &str, prefix: &str) -> Option<String> {
    engine.complete(before, prefix).map(|(suffix, _)| suffix)
}

fn observe_times(engine: &mut NgramPredictor, prompt: &str, times: usize) {
    for _ in 0..times {
        engine.observe(prompt);
    }
}

#[test]
fn finished_word_holds_its_own_mass() {
    let dir = tempfile::tempdir().unwrap();
    let mut engine = open(dir.path());
    assert_eq!(suffix(&mut engine, "", "th").as_deref(), Some("e"));
    // `the` is itself the likeliest word, so no longer word clears the gate.
    assert_eq!(suffix(&mut engine, "", "the"), None);
}

#[test]
fn ghost_stays_while_typing_through_it() {
    let dir = tempfile::tempdir().unwrap();
    let mut engine = open(dir.path());
    observe_times(&mut engine, "can you refactor the parser module", 20);
    observe_times(&mut engine, "the refactoring went well overall", 8);
    assert_eq!(
        suffix(&mut engine, "can you ", "re").as_deref(),
        Some("factor")
    );
    // Typing the next letter of the shown word must not swap it for a rival.
    assert_eq!(
        suffix(&mut engine, "can you ", "ref").as_deref(),
        Some("actor")
    );
    assert_eq!(
        suffix(&mut engine, "can you ", "refa").as_deref(),
        Some("ctor")
    );
}

#[test]
fn one_off_typos_are_never_offered() {
    let dir = tempfile::tempdir().unwrap();
    let mut engine = open(dir.path());
    engine.observe("please check the iterastion count and the zqorbit value");
    let typo = "iterastion";
    for k in 2..typo.len() {
        assert_ne!(
            suffix(&mut engine, "check the ", &typo[..k]).as_deref(),
            Some(&typo[k..])
        );
    }
    assert_ne!(suffix(&mut engine, "the ", "zq").as_deref(), Some("orbit"));
    // Jargon the user keeps typing graduates.
    observe_times(&mut engine, "please check the zqorbit value", 2);
    assert_eq!(suffix(&mut engine, "the ", "zq").as_deref(), Some("orbit"));
}

#[test]
fn learned_casing_beats_the_allcaps_rule() {
    let dir = tempfile::tempdir().unwrap();
    let mut engine = open(dir.path());
    observe_times(&mut engine, "log in with OAuth and open the PRs page", 4);
    assert_eq!(
        suffix(&mut engine, "log in with ", "OA").as_deref(),
        Some("uth")
    );
    assert_eq!(suffix(&mut engine, "open the ", "PR").as_deref(), Some("s"));
    assert_eq!(
        suffix(&mut engine, "log in with ", "oa").as_deref(),
        Some("uth")
    );
}

#[test]
fn snapshot_round_trips_and_rejects_unknown_versions() {
    let dir = tempfile::tempdir().unwrap();
    let queries = [
        ("can you ", "ref"),
        ("log in with ", "OA"),
        ("", "th"),
        ("please check the ", "zq"),
        ("the ", "par"),
    ];
    let mut engine = open(dir.path());
    observe_times(&mut engine, "can you refactor the parser module", 5);
    observe_times(
        &mut engine,
        "log in with OAuth then check the zqorbit value",
        3,
    );
    let expected: Vec<_> = queries
        .iter()
        .map(|&(before, prefix)| engine.complete(before, prefix))
        .collect();
    engine.persist().unwrap();
    drop(engine);

    let mut restored = open(dir.path());
    let actual: Vec<_> = queries
        .iter()
        .map(|&(before, prefix)| restored.complete(before, prefix))
        .collect();
    assert_eq!(actual, expected);
    assert!(actual.iter().filter(|hint| hint.is_some()).count() >= 4);

    let path = dir.path().join("ngram.snapshot");
    let bytes = std::fs::read(&path).unwrap();
    let mut future = bytes.clone();
    future[8] = future[8].wrapping_add(1);
    std::fs::write(&path, &future).unwrap();
    assert!(NgramPredictor::open(dir.path()).is_err());
    std::fs::write(&path, &bytes[..bytes.len() - 7]).unwrap();
    assert!(NgramPredictor::open(dir.path()).is_err());
}

#[test]
fn single_letters_complete_from_context_but_finished_words_stay_bare() {
    let dir = tempfile::tempdir().unwrap();
    let mut engine = open(dir.path());
    observe_times(
        &mut engine,
        "can you help me figure out why the build fails",
        3,
    );
    observe_times(&mut engine, "I need to figure out the release notes", 2);
    assert_eq!(
        suffix(&mut engine, "Can you help me figure ", "o").as_deref(),
        Some("ut")
    );
    // `I` is a word of its own: no ghost after the single letter.
    assert_eq!(suffix(&mut engine, "", "I"), None);
}
