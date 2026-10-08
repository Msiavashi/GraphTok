use rs_gigatoken::load_tokenizer::hf::{HfTokenizer, load_hf_slice};

#[test]
fn loads_gpt2_tokenizer_without_python_layer() {
    let path =
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/gpt2_tokenizer.json");
    let data = std::fs::read(path).expect("fixture must exist");

    let tokenizer = load_hf_slice(&data).expect("GPT-2 tokenizer should load");

    match tokenizer {
        HfTokenizer::Bpe(mut tokenizer) => {
            let mut ids = Vec::new();
            tokenizer.memoized_encode(
                rs_gigatoken::pretokenize::pretokenize_as_iter(b"Hello, world!"),
                |tokens| ids.extend_from_slice(tokens),
            );

            assert!(!ids.is_empty());
            assert_eq!(tokenizer.decode(&ids).collect::<Vec<_>>(), b"Hello, world!");
        }
        HfTokenizer::SentencePiece(_) => panic!("GPT-2 fixture must use byte-level BPE"),
    }
}

#[test]
fn native_padding_options_cover_prefix_suffix_and_left_truncation() {
    let mut options = rs_gigatoken::EncodeOptions::new(0);
    options.max_length = Some(5);
    options.pad_to_max_length = true;
    options.truncate = true;
    options.truncate_left = true;
    options.prefix = vec![101];
    options.suffix = vec![102];

    let (flat, width, lengths) = rs_gigatoken::pad_truncate_ragged(&[1, 2, 3, 4], &[4], &options)
        .expect("valid padding options");

    assert_eq!(width, 5);
    assert_eq!(lengths, vec![5]);
    assert_eq!(flat, vec![101, 2, 3, 4, 102]);
}

#[test]
fn substring_matcher_exposes_zero_overhead_helpers() {
    let matcher =
        rs_gigatoken::SubstringMatcher::new([b"<tool>".as_slice(), b"<tool_result>".as_slice()])
            .expect("matcher should compile");

    assert!(matcher.contains(b"a <tool_result> b"));
    assert_eq!(matcher.find(b"a <tool_result> b"), Some((1, 2, 15)));
    assert_eq!(
        matcher.find_all(b"<tool> <tool>"),
        vec![(0, 0, 6), (0, 7, 13)]
    );
}

#[test]
fn byte_level_roundtrip_unicode_matrix() {
    let path =
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/gpt2_tokenizer.json");
    let data = std::fs::read(path).expect("fixture must exist");
    let tokenizer = load_hf_slice(&data).expect("fixture should load");

    let HfTokenizer::Bpe(mut tokenizer) = tokenizer else {
        panic!("fixture must use byte-level BPE");
    };

    for text in [
        "Hello, world!",
        "नमस्ते दुनिया",
        "你好，世界",
        "こんにちは世界",
        "emoji 🚀 café",
        "line 1\\nline 2\\n",
    ] {
        let mut ids = Vec::new();
        tokenizer.memoized_encode_flat(
            rs_gigatoken::pretokenize::pretokenize_as_iter(text.as_bytes()),
            &mut ids,
        );
        let decoded: Vec<u8> = tokenizer
            .decode(
                &ids.iter()
                    .copied()
                    .map(rs_gigatoken::TokenId)
                    .collect::<Vec<_>>(),
            )
            .collect();
        assert_eq!(decoded, text.as_bytes(), "round-trip failed for {text:?}");
    }
}

#[test]
fn forbidden_matcher_is_opt_in_on_the_bpe_hot_path() {
    let path =
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/gpt2_tokenizer.json");
    let data = std::fs::read(path).expect("fixture must exist");
    let tokenizer = load_hf_slice(&data).expect("fixture should load");
    let HfTokenizer::Bpe(mut tokenizer) = tokenizer else {
        panic!("fixture must use byte-level BPE");
    };

    let matcher = rs_gigatoken::SubstringMatcher::new([b"SECRET".as_slice()]).unwrap();
    assert!(
        tokenizer
            .encode_with_forbidden(b"normal text", &matcher)
            .is_ok()
    );
    assert!(
        tokenizer
            .encode_with_forbidden(b"contains SECRET text", &matcher)
            .is_err()
    );
}
