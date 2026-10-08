//! Single-threaded SentencePiece encode throughput on OWT, mirroring
//! `encode_st` (GPT-2 byte-level) for a like-for-like comparison.
//!
//! Select the tokenizer with SP_TOKENIZER=tinyllama (default) or sp4096;
//! cap the input with ENCODE_MB like `encode_st`.

use rs_gigatoken::load_tokenizer::hf::load_hf_sentencepiece;
use rs_gigatoken::load_tokenizer::hub;
use std::path::PathBuf;
use std::time::Instant;

mod common;

fn main() {
    let which = std::env::var("SP_TOKENIZER").unwrap_or_else(|_| "tinyllama".to_string());
    let tokenizer_path = match which.as_str() {
        "tinyllama" => {
            let repo = "TinyLlama/TinyLlama-1.1B-Chat-v1.0";
            let path =
                PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("data/tinyllama_tokenizer.json");

            if path.exists() {
                eprintln!("Loading {which} tokenizer from {path:?}...");
                path
            } else {
                eprintln!(
                    "Loading {which} tokenizer.json from HuggingFace Hub repository {repo:?}..."
                );
                hub::hub_file(repo, "tokenizer.json", "main")
                    .expect("Could not fetch tokenizer.json from the HuggingFace Hub")
            }
        }
        "sp4096" => {
            let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join("data/fineweb_4096_bpe_tokenizer.json");

            if path.exists() {
                eprintln!("Loading {which} tokenizer from {path:?}...");
                path
            } else {
                eprintln!("Skipping encode_st_sp benchmark: tokenizer not found at {path:?}.");
                eprintln!(
                    "Set SP_TOKENIZER=tinyllama or provide data/fineweb_4096_bpe_tokenizer.json."
                );
                return;
            }
        }
        other => {
            eprintln!("Skipping encode_st_sp benchmark: unknown SP_TOKENIZER {other:?}.");
            eprintln!("Use SP_TOKENIZER=tinyllama or SP_TOKENIZER=sp4096.");
            return;
        }
    };

    let tokenizer = match load_hf_sentencepiece(&tokenizer_path) {
        Ok(tokenizer) => tokenizer,
        Err(err) => {
            eprintln!("Skipping encode_st_sp benchmark: could not load tokenizer:");
            eprintln!("{err}");
            return;
        }
    };

    let owt_path = std::env::home_dir().unwrap().join("data/owt_train.txt");
    if !owt_path.exists() {
        eprintln!(
            "Skipping encode_st_sp benchmark: OpenWebText dataset not found at {owt_path:?}."
        );
        eprintln!("Install ~/data/owt_train.txt to run the benchmark.");
        return;
    }

    if !common::has_owt() {
        eprintln!("Skipping benchmark: OpenWebText dataset not found at ~/data/owt_train.txt.");
        eprintln!("Install ~/data/owt_train.txt to run this benchmark.");
        return;
    }
    let input = common::load_owt_input(None);
    let size_gb = input.len() as f64 / 1e9;
    let text = std::str::from_utf8(&input).expect("input must be UTF-8");

    eprintln!("Encoding (single-threaded)...");
    // Count-only callback, mirroring encode_st's measurement of the GPT-2
    // path (the full encode runs; only output materialization is skipped).
    let passes: usize = std::env::var("SP_PASSES")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(2);
    let mut state = rs_gigatoken::EncodeState::new();
    for pass in 1..=passes {
        let mut total_tokens: usize = 0;
        let start = Instant::now();
        tokenizer.encode_raw_cb(&mut state, text, &mut |tokens: &[_]| {
            total_tokens += tokens.len();
        });
        let elapsed = start.elapsed().as_secs_f64();
        let throughput_gb = size_gb / elapsed;
        eprintln!(
            "pass {pass} (cache {}): {total_tokens} tokens in {elapsed:.2}s — {throughput_gb:.2} GB/s ({:.0} MB/s), {} cached units",
            if pass == 1 { "cold" } else { "warm" },
            throughput_gb * 1000.0,
            state.cache_size()
        );
    }
}
