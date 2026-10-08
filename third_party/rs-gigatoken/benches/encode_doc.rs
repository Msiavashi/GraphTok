//! Whole-document multithreaded encode benchmark, mirroring
//! `BPETokenizer.encode_files` on a single plain-text file: the entire input
//! is ONE document handed to the library's parallel encode path
//! (`encode_docs_ragged`), which splits it at pretoken-safe boundaries
//! (token-identical to a serial pass), encodes with a persistent worker
//! pool, and gathers one flat id buffer.
//!
//! One round per process by default: a first pass over a fresh dataset is
//! the workload that matters, and everything a second in-process round
//! reuses — worker pretoken caches, allocator arenas, faulted pages, grown
//! buffer capacities — makes later rounds unrealistically fast. Restart the
//! binary to collect independent samples; ENCODE_ROUNDS>1 remains available
//! for cache-behavior experiments.
//!
//! Run with: cargo bench --bench encode_doc              (2 GB default)
//!           ENCODE_MB=500 cargo bench --bench encode_doc
//!           TOKENIZER_JSON=data/qwen3_5_tokenizer.json cargo bench --bench encode_doc

use rs_gigatoken::load_tokenizer::hf::load_hf_bpe;
use rs_gigatoken::load_tokenizer::hub;
use rs_gigatoken::{WorkerPool, encode_docs_ragged};
use std::hint::black_box;
use std::path::PathBuf;
use std::time::Instant;

mod common;

const DEFAULT_MB: usize = 2000;

fn main() {
    common::allow_thp();
    let tokenizer_json =
        std::env::var("TOKENIZER_JSON").unwrap_or_else(|_| "allenai/OLMo-3-1025-7B".to_string());

    let tokenizer_path = {
        let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(&tokenizer_json);

        if path.exists() {
            eprintln!("Loading tokenizer from {path:?}...");
            path
        } else if hub::looks_like_repo_id(&tokenizer_json) {
            eprintln!(
                "Loading tokenizer.json from HuggingFace Hub repository {tokenizer_json:?}..."
            );

            hub::hub_file(&tokenizer_json, "tokenizer.json", "main")
                .expect("Could not fetch tokenizer.json from the HuggingFace Hub")
        } else {
            eprintln!("Skipping encode_doc benchmark: tokenizer not found at {path:?}.");
            eprintln!(
                "Set TOKENIZER_JSON to a local tokenizer.json or a HuggingFace repository ID."
            );
            return;
        }
    };

    let tokenizer = match load_hf_bpe(&tokenizer_path) {
        Ok(tokenizer) => tokenizer,
        Err(err) => {
            eprintln!("Skipping encode_doc benchmark: could not load tokenizer:");
            eprintln!("{err}");
            return;
        }
    };

    if !common::has_owt() {
        eprintln!("Skipping benchmark: OpenWebText dataset not found at ~/data/owt_train.txt.");
        eprintln!("Install ~/data/owt_train.txt to run this benchmark.");
        return;
    }

    let input = common::load_owt_input(Some(DEFAULT_MB));
    let size_mb = input.len() as f64 / 1e6;
    eprintln!("1 document, {} threads\n", rayon::current_num_threads());

    let rounds: usize = std::env::var("ENCODE_ROUNDS")
        .ok()
        .and_then(|v| v.trim().parse().ok())
        .unwrap_or(1);
    for round in 0..rounds {
        // Fresh worker pool per round so extra rounds at least start with
        // cold pretoken caches (the pool retains one forked tokenizer per
        // rayon thread).
        let workers = WorkerPool::new();
        let t0 = Instant::now();
        let (flat, lens) = encode_docs_ragged(&workers, &tokenizer, &[&input]);
        black_box((&flat, lens));
        let elapsed = t0.elapsed().as_secs_f64();
        eprintln!(
            "round {round}: {} tokens in {elapsed:.3}s — {:.0} MB/s",
            flat.len(),
            size_mb / elapsed
        );
    }
}
