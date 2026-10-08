//! Whole-file parallel encode benchmark. For both backends the entire input
//! is ONE document handed to the library's parallel encode path
//! (`encode_docs_ragged` / `sp_encode_docs_ragged`) — the same chunking
//! policy, safe-boundary document splitting, and (for BPE) persistent worker
//! pool as `encode_batch` / `encode_files` — so this measures gigatoken's
//! own parallelism, not a bench-local split.
//!
//! Run with: cargo bench --bench encode                 (full OWT)
//!           ENCODE_MB=500 cargo bench --bench encode
//!           TOKENIZER_JSON=data/qwen3_5_tokenizer.json cargo bench --bench encode
//!           TOKENIZER_JSON=meta-llama/Llama-3.1-8B cargo bench --bench encode
//!             (a HuggingFace repo id: resolved via the standard HF cache,
//!              downloaded into it on a miss)

use rs_gigatoken::load_tokenizer::hf::{HfTokenizer, load_hf_slice};
use rs_gigatoken::load_tokenizer::hub;
use rs_gigatoken::{WorkerPool, encode_docs_ragged, sp_encode_docs_ragged};
use std::hint::black_box;
use std::path::PathBuf;
use std::time::Instant;

mod common;

fn main() {
    common::allow_thp();
    let tokenizer_json =
        std::env::var("TOKENIZER_JSON").unwrap_or_else(|_| "openai-community/gpt2".to_string());

    let data = if hub::looks_like_repo_id(&tokenizer_json) {
        eprintln!("Loading tokenizer.json from HuggingFace Hub repository {tokenizer_json:?}...");

        let tokenizer_path = hub::hub_file(
            &tokenizer_json,
            "tokenizer.json",
            "main",
        )
        .unwrap_or_else(|err| {
            panic!(
                "Could not fetch tokenizer.json for HuggingFace repository                  {tokenizer_json:?}: {err}"
            )
        });

        std::fs::read(&tokenizer_path).unwrap_or_else(|err| {
            panic!("Could not read downloaded tokenizer.json at {tokenizer_path:?}: {err}")
        })
    } else {
        let tokenizer_path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(&tokenizer_json);

        eprintln!("Loading tokenizer from {tokenizer_path:?}...");

        std::fs::read(&tokenizer_path).unwrap_or_else(|err| {
            panic!("Could not read tokenizer.json at {tokenizer_path:?}: {err}")
        })
    };
    let tokenizer = load_hf_slice(&data).expect("Could not load tokenizer");

    if !common::has_owt() {
        eprintln!("Skipping benchmark: OpenWebText dataset not found at ~/data/owt_train.txt.");
        eprintln!("Install ~/data/owt_train.txt to run this benchmark.");
        return;
    }
    let input = common::load_owt_input(None);
    let size_gb = input.len() as f64 / 1e9;

    let workers = WorkerPool::new();
    let (n_tokens, elapsed) = match &tokenizer {
        HfTokenizer::Bpe(tokenizer) => {
            eprintln!(
                "Encoding (ByteLevel BPE, 1 document, {} threads)...",
                rayon::current_num_threads()
            );
            let start = Instant::now();
            let (ids, lens) = encode_docs_ragged(&workers, tokenizer, &[&input]);
            black_box((&ids, lens));
            (ids.len(), start.elapsed().as_secs_f64())
        }
        HfTokenizer::SentencePiece(tokenizer) => {
            // Trailing partial UTF-8 char (from the ENCODE_MB byte cap) is
            // dropped; SP encoding takes str input.
            let text = match std::str::from_utf8(&input) {
                Ok(text) => text,
                Err(e) => std::str::from_utf8(&input[..e.valid_up_to()]).unwrap(),
            };
            eprintln!(
                "Encoding (SentencePiece, 1 document, {} threads)...",
                rayon::current_num_threads()
            );
            let start = Instant::now();
            let (ids, lens) = sp_encode_docs_ragged(tokenizer, &[text]);
            black_box((&ids, lens));
            (ids.len(), start.elapsed().as_secs_f64())
        }
    };
    let throughput_gb = size_gb / elapsed;

    eprintln!(
        "{n_tokens} tokens in {elapsed:.2}s — {throughput_gb:.2} GB/s ({:.0} MB/s)",
        throughput_gb * 1000.0
    );
}
