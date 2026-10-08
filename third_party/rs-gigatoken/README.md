# rs-gigatoken

High-throughput, CPU-optimized tokenization for Rust.

`rs-gigatoken` is the Rust-native distribution of the Gigatoken tokenizer
engine. It is designed for applications that need very high-throughput BPE
tokenization without a Python runtime or compatibility layer.

## Features

- Byte-level BPE / Tiktoken-style tokenization
- Hugging Face `tokenizer.json` loading for supported BPE models
- SentencePiece-style BPE with byte fallback
- GPT-2, GPT-4/cl100k, Qwen, OLMo, DeepSeek, O200k, Nemotron and Kimi
  pretokenization schemes
- Parallel document encoding with persistent worker-local caches
- Parallel/serial file encoding for text, JSONL and Parquet inputs
- File-oriented batch processing
- `.tiktoken` vocabulary loading
- BPE training
- Rust-native padding/truncation, wrapping and reusable substring matching
- CPU/SIMD optimized hot paths
- No Python, PyO3, NumPy, Maturin, Transformers or Tiktoken runtime
  dependencies

## Installation

Add the crate to your application:

```toml
[dependencies]
rs-gigatoken = "0.10"
```

The Rust import name is `rs_gigatoken`.

## Basic usage

Load a Hugging Face `tokenizer.json` and encode text:

```rust
use rs_gigatoken::load_tokenizer::hf::load_hf_bpe;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut tokenizer = load_hf_bpe("tokenizer.json")?;

    let mut tokens = Vec::new();
    tokenizer.memoized_encode(
        rs_gigatoken::pretokenize::pretokenize_as_iter(b"Hello, world!"),
        |ids| tokens.extend_from_slice(ids),
    );

    println!("{tokens:?}");
    Ok(())
}
```

For a tokenizer whose format is known, the concrete Rust tokenizer types are
also available directly:

```rust
use rs_gigatoken::Tokenizer;

let mut tokenizer = /* construct or load a Tokenizer */;
let mut ids = Vec::new();

tokenizer.memoized_encode(
    rs_gigatoken::pretokenize::pretokenize_as_iter(b"Hello, world!"),
    |tokens| ids.extend_from_slice(tokens),
);
```

## Parallel encoding

The batch engine returns one flat token buffer plus one length per document.
This avoids allocating a separate `Vec` for every document.

```rust
use rs_gigatoken::{
    WorkerPool,
    encode_docs_ragged,
    load_tokenizer::hf::load_hf_bpe,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let tokenizer = load_hf_bpe("tokenizer.json")?;
    let workers = WorkerPool::new();

    let documents = [
        b"first document".as_slice(),
        b"second document".as_slice(),
    ];

    let (tokens, lengths) =
        encode_docs_ragged(&workers, &tokenizer, &documents);

    println!("{} tokens across {:?}", tokens.len(), lengths);
    Ok(())
}
```

## File encoding and output assembly

The Rust API can encode paths directly while preserving the upstream low-copy
file strategy (mmap for plain files, decompression for `.gz`/`.zst`, and row-wise
Parquet handling):

```rust
use rs_gigatoken::{encode_files_ragged, DocFormat, WorkerPool};

let workers = WorkerPool::new();
let paths = vec!["data.txt".into()];
let format = DocFormat::Text { separator: None };
let (tokens, lengths) = encode_files_ragged(&workers, &tokenizer, &paths, &format)?;
```

For fixed-width model input, `EncodeOptions` (also available as the legacy
name `PadTruncate`) and `pad_truncate_ragged` assemble a flat row-major matrix
without allocating one vector per row. Prefix/suffix IDs count toward
`max_length`, and left/right padding and truncation are supported. The batch
engine also exposes `encode_docs_padded` and its serial counterpart.

## Loading Hugging Face tokenizers

Supported BPE tokenizer JSON files can be loaded directly:

```rust
use rs_gigatoken::load_tokenizer::hf::{load_hf_bpe, load_hf_slice};

let tokenizer = load_hf_bpe("tokenizer.json")?;
```

Raw SentencePiece BPE `.model` files with byte fallback can also be loaded
directly. The protobuf is parsed only during model loading; the encoding path
continues to use the same optimized Rust SentencePiece backend:

```rust
use rs_gigatoken::load_sentencepiece_model;

let tokenizer = load_sentencepiece_model("tokenizer.model")?;
```

Only SentencePiece BPE models with byte fallback are accepted. Unigram, Word,
and other SentencePiece model types are rejected explicitly rather than being
misinterpreted as BPE.

`load_hf_slice` returns an `HfTokenizer`, allowing callers to dispatch between
byte-level BPE and SentencePiece-style BPE:

```rust
use rs_gigatoken::load_tokenizer::hf::load_hf_slice;

let data = std::fs::read("tokenizer.json")?;

match load_hf_slice(&data)? {
    rs_gigatoken::HfTokenizer::Bpe(tokenizer) => {
        // byte-level BPE
        println!("{}", tokenizer.vocab_size());
    }
    rs_gigatoken::HfTokenizer::SentencePiece(tokenizer) => {
        // SentencePiece-style BPE
        println!("{}", tokenizer.vocab_size());
    }
}
```

## Hugging Face Hub

The crate includes a small Rust-native Hub cache/downloader used by the
benchmark and loader tooling. It does not require `huggingface_hub`,
Transformers, Python, or another runtime.

```rust
use rs_gigatoken::load_tokenizer::hub;

let path = hub::hub_file(
    "openai-community/gpt2",
    "tokenizer.json",
    "main",
)?;
```

## BPE training

The BPE training engine is available directly through
`rs_gigatoken::bpe_train`.

## Architecture

```text
Rust application
      │
      ▼
rs-gigatoken
      │
      ├── tokenizer loaders
      ├── pretokenizers
      ├── BPE engine
      ├── SentencePiece BPE
      ├── parallel batch engine
      ├── file ingestion
      └── BPE trainer
```

There is deliberately no Python translation layer in this repository.

## Performance

The engine is optimized for large-scale CPU tokenization workloads. See the
benchmark programs under `benches/` for reproducible throughput experiments.

Benchmark results in the upstream project are hardware- and workload-specific;
measure on the CPU and tokenizer configuration relevant to your application.

## License

MIT. See `LICENSE`.
