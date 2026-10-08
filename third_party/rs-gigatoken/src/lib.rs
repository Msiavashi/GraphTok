#![feature(portable_simd)]

//! `rs-gigatoken` — high-throughput, Rust-native tokenization.
//!
//! The crate is a native Rust tokenizer engine with no foreign-language runtime
//! or compatibility adapters. Tokenizers, model loading, batching, file
//! ingestion, and BPE training are exposed directly as Rust APIs.

#[cfg(test)]
pub(crate) mod test_hub;

pub mod api;
pub mod batch;
pub mod bpe;
pub mod bpe_train;
pub mod input;
pub mod load_tokenizer;
pub mod pretokenize;
pub mod token;

pub use crate::api::{
    EncodeOptions, PadTruncate, SubstringMatcher, pad_truncate_ragged, wrap_truncate,
};
pub use crate::batch::{
    WorkerPool, encode_docs_padded, encode_docs_padded_serial, encode_docs_ragged,
    encode_docs_ragged_serial, encode_files_ragged, encode_files_ragged_serial,
    sp_encode_docs_padded, sp_encode_docs_ragged,
};
pub use crate::bpe::sentencepiece::{EncodeState, SentencePieceBPE};
pub use crate::bpe::tiktoken::Tokenizer;
pub use crate::bpe::tiktoken::Tokenizer as ByteLevelBpeTokenizer;
pub use crate::bpe_train::{BPEResult, TieBreaking};
pub use crate::input::file_source::{ContentFormat, DocFormat, FileSourceSpec};
pub use crate::load_tokenizer::hf::HfTokenizer;
pub use crate::load_tokenizer::sentencepiece::{
    load_sentencepiece_model, load_sentencepiece_slice,
};
pub use crate::pretokenize::PretokenizerType;
pub use crate::token::TokenId;
