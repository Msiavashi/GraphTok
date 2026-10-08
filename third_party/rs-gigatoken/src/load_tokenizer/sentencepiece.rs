//! Loader for raw SentencePiece `.model` protobuf files.
//!
//! The tokenizer engine intentionally supports the same subset as the
//! upstream Rust backend: SentencePiece **BPE** models with byte fallback.
//! This module parses only the protobuf fields needed to reconstruct an
//! equivalent `tokenizer.json`, then reuses the existing HF loader. Keeping
//! the conversion at load time avoids adding protobuf machinery to any hot
//! encoding path.

use base64::Engine;
use eyre::{Result, ensure, eyre};
use serde::Serialize;
use std::collections::HashMap;
use std::path::Path;

const MODEL_PIECES: u64 = 1;
const MODEL_TRAINER_SPEC: u64 = 2;
const MODEL_NORMALIZER_SPEC: u64 = 3;

const PIECE_PIECE: u64 = 1;
const PIECE_SCORE: u64 = 2;
const PIECE_TYPE: u64 = 3;

const TRAINER_MODEL_TYPE: u64 = 3;
const TRAINER_TREAT_WHITESPACE_AS_SUFFIX: u64 = 24;
const TRAINER_BYTE_FALLBACK: u64 = 35;
const TRAINER_UNK_PIECE: u64 = 45;

const NORM_PRECOMPILED_CHARSMAP: u64 = 2;
const NORM_ADD_DUMMY_PREFIX: u64 = 3;
const NORM_REMOVE_EXTRA_WHITESPACES: u64 = 4;
const NORM_ESCAPE_WHITESPACES: u64 = 5;

const TYPE_NORMAL: u64 = 1;
const TYPE_CONTROL: u64 = 3;
const TYPE_USER_DEFINED: u64 = 4;

#[derive(Debug, Clone)]
enum ProtoValue {
    Varint(u64),
    Bytes(Vec<u8>),
    Fixed32([u8; 4]),
    Fixed64([u8; 8]),
}

fn read_varint(data: &[u8], pos: &mut usize) -> Result<u64> {
    let mut value = 0u64;
    let mut shift = 0u32;
    loop {
        ensure!(*pos < data.len(), "truncated SentencePiece protobuf varint");
        let byte = data[*pos];
        *pos += 1;
        value |= u64::from(byte & 0x7f) << shift;
        if byte & 0x80 == 0 {
            return Ok(value);
        }
        shift += 7;
        ensure!(shift < 64, "invalid SentencePiece protobuf varint");
    }
}

fn fields(data: &[u8]) -> Result<Vec<(u64, ProtoValue)>> {
    let mut out = Vec::new();
    let mut pos = 0usize;
    while pos < data.len() {
        let key = read_varint(data, &mut pos)?;
        let field = key >> 3;
        let wire = key & 7;
        ensure!(field != 0, "invalid SentencePiece protobuf field 0");
        let value = match wire {
            0 => ProtoValue::Varint(read_varint(data, &mut pos)?),
            1 => {
                ensure!(
                    data.len() - pos >= 8,
                    "truncated SentencePiece fixed64 field"
                );
                let mut bytes = [0u8; 8];
                bytes.copy_from_slice(&data[pos..pos + 8]);
                pos += 8;
                ProtoValue::Fixed64(bytes)
            }
            2 => {
                let len = read_varint(data, &mut pos)?;
                let len = usize::try_from(len)
                    .map_err(|_| eyre!("SentencePiece protobuf field is too large"))?;
                ensure!(
                    len <= data.len() - pos,
                    "truncated SentencePiece bytes field"
                );
                let bytes = data[pos..pos + len].to_vec();
                pos += len;
                ProtoValue::Bytes(bytes)
            }
            5 => {
                ensure!(
                    data.len() - pos >= 4,
                    "truncated SentencePiece fixed32 field"
                );
                let mut bytes = [0u8; 4];
                bytes.copy_from_slice(&data[pos..pos + 4]);
                pos += 4;
                ProtoValue::Fixed32(bytes)
            }
            other => {
                return Err(eyre!(
                    "unsupported SentencePiece protobuf wire type {other}"
                ));
            }
        };
        out.push((field, value));
    }
    Ok(out)
}

fn scalar_map(data: &[u8]) -> Result<HashMap<u64, ProtoValue>> {
    let mut map = HashMap::new();
    for (field, value) in fields(data)? {
        map.insert(field, value);
    }
    Ok(map)
}

fn varint(map: &HashMap<u64, ProtoValue>, field: u64, default: u64) -> u64 {
    match map.get(&field) {
        Some(ProtoValue::Varint(value)) => *value,
        _ => default,
    }
}

fn bytes(map: &HashMap<u64, ProtoValue>, field: u64) -> Vec<u8> {
    match map.get(&field) {
        Some(ProtoValue::Bytes(value)) => value.clone(),
        _ => Vec::new(),
    }
}

fn fixed32_f32(map: &HashMap<u64, ProtoValue>, field: u64) -> f32 {
    match map.get(&field) {
        Some(ProtoValue::Fixed32(value)) => f32::from_le_bytes(*value),
        _ => 0.0,
    }
}

#[derive(Debug, Clone)]
struct Piece {
    content: String,
    score: f32,
    kind: u64,
}

#[derive(Serialize)]
struct TokenizerJsonOut {
    version: &'static str,
    truncation: Option<()>,
    padding: Option<()>,
    added_tokens: Vec<AddedTokenOut>,
    normalizer: NormalizerOut,
    pre_tokenizer: Option<()>,
    post_processor: Option<()>,
    decoder: Option<()>,
    model: ModelOut,
}

#[derive(Serialize)]
struct AddedTokenOut {
    id: usize,
    content: String,
    single_word: bool,
    lstrip: bool,
    rstrip: bool,
    normalized: bool,
    special: bool,
}

#[derive(Serialize)]
struct NormalizerOut {
    #[serde(rename = "type")]
    kind: &'static str,
    normalizers: Vec<NormalizerStepOut>,
}

#[derive(Serialize)]
#[serde(tag = "type")]
enum NormalizerStepOut {
    Precompiled {
        precompiled_charsmap: String,
    },
    Strip {
        strip_left: bool,
        strip_right: bool,
    },
    Replace {
        pattern: PatternOut,
        content: &'static str,
    },
    Prepend {
        prepend: &'static str,
    },
}

#[derive(Serialize)]
enum PatternOut {
    #[serde(rename = "Regex")]
    Regex(String),
    #[serde(rename = "String")]
    String(String),
}

#[derive(Serialize)]
struct ModelOut {
    #[serde(rename = "type")]
    kind: &'static str,
    dropout: Option<()>,
    unk_token: String,
    continuing_subword_prefix: Option<()>,
    end_of_word_suffix: Option<()>,
    fuse_unk: bool,
    byte_fallback: bool,
    ignore_merges: bool,
    vocab: HashMap<String, usize>,
    merges: Vec<[String; 2]>,
}

/// Convert a raw SentencePiece BPE model to tokenizer.json bytes.
///
/// This is a load-time operation. The generated JSON is immediately consumed
/// by the existing native HF loader and is never used during encoding.
pub fn sentencepiece_to_tokenizer_json(data: &[u8]) -> Result<Vec<u8>> {
    let mut pieces = Vec::new();
    let mut trainer = HashMap::new();
    let mut normalizer = HashMap::new();

    for (field, value) in fields(data)? {
        match field {
            MODEL_PIECES => {
                let ProtoValue::Bytes(piece_data) = value else {
                    return Err(eyre!("SentencePiece piece field has invalid wire type"));
                };
                let piece = scalar_map(&piece_data)?;
                let content = bytes(&piece, PIECE_PIECE);
                let content = String::from_utf8(content)
                    .map_err(|_| eyre!("SentencePiece model contains a non-UTF-8 piece"))?;
                pieces.push(Piece {
                    content,
                    score: fixed32_f32(&piece, PIECE_SCORE),
                    kind: varint(&piece, PIECE_TYPE, TYPE_NORMAL),
                });
            }
            MODEL_TRAINER_SPEC => {
                let ProtoValue::Bytes(spec) = value else {
                    return Err(eyre!("SentencePiece trainer_spec has invalid wire type"));
                };
                trainer = scalar_map(&spec)?;
            }
            MODEL_NORMALIZER_SPEC => {
                let ProtoValue::Bytes(spec) = value else {
                    return Err(eyre!("SentencePiece normalizer_spec has invalid wire type"));
                };
                normalizer = scalar_map(&spec)?;
            }
            _ => {}
        }
    }

    ensure!(
        !pieces.is_empty(),
        "not a SentencePiece model: no pieces found"
    );

    let model_type = varint(&trainer, TRAINER_MODEL_TYPE, 1);
    ensure!(
        model_type == 2,
        "only BPE SentencePiece models are supported, got model_type {model_type}"
    );
    ensure!(
        varint(&trainer, TRAINER_BYTE_FALLBACK, 0) != 0,
        "only byte_fallback SentencePiece models are supported"
    );
    ensure!(
        varint(&normalizer, NORM_ESCAPE_WHITESPACES, 1) != 0,
        "SentencePiece models with escape_whitespaces=false are not supported"
    );
    ensure!(
        varint(&trainer, TRAINER_TREAT_WHITESPACE_AS_SUFFIX, 0) == 0,
        "SentencePiece models with treat_whitespace_as_suffix are not supported"
    );

    let mut vocab = HashMap::with_capacity(pieces.len());
    let mut scores = HashMap::with_capacity(pieces.len());
    for (id, piece) in pieces.iter().enumerate() {
        vocab.insert(piece.content.clone(), id);
        scores.insert(piece.content.clone(), piece.score);
    }

    let mut merge_candidates = Vec::<(String, String, f32)>::new();
    for (merged, &score) in &scores {
        let mut boundaries = merged.char_indices().map(|(i, _)| i).collect::<Vec<_>>();
        boundaries.push(merged.len());
        for pair in boundaries.windows(2).skip(1).map(|w| (w[0], w[1])) {
            let split = pair.0;
            let left = &merged[..split];
            let right = &merged[split..];
            if vocab.contains_key(left) && vocab.contains_key(right) {
                merge_candidates.push((left.to_string(), right.to_string(), score));
            }
        }
    }
    merge_candidates.sort_by(|a, b| {
        b.2.total_cmp(&a.2)
            .then_with(|| b.0.len().cmp(&a.0.len()))
            .then_with(|| b.1.len().cmp(&a.1.len()))
    });
    let merges = merge_candidates
        .into_iter()
        .map(|(left, right, _)| [left, right])
        .collect();

    let mut normalizers = Vec::new();
    let charsmap = bytes(&normalizer, NORM_PRECOMPILED_CHARSMAP);
    if !charsmap.is_empty() {
        normalizers.push(NormalizerStepOut::Precompiled {
            precompiled_charsmap: base64::engine::general_purpose::STANDARD.encode(charsmap),
        });
    }
    if varint(&normalizer, NORM_REMOVE_EXTRA_WHITESPACES, 1) != 0 {
        normalizers.push(NormalizerStepOut::Strip {
            strip_left: true,
            strip_right: true,
        });
        normalizers.push(NormalizerStepOut::Replace {
            pattern: PatternOut::Regex(" {2,}".to_string()),
            content: " ",
        });
    }
    if varint(&normalizer, NORM_ADD_DUMMY_PREFIX, 1) != 0 {
        normalizers.push(NormalizerStepOut::Prepend { prepend: "▁" });
    }
    normalizers.push(NormalizerStepOut::Replace {
        pattern: PatternOut::String(" ".to_string()),
        content: "▁",
    });

    let unk_piece = match trainer.get(&TRAINER_UNK_PIECE) {
        Some(ProtoValue::Bytes(bytes)) => String::from_utf8(bytes.clone())
            .map_err(|_| eyre!("SentencePiece unk_piece is not valid UTF-8"))?,
        _ => "<unk>".to_string(),
    };

    let added_tokens = pieces
        .iter()
        .enumerate()
        .filter(|(_, piece)| piece.kind == TYPE_CONTROL || piece.kind == TYPE_USER_DEFINED)
        .map(|(id, piece)| AddedTokenOut {
            id,
            content: piece.content.clone(),
            single_word: false,
            lstrip: false,
            rstrip: false,
            normalized: false,
            special: piece.kind == TYPE_CONTROL,
        })
        .collect();

    let output = TokenizerJsonOut {
        version: "1.0",
        truncation: None,
        padding: None,
        added_tokens,
        normalizer: NormalizerOut {
            kind: "Sequence",
            normalizers,
        },
        pre_tokenizer: None,
        post_processor: None,
        decoder: None,
        model: ModelOut {
            kind: "BPE",
            dropout: None,
            unk_token: unk_piece,
            continuing_subword_prefix: None,
            end_of_word_suffix: None,
            fuse_unk: true,
            byte_fallback: true,
            ignore_merges: false,
            vocab,
            merges,
        },
    };

    sonic_rs::to_vec(&output)
        .map_err(|e| eyre!("failed to serialize generated tokenizer JSON: {e}"))
}

/// Load a SentencePiece BPE `.model` from bytes.
pub fn load_sentencepiece_slice(
    data: &[u8],
) -> Result<crate::bpe::sentencepiece::SentencePieceBPE> {
    let json = sentencepiece_to_tokenizer_json(data)?;
    match super::hf::load_hf_slice(&json)? {
        super::hf::HfTokenizer::SentencePiece(tokenizer) => Ok(tokenizer),
        super::hf::HfTokenizer::Bpe(_) => Err(eyre!(
            "generated SentencePiece model did not select the SentencePiece backend"
        )),
    }
}

/// Load a SentencePiece BPE `.model` from a path.
pub fn load_sentencepiece_model(
    path: impl AsRef<Path>,
) -> Result<crate::bpe::sentencepiece::SentencePieceBPE> {
    let path = path.as_ref();
    let data = std::fs::read(path)
        .map_err(|e| eyre!("failed to read SentencePiece model {}: {e}", path.display()))?;
    load_sentencepiece_slice(&data)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn varint(mut value: u64, out: &mut Vec<u8>) {
        while value >= 0x80 {
            out.push((value as u8) | 0x80);
            value >>= 7;
        }
        out.push(value as u8);
    }

    fn field_bytes(field: u64, bytes: &[u8], out: &mut Vec<u8>) {
        varint((field << 3) | 2, out);
        varint(bytes.len() as u64, out);
        out.extend_from_slice(bytes);
    }

    fn field_varint(field: u64, value: u64, out: &mut Vec<u8>) {
        varint(field << 3, out);
        varint(value, out);
    }

    fn piece(text: &str, score: f32) -> Vec<u8> {
        let mut out = Vec::new();
        field_bytes(PIECE_PIECE, text.as_bytes(), &mut out);
        varint((PIECE_SCORE << 3) | 5, &mut out);
        out.extend_from_slice(&score.to_le_bytes());
        field_varint(PIECE_TYPE, TYPE_NORMAL, &mut out);
        out
    }

    #[test]
    fn parses_minimal_bpe_model() {
        let mut trainer = Vec::new();
        field_varint(TRAINER_MODEL_TYPE, 2, &mut trainer);
        field_varint(TRAINER_BYTE_FALLBACK, 1, &mut trainer);

        let mut normalizer = Vec::new();
        field_varint(NORM_ADD_DUMMY_PREFIX, 1, &mut normalizer);
        field_varint(NORM_REMOVE_EXTRA_WHITESPACES, 1, &mut normalizer);
        field_varint(NORM_ESCAPE_WHITESPACES, 1, &mut normalizer);

        let mut model = Vec::new();
        for (text, score) in [("<unk>", 0.0), ("a", 1.0), ("b", 1.0), ("ab", 2.0)] {
            field_bytes(MODEL_PIECES, &piece(text, score), &mut model);
        }
        field_bytes(MODEL_TRAINER_SPEC, &trainer, &mut model);
        field_bytes(MODEL_NORMALIZER_SPEC, &normalizer, &mut model);

        let json = sentencepiece_to_tokenizer_json(&model).unwrap();
        let text = String::from_utf8(json).unwrap();
        assert!(text.contains("\"byte_fallback\":true"));
        assert!(text.contains("\"ab\""));
        assert!(text.contains("\"merges\":[[\"a\",\"b\"]]"));
    }

    #[test]
    fn rejects_unigram_model() {
        let mut trainer = Vec::new();
        field_varint(TRAINER_MODEL_TYPE, 1, &mut trainer);
        field_varint(TRAINER_BYTE_FALLBACK, 1, &mut trainer);
        let mut model = Vec::new();
        field_bytes(MODEL_TRAINER_SPEC, &trainer, &mut model);
        field_bytes(MODEL_PIECES, &piece("<unk>", 0.0), &mut model);

        let error = sentencepiece_to_tokenizer_json(&model)
            .unwrap_err()
            .to_string();
        assert!(error.contains("only BPE SentencePiece models are supported"));
    }
}
