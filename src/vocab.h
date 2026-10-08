// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 the GraphTok contributors

#pragma once
#include "tokenizer.cuh"
#include "pretokenize.h"
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>
#include <unordered_map>

namespace gbpe {

enum class NormalizerKind : uint8_t {
    None = 0,
    NFC  = 1,
};

// Loaded representation of the HF tokenizer.json (model.type == "BPE").
// Strings are the GPT-2 byte-level-encoded forms (e.g. "Ġthe", "Ġand").
struct HostVocab {
    // tokens_by_id[id] -> bytes (in the GPT-2 byte-level-encoded alphabet,
    // i.e. each *char* of the string maps to one input byte via the
    // byte-to-unicode table). Length is decoded back to raw bytes by
    // gpt2_token_decoded_bytes(id).
    std::vector<std::string> tokens_by_id;          // encoded (display) form
    std::vector<std::vector<uint8_t>> token_bytes;  // decoded raw bytes (post byte-level)
    std::unordered_map<std::string, uint32_t> id_by_token;
    uint32_t model_vocab_size = 0;  // excludes added-only decoder entries

    // Per-vocab byte->id table (256 entries). For GPT-2 the *initial* alphabet
    // is exactly the 256 single-byte tokens "!" .. "Ā" .. "ÿ" encoded via the
    // byte-level table; this is the lookup we use to seed the pre-token's
    // initial parts[].
    std::vector<uint32_t> byte_to_id;  // size 256

    // Merge rules in rank order: merges[r] = (left_id, right_id, new_id).
    struct Merge {
        uint32_t left_id;
        uint32_t right_id;
        uint32_t new_id;
    };
    std::vector<Merge> merges;

    // Auto-detected pre-tokenizer regex kind. The loader inspects the
    // tokenizer.json pre_tokenizer field and selects one of the known
    // RegexKind variants.
    RegexKind regex_kind = RegexKind::GPT2;

    // HF normalizer applied before pre-tokenization. Byte-level Qwen-2.5 uses
    // NFC; ignoring it is not bit-exact on decomposed multilingual text.
    NormalizerKind normalizer_kind = NormalizerKind::None;

    // HF BPE `ignore_merges`: when true, a complete pre-token already present
    // in the model vocabulary is emitted directly before ordinary greedy BPE.
    // Llama-3 enables this and it is observable on multilingual text.
    bool ignore_merges = false;

    // ---- F4: Special-token insertion (BOS/EOS/chat-template) ----
    struct SpecialToken {
        uint32_t    id;
        std::string content;
        bool        special = true;
        bool        normalized = false;
    };
    std::vector<SpecialToken> added_tokens;
    std::unordered_map<std::string, uint32_t> special_id_by_content;
    std::vector<uint32_t> bos_ids;
    std::vector<uint32_t> eos_ids;

    // ---- F1b: SP-family extras (Gemma 3). Populated when model.byte_fallback == true.
    bool byte_fallback = false;
    std::string unk_token;
    std::vector<uint32_t> byte_fallback_id;     // size 256 (SP) or 0 (other)
    // Reverse lookup for the decode post-processor: indexed by token ID,
    // -1 for ordinary tokens and 0..255 for a <0xHH> fallback token.
    std::vector<int16_t> byte_fallback_byte_by_id;
    uint32_t max_token_byte_len = 0;
    std::unordered_map<uint64_t, uint64_t> host_merges;
};

// Encode-time options applied to the bare BPE ids produced by the kernel.
struct EncodeOptions {
    // Prepend hv.bos_ids (mirrors HF `add_special_tokens=True` BOS behavior).
    bool add_bos = false;
    // Append hv.eos_ids when populated; otherwise append a sensible canonical
    // EOS from added_tokens if one can be identified (Llama-3: <|end_of_text|>,
    // Qwen-2.5: <|endoftext|>, DeepSeek-V3: <｜end▁of▁sentence｜>, GPT-2:
    // <|endoftext|>). No-op when neither source applies.
    bool add_eos = false;
};

// Apply the requested specials to a bare-encoder output in-place (prepend/
// append). Pure host function; does not touch the GPU.
void apply_encode_options(const HostVocab& hv,
                          const EncodeOptions& opts,
                          std::vector<uint32_t>& ids);

// Load HF-style tokenizer.json from disk. Returns false on parse failure.
bool load_hf_tokenizer_json(const std::string& path, HostVocab& out);

// Apply the tokenizer.json normalizer to raw input text. Returns `text` by
// value to keep call sites simple; normalizer_kind==None returns a copy.
std::string normalize_for_vocab(const HostVocab& hv, const std::string& text);

// True when normalize_for_vocab(hv, text) would return `text` unchanged
// (no normalizer, or text already in NFC). Lets callers skip the copy.
bool is_normalized_for_vocab(const HostVocab& hv, std::string_view text);

// normalize_for_vocab without the copies: returns false, leaving `out`
// untouched, when `text` is already normalized (use `text` as is); otherwise
// writes the normalized text to `out` and returns true. Scans an unchanged
// text once, and copies the prefix before the first change without
// rescanning it.
bool normalize_view_for_vocab(const HostVocab& hv, std::string_view text, std::string& out);

// Apply the serialized SP decoder after TokenizerCtx::decode() has gathered
// raw model-token spellings.  For byte-fallback vocabularies this performs
// Replace(U+2581 -> " ") followed by ByteFallback, using token IDs rather
// than scanning concatenated text so markers cannot be recognized across a
// token boundary.  The result may contain arbitrary bytes; callers choosing a
// Unicode string must apply the reference decoder's UTF-8 replacement policy.
void postprocess_decoded_bytes(const HostVocab& hv,
                               const uint32_t* token_ids,
                               uint32_t n_tokens,
                               const std::vector<uint8_t>& raw_bytes,
                               std::vector<uint8_t>& decoded_bytes);

// Same serialized SP decoder, expressed as valid UTF-8 for a text API. A
// malformed contiguous ByteFallback run becomes one U+FFFD per fallback byte,
// matching Hugging Face tokenizers' ByteFallback + Fuse behavior.
void postprocess_decoded_utf8(const HostVocab& hv,
                              const uint32_t* token_ids,
                              uint32_t n_tokens,
                              const std::vector<uint8_t>& raw_bytes,
                              std::vector<uint8_t>& decoded_utf8);

// Build a packed device-side VocabPack from the host vocab. Allocates GPU
// memory via cudaMalloc; caller owns and must free with free_vocab_pack().
VocabPack build_vocab_pack(const HostVocab& hv);
void free_vocab_pack(VocabPack& vp);

// Byte<->unicode-char table used by the GPT-2 byte-level encoder. Returns
// a length-256 array where bytes_to_unicode[b] is the unicode codepoint
// (always 0x21..0x1FF). Used both at vocab load time to decode the
// tokenizer.json strings back to bytes, and at host-pretok time to encode
// raw input bytes into the GPT-2 representation.
const uint32_t* gpt2_byte_to_unicode();   // [256]
// Inverse: unicode codepoint (in 0x21..0x1FF) -> raw byte. Sparse, only the
// 256 mapped codepoints are valid.
const int32_t* gpt2_unicode_to_byte();    // [0x200], -1 if not a mapped codepoint

// Hash function used in the device merge table. Exposed so the host-side
// table builder uses the same hash.
#ifdef __CUDACC__
#  define GBPE_HD __host__ __device__
#else
#  define GBPE_HD
#endif

// ---- Slot64 path (GPT-2): both ids fit in 16 bits, rank fits in 16 bits.
//   slot = (left << 48) | (right << 32) | (new_id << 16) | rank
inline GBPE_HD uint32_t pack_key32(uint32_t l, uint32_t r) {
    return (l << 16) | (r & 0xFFFFu);
}
inline GBPE_HD uint64_t pack_slot64(uint32_t l, uint32_t r, uint32_t new_id, uint32_t rank) {
    uint64_t key = static_cast<uint64_t>(pack_key32(l, r)) << 32;
    uint64_t val = (static_cast<uint64_t>(new_id & 0xFFFFu) << 16) | (rank & 0xFFFFu);
    return key | val;
}
// Murmur3-style integer finalizer on the 32-bit key.
inline GBPE_HD uint32_t merge_hash_key32(uint32_t key, uint32_t mask) {
    uint32_t x = key;
    x ^= x >> 16;
    x *= 0x85ebca6bu;
    x ^= x >> 13;
    x *= 0xc2b2ae35u;
    x ^= x >> 16;
    return x & mask;
}

// ---- Slot128 path (Llama-3): keys 64-bit (left:32 | right:32), vals 64-bit
//      (new_id:32 | rank:32). Rank == 0xFFFFFFFFu marks empty.
inline GBPE_HD uint64_t pack_key64(uint32_t l, uint32_t r) {
    return (static_cast<uint64_t>(l) << 32) | static_cast<uint64_t>(r);
}
inline GBPE_HD uint64_t pack_val64(uint32_t new_id, uint32_t rank) {
    return (static_cast<uint64_t>(new_id) << 32) | static_cast<uint64_t>(rank);
}
// 64-bit Murmur3-style finalizer.
inline GBPE_HD uint32_t merge_hash_key64(uint64_t key, uint32_t mask) {
    uint64_t x = key;
    x ^= x >> 33;
    x *= 0xff51afd7ed558ccdULL;
    x ^= x >> 33;
    x *= 0xc4ceb9fe1a85ec53ULL;
    x ^= x >> 33;
    return static_cast<uint32_t>(x) & mask;
}

// Raw-byte hash for HF BPE model.ignore_merges. Every device lookup verifies
// the candidate bytes after probing, so a hash collision cannot affect output.
inline GBPE_HD uint64_t raw_token_hash64(const uint8_t* bytes, uint32_t len) {
    uint64_t h = 1469598103934665603ULL;
    for (uint32_t i = 0; i < len; ++i) {
        h ^= static_cast<uint64_t>(bytes[i]);
        h *= 1099511628211ULL;
    }
    h ^= static_cast<uint64_t>(len);
    h *= 0x9E3779B185EBCA87ULL;
    return h;
}

inline GBPE_HD uint32_t raw_token_hash_slot(uint64_t hash, uint32_t mask) {
    return merge_hash_key64(hash, mask);
}

}  // namespace gbpe
