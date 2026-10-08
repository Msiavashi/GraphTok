// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 the GraphTok contributors

#pragma once
#include "build_config.h"
#include "tokenizer.cuh"
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>
#include <unordered_map>

namespace gbpe {

// The regex pattern set is auto-selected from the vocab. GPT-2, Llama-3, and
// Qwen2.5 all use a single regex; DeepSeek-V3 uses a sequence of three regexes
// applied in turn; Gemma 3 uses no regex at all (literal Split + Replace
// normalizer). Pre-tokenization now runs on the GPU (the raw_text path);
// RegexKind is retained only to classify a vocab and select the GPU boundary
// stage (see vocab.cc / detect_regex_kind) and the SP-family host path below.
enum class RegexKind { GPT2, Llama3, Qwen25, DeepSeekV3, Gemma3 };

#if GBPE_HAVE_FAMILY_SP

// SP-family pre-tokenizer + initial-tokenizer combined.
//
// Pipeline (Gemma 3):
//   1. Replace " " -> "▁" (U+2581) over the input.
//   2. Apply the literal Split(" ", MergedWithPrevious). Because normalization
//      has already replaced every literal space, the split has no matches and
//      the complete normalized input is one pre-token (added-token extraction,
//      when enabled by a higher-level API, creates separate spans before this).
//   3. Walk the span char-by-char and produce a sequence of initial token
//      ids by longest-prefix lookup against the vocab. Unmatched chars expand
//      to one byte_fallback id per UTF-8 byte (vocab token "<0xXX>").
//
// Output layout mirrors the byte-level pre-tokenizer's two-bucket scheme but
// stores uint32_t token ids (not raw bytes) and lens count IDS, not bytes.
// Pre-tokens whose initial-id count exceeds MAX_PRETOKEN_LEN_LONG go into the
// `overflow` list — the host BPE-encodes those directly and the GPU pipeline
// splices their results into the output via per_pre_count/per_pre_tokens.
struct SPResult {
    // SHORT bucket (count <= MAX_PRETOKEN_LEN_SHORT initial ids).
    uint32_t* short_ids;       // [cap_short * MAX_PRETOKEN_LEN_SHORT]
    uint16_t* short_lens;      // [cap_short] -- # of initial ids
    uint32_t* short_orig_idx;  // [cap_short]
    uint32_t  n_short;
    uint32_t  cap_short;
    // LONG bucket (MAX_PRETOKEN_LEN_SHORT < count <= MAX_PRETOKEN_LEN_LONG).
    uint32_t* long_ids;        // [cap_long * MAX_PRETOKEN_LEN_LONG]
    uint16_t* long_lens;       // [cap_long]
    uint32_t* long_orig_idx;   // [cap_long]
    uint32_t  n_long;
    uint32_t  cap_long;
    // OVERFLOW: pre-tokens with > MAX_PRETOKEN_LEN_LONG initial ids. The
    // pre-tokenizer fills overflow_* by appending the host-BPE'd RESULT ids
    // for each overflow pre-token; the GPU pipeline copies these directly
    // into the per_pre_tokens device rows at orig_idx.
    // Layout: overflow_orig_idx[i] gives the destination orig_idx;
    // overflow_result_ids[overflow_offsets[i] .. overflow_offsets[i+1]] gives
    // the post-BPE token ids (must be <= MAX_PRETOKEN_LEN_LONG entries).
    uint32_t* overflow_orig_idx;     // [cap_overflow]
    uint32_t* overflow_result_ids;   // [cap_overflow * MAX_PRETOKEN_LEN_LONG]
    uint16_t* overflow_result_lens;  // [cap_overflow] -- result count
    uint32_t  n_overflow;
    uint32_t  cap_overflow;

    uint32_t  n_total;  // n_short + n_long + n_overflow
};

// Vocab info needed by the SP pre-tokenizer to do longest-prefix init-token
// matching and byte-fallback. Built from HostVocab; see vocab.cc.
struct SPVocabInfo {
    // Map from raw byte sequence (UTF-8) -> token id. Keys are token strings
    // as they appear in tokenizer.json's vocab (raw UTF-8). Used for
    // longest-prefix lookup at initial-tokenize time.
    const std::unordered_map<std::string, uint32_t>* id_by_token = nullptr;
    // byte_fallback_id[b] = vocab id of "<0xXX>" token (for b = 0..255). Used
    // for any UTF-8 byte that doesn't match a vocab token at the current pos.
    const uint32_t* byte_fallback = nullptr;   // [256]
    // Maximum token byte length, for capping the longest-prefix probe.
    uint32_t max_token_byte_len = 0;
    // For host-side BPE of overflow pre-tokens. Keys = (left:32 | right:32);
    // values = (new_id:32 | rank:32). Same packing as the device Slot128.
    const std::unordered_map<uint64_t, uint64_t>* host_merges = nullptr;
};

class SPPreTokenizer {
public:
    explicit SPPreTokenizer(const SPVocabInfo& vi);

    // Apply pipeline to `text`. Caller pre-allocates buffers in `r`.
    void apply(const std::string& text, SPResult& r);

private:
    SPVocabInfo vi_;
};

#endif  // GBPE_HAVE_FAMILY_SP

// Round a required element count to a graph-stable uint32 power-of-two shape.
// Values above 2^31 cannot be represented as a power of two in uint32_t and
// must be rejected rather than wrapping an allocation/count to zero.
inline uint32_t round_graph_capacity(size_t required, uint32_t floor = 1024) {
    const size_t base = required < floor ? floor : required;
    if (base > 0x80000000ull) {
        throw std::length_error("cuTokenize: requested graph capacity exceeds 2^31");
    }
    uint32_t p = 1;
    while (p < base) p <<= 1;
    return p;
}

// Checked byte capacity used for input/output buffers that carry a small
// fixed slack. Keeping the addition in size_t avoids uint32 wraparound.
inline uint32_t checked_byte_capacity(size_t text_bytes, uint32_t slack = 1024) {
    if (text_bytes > static_cast<size_t>(std::numeric_limits<uint32_t>::max()) - slack) {
        throw std::length_error("cuTokenize: input is too large for uint32 offsets");
    }
    return static_cast<uint32_t>(text_bytes + slack);
}

// Every non-empty pre-token contains at least one input byte, so the total
// pre-token count is bounded by the byte count (Qwen can approach this bound).
inline uint32_t estimate_max_pretokens(size_t text_bytes) {
    return round_graph_capacity(text_bytes);
}

// A long-bucket pre-token contains at least 33 initial byte/ID elements. Thus
// ceil(input_elements/33) is a true bound on how many long-bucket rows can
// exist. The previous input/64 heuristic failed on repeated 33-byte tokens.
inline uint32_t estimate_max_long_pretokens(size_t input_elements,
                                            uint32_t floor = 1024) {
    const size_t count = input_elements == 0 ? 0 : (input_elements + 32u) / 33u;
    return round_graph_capacity(count, floor);
}

}  // namespace gbpe
