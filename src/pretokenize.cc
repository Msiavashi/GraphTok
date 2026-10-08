// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 the cuTokenize contributors

#include "pretokenize.h"
#include "build_config.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace gbpe {

// =============================================================================
//  SP-family pre-tokenizer (Gemma 3).
// =============================================================================
#if GBPE_HAVE_FAMILY_SP

namespace {

// "▁" is U+2581, UTF-8 = E2 96 81. Three bytes.
constexpr uint8_t LOWBAR_B0 = 0xE2;
constexpr uint8_t LOWBAR_B1 = 0x96;
constexpr uint8_t LOWBAR_B2 = 0x81;

// Apply the Gemma 3 normalizer (" " -> "▁") over `text` into `norm`.
void normalize_replace_space(const std::string& text, std::vector<uint8_t>& norm) {
    norm.clear();
    norm.reserve(text.size() + text.size() / 4);  // ~25% growth worst case
    for (unsigned char c : text) {
        if (c == ' ') {
            norm.push_back(LOWBAR_B0);
            norm.push_back(LOWBAR_B1);
            norm.push_back(LOWBAR_B2);
        } else {
            norm.push_back(c);
        }
    }
}

// Walk one segment (raw UTF-8 bytes) and emit initial token ids: one id per
// Unicode codepoint, looked up by the codepoint's UTF-8 bytes. Codepoints not
// in vocab fall back to one byte_fallback id per UTF-8 byte. Matches the HF
// tokenizers BPE-with-byte-fallback initial-parts behavior. Appends ids to
// `out`. Returns number of ids appended.
size_t emit_initial_ids(const uint8_t* seg, size_t n,
                        const SPVocabInfo& vi,
                        std::vector<uint32_t>& out)
{
    const auto& map = *vi.id_by_token;
    size_t added = 0;
    size_t i = 0;
    std::string key;
    while (i < n) {
        unsigned char b0 = seg[i];
        size_t cp_len;
        if (b0 < 0x80)               cp_len = 1;
        else if ((b0 & 0xE0) == 0xC0) cp_len = 2;
        else if ((b0 & 0xF0) == 0xE0) cp_len = 3;
        else if ((b0 & 0xF8) == 0xF0) cp_len = 4;
        else                          cp_len = 1;  // malformed — treat as 1 byte
        if (i + cp_len > n) cp_len = n - i;        // truncated trailing bytes

        key.assign(reinterpret_cast<const char*>(seg + i), cp_len);
        auto it = map.find(key);
        if (it != map.end()) {
            out.push_back(it->second);
            ++added;
        } else {
            // Char not in vocab; expand to byte_fallback ids.
            for (size_t k = 0; k < cp_len; ++k) {
                out.push_back(vi.byte_fallback[seg[i + k]]);
                ++added;
            }
        }
        i += cp_len;
    }
    return added;
}

// Host-side BPE merge loop over an initial-id sequence using vi.host_merges.
// In-place: modifies `parts`.
void host_bpe(std::vector<uint32_t>& parts,
              const std::unordered_map<uint64_t, uint64_t>& mm)
{
    while (parts.size() >= 2) {
        uint32_t best_rank = 0xFFFFFFFFu;
        size_t best_i = 0;
        uint32_t best_new = 0;
        for (size_t i = 0; i + 1 < parts.size(); ++i) {
            uint64_t k = (static_cast<uint64_t>(parts[i]) << 32) | parts[i+1];
            auto it = mm.find(k);
            if (it != mm.end()) {
                uint32_t rank = static_cast<uint32_t>(it->second & 0xFFFFFFFFu);
                if (rank < best_rank) {
                    best_rank = rank;
                    best_i = i;
                    best_new = static_cast<uint32_t>(it->second >> 32);
                }
            }
        }
        if (best_rank == 0xFFFFFFFFu) break;
        parts[best_i] = best_new;
        parts.erase(parts.begin() + best_i + 1);
    }
}

}  // namespace

SPPreTokenizer::SPPreTokenizer(const SPVocabInfo& vi) : vi_(vi) {}

void SPPreTokenizer::apply(const std::string& text, SPResult& r) {
    r.n_short = 0;
    r.n_long = 0;
    r.n_overflow = 0;
    r.n_total = 0;

    // Step 1: normalize. (Replace " " -> "▁".)
    std::vector<uint8_t> norm;
    normalize_replace_space(text, norm);
    const size_t N = norm.size();

    // Step 2: the tokenizer's literal Split(" ", MergedWithPrevious) runs
    // after the Replace normalizer. There are therefore no literal spaces left
    // to match: the normalized input is a single segment. The previous code
    // split on U+2581, which is not what tokenizer.json specifies.
    const uint32_t seg_starts[2] = {0, static_cast<uint32_t>(N)};

    // Step 3: for each segment, emit initial ids and bucket.
    std::vector<uint32_t> ids_scratch;
    ids_scratch.reserve(MAX_PRETOKEN_LEN_LONG);
    for (size_t s = 0; s < 1; ++s) {
        uint32_t seg_start = seg_starts[s];
        uint32_t seg_end   = seg_starts[s + 1];
        if (seg_end <= seg_start) continue;
        const uint8_t* seg = norm.data() + seg_start;
        size_t seg_n = seg_end - seg_start;

        ids_scratch.clear();
        emit_initial_ids(seg, seg_n, vi_, ids_scratch);
        if (ids_scratch.empty()) continue;

        const uint32_t orig_idx = r.n_total;
        const size_t cnt = ids_scratch.size();

        if (cnt <= MAX_PRETOKEN_LEN_SHORT) {
            if (r.n_short >= r.cap_short) {
                std::fprintf(stderr,
                    "[pretok-sp] short bucket overflow at %u (cap=%u)\n",
                    r.n_short, r.cap_short);
                std::abort();
            }
            uint32_t* row = r.short_ids + static_cast<size_t>(r.n_short)
                            * MAX_PRETOKEN_LEN_SHORT;
            std::memcpy(row, ids_scratch.data(), cnt * sizeof(uint32_t));
            std::memset(row + cnt, 0,
                        (MAX_PRETOKEN_LEN_SHORT - cnt) * sizeof(uint32_t));
            r.short_lens[r.n_short]     = static_cast<uint16_t>(cnt);
            r.short_orig_idx[r.n_short] = orig_idx;
            ++r.n_short;
        } else if (cnt <= MAX_PRETOKEN_LEN_LONG) {
            if (r.n_long >= r.cap_long) {
                std::fprintf(stderr,
                    "[pretok-sp] long bucket overflow at %u (cap=%u)\n",
                    r.n_long, r.cap_long);
                std::abort();
            }
            uint32_t* row = r.long_ids + static_cast<size_t>(r.n_long)
                            * MAX_PRETOKEN_LEN_LONG;
            std::memcpy(row, ids_scratch.data(), cnt * sizeof(uint32_t));
            std::memset(row + cnt, 0,
                        (MAX_PRETOKEN_LEN_LONG - cnt) * sizeof(uint32_t));
            r.long_lens[r.n_long]     = static_cast<uint16_t>(cnt);
            r.long_orig_idx[r.n_long] = orig_idx;
            ++r.n_long;
        } else {
            // Overflow: host-BPE the whole sequence and stash result.
            if (r.n_overflow >= r.cap_overflow) {
                std::fprintf(stderr,
                    "[pretok-sp] overflow bucket full at %u (cap=%u)\n",
                    r.n_overflow, r.cap_overflow);
                std::abort();
            }
            std::vector<uint32_t> tmp = ids_scratch;  // copy
            host_bpe(tmp, *vi_.host_merges);
            if (tmp.size() > MAX_PRETOKEN_LEN_LONG) {
                std::fprintf(stderr,
                    "[pretok-sp] overflow pre-token BPE result %zu > LONG_LEN=%d, can't fit row stride\n",
                    tmp.size(), MAX_PRETOKEN_LEN_LONG);
                std::abort();
            }
            uint32_t* row = r.overflow_result_ids
                            + static_cast<size_t>(r.n_overflow)
                              * MAX_PRETOKEN_LEN_LONG;
            std::memcpy(row, tmp.data(), tmp.size() * sizeof(uint32_t));
            r.overflow_result_lens[r.n_overflow]
                = static_cast<uint16_t>(tmp.size());
            r.overflow_orig_idx[r.n_overflow] = orig_idx;
            ++r.n_overflow;
        }
        ++r.n_total;
    }
}

#endif  // GBPE_HAVE_FAMILY_SP

}  // namespace gbpe
