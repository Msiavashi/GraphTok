// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 the GraphTok contributors
//
// Host encoder for small byte-level inputs. A graph replay costs ~90 us of
// fixed latency regardless of input size, so below a few kilobytes a single
// CPU thread wins. This path reuses the GPU's boundary predicates
// (pretok_boundary.h) and builds its merge table with the same hash and
// insertion order as the device table, so its output is the same token
// sequence by construction. Gemma (SentencePiece) and DeepSeek-V3 (three-pass
// splitter) are not handled here and report supported() == false.
#pragma once
#include "vocab.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace gbpe {

// Pre-token cache counters. Present only in a build configured with
// -DGBPE_HOST_STATS=1; the shipped build compiles these away entirely.
#if defined(GBPE_HOST_STATS) && GBPE_HOST_STATS
struct HostCacheStats {
    unsigned long long short_lookups, short_hits, short_evictions;
    unsigned long long wide_lookups, wide_hits, wide_evictions;
    unsigned long long uncacheable;
};
HostCacheStats host_cache_stats();
void host_cache_stats_reset();
#endif

class HostEncoder {
public:
    explicit HostEncoder(const HostVocab& hv);
    ~HostEncoder();

    // False for tokenizer families this encoder does not implement; callers
    // must then use the GPU path.
    bool supported() const { return supported_; }

    // Bare encode (no BOS/EOS) of already-normalized UTF-8 bytes.
    void encode(const uint8_t* text, size_t len, std::vector<uint32_t>& out) const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
    bool supported_ = false;
};

}  // namespace gbpe
