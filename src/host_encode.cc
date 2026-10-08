// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 the GraphTok contributors
#include "host_encode.h"
#include "pretok_boundary.h"
#include "pretok_boundary_simd.h"
#include "gbpe_class_table.h"

#include <algorithm>
#include <cstring>

#if defined(__linux__)
#  include <sys/mman.h>
#  include <cstdlib>
#endif

// Stage cutoff for attribution builds. 0 (default) runs the whole encoder;
// 1 stops after the classify pass, 2 after the boundary scan, so the cost of
// each stage is a difference of medians between builds. A truncated build
// emits WRONG TOKENS by construction and exists only for measurement.
#ifndef GBPE_HOST_STOP_AFTER
#define GBPE_HOST_STOP_AFTER 0
#endif

namespace gbpe {

namespace hugepage {

// Allocator that backs large tables with transparent huge pages.
//
// The host encoder's random-access tables total ~50 MiB (26 MiB hashed cache,
// 8 MiB short cache, up to 16 MiB merge table). At 4 KiB pages that is ~12800
// pages against an L2 dTLB of order 1-2k entries, so a random probe misses the
// TLB more often than it misses the data cache, and every such miss is a page
// walk. 2 MiB pages cut the entry count by 512x.
//
// This is NOT the refuted pre-faulting experiment. That measured the cost of
// *taking* the page faults and correctly found nothing: faults are a one-off,
// while TLB reach is paid on every probe. It is also the natural follow-up to
// the A1 cache sizing, which found that growing a table past L2 costs nothing
// per probe -- precisely because the cost that does scale is TLB reach.
//
// THP on the reference host is `madvise`, so the advice is required (nothing
// happens implicitly) but no privilege is needed. Every step degrades safely:
// if posix_memalign or madvise fails, the table simply lives on 4 KiB pages.
constexpr size_t kHugePage = size_t{2} << 20;

inline void* allocate(size_t bytes) {
#if defined(__linux__) && defined(MADV_HUGEPAGE)
    if (bytes >= kHugePage) {
        // Round the length up as well: madvise promotes only fully covered
        // 2 MiB regions, so a partial tail would stay on 4 KiB pages.
        const size_t rounded = (bytes + kHugePage - 1) & ~(kHugePage - 1);
        void* p = nullptr;
        if (posix_memalign(&p, kHugePage, rounded) == 0 && p != nullptr) {
            (void)madvise(p, rounded, MADV_HUGEPAGE);   // advisory; may fail
            return p;
        }
    }
#endif
    return ::operator new(bytes);
}

inline void deallocate(void* p, size_t bytes) noexcept {
    if (p == nullptr) return;
#if defined(__linux__) && defined(MADV_HUGEPAGE)
    if (bytes >= kHugePage) { std::free(p); return; }
#endif
    ::operator delete(p);
}

}  // namespace hugepage

// std::vector-compatible allocator using the above. Stateless and always
// equal, so vectors using it move and swap normally.
template <typename T>
struct HugeAlloc {
    using value_type = T;
    HugeAlloc() noexcept = default;
    template <typename U> HugeAlloc(const HugeAlloc<U>&) noexcept {}
    T* allocate(size_t n) {
        return static_cast<T*>(hugepage::allocate(n * sizeof(T)));
    }
    void deallocate(T* p, size_t n) noexcept {
        hugepage::deallocate(p, n * sizeof(T));
    }
    template <typename U> bool operator==(const HugeAlloc<U>&) const noexcept { return true; }
    template <typename U> bool operator!=(const HugeAlloc<U>&) const noexcept { return false; }
};

template <typename T>
using HugeVector = std::vector<T, HugeAlloc<T>>;

// Pre-token cache counters, compiled out entirely unless
// -DGBPE_HOST_STATS=1 is passed. The shipped build must never carry
// instrumentation: a previous session left an instrumented .so in place and
// inflated every benchmark by 33%.
#if defined(GBPE_HOST_STATS) && GBPE_HOST_STATS
struct HostStats {
    uint64_t short_lookups = 0, short_hits = 0, short_evictions = 0;
    uint64_t wide_lookups = 0, wide_hits = 0, wide_evictions = 0;
    uint64_t uncacheable = 0;
};
HostStats g_host_stats;
#define GBPE_STAT(field) (++::gbpe::g_host_stats.field)
#define GBPE_STAT_IF(cond, field) do { if (cond) ++::gbpe::g_host_stats.field; } while (0)
#else
#define GBPE_STAT(field) ((void)0)
#define GBPE_STAT_IF(cond, field) ((void)0)
#endif

#if defined(GBPE_HOST_STATS) && GBPE_HOST_STATS
HostCacheStats host_cache_stats() {
    const HostStats& s = g_host_stats;
    return HostCacheStats{s.short_lookups, s.short_hits, s.short_evictions,
                          s.wide_lookups, s.wide_hits, s.wide_evictions,
                          s.uncacheable};
}
void host_cache_stats_reset() { g_host_stats = HostStats{}; }
#endif

namespace {

constexpr uint32_t kDead = 0xFFFFFFFFu;

// Merge table with the device layout: same pack/hash functions, same
// capacity, same insertion order, so a lookup here probes the same slots the
// kernel does. Slot64 for vocab/merges <= 64k (GPT-2), Slot128 otherwise.
struct MergeTable {
    bool slot64 = false;
    uint32_t mask = 0;
    // Huge-page backed: these are probed at random and are 1-16 MiB, so TLB
    // reach rather than cache residency is what scales. See HugeAlloc.
    HugeVector<uint64_t> slots64;
    HugeVector<uint64_t> keys128;
    HugeVector<uint64_t> vals128;

    void build(const HostVocab& hv) {
        uint32_t cap = 4;
        while (cap < hv.merges.size() * 2) cap <<= 1;
        mask = cap - 1;
        slot64 = hv.model_vocab_size <= 65535 && hv.merges.size() <= 65535;
        if (slot64) {
            constexpr uint64_t EMPTY_SLOT = 0x000000000000FFFFULL;
            slots64.assign(cap, EMPTY_SLOT);
            for (uint32_t rank = 0; rank < hv.merges.size(); ++rank) {
                const auto& m = hv.merges[rank];
                uint64_t slot = pack_slot64(m.left_id, m.right_id, m.new_id, rank);
                uint32_t idx = merge_hash_key32(pack_key32(m.left_id, m.right_id), mask);
                while ((slots64[idx] & 0xFFFFu) != 0xFFFFu) idx = (idx + 1) & mask;
                slots64[idx] = slot;
            }
        } else {
            keys128.assign(cap, 0);
            vals128.assign(cap, static_cast<uint64_t>(kDead));
            for (uint32_t rank = 0; rank < hv.merges.size(); ++rank) {
                const auto& m = hv.merges[rank];
                uint64_t k64 = pack_key64(m.left_id, m.right_id);
                uint32_t idx = merge_hash_key64(k64, mask);
                while ((vals128[idx] & 0xFFFFFFFFu) != kDead) idx = (idx + 1) & mask;
                keys128[idx] = k64;
                vals128[idx] = pack_val64(m.new_id, rank);
            }
        }
    }

    // Returns rank (kDead if no rule) and writes new_id.
    uint32_t probe(uint32_t left, uint32_t right, uint32_t* new_id) const {
        if (slot64) {
            const uint32_t want = pack_key32(left, right);
            uint32_t idx = merge_hash_key32(want, mask);
            for (int p = 0; p < 64; ++p) {
                const uint64_t s = slots64[idx];
                const uint32_t rank16 = static_cast<uint32_t>(s & 0xFFFFu);
                if (rank16 == 0xFFFFu) return kDead;
                if (static_cast<uint32_t>(s >> 32) == want) {
                    *new_id = static_cast<uint32_t>((s >> 16) & 0xFFFFu);
                    return rank16;
                }
                idx = (idx + 1) & mask;
            }
            return kDead;
        }
        const uint64_t want = pack_key64(left, right);
        uint32_t idx = merge_hash_key64(want, mask);
        for (int p = 0; p < 64; ++p) {
            const uint64_t v = vals128[idx];
            const uint32_t rank = static_cast<uint32_t>(v & 0xFFFFFFFFu);
            if (rank == kDead) return kDead;
            if (keys128[idx] == want) {
                *new_id = static_cast<uint32_t>(v >> 32);
                return rank;
            }
            idx = (idx + 1) & mask;
        }
        return kDead;
    }
};

// AddedVocabulary literal matcher. The loader already rejects assets where a
// literal contains another literal's first byte, so a left-to-right scan that
// keeps the longest terminal at each position reproduces the GPU matcher.
struct AddedTrie {
    struct Node {
        std::unordered_map<uint8_t, uint32_t> next;
        uint32_t token_id = kDead;
    };
    std::vector<Node> nodes;
    uint32_t root[256];
    uint32_t max_bytes = 0;

    void build(const HostVocab& hv) {
        for (auto& r : root) r = kDead;
        nodes.assign(1, Node{});
        for (const auto& tok : hv.added_tokens) {
            if (tok.content.empty()) continue;
            uint32_t node = 0;
            for (unsigned char byte : tok.content) {
                auto [it, inserted] = nodes[node].next.emplace(
                    byte, static_cast<uint32_t>(nodes.size()));
                if (inserted) nodes.emplace_back();
                node = it->second;
            }
            nodes[node].token_id = tok.id;
            max_bytes = std::max<uint32_t>(max_bytes,
                                           static_cast<uint32_t>(tok.content.size()));
        }
        for (const auto& e : nodes.front().next) root[e.first] = e.second;
    }

    // Longest literal starting at text[i]; returns its byte length or 0.
    uint32_t match(const uint8_t* text, size_t len, size_t i, uint32_t* id) const {
        if (max_bytes == 0) return 0;
        uint32_t node = root[text[i]];
        if (node == kDead) return 0;
        uint32_t best_id = nodes[node].token_id;
        uint32_t best_len = best_id == kDead ? 0u : 1u;
        const size_t limit = std::min<size_t>(len - i, max_bytes);
        for (size_t off = 1; off < limit; ++off) {
            auto it = nodes[node].next.find(text[i + off]);
            if (it == nodes[node].next.end()) break;
            node = it->second;
            if (nodes[node].token_id != kDead) {
                best_id = nodes[node].token_id;
                best_len = static_cast<uint32_t>(off + 1);
            }
        }
        if (best_len) *id = best_id;
        return best_len;
    }
};

}  // namespace

struct HostEncoder::Impl {
    RegexKind kind = RegexKind::GPT2;
    bool fold_long_s = false;
    bool ignore_merges = false;
    uint32_t byte_to_id[256];
    // ASCII bulk-classification tables, built once in the constructor.
    // ascii_cls[b] is gbpe_classify(b) for b < 0x80; ascii_ok[b] says the
    // classification loop may take b in its bulk run, i.e. b is ASCII and
    // starts no added token. Both are 0 for b >= 0x80, so a non-ASCII byte
    // always falls through to the general path.
    uint8_t ascii_cls[256] = {};
    uint8_t ascii_ok[256] = {};
    // Vector ASCII path: usable when the CPU has AVX2 and at most 4 distinct
    // ASCII bytes begin an added token (every shipped vocabulary has one, the
    // '<' of the chat-template literals), so the run scan can test them with
    // four broadcast compares. added_roots_ is padded by repetition, or 0xFF
    // when there are none, which matches no ASCII byte.
    bool avx2_ascii_ = false;
    uint8_t added_roots_[4] = {0xFF, 0xFF, 0xFF, 0xFF};
    MergeTable merges;
    AddedTrie added;
    // Raw pre-token bytes -> vocab id, for HF ignore_merges. Same predicate
    // as the device direct lookup: an exact raw-byte match in the vocabulary.
    //
    // Open-addressed and flat rather than unordered_map<string, uint32_t>: the
    // lookup runs on every cache miss, and the map form built a temporary
    // std::string per probe (a heap allocation above 15 bytes) and then hashed
    // it again, when the caller has already hashed those exact bytes for the
    // pre-token cache. Entries store the token id plus the byte span, so a hit
    // is verified byte-for-byte exactly as before.
    struct RawEntry {
        uint64_t hash = 0;
        uint32_t id = kDead;
        uint32_t off = 0;     // into raw_blob
        uint32_t len = 0;
    };
    std::vector<RawEntry> raw_table;   // power of two, kDead id = empty
    std::vector<uint8_t> raw_blob;
    uint32_t raw_mask = 0;

    // Look up `bytes` given the hash the caller already computed.
    uint32_t raw_lookup(const uint8_t* bytes, uint32_t len, uint64_t hash) const {
        if (raw_table.empty()) return kDead;
        uint32_t idx = static_cast<uint32_t>(hash) & raw_mask;
        for (;;) {
            const RawEntry& e = raw_table[idx];
            if (e.id == kDead) return kDead;
            if (e.hash == hash && e.len == len &&
                std::memcmp(raw_blob.data() + e.off, bytes, len) == 0) {
                return e.id;
            }
            idx = (idx + 1u) & raw_mask;
        }
    }

    // Pre-token result cache. A byte-level pre-token's encoding depends only
    // on its bytes, so a hit is exact. Direct-mapped, overwrite on collision,
    // keys verified byte-for-byte. Real text is Zipfian, so most pre-tokens of
    // a request are repeats and skip the merge loop entirely.
    struct CacheEntry {
        uint64_t hash = 0;
        uint8_t  len = 0;       // 0 = empty
        uint8_t  n_ids = 0;
        uint8_t  bytes[30];
        uint32_t ids[16];
    };
    // 104 bytes, so a probe straddles 1.6 cache lines and the 16-bit tier is
    // 6.5 MiB -- not the 48 B/3 MiB assumed in some earlier notes. Pinned so
    // the footprint cannot drift silently when a field is added.
    static_assert(sizeof(CacheEntry) == 104, "unexpected hashed cache entry size");
    // 18 bits = 262144 entries x 104 B = 26 MiB. Overridable at compile time
    // (-DGBPE_CACHE_BITS=n).
    //
    // Raised from 16 after measuring the 1-3 KiB band, where the wide tier was
    // the binding constraint on multibyte text: at 16 bits multilingual hit
    // only 67.8% with 30.7% of lookups evicting a live entry, against 91% /
    // 7% on prose. 17 bits lifts that to 75.7%/20.3% and 18 to 81.0%/12.1%.
    //
    // Being well past L2 turns out not to matter, which is the opposite of what
    // the 4-way associativity experiment suggested: the probe pattern is one
    // line per lookup (the key compare is the same line as the ids), so a
    // bigger table costs TLB reach, not extra lines per probe. Measured
    // monotonic at 2 KiB multilingual -- 54.2 / 49.5 / 47.1 us at 16 / 17 / 18
    // -- with books and code flat to slightly better, never worse.
#ifndef GBPE_CACHE_BITS
#define GBPE_CACHE_BITS 18
#endif
    static constexpr uint32_t kCacheBits = GBPE_CACHE_BITS;
    static constexpr uint32_t kCacheMask = (1u << kCacheBits) - 1u;
    static constexpr uint32_t kCacheMaxLen = 30;
    static constexpr uint32_t kCacheMaxIds = 16;
    mutable HugeVector<CacheEntry> cache;

    // Short-key cache. A pre-token of 8 bytes or fewer IS a 64-bit word, so
    // it needs no hash mixing and no memcmp: the slot comes from one multiply
    // and the key comparison is a single 64-bit compare. 84% of pre-tokens
    // qualify, and on the real distribution this path is 1.45x the hashed one
    // (7.77 vs 11.23 ns per lookup). Longer keys keep the hashed cache above.
    //
    // 32 bytes per entry (see the static_assert below), so the table is 2 MiB
    // at 16 bits and two entries share a cache line -- which is what makes the
    // 2-way set below free on a hit. Capped at 3 ids, which covers 99.6% of
    // pre-tokens; anything longer falls through to the hashed cache, which
    // stores 16.
    struct ShortEntry {
        uint64_t key = 0;
        uint32_t ids[3] = {};
        uint8_t  len = 0;       // 0 = empty
        uint8_t  n_ids = 0;
        uint8_t  pad[2] = {};
        uint32_t age = 0;       // LRU tick within its set
    };
    static_assert(sizeof(ShortEntry) == 32, "unexpected short entry padding");
    // 18 bits = 262144 entries x 32 B = 8 MiB. Overridable at compile time
    // (-DGBPE_SHORT_BITS=n).
    //
    // Raised from 16 alongside kCacheBits. This tier was never the binding
    // constraint -- it already hit 99.2% on prose and 95.5% on multilingual --
    // but widening it on top of the wide tier still paid on every corpus,
    // worth another 2-6% at 2-3 KiB, because the 4.0-4.3% of multilingual
    // lookups that evict here each cost a 367 ns miss.
#ifndef GBPE_SHORT_BITS
#define GBPE_SHORT_BITS 18
#endif
    static constexpr uint32_t kShortBits = GBPE_SHORT_BITS;
    // 2-way set-associative. Direct-mapped, this cache thrashes on text with a
    // wide pre-token vocabulary: simulated over the real access stream,
    // multilingual hits 76.3% with 14.4% evictions against 94.1%/2.6% on
    // prose. Pairing the same slots into sets lifts that to 79.9% and 95.6%
    // for no extra memory -- a miss here costs 367 ns, so the trade is worth
    // one extra compare on a hit.
    static constexpr uint32_t kShortWays = 2;
    static constexpr uint32_t kShortMaxIds = 3;
    // Huge-page backed, which also gives the 2-way sets 2 MiB alignment: a set
    // is 64 B, so with the base aligned no set straddles two cache lines.
    mutable HugeVector<ShortEntry> short_cache;
    mutable uint32_t short_tick = 0;   // LRU clock

    // The 64-bit value of a pre-token of `len` <= 8 bytes, and its slot.
    // Requires 8 readable bytes at `bytes`; the caller checks.
    static inline uint64_t short_key(const uint8_t* bytes, uint32_t len) {
        uint64_t k;
        std::memcpy(&k, bytes, 8);
        return (len == 8) ? k : (k & ((1ULL << (len * 8)) - 1ULL));
    }
    // Index of the set this key belongs to, in entries (ways are adjacent).
    static inline uint32_t short_slot(uint64_t key) {
        const uint32_t set = static_cast<uint32_t>(
            (key * 0x9E3779B185EBCA87ULL) >> (64 - kShortBits + 1));
        return set * kShortWays;
    }

    // Scratch reused across calls (single-threaded use per Tokenizer).
    mutable std::vector<uint8_t>  cls;
    mutable std::vector<uint8_t>  b0;
    mutable std::vector<uint32_t> byte_pos;
    mutable std::vector<uint32_t> direct_id;
    // Boundary list from the scan, consumed by emit_spans. One entry per
    // pre-token, so ~490 at 2 KiB -- a couple of KiB, L1-resident. Whole-input
    // rather than fixed 256-span chunks: there is no flush seam to get wrong,
    // and the host path is capped at host_max_bytes anyway.
    mutable std::vector<uint32_t> bnd;

    mutable std::vector<uint32_t> drun;
    mutable std::vector<uint32_t> parts;
    mutable std::vector<uint32_t> pair_rank;
    mutable std::vector<uint32_t> pair_new;
    // Heap-merge scratch (see bpe_merge_heap).
    struct HeapCand {
        uint32_t rank, pos, new_id, gen;
        // Ties break to the leftmost position, matching the linear loop's
        // argmin. Compaction preserves order, so "leftmost compacted index"
        // and "smallest original index" agree.
        bool operator>(const HeapCand& o) const {
            return rank != o.rank ? rank > o.rank : pos > o.pos;
        }
    };
    mutable std::vector<uint32_t> hsym, hprev, hnext, hgen;
    mutable std::vector<HeapCand> hheap;

    // Hash for the private pre-token cache. NOT raw_token_hash64: that one is
    // shared with the device tables and must stay byte-identical there, and
    // its FNV loop is a chain of dependent multiplies -- four of them for the
    // 4-byte pre-tokens that dominate real text. This cache is host-only and
    // its keys are verified byte-for-byte on every hit, so any mixing that
    // spreads well is sound. Read up to 8 bytes at a time and mix once.
    // Mix the two 64-bit words a key reduces to. Shared by both readers below
    // so they cannot disagree.
    static inline uint64_t cache_mix(uint64_t a, uint64_t b, uint32_t len) {
        uint64_t h = a * 0x9E3779B185EBCA87ULL + (b ^ len) * 0xC2B2AE3D27D4EB4FULL;
        h ^= h >> 29;
        h *= 0xBF58476D1CE4E5B9ULL;
        h ^= h >> 32;
        return h;
    }

    // Branchless form, for keys with at least 8 readable bytes at `bytes`.
    // Pre-token lengths vary call to call, so the length dispatch in the safe
    // version below mispredicts constantly: measured 8.27 ns/key branchy vs
    // 4.89 branchless on the real length distribution. Two unaligned 8-byte
    // loads, masked to the key length, give the same value as reading exactly
    // `len` bytes -- the mask discards whatever followed the key.
    static inline uint64_t cache_hash_wide(const uint8_t* bytes, uint32_t len) {
        uint64_t a, b;
        std::memcpy(&a, bytes, 8);
        std::memcpy(&b, bytes + (len > 8 ? len - 8 : 0), 8);
        const uint64_t mask = (len >= 8) ? ~0ULL : ((1ULL << (len * 8)) - 1ULL);
        return cache_mix(a & mask, b & mask, len);
    }

    // Safe form: reads exactly `len` bytes. Used when fewer than 8 bytes remain
    // in the caller's buffer, and by the constructor when hashing vocabulary
    // entries. Must produce the same value as cache_hash_wide for every key.
    static uint64_t cache_hash(const uint8_t* bytes, uint32_t len) {
        uint64_t a = 0, b = 0;
        if (len >= 8) {
            std::memcpy(&a, bytes, 8);
            std::memcpy(&b, bytes + len - 8, 8);
        } else {
            // Little-endian: byte i of the key is byte i of the word, which is
            // what the masked 8-byte load above yields.
            for (uint32_t i = 0; i < len; ++i)
                a |= static_cast<uint64_t>(bytes[i]) << (i * 8);
            b = a;
        }
        return cache_mix(a, b, len);
    }

    // `readable` is how many bytes may be read at `bytes` (the pre-token's own
    // length plus whatever else remains in the caller's buffer). The wide hash
    // needs 8; the last pre-token of an input usually has fewer, so it takes
    // the safe path. Both hashes are verified to produce identical values.
    void bpe_pretoken(const uint8_t* bytes, uint32_t len, size_t readable,
                      std::vector<uint32_t>& out) const {
        // Short keys (84% of pre-tokens) need no hash and no memcmp.
        if (len != 0 && len <= 8 && readable >= 8) {
            GBPE_STAT(short_lookups);
            const uint64_t key = short_key(bytes, len);
            ShortEntry* set = short_cache.data() + short_slot(key);
            for (uint32_t w = 0; w < kShortWays; ++w) {
                ShortEntry& se = set[w];
                if (se.len == len && se.key == key) {
                    GBPE_STAT(short_hits);
                    se.age = ++short_tick;
                    const uint32_t cnt = se.n_ids;
                    for (uint32_t i = 0; i < cnt; ++i) out.push_back(se.ids[i]);
                    return;
                }
            }
            const size_t begin = out.size();
            bpe_pretoken_uncached(bytes, len, out, cache_hash_wide(bytes, len));
            const size_t produced = out.size() - begin;
            if (produced <= kShortMaxIds) {
                uint32_t victim = 0;
                for (uint32_t w = 1; w < kShortWays; ++w)
                    if (set[w].age < set[victim].age) victim = w;
                ShortEntry& se = set[victim];
                // An occupied victim means a capacity/conflict eviction, which
                // is what the sizing sweep is trying to remove.
                GBPE_STAT_IF(se.len != 0, short_evictions);
                se.key = key;
                se.len = static_cast<uint8_t>(len);
                se.n_ids = static_cast<uint8_t>(produced);
                se.age = ++short_tick;
                std::memcpy(se.ids, out.data() + begin, produced * sizeof(uint32_t));
            }
            return;
        }
        CacheEntry* slot = nullptr;
        uint64_t hash = 0;
        if (len <= kCacheMaxLen) {
            GBPE_STAT(wide_lookups);
            hash = (readable >= 8) ? cache_hash_wide(bytes, len)
                                   : cache_hash(bytes, len);
            slot = &cache[static_cast<uint32_t>(hash) & kCacheMask];
            if (slot->len == len && slot->hash == hash &&
                std::memcmp(slot->bytes, bytes, len) == 0) {
                GBPE_STAT(wide_hits);
                out.insert(out.end(), slot->ids, slot->ids + slot->n_ids);
                return;
            }
        } else {
            GBPE_STAT(uncacheable);
        }
        const size_t out_begin = out.size();
        bpe_pretoken_uncached(bytes, len, out, hash);
        if (slot != nullptr && out.size() - out_begin <= kCacheMaxIds) {
            GBPE_STAT_IF(slot->len != 0, wide_evictions);
            slot->hash = hash;
            slot->len = static_cast<uint8_t>(len);
            slot->n_ids = static_cast<uint8_t>(out.size() - out_begin);
            std::memcpy(slot->bytes, bytes, len);
            std::memcpy(slot->ids, out.data() + out_begin, slot->n_ids * sizeof(uint32_t));
        }
    }

    // `hash` is the caller's cache_hash of these bytes, or 0 when unknown
    // (len > kCacheMaxLen); the raw table recomputes it in that case.
    // Same result as the linear loop, in O(L log L) instead of O(R*L).
    // Symbols live in a doubly-linked list; a min-heap holds candidate merges.
    // A candidate is stale iff its position's version moved since it was
    // pushed, so every mutation bumps the versions it invalidates -- including
    // marking the consumed node dead, without which its pending candidates
    // fire on a node that is no longer in the list and silently drop a merge.
    void bpe_merge_heap(const uint8_t* bytes, uint32_t len,
                        std::vector<uint32_t>& out) const {
        hsym.assign(len, 0); hprev.assign(len, 0);
        hnext.assign(len, 0); hgen.assign(len, 0);
        for (uint32_t i = 0; i < len; ++i) {
            hsym[i] = byte_to_id[bytes[i]];
            hprev[i] = i ? i - 1 : kDead;
            hnext[i] = (i + 1 < len) ? i + 1 : kDead;
        }
        hheap.clear();
        auto push = [&](uint32_t i) {
            const uint32_t j = hnext[i];
            if (j == kDead) return;
            uint32_t nid = 0;
            const uint32_t rk = merges.probe(hsym[i], hsym[j], &nid);
            if (rk == kDead) return;
            hheap.push_back(HeapCand{rk, i, nid, hgen[i]});
            std::push_heap(hheap.begin(), hheap.end(), std::greater<HeapCand>());
        };
        for (uint32_t i = 0; i + 1 < len; ++i) {
            uint32_t nid = 0;
            const uint32_t rk = merges.probe(hsym[i], hsym[i + 1], &nid);
            if (rk != kDead) hheap.push_back(HeapCand{rk, i, nid, 0});
        }
        std::make_heap(hheap.begin(), hheap.end(), std::greater<HeapCand>());
        while (!hheap.empty()) {
            std::pop_heap(hheap.begin(), hheap.end(), std::greater<HeapCand>());
            const HeapCand c = hheap.back();
            hheap.pop_back();
            const uint32_t i = c.pos;
            if (c.gen != hgen[i]) continue;          // superseded
            const uint32_t j = hnext[i];
            if (j == kDead) continue;
            hsym[i] = c.new_id;
            const uint32_t k = hnext[j];
            hnext[i] = k;
            if (k != kDead) hprev[k] = i;
            hgen[j] = kDead;                         // j is consumed
            ++hgen[i];
            const uint32_t pv = hprev[i];
            if (pv != kDead) ++hgen[pv];             // its right symbol changed
            push(i);
            if (pv != kDead) push(pv);
        }
        for (uint32_t i = 0; i != kDead; i = hnext[i]) out.push_back(hsym[i]);
    }

    // Above this length the heap's bookkeeping is cheaper than rescanning.
    // Measured crossover on real pre-tokens: 48 bytes.
    static constexpr uint32_t kHeapMergeMinLen = 48;

    void bpe_pretoken_uncached(const uint8_t* bytes, uint32_t len,
                               std::vector<uint32_t>& out, uint64_t hash = 0) const {
        if (ignore_merges) {
            const uint64_t h = hash ? hash : cache_hash(bytes, len);
            const uint32_t id = raw_lookup(bytes, len, h);
            if (id != kDead) { out.push_back(id); return; }
        }
        if (len >= kHeapMergeMinLen) { bpe_merge_heap(bytes, len, out); return; }
        parts.resize(len);
        for (uint32_t i = 0; i < len; ++i) parts[i] = byte_to_id[bytes[i]];
        if (len < 2) { out.insert(out.end(), parts.begin(), parts.end()); return; }

        // Cached adjacent-pair ranks; a merge invalidates only its two
        // neighbours. Argmin over (rank, index) picks the leftmost among equal
        // ranks, matching the warp-wide (rank, slot) minimum in bpe_kernel.
        uint32_t n = len;
        pair_rank.resize(n);
        pair_new.resize(n);
        // Memoizing this first round over a dense 65536-entry byte-pair grid was
        // measured and rejected.
        for (uint32_t i = 0; i + 1 < n; ++i) {
            pair_rank[i] = merges.probe(parts[i], parts[i + 1], &pair_new[i]);
        }
        for (;;) {
            uint32_t best = kDead, bi = 0;
            for (uint32_t i = 0; i + 1 < n; ++i) {
                if (pair_rank[i] < best) { best = pair_rank[i]; bi = i; }
            }
            if (best == kDead) break;
            parts[bi] = pair_new[bi];
            for (uint32_t i = bi + 1; i + 1 < n; ++i) {
                parts[i] = parts[i + 1];
                pair_rank[i] = pair_rank[i + 1];
                pair_new[i] = pair_new[i + 1];
            }
            --n;
            if (bi + 1 < n) pair_rank[bi] = merges.probe(parts[bi], parts[bi + 1], &pair_new[bi]);
            if (bi > 0)     pair_rank[bi - 1] = merges.probe(parts[bi - 1], parts[bi], &pair_new[bi - 1]);
        }
        out.insert(out.end(), parts.begin(), parts.begin() + n);
    }

    // Boundary scan for the Llama/Qwen families, emitting one pre-token per
    // boundary run. The body is the same two-step decision the scalar loop
    // made -- local-window fast path first, full predicate on kFastUnknown --
    // but where AVX2 is available the fast path is evaluated 32 positions at a
    // time into a resolved/answer bitmap pair, and only the lanes it declines
    // pay the scalar cascade. `llamaqwen_fast_x32` is verified lane-for-lane
    // against `llamaqwen_fast` over random adversarial arrays and all three
    // corpora, so this changes throughput only, never token IDs.
    template<bool kDigitCap, typename EmitFn>
    void scan_llamaqwen(const uint8_t* C, const uint8_t* B,
                        const uint32_t* drun_ptr, uint32_t n,
                        const uint32_t* D, EmitFn&& emit, uint32_t& start) const {
        uint32_t c = 1;
#if GBPE_HAVE_AVX2_BOUNDARY
        // The vector window needs c-3 .. c+32 in range and a single document
        // (D == nullptr); the head, the tail and every multi-document encode
        // go through the scalar path below.
        if (D == nullptr && n >= 40 && cpu_has_avx2()) {
            for (; c < 3u && c < n; ++c) {
                if (!boundary_at<kDigitCap>(C, B, drun_ptr, n, D, c)) continue;
                emit(start, c); start = c;
            }
            for (; c + 33u <= n; c += 32u) {
                uint32_t resolved = 0, answer = 0;
                llamaqwen_fast_x32(C, B, c, &resolved, &answer);
                // Lanes the vector path declined still need the cascade; doing
                // them here keeps boundaries in ascending order for emit().
                uint32_t todo = ~resolved & 0xFFFFFFFFu;
                while (todo) {
                    const uint32_t j = static_cast<uint32_t>(__builtin_ctz(todo));
                    todo &= todo - 1u;
                    if (llamaqwen_boundary<kDigitCap>(C, B, drun_ptr, n, D, c + j))
                        answer |= 1u << j;
                }
                while (answer) {
                    const uint32_t j = static_cast<uint32_t>(__builtin_ctz(answer));
                    answer &= answer - 1u;
                    emit(start, c + j); start = c + j;
                }
            }
        }
#endif
        for (; c <= n; ++c) {
            if (c < n && !boundary_at<kDigitCap>(C, B, drun_ptr, n, D, c)) continue;
            emit(start, c); start = c;
        }
    }

    // Scalar two-step decision, shared by the head/tail and the fallback.
    template<bool kDigitCap>
    static bool boundary_at(const uint8_t* C, const uint8_t* B,
                            const uint32_t* drun_ptr, uint32_t n,
                            const uint32_t* D, uint32_t c) {
        const FastBoundary f = llamaqwen_fast(C, B, n, D, c);
        if (f != kFastUnknown) return f == kFastYes;
        return llamaqwen_boundary<kDigitCap>(C, B, drun_ptr, n, D, c);
    }

    void encode(const uint8_t* text, size_t len, std::vector<uint32_t>& out) const {
        out.clear();
        if (len == 0) return;

        // Stage 1: dense per-codepoint arrays, exactly as k1/gather build them.
        // An added-token literal is one CLS_ADDED entry covering all its bytes;
        // bytes inside it are not codepoint starts. A byte is a start iff it is
        // not a UTF-8 continuation byte, so leading orphan continuation bytes
        // belong to no pre-token and are dropped, as on the device.
        // At most one entry per byte; size once and write by index.
        if (cls.size() < len) {
            cls.resize(len); b0.resize(len); byte_pos.resize(len); direct_id.resize(len);
        }
        uint32_t n = 0;
        size_t i = 0;
        const bool has_added = added.max_bytes != 0;
        while (i < len) {
            // Bulk ASCII run. Real text is overwhelmingly ASCII (96.3% of the
            // books corpus by byte), and for a byte < 0x80 that starts no added
            // token the three arrays advance in lockstep with the input: one
            // codepoint per byte, class from a 256-entry table, b0 the byte
            // itself. Doing that in a tight loop over a whole run lets the
            // compiler vectorise the copy and keeps the general path below for
            // the rare multi-byte or added-token case.
            //
            // ascii_ok[] folds in the added-token root test so the run does not
            // have to re-check it per byte; when the vocabulary has no added
            // tokens every ASCII byte qualifies.
            size_t run = i;
#if GBPE_HAVE_AVX2_BOUNDARY
            // Find the run 32 bytes at a time when the tail is long enough.
            if (avx2_ascii_) {
                while (run + 32 <= len) {
                    const uint32_t k = ascii_run_32(text + run, added_roots_);
                    run += k;
                    if (k != 32u) break;
                }
            }
#endif
            while (run < len && ascii_ok[text[run]]) ++run;
            if (run > i) {
                size_t k = i;
#if GBPE_HAVE_AVX2_BOUNDARY
                // Classify the run 32 bytes at a time: for ASCII the class is
                // three range comparisons, no table lookup (10.5x the scalar
                // loop). b0 is the byte itself, so it is a straight copy.
                if (avx2_ascii_) {
                    for (; k + 32 <= run; k += 32) {
                        classify_ascii_32(text + k, cls.data() + n);
                        std::memcpy(b0.data() + n, text + k, 32);
                        for (uint32_t j = 0; j < 32; ++j)
                            byte_pos[n + j] = static_cast<uint32_t>(k + j);
                        n += 32;
                    }
                }
#endif
                for (; k < run; ++k) {
                    const uint8_t b = text[k];
                    cls[n] = ascii_cls[b];
                    b0[n] = b;
                    byte_pos[n] = static_cast<uint32_t>(k);
                    ++n;
                }
                i = run;
                if (i >= len) break;
            }
            const uint8_t byte = text[i];
            if (has_added && added.root[byte] != kDead) {
                uint32_t id = 0;
                const uint32_t alen = added.match(text, len, i, &id);
                if (alen) {
                    cls[n] = CLS_ADDED; b0[n] = 0; byte_pos[n] = static_cast<uint32_t>(i);
                    direct_id[n] = id;
                    ++n; i += alen;
                    continue;
                }
            }
            if ((byte & 0xC0u) == 0x80u) { ++i; continue; }
            if (byte < 0x80u) {
                cls[n] = gbpe_classify(byte);
                b0[n] = byte;
            } else {
                int width = 1;
                const uint32_t cp = decode_cp(text + i, static_cast<uint32_t>(len - i), &width);
                cls[n] = gbpe_classify(cp);
                const bool long_s = fold_long_s && byte == 0xC5u && i + 1 < len && text[i + 1] == 0xBFu;
                b0[n] = long_s ? static_cast<uint8_t>('s') : byte;
            }
            byte_pos[n] = static_cast<uint32_t>(i);
            ++n; ++i;   // the next start is found by the continuation test above
        }
        if (n == 0) return;

        // Digit-run starts (Llama only): drun[c] is the ordinal of the first
        // codepoint of c's digit run, read by the predicate's \p{N}{1,3} rule.
        //
        // Only the digit entries are ever read, and digits are rare in real
        // text (0.015% of the books corpus), so instead of the device's
        // inclusive max-scan over every codepoint we fill runs as we find
        // them. Non-digit entries are left untouched and never read. The
        // walk is forward-only over each run, so a single long run costs its
        // own length once, not once per position.
        const uint32_t* drun_ptr = nullptr;
        if (kind == RegexKind::Llama3) {
            if (drun.size() < n) drun.resize(n);
            for (uint32_t c = 0; c < n; ) {
                if (cls[c] != CLS_N) { ++c; continue; }
                const uint32_t run_start = c;
                do { drun[c] = run_start; ++c; } while (c < n && cls[c] == CLS_N);
            }
            drun_ptr = drun.data();
        }

        // Stage 2: boundaries, then one pre-token per boundary run. A pre-token
        // extends to the next boundary's first byte, so trailing bytes that
        // start no codepoint are absorbed by it (k3 semantics).
        const uint8_t* C = cls.data();
        const uint8_t* B = b0.data();
        const uint32_t* D = nullptr;   // single document: range test only
#if GBPE_HOST_STOP_AFTER == 1
        // Attribution build: stop after classify. Output is deliberately wrong.
        out.push_back(n);
        return;
#endif
        // Stage 2 collects boundaries; Stage 3 below turns them into tokens.
        //
        // These were one fused pass: the scan called an emit lambda per span,
        // which hashed, probed the pre-token cache and merged before the scan
        // looked at the next position. That threw away the lookahead the
        // vector path had already computed -- llamaqwen_fast_x32 resolves 32
        // positions into a bitmap, so the next ~32 span boundaries are known
        // before any of them is needed -- and made every cache probe a
        // serially dependent random access into a 34 MiB table pair with
        // nothing to overlap its latency.
        //
        // Splitting them is not the refuted "fuse the passes" experiment; it
        // is its inverse. That one merged classify+boundary+emit into a single
        // loop and lost 2.12 us to 4.11 because a fused loop will not
        // vectorise. Separate passes won there and win here.
        //
        // Measured justification: holding the hit rate at 99.9% and growing
        // only the working set, multilingual 2 KiB costs 22.4 us over ~2.9k
        // keys and 26.5 us over ~23k. That 4 us is pure memory latency on
        // cache *hits*, which is what the prefetch ladder in Stage 3 hides.
        bnd.clear();
#if GBPE_HOST_STOP_AFTER == 3
        // Attribution build: run the boundary predicate but throw the answers
        // away instead of appending them, so (stop-2 minus stop-3) isolates the
        // cost of the drain -- one push_back per boundary -- from the predicate
        // kernel itself. Accumulated into a sink the compiler cannot discard.
        uint32_t sink = 0;
        auto collect = [&](uint32_t start, uint32_t c) {
            (void)start;
            sink += c;
        };
#else
        auto collect = [&](uint32_t start, uint32_t c) {
            (void)start;
            bnd.push_back(c);
        };
#endif
        // One family-specialised loop each, so the predicate inlines.
        uint32_t start = 0;
        switch (kind) {
            case RegexKind::GPT2:
                for (uint32_t c = 1; c <= n; ++c) {
                    if (c < n) {
                        const FastBoundary f = gpt2_fast(C, B, n, D, c);
                        if (f == kFastNo) continue;
                        if (f == kFastUnknown && !gpt2_boundary(C, B, n, D, c)) continue;
                    }
                    collect(start, c); start = c;
                }
                break;
            // The inlined local-window check answers ~88% of prose positions
            // without entering the full predicate (which builds a frame for
            // its lambda cascade on entry, ~170 instructions even for a
            // position it resolves immediately).
            case RegexKind::Llama3:
                scan_llamaqwen<true>(C, B, drun_ptr, n, D, collect, start);
                break;
            case RegexKind::Qwen25:
                scan_llamaqwen<false>(C, B, nullptr, n, D, collect, start);
                break;
            default:
                for (uint32_t c = 1; c <= n; ++c) { collect(start, c); start = c; }
        }

#if GBPE_HOST_STOP_AFTER == 3
        out.push_back(sink);
        return;
#endif
#if GBPE_HOST_STOP_AFTER == 2
        // Attribution build: stop after the boundary scan, before any cache
        // probe or merge. Output is deliberately wrong.
        out.push_back(static_cast<uint32_t>(bnd.size()));
        return;
#endif
        emit_spans(text, len, n, out);
    }

    // Stage 3. Walks the boundary list built by the scan, prefetching each
    // span's cache line a fixed distance ahead of the probe that needs it.
    //
    // Order is preserved exactly: spans are walked ascending and appended in
    // that order, so token IDs, LRU age bumps and cache insertions all happen
    // in the same sequence as the fused version. Cache *contents* are
    // bit-identical too, not just the output. Only the timing of the loads
    // moves, and a prefetch has no architectural effect.
    void emit_spans(const uint8_t* text, size_t len, uint32_t n,
                    std::vector<uint32_t>& out) const {
        const uint32_t m = static_cast<uint32_t>(bnd.size());
        if (m == 0) return;

        // Every byte yields at most one token (one symbol per byte, merges only
        // reduce, an added-token literal collapses many bytes to one id), so
        // len is an exact upper bound and the append loop never reallocates.
        out.reserve(out.size() + len);

        // Software prefetch of these probes was measured and rejected, in both
        // the naive form and a two-pass form that derives every span's slot up
        // front so the prefetch costs nothing to address.
        // The win this split
        // delivers is batching, one reserve and a tighter loop -- not latency
        // hiding.
        for (uint32_t i = 0; i < m; ++i) {
            const uint32_t s = (i == 0) ? 0u : bnd[i - 1];
            const uint32_t c = bnd[i];
            if (cls[s] == CLS_ADDED) {
                out.push_back(direct_id[s]);
                continue;
            }
            const uint32_t sb = byte_pos[s];
            const uint32_t eb = (c < n) ? byte_pos[c] : static_cast<uint32_t>(len);
            bpe_pretoken(text + sb, eb - sb, len - sb, out);
        }
    }
};

HostEncoder::HostEncoder(const HostVocab& hv) : impl_(new Impl) {
    Impl& im = *impl_;
    im.kind = hv.regex_kind;
    const bool byte_level = !hv.byte_fallback &&
        (hv.regex_kind == RegexKind::GPT2 || hv.regex_kind == RegexKind::Llama3 ||
         hv.regex_kind == RegexKind::Qwen25);
    if (!byte_level || hv.byte_to_id.size() != 256) return;
    im.fold_long_s = hv.regex_kind == RegexKind::Llama3 || hv.regex_kind == RegexKind::Qwen25;
    im.ignore_merges = hv.ignore_merges;
    for (int b = 0; b < 256; ++b) im.byte_to_id[b] = hv.byte_to_id[b];
    im.merges.build(hv);
    im.added.build(hv);
    // ASCII bulk tables. Built after the trie so ascii_ok can exclude bytes
    // that begin an added token; those must go through the general path.
    for (int b = 0; b < 128; ++b) {
        im.ascii_cls[b] = gbpe_classify(static_cast<uint32_t>(b));
        const bool starts_added =
            im.added.max_bytes != 0 && im.added.root[b] != kDead;
        im.ascii_ok[b] = starts_added ? 0u : 1u;
    }
    // Collect the ASCII bytes that begin an added token. The vector run scan
    // tests up to four of them; with more, it stays on the scalar path.
    {
        uint8_t roots[5];
        int n_roots = 0;
        for (int b = 0; b < 128 && n_roots <= 4; ++b) {
            if (im.added.max_bytes != 0 && im.added.root[b] != kDead)
                roots[n_roots++] = static_cast<uint8_t>(b);
        }
        if (n_roots <= 4) {
            for (int k = 0; k < 4; ++k)
                im.added_roots_[k] = (n_roots == 0) ? 0xFFu : roots[k % n_roots];
            im.avx2_ascii_ = cpu_has_avx2();
        }
    }
    im.cache.assign(1u << Impl::kCacheBits, Impl::CacheEntry{});
    im.short_cache.assign(1u << Impl::kShortBits, Impl::ShortEntry{});
    if (hv.ignore_merges) {
        size_t n_tokens = 0, total_bytes = 0;
        for (const auto& tb : hv.token_bytes) {
            if (tb.empty()) continue;
            ++n_tokens; total_bytes += tb.size();
        }
        uint32_t cap = 16;
        while (cap < n_tokens * 2) cap <<= 1;     // load factor <= 0.5
        im.raw_mask = cap - 1;
        im.raw_table.assign(cap, Impl::RawEntry{});
        im.raw_blob.reserve(total_bytes);
        for (uint32_t id = 0; id < hv.token_bytes.size(); ++id) {
            const auto& tb = hv.token_bytes[id];
            if (tb.empty()) continue;
            const uint32_t off = static_cast<uint32_t>(im.raw_blob.size());
            im.raw_blob.insert(im.raw_blob.end(), tb.begin(), tb.end());
            // Hash from the source, not from raw_blob: the reserve above makes
            // reallocation impossible, but not depending on that keeps this
            // correct if the sizing ever drifts.
            const uint64_t h = Impl::cache_hash(
                reinterpret_cast<const uint8_t*>(tb.data()),
                static_cast<uint32_t>(tb.size()));
            uint32_t idx = static_cast<uint32_t>(h) & im.raw_mask;
            while (im.raw_table[idx].id != kDead) idx = (idx + 1u) & im.raw_mask;
            im.raw_table[idx] = Impl::RawEntry{h, id, off,
                                               static_cast<uint32_t>(tb.size())};
        }
    }
    supported_ = true;
}

HostEncoder::~HostEncoder() = default;

void HostEncoder::encode(const uint8_t* text, size_t len, std::vector<uint32_t>& out) const {
    impl_->encode(text, len, out);
}

}  // namespace gbpe
