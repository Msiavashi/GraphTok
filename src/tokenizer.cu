// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 the cuTokenize contributors

#include "tokenizer.cuh"
#include "vocab.h"
#include "build_config.h"
#include "pretok_boundary.h"

#include <cuda_runtime.h>
#include <cub/cub.cuh>
#include <stdexcept>

// Max reduction op for CUB scans. Defined locally so the build works on both
// CUDA 12.x (CCCL 2.x, which has cub::Max) and 13.x (CCCL 3.x, which renamed it
// to cuda::maximum<>) — neither name is portable across the floor we support.
namespace gbpe { struct MaxOp {
    template<typename T> __host__ __device__ T operator()(const T& a, const T& b) const {
        return a > b ? a : b;
    }
};

struct U8ToU32 {
    __host__ __device__ uint32_t operator()(uint8_t v) const {
        return static_cast<uint32_t>(v);
    }
};

using U8ToU32Iterator =
    cub::TransformInputIterator<uint32_t, U8ToU32, const uint8_t*>;

// Two independent exclusive sums over the same domain, scanned as one
// sequence so they cost one graph node instead of two. Used for the
// short/long bucket offsets, which k3 produces and k4 consumes together.
struct ShortLong {
    uint32_t s, l;
    __host__ __device__ ShortLong operator+(const ShortLong& o) const {
        return ShortLong{s + o.s, l + o.l};
    }
};

// Reads the two separate flag arrays as one pair sequence.
struct ShortLongZipIn {
    const uint32_t* s;
    const uint32_t* l;
    using value_type = ShortLong;
    using reference = ShortLong;
    using pointer = void;
    using difference_type = int;
    using iterator_category = std::random_access_iterator_tag;
    __host__ __device__ ShortLong operator[](int i) const {
        return ShortLong{s[i], l[i]};
    }
    __host__ __device__ ShortLongZipIn operator+(int i) const {
        return ShortLongZipIn{s + i, l + i};
    }
    __host__ __device__ ShortLong operator*() const { return (*this)[0]; }
};

// A proxy whose assignment scatters the pair back into the two result arrays,
// so no caller downstream has to change how it reads them.
struct ShortLongRef {
    uint32_t* s;
    uint32_t* l;
    __host__ __device__ void operator=(const ShortLong& v) const { *s = v.s; *l = v.l; }
};
struct ShortLongZipOut {
    uint32_t* s;
    uint32_t* l;
    using value_type = ShortLong;
    using reference = ShortLongRef;
    using pointer = void;
    using difference_type = int;
    using iterator_category = std::random_access_iterator_tag;
    __host__ __device__ ShortLongRef operator[](int i) const {
        return ShortLongRef{s + i, l + i};
    }
    __host__ __device__ ShortLongZipOut operator+(int i) const {
        return ShortLongZipOut{s + i, l + i};
    }
    __host__ __device__ ShortLongRef operator*() const { return (*this)[0]; }
};
}  // namespace gbpe

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace gbpe {

#define CUDA_CHECK(call) do {                                       \
    cudaError_t err__ = (call);                                     \
    if (err__ != cudaSuccess) {                                     \
        std::fprintf(stderr, "CUDA error %s:%d: %s\n",              \
                     __FILE__, __LINE__, cudaGetErrorString(err__));\
        std::abort();                                               \
    }                                                               \
} while (0)


// =============================================================================
//  Merge-table lookup. Open-addressing, linear probe over packed 64-bit slots.
//  Layout: [left:16][right:16][new_id:16][rank:16], rank==0xFFFF == empty.
//  ONE 64-bit load per probe (was two 64-bit loads with split key/val arrays).
// =============================================================================

// Slot policies: abstract the merge-table layout. Kernel body calls
// Slot::probe(args, l, r) -> {rank, new_id}, regardless of underlying layout.

struct ProbeResult { uint32_t rank; uint32_t new_id; };

// A lookup is split in two: issue() hashes the pair and starts the load of
// its home slot, finish() examines it and walks on if needed. Two lookups
// issued before either is finished wait on one memory latency together,
// where two probe() calls in a row wait on two. (Loading several slots per
// step as well was measured: no gain at small inputs, up to 8% slower at
// 1M tokens from the extra traffic.)

#if !GBPE_USE_SLOT128_ONLY
// Slot64 -- one 64-bit load per probe. GPT-2 path (vocab<=64k, merges<=64k).
struct Slot64 {
    struct Args {
        const uint64_t* __restrict__ slots;
        uint32_t mask;
    };
    struct Pending {
        uint32_t want_key;
        uint32_t idx;
        uint64_t s;
    };
    static __device__ __forceinline__ void issue(
        Args a, uint32_t left_id, uint32_t right_id, Pending& p)
    {
        p.want_key = pack_key32(left_id, right_id);
        p.idx = merge_hash_key32(p.want_key, a.mask);
        p.s = a.slots[p.idx];
    }
    static __device__ __forceinline__ ProbeResult finish(Args a, const Pending& p)
    {
        uint32_t idx = p.idx;
        uint64_t s = p.s;
        #pragma unroll 4
        for (int probe = 0; probe < 64; ++probe) {
            if (probe > 0) s = a.slots[idx];
            uint32_t rank16 = static_cast<uint32_t>(s & 0xFFFFu);
            if (rank16 == 0xFFFFu) return {RANK_NONE, TOKEN_DEAD};
            uint32_t k = static_cast<uint32_t>(s >> 32);
            if (k == p.want_key) {
                return {rank16,
                        static_cast<uint32_t>((s >> 16) & 0xFFFFu)};
            }
            idx = (idx + 1) & a.mask;
        }
        return {RANK_NONE, TOKEN_DEAD};
    }
    static __device__ __forceinline__ ProbeResult probe(
        Args a, uint32_t left_id, uint32_t right_id)
    {
        Pending p;
        issue(a, left_id, right_id, p);
        return finish(a, p);
    }
};

#endif  // !GBPE_USE_SLOT128_ONLY

#if !GBPE_USE_SLOT64_ONLY
// Slot128 -- two 64-bit loads per probe. Llama-3 / Qwen / DeepSeek / Gemma path.
struct Slot128 {
    struct Args {
        const uint64_t* __restrict__ keys;
        const uint64_t* __restrict__ vals;
        uint32_t mask;
    };
    struct Pending {
        uint64_t want_key;
        uint32_t idx;
        uint64_t v, k;
    };
    static __device__ __forceinline__ void issue(
        Args a, uint32_t left_id, uint32_t right_id, Pending& p)
    {
        p.want_key = pack_key64(left_id, right_id);
        p.idx = merge_hash_key64(p.want_key, a.mask);
        // Both loads issue together: the probe waits on one memory latency
        // instead of two in series. keys[idx] is in bounds for an empty slot
        // too, and its value is ignored there.
        p.v = a.vals[p.idx];
        p.k = a.keys[p.idx];
    }
    static __device__ __forceinline__ ProbeResult finish(Args a, const Pending& p)
    {
        uint32_t idx = p.idx;
        uint64_t v = p.v, k = p.k;
        #pragma unroll 4
        for (int probe = 0; probe < 64; ++probe) {
            if (probe > 0) {
                v = a.vals[idx];
                k = a.keys[idx];
            }
            uint32_t rank = static_cast<uint32_t>(v & 0xFFFFFFFFu);
            if (rank == 0xFFFFFFFFu) return {RANK_NONE, TOKEN_DEAD};
            if (k == p.want_key) {
                return {rank, static_cast<uint32_t>(v >> 32)};
            }
            idx = (idx + 1) & a.mask;
        }
        return {RANK_NONE, TOKEN_DEAD};
    }
    static __device__ __forceinline__ ProbeResult probe(
        Args a, uint32_t left_id, uint32_t right_id)
    {
        Pending p;
        issue(a, left_id, right_id, p);
        return finish(a, p);
    }
};
#endif  // !GBPE_USE_SLOT64_ONLY

// Optional full-pre-token lookup for HF BPE model.ignore_merges. Hash hits are
// always verified against the packed raw-byte vocabulary storage, so the hash
// table is a performance index only and cannot introduce a collision error.
struct DirectTokenArgs {
    const uint64_t* __restrict__ hashes;
    const uint32_t* __restrict__ ids;
    uint32_t mask;
    const uint8_t*  __restrict__ token_bytes;
    const uint32_t* __restrict__ token_offsets;
    const uint16_t* __restrict__ token_lens;
};

__device__ __forceinline__ uint32_t direct_token_lookup(
    DirectTokenArgs a, const uint8_t* __restrict__ bytes, uint32_t len)
{
    if (a.ids == nullptr || len == 0) return TOKEN_DEAD;
    const uint64_t hash = raw_token_hash64(bytes, len);
    uint32_t slot = raw_token_hash_slot(hash, a.mask);
    // The table is allocated at <=0.5 load, but do not impose a correctness
    // limit on a rare collision/probe chain.
    for (uint32_t probe = 0; probe <= a.mask; ++probe) {
        const uint32_t id = a.ids[slot];
        if (id == TOKEN_DEAD) return TOKEN_DEAD;
        if (a.hashes[slot] == hash &&
            static_cast<uint32_t>(a.token_lens[id]) == len) {
            const uint8_t* candidate = a.token_bytes + a.token_offsets[id];
            bool same = true;
            #pragma unroll 1
            for (uint32_t i = 0; i < len; ++i) {
                if (candidate[i] != bytes[i]) { same = false; break; }
            }
            if (same) return id;
        }
        slot = (slot + 1u) & a.mask;
    }
    return TOKEN_DEAD;
}


// =============================================================================
//  bpe_kernel<LEN>: one warp per pre-token, two warps per block.
//
//  Templated on the per-pre-token byte capacity LEN (32 for short bucket,
//  64 for long bucket). SLOTS_PER_THREAD = LEN / 32, so:
//    LEN=32: 1 slot per thread, 1-row argmin, 31-iter merge loop
//    LEN=64: 2 slots per thread, 2-row argmin, 63-iter merge loop
//  The compiler constexpr-eliminates row-1 paths when SLOTS_PER_THREAD==1.
//
//  RULES OF WARP COOPERATION (every line obeys these):
//    1. Every warp-level op (__ballot_sync, __shfl_*_sync, warp reduce) runs
//       on ALL 32 lanes with mask 0xFFFFFFFF.
//    2. Lanes with slot >= N do NOT early-return. They carry TOKEN_DEAD
//       and RANK_NONE.
//    3. Only the whole warp may early-return for an inactive bucket row.
// =============================================================================

__device__ __forceinline__ uint64_t pack_rank_idx(uint32_t rank, uint32_t idx) {
    return (static_cast<uint64_t>(rank) << 32) | static_cast<uint64_t>(idx);
}

// Find the index of the next-alive slot strictly greater than `from`, or
// 0xFFFFFFFFu if none. `a0`/`a1` are the 32-bit ballot masks for rows 0/1.
// Branchless-ish; safe to call from all lanes with identical args.
__device__ __forceinline__ uint32_t next_alive_after(uint32_t a0, uint32_t a1, uint32_t from)
{
    if (from < WARP_SIZE - 1) {
        // bits > from in row 0
        uint32_t m0 = a0 & ~((1u << (from + 1)) - 1u);
        if (m0) return __ffs(m0) - 1;
        if (a1) return WARP_SIZE + (__ffs(a1) - 1);
        return 0xFFFFFFFFu;
    }
    if (from == WARP_SIZE - 1) {
        if (a1) return WARP_SIZE + (__ffs(a1) - 1);
        return 0xFFFFFFFFu;
    }
    uint32_t rel = from - WARP_SIZE;
    if (rel >= WARP_SIZE - 1) return 0xFFFFFFFFu;
    uint32_t m1 = a1 & ~((1u << (rel + 1)) - 1u);
    if (m1) return WARP_SIZE + (__ffs(m1) - 1);
    return 0xFFFFFFFFu;
}

// Find the index of the previous-alive slot strictly less than `from`, or
// 0xFFFFFFFFu if none.
__device__ __forceinline__ uint32_t prev_alive_before(uint32_t a0, uint32_t a1, uint32_t from)
{
    if (from >= WARP_SIZE) {
        uint32_t rel = from - WARP_SIZE;
        uint32_t m1 = (rel > 0) ? (a1 & ((1u << rel) - 1u)) : 0u;
        if (m1) return WARP_SIZE + (31 - __clz(m1));
        if (a0) return 31 - __clz(a0);
        return 0xFFFFFFFFu;
    }
    uint32_t m0 = (from > 0) ? (a0 & ((1u << from) - 1u)) : 0u;
    if (m0) return 31 - __clz(m0);
    return 0xFFFFFFFFu;
}

// Shuffle the token at `slot` from its owning lane to every lane.
// Both rows are shuffled unconditionally (so all 32 lanes participate in both
// __shfl_sync calls); we then select based on which row contains `slot`.
__device__ __forceinline__ uint32_t shfl_slot(uint32_t parts0, uint32_t parts1, uint32_t slot)
{
    uint32_t lane = slot & (WARP_SIZE - 1);
    uint32_t v0 = __shfl_sync(0xFFFFFFFFu, parts0, lane);
    uint32_t v1 = __shfl_sync(0xFFFFFFFFu, parts1, lane);
    return (slot < WARP_SIZE) ? v0 : v1;
}

// Minimum of one uint32 across the warp: a single redux instruction on
// sm_80+, a shuffle tree elsewhere.
__device__ __forceinline__ uint32_t warp_min_u32(uint32_t v)
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    return __reduce_min_sync(0xFFFFFFFFu, v);
#else
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        const uint32_t o = __shfl_xor_sync(0xFFFFFFFFu, v, off);
        v = o < v ? o : v;
    }
    return v;
#endif
}

// The token at `slot` broadcast to every lane. With one slot per lane there
// is no second row to shuffle.
template<int SLOTS>
__device__ __forceinline__ uint32_t slot_value(uint32_t parts0, uint32_t parts1, uint32_t slot)
{
    if constexpr (SLOTS == 1) {
        return __shfl_sync(0xFFFFFFFFu, parts0, slot & (WARP_SIZE - 1));
    } else {
        return shfl_slot(parts0, parts1, slot);
    }
}

// kFromIds == false: inputs are raw bytes; translate via byte_to_id.
// kFromIds == true : inputs are uint32_t initial token ids; read directly.
template<int LEN, typename Slot, bool kFromIds = false>
__global__ __launch_bounds__(BLOCK_THREADS)
void bpe_kernel(
    const void*     __restrict__ pretoken_input,   // uint8_t* or uint32_t*
    const uint16_t* __restrict__ pretoken_lens,    // [n_pretokens]
    const uint32_t* __restrict__ orig_idx,         // [n_pretokens]
    uint32_t                    n_pretokens,
    const uint32_t* __restrict__ byte_to_id,
    typename Slot::Args         slot_args,
    DirectTokenArgs             direct_args,
    uint32_t* __restrict__      out_tokens,
    uint32_t* __restrict__      out_count,
    const uint32_t* __restrict__ n_active_ptr = nullptr)
{
    static_assert(LEN == MAX_PRETOKEN_LEN_SHORT || LEN == MAX_PRETOKEN_LEN_LONG,
                  "bpe_kernel only instantiated for the two bucket sizes");
    constexpr int SLOTS = LEN / WARP_SIZE;
    static_assert(SLOTS == 1 || SLOTS == 2, "SLOTS must be 1 or 2");

    const uint32_t warp_id = threadIdx.x >> 5;
    const uint32_t tid = threadIdx.x & (WARP_SIZE - 1);
    uint32_t n_active = n_pretokens;
    if (n_active_ptr != nullptr) {
        n_active = min(n_active, *n_active_ptr);
    }
    const uint32_t warp_stride = gridDim.x * WARPS_PER_BLOCK;
    for (uint32_t pre_idx = blockIdx.x * WARPS_PER_BLOCK + warp_id;
         pre_idx < n_active;
         pre_idx += warp_stride) {
        const uint32_t N = pretoken_lens[pre_idx];
        if (N == 0) {
            // Tail block (pretoken_lens was zeroed by encode() past
            // n_short/n_long). d_per_pre_count is also pre-zeroed past
            // n_total, so no write needed — and writing through
            // orig_idx[pre_idx] would be invalid here because orig_idx's tail
            // past n_short/n_long is uninitialized garbage. Skip.
            continue;
        }
        const uint32_t out_idx = orig_idx[pre_idx];

        // HF's BPE `ignore_merges` is semantically observable: if the complete
        // ByteLevel pre-token is itself a vocabulary item, emit it directly rather
        // than deriving it through pair merges. SP inputs are initial IDs instead
        // of raw bytes, so they intentionally bypass this lookup.
        if constexpr (!kFromIds) {
            uint32_t direct_id = TOKEN_DEAD;
            if (tid == 0 && direct_args.ids != nullptr) {
                const uint8_t* raw = static_cast<const uint8_t*>(pretoken_input) +
                                     pre_idx * static_cast<size_t>(LEN);
                direct_id = direct_token_lookup(direct_args, raw, N);
            }
            direct_id = __shfl_sync(0xFFFFFFFFu, direct_id, 0);
            if (direct_id != TOKEN_DEAD) {
                if (tid == 0) {
                    out_tokens[pre_idx * static_cast<size_t>(LEN)] = direct_id;
                    out_count[out_idx] = 1;
                }
                continue;
            }
        }

        const uint32_t my_slot0 = tid;
        const uint32_t my_slot1 = tid + WARP_SIZE;  // unused when SLOTS == 1

        // ---- Step 1: initial parts[]. Read from either uint8 byte rows (byte-
        // level family, translate via byte_to_id) or uint32 id rows (SP family,
        // direct read).
        uint32_t p0;
        uint32_t p1 = TOKEN_DEAD;
        if constexpr (kFromIds) {
            const uint32_t* my_row =
                static_cast<const uint32_t*>(pretoken_input) + pre_idx * static_cast<size_t>(LEN);
            p0 = (my_slot0 < N) ? my_row[my_slot0] : TOKEN_DEAD;
            if constexpr (SLOTS == 2) {
                p1 = (my_slot1 < N) ? my_row[my_slot1] : TOKEN_DEAD;
            }
        } else {
            const uint8_t* my_row =
                static_cast<const uint8_t*>(pretoken_input) + pre_idx * static_cast<size_t>(LEN);
            p0 = (my_slot0 < N) ? byte_to_id[my_row[my_slot0]] : TOKEN_DEAD;
            if constexpr (SLOTS == 2) {
                p1 = (my_slot1 < N) ? byte_to_id[my_row[my_slot1]] : TOKEN_DEAD;
            }
        }

        // ---- Step 2: initial alive masks.
        uint32_t a0 = __ballot_sync(0xFFFFFFFFu, p0 != TOKEN_DEAD);
        uint32_t a1 = 0u;
        if constexpr (SLOTS == 2) {
            a1 = __ballot_sync(0xFFFFFFFFu, p1 != TOKEN_DEAD);
        }

        // ---- Step 3: initial ranks. Each slot also keeps the id its pair
        // would merge into (n0/n1), from the same probe: the merge step then
        // reads the merged id from a register instead of probing again, so a
        // merge iteration waits on one table lookup instead of two in series.
        // A slot's cached id is refreshed whenever its pair changes (it is the
        // merged slot or its left neighbour), so it always equals what a fresh
        // probe of its current pair would return.
        uint32_t next0 = next_alive_after(a0, a1, my_slot0);
        uint32_t safe_next0 = (next0 == 0xFFFFFFFFu) ? 0u : next0;
        uint32_t right0 = slot_value<SLOTS>(p0, p1, safe_next0);
        uint32_t r0 = RANK_NONE, n0 = TOKEN_DEAD;
        if (p0 != TOKEN_DEAD && next0 != 0xFFFFFFFFu) {
            const ProbeResult pr = Slot::probe(slot_args, p0, right0);
            r0 = pr.rank;
            n0 = pr.new_id;
        }
        uint32_t r1 = RANK_NONE, n1 = TOKEN_DEAD;
        if constexpr (SLOTS == 2) {
            uint32_t next1 = next_alive_after(a0, a1, my_slot1);
            uint32_t safe_next1 = (next1 == 0xFFFFFFFFu) ? 0u : next1;
            uint32_t right1 = slot_value<SLOTS>(p0, p1, safe_next1);
            if (p1 != TOKEN_DEAD && next1 != 0xFFFFFFFFu) {
                const ProbeResult pr = Slot::probe(slot_args, p1, right1);
                r1 = pr.rank;
                n1 = pr.new_id;
            }
        }

// ---- Main merge loop. Bound = LEN - 1.
#pragma unroll 1
        for (int iter = 0; iter < LEN - 1; ++iter) {
            // Argmin over (rank, slot): the lowest rank, ties to the lowest
            // slot. Ranks are below 2^26 (checked when the table is built)
            // and slots below 64, so rank<<6|slot orders exactly like the
            // (rank, slot) pair and one warp reduction finds it.
            uint32_t key = (r0 == RANK_NONE) ? 0xFFFFFFFFu : ((r0 << 6) | my_slot0);
            if constexpr (SLOTS == 2) {
                const uint32_t key1 =
                    (r1 == RANK_NONE) ? 0xFFFFFFFFu : ((r1 << 6) | my_slot1);
                key = key < key1 ? key : key1;
            }
            key = warp_min_u32(key);
            if (key == 0xFFFFFFFFu) break;
            uint32_t min_slot = key & 63u;

            uint32_t right_slot = next_alive_after(a0, a1, min_slot);
            uint32_t prev_slot = prev_alive_before(a0, a1, min_slot);

            uint32_t new_tok = slot_value<SLOTS>(n0, n1, min_slot);

            if (my_slot0 == min_slot) p0 = new_tok;
            if (my_slot0 == right_slot) p0 = TOKEN_DEAD;
            if constexpr (SLOTS == 2) {
                if (my_slot1 == min_slot) p1 = new_tok;
                if (my_slot1 == right_slot) p1 = TOKEN_DEAD;
            }

            if (right_slot < WARP_SIZE) {
                a0 &= ~(1u << right_slot);
            } else {
                if constexpr (SLOTS == 2) {
                    a1 &= ~(1u << (right_slot - WARP_SIZE));
                }
                // SLOTS==1: right_slot can never be >= WARP_SIZE (next_alive_after
                // with a1==0 only returns indices in [0,32) or 0xFFFFFFFF).
            }

            uint32_t next_after = next_alive_after(a0, a1, min_slot);
            uint32_t right2 = slot_value<SLOTS>(p0, p1, (next_after == 0xFFFFFFFFu) ? 0u : next_after);
            uint32_t prev_tok = slot_value<SLOTS>(p0, p1, (prev_slot == 0xFFFFFFFFu) ? 0u : prev_slot);
            // Both lookups are issued before either is examined, so their
            // loads are in flight together. A missing neighbour's lookup is
            // still issued (its table loads are in bounds) and discarded.
            typename Slot::Pending pend_min, pend_prev;
            Slot::issue(slot_args, new_tok, right2, pend_min);
            Slot::issue(slot_args, prev_tok, new_tok, pend_prev);
            ProbeResult at_min{RANK_NONE, TOKEN_DEAD};
            if (next_after != 0xFFFFFFFFu) {
                at_min = Slot::finish(slot_args, pend_min);
            }
            ProbeResult at_prev{RANK_NONE, TOKEN_DEAD};
            if (prev_slot != 0xFFFFFFFFu) {
                at_prev = Slot::finish(slot_args, pend_prev);
            }

            if (my_slot0 == min_slot) { r0 = at_min.rank; n0 = at_min.new_id; }
            if (my_slot0 == right_slot) r0 = RANK_NONE;
            if (prev_slot != 0xFFFFFFFFu && my_slot0 == prev_slot) {
                r0 = at_prev.rank;
                n0 = at_prev.new_id;
            }
            if constexpr (SLOTS == 2) {
                if (my_slot1 == min_slot) { r1 = at_min.rank; n1 = at_min.new_id; }
                if (my_slot1 == right_slot) r1 = RANK_NONE;
                if (prev_slot != 0xFFFFFFFFu && my_slot1 == prev_slot) {
                    r1 = at_prev.rank;
                    n1 = at_prev.new_id;
                }
            }
        }

        // ---- Compaction. Outputs go to this bucket's local row `pre_idx` at the
        // bucket's own stride (LEN: 32 short / 64 long). flatten_kernel_bucketed
        // maps pre_idx -> orig_idx for the count/scan/destination lookups, so the
        // short bucket needs only a 32-wide row. The count is still written by
        // orig_idx below (out_count[out_idx]) so the orig_idx-keyed scan is intact.
        uint32_t cnt;
        if constexpr (SLOTS == 1) {
            cnt = __popc(a0);
        } else {
            cnt = __popc(a0) + __popc(a1);
        }
        uint32_t* out_row = out_tokens + pre_idx * static_cast<size_t>(LEN);

        if (p0 != TOKEN_DEAD && my_slot0 < N) {
            uint32_t below = __popc(a0 & ((1u << my_slot0) - 1u));
            out_row[below] = p0;
        }
        if constexpr (SLOTS == 2) {
            if (p1 != TOKEN_DEAD && my_slot1 < N) {
                uint32_t rel = my_slot1 - WARP_SIZE;
                uint32_t below = __popc(a0) + (rel > 0 ? __popc(a1 & ((1u << rel) - 1u)) : 0u);
                out_row[below] = p1;
            }
        }

        if (tid == 0) out_count[out_idx] = cnt;
    }
}

#if GBPE_GPU_PRETOK
// =============================================================================
//  Generic overflow BPE: one warp per >64-element pre-token.
//
//  The fast 32/64 kernels keep all state in registers. This correctness path
//  stores state in global memory and maintains the exact (rank,index) minimum
//  in a power-of-two tournament tree. Initial construction is O(L/warp); each
//  exact greedy merge updates at most three leaf-to-root paths in O(log L).
// =============================================================================
// Overflow pre-tokens are rare (well under 1% of pre-tokens) but each one is
// a long serial merge chain, so on multilingual text this kernel dominated
// the graph when only 64 warps serviced ~900 of them. Spread them over every
// SM: independent warps per block, enough blocks to cover an H100. Three
// warps per block (not four) so the shared-memory working set below stays
// inside the 48 KiB static limit; 264 blocks x 3 = 792 warps still vastly
// exceeds the measured overflow count (<= ~800 at 1 MiB), so this does not
// constrain parallelism.
constexpr uint32_t OVERFLOW_BPE_WARPS_PER_BLOCK = 3;
constexpr uint32_t OVERFLOW_BPE_BLOCKS = 264;
constexpr uint32_t OVERFLOW_BPE_THREADS = WARP_SIZE * OVERFLOW_BPE_WARPS_PER_BLOCK;

// The merge loop's cost is the LATENCY of one warp's dependent chain, not
// throughput: only lane 0 advances the list, and each iteration walks ~7
// dependent global levels. Staging the per-pre-token working set in shared
// memory cuts those levels to shared-memory latency. The merge table probe
// still lives in global memory, so this shortens the chain rather than
// removing it.
//
// Bound chosen from the measured pre-token length distribution: after the
// regex pre-tokenizer, the longest overflow pre-token is 192 B (llama3/qwen2.5
// /GPT-2) and 488 B (DeepSeek-V3) across code/multilingual/books at up to
// 1 MiB. 512 covers every one of them at the benchmarked sizes; the rare
// longer pre-token goes to the long-entry blocks (overflow_long_entry).
//
// Budget per warp: tree 2*512*8 B = 8 KiB, parts/prev/next 512*4 B each =
// 6 KiB, total 14 KiB. At 3 warps/block that is 42 KiB, inside the 48 KiB
// static limit (5 blocks/SM on sm_90), so no dynamic-shared opt-in is needed.
constexpr uint32_t OVERFLOW_SMEM_MAX_LEN = 512;

template<typename TreeT>
__device__ __forceinline__ void overflow_rebuild_paths(
    TreeT* tree, uint32_t leaf_count, uint32_t lane,
    uint32_t changed0, uint32_t changed1, uint32_t changed2)
{
    __syncwarp();  // make the changed leaves visible before rebuilding
    changed0 = __shfl_sync(0xFFFFFFFFu, changed0, 0);
    changed1 = __shfl_sync(0xFFFFFFFFu, changed1, 0);
    changed2 = __shfl_sync(0xFFFFFFFFu, changed2, 0);
    const uint32_t changed = lane == 0 ? changed0
                           : lane == 1 ? changed1
                           : lane == 2 ? changed2 : 0xFFFFFFFFu;
    uint32_t node = changed == 0xFFFFFFFFu
                  ? 0xFFFFFFFFu : leaf_count + changed;
    for (uint32_t width = leaf_count; width > 1; width >>= 1) {
        const uint32_t parent = node == 0xFFFFFFFFu
                              ? 0xFFFFFFFFu : node >> 1;
        const uint32_t parent0 = __shfl_sync(0xFFFFFFFFu, parent, 0);
        const uint32_t parent1 = __shfl_sync(0xFFFFFFFFu, parent, 1);
        const bool owns_parent = lane < 3 && parent != 0xFFFFFFFFu &&
            (lane == 0 || parent != parent0) &&
            (lane < 2 || parent != parent1);
        if (owns_parent) {
            const uint64_t a = tree[parent << 1];
            const uint64_t b = tree[(parent << 1) + 1];
            tree[parent] = (a < b) ? a : b;
        }
        __syncwarp();
        node = parent;
    }
}

// The exact greedy merge loop, parameterised only on WHERE the working arrays
// live. Shared and global paths instantiate the identical code with identical
// indices; nothing about the merge order, tie-breaking or probe sequence
// depends on the storage class. `parts`/`prev`/`next`/`tree` are pre-token
// local (index 0 == first element), so the caller supplies either
// `smem + warp_slot` or `global + base`.
template<typename Slot, typename PartsT, typename LinkT, typename TreeT>
__device__ __forceinline__ void overflow_merge_loop(
    PartsT* parts, LinkT* prev, LinkT* next, TreeT* tree,
    uint32_t len, uint32_t leaf_count, uint32_t lane,
    typename Slot::Args slot_args, uint32_t* __restrict__ error_count)
{
    // Initial adjacent-pair ranks become the tree leaves. Padding leaves
    // carry RANK_NONE so they can never win a real reduction.
    for (uint32_t i = lane; i < leaf_count; i += WARP_SIZE) {
        uint32_t rank = RANK_NONE;
        if (i + 1 < len) {
            rank = Slot::probe(slot_args, parts[i], parts[i + 1]).rank;
        }
        tree[leaf_count + i] = pack_rank_idx(rank, i);
    }
    __syncwarp();

    // Build one complete-tree level at a time. Every child level is fully
    // visible before any lane consumes it to build the next parent level.
    for (uint32_t width = leaf_count >> 1; width > 0; width >>= 1) {
        for (uint32_t i = lane; i < width; i += WARP_SIZE) {
            uint32_t node = width + i;
            TreeT a = tree[node << 1];
            TreeT b = tree[(node << 1) + 1];
            tree[node] = (a < b) ? a : b;
        }
        __syncwarp();
    }

    // The root is the exact global argmin. A merge changes only the leaf
    // at the merged-left position, the killed-right position, and the
    // predecessor position; three lanes rebuild those paths in parallel
    // and deduplicate them as they converge.
    for (uint32_t iter = 0; iter + 1 < len; ++iter) {
        uint64_t best = (lane == 0) ? tree[1] : 0;
        best = __shfl_sync(0xFFFFFFFFu, best, 0);
        const uint32_t min_rank = static_cast<uint32_t>(best >> 32);
        if (min_rank == RANK_NONE) break;
        const uint32_t left = static_cast<uint32_t>(best);

        uint32_t changed0 = 0xFFFFFFFFu;
        uint32_t changed1 = 0xFFFFFFFFu;
        uint32_t changed2 = 0xFFFFFFFFu;

        // All list mutation and both rank probes stay on lane 0, exactly as
        // in the global-memory original. Splitting the `after` and `before`
        // probes across lanes 0 and 1 was measured twice -- once as a
        // divergent branch, once as a select-driven uniform probe -- and both
        // were slower (-1.3% and -7% of overflow cost). The `before` probe is
        // not on the critical path: it has no consumer until the rebuild, so
        // lane 0 already overlaps it, and paying a cross-lane select plus an
        // extra warp's worth of loads only lengthens the dependent chain.
        if (lane == 0) {
            const uint32_t right  = next[left];
            const uint32_t before = prev[left];
            if (right == 0xFFFFFFFFu || parts[right] == TOKEN_DEAD) {
                // A valid tree must never nominate a dead/no-right leaf.
                // Preserve memory safety for this replay, but surface the
                // non-bit-exact state to the host instead of silently
                // emitting a shifted suffix.
                if (error_count != nullptr) atomicAdd(error_count, 1u);
                tree[leaf_count + left] = pack_rank_idx(RANK_NONE, left);
                changed0 = left;
            } else {
                const uint32_t after     = next[right];
                const uint32_t left_id   = parts[left];
                const uint32_t right_id  = parts[right];
                const uint32_t new_id =
                    Slot::probe(slot_args, left_id, right_id).new_id;

                parts[left]  = new_id;
                parts[right] = TOKEN_DEAD;
                next[left] = after;
                if (after != 0xFFFFFFFFu) prev[after] = left;

                uint32_t left_rank = (after == 0xFFFFFFFFu)
                    ? RANK_NONE
                    : Slot::probe(slot_args, new_id, parts[after]).rank;
                tree[leaf_count + left] = pack_rank_idx(left_rank, left);
                tree[leaf_count + right] = pack_rank_idx(RANK_NONE, right);

                if (before != 0xFFFFFFFFu) {
                    uint32_t before_rank =
                        Slot::probe(slot_args, parts[before], new_id).rank;
                    tree[leaf_count + before] = pack_rank_idx(before_rank, before);
                }
                changed0 = left;
                changed1 = right;
                changed2 = before;
            }
        }
        overflow_rebuild_paths(tree, leaf_count, lane, changed0, changed1, changed2);
    }
}

// =============================================================================
//  Long overflow pre-tokens (> OVERFLOW_SMEM_MAX_LEN): block-cooperative,
//  round-batched exact BPE.
//
//  The serial loop above does one merge per iteration; on a >512-element
//  pre-token its tree lives in global memory and each merge costs ~3 us, so a
//  single 10,000-character run of one repeated character (one 10 KB
//  pre-token) took ~28 ms. Such pre-tokens are runs of spaces, '=', '-', '.',
//  letters or CJK, and they are highly repetitive: sequential BPE applies the
//  same merge to many places.
//
//  One round: r = the minimum pair rank. Sequential BPE, processing (rank,
//  index) in order, merges the rank-r pairs left to right; within a maximal
//  chain of consecutive rank-r pairs it takes every other pair. That is exact
//  as long as no pair created along the way ranks <= r (it would be merged
//  first). Each selected merge is checked against the neighbours it has at
//  the moment sequential BPE applies it (the result of the selected pair two
//  to the left, or the original left symbol; the original right symbol); the
//  round applies the selected merges up to and including the first one that
//  fails the check -- exactly the sequential steps -- and the next round
//  restarts from the true minimum. Runs of one character finish in ~3-7
//  rounds instead of ~n serial merges.
//
//  Rounds cost O(n / threads) each, so on non-repetitive text (about one
//  merge per round) they are no better than the serial loop. A round that
//  applies fewer than max(2, n/1024) merges hands the compacted remainder to
//  the unchanged serial loop (warp 0; shared-memory path once n <= 512).
//
//  These run on OVERFLOW_LONG_BLOCKS extra blocks of the same launch (no new
//  graph node). The warp blocks skip long entries; long block b takes the
//  long entries whose overflow index is b mod OVERFLOW_LONG_BLOCKS. Every
//  trip count below is block-uniform, so the __syncthreads() are safe.
//
//  Scratch (all disjoint per pre-token, no new allocation): symbols
//  double-buffer between parts+base and prev+base, pair ranks live in
//  next+base, and the selected merge results in the first 4*len bytes of the
//  pre-token's tree region.
// =============================================================================
constexpr uint32_t OVERFLOW_LONG_BLOCKS = 128;
constexpr uint32_t OVERFLOW_BPE_GRID = OVERFLOW_BPE_BLOCKS + OVERFLOW_LONG_BLOCKS;
constexpr uint32_t OVERFLOW_SEG_BREAK = 0x80000000u;
constexpr uint32_t OVERFLOW_LONG_MAX_UNPRODUCTIVE = 4;
constexpr uint32_t OVERFLOW_RANK_STALE = 0xFFFFFFFEu;   // never a real rank

// Segmented count of trailing rank-r pairs: a value is either a plain count
// (no break in the segment) or SEG_BREAK | count-after-the-last-break.
__device__ __forceinline__ uint32_t overflow_seg_op(uint32_t a, uint32_t b) {
    return (b & OVERFLOW_SEG_BREAK)
        ? b : ((a & OVERFLOW_SEG_BREAK) | ((a & ~OVERFLOW_SEG_BREAK) + b));
}
__device__ __forceinline__ uint32_t overflow_add_op(uint32_t a, uint32_t b) {
    return a + b;
}

__device__ __forceinline__ uint32_t overflow_block_min(uint32_t v, uint32_t* s_warp) {
    const uint32_t lane = threadIdx.x & (WARP_SIZE - 1);
    const uint32_t w = threadIdx.x / WARP_SIZE;
    #pragma unroll
    for (uint32_t d = 16; d > 0; d >>= 1) v = min(v, __shfl_xor_sync(0xFFFFFFFFu, v, d));
    if (lane == 0) s_warp[w] = v;
    __syncthreads();
    uint32_t m = s_warp[0];
    #pragma unroll
    for (uint32_t k = 1; k < OVERFLOW_BPE_WARPS_PER_BLOCK; ++k) m = min(m, s_warp[k]);
    __syncthreads();
    return m;
}

// Block-wide exclusive scan (associative, order-preserving `op`); also
// returns the block total.
template<typename Op>
__device__ __forceinline__ uint32_t overflow_block_scan(
    uint32_t v, Op op, uint32_t* s_warp, uint32_t* total)
{
    const uint32_t lane = threadIdx.x & (WARP_SIZE - 1);
    const uint32_t w = threadIdx.x / WARP_SIZE;
    uint32_t inc = v;
    #pragma unroll
    for (uint32_t d = 1; d < WARP_SIZE; d <<= 1) {
        const uint32_t t = __shfl_up_sync(0xFFFFFFFFu, inc, d);
        if (lane >= d) inc = op(t, inc);
    }
    uint32_t excl = __shfl_up_sync(0xFFFFFFFFu, inc, 1);
    if (lane == WARP_SIZE - 1) s_warp[w] = inc;
    __syncthreads();
    uint32_t prefix = 0, all = 0;   // 0 is the identity of both ops
    #pragma unroll
    for (uint32_t k = 0; k < OVERFLOW_BPE_WARPS_PER_BLOCK; ++k) {
        const uint32_t x = s_warp[k];
        if (k < w) prefix = op(prefix, x);
        all = op(all, x);
    }
    __syncthreads();
    *total = all;
    return lane == 0 ? prefix : op(prefix, excl);
}

// Serial exact BPE of `len` ids read from `src` (warp-wide call), the same
// shared/global paths the warp blocks use. Writes the compacted result to
// parts[base..) and returns its length (valid on lane 0). `src` may alias
// parts+base or prev+base: each lane reads src[i] before it writes index i.
template<typename Slot>
__device__ __forceinline__ uint32_t overflow_serial_from_ids(
    const uint32_t* src, uint32_t base, uint32_t len, uint32_t lane,
    uint32_t* s_parts, uint32_t* s_prev, uint32_t* s_next, uint64_t* s_tree,
    uint32_t* parts, uint32_t* prev, uint32_t* next, uint64_t* tree_storage,
    typename Slot::Args slot_args, uint32_t* error_count)
{
    uint32_t leaf_count = 1;
    while (leaf_count < len) leaf_count <<= 1;
    uint32_t count = 0;
    if (len <= OVERFLOW_SMEM_MAX_LEN) {
        for (uint32_t i = lane; i < len; i += WARP_SIZE) {
            s_parts[i] = src[i];
            s_prev[i] = (i == 0) ? 0xFFFFFFFFu : i - 1;
            s_next[i] = (i + 1 < len) ? i + 1 : 0xFFFFFFFFu;
        }
        __syncwarp();
        overflow_merge_loop<Slot>(s_parts, s_prev, s_next, s_tree,
                                  len, leaf_count, lane, slot_args, error_count);
        if (lane == 0) {
            for (uint32_t i = 0; i < len; ++i) {
                uint32_t token = s_parts[i];
                if (token != TOKEN_DEAD) parts[base + count++] = token;
            }
        }
    } else {
        // 2*leaf_count < 4*len, so assigning four uint64 slots per original
        // input position gives every disjoint pre-token a disjoint tree
        // region without a runtime allocator or another prefix scan.
        uint64_t* tree = tree_storage + static_cast<size_t>(base) * 4;
        // Links are local indices within this descriptor; UINT32_MAX ends.
        for (uint32_t i = lane; i < len; i += WARP_SIZE) {
            const uint32_t id = src[i];
            parts[base + i] = id;
            prev[base + i] = (i == 0) ? 0xFFFFFFFFu : i - 1;
            next[base + i] = (i + 1 < len) ? i + 1 : 0xFFFFFFFFu;
        }
        __syncwarp();
        overflow_merge_loop<Slot>(parts + base, prev + base, next + base,
                                  tree, len, leaf_count, lane, slot_args,
                                  error_count);
        // Serial compaction is safe in place: destination <= source index.
        if (lane == 0) {
            for (uint32_t i = 0; i < len; ++i) {
                uint32_t token = parts[base + i];
                if (token != TOKEN_DEAD) parts[base + count++] = token;
            }
        }
    }
    __syncwarp();
    return count;
}

// One long pre-token, whole block. Returns with the block converged.
template<typename Slot, bool kFromIds>
__device__ __forceinline__ void overflow_long_entry(
    const void* __restrict__ input, uint32_t base, uint32_t len, uint32_t orig,
    const uint32_t* __restrict__ byte_to_id,
    typename Slot::Args slot_args, DirectTokenArgs direct_args,
    uint32_t* __restrict__ parts, uint32_t* __restrict__ prev,
    uint32_t* __restrict__ next, uint64_t* __restrict__ tree_storage,
    uint32_t* __restrict__ out_count, uint32_t* __restrict__ error_count,
    uint32_t* s_parts, uint32_t* s_prev, uint32_t* s_next, uint64_t* s_tree,
    uint32_t* s_warp, uint32_t* s_bcast)
{
    constexpr uint32_t T = OVERFLOW_BPE_THREADS;
    const uint32_t tid = threadIdx.x;

    if constexpr (!kFromIds) {
        // Exact ignore_merges behavior, as on the warp path.
        if (tid == 0) {
            *s_bcast = (direct_args.ids != nullptr)
                ? direct_token_lookup(direct_args,
                      static_cast<const uint8_t*>(input) + base, len)
                : TOKEN_DEAD;
        }
        __syncthreads();
        const uint32_t direct_id = *s_bcast;
        __syncthreads();
        if (direct_id != TOKEN_DEAD) {
            if (tid == 0) { parts[base] = direct_id; out_count[orig] = 1; }
            return;
        }
    }

    // Pair i's rank and merge result stay valid across rounds unless one
    // of its two symbols changed; only those pairs are probed again.
    uint32_t* scratch = reinterpret_cast<uint32_t*>(tree_storage + static_cast<size_t>(base) * 4);
    uint32_t* cur = parts + base;          // symbols (double-buffered)
    uint32_t* alt = prev + base;
    uint32_t* rk  = next + base;           // pair ranks, OVERFLOW_RANK_STALE = re-probe
    uint32_t* rk_alt = scratch + len;
    uint32_t* id  = scratch + 2 * len;     // pair merge results
    uint32_t* id_alt = scratch + 3 * len;
    uint32_t* nid = scratch;               // this round: result if selected, else TOKEN_DEAD
    for (uint32_t i = tid; i < len; i += T) {
        if constexpr (kFromIds) {
            cur[i] = static_cast<const uint32_t*>(input)[base + i];
        } else {
            cur[i] = byte_to_id[static_cast<const uint8_t*>(input)[base + i]];
        }
        rk[i] = OVERFLOW_RANK_STALE;
    }
    __syncthreads();

    uint32_t n = len;
    bool finished = false;
    uint32_t unproductive = 0;
    for (;;) {
        const uint32_t chunk = (n + T - 1) / T;
        const uint32_t lo = min(tid * chunk, n);
        const uint32_t hi = min(lo + chunk, n);
        const uint32_t phi = min(hi, n - 1);          // pairs [lo, phi)

        // Every pass walks the thread's contiguous segment in batches of B:
        // the batch's loads (and probes) are issued before any is consumed,
        // so a batch waits on one memory latency instead of B in a row.
        constexpr uint32_t B = 8;

        // Pass 1: probe the stale pairs; r = the minimum pair rank.
        uint32_t local_min = RANK_NONE;
        for (uint32_t i0 = lo; i0 < phi; i0 += B) {
            uint32_t x[B];
            #pragma unroll
            for (uint32_t k = 0; k < B; ++k) x[k] = (i0 + k < phi) ? rk[i0 + k] : 0u;
            uint32_t sym[B + 1];
            #pragma unroll
            for (uint32_t k = 0; k <= B; ++k) {
                const bool need = (k < B && x[k] == OVERFLOW_RANK_STALE) ||
                                  (k > 0 && x[k - 1] == OVERFLOW_RANK_STALE);
                sym[k] = (need && i0 + k <= phi) ? cur[i0 + k] : 0u;
            }
            typename Slot::Pending pd[B];
            #pragma unroll
            for (uint32_t k = 0; k < B; ++k)
                if (i0 + k < phi && x[k] == OVERFLOW_RANK_STALE)
                    Slot::issue(slot_args, sym[k], sym[k + 1], pd[k]);
            #pragma unroll
            for (uint32_t k = 0; k < B; ++k) {
                if (i0 + k >= phi) continue;
                if (x[k] == OVERFLOW_RANK_STALE) {
                    const ProbeResult p = Slot::finish(slot_args, pd[k]);
                    x[k] = p.rank;
                    rk[i0 + k] = p.rank;
                    id[i0 + k] = p.new_id;
                }
                local_min = min(local_min, x[k]);
            }
        }
        const uint32_t r = overflow_block_min(local_min, s_warp);
        if (r == RANK_NONE) { finished = true; break; }

        // Pass 2: left-to-right greedy selection inside each chain of
        // consecutive rank-r pairs (even offsets). Unselected -> TOKEN_DEAD.
        uint32_t seg = 0;
        for (uint32_t i0 = lo; i0 < phi; i0 += B) {
            uint32_t x[B];
            #pragma unroll
            for (uint32_t k = 0; k < B; ++k) x[k] = (i0 + k < phi) ? rk[i0 + k] : 0u;
            #pragma unroll
            for (uint32_t k = 0; k < B; ++k)
                if (i0 + k < phi) seg = (x[k] == r) ? seg + 1u : OVERFLOW_SEG_BREAK;
        }
        uint32_t unused;
        uint32_t run = overflow_block_scan(seg, overflow_seg_op, s_warp, &unused)
                     & ~OVERFLOW_SEG_BREAK;
        for (uint32_t i0 = lo; i0 < phi; i0 += B) {
            uint32_t x[B], y[B];
            #pragma unroll
            for (uint32_t k = 0; k < B; ++k) {
                x[k] = (i0 + k < phi) ? rk[i0 + k] : 0u;
                y[k] = (i0 + k < phi) ? id[i0 + k] : 0u;
            }
            #pragma unroll
            for (uint32_t k = 0; k < B; ++k) {
                if (i0 + k >= phi) continue;
                if (x[k] == r) {
                    nid[i0 + k] = (run & 1u) ? TOKEN_DEAD : y[k];
                    ++run;
                } else {
                    nid[i0 + k] = TOKEN_DEAD;
                    run = 0;
                }
            }
        }
        __syncthreads();

        // Pass 3: first selected merge whose at-the-time neighbour pair
        // ranks <= r (sequential BPE would take that pair next): the left
        // neighbour is the result of the selected pair two to the left, else
        // the original left symbol; the right one is the original symbol.
        uint32_t local_v = 0xFFFFFFFFu;
        for (uint32_t i0 = lo; i0 < phi; i0 += B) {
            uint32_t m[B + 2];                         // nid[i0-2 .. i0+B)
            #pragma unroll
            for (uint32_t k = 0; k < B + 2; ++k) {
                const uint32_t idx = i0 + k - 2u;
                m[k] = (i0 + k >= 2u && idx < n - 1) ? nid[idx] : TOKEN_DEAD;
            }
            uint32_t cs[B + 3];                        // cur[i0-1 .. i0+B+2)
            #pragma unroll
            for (uint32_t k = 0; k < B + 3; ++k) {
                const uint32_t idx = i0 + k - 1u;
                cs[k] = (i0 + k >= 1u && idx < n) ? cur[idx] : 0u;
            }
            typename Slot::Pending pl[B], pr[B];
            bool hl[B], hr[B];
            #pragma unroll
            for (uint32_t k = 0; k < B; ++k) {
                const uint32_t i = i0 + k;
                const uint32_t mid = m[k + 2];
                const bool sel = i < phi && mid != TOKEN_DEAD;
                hl[k] = sel && i >= 1u;
                hr[k] = sel && i + 2u < n;
                if (hl[k]) Slot::issue(slot_args, m[k] != TOKEN_DEAD ? m[k] : cs[k],
                                       mid, pl[k]);
                if (hr[k]) Slot::issue(slot_args, mid, cs[k + 3], pr[k]);
            }
            #pragma unroll
            for (uint32_t k = 0; k < B; ++k) {
                const bool bad = (hl[k] && Slot::finish(slot_args, pl[k]).rank <= r) ||
                                 (hr[k] && Slot::finish(slot_args, pr[k]).rank <= r);
                if (bad) local_v = min(local_v, i0 + k);
            }
        }
        const uint32_t v = overflow_block_min(local_v, s_warp);

        // Pass 4: apply the selected merges at index <= v; compact into alt.
        // Element j dies iff pair j-1 was applied.
        uint32_t keep = 0;
        for (uint32_t j0 = lo; j0 < hi; j0 += B) {
            uint32_t q[B];
            #pragma unroll
            for (uint32_t k = 0; k < B; ++k) {
                const uint32_t j = j0 + k;
                q[k] = (j < hi && j >= 1u) ? nid[j - 1] : TOKEN_DEAD;
            }
            #pragma unroll
            for (uint32_t k = 0; k < B; ++k) {
                const uint32_t j = j0 + k;
                if (j < hi) keep += (q[k] != TOKEN_DEAD && j - 1 <= v) ? 0u : 1u;
            }
        }
        uint32_t total;
        uint32_t out = overflow_block_scan(keep, overflow_add_op, s_warp, &total);
        // A kept element's pair (to the next kept element) keeps its rank
        // iff neither element is a merge result of this round.
        for (uint32_t j0 = lo; j0 < hi; j0 += B) {
            uint32_t q[B], w[B + 1], c[B], x[B], y[B];
            #pragma unroll
            for (uint32_t k = 0; k < B; ++k) {
                const uint32_t j = j0 + k;
                q[k] = (j < hi && j >= 1u) ? nid[j - 1] : TOKEN_DEAD;
                c[k] = (j < hi) ? cur[j] : 0u;
                x[k] = (j < hi && j + 1 < n) ? rk[j] : OVERFLOW_RANK_STALE;
                y[k] = (j < hi && j + 1 < n) ? id[j] : 0u;
            }
            #pragma unroll
            for (uint32_t k = 0; k <= B; ++k) {
                const uint32_t j = j0 + k;
                w[k] = (j <= hi && j + 1 < n) ? nid[j] : TOKEN_DEAD;
            }
            #pragma unroll
            for (uint32_t k = 0; k < B; ++k) {
                const uint32_t j = j0 + k;
                if (j >= hi) continue;
                if (q[k] != TOKEN_DEAD && j - 1 <= v) continue;
                const bool merged      = w[k] != TOKEN_DEAD && j <= v;
                const bool next_merged = w[k + 1] != TOKEN_DEAD && j + 1 <= v;
                alt[out] = merged ? w[k] : c[k];
                const bool fresh = !merged && !next_merged;
                rk_alt[out] = fresh ? x[k] : OVERFLOW_RANK_STALE;
                if (fresh) id_alt[out] = y[k];
                ++out;
            }
        }
        __syncthreads();

        const uint32_t merges = n - total;
        const uint32_t n_before = n;
        n = total;
        uint32_t* t = cur; cur = alt; alt = t;
        t = rk; rk = rk_alt; rk_alt = t;
        t = id; id = id_alt; id_alt = t;
        // A round that merges little costs more than the serial loop would;
        // after a few, the remainder goes to it (bounds the cost on
        // non-repetitive text; a lone low-rank merge at the edge of a run,
        // e.g. " AAAA...", must not end the batching).
        if (merges < 2u || merges * 1024u < n_before) {
            if (++unproductive >= OVERFLOW_LONG_MAX_UNPRODUCTIVE) break;
        }
    }

    if (finished) {
        if (cur != parts + base) {
            for (uint32_t i = tid; i < n; i += T) parts[base + i] = cur[i];
        }
        if (tid == 0) out_count[orig] = n;
    } else if (tid < WARP_SIZE) {
        // Unproductive round: finish the remainder serially on warp 0.
        const uint32_t count = overflow_serial_from_ids<Slot>(
            cur, base, n, tid, s_parts, s_prev, s_next, s_tree,
            parts, prev, next, tree_storage, slot_args, error_count);
        if (tid == 0) out_count[orig] = count;
    }
    __syncthreads();
}

// The long-entry path's batched probes want far more registers than the warp
// path (46); uncapped, the kernel compiled to ~250 and the warp blocks lost
// residency (5 -> 2 blocks/SM). Cap at the shared-memory-bound occupancy.
template<typename Slot, bool kFromIds = false>
__global__ void __launch_bounds__(OVERFLOW_BPE_THREADS, 5) overflow_bpe_kernel(
    const void* __restrict__ input,              // raw uint8 bytes or flat uint32 ids
    const uint32_t* __restrict__ overflow_start,
    const uint32_t* __restrict__ overflow_len,
    const uint32_t* __restrict__ overflow_orig,
    const uint32_t* __restrict__ n_overflow,
    const uint32_t* __restrict__ byte_to_id,
    typename Slot::Args slot_args,
    DirectTokenArgs direct_args,
    uint32_t* __restrict__ parts,
    uint32_t* __restrict__ prev,
    uint32_t* __restrict__ next,
    uint64_t* __restrict__ tree_storage,
    uint32_t* __restrict__ out_count,
    uint32_t* __restrict__ error_count)
{
    // Per-warp working set. Each warp owns a disjoint slice and only ever
    // synchronises with __syncwarp(), so no block-wide barrier is needed (and
    // none may be added: warps take different grid-stride trip counts).
    __shared__ uint64_t smem_tree[OVERFLOW_BPE_WARPS_PER_BLOCK *
                                  2u * OVERFLOW_SMEM_MAX_LEN];
    __shared__ uint32_t smem_parts[OVERFLOW_BPE_WARPS_PER_BLOCK *
                                   OVERFLOW_SMEM_MAX_LEN];
    __shared__ uint32_t smem_prev[OVERFLOW_BPE_WARPS_PER_BLOCK *
                                  OVERFLOW_SMEM_MAX_LEN];
    __shared__ uint32_t smem_next[OVERFLOW_BPE_WARPS_PER_BLOCK *
                                  OVERFLOW_SMEM_MAX_LEN];

    const uint32_t lane = threadIdx.x & (WARP_SIZE - 1);
    const uint32_t warp_in_block = threadIdx.x / WARP_SIZE;

    if (blockIdx.x >= OVERFLOW_BPE_BLOCKS) {
        // Long-entry block (see overflow_long_entry). Chunks of T entries:
        // thread t looks at overflow index q + t*K + b; the block then walks
        // the long ones in order. All control flow is block-uniform.
        __shared__ uint32_t s_warp[OVERFLOW_BPE_WARPS_PER_BLOCK];
        __shared__ uint32_t s_mask[OVERFLOW_BPE_WARPS_PER_BLOCK];
        __shared__ uint32_t s_bcast;
        constexpr uint32_t K = OVERFLOW_LONG_BLOCKS;
        constexpr uint32_t T = OVERFLOW_BPE_THREADS;
        const uint32_t b = blockIdx.x - OVERFLOW_BPE_BLOCKS;
        const uint32_t n_ov = *n_overflow;
        for (uint32_t q = 0; q < n_ov; q += T * K) {
            const uint32_t oi = q + threadIdx.x * K + b;
            const bool is_long = oi < n_ov && overflow_len[oi] > OVERFLOW_SMEM_MAX_LEN;
            const uint32_t m = __ballot_sync(0xFFFFFFFFu, is_long);
            if (lane == 0) s_mask[warp_in_block] = m;
            __syncthreads();
            for (uint32_t w = 0; w < OVERFLOW_BPE_WARPS_PER_BLOCK; ++w) {
                uint32_t bits = s_mask[w];
                while (bits) {
                    const uint32_t j = w * WARP_SIZE + (__ffs(bits) - 1);
                    bits &= bits - 1;
                    const uint32_t e = q + j * K + b;
                    overflow_long_entry<Slot, kFromIds>(
                        input, overflow_start[e], overflow_len[e], overflow_orig[e],
                        byte_to_id, slot_args, direct_args,
                        parts, prev, next, tree_storage, out_count, error_count,
                        smem_parts, smem_prev, smem_next, smem_tree,
                        s_warp, &s_bcast);
                }
            }
            __syncthreads();   // before the next chunk rewrites s_mask
        }
        return;
    }

    const uint32_t global_warp = blockIdx.x * (blockDim.x / WARP_SIZE) + warp_in_block;
    const uint32_t warp_stride = OVERFLOW_BPE_BLOCKS * (blockDim.x / WARP_SIZE);
    for (uint32_t oi = global_warp; oi < *n_overflow; oi += warp_stride) {
        const uint32_t base = overflow_start[oi];
        const uint32_t len  = overflow_len[oi];
        if (len > OVERFLOW_SMEM_MAX_LEN) continue;   // a long-entry block owns it

        // Keep the exact ignore_merges behavior for direct-vocab pre-tokens
        // above the 64-element register buckets as well.
        if constexpr (!kFromIds) {
            uint32_t direct_id = TOKEN_DEAD;
            if (lane == 0 && direct_args.ids != nullptr) {
                const uint8_t* raw = static_cast<const uint8_t*>(input) + base;
                direct_id = direct_token_lookup(direct_args, raw, len);
            }
            direct_id = __shfl_sync(0xFFFFFFFFu, direct_id, 0);
            if (direct_id != TOKEN_DEAD) {
                if (lane == 0) {
                    parts[base] = direct_id;
                    out_count[overflow_orig[oi]] = 1;
                }
                __syncwarp();
                continue;
            }
        }
        uint32_t leaf_count = 1;
        while (leaf_count < len) leaf_count <<= 1;

        // Long entries were skipped above, so the working set fits the warp's
        // shared-memory slice.
        uint32_t count = 0;
        {
            uint32_t* s_parts = smem_parts + warp_in_block * OVERFLOW_SMEM_MAX_LEN;
            uint32_t* s_prev  = smem_prev  + warp_in_block * OVERFLOW_SMEM_MAX_LEN;
            uint32_t* s_next  = smem_next  + warp_in_block * OVERFLOW_SMEM_MAX_LEN;
            uint64_t* s_tree  = smem_tree  + warp_in_block * (2u * OVERFLOW_SMEM_MAX_LEN);

            for (uint32_t i = lane; i < len; i += WARP_SIZE) {
                if constexpr (kFromIds) {
                    s_parts[i] = static_cast<const uint32_t*>(input)[base + i];
                } else {
                    uint8_t b = static_cast<const uint8_t*>(input)[base + i];
                    s_parts[i] = byte_to_id[b];
                }
                s_prev[i] = (i == 0) ? 0xFFFFFFFFu : i - 1;
                s_next[i] = (i + 1 < len) ? i + 1 : 0xFFFFFFFFu;
            }
            __syncwarp();

            overflow_merge_loop<Slot>(s_parts, s_prev, s_next, s_tree,
                                      len, leaf_count, lane, slot_args,
                                      error_count);

            // Compaction writes through to global: flatten reads `parts`.
            if (lane == 0) {
                for (uint32_t i = 0; i < len; ++i) {
                    uint32_t token = s_parts[i];
                    if (token != TOKEN_DEAD) parts[base + count++] = token;
                }
            }
        }
        if (lane == 0) out_count[overflow_orig[oi]] = count;
        __syncwarp();
    }
}

__global__ void flatten_gpu_overflow_kernel(
    const uint32_t* __restrict__ parts,
    const uint32_t* __restrict__ overflow_start,
    const uint32_t* __restrict__ overflow_orig,
    const uint32_t* __restrict__ n_overflow,
    const uint32_t* __restrict__ per_pre_count,
    const uint32_t* __restrict__ scan_out,
    uint32_t* __restrict__ out_tokens)
{
    for (uint32_t oi = blockIdx.x; oi < *n_overflow; oi += gridDim.x) {
        uint32_t orig = overflow_orig[oi];
        uint32_t count = per_pre_count[orig];
        uint32_t src = overflow_start[oi];
        uint32_t dst = scan_out[orig];
        for (uint32_t k = threadIdx.x; k < count; k += blockDim.x) {
            out_tokens[dst + k] = parts[src + k];
        }
    }
}
#endif


// =============================================================================
//  flatten_kernel_bucketed: gather per-pre-token rows into a contiguous output.
//
//  Templated on the bucket's row stride (STRIDE = 32 short / 64 long). Each
//  warp is a BUCKET-LOCAL index `i`; it maps i -> orig_idx via orig_idx_arr,
//  then reads cnt/dst from the orig_idx-keyed per_pre_count / scan_out arrays.
//  A grid-stride loop lets graph capture use a bounded grid while preserving
//  sequence-parallel processing across all resident warps.
//
//  TAIL GUARD: lens_arr[i] == 0 marks an unused (padded) bucket slot whose
//  orig_idx_arr[i] is uninitialized garbage; we must skip it BEFORE reading
//  orig_idx_arr[i]. (lens==0 is never a real pre-token — the regex emits no
//  zero-byte pre-tokens; this mirrors bpe_kernel's N==0 guard.)
// =============================================================================
template<int STRIDE>
__global__
void flatten_kernel_bucketed(
    const uint32_t* __restrict__ per_pre_tokens,   // bucket-local [cap_bucket, STRIDE]
    const uint32_t* __restrict__ orig_idx_arr,     // [cap_bucket] local -> global orig_idx
    const uint16_t* __restrict__ lens_arr,         // [cap_bucket] tail guard
    const uint32_t* __restrict__ per_pre_count,    // [cap_pretokens] keyed by orig_idx
    const uint32_t* __restrict__ scan_out,         // [cap_pretokens+1] keyed by orig_idx
    uint32_t* __restrict__       out_tokens,
    uint32_t                      capacity_rows,
    const uint32_t* __restrict__ n_active_ptr = nullptr)
{
    const uint32_t warp_id = threadIdx.x / WARP_SIZE;
    const uint32_t lane = threadIdx.x % WARP_SIZE;
    const uint32_t warps_per_block = blockDim.x / WARP_SIZE;
    uint32_t n_active = capacity_rows;
    if (n_active_ptr != nullptr) {
        n_active = min(n_active, *n_active_ptr);
    }
    const uint32_t warp_stride = gridDim.x * warps_per_block;
    for (uint32_t i = blockIdx.x * warps_per_block + warp_id;
         i < n_active;
         i += warp_stride) {
        if (n_active_ptr == nullptr && lens_arr[i] == 0) continue;
        const uint32_t glob     = orig_idx_arr[i];
        const uint32_t cnt      = per_pre_count[glob];
        const uint32_t dst_base = scan_out[glob];
        const uint32_t* src = per_pre_tokens + static_cast<size_t>(i) * STRIDE;
        for (uint32_t k = lane; k < cnt; k += WARP_SIZE) {
            out_tokens[dst_base + k] = src[k];
        }
    }
}

#if GBPE_HAVE_FAMILY_SP
// SP-family overflow flatten: 64-wide rows keyed by a dedicated local index.
// n_overflow is exact (no tail padding), passed as a launch-time bound.
__global__
void flatten_overflow_kernel(
    const uint32_t* __restrict__ per_pre_tokens,   // [cap_overflow, 64]
    const uint32_t* __restrict__ overflow_orig_idx,// [cap_overflow] local -> global
    const uint32_t* __restrict__ n_overflow_ptr,   // device scalar, set per encode()
    const uint32_t* __restrict__ per_pre_count,    // [cap_pretokens] keyed by orig_idx
    const uint32_t* __restrict__ scan_out,         // [cap_pretokens+1] keyed by orig_idx
    uint32_t* __restrict__       out_tokens)
{
    const uint32_t i = blockIdx.x;
    if (i >= *n_overflow_ptr) return;              // graph-safe dynamic bound
    const uint32_t glob     = overflow_orig_idx[i];
    const uint32_t cnt      = per_pre_count[glob];
    const uint32_t dst_base = scan_out[glob];
    const uint32_t* src = per_pre_tokens + static_cast<size_t>(i) * MAX_PRETOKEN_LEN_LONG;
    for (uint32_t k = threadIdx.x; k < cnt; k += blockDim.x) {
        out_tokens[dst_base + k] = src[k];
    }
}
#endif


// =============================================================================
//  GPU pre-tokenizer stage (GPT-2). Translates the proven byte-exact predicate
//  (the Python reference ref_pretokenize_gpt2) into graph-static
//  kernels. Classes: 0=L 1=N 2=S(whitespace) 3=O(other). The class table is
//  generated from tiktoken's regex engine.
// =============================================================================
#if GBPE_GPU_PRETOK
#include "gbpe_class_table.h"   // gbpe_classify(cp)->0/1/2/3 ; from generated/

// The header's gbpe_class_blocks/leaves are file-scope *host* const arrays; nvcc
// rejects referencing them from device code (and gbpe_classify, being
// __host__ __device__, references them). Mirror them into __constant__ memory
// (small: 8512 + 8704 bytes, well under the 64KB limit) and provide a __device__
// classify that the pretok kernels use. The kernels call gbpe_classify(cp); the
// macro below redirects those calls to the device version WITHOUT editing the
// kernel bodies (advisor-verified logic stays untouched).
__constant__ uint8_t  d_gbpe_class_leaves[133 * 64];
__constant__ uint16_t d_gbpe_class_blocks[4352];

__device__ __forceinline__ uint8_t gbpe_classify_dev(uint32_t cp) {
    uint32_t over = (uint32_t)(cp > GBPE_CLASS_MAX_CP);
    uint32_t blk  = (cp >> 8) & 0x1FFFu;
    uint32_t leaf = d_gbpe_class_blocks[blk * (1u - over)];
    uint32_t lo   = cp & 0xFFu;
    uint32_t byte = d_gbpe_class_leaves[leaf * 64u + (lo >> 2)];
    uint32_t cls  = (byte >> ((lo & 3u) * 2u)) & 0x3u;
    return (uint8_t)(cls | (over * 3u));
}
#define gbpe_classify gbpe_classify_dev

#if GBPE_HAVE_VOCAB_DEEPSEEK_V3
// Wide property-bit table (DeepSeek only): packs \p{L}\p{N}\s\p{M}\p{P}\p{S}.
// cjk/apl/apc are hardcoded ranges (no table). Mirrors generated gbpe_wide_*.
__constant__ uint8_t  d_gbpe_wide_leaves[152 * 256];
__constant__ uint16_t d_gbpe_wide_blocks[4352];
__device__ __forceinline__ uint8_t gbpe_class_bits_dev(uint32_t cp) {
    uint32_t over = (uint32_t)(cp > GBPE_CLASS_MAX_CP);
    uint32_t blk  = (cp >> 8) & 0x1FFFu;
    uint32_t leaf = d_gbpe_wide_blocks[blk * (1u - over)];
    uint32_t lo   = cp & 0xFFu;
    uint8_t  bits = d_gbpe_wide_leaves[leaf * 256u + lo];
    return (uint8_t)(bits * (1u - over));
}
#endif

// CLS_* / ADDED_MISS / decode_cp and the boundary predicates now live in
// pretok_boundary.h so the host small-input encoder shares them verbatim.

#if GBPE_GPU_PRETOK
// -----------------------------------------------------------------------------
//  Document-locality helpers (batched encode).
//
//  offs[0..n_docs] is the exclusive-prefix layout of the concatenated documents
//  in d_text_bytes: document d occupies bytes [offs[d], offs[d+1]). The array is
//  non-decreasing; empty documents show up as duplicate offsets.
//
//  Every byte-domain read in the pre-tokenizer must be clamped to the document
//  that owns the reading byte, otherwise a doc seam behaves like ordinary text
//  and (a) added-token literals match across it and (b) UTF-8 decoding of a
//  truncated tail consumes the next document's bytes.
//
//  At n_docs == 1 these return exactly {0, len}, so the single-document path is
//  bit-identical to the pre-batch behaviour by construction.
// -----------------------------------------------------------------------------

// Index d of the document containing byte i: the largest d with
// offs[d] <= i < offs[d+1]. Upper-bound binary search over offs[0..n_docs];
// the strict right edge makes duplicate offsets (empty docs) skip themselves.
// Callers must have already checked i < offs[n_docs] (i.e. i < len).
__device__ __forceinline__ uint32_t doc_index_for_byte(
    const uint32_t* __restrict__ offs, uint32_t n_docs, uint32_t i)
{
    // upper_bound(offs[0..n_docs], i) - 1
    uint32_t lo = 0u, hi = n_docs;  // search in [0, n_docs]
    while (lo < hi) {
        const uint32_t mid = lo + (hi - lo + 1u) / 2u;  // mid in (lo, hi]
        if (offs[mid] <= i) lo = mid; else hi = mid - 1u;
    }
    return lo;
}

// Exclusive byte end of the document containing byte i.
__device__ __forceinline__ uint32_t doc_end_for_byte(
    const uint32_t* __restrict__ offs, uint32_t n_docs, uint32_t i)
{
    return offs[doc_index_for_byte(offs, n_docs, i) + 1u];
}

// Shared raw-byte AddedVocabulary matcher. The loader proves that starts of
// literals in this trie cannot occur inside another literal, so expansion of
// the longest match at each start is race-free and requires no host decision,
// atomics, or additional scan.
__device__ __forceinline__ uint32_t added_find_edge(
    uint32_t node, uint8_t byte,
    const uint32_t* __restrict__ edge_begin,
    const uint16_t* __restrict__ edge_count,
    const uint64_t* __restrict__ edges)
{
    uint32_t lo = edge_begin[node];
    uint32_t hi = lo + edge_count[node];
    // Node degrees are tiny for current assets. Linear lookup is cheaper for
    // the common case; binary search bounds future high-degree nodes.
    if (hi - lo <= 8u) {
        for (uint32_t at = lo; at < hi; ++at) {
            const uint64_t edge = edges[at];
            const uint8_t edge_byte = static_cast<uint8_t>(edge);
            if (edge_byte == byte) return static_cast<uint32_t>(edge >> 8);
            if (edge_byte > byte) break;
        }
        return ADDED_MISS;
    }
    while (lo < hi) {
        const uint32_t mid = lo + (hi - lo) / 2u;
        const uint64_t edge = edges[mid];
        const uint8_t edge_byte = static_cast<uint8_t>(edge);
        if (edge_byte < byte) lo = mid + 1u;
        else hi = mid;
    }
    if (lo < edge_begin[node] + edge_count[node]) {
        const uint64_t edge = edges[lo];
        if (static_cast<uint8_t>(edge) == byte) {
            return static_cast<uint32_t>(edge >> 8);
        }
    }
    return ADDED_MISS;
}

__device__ __forceinline__ void added_match_at(uint32_t i,
    const uint8_t* text, const uint32_t* len_ptr,
    uint32_t cap_bytes, const uint32_t* root,
    const uint32_t* edge_begin,
    const uint16_t* edge_count,
    const uint32_t* terminal_id,
    const uint64_t* edges,
    uint32_t trie_nodes, uint32_t max_literal_bytes,
    uint32_t* direct_id,
    uint32_t* direct_len,
    uint32_t* owner_start,
    const uint32_t* doc_offs,
    const uint32_t* n_docs_ptr)
{
    if (i >= cap_bytes) return;
    direct_id[i] = ADDED_MISS;
    direct_len[i] = 0u;
    owner_start[i] = 0u;
    const uint32_t len = *len_ptr;
    if (i >= len || trie_nodes == 0 || max_literal_bytes == 0) return;

    uint32_t node = root[text[i]];
    if (node == ADDED_MISS || node >= trie_nodes) return;
    uint32_t best_id = terminal_id[node];
    uint32_t best_len = best_id == ADDED_MISS ? 0u : 1u;
    // Doc-local: a literal may never straddle a document seam, so the walk is
    // bounded by the end of the document owning byte i (== len when n_docs==1).
    const uint32_t available =
        doc_end_for_byte(doc_offs, *n_docs_ptr, i) - i;
    const uint32_t limit = available < max_literal_bytes
        ? available : max_literal_bytes;
    for (uint32_t offset = 1; offset < limit; ++offset) {
        node = added_find_edge(node, text[i + offset], edge_begin,
                               edge_count, edges);
        if (node == ADDED_MISS || node >= trie_nodes) break;
        const uint32_t id = terminal_id[node];
        if (id != ADDED_MISS) {
            best_id = id;
            best_len = offset + 1u;
        }
    }
    direct_id[i] = best_id;
    direct_len[i] = best_len;
}
__global__ void added_match(
    const uint8_t* __restrict__ text, const uint32_t* __restrict__ len_ptr,
    uint32_t cap_bytes, const uint32_t* __restrict__ root,
    const uint32_t* __restrict__ edge_begin,
    const uint16_t* __restrict__ edge_count,
    const uint32_t* __restrict__ terminal_id,
    const uint64_t* __restrict__ edges,
    uint32_t trie_nodes, uint32_t max_literal_bytes,
    uint32_t* __restrict__ direct_id,
    uint32_t* __restrict__ direct_len,
    uint32_t* __restrict__ owner_start,
    const uint32_t* __restrict__ doc_offs,
    const uint32_t* __restrict__ n_docs_ptr)
{
    added_match_at(blockIdx.x * blockDim.x + threadIdx.x
       , text, len_ptr
       , cap_bytes, root
       , edge_begin
       , edge_count
       , terminal_id
       , edges
       , trie_nodes, max_literal_bytes
       , direct_id
       , direct_len
       , owner_start
       , doc_offs
       , n_docs_ptr);
}

__device__ __forceinline__ void added_expand_owners_at(uint32_t i,
    const uint32_t* direct_len,
    uint32_t cap_bytes, uint32_t* owner_start)
{
    if (i >= cap_bytes || direct_len[i] == 0u) return;
    const uint32_t match_end = i + direct_len[i];
    const uint32_t end = match_end < cap_bytes ? match_end : cap_bytes;
    for (uint32_t at = i; at < end; ++at) owner_start[at] = i + 1u;
}
__global__ void added_expand_owners(
    const uint32_t* __restrict__ direct_len,
    uint32_t cap_bytes, uint32_t* __restrict__ owner_start)
{
    added_expand_owners_at(blockIdx.x * blockDim.x + threadIdx.x
       , direct_len
       , cap_bytes, owner_start);
}
#endif

#if GBPE_HAVE_FAMILY_SP && GBPE_GEMMA_GPU_PRETOK
__device__ __forceinline__ bool sp_is_continuation(uint8_t b) {
    return (b & 0xC0u) == 0x80u;
}

// Return the canonical UTF-8 width at i, or zero for an invalid/truncated
// sequence. This rejects overlong forms, surrogate scalars, and values above
// U+10FFFF.
__device__ __forceinline__ uint8_t sp_valid_utf8_width(
    const uint8_t* text, uint32_t len, uint32_t i)
{
    if (i >= len) return 0;
    const uint8_t b0 = text[i];
    const uint32_t avail = len - i;
    if (b0 <= 0x7Fu) return 1;
    if (b0 >= 0xC2u && b0 <= 0xDFu) {
        return avail >= 2 && sp_is_continuation(text[i + 1]) ? 2 : 0;
    }
    if (b0 >= 0xE0u && b0 <= 0xEFu) {
        if (avail < 3 || !sp_is_continuation(text[i + 2])) return 0;
        const uint8_t b1 = text[i + 1];
        const bool valid_b1 = b0 == 0xE0u ? (b1 >= 0xA0u && b1 <= 0xBFu)
                            : b0 == 0xEDu ? (b1 >= 0x80u && b1 <= 0x9Fu)
                                         : sp_is_continuation(b1);
        return valid_b1 ? 3 : 0;
    }
    if (b0 >= 0xF0u && b0 <= 0xF4u) {
        if (avail < 4 || !sp_is_continuation(text[i + 2]) ||
            !sp_is_continuation(text[i + 3])) return 0;
        const uint8_t b1 = text[i + 1];
        const bool valid_b1 = b0 == 0xF0u ? (b1 >= 0x90u && b1 <= 0xBFu)
                            : b0 == 0xF4u ? (b1 >= 0x80u && b1 <= 0x8Fu)
                                         : sp_is_continuation(b1);
        return valid_b1 ? 4 : 0;
    }
    return 0;
}

// A byte is suppressed only when a strictly valid scalar starting in the
// previous three positions owns it. Invalid/unclaimed continuation bytes are
// emitted through byte fallback instead of being dropped.
__device__ __forceinline__ bool sp_covered_by_valid_lead(
    const uint8_t* text, uint32_t len, uint32_t i)
{
    #pragma unroll
    for (uint32_t distance = 1; distance <= 3; ++distance) {
        if (i < distance) break;
        const uint32_t lead = i - distance;
        if (sp_valid_utf8_width(text, len, lead) > distance) return true;
    }
    return false;
}

__device__ __forceinline__ uint32_t sp_normalized_cp(
    const uint8_t* text, uint32_t i, uint8_t width)
{
    int decoded_width = 1;
    uint32_t cp = decode_cp(text + i, width, &decoded_width);
    return cp == 0x20u ? 0x2581u : cp;
}

constexpr uint32_t SP_ADDED_MISS = ADDED_MISS;

// AddedVocabulary extraction precedes Gemma's Replace normalizer. Raw U+2581
// runs need semantic boundaries because U+2581 intentionally merges into
// following word pieces. Newline/tab added runs are reconstructed by BPE; the
// performance-only component cuts below are separately merge-table-proven.
__device__ __forceinline__ bool sp_is_raw_lowbar_at(
    const uint8_t* text, uint32_t len, uint32_t i)
{
    return i + 2 < len && text[i] == 0xE2u && text[i + 1] == 0x96u &&
           text[i + 2] == 0x81u;
}

__device__ __forceinline__ bool sp_is_inside_raw_lowbar(
    const uint8_t* text, uint32_t i)
{
    return (i >= 1 && text[i - 1] == 0xE2u && text[i] == 0x96u) ||
           (i >= 2 && text[i - 2] == 0xE2u && text[i - 1] == 0x96u &&
            text[i] == 0x81u);
}

// Gemma's tokenizer JSON normalizes ASCII spaces to U+2581 before its Split
// pre-tokenizer runs, so that Split is a no-op and an ordinary document would
// otherwise become one enormous BPE descriptor.  The loader proves that a
// maximal normalized-space run cannot merge with the token on its left, with
// the sole asset-specific exception `>` + `\u2581</`.  It also proves newline
// and tab runs cannot merge with either neighbouring alphabet.  Starting a
// segment at those run boundaries therefore only decomposes independent BPE
// components; it does not change greedy merge order or token IDs.
__device__ __forceinline__ bool sp_starts_independent_bpe_component(
    const uint8_t* text, uint32_t len, uint32_t i)
{
    if (i == 0 || i >= len) return false;
    const uint8_t byte = text[i];
    if (byte != 0x20u && byte != 0x0Au && byte != 0x09u) return false;
    if (text[i - 1] == byte) return false;  // keep each homogeneous run intact

    if (byte == 0x20u) {
        // A single raw U+2581 is ordinary input (AddedVocabulary starts at a
        // run of two), so it must remain able to merge with a following space
        // after that space is normalized to another U+2581.
        if (i >= 3 && sp_is_raw_lowbar_at(text, len, i - 3)) return false;

        // This is the only non-lowbar-left / lowbar-right merge accepted by
        // the loader.  Keep the three raw pieces in one BPE component so the
        // rank-257238 Gemma token remains reachable.
        const bool html_close_exception = text[i - 1] == static_cast<uint8_t>('>') &&
            i + 2 < len && text[i + 1] == static_cast<uint8_t>('<') &&
            text[i + 2] == static_cast<uint8_t>('/');
        if (html_close_exception) return false;
    }
    return true;
}

__global__ void sp_added_run_seed(
    const uint8_t* __restrict__ text, const uint32_t* __restrict__ len_ptr,
    uint32_t cap_bytes, uint32_t* __restrict__ lowbar_run_seed)
{
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= cap_bytes) return;
    const uint32_t len = *len_ptr;
    lowbar_run_seed[i] = 0u;
    if (i >= len) return;
    if (sp_is_raw_lowbar_at(text, len, i)) {
        lowbar_run_seed[i] =
            i >= 3 && sp_is_raw_lowbar_at(text, len, i - 3) ? 0u : i + 1u;
    } else if (!sp_is_inside_raw_lowbar(text, i)) {
        lowbar_run_seed[i] = i + 1u;
    }
}

// Keep the rare raw-U+2581 path out of the hot ordinary-text analysis kernel.
// The scan is monotonic: every byte in one raw run has its start marker, and
// the first following byte has a larger marker, so this lower-bound search is
// bounded by log2(capacity) and needs no extra graph node.
__device__ __noinline__ uint32_t sp_added_lowbar_id(
    uint32_t i, uint32_t len,
    const uint32_t* __restrict__ lowbar_run_start,
    const uint32_t* __restrict__ lowbar_run_ids)
{
    const uint32_t start_mark = lowbar_run_start[i];
    if (start_mark == 0u) return SP_ADDED_MISS;
    const uint32_t start = start_mark - 1u;
    const uint32_t offset = (i - start) / 3u;
    if ((offset % 31u) != 0u) return SP_ADDED_MISS;
    uint32_t lo = start;
    uint32_t hi = len;
    while (lo < hi) {
        const uint32_t mid = lo + (hi - lo) / 2u;
        if (lowbar_run_start[mid] == start_mark) lo = mid + 1u;
        else hi = mid;
    }
    const uint32_t run_length = (lo - start) / 3u;
    if (run_length <= offset) return SP_ADDED_MISS;
    const uint32_t remaining = run_length - offset;
    if (remaining < 2u) return SP_ADDED_MISS;
    const uint32_t take = remaining < 31u ? remaining : 31u;
    return lowbar_run_ids[take];
}

// Pass 1: emission length plus Gemma segment boundary per input byte. Valid
// continuation bytes emit zero; malformed bytes emit one fallback ID.
__global__ void sp_pretok_analyze(
    const uint8_t* __restrict__ text, const uint32_t* __restrict__ len_ptr,
    uint32_t cap_bytes, const uint32_t* __restrict__ cp_to_id,
    const uint32_t* __restrict__ direct_owner_start,
    const uint32_t* __restrict__ lowbar_run_start,
    uint8_t* __restrict__ emit_len, uint32_t* __restrict__ is_boundary)
{
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i > cap_bytes) return;
    if (i == cap_bytes) { emit_len[i] = 0; return; }
    uint32_t len = *len_ptr;
    if (i >= len) {
        emit_len[i] = 0;
        is_boundary[i] = 0;
        return;
    }
    const uint32_t owner = direct_owner_start[i];
    if (owner != 0) {
        const bool is_direct_start = owner - 1u == i;
        emit_len[i] = is_direct_start ? 1u : 0u;
        is_boundary[i] = is_direct_start ? 1u : 0u;
        return;
    }
    const bool follows_direct = i > 0 && direct_owner_start[i - 1] != 0;
    const bool raw_lowbar = sp_is_raw_lowbar_at(text, len, i);
    const bool previous_raw_lowbar =
        i >= 3 && sp_is_raw_lowbar_at(text, len, i - 3);
    const bool next_raw_lowbar =
        raw_lowbar && sp_is_raw_lowbar_at(text, len, i + 3);
    if (raw_lowbar && lowbar_run_start[i] != 0u) {
        const uint32_t start_mark = lowbar_run_start[i];
        const uint32_t start = start_mark - 1u;
        const uint32_t offset = (i - start) / 3u;
        if ((offset % 31u) != 0u) {
            emit_len[i] = 0;
            is_boundary[i] = 0;
            return;
        }
        // At a 31-scalar chunk boundary, another raw scalar proves that at
        // least two remain and AddedVocabulary must consume a greedy chunk.
        // The only non-direct case is the final one-scalar remainder.
        if (next_raw_lowbar) {
            emit_len[i] = 1;
            is_boundary[i] = 1;
            return;
        }
    }
    // A single raw U+2581 remains ordinary text and must still be allowed to
    // merge into the following word. Longer runs were already greedily
    // materialized as direct AddedVocabulary tokens. If their remainder is
    // one U+2581, that final scalar starts the following ordinary span.
    const bool starts_added_lowbar_run =
        raw_lowbar && !previous_raw_lowbar && next_raw_lowbar;
    bool starts_lowbar_tail = false;
    if (raw_lowbar && !next_raw_lowbar && lowbar_run_start[i] != 0) {
        const uint32_t run_start = lowbar_run_start[i] - 1u;
        const uint32_t offset = (i - run_start) / 3u;
        starts_lowbar_tail = offset > 0 && (offset + 1u) % 31u == 1u;
    }
    bool follows_consumed_lowbar_run = false;
    if (previous_raw_lowbar && lowbar_run_start[i - 1] != 0) {
        const uint32_t run_start = lowbar_run_start[i - 1] - 1u;
        const uint32_t last_lowbar = i - 3u;
        const uint32_t run_length = (last_lowbar - run_start) / 3u + 1u;
        follows_consumed_lowbar_run =
            run_length >= 2u && run_length % 31u != 1u;
    }
    const bool starts_plain_span = i == 0 || follows_direct ||
        starts_added_lowbar_run || starts_lowbar_tail ||
        follows_consumed_lowbar_run ||
        sp_starts_independent_bpe_component(text, len, i);
    const uint8_t width = sp_valid_utf8_width(text, len, i);
    if (width == 0) {
        if (sp_covered_by_valid_lead(text, len, i)) {
            emit_len[i] = 0;
            is_boundary[i] = 0;
        } else {
            emit_len[i] = 1;
            is_boundary[i] = starts_plain_span ? 1u : 0u;
        }
        return;
    }
    uint32_t cp = sp_normalized_cp(text, i, width);
    uint32_t id = cp <= 0x10FFFFu ? cp_to_id[cp] : 0xFFFFFFFFu;
    emit_len[i] = id != 0xFFFFFFFFu ? uint8_t{1} : width;
    // Replace(" " -> "▁") precedes Split(" ", MergedWithPrevious), so the
    // Split component cannot match.  We additionally partition that otherwise
    // huge ordinary span only at merge-table-proven independent components.
    is_boundary[i] = starts_plain_span ? 1u : 0u;
}

// Pass 3: repeat the direct lookup and materialize initial token IDs. Misses
// emit the vocab's <0xXX> IDs, not raw byte values.
__global__ void sp_pretok_emit_ids(
    const uint8_t* __restrict__ text, const uint32_t* __restrict__ len_ptr,
    uint32_t cap_bytes, const uint32_t* __restrict__ cp_to_id,
    const uint32_t* __restrict__ byte_fallback,
    const uint32_t* __restrict__ direct_id,
    const uint32_t* __restrict__ lowbar_run_start,
    const uint32_t* __restrict__ lowbar_run_ids,
    const uint8_t* __restrict__ emit_len,
    const uint32_t* __restrict__ emit_offset,
    uint32_t* __restrict__ flat_ids)
{
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= cap_bytes || emit_len[i] == 0) return;
    uint32_t len = *len_ptr;
    uint32_t pos = emit_offset[i];
    if (direct_id[i] != SP_ADDED_MISS) {
        flat_ids[pos] = direct_id[i];
        return;
    }
    if (sp_is_raw_lowbar_at(text, len, i) &&
        sp_is_raw_lowbar_at(text, len, i + 3u)) {
        const uint32_t id = sp_added_lowbar_id(
            i, len, lowbar_run_start, lowbar_run_ids);
        if (id != SP_ADDED_MISS) {
            flat_ids[pos] = id;
            return;
        }
    }
    const uint8_t width = sp_valid_utf8_width(text, len, i);
    if (width == 0) {
        flat_ids[pos] = byte_fallback[text[i]];
        return;
    }
    uint32_t cp = sp_normalized_cp(text, i, width);
    uint32_t id = cp <= 0x10FFFFu ? cp_to_id[cp] : 0xFFFFFFFFu;
    if (id != 0xFFFFFFFFu) {
        flat_ids[pos] = id;
    } else {
        for (uint32_t k = 0; k < width; ++k) {
            flat_ids[pos + k] = byte_fallback[text[i + k]];
        }
    }
}

__global__ void sp_pretok_segment_starts(
    const uint32_t* __restrict__ is_boundary,
    const uint32_t* __restrict__ segment_idx,
    const uint32_t* __restrict__ emit_offset,
    uint32_t cap_bytes, uint32_t* __restrict__ segment_start,
    uint32_t* __restrict__ n_segments)
{
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < cap_bytes && is_boundary[i]) {
        segment_start[segment_idx[i]] = emit_offset[i];
    }
    if (i == 0) {
        uint32_t last = cap_bytes - 1;
        uint32_t n = segment_idx[last] + is_boundary[last];
        *n_segments = n;
        segment_start[n] = emit_offset[cap_bytes];
    }
}

__global__ void sp_pretok_classify_segments(
    const uint32_t* __restrict__ segment_start,
    const uint32_t* __restrict__ n_segments, uint32_t cap_bytes,
    uint32_t* __restrict__ is_short, uint32_t* __restrict__ is_long,
    uint32_t* __restrict__ per_pre_count, uint32_t* __restrict__ n_overflow)
{
    uint32_t s = blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= cap_bytes) return;
    uint32_t n = *n_segments;
    if (s >= n) { is_short[s] = 0; is_long[s] = 0; return; }
    uint32_t count = segment_start[s + 1] - segment_start[s];
    is_short[s] = count <= MAX_PRETOKEN_LEN_SHORT ? 1u : 0u;
    is_long[s] = (count > MAX_PRETOKEN_LEN_SHORT &&
                  count <= MAX_PRETOKEN_LEN_LONG) ? 1u : 0u;
    per_pre_count[s] = 0;
    if (count > MAX_PRETOKEN_LEN_LONG) atomicAdd(n_overflow, 1u);
}

__global__ void sp_pretok_materialize_buckets(
    const uint32_t* __restrict__ flat_ids,
    const uint32_t* __restrict__ segment_start,
    const uint32_t* __restrict__ n_segments,
    const uint32_t* __restrict__ is_short,
    const uint32_t* __restrict__ is_long,
    const uint32_t* __restrict__ short_local,
    const uint32_t* __restrict__ long_local,
    uint32_t cap_bytes, uint32_t cap_short, uint32_t cap_long,
    uint32_t* __restrict__ short_ids, uint16_t* __restrict__ short_lens,
    uint32_t* __restrict__ short_orig_idx,
    uint32_t* __restrict__ long_ids, uint16_t* __restrict__ long_lens,
    uint32_t* __restrict__ long_orig_idx,
    uint32_t* __restrict__ overflow_start,
    uint32_t* __restrict__ overflow_len,
    uint32_t* __restrict__ overflow_orig,
    uint32_t cap_overflow,
    uint32_t* __restrict__ pretok_errors)
{
    uint32_t s = blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= cap_bytes || s >= *n_segments) return;
    uint32_t begin = segment_start[s];
    uint32_t count = segment_start[s + 1] - begin;
    if (is_short[s]) {
        uint32_t row = short_local[s];
        if (row >= cap_short) { atomicAdd(pretok_errors, 1u); return; }
        uint32_t* dst = short_ids + static_cast<size_t>(row) * MAX_PRETOKEN_LEN_SHORT;
        for (uint32_t k = 0; k < count; ++k) dst[k] = flat_ids[begin + k];
        short_lens[row] = static_cast<uint16_t>(count);
        short_orig_idx[row] = s;
    } else if (is_long[s]) {
        uint32_t row = long_local[s];
        if (row >= cap_long) { atomicAdd(pretok_errors, 1u); return; }
        uint32_t* dst = long_ids + static_cast<size_t>(row) * MAX_PRETOKEN_LEN_LONG;
        for (uint32_t k = 0; k < count; ++k) dst[k] = flat_ids[begin + k];
        long_lens[row] = static_cast<uint16_t>(count);
        long_orig_idx[row] = s;
    } else {
        uint32_t row = s - short_local[s] - long_local[s];
        if (row >= cap_overflow) { atomicAdd(pretok_errors, 1u); return; }
        overflow_start[row] = begin;
        overflow_len[row] = count;
        overflow_orig[row] = s;
    }
}
#endif

// k1: one thread per byte. Mark codepoint starts and classify them.
__device__ __forceinline__ void pretok_k1_classify_at(uint32_t i,
    const uint8_t* text, const uint32_t* len_ptr,
    uint32_t cap_bytes,
    const uint32_t* added_owner_start,
    uint32_t* is_start, uint8_t* byte_class,
    uint32_t* n_overflow_reset,
    uint32_t* pretok_errors_reset,
    const uint32_t* doc_offs,
    const uint32_t* n_docs_ptr
#if GBPE_HAVE_VOCAB_DEEPSEEK_V3
    , uint32_t* byte_cp
#endif
    )
{
    if (i >= cap_bytes) return;
    if (i == 0) {
        if (n_overflow_reset) *n_overflow_reset = 0u;  // ordered before k3's atomicAdd
        // Replaces a per-replay memset node: nothing before k1 touches the
        // error counter, and every later increment is an atomic.
        if (pretok_errors_reset) *pretok_errors_reset = 0u;
    }
    uint32_t len = *len_ptr;
    if (i >= len) { is_start[i] = 0; byte_class[i] = CLS_CONT; return; }
    const uint32_t owner = added_owner_start[i];
    if (owner != 0u) {
        const bool direct_start = owner - 1u == i;
        is_start[i] = direct_start ? 1u : 0u;
        byte_class[i] = direct_start ? CLS_ADDED : CLS_CONT;
#if GBPE_HAVE_VOCAB_DEEPSEEK_V3
        byte_cp[i] = 0u;
#endif
        return;
    }
    uint8_t b = text[i];
    bool start = (b & 0xC0u) != 0x80u;
    is_start[i] = start ? 1u : 0u;
    if (start) {
        int width;
        // Doc-local available length: decode_cp does not validate continuation
        // bytes, so a document ending in a lone lead byte would otherwise
        // consume the NEXT document's first bytes and corrupt the PREVIOUS
        // document's last codepoint. The is_start rule above is byte-local and
        // needs no clamp.
        const uint32_t doc_end = doc_end_for_byte(doc_offs, *n_docs_ptr, i);
        const uint32_t cp = decode_cp(text + i, doc_end - i, &width);
        byte_class[i] = gbpe_classify(cp);
#if GBPE_HAVE_VOCAB_DEEPSEEK_V3
        byte_cp[i] = cp;
#endif
    } else {
        byte_class[i] = CLS_CONT;
    }
}
__global__ void pretok_k1_classify(
    const uint8_t* __restrict__ text, const uint32_t* __restrict__ len_ptr,
    uint32_t cap_bytes,
    const uint32_t* __restrict__ added_owner_start,
    uint32_t* __restrict__ is_start, uint8_t* __restrict__ byte_class,
    uint32_t* __restrict__ n_overflow_reset,
    uint32_t* __restrict__ pretok_errors_reset,
    const uint32_t* __restrict__ doc_offs,
    const uint32_t* __restrict__ n_docs_ptr
#if GBPE_HAVE_VOCAB_DEEPSEEK_V3
    , uint32_t* __restrict__ byte_cp
#endif
    )
{
    pretok_k1_classify_at(blockIdx.x * blockDim.x + threadIdx.x
       , text, len_ptr
       , cap_bytes
       , added_owner_start
       , is_start, byte_class
       , n_overflow_reset
       , pretok_errors_reset
       , doc_offs
       , n_docs_ptr
#if GBPE_HAVE_VOCAB_DEEPSEEK_V3
       , byte_cp
#endif
        );
}

// k1b-gather: one thread per byte; scatter dense per-codepoint arrays using the
// exclusive-scan cp ordinal. (Scan itself is a separate CUB call.)
__device__ __forceinline__ void pretok_gather_cp_at(uint32_t i,
    const uint8_t* text, const uint32_t* len_ptr,
    uint32_t cap_bytes,
    const uint32_t* is_start,
    const uint8_t* byte_class,
    const uint32_t* cp_index,
    uint8_t* cls, uint32_t* cp_byte_pos,
    uint8_t* cp_byte0, uint32_t* ncp_ptr,
    bool fold_long_s,
    const uint32_t* doc_offs,
    const uint32_t* n_docs_ptr,
    uint32_t* cp_doc_start,
    uint32_t* digit_seed   // nullable: seed for the digit-run max-scan
#if GBPE_HAVE_VOCAB_DEEPSEEK_V3
    , const uint32_t* byte_cp,
    uint32_t* cp_cp
#endif
    )
{
    if (i >= cap_bytes) return;
    uint32_t len = *len_ptr;
    // ncp = exclusive-scan total of is_start = cp_index[cap_bytes]. The scan runs
    // over cap_bytes+1, so cp_index[cap_bytes] holds the total. One thread copies
    // it to the d_ncp scalar (graph-static).
    if (i == 0) *ncp_ptr = cp_index[cap_bytes];
    if (i >= len || !is_start[i]) return;
    uint32_t c = cp_index[i];
    cls[c]         = byte_class[i];
    cp_byte_pos[c] = i;
    const uint32_t d = doc_index_for_byte(doc_offs, *n_docs_ptr, i);
    const uint32_t doc_end = doc_offs[d + 1u];
    // Doc-start contract consumed by the boundary stage: cp_doc_start[c]
    // is the ordinal of the first codepoint of c's document, so `c` is a
    // doc-start codepoint iff cp_doc_start[c] == c, and two codepoints are in
    // the same document iff their cp_doc_start values are equal. cp_index is the
    // per-byte exclusive scan of is_start, so cp_index at the document's first
    // byte is that ordinal even if the document begins with continuation bytes.
    // At n_docs == 1 this is 0 for every c.
    cp_doc_start[c] = cp_index[doc_offs[d]];
    // Digit-run seed (was its own kernel): a digit seeds 0, or its own
    // ordinal at a document start; anything else seeds c+1. Entries past
    // ncp are never read by the boundary stage, so the tail is left as is.
    if (digit_seed != nullptr) {
        const bool is_digit = byte_class[i] == CLS_N;
        const bool doc_start = cp_doc_start[c] == c;
        digit_seed[c] = is_digit ? (doc_start ? c : 0u) : (c + 1u);
    }
    // U+017F is the sole non-ASCII fold used by the contraction literals. The
    // peek is clamped to the document end so a split \xC5|\xBF does not fold.
    const bool long_s = fold_long_s && text[i] == 0xC5u && i + 1u < doc_end &&
                        text[i + 1u] == 0xBFu;
    cp_byte0[c] = byte_class[i] == CLS_ADDED ? 0u
        : long_s ? static_cast<uint8_t>('s') : text[i];
#if GBPE_HAVE_VOCAB_DEEPSEEK_V3
    cp_cp[c] = byte_cp[i];
#endif
}
__global__ void pretok_gather_cp(
    const uint8_t* __restrict__ text, const uint32_t* __restrict__ len_ptr,
    uint32_t cap_bytes,
    const uint32_t* __restrict__ is_start,
    const uint8_t* __restrict__ byte_class,
    const uint32_t* __restrict__ cp_index,
    uint8_t* __restrict__ cls, uint32_t* __restrict__ cp_byte_pos,
    uint8_t* __restrict__ cp_byte0, uint32_t* __restrict__ ncp_ptr,
    bool fold_long_s,
    const uint32_t* __restrict__ doc_offs,
    const uint32_t* __restrict__ n_docs_ptr,
    uint32_t* __restrict__ cp_doc_start,
    uint32_t* __restrict__ digit_seed   // nullable: seed for the digit-run max-scan
#if GBPE_HAVE_VOCAB_DEEPSEEK_V3
    , const uint32_t* __restrict__ byte_cp,
    uint32_t* __restrict__ cp_cp
#endif
    )
{
    pretok_gather_cp_at(blockIdx.x * blockDim.x + threadIdx.x
       , text, len_ptr
       , cap_bytes
       , is_start
       , byte_class
       , cp_index
       , cls, cp_byte_pos
       , cp_byte0, ncp_ptr
       , fold_long_s
       , doc_offs
       , n_docs_ptr
       , cp_doc_start
       , digit_seed
#if GBPE_HAVE_VOCAB_DEEPSEEK_V3
       , byte_cp
       , cp_cp
#endif
        );
}

// =============================================================================
//  Shared boundary-stage helpers (DRY). Bounded-window codepoint accessors and
//  the contraction predicate, reused by the GPT-2 and Llama/Qwen boundary
//  kernels. (DeepSeek shares only the accessors — its regex has no contraction
//  alternative.) All operate on the dense per-codepoint arrays cls[]/cp_byte0[].
// =============================================================================

// k2: one thread per codepoint. The proven GPT-2 boundary predicate, local
// window {c-2..c+2} + the two ws flags computed inline (also local).
#if GBPE_HAVE_VOCAB_GPT2
__device__ __forceinline__ void pretok_k2_boundaries_at(uint32_t c,
    const uint8_t* cls, const uint8_t* cp_byte0,
    const uint32_t* ncp_ptr, uint32_t cap_bytes,
    uint32_t* is_boundary,
    const uint32_t* cp_doc_start)
{
    if (c >= cap_bytes) return;
    uint32_t n = *ncp_ptr;
    // Grid covers cap_bytes (>= ncp). Inactive lanes MUST write 0 so the
    // downstream fixed-length scans over the tail are correct.
    if (c >= n) { is_boundary[c] = 0u; return; }
    // Same local-window fast path the host uses: most lanes answer from a few
    // loads and skip the cascade's register-heavy frame entirely.
    const FastBoundary f = gpt2_fast(cls, cp_byte0, n, cp_doc_start, c);
    if (f != kFastUnknown) {
        is_boundary[c] = (f == kFastYes) ? 1u : 0u;
        return;
    }
    is_boundary[c] = gpt2_boundary(cls, cp_byte0, n, cp_doc_start, c) ? 1u : 0u;
}
__global__ void pretok_k2_boundaries(
    const uint8_t* __restrict__ cls, const uint8_t* __restrict__ cp_byte0,
    const uint32_t* __restrict__ ncp_ptr, uint32_t cap_bytes,
    uint32_t* __restrict__ is_boundary,
    const uint32_t* __restrict__ cp_doc_start)
{
    pretok_k2_boundaries_at(blockIdx.x * blockDim.x + threadIdx.x
       , cls, cp_byte0
       , ncp_ptr, cap_bytes
       , is_boundary
       , cp_doc_start);
}
#endif  // GBPE_HAVE_VOCAB_GPT2

// Digit-run seed: dseed[c] = (cls[c]!=N) ? (c+1) : 0. An InclusiveScan(Max) of
// this yields drun_start[c] = the run-start ordinal, so digit_off = c-drun_start.
// (Scan, not a local window — a long digit run is one run but many pre-tokens.)
__global__ void pretok_digit_seed(
    const uint8_t* __restrict__ cls, const uint32_t* __restrict__ ncp_ptr,
    uint32_t cap_bytes, uint32_t* __restrict__ dseed,
    const uint32_t* __restrict__ cp_doc_start)
{
    uint32_t c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= cap_bytes) return;
    uint32_t n = *ncp_ptr;
    // A digit that starts a document must start its own digit run, otherwise a
    // run straddling the seam would inherit the previous document's phase
    // (["12","345"] -> "123|45" instead of "12|345"). Seeding c (instead of 0)
    // works because every earlier seed is <= position+1 <= c, so the
    // InclusiveScan(Max) yields drun_start = c for the doc-start digit and for
    // the rest of its run; off = c - drun_start[c] is then document-relative.
    // At n_docs == 1 the doc-start digit is c == 0 and seeding 0 == c, so this
    // is identical to the old all-zero behaviour.
    const bool doc_start = (c < n) && (cp_doc_start[c] == c);
    dseed[c] = (c < n && cls[c] == CLS_N) ? (doc_start ? c : 0u) : (c + 1u);
}

// k2 boundary stage for Llama-3. Local window {k-2..k+1} + the digit-run scan
// (drun_start) for the 3-cap + in-run walks for whitespace/O-run CRLF.
// Translates the proven
// ref_pretokenize_llama3 byte-for-byte. kCaseInsensitive contractions.
#if GBPE_HAVE_VOCAB_LLAMA3 || GBPE_HAVE_VOCAB_QWEN25
template<bool kDigitCap>   // Llama: true (\p{N}{1,3}); Qwen: false (single \p{N})
__device__ __forceinline__ void pretok_k2_llamaqwen_at(uint32_t c,
    const uint8_t* cls, const uint8_t* cp_byte0,
    const uint32_t* drun_start,
    const uint32_t* ncp_ptr, uint32_t cap_bytes,
    uint32_t* is_boundary,
    const uint32_t* cp_doc_start)
{
    if (c >= cap_bytes) return;
    uint32_t n = *ncp_ptr;
    if (c >= n) { is_boundary[c] = 0u; return; }
    // The local-window fast path resolves ~88% of prose positions from a few
    // loads. Lanes that take it skip the full predicate's register-heavy
    // cascade entirely, so a warp only pays for it when some lane needs it.
    const FastBoundary f = llamaqwen_fast(cls, cp_byte0, n, cp_doc_start, c);
    if (f != kFastUnknown) {
        is_boundary[c] = (f == kFastYes) ? 1u : 0u;
        return;
    }
    is_boundary[c] = llamaqwen_boundary<kDigitCap>(
        cls, cp_byte0, drun_start, n, cp_doc_start, c) ? 1u : 0u;
}
template<bool kDigitCap>   // Llama: true (\p{N}{1,3}); Qwen: false (single \p{N})
__global__ void pretok_k2_llamaqwen(
    const uint8_t* __restrict__ cls, const uint8_t* __restrict__ cp_byte0,
    const uint32_t* __restrict__ drun_start,
    const uint32_t* __restrict__ ncp_ptr, uint32_t cap_bytes,
    uint32_t* __restrict__ is_boundary,
    const uint32_t* __restrict__ cp_doc_start)
{
    pretok_k2_llamaqwen_at<kDigitCap>(blockIdx.x * blockDim.x + threadIdx.x
       , cls, cp_byte0
       , drun_start
       , ncp_ptr, cap_bytes
       , is_boundary
       , cp_doc_start);
}
#endif  // GBPE_HAVE_VOCAB_LLAMA3 || GBPE_HAVE_VOCAB_QWEN25

// Boundary stage for DeepSeek-V3. Faithful translation of
// the Python reference ref_pretokenize_deepseek_parallel. Passes 1/2
// split the input first; pass 3 must then treat every resulting span edge as
// its own regex BOS/EOF (rather than reading global left context across a
// digit/CJK split).
//
// The upstream seed + inclusive-max scan materialize each codepoint's pass-3
// span start before k2 runs. This is graph-static and O(N); k2 tests span
// membership in O(1) instead of making many threads walk to both span edges.
// Long pre-tokens are handled by the generic BPE path.
//   is_boundary = pass1 || pass2 || (pass3 && !is_N); [0]=1.
#if GBPE_HAVE_VOCAB_DEEPSEEK_V3
__device__ __forceinline__ bool deepseek_is_number(
    const uint32_t* __restrict__ cp_cp, int j, int n)
{
    return j >= 0 && j < n &&
           (gbpe_class_bits_dev(cp_cp[j]) & GBPE_WB_N) != 0;
}

__device__ __forceinline__ bool deepseek_pass1_boundary(
    const uint32_t* __restrict__ cp_cp,
    const uint32_t* __restrict__ drun_start,
    int j, int n, const uint32_t* __restrict__ cp_doc_start)
{
    if (j < 0 || j >= n) return false;
    // Document start == regex BOS. Short-circuits BEFORE any j-1 read, so the
    // cross-document neighbour is never consulted.
    if (cp_doc_start[j] == (uint32_t)j) return true;
    // drun_start is document-local (see pretok_digit_seed).
    if (deepseek_is_number(cp_cp, j, n)) {
        return (static_cast<uint32_t>(j) - drun_start[j]) % 3u == 0u;
    }
    return deepseek_is_number(cp_cp, j - 1, n);
}

__device__ __forceinline__ bool deepseek_pass2_boundary(
    const uint32_t* __restrict__ cp_cp, int j, int n,
    const uint32_t* __restrict__ cp_doc_start)
{
    if (j < 0 || j >= n) return false;
    if (cp_doc_start[j] == (uint32_t)j) return true;   // before the j-1 read
    return gbpe_is_cjk(cp_cp[j]) != gbpe_is_cjk(cp_cp[j - 1]);
}

// Seed an inclusive Max scan. A non-zero entry is both an upstream-boundary
// flag and its own codepoint index. Boundary zero is represented by zero too;
// consequently every codepoint before the next boundary correctly scans to 0.
__device__ __forceinline__ void pretok_deepseek_span_seed_at(uint32_t c,
    const uint32_t* cp_cp,
    const uint8_t* cls,
    const uint32_t* drun_start,
    const uint32_t* ncp_ptr, uint32_t cap_bytes,
    uint32_t* upstream_seed,
    const uint32_t* cp_doc_start)
{
    if (c >= cap_bytes) return;
    const uint32_t n = *ncp_ptr;
    if (c >= n) {
        upstream_seed[c] = 0u;
        return;
    }
    // No structural change is needed for batching here: pass 1 now returns true
    // at every document start, so the seed is `c` there and the downstream
    // InclusiveScan(Max) can never propagate a span id from the previous
    // document. Every `span_start[j] == span_id` test in pretok_k2_deepseek is
    // therefore document-local for free.
    const bool added_boundary = cls[c] == CLS_ADDED ||
        (c > 0 && cls[c - 1] == CLS_ADDED);
    const bool upstream = added_boundary ||
        deepseek_pass1_boundary(cp_cp, drun_start, static_cast<int>(c),
                                static_cast<int>(n), cp_doc_start) ||
        deepseek_pass2_boundary(cp_cp, static_cast<int>(c),
                                static_cast<int>(n), cp_doc_start);
    upstream_seed[c] = upstream ? c : 0u;
}
__global__ void pretok_deepseek_span_seed(
    const uint32_t* __restrict__ cp_cp,
    const uint8_t* __restrict__ cls,
    const uint32_t* __restrict__ drun_start,
    const uint32_t* __restrict__ ncp_ptr, uint32_t cap_bytes,
    uint32_t* __restrict__ upstream_seed,
    const uint32_t* __restrict__ cp_doc_start)
{
    pretok_deepseek_span_seed_at(blockIdx.x * blockDim.x + threadIdx.x
       , cp_cp
       , cls
       , drun_start
       , ncp_ptr, cap_bytes
       , upstream_seed
       , cp_doc_start);
}

__device__ __forceinline__ void pretok_k2_deepseek_at(uint32_t cu,
    const uint32_t* ncp_ptr, uint32_t cap_bytes,
    uint32_t* is_boundary,
    const uint32_t* cp_cp,
    const uint8_t* cls,
    const uint32_t* upstream_seed,
    const uint32_t* span_start,
    const uint32_t* cp_doc_start)
{
    if (cu >= cap_bytes) return;
    uint32_t n = *ncp_ptr;
    if (cu >= n) { is_boundary[cu] = 0u; return; }
    if (cls[cu] == CLS_ADDED) { is_boundary[cu] = 1u; return; }
    const uint32_t my_doc = cp_doc_start[cu];
    if (my_doc == cu) { is_boundary[cu] = 1u; return; }
    int c = (int)cu;
    int N = (int)n;
    // Document clamp for the raw (non-span) fast-path accessors below. The span
    // accessors get this for free via span_start (see pretok_deepseek_span_seed),
    // but the fast paths bypass the span machinery entirely.
    auto in_doc = [&](int j) -> bool {
        return j >= 0 && j < N && cp_doc_start[j] == my_doc;
    };

    const uint32_t SP = 0x20u, CR = 0x0Du, LF = 0x0Au;
    // Pass 3 receives the spans produced by passes 1/2, so all of its property
    // accessors below are span-local. The seed is retained after the scan and
    // gives the upstream-boundary bit without recomputing either regex pass.
    const bool upstream = upstream_seed[c] != 0u;
    auto raw_is_cjk = [&](int j) -> bool {
        return in_doc(j) && gbpe_is_cjk(cp_cp[j]);
    };

    // Fast decisions for strict run interiors. They are independent of the
    // pass-1/pass-2 span bounds, so they avoid the remaining local run
    // reconstruction for homogeneous letter/space/newline/punctuation inputs.
    auto raw_bits = [&](int j) -> uint8_t {
        return in_doc(j) ? gbpe_class_bits_dev(cp_cp[j]) : 0u;
    };
    auto raw_is_LM = [&](int j) -> bool {
        return (raw_bits(j) & (GBPE_WB_L | GBPE_WB_M)) != 0;
    };
    auto raw_is_PS = [&](int j) -> bool {
        return (raw_bits(j) & (GBPE_WB_P | GBPE_WB_SY)) != 0;
    };
    auto raw_is_ws = [&](int j) -> bool {
        return (raw_bits(j) & GBPE_WB_WS) != 0;
    };
    auto raw_is_crlf = [&](int j) -> bool {
        return in_doc(j) && (cp_cp[j] == CR || cp_cp[j] == LF);
    };
    auto raw_is_gap = [&](int j) -> bool {
        const uint8_t bits = raw_bits(j);
        return in_doc(j) &&
               !(bits & (GBPE_WB_L | GBPE_WB_M | GBPE_WB_N |
                         GBPE_WB_P | GBPE_WB_SY | GBPE_WB_WS)) &&
               !raw_is_cjk(j);
    };
    const bool touches_added = (in_doc(c - 1) && cls[c - 1] == CLS_ADDED) ||
        (in_doc(c + 1) && cls[c + 1] == CLS_ADDED);
    if (!upstream && !touches_added) {
        if (raw_is_LM(c) && raw_is_LM(c - 1) &&
            !(gbpe_is_apl(cp_cp[c - 1]) && !gbpe_is_apl(cp_cp[c]))) {
            is_boundary[c] = 0u;
            return;
        }
        if (raw_is_PS(c) && raw_is_PS(c - 1)) {
            is_boundary[c] = 0u;
            return;
        }
        if (raw_is_crlf(c) && raw_is_crlf(c - 1)) {
            is_boundary[c] = 0u;
            return;
        }
        if (raw_is_ws(c) && raw_is_ws(c - 1) && raw_is_ws(c + 1) &&
            !raw_is_crlf(c) && !raw_is_crlf(c - 1)) {
            is_boundary[c] = 0u;
            return;
        }
        if (raw_is_gap(c) && raw_is_gap(c - 1) && raw_is_gap(c + 1)) {
            is_boundary[c] = 0u;
            return;
        }
    }
    const uint32_t span_id = span_start[c];
    const int span_lo = static_cast<int>(span_id);
    auto in_span = [&](int j) -> bool {
        return j >= 0 && j < N && span_start[j] == span_id;
    };
    auto cp_at = [&](int j) -> uint32_t { return in_span(j) ? cp_cp[j] : 0u; };
    auto is_L_at  = [&](int j) -> bool { return in_span(j) && (gbpe_class_bits_dev(cp_cp[j]) & GBPE_WB_L); };
    auto is_N_at  = [&](int j) -> bool { return in_span(j) && (gbpe_class_bits_dev(cp_cp[j]) & GBPE_WB_N); };
    auto is_M_at  = [&](int j) -> bool { return in_span(j) && (gbpe_class_bits_dev(cp_cp[j]) & GBPE_WB_M); };
    auto is_P_at  = [&](int j) -> bool { return in_span(j) && (gbpe_class_bits_dev(cp_cp[j]) & GBPE_WB_P); };
    auto is_S_at  = [&](int j) -> bool { return in_span(j) && (gbpe_class_bits_dev(cp_cp[j]) & GBPE_WB_SY); };
    auto is_ws_at = [&](int j) -> bool { return in_span(j) && (gbpe_class_bits_dev(cp_cp[j]) & GBPE_WB_WS); };
    auto is_cjk_at = [&](int j) -> bool { return in_span(j) && gbpe_is_cjk(cp_cp[j]); };
    auto is_apl_at = [&](int j) -> bool { return in_span(j) && gbpe_is_apl(cp_cp[j]); };
    auto is_apc_at = [&](int j) -> bool { return in_span(j) && gbpe_is_apc(cp_cp[j]); };
    auto is_crlf   = [&](int j) -> bool { uint32_t p=cp_at(j); return in_span(j) && (p==CR||p==LF); };
    auto is_word   = [&](int j) -> bool { return is_L_at(j) || is_M_at(j); };
    auto is_gap_at = [&](int j) -> bool {
        return in_span(j) && !is_L_at(j) && !is_M_at(j) && !is_N_at(j)
            && !is_P_at(j) && !is_S_at(j) && !is_ws_at(j) && !is_cjk_at(j);
    };

    // ---- other_trailing_nl[j]: CR/LF after a P/S run (chained). Walk back over
    // contiguous CR/LF; the char before that CR/LF run must be P or Sym. ----
    auto other_trailing_nl = [&](int j) -> bool {
        if (j <= span_lo) return false;
        if (!is_crlf(j)) return false;
        int t = j - 1;
        while (t >= span_lo && is_crlf(t)) --t;
        return t >= span_lo && (is_P_at(t) || is_S_at(t));
    };
    // in_ws_run[j] = is_ws AND NOT other_trailing_nl.
    auto in_ws_run = [&](int j) -> bool {
        return is_ws_at(j) && !other_trailing_nl(j);
    };

    // ---- alt1_start[j]: ascii-punct at j with ascii-letter at j+1, not blocked
    // by a preceding P/S/literal-space. ----
    auto alt1_start = [&](int j) -> bool {
        if (!is_apc_at(j)) return false;
        if (!(in_span(j + 1) && is_apl_at(j + 1))) return false;
        if (j == span_lo) return true;
        bool blocked = is_P_at(j - 1) || is_S_at(j - 1) || cp_at(j - 1) == SP;
        return !blocked;
    };
    // in_alt1_tail[j]: j is in the ascii-letter tail of some alt1_start. Walk back
    // over contiguous ascii letters; the char before the run must be an alt1_start.
    // Exact run boundaries remain required for overflow pre-tokens. This can
    // be O(run^2) across the boundary kernel for pathological runs, but keeps
    // the rare correctness path local to the existing predicate.
    const int WALK_CAP = N + 1;
    auto in_alt1_tail = [&](int j) -> bool {
        if (!is_apl_at(j)) return false;
        int rs = j; int w = 0;
        while (rs - 1 >= span_lo && is_apl_at(rs - 1) && ++w <= WALK_CAP) --rs;
        // run of ascii letters [rs, ...]; alt1 tail iff alt1_start(rs-1).
        return alt1_start(rs - 1);
    };

    // ---- whitespace-run flags (alt3 ` ?`, alt4 \s*[\r\n]+, alt5/6). Reconstruct
    // the run [a,bnd) containing index j via in_ws_run walks; mirror _pass3 logic.
    // Provides: nl_token_end(j), ws_split_before_last(j), ws_last_trailing(j),
    // ws_last_is_space(j). ----
    // tail_subrun(p,bnd): marks last of a ws subrun [p,bnd) when bnd is a true
    // pass3 span-interior boundary (not EOF / digit / CJK).
    // We answer per-query rather than materializing arrays.
    auto ws_run_bounds = [&](int j, int& a, int& bnd) {
        int aw = 0; a = j; while (a - 1 >= span_lo && in_ws_run(a - 1) && ++aw <= WALK_CAP) --a;
        int bw = 0; bnd = j; while (in_span(bnd) && in_ws_run(bnd) && ++bw <= WALK_CAP) ++bnd;
    };
    // last CR/LF index within [a,bnd), else -1.
    auto ws_last_nl = [&](int a, int bnd) -> int {
        for (int t = bnd - 1; t >= a; --t) if (is_crlf(t)) return t;
        return -1;
    };
    // nl_token_end(j): j is the last CR/LF of its ws run's leading \s*[\r\n]+ token.
    auto nl_token_end = [&](int j) -> bool {
        if (!in_ws_run(j)) return false;
        int a, bnd; ws_run_bounds(j, a, bnd);
        int nl = ws_last_nl(a, bnd);
        return nl == j;
    };
    // For the trailing subrun [p,bnd): determine the "last" index & its flags.
    // tail subrun exists only when bnd is interior to the current pass-3 span.
    // Returns last index (or -1), plus split/is_space via out params.
    auto tail_last = [&](int p, int bnd, bool& split, bool& is_space) -> int {
        split = false; is_space = false;
        if (p >= bnd) return -1;
        if (!in_span(bnd)) return -1;
        int last = bnd - 1;
        if (last > p) split = true;
        if (cp_at(last) == SP) is_space = true;
        return last;
    };
    // Compute the trailing-subrun "last" index for the run containing j.
    auto run_tail_last = [&](int j, bool& split, bool& is_space) -> int {
        split = false; is_space = false;
        int a, bnd; ws_run_bounds(j, a, bnd);
        int nl = ws_last_nl(a, bnd);
        int p = (nl >= 0) ? nl + 1 : a;
        return tail_last(p, bnd, split, is_space);
    };
    auto ws_split_before_last = [&](int j) -> bool {
        if (!in_ws_run(j)) return false;
        bool sp, isp; int last = run_tail_last(j, sp, isp);
        return last == j && sp;
    };
    auto ws_last_trailing = [&](int j) -> bool {
        if (!in_ws_run(j)) return false;
        bool sp, isp; int last = run_tail_last(j, sp, isp);
        return last == j;
    };
    auto ws_last_is_space = [&](int j) -> bool {
        if (!in_ws_run(j)) return false;
        bool sp, isp; int last = run_tail_last(j, sp, isp);
        return last == j && isp;
    };

    // ---- gap-run flags: gap_split_before_last(j), gap_last_attaches not needed
    // directly (attach handled below). Reconstruct run [a,bnd). ----
    auto gap_split_before_last = [&](int j) -> bool {
        if (!is_gap_at(j)) return false;
        int aw = 0; int a = j; while (a - 1 >= span_lo && is_gap_at(a - 1) && ++aw <= WALK_CAP) --a;
        int bw = 0; int bnd = j; while (in_span(bnd) && is_gap_at(bnd) && ++bw <= WALK_CAP) ++bnd;
        int last = bnd - 1;
        if (j != last) return false;
        bool attaches = (in_span(bnd) && (is_L_at(bnd) || is_M_at(bnd))
                         && !is_cjk_at(bnd) && !is_N_at(bnd));
        return attaches && (last > a);
    };

    // ---- attach_prefix(j): does cp[j] attach forward as alt2 prefix to a word
    // body at j+1? (Python indexes attach_prefix at k-1, body at k.) ----
    auto is_prefixable = [&](int j) -> bool {
        return in_span(j) && !is_crlf(j) && !is_L_at(j)
            && !is_P_at(j) && !is_S_at(j);
    };
    auto attach_prefix = [&](int j) -> bool {
        // j attaches to a word at j+1.
        int k = j + 1;
        if (!is_word(k)) return false;
        if (is_word(j)) return false;          // part of same body, not a prefix
        if (!is_prefixable(j)) return false;
        if (is_ws_at(j)) {
            uint32_t pj = cp_at(j);
            return ws_last_trailing(j) && !(pj == CR || pj == LF);
        }
        return true;                            // gap char: always attaches forward
    };

    // ---- pass3 boundary decision (local cascade, mirrors lines 544-592). ----
    bool continues = false;
    bool cur_word = is_word(c);
    bool prev_word = is_word(c - 1);

    if (in_alt1_tail(c)) {
        continues = true;
    } else if (alt1_start(c)) {
        continues = false;
    } else if (other_trailing_nl(c)) {
        continues = true;
    } else if (attach_prefix(c - 1) && cur_word) {
        continues = true;
    } else if (cur_word) {
        continues = prev_word && !in_alt1_tail(c - 1);
    } else if (is_P_at(c) || is_S_at(c)) {
        if (is_P_at(c - 1) || is_S_at(c - 1)) {
            continues = true;
        } else if (cp_at(c - 1) == SP && ws_last_is_space(c - 1)) {
            continues = true;
        } else {
            continues = false;
        }
    } else if (is_ws_at(c)) {
        if (nl_token_end(c - 1)) {
            continues = false;
        } else if (ws_split_before_last(c)) {
            continues = false;
        } else if (in_ws_run(c - 1)) {
            continues = true;
        } else {
            continues = false;
        }
    } else if (is_gap_at(c)) {
        if (gap_split_before_last(c)) {
            continues = false;
        } else if (is_gap_at(c - 1)) {
            continues = true;
        } else {
            continues = false;
        }
    } else {
        continues = false;
    }
    bool p3 = !continues;
    // Pass 1 extracts digits before pass 3. CJK spans from pass 2 are still
    // presented to pass 3 by the HF Split sequence; pass 3 can split
    // punctuation/symbols inside those spans (e.g. "アントニ・ガウディ").
    bool p3m = p3 && !is_N_at(c);

    is_boundary[c] = (upstream || p3m) ? 1u : 0u;
}
__global__ void pretok_k2_deepseek(
    const uint32_t* __restrict__ ncp_ptr, uint32_t cap_bytes,
    uint32_t* __restrict__ is_boundary,
    const uint32_t* __restrict__ cp_cp,
    const uint8_t* __restrict__ cls,
    const uint32_t* __restrict__ upstream_seed,
    const uint32_t* __restrict__ span_start,
    const uint32_t* __restrict__ cp_doc_start)
{
    pretok_k2_deepseek_at(blockIdx.x * blockDim.x + threadIdx.x
       , ncp_ptr, cap_bytes
       , is_boundary
       , cp_cp
       , cls
       , upstream_seed
       , span_start
       , cp_doc_start);
}
#endif  // GBPE_HAVE_VOCAB_DEEPSEEK_V3

// Next boundary codepoint after boundary c (n if none). Pre-tokens are
// mostly short, so the first 64 positions are checked one by one exactly as
// before. Past that (a long run: the plain walk cost ~0.7 ms per
// 10K-codepoint pre-token in each of k3 and k4) it searches tok_orig_idx,
// the exclusive scan of is_boundary: every position in (c, next boundary]
// holds tok_orig_idx[c] + 1 and every later one more, so a galloping plus
// binary search finds the last such position in O(log length) loads.
__device__ __forceinline__ uint32_t pretok_next_boundary(
    const uint32_t* is_boundary, const uint32_t* tok_orig_idx,
    uint32_t c, uint32_t n)
{
    uint32_t nb = c + 1;
    const uint32_t scalar_end = min(n, c + 65u);
    while (nb < scalar_end && !is_boundary[nb]) ++nb;
    if (nb < scalar_end || nb >= n) return nb;
    const uint32_t want = tok_orig_idx[c] + 1u;
    uint32_t lo = nb - 1;          // tok_orig_idx[lo] == want
    uint32_t hi = n;               // n, or tok_orig_idx[hi] > want
    for (uint32_t step = 64u; lo + step < n; step <<= 1) {
        if (tok_orig_idx[lo + step] > want) { hi = lo + step; break; }
        lo += step;
    }
    while (hi - lo > 1u) {
        const uint32_t mid = lo + (hi - lo) / 2u;
        if (tok_orig_idx[mid] > want) hi = mid; else lo = mid;
    }
    // lo is the last position before the next boundary's successor: the
    // boundary itself, or n-1 when no boundary follows c.
    return is_boundary[lo] ? lo : n;
}

// k3-len/bucket: one thread per codepoint. For boundary codepoints, compute the
// pre-token byte length (next boundary's byte pos - this byte pos) and set the
// is_short/is_long flags (indexed by codepoint position; dense-packed via scan).
__device__ __forceinline__ void pretok_k3_lenbucket_at(uint32_t c,
    const uint32_t* is_boundary, const uint32_t* cp_byte_pos,
    const uint8_t* cls,
    const uint32_t* tok_orig_idx,
    const uint32_t* direct_id_by_byte,
    const uint32_t* ncp_ptr, const uint32_t* text_len_ptr,
    uint32_t cap_bytes,
    uint32_t* is_short, uint32_t* is_long,
    uint32_t* overflow_local,
    uint32_t* n_overflow,
    uint32_t* per_pre_count,
    uint32_t* direct_ids,
    const uint32_t* doc_offs,
    const uint32_t* n_docs_ptr)
{
    if (c >= cap_bytes) return;
    uint32_t n = *ncp_ptr;
    // Inactive/non-boundary lanes write 0 (fixed-length scans over the tail).
    if (c >= n || !is_boundary[c]) { is_short[c] = 0; is_long[c] = 0; return; }
    if (cls[c] == CLS_ADDED) {
        is_short[c] = 0u;
        is_long[c] = 0u;
        const uint32_t orig = tok_orig_idx[c];
        per_pre_count[orig] = 1u;
        direct_ids[orig] = direct_id_by_byte[cp_byte_pos[c]];
        return;
    }
    // find next boundary codepoint
    const uint32_t nb = pretok_next_boundary(is_boundary, tok_orig_idx, c, n);
    uint32_t start_byte = cp_byte_pos[c];
    uint32_t end_byte   = (nb < n) ? cp_byte_pos[nb] : *text_len_ptr;
    // Doc-local byte extent. A pre-token runs to the next pre-token's first
    // byte, so trailing bytes that start no codepoint (orphan UTF-8
    // continuations) are absorbed by the preceding pre-token. Clamping to the
    // document end keeps that absorption inside the document: doc-leading
    // orphan continuation bytes are dropped exactly as they are when the
    // document is encoded on its own. Identity at n_docs == 1.
    const uint32_t doc_end_k3 =
        doc_end_for_byte(doc_offs, *n_docs_ptr, start_byte);
    if (end_byte > doc_end_k3) end_byte = doc_end_k3;
    uint32_t blen = end_byte - start_byte;
    is_short[c] = (blen <= MAX_PRETOKEN_LEN_SHORT) ? 1u : 0u;
    is_long[c]  = (blen > MAX_PRETOKEN_LEN_SHORT && blen <= MAX_PRETOKEN_LEN_LONG) ? 1u : 0u;
    // Count >64B pre-tokens for the generic graph-resident merge path.
    if (blen > MAX_PRETOKEN_LEN_LONG) {
        overflow_local[c] = atomicAdd(n_overflow, 1u);
    }
}
__global__ void pretok_k3_lenbucket(
    const uint32_t* __restrict__ is_boundary, const uint32_t* __restrict__ cp_byte_pos,
    const uint8_t* __restrict__ cls,
    const uint32_t* __restrict__ tok_orig_idx,
    const uint32_t* __restrict__ direct_id_by_byte,
    const uint32_t* __restrict__ ncp_ptr, const uint32_t* __restrict__ text_len_ptr,
    uint32_t cap_bytes,
    uint32_t* __restrict__ is_short, uint32_t* __restrict__ is_long,
    uint32_t* __restrict__ overflow_local,
    uint32_t* __restrict__ n_overflow,
    uint32_t* __restrict__ per_pre_count,
    uint32_t* __restrict__ direct_ids,
    const uint32_t* __restrict__ doc_offs,
    const uint32_t* __restrict__ n_docs_ptr)
{
    pretok_k3_lenbucket_at(blockIdx.x * blockDim.x + threadIdx.x
       , is_boundary, cp_byte_pos
       , cls
       , tok_orig_idx
       , direct_id_by_byte
       , ncp_ptr, text_len_ptr
       , cap_bytes
       , is_short, is_long
       , overflow_local
       , n_overflow
       , per_pre_count
       , direct_ids
       , doc_offs
       , n_docs_ptr);
}

// k4: one thread per codepoint (boundary codepoints do work). Materialize the
// bucketed rows + lens + orig_idx. Tails are pre-zeroed by graph memsets.
__device__ __forceinline__ void pretok_k4_materialize_at(uint32_t c,
    const uint8_t* text,
    const uint32_t* is_boundary, const uint32_t* cp_byte_pos,
    const uint32_t* tok_orig_idx,
    const uint32_t* is_short, const uint32_t* is_long,
    const uint32_t* short_local, const uint32_t* long_local,
    const uint32_t* overflow_local,
    const uint8_t* cls,
    const uint32_t* ncp_ptr, const uint32_t* text_len_ptr,
    uint8_t* short_bytes, uint16_t* short_lens,
    uint32_t* short_orig_idx,
    uint8_t* long_bytes, uint16_t* long_lens,
    uint32_t* long_orig_idx,
    uint32_t* overflow_start,
    uint32_t* overflow_len,
    uint32_t* overflow_orig,
    uint32_t cap_short, uint32_t cap_long, uint32_t cap_overflow,
    uint32_t* pretok_errors,
    const uint32_t* doc_offs,
    const uint32_t* n_docs_ptr,
    // Live-count extraction (was pretok_extract_counts, its own node). The
    // bucket scans it reads are complete before this kernel starts, and
    // nothing in this kernel reads the counts it writes.
    uint32_t capb, uint32_t cap_total,
    uint32_t* n_short_out, uint32_t* n_long_out,
    uint32_t* n_total_out, uint32_t* n_overflow_out)
{
    if (c == 0 && n_short_out != nullptr) {
        const uint32_t last = capb - 1;
        const uint32_t short_count = short_local[last]  + is_short[last];
        const uint32_t long_count  = long_local[last]   + is_long[last];
        const uint32_t total_count = tok_orig_idx[last] + is_boundary[last];
        if (short_count > cap_short || long_count > cap_long ||
            total_count > cap_total) {
            atomicAdd(pretok_errors, 1u);
            *n_short_out = 0; *n_long_out = 0; *n_total_out = 0; *n_overflow_out = 0;
        } else {
            *n_short_out = short_count;
            *n_long_out  = long_count;
            *n_total_out = total_count;
        }
    }
    uint32_t n = *ncp_ptr;
    if (c >= n || !is_boundary[c]) return;
    if (cls[c] == CLS_ADDED) return;
    const uint32_t nb = pretok_next_boundary(is_boundary, tok_orig_idx, c, n);
    uint32_t start_byte = cp_byte_pos[c];
    uint32_t end_byte   = (nb < n) ? cp_byte_pos[nb] : *text_len_ptr;
    // Same doc-local clamp as k3 — the two must agree or the bucket decision
    // and the copied byte range diverge.
    const uint32_t doc_end_k4 =
        doc_end_for_byte(doc_offs, *n_docs_ptr, start_byte);
    if (end_byte > doc_end_k4) end_byte = doc_end_k4;
    uint32_t blen = end_byte - start_byte;
    uint32_t oidx = tok_orig_idx[c];
    if (is_short[c]) {
        uint32_t row = short_local[c];
        if (row >= cap_short) { atomicAdd(pretok_errors, 1u); return; }  // guard OOB
        uint8_t* dst = short_bytes + (size_t)row * MAX_PRETOKEN_LEN_SHORT;
        for (uint32_t k = 0; k < blen; ++k) dst[k] = text[start_byte + k];
        short_lens[row] = (uint16_t)blen;
        short_orig_idx[row] = oidx;
    } else if (is_long[c]) {
        uint32_t row = long_local[c];
        if (row >= cap_long) { atomicAdd(pretok_errors, 1u); return; }   // guard OOB
        uint8_t* dst = long_bytes + (size_t)row * MAX_PRETOKEN_LEN_LONG;
        for (uint32_t k = 0; k < blen; ++k) dst[k] = text[start_byte + k];
        long_lens[row] = (uint16_t)blen;
        long_orig_idx[row] = oidx;
    } else {
        uint32_t row = overflow_local[c];
        if (row >= cap_overflow) { atomicAdd(pretok_errors, 1u); return; }
        overflow_start[row] = start_byte;
        overflow_len[row] = blen;
        overflow_orig[row] = oidx;
    }
}
__global__ void pretok_k4_materialize(
    const uint8_t* __restrict__ text,
    const uint32_t* __restrict__ is_boundary, const uint32_t* __restrict__ cp_byte_pos,
    const uint32_t* __restrict__ tok_orig_idx,
    const uint32_t* __restrict__ is_short, const uint32_t* __restrict__ is_long,
    const uint32_t* __restrict__ short_local, const uint32_t* __restrict__ long_local,
    const uint32_t* __restrict__ overflow_local,
    const uint8_t* __restrict__ cls,
    const uint32_t* __restrict__ ncp_ptr, const uint32_t* __restrict__ text_len_ptr,
    uint8_t* __restrict__ short_bytes, uint16_t* __restrict__ short_lens,
    uint32_t* __restrict__ short_orig_idx,
    uint8_t* __restrict__ long_bytes, uint16_t* __restrict__ long_lens,
    uint32_t* __restrict__ long_orig_idx,
    uint32_t* __restrict__ overflow_start,
    uint32_t* __restrict__ overflow_len,
    uint32_t* __restrict__ overflow_orig,
    uint32_t cap_short, uint32_t cap_long, uint32_t cap_overflow,
    uint32_t* __restrict__ pretok_errors,
    const uint32_t* __restrict__ doc_offs,
    const uint32_t* __restrict__ n_docs_ptr,
    // Live-count extraction (was pretok_extract_counts, its own node). The
    // bucket scans it reads are complete before this kernel starts, and
    // nothing in this kernel reads the counts it writes.
    uint32_t capb, uint32_t cap_total,
    uint32_t* __restrict__ n_short_out, uint32_t* __restrict__ n_long_out,
    uint32_t* __restrict__ n_total_out, uint32_t* __restrict__ n_overflow_out)
{
    pretok_k4_materialize_at(blockIdx.x * blockDim.x + threadIdx.x
       , text
       , is_boundary, cp_byte_pos
       , tok_orig_idx
       , is_short, is_long
       , short_local, long_local
       , overflow_local
       , cls
       , ncp_ptr, text_len_ptr
       , short_bytes, short_lens
       , short_orig_idx
       , long_bytes, long_lens
       , long_orig_idx
       , overflow_start
       , overflow_len
       , overflow_orig
       , cap_short, cap_long, cap_overflow
       , pretok_errors
       , doc_offs
       , n_docs_ptr
       , capb, cap_total
       , n_short_out, n_long_out
       , n_total_out, n_overflow_out);
}

__global__ void flatten_direct_added_kernel(
    const uint8_t* __restrict__ cls,
    const uint32_t* __restrict__ tok_orig_idx,
    const uint32_t* __restrict__ ncp,
    const uint32_t* __restrict__ direct_ids,
    const uint32_t* __restrict__ scan_out,
    uint32_t cap_bytes, uint32_t* __restrict__ out_tokens)
{
    const uint32_t c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= cap_bytes || c >= *ncp || cls[c] != CLS_ADDED) return;
    const uint32_t orig = tok_orig_idx[c];
    out_tokens[scan_out[orig]] = direct_ids[orig];
}

// All four flatten stages in one launch, partitioned by block range:
// [0, short_blocks) short rows, then long rows, then GPU-overflow
// descriptors, then the per-codepoint direct (added-token) writes. Every
// stage writes a disjoint region of out_tokens, so fusing them only removes
// three graph nodes from the critical path.
// Per-document token offsets, computed IN-GRAPH from device-resident data so a
// batched encode needs no extra host round trip. For document d, walk
//   doc byte start -> codepoint ordinal -> owning pre-token -> token offset
// using the same arrays the flatten stage consumes. Entry [n_docs] is the grand
// total, so the n_docs+1 entries are the running sums the caller wants.
//
// The grid is capacity-sized (cap_docs+1) and the live count is read from
// device memory, so this node is graph-static across replays.
//
// NOTE: tok_orig_idx[c] is the pre-token that OWNS codepoint c, so this lookup
// is exact only if every document begins on a pre-token boundary. It does: the
// doc-clamp changes force each document-start codepoint to be a boundary (and
// clamp every neighbour read and byte extent to the owning document), so no
// pre-token can straddle a seam. The chain is therefore exact for all
// byte-level families on arbitrary seams. SP-family contexts never launch it.
// Body shared by the standalone kernel and flatten_all_kernel's doc-offset
// partition, so the two cannot drift.
__device__ __forceinline__ void doc_token_offsets_body(
    const uint32_t* __restrict__ doc_byte_offsets,  // [cap_docs+1]
    const uint32_t* __restrict__ n_docs_ptr,
    uint32_t cap_docs, uint32_t cap_bytes,
    const uint32_t* __restrict__ cp_index,          // [cap_bytes+1]
    const uint32_t* __restrict__ tok_orig_idx,      // [cap_bytes]
    const uint32_t* __restrict__ ncp_ptr,
    const uint32_t* __restrict__ n_total_ptr,
    const uint32_t* __restrict__ scan_out,          // [cap_pretokens+1]
    uint32_t* __restrict__ out_total,
    uint32_t* __restrict__ doc_token_offsets,       // [cap_docs+1]
    uint32_t d)
{
    const uint32_t n = *n_docs_ptr;
    if (d > cap_docs || d > n) return;
    if (d == n) {
        // Also finalizes the output total (was pretok_finalize_total): the
        // count scan is complete, and scan_out at the live n_total is the
        // total because per_pre_count is zero past it.
        const uint32_t total = scan_out[*n_total_ptr];
        *out_total = total;
        doc_token_offsets[d] = total;
        return;
    }
    const uint32_t B = doc_byte_offsets[d];
    const uint32_t c = cp_index[B];
    const uint32_t o = (c >= *ncp_ptr || c >= cap_bytes) ? *n_total_ptr : tok_orig_idx[c];
    doc_token_offsets[d] = scan_out[o];
}

template<int STRIDE_S, int STRIDE_L>
__global__ void flatten_all_kernel(
    // short / long buckets (flatten_kernel_bucketed)
    const uint32_t* __restrict__ short_tokens, const uint32_t* __restrict__ short_orig,
    const uint16_t* __restrict__ short_lens, uint32_t short_rows,
    const uint32_t* __restrict__ n_short_ptr, uint32_t short_blocks,
    const uint32_t* __restrict__ long_tokens, const uint32_t* __restrict__ long_orig,
    const uint16_t* __restrict__ long_lens, uint32_t long_rows,
    const uint32_t* __restrict__ n_long_ptr, uint32_t long_blocks,
    // GPU overflow (flatten_gpu_overflow_kernel)
    const uint32_t* __restrict__ overflow_parts, const uint32_t* __restrict__ overflow_start,
    const uint32_t* __restrict__ overflow_orig, const uint32_t* __restrict__ n_overflow,
    uint32_t overflow_blocks,
    // direct added tokens (flatten_direct_added_kernel)
    const uint8_t* __restrict__ cls, const uint32_t* __restrict__ tok_orig_idx,
    const uint32_t* __restrict__ ncp, const uint32_t* __restrict__ direct_ids,
    uint32_t cap_bytes, uint32_t direct_blocks,
    // per-document token offsets (was pretok_doc_token_offsets); zero blocks
    // disables the partition, for the SP front end which has no doc walk
    const uint32_t* __restrict__ doc_byte_offsets,
    const uint32_t* __restrict__ n_docs_ptr, uint32_t cap_docs,
    const uint32_t* __restrict__ cp_index, const uint32_t* __restrict__ n_total_ptr,
    uint32_t* __restrict__ out_total, uint32_t* __restrict__ doc_token_offsets,
    uint32_t doc_offset_blocks,
    // shared
    const uint32_t* __restrict__ per_pre_count, const uint32_t* __restrict__ scan_out,
    uint32_t* __restrict__ out_tokens)
{
    uint32_t b = blockIdx.x;
    const uint32_t warp_id = threadIdx.x / WARP_SIZE;
    const uint32_t lane = threadIdx.x % WARP_SIZE;
    const uint32_t warps_per_block = blockDim.x / WARP_SIZE;
    if (b < short_blocks) {
        const uint32_t n_active = n_short_ptr ? min(short_rows, *n_short_ptr) : short_rows;
        const uint32_t warp_stride = short_blocks * warps_per_block;
        for (uint32_t i = b * warps_per_block + warp_id; i < n_active; i += warp_stride) {
            if (n_short_ptr == nullptr && short_lens[i] == 0) continue;
            const uint32_t glob = short_orig[i];
            const uint32_t cnt = per_pre_count[glob];
            const uint32_t dst_base = scan_out[glob];
            const uint32_t* src = short_tokens + static_cast<size_t>(i) * STRIDE_S;
            for (uint32_t k = lane; k < cnt; k += WARP_SIZE) out_tokens[dst_base + k] = src[k];
        }
        return;
    }
    b -= short_blocks;
    if (b < long_blocks) {
        const uint32_t n_active = n_long_ptr ? min(long_rows, *n_long_ptr) : long_rows;
        const uint32_t warp_stride = long_blocks * warps_per_block;
        for (uint32_t i = b * warps_per_block + warp_id; i < n_active; i += warp_stride) {
            if (n_long_ptr == nullptr && long_lens[i] == 0) continue;
            const uint32_t glob = long_orig[i];
            const uint32_t cnt = per_pre_count[glob];
            const uint32_t dst_base = scan_out[glob];
            const uint32_t* src = long_tokens + static_cast<size_t>(i) * STRIDE_L;
            for (uint32_t k = lane; k < cnt; k += WARP_SIZE) out_tokens[dst_base + k] = src[k];
        }
        return;
    }
    b -= long_blocks;
    if (b < overflow_blocks) {
        for (uint32_t oi = b; oi < *n_overflow; oi += overflow_blocks) {
            const uint32_t orig = overflow_orig[oi];
            const uint32_t count = per_pre_count[orig];
            const uint32_t src = overflow_start[oi];
            const uint32_t dst = scan_out[orig];
            for (uint32_t k = threadIdx.x; k < count; k += blockDim.x) {
                out_tokens[dst + k] = overflow_parts[src + k];
            }
        }
        return;
    }
    b -= overflow_blocks;
    if (b < direct_blocks) {
        const uint32_t c = b * blockDim.x + threadIdx.x;
        if (c >= cap_bytes || c >= *ncp || cls[c] != CLS_ADDED) return;
        const uint32_t orig = tok_orig_idx[c];
        out_tokens[scan_out[orig]] = direct_ids[orig];
        return;
    }
    // Per-document token offsets, folded in as one more partition so it costs
    // no graph node of its own (a node is ~1.2 us of launch on this device,
    // and this kernel is one thread per document -- usually one document).
    // It reads only the completed count scan and writes doc_token_offsets and
    // out_total, which no other partition touches, so partition order is
    // irrelevant.
    b -= direct_blocks;
    if (doc_offset_blocks != 0u && b < doc_offset_blocks) {
        const uint32_t d = b * blockDim.x + threadIdx.x;
        doc_token_offsets_body(doc_byte_offsets, n_docs_ptr, cap_docs, cap_bytes,
                               cp_index, tok_orig_idx, ncp, n_total_ptr,
                               scan_out, out_total, doc_token_offsets, d);
    }
}

// Extract live bucket counts from the pretok scans (graph-static, 1 thread).
// is_short/is_long/is_boundary tails are zeroed by k2/k3, so the inclusive total
// = exclusive_scan[capb-1] + flag[capb-1]. These device scalars let bpe_kernel
// and flatten bound themselves BY LENGTH instead of relying on zeroed tails — so
// no per-prompt memset of the lens / count arrays is needed.
__global__ void pretok_extract_counts(
    const uint32_t* __restrict__ short_local, const uint32_t* __restrict__ is_short,
    const uint32_t* __restrict__ long_local,  const uint32_t* __restrict__ is_long,
    const uint32_t* __restrict__ tok_orig_idx,const uint32_t* __restrict__ is_boundary,
    uint32_t capb, uint32_t cap_short, uint32_t cap_long, uint32_t cap_total,
    uint32_t* __restrict__ n_short, uint32_t* __restrict__ n_long,
    uint32_t* __restrict__ n_total, uint32_t* __restrict__ n_overflow,
    uint32_t* __restrict__ pretok_errors)
{
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        uint32_t last = capb - 1;
        const uint32_t short_count = short_local[last]  + is_short[last];
        const uint32_t long_count  = long_local[last]   + is_long[last];
        const uint32_t total_count = tok_orig_idx[last] + is_boundary[last];
        // k4 guards individual writes, but every downstream graph node still
        // consumes these live counts. Never let an underestimated context
        // capacity turn that recoverable error into an OOB scan/descriptor read.
        if (short_count > cap_short || long_count > cap_long ||
            total_count > cap_total) {
            atomicAdd(pretok_errors, 1u);
            *n_short = 0;
            *n_long = 0;
            *n_total = 0;
            *n_overflow = 0;
            return;
        }
        *n_short = short_count;
        *n_long  = long_count;
        *n_total = total_count;
    }
}

// Finalize the output token total after the count-scan: read scan_out at the
// DEVICE index n_total (not the fixed capacity index, whose slot is unwritten
// garbage now that per_pre_count is no longer zeroed). 1 thread, graph-static.
__global__ void pretok_finalize_total(
    const uint32_t* __restrict__ scan_out, const uint32_t* __restrict__ n_total,
    uint32_t* __restrict__ out_total)
{
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        *out_total = scan_out[*n_total];
    }
}


__global__ void pretok_doc_token_offsets(
    const uint32_t* __restrict__ doc_byte_offsets,  // [cap_docs+1]
    const uint32_t* __restrict__ n_docs_ptr,
    uint32_t cap_docs, uint32_t cap_bytes,
    const uint32_t* __restrict__ cp_index,          // [cap_bytes+1]
    const uint32_t* __restrict__ tok_orig_idx,      // [cap_bytes]
    const uint32_t* __restrict__ ncp_ptr,
    const uint32_t* __restrict__ n_total_ptr,
    const uint32_t* __restrict__ scan_out,          // [cap_pretokens+1]
    uint32_t* __restrict__ out_total,
    uint32_t* __restrict__ doc_token_offsets)       // [cap_docs+1]
{
    doc_token_offsets_body(doc_byte_offsets, n_docs_ptr, cap_docs, cap_bytes,
                           cp_index, tok_orig_idx, ncp_ptr, n_total_ptr,
                           scan_out, out_total, doc_token_offsets,
                           blockIdx.x * blockDim.x + threadIdx.x);
}


// =============================================================================
//  Single-CTA pre-tokenizer for small size classes.
//
//  The byte-level front end above is eight kernels and five scans (each scan
//  is two graph nodes), and at a few KiB every one of them is launch-bound:
//  about 35 us of a 55 us graph at 128 tokens. For small capacities one CTA
//  runs the same per-element bodies (the *_at functions the standalone
//  kernels call) stage by stage with __syncthreads between them, on work
//  arrays in shared memory, and block scans in place of the CUB device
//  scans. Every buffer a later graph node reads ends up holding the same
//  values as after the multi-kernel sequence, so everything downstream is
//  unchanged. One SM cannot hide the latency of more than a few bytes per
//  thread, so it is used only up to TINY_PRETOK_THREADS * TINY_PRETOK_ITEMS
//  bytes of capacity (measured: at 4 KiB it still wins with the class full,
//  3 KiB of text; at ~4.7 KiB of text it loses), and only where the work
//  arrays fit in shared memory (4 KiB on sm_90, 2 KiB on sm_80).
// =============================================================================
constexpr uint32_t TINY_PRETOK_THREADS = 1024;
constexpr uint32_t TINY_PRETOK_ITEMS = 4;

// Block-scan scratch shared by every scan of one kernel, so the static
// shared memory is one scan's worth, not one per scan.
union TinyScanStorage {
    typename cub::BlockScan<uint32_t, TINY_PRETOK_THREADS>::TempStorage u32;
    typename cub::BlockScan<ShortLong, TINY_PRETOK_THREADS>::TempStorage pair;
};
template<typename T> __device__ __forceinline__
typename cub::BlockScan<T, TINY_PRETOK_THREADS>::TempStorage& scan_storage(TinyScanStorage& s);
template<> __device__ __forceinline__
cub::BlockScan<uint32_t, TINY_PRETOK_THREADS>::TempStorage& scan_storage<uint32_t>(TinyScanStorage& s) {
    return s.u32;
}
template<> __device__ __forceinline__
cub::BlockScan<ShortLong, TINY_PRETOK_THREADS>::TempStorage& scan_storage<ShortLong>(TinyScanStorage& s) {
    return s.pair;
}

// Scan of n elements by one CTA: each thread reduces a contiguous chunk,
// the chunk totals are block-scanned, and each thread then rescans its chunk
// from its prefix. `load` and `store` must not alias (none of the scans below
// is in place). Ends with a barrier, so the output is visible to the CTA
// and `storage` is free for the next scan.
template<typename T, typename Op, typename Load, typename Store>
__device__ __forceinline__ void cta_scan(TinyScanStorage& storage, uint32_t n,
                                         bool exclusive, T identity,
                                         Op op, Load load, Store store)
{
    using BS = cub::BlockScan<T, TINY_PRETOK_THREADS>;
    typename BS::TempStorage& tmp = scan_storage<T>(storage);
    const uint32_t per = (n + TINY_PRETOK_THREADS - 1u) / TINY_PRETOK_THREADS;
    const uint32_t lo = min(n, threadIdx.x * per);
    const uint32_t hi = min(n, lo + per);
    T acc = identity;
    for (uint32_t j = lo; j < hi; ++j) acc = op(acc, load(j));
    T run;
    BS(tmp).ExclusiveScan(acc, run, identity, op);
    for (uint32_t j = lo; j < hi; ++j) {
        const T v = load(j);
        if (exclusive) { store(j, run); run = op(run, v); }
        else           { run = op(run, v); store(j, run); }
    }
    __syncthreads();
}

struct TinyPretokArgs {
    const uint8_t* text; const uint32_t* text_len; uint32_t capb;
    // added-token trie
    const uint32_t* added_root; const uint32_t* added_edge_begin;
    const uint16_t* added_edge_count; const uint32_t* added_token_id;
    const uint64_t* added_edges; uint32_t added_trie_nodes; uint32_t added_max_bytes;
    // work buffers (see TokenizerCtx for what each holds when)
    uint32_t* drun_seed; uint32_t* drun_start; uint32_t* is_long; uint32_t* is_short;
    uint32_t* tok_orig_idx; uint32_t* is_start; uint8_t* byte_class; uint32_t* cp_index;
    uint8_t* cls; uint32_t* cp_byte_pos; uint8_t* cp_byte0; uint32_t* ncp;
    uint32_t* cp_doc_start; uint32_t* is_boundary; uint32_t* short_local; uint32_t* long_local;
    uint32_t* byte_cp; uint32_t* cp_cp;
    const uint32_t* doc_offs; const uint32_t* n_docs;
    uint32_t* n_overflow; uint32_t* pretok_errors;
    uint32_t* per_pre_count; uint32_t* direct_ids;
    // outputs
    uint8_t* short_bytes; uint16_t* short_lens; uint32_t* short_orig_idx;
    uint8_t* long_bytes; uint16_t* long_lens; uint32_t* long_orig_idx;
    uint32_t* overflow_start; uint32_t* overflow_len; uint32_t* overflow_orig;
    uint32_t cap_short, cap_long, cap_pretokens;
    uint32_t* n_short; uint32_t* n_long; uint32_t* n_total;
    PretokKind kind; bool fold_long_s; bool needs_digit_scan;
};

// Dynamic shared memory pretok_tiny_kernel needs for a capacity of capb
// bytes: its per-byte and per-codepoint work arrays (the DeepSeek scalars
// stay in global memory).
constexpr uint32_t kTinyWorkU32 = 12;   // uint32 arrays, capb+1 entries each
constexpr uint32_t kTinyWorkU8 = 3;     // uint8 arrays, capb entries each
__host__ __device__ constexpr size_t tiny_pretok_smem_bytes(uint32_t capb) {
    return kTinyWorkU32 * 4ull * (capb + 1ull) + kTinyWorkU8 * ((capb + 4ull) & ~3ull);
}

__global__ __launch_bounds__(TINY_PRETOK_THREADS)
void pretok_tiny_kernel(const TinyPretokArgs g)
{
    constexpr uint32_t T = TINY_PRETOK_THREADS;
    const uint32_t capb = g.capb;
    // The work arrays live in shared memory: a stage reads what other
    // threads wrote in the stage before, which global memory would serve
    // from L2 on every access (stores do not fill L1). Only what later
    // graph nodes read is copied out at the end; everything a stage writes
    // to global memory directly (counts, buckets, per-pre-token outputs)
    // goes there as in the multi-kernel sequence.
    extern __shared__ uint32_t tiny_smem[];
    __shared__ TinyScanStorage scan_tmp;
    TinyPretokArgs a = g;
    {
        uint32_t* w = tiny_smem;
        uint32_t** u32[kTinyWorkU32] = {
            &a.drun_seed, &a.drun_start, &a.is_long, &a.is_short,
            &a.tok_orig_idx, &a.is_start, &a.cp_index, &a.cp_byte_pos,
            &a.cp_doc_start, &a.is_boundary, &a.short_local, &a.long_local};
        for (uint32_t k = 0; k < kTinyWorkU32; ++k) { *u32[k] = w; w += capb + 1u; }
        uint8_t* w8 = reinterpret_cast<uint8_t*>(w);
        const uint32_t stride8 = (capb + 4u) & ~3u;
        a.byte_class = w8; a.cls = w8 + stride8; a.cp_byte0 = w8 + 2u * stride8;
        // The CUB scan reads is_start[capb], which k1 never writes (the
        // global buffer is zeroed at allocation).
        if (threadIdx.x == 0) a.is_start[capb] = 0u;
    }
    const auto sum = [](uint32_t x, uint32_t y) { return x + y; };
    const auto max = [](uint32_t x, uint32_t y) { return x > y ? x : y; };

    for (uint32_t i = threadIdx.x; i < capb; i += T)
        added_match_at(i, a.text, a.text_len, capb, a.added_root,
            a.added_edge_begin, a.added_edge_count, a.added_token_id,
            a.added_edges, a.added_trie_nodes, a.added_max_bytes,
            a.drun_seed, a.is_long, a.tok_orig_idx, a.doc_offs, a.n_docs);
    __syncthreads();
    for (uint32_t i = threadIdx.x; i < capb; i += T)
        added_expand_owners_at(i, a.is_long, capb, a.tok_orig_idx);
    __syncthreads();
    for (uint32_t i = threadIdx.x; i < capb; i += T)
        pretok_k1_classify_at(i, a.text, a.text_len, capb, a.tok_orig_idx,
            a.is_start, a.byte_class, a.n_overflow, a.pretok_errors,
            a.doc_offs, a.n_docs
#if GBPE_HAVE_VOCAB_DEEPSEEK_V3
            , a.byte_cp
#endif
            );
    __syncthreads();
    cta_scan<uint32_t>(scan_tmp, capb + 1u, true, 0u, sum,
        [&](uint32_t j) { return a.is_start[j]; },
        [&](uint32_t j, uint32_t v) { a.cp_index[j] = v; });
    for (uint32_t i = threadIdx.x; i < capb; i += T)
        pretok_gather_cp_at(i, a.text, a.text_len, capb, a.is_start,
            a.byte_class, a.cp_index, a.cls, a.cp_byte_pos, a.cp_byte0, a.ncp,
            a.fold_long_s, a.doc_offs, a.n_docs, a.cp_doc_start,
            a.needs_digit_scan ? a.is_short : nullptr
#if GBPE_HAVE_VOCAB_DEEPSEEK_V3
            , a.byte_cp, a.cp_cp
#endif
            );
    __syncthreads();
    if (a.needs_digit_scan) {
        cta_scan<uint32_t>(scan_tmp, capb, false, 0u, max,
            [&](uint32_t j) { return a.is_short[j]; },
            [&](uint32_t j, uint32_t v) { a.drun_start[j] = v; });
    }
    switch (a.kind) {
#if GBPE_HAVE_VOCAB_LLAMA3
        case PretokKind::Llama3:
            for (uint32_t c = threadIdx.x; c < capb; c += T)
                pretok_k2_llamaqwen_at<true>(c, a.cls, a.cp_byte0, a.drun_start,
                    a.ncp, capb, a.is_boundary, a.cp_doc_start);
            break;
#endif
#if GBPE_HAVE_VOCAB_QWEN25
        case PretokKind::Qwen25:
            for (uint32_t c = threadIdx.x; c < capb; c += T)
                pretok_k2_llamaqwen_at<false>(c, a.cls, a.cp_byte0, a.drun_start,
                    a.ncp, capb, a.is_boundary, a.cp_doc_start);
            break;
#endif
#if GBPE_HAVE_VOCAB_DEEPSEEK_V3
        case PretokKind::DeepSeekV3:
            for (uint32_t c = threadIdx.x; c < capb; c += T)
                pretok_deepseek_span_seed_at(c, a.cp_cp, a.cls, a.drun_start,
                    a.ncp, capb, a.is_short, a.cp_doc_start);
            __syncthreads();
            cta_scan<uint32_t>(scan_tmp, capb, false, 0u, max,
                [&](uint32_t j) { return a.is_short[j]; },
                [&](uint32_t j, uint32_t v) { a.short_local[j] = v; });
            for (uint32_t c = threadIdx.x; c < capb; c += T)
                pretok_k2_deepseek_at(c, a.ncp, capb, a.is_boundary, a.cp_cp,
                    a.cls, a.is_short, a.short_local, a.cp_doc_start);
            break;
#endif
#if GBPE_HAVE_VOCAB_GPT2
        case PretokKind::GPT2:
            for (uint32_t c = threadIdx.x; c < capb; c += T)
                pretok_k2_boundaries_at(c, a.cls, a.cp_byte0, a.ncp, capb,
                    a.is_boundary, a.cp_doc_start);
            break;
#endif
        default: break;
    }
    __syncthreads();
    cta_scan<uint32_t>(scan_tmp, capb, true, 0u, sum,
        [&](uint32_t j) { return a.is_boundary[j]; },
        [&](uint32_t j, uint32_t v) { a.tok_orig_idx[j] = v; });
    for (uint32_t c = threadIdx.x; c < capb; c += T)
        pretok_k3_lenbucket_at(c, a.is_boundary, a.cp_byte_pos, a.cls,
            a.tok_orig_idx, a.drun_seed, a.ncp, a.text_len, capb,
            a.is_short, a.is_long, a.drun_start, a.n_overflow,
            a.per_pre_count, a.direct_ids, a.doc_offs, a.n_docs);
    __syncthreads();
    cta_scan<ShortLong>(scan_tmp, capb, true, ShortLong{0u, 0u},
        [](ShortLong x, ShortLong y) { return x + y; },
        [&](uint32_t j) { return ShortLong{a.is_short[j], a.is_long[j]}; },
        [&](uint32_t j, ShortLong v) { a.short_local[j] = v.s; a.long_local[j] = v.l; });
    for (uint32_t c = threadIdx.x; c < capb; c += T)
        pretok_k4_materialize_at(c, a.text, a.is_boundary, a.cp_byte_pos,
            a.tok_orig_idx, a.is_short, a.is_long, a.short_local, a.long_local,
            a.drun_start, a.cls, a.ncp, a.text_len,
            a.short_bytes, a.short_lens, a.short_orig_idx,
            a.long_bytes, a.long_lens, a.long_orig_idx,
            a.overflow_start, a.overflow_len, a.overflow_orig,
            a.cap_short, a.cap_long, a.cap_pretokens, a.pretok_errors,
            a.doc_offs, a.n_docs,
            capb, a.cap_pretokens, a.n_short, a.n_long, a.n_total, a.n_overflow);
    // What the flatten stage reads: codepoint classes, pre-token owners, and
    // the codepoint index (per-document offsets).
    for (uint32_t i = threadIdx.x; i < capb; i += T) {
        g.cls[i] = a.cls[i];
        g.tok_orig_idx[i] = a.tok_orig_idx[i];
        g.cp_index[i] = a.cp_index[i];
    }
    if (threadIdx.x == 0) g.cp_index[capb] = a.cp_index[capb];
}

// Count scan + flatten for the same small classes, as one single-CTA
// launch: the block scan replaces the two-node CUB scan, then every stage of
// flatten_all_kernel runs at once with one thread per bucket row (a warp per
// overflow pre-token), since one CTA has too few warps for a warp per row.
struct TinyTailArgs {
    const uint32_t* short_tokens; const uint32_t* short_orig; const uint32_t* n_short;
    const uint32_t* long_tokens; const uint32_t* long_orig; const uint32_t* n_long;
    const uint32_t* overflow_parts; const uint32_t* overflow_start;
    const uint32_t* overflow_orig; const uint32_t* n_overflow;
    const uint8_t* cls; const uint32_t* tok_orig_idx; const uint32_t* ncp;
    const uint32_t* direct_ids; uint32_t cap_bytes;
    const uint32_t* doc_byte_offsets; const uint32_t* n_docs; uint32_t cap_docs;
    const uint32_t* cp_index; const uint32_t* n_total;
    uint32_t* out_total; uint32_t* doc_token_offsets;
    const uint32_t* per_pre_count; uint32_t* scan_out; uint32_t cap_pretokens;
    uint32_t* out_tokens;
};

template<int STRIDE_S, int STRIDE_L>
__global__ __launch_bounds__(TINY_PRETOK_THREADS)
void tiny_tail_kernel(const TinyTailArgs a)
{
    constexpr uint32_t T = TINY_PRETOK_THREADS;
    __shared__ TinyScanStorage scan_tmp;
    cta_scan<uint32_t>(scan_tmp, a.cap_pretokens + 1u, true, 0u,
        [](uint32_t x, uint32_t y) { return x + y; },
        [&](uint32_t j) { return a.per_pre_count[j]; },
        [&](uint32_t j, uint32_t v) { a.scan_out[j] = v; });
    const uint32_t tid = threadIdx.x;
    const uint32_t n_short = *a.n_short;
    for (uint32_t i = tid; i < n_short; i += T) {
        const uint32_t glob = a.short_orig[i];
        const uint32_t cnt = a.per_pre_count[glob];
        const uint32_t dst = a.scan_out[glob];
        const uint32_t* src = a.short_tokens + static_cast<size_t>(i) * STRIDE_S;
        for (uint32_t k = 0; k < cnt; ++k) a.out_tokens[dst + k] = src[k];
    }
    const uint32_t n_long = a.n_long ? *a.n_long : 0u;
    for (uint32_t i = tid; i < n_long; i += T) {
        const uint32_t glob = a.long_orig[i];
        const uint32_t cnt = a.per_pre_count[glob];
        const uint32_t dst = a.scan_out[glob];
        const uint32_t* src = a.long_tokens + static_cast<size_t>(i) * STRIDE_L;
        for (uint32_t k = 0; k < cnt; ++k) a.out_tokens[dst + k] = src[k];
    }
    const uint32_t lane = tid % WARP_SIZE;
    for (uint32_t oi = tid / WARP_SIZE; oi < *a.n_overflow; oi += T / WARP_SIZE) {
        const uint32_t orig = a.overflow_orig[oi];
        const uint32_t count = a.per_pre_count[orig];
        const uint32_t src = a.overflow_start[oi];
        const uint32_t dst = a.scan_out[orig];
        for (uint32_t k = lane; k < count; k += WARP_SIZE)
            a.out_tokens[dst + k] = a.overflow_parts[src + k];
    }
    const uint32_t ncp = *a.ncp;
    for (uint32_t c = tid; c < a.cap_bytes && c < ncp; c += T) {
        if (a.cls[c] != CLS_ADDED) continue;
        const uint32_t orig = a.tok_orig_idx[c];
        a.out_tokens[a.scan_out[orig]] = a.direct_ids[orig];
    }
    for (uint32_t d = tid; d <= a.cap_docs; d += T) {
        doc_token_offsets_body(a.doc_byte_offsets, a.n_docs, a.cap_docs, a.cap_bytes,
                               a.cp_index, a.tok_orig_idx, a.ncp, a.n_total,
                               a.scan_out, a.out_total, a.doc_token_offsets, d);
    }
}
#endif  // GBPE_GPU_PRETOK


// =============================================================================
//  Host-side context + CUDA graph capture.
// =============================================================================

// An unset variable uses the large-context-tuned default. A value of zero
// restores the capacity-sized comparison grid; any positive value bounds the
// captured launch to that many CTAs per SM. Kernels consume all device-resident
// active rows through grid-stride loops.
static uint32_t bounded_grid_from_env(const char* name,
                                      uint32_t capacity_grid,
                                      uint32_t sm_count,
                                      uint32_t default_blocks_per_sm) {
    const char* text = std::getenv(name);
    unsigned long blocks_per_sm = default_blocks_per_sm;
    if (text == nullptr || *text == '\0') {
        // Use the measured large-context default.
    } else {
        char* end = nullptr;
        blocks_per_sm = std::strtoul(text, &end, 10);
        if (*text == '-' || end == text || *end != '\0' ||
            blocks_per_sm > UINT32_MAX) {
            throw std::invalid_argument(std::string(name) +
                                        " must be a non-negative integer");
        }
    }
    if (blocks_per_sm == 0) return capacity_grid;
    const uint64_t requested =
        static_cast<uint64_t>(sm_count) * blocks_per_sm;
    const uint32_t bounded = requested > capacity_grid
                           ? capacity_grid
                           : static_cast<uint32_t>(requested);
    return bounded == 0 ? 1 : bounded;
}

static uint32_t work_grid_scale(uint32_t capacity_rows) {
    constexpr uint32_t kReferenceRows = 1u << 19;  // 524,288
    const uint32_t ratio = static_cast<uint32_t>(
        (static_cast<uint64_t>(capacity_rows) + kReferenceRows - 1) /
        kReferenceRows);
    uint32_t scale = 1;
    while (scale * scale < ratio) ++scale;
    return scale;
}

static uint32_t flatten_threads_from_env() {
    const char* text = std::getenv("GBPE_FLATTEN_THREADS");
    if (text == nullptr || *text == '\0') return BLOCK_THREADS;
    char* end = nullptr;
    const unsigned long threads = std::strtoul(text, &end, 10);
    if (end == text || *end != '\0' ||
        (threads != 32 && threads != 64 &&
         threads != 128 && threads != 256)) {
        throw std::invalid_argument(
            "GBPE_FLATTEN_THREADS must be one of 32, 64, 128, or 256");
    }
    return static_cast<uint32_t>(threads);
}

TokenizerCtx::TokenizerCtx(const VocabPack& vp,
                           uint32_t max_short_pretokens,
                           uint32_t max_long_pretokens,
                           uint32_t max_input_bytes,
                           uint32_t max_output_tokens,
                           uint32_t max_decode_tokens,
                           uint32_t max_decode_bytes,
                           uint32_t max_batch_docs,
                           bool use_cuda_graph)
    : vocab(vp),
      cap_pretokens(max_short_pretokens + max_long_pretokens),
      cap_short(max_short_pretokens),
      cap_long(max_long_pretokens),
      cap_input_bytes(max_input_bytes),
      cap_output_tokens(max_output_tokens),
      cap_docs(max_batch_docs > 0 ? max_batch_docs : 1u),
      use_cuda_graph_(use_cuda_graph)
{
    cap_decode_tokens = max_decode_tokens;
    cap_decode_bytes  = max_decode_bytes;
#if GBPE_HAVE_FAMILY_SP
    sp_mode_ = vp.sp_family;
#if !GBPE_GEMMA_GPU_PRETOK
    // Fixed overflow cap mirrors the host-side kOverflowCap (main.cc /
    // python_bindings.cc). Overflow pre-tokens are rare (a handful per million);
    // 1024 is the host-side staging cap, so the device buffer matches it.
    if (sp_mode_) cap_overflow = 1024;
#endif
#endif
    allocate();
    if (cap_decode_tokens > 0 && cap_decode_bytes > 0) {
        allocate_decode();
    }
}

void TokenizerCtx::allocate() {
    // Per-bucket inputs.
    CUDA_CHECK(cudaMalloc(&d_short_bytes,
                          static_cast<size_t>(cap_short) * MAX_PRETOKEN_LEN_SHORT));
    CUDA_CHECK(cudaMalloc(&d_short_lens,
                          static_cast<size_t>(cap_short) * sizeof(uint16_t)));
    CUDA_CHECK(cudaMemset(d_short_lens, 0,
                          static_cast<size_t>(cap_short) * sizeof(uint16_t)));
    CUDA_CHECK(cudaMalloc(&d_short_orig_idx,
                          static_cast<size_t>(cap_short) * sizeof(uint32_t)));

    if (cap_long > 0) {
        CUDA_CHECK(cudaMalloc(&d_long_bytes,
                              static_cast<size_t>(cap_long) * MAX_PRETOKEN_LEN_LONG));
        CUDA_CHECK(cudaMalloc(&d_long_lens,
                              static_cast<size_t>(cap_long) * sizeof(uint16_t)));
        CUDA_CHECK(cudaMemset(d_long_lens, 0,
                              static_cast<size_t>(cap_long) * sizeof(uint16_t)));
        CUDA_CHECK(cudaMalloc(&d_long_orig_idx,
                              static_cast<size_t>(cap_long) * sizeof(uint32_t)));
    }

    // Shared outputs keyed by orig_idx in [0, cap_pretokens). +1 for the
    // scan sentinel slot.
    CUDA_CHECK(cudaMalloc(&d_per_pre_count,
                          static_cast<size_t>(cap_pretokens + 1) * sizeof(uint32_t)));
    CUDA_CHECK(cudaMemset(d_per_pre_count, 0,
                          static_cast<size_t>(cap_pretokens + 1) * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_per_pre_tokens_short,
                          static_cast<size_t>(cap_short) * MAX_PRETOKEN_LEN_SHORT
                          * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_per_pre_tokens_long,
                          static_cast<size_t>(cap_long > 0 ? cap_long : 1) * MAX_PRETOKEN_LEN_LONG
                          * sizeof(uint32_t)));
#if GBPE_HAVE_FAMILY_SP
    if (sp_mode_ && cap_overflow > 0) {
        CUDA_CHECK(cudaMalloc(&d_per_pre_tokens_overflow,
                              static_cast<size_t>(cap_overflow) * MAX_PRETOKEN_LEN_LONG
                              * sizeof(uint32_t)));
        CUDA_CHECK(cudaMalloc(&d_overflow_orig_idx,
                              static_cast<size_t>(cap_overflow) * sizeof(uint32_t)));
        CUDA_CHECK(cudaMalloc(&d_overflow_n, sizeof(uint32_t)));
        CUDA_CHECK(cudaMemset(d_overflow_n, 0, sizeof(uint32_t)));
    }
#endif
    CUDA_CHECK(cudaMalloc(&d_scan_out,
                          static_cast<size_t>(cap_pretokens + 1) * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_out_tokens,
                          static_cast<size_t>(cap_output_tokens) * sizeof(uint32_t)));
#if GBPE_GPU_PRETOK
    CUDA_CHECK(cudaMalloc(&d_direct_ids,
                          static_cast<size_t>(cap_pretokens) * sizeof(uint32_t)));
#endif

    // Pinned host staging.
    CUDA_CHECK(cudaMallocHost(&h_pin_short_bytes,
                              static_cast<size_t>(cap_short) * MAX_PRETOKEN_LEN_SHORT));
    CUDA_CHECK(cudaMallocHost(&h_pin_short_lens,
                              static_cast<size_t>(cap_short) * sizeof(uint16_t)));
    CUDA_CHECK(cudaMallocHost(&h_pin_short_orig_idx,
                              static_cast<size_t>(cap_short) * sizeof(uint32_t)));
    if (cap_long > 0) {
        CUDA_CHECK(cudaMallocHost(&h_pin_long_bytes,
                                  static_cast<size_t>(cap_long) * MAX_PRETOKEN_LEN_LONG));
        CUDA_CHECK(cudaMallocHost(&h_pin_long_lens,
                                  static_cast<size_t>(cap_long) * sizeof(uint16_t)));
        CUDA_CHECK(cudaMallocHost(&h_pin_long_orig_idx,
                                  static_cast<size_t>(cap_long) * sizeof(uint32_t)));
    }

#if GBPE_HAVE_FAMILY_SP
    if (sp_mode_) {
        // SP family: GPU BPE consumes uint32 initial-token ID rows.
        CUDA_CHECK(cudaMalloc(&d_short_ids,
            static_cast<size_t>(cap_short) * MAX_PRETOKEN_LEN_SHORT * sizeof(uint32_t)));
#if !GBPE_GEMMA_GPU_PRETOK
        CUDA_CHECK(cudaMallocHost(&h_pin_short_ids,
            static_cast<size_t>(cap_short) * MAX_PRETOKEN_LEN_SHORT * sizeof(uint32_t)));
#endif
        if (cap_long > 0) {
            CUDA_CHECK(cudaMalloc(&d_long_ids,
                static_cast<size_t>(cap_long) * MAX_PRETOKEN_LEN_LONG * sizeof(uint32_t)));
#if !GBPE_GEMMA_GPU_PRETOK
            CUDA_CHECK(cudaMallocHost(&h_pin_long_ids,
                static_cast<size_t>(cap_long) * MAX_PRETOKEN_LEN_LONG * sizeof(uint32_t)));
#endif
        }
    }
#endif

    {
        size_t bytes = 0;
        cub::DeviceScan::ExclusiveSum(nullptr, bytes,
            d_per_pre_count, d_scan_out,
            cap_pretokens + 1, /*stream=*/0);
        scan_temp_bytes = bytes;
        CUDA_CHECK(cudaMalloc(&d_scan_temp, scan_temp_bytes));
    }

#if GBPE_GPU_PRETOK
    {
        const size_t capb  = cap_input_bytes;
        const size_t capb1 = static_cast<size_t>(cap_input_bytes) + 1;
        // Device input + length scalar.
        CUDA_CHECK(cudaMalloc(&d_text_bytes, capb));
        // Header block: [0] text_len, [1] n_docs, [2..] doc byte offsets.
        // Zero length so the warmup capture sees len=0 -> ncp=0 -> all pretok
        // kernels early-return cleanly (no OOB reads on uninitialized text);
        // n_docs=1 and zero offsets describe one empty document.
        {
            const size_t hdr_words = 2 + static_cast<size_t>(cap_docs) + 1;
            CUDA_CHECK(cudaMalloc(&d_hdr, hdr_words * sizeof(uint32_t)));
            CUDA_CHECK(cudaMemset(d_hdr, 0, hdr_words * sizeof(uint32_t)));
            const uint32_t one = 1u;
            CUDA_CHECK(cudaMemcpy(d_hdr + 1, &one, sizeof(uint32_t),
                                  cudaMemcpyHostToDevice));
            d_text_len = d_hdr + 0;
        }
        // Per-byte workspaces. is_start is sized cap+1 (the ExclusiveSum runs
        // over cap+1 items so cp_index[cap] holds the grand total); element
        // [cap] is never written by k1, so zero it once here.
        CUDA_CHECK(cudaMalloc(&d_is_start, capb1 * sizeof(uint32_t)));
        CUDA_CHECK(cudaMemset(d_is_start, 0, capb1 * sizeof(uint32_t)));
        CUDA_CHECK(cudaMalloc(&d_byte_class, capb1));
        CUDA_CHECK(cudaMalloc(&d_cp_index, capb1 * sizeof(uint32_t)));
        // Per-codepoint dense arrays (ncp <= nbytes, so cap is an upper bound).
        CUDA_CHECK(cudaMalloc(&d_cls, capb));
        CUDA_CHECK(cudaMalloc(&d_cp_byte_pos, capb * sizeof(uint32_t)));
        CUDA_CHECK(cudaMalloc(&d_cp_byte0, capb));
#if GBPE_HAVE_VOCAB_DEEPSEEK_V3
        CUDA_CHECK(cudaMalloc(&d_byte_cp, capb * sizeof(uint32_t)));
        CUDA_CHECK(cudaMalloc(&d_cp_cp,   capb * sizeof(uint32_t)));
#endif
        CUDA_CHECK(cudaMalloc(&d_drun_seed,  capb * sizeof(uint32_t)));
        CUDA_CHECK(cudaMalloc(&d_drun_start, capb * sizeof(uint32_t)));
        CUDA_CHECK(cudaMalloc(&d_is_boundary, capb * sizeof(uint32_t)));
        CUDA_CHECK(cudaMalloc(&d_tok_orig_idx, capb * sizeof(uint32_t)));
        CUDA_CHECK(cudaMalloc(&d_is_short, capb * sizeof(uint32_t)));
        CUDA_CHECK(cudaMalloc(&d_is_long, capb * sizeof(uint32_t)));
        CUDA_CHECK(cudaMalloc(&d_short_local, capb * sizeof(uint32_t)));
        CUDA_CHECK(cudaMalloc(&d_long_local, capb * sizeof(uint32_t)));
        CUDA_CHECK(cudaMalloc(&d_ncp, sizeof(uint32_t)));
        CUDA_CHECK(cudaMemset(d_ncp, 0, sizeof(uint32_t)));
        // Live-count device scalars (filled by pretok_extract_counts /
        // pretok_finalize_total each launch). Zero-init so the warmup capture
        // (len=0) sees 0 counts and every bounded kernel cleanly no-ops.
        CUDA_CHECK(cudaMalloc(&d_n_short, sizeof(uint32_t)));
        CUDA_CHECK(cudaMalloc(&d_n_long,  sizeof(uint32_t)));
        CUDA_CHECK(cudaMalloc(&d_n_total, sizeof(uint32_t)));
        // Mapped pinned status: the graph writes the token total and the
        // error count straight into host memory; the host reads them after
        // the stream sync without a D2H copy.
        CUDA_CHECK(cudaHostAlloc(reinterpret_cast<void**>(&h_pin_status),
                                 2 * sizeof(uint32_t), cudaHostAllocMapped));
        h_pin_status[0] = 0u;
        h_pin_status[1] = 0u;
        CUDA_CHECK(cudaHostGetDevicePointer(reinterpret_cast<void**>(&d_status),
                                            h_pin_status, 0));
        d_out_total = d_status + 0;
        CUDA_CHECK(cudaMalloc(&d_n_overflow, sizeof(uint32_t)));
        CUDA_CHECK(cudaMemset(d_n_overflow, 0, sizeof(uint32_t)));
        bool gpu_pretok_ctx = true;
#if GBPE_HAVE_FAMILY_SP && !GBPE_GEMMA_GPU_PRETOK
        gpu_pretok_ctx = !sp_mode_;
#endif
        if (gpu_pretok_ctx) {
            d_pretok_errors = d_status + 1;   // mapped status block, see above
            CUDA_CHECK(cudaMalloc(&d_overflow_start,
                                  static_cast<size_t>(cap_pretokens) * sizeof(uint32_t)));
            CUDA_CHECK(cudaMalloc(&d_overflow_len,
                                  static_cast<size_t>(cap_pretokens) * sizeof(uint32_t)));
            CUDA_CHECK(cudaMalloc(&d_overflow_orig,
                                  static_cast<size_t>(cap_pretokens) * sizeof(uint32_t)));
            CUDA_CHECK(cudaMalloc(&d_overflow_parts, capb * sizeof(uint32_t)));
            CUDA_CHECK(cudaMalloc(&d_overflow_prev,  capb * sizeof(uint32_t)));
            CUDA_CHECK(cudaMalloc(&d_overflow_next,  capb * sizeof(uint32_t)));
            CUDA_CHECK(cudaMalloc(&d_overflow_tree,
                                  capb * 4 * sizeof(uint64_t)));
            CUDA_CHECK(cudaMemset(d_pretok_errors, 0, sizeof(uint32_t)));
        }
        CUDA_CHECK(cudaMemset(d_n_short, 0, sizeof(uint32_t)));
        CUDA_CHECK(cudaMemset(d_n_long,  0, sizeof(uint32_t)));
        CUDA_CHECK(cudaMemset(d_n_total, 0, sizeof(uint32_t)));
        // d_out_total is the mapped status word, zeroed on the host above.
#if GBPE_HAVE_FAMILY_SP && GBPE_GEMMA_GPU_PRETOK
        if (sp_mode_) {
            CUDA_CHECK(cudaMalloc(&d_sp_flat_ids, capb * sizeof(uint32_t)));
            CUDA_CHECK(cudaMalloc(&d_sp_segment_start, capb1 * sizeof(uint32_t)));
        }
#endif

        // ---- Batched-encode plumbing. cap_docs+1 offsets (the trailing entry
        // is the total). Initialized for a single EMPTY document so the len=0
        // warmup + graph capture see a well-defined batch (mirrors the
        // d_text_len zero-init above) and every doc-indexed kernel no-ops.
        const size_t capd1 = static_cast<size_t>(cap_docs) + 1;
        // n_docs and the byte offsets live in the header block (initialized
        // above to one empty document).
        d_n_docs = d_hdr + 1;
        d_doc_byte_offsets = d_hdr + 2;
        CUDA_CHECK(cudaMalloc(&d_doc_token_offsets, capd1 * sizeof(uint32_t)));
        CUDA_CHECK(cudaMemset(d_doc_token_offsets, 0, capd1 * sizeof(uint32_t)));
        // Dense per-codepoint doc-start array: allocated now, written by
        // pretok_gather_cp and consumed by the doc-boundary clamps.
        // Zero-init so nothing reads uninitialized memory in the meantime.
        CUDA_CHECK(cudaMalloc(&d_cp_doc_start, capb * sizeof(uint32_t)));
        CUDA_CHECK(cudaMemset(d_cp_doc_start, 0, capb * sizeof(uint32_t)));

        // Pinned staging for raw text + length.
        CUDA_CHECK(cudaMallocHost(&h_pin_text_bytes, capb));
        CUDA_CHECK(cudaMallocHost(&h_pin_hdr, (2 + capd1) * sizeof(uint32_t)));
        std::memset(h_pin_hdr, 0, (2 + capd1) * sizeof(uint32_t));
        h_pin_text_len = h_pin_hdr + 0;
        h_pin_n_docs = h_pin_hdr + 1;
        h_pin_doc_byte_offsets = h_pin_hdr + 2;
        CUDA_CHECK(cudaMallocHost(&h_pin_doc_token_offsets, capd1 * sizeof(uint32_t)));
        CUDA_CHECK(cudaMallocHost(&h_pin_out_tokens,
                                  static_cast<size_t>(cap_output_tokens) * sizeof(uint32_t)));
        std::memset(h_pin_doc_byte_offsets, 0, capd1 * sizeof(uint32_t));
        std::memset(h_pin_doc_token_offsets, 0, capd1 * sizeof(uint32_t));
        *h_pin_n_docs = 1;

        // CUB temp for the pretok scans: take the maximum requirement.
        size_t need = 0, b = 0;
        auto bump = [&](){ if (b > need) need = b; b = 0; };
        cub::DeviceScan::ExclusiveSum(nullptr, b, d_is_start, d_cp_index,
                                      (int)(cap_input_bytes + 1), /*stream=*/0);
        bump();
#if GBPE_HAVE_FAMILY_SP && GBPE_GEMMA_GPU_PRETOK
        {
            U8ToU32Iterator emit_len_u32(d_byte_class, U8ToU32{});
            cub::DeviceScan::ExclusiveSum(nullptr, b, emit_len_u32, d_cp_index,
                                          (int)(cap_input_bytes + 1), 0);
            bump();
        }
#endif
        cub::DeviceScan::ExclusiveSum(nullptr, b, d_is_boundary, d_tok_orig_idx,
                                      (int)cap_input_bytes, 0);
        bump();
        // The short/long bucket offsets are scanned together as one pair-valued
        // sequence (one graph node instead of two), so size the workspace for
        // that scan, not for two uint32 scans.
        {
            gbpe::ShortLongZipIn zin{d_is_short, d_is_long};
            gbpe::ShortLongZipOut zout{d_short_local, d_long_local};
            cub::DeviceScan::ExclusiveSum(nullptr, b, zin, zout,
                                          (int)cap_input_bytes, 0);
            bump();
        }
        // Digit-run-start scan (Llama 3-cap / DeepSeek pass1): InclusiveScan(Max).
        cub::DeviceScan::InclusiveScan(nullptr, b, d_drun_seed, d_drun_start,
                                       gbpe::MaxOp{}, (int)cap_input_bytes, 0);
        bump();
        pt_scan_temp_bytes = need;
        CUDA_CHECK(cudaMalloc(&d_pt_scan_temp, pt_scan_temp_bytes));

        // Upload the classification table into __constant__ memory (once,
        // outside capture). gbpe_classify_dev() reads these on device.
        CUDA_CHECK(cudaMemcpyToSymbol(d_gbpe_class_leaves, gbpe_class_leaves,
                                      sizeof(gbpe_class_leaves)));
        CUDA_CHECK(cudaMemcpyToSymbol(d_gbpe_class_blocks, gbpe_class_blocks,
                                      sizeof(gbpe_class_blocks)));
#if GBPE_HAVE_VOCAB_DEEPSEEK_V3
        // Wide property-bit table (DeepSeek): \p{L}\p{N}\s\p{M}\p{P}\p{S}. MUST be
        // uploaded or gbpe_class_bits_dev() reads zeros -> every cp is gap.
        CUDA_CHECK(cudaMemcpyToSymbol(d_gbpe_wide_leaves, gbpe_wide_leaves,
                                      sizeof(gbpe_wide_leaves)));
        CUDA_CHECK(cudaMemcpyToSymbol(d_gbpe_wide_blocks, gbpe_wide_blocks,
                                      sizeof(gbpe_wide_blocks)));
#endif
        // d_class_blocks/d_class_leaves members stay nullptr (unused).
    }
#endif

    cudaStream_t s;
    CUDA_CHECK(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking));
    stream_ = s;
    cudaEvent_t e0, e1;
    CUDA_CHECK(cudaEventCreate(&e0));
    CUDA_CHECK(cudaEventCreate(&e1));
    ev_start_ = e0;
    ev_stop_  = e1;
    {
        cudaEvent_t w;
        CUDA_CHECK(cudaEventCreateWithFlags(
            &w, cudaEventBlockingSync | cudaEventDisableTiming));
        ev_wait_ = w;
        const char* mode = std::getenv("GBPE_GPU_WAIT");
        blocking_wait_ = mode != nullptr && std::strcmp(mode, "block") == 0;
    }
    // Side streams + fork/join events for the parallel BPE branches. The
    // events carry no timing so they stay cheap inside the captured graph.
    for (int k = 0; k < 2; ++k) {
        cudaStream_t a;
        CUDA_CHECK(cudaStreamCreateWithFlags(&a, cudaStreamNonBlocking));
        aux_stream_[k] = a;
        cudaEvent_t j;
        CUDA_CHECK(cudaEventCreateWithFlags(&j, cudaEventDisableTiming));
        join_ev_[k] = j;
    }
    cudaEvent_t f;
    CUDA_CHECK(cudaEventCreateWithFlags(&f, cudaEventDisableTiming));
    fork_ev_ = f;

    GBPE_LOG(
        "[ctx] cap_short=%u cap_long=%u (total=%u) cap_out=%u  scan_temp=%zu B\n",
        cap_short, cap_long, cap_pretokens, cap_output_tokens, scan_temp_bytes);
}

void TokenizerCtx::prepare_launch_config() {
    if (launch_config_ready_) return;
    LaunchConfig& config = launch_config_;
    const uint32_t short_capacity_grid =
        (cap_short + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
    const uint32_t long_capacity_grid  = (cap_long > 0)
        ? (cap_long + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK
        : 0;
    config.flatten_threads = flatten_threads_from_env();
    const uint32_t flatten_warps = config.flatten_threads / WARP_SIZE;
    const uint32_t short_flatten_capacity_grid =
        (cap_short + flatten_warps - 1) / flatten_warps;
    const uint32_t long_flatten_capacity_grid = (cap_long > 0)
        ? (cap_long + flatten_warps - 1) / flatten_warps
        : 0;
    int active_device = 0;
    int sm_count_int = 0;
    CUDA_CHECK(cudaGetDevice(&active_device));
    CUDA_CHECK(cudaDeviceGetAttribute(
        &sm_count_int, cudaDevAttrMultiProcessorCount, active_device));
    const uint32_t sm_count = static_cast<uint32_t>(sm_count_int);
    const uint32_t grid_scale = work_grid_scale(cap_short);
    // The sqrt(capacity) progression preserves enough independent warps for
    // long contexts without returning to a capacity-sized launch. The 100k
    // and 1M endpoints are 192/24 and 512/64 CTAs per SM respectively.
    const uint32_t default_bpe_blocks_per_sm =
        192 + 160 * (grid_scale - 1);
    const uint32_t default_flatten_blocks_per_sm =
        24 + 20 * (grid_scale - 1);
    config.short_grid = bounded_grid_from_env(
        "GBPE_BPE_BLOCKS_PER_SM", short_capacity_grid, sm_count,
        default_bpe_blocks_per_sm);
    config.long_grid = long_capacity_grid > 0
        ? bounded_grid_from_env(
              "GBPE_BPE_BLOCKS_PER_SM", long_capacity_grid, sm_count,
              default_bpe_blocks_per_sm)
        : 0;
    config.short_flatten_grid = bounded_grid_from_env(
        "GBPE_FLATTEN_BLOCKS_PER_SM", short_flatten_capacity_grid, sm_count,
        default_flatten_blocks_per_sm);
    config.long_flatten_grid = long_flatten_capacity_grid > 0
        ? bounded_grid_from_env(
              "GBPE_FLATTEN_BLOCKS_PER_SM",
              long_flatten_capacity_grid,
              sm_count,
              default_flatten_blocks_per_sm)
        : 0;
    GBPE_LOG(
        "[ctx] work grids bpe=%u/%u flatten=%u/%u@%uT "
        "(capacity %u/%u, %u/%u)\n",
        config.short_grid, config.long_grid,
        config.short_flatten_grid, config.long_flatten_grid,
        config.flatten_threads,
        short_capacity_grid, long_capacity_grid,
        short_flatten_capacity_grid, long_flatten_capacity_grid);

#if GBPE_GPU_PRETOK
    // GPU pre-tokenizer configuration is fixed for this context and capacity.
#if GBPE_HAVE_FAMILY_SP
#if GBPE_GEMMA_GPU_PRETOK
    config.use_gpu_pretok = true;
#else
    config.use_gpu_pretok = !sp_mode_;
#endif
#else
    config.use_gpu_pretok = true;
#endif
    config.pretok_grid = (cap_input_bytes + 255u) / 256u;
    config.sp_pretok_grid = (cap_input_bytes + 1u + 255u) / 256u;
    config.doc_offsets_grid = (cap_docs + 1u + 255u) / 256u;
    config.direct_added_grid = (cap_input_bytes + 255u) / 256u;
    if (config.use_gpu_pretok) {
        config.n_short_ptr = d_n_short;
        config.n_long_ptr = d_n_long;
    }
    config.emit_doc_offsets = config.use_gpu_pretok;
    {
        int smem_optin = 0;
        CUDA_CHECK(cudaDeviceGetAttribute(
            &smem_optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, active_device));
        // The kernel's static shared memory (its block scans) counts
        // against the same per-block limit.
        cudaFuncAttributes tiny_attr;
        CUDA_CHECK(cudaFuncGetAttributes(&tiny_attr, pretok_tiny_kernel));
        const size_t smem = tiny_pretok_smem_bytes(cap_input_bytes);
        config.tiny_pretok = config.use_gpu_pretok &&
            cap_input_bytes <= TINY_PRETOK_THREADS * TINY_PRETOK_ITEMS &&
            smem + tiny_attr.sharedSizeBytes <= static_cast<size_t>(smem_optin);
#if GBPE_HAVE_FAMILY_SP
        if (sp_mode_) config.tiny_pretok = false;
#endif
        if (config.tiny_pretok) {
            CUDA_CHECK(cudaFuncSetAttribute(pretok_tiny_kernel,
                cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(smem)));
        }
    }
#if GBPE_HAVE_FAMILY_SP
    config.emit_doc_offsets = config.emit_doc_offsets && !sp_mode_;
#endif
#endif  // GBPE_GPU_PRETOK
    launch_config_ready_ = true;
}

void TokenizerCtx::capture_graph(void* eager_stream) {
    cudaStream_t s = eager_stream != nullptr
        ? static_cast<cudaStream_t>(eager_stream)
        : static_cast<cudaStream_t>(stream_);
    prepare_launch_config();
    const LaunchConfig& config = launch_config_;
    const uint32_t short_grid = config.short_grid;
    const uint32_t long_grid = config.long_grid;
    const uint32_t flatten_threads = config.flatten_threads;
    const uint32_t short_flatten_grid = config.short_flatten_grid;
    const uint32_t long_flatten_grid = config.long_flatten_grid;

    // Device-count pointers for length-bounded dead-slot skipping. Their
    // values are cached with the launch plan; no device query or grid
    // selection occurs on a measured eager replay.
    const uint32_t* n_short_ptr = nullptr;
    const uint32_t* n_long_ptr = nullptr;
#if GBPE_GPU_PRETOK
    const bool use_gpu_pretok = config.use_gpu_pretok;
    const bool emit_doc_offsets = config.emit_doc_offsets;
    n_short_ptr = config.n_short_ptr;
    n_long_ptr = config.n_long_ptr;
#endif

    const DirectTokenArgs direct_args{
        vocab.direct_token_hashes,
        vocab.direct_token_ids,
        vocab.direct_token_mask,
        vocab.vocab_token_bytes_concat,
        vocab.vocab_token_offset,
        vocab.vocab_token_len,
    };

    // Lambda that dispatches a BPE kernel launch. In multi-slot builds it
    // branches on vocab.kind at runtime. In single-slot builds the guard
    // strips the dead branch so only one Slot template is instantiated. SP-
    // family vocabs use the kFromIds=true overload and the uint32 id buffer;
    // byte-level family uses kFromIds=false with the uint8 byte buffer.
    auto launch_short = [&](uint32_t grid, uint32_t n) {
#if GBPE_HAVE_FAMILY_SP
        if (sp_mode_) {
            Slot128::Args sa{vocab.keys128, vocab.vals128, vocab.merge_table_mask};
            bpe_kernel<MAX_PRETOKEN_LEN_SHORT, Slot128, /*kFromIds=*/true>
                <<<grid, BLOCK_THREADS, 0, s>>>(
                    d_short_ids, d_short_lens, d_short_orig_idx, n,
                    vocab.byte_to_id, sa,
                    direct_args,
                    d_per_pre_tokens_short, d_per_pre_count, n_short_ptr);
            return;
        }
#endif
#if !GBPE_USE_SLOT128_ONLY
        if (vocab.kind == SlotKind::Slot64) {
            Slot64::Args sa{vocab.slots64, vocab.merge_table_mask};
            bpe_kernel<MAX_PRETOKEN_LEN_SHORT, Slot64>
                <<<grid, BLOCK_THREADS, 0, s>>>(
                    d_short_bytes, d_short_lens, d_short_orig_idx, n,
                    vocab.byte_to_id, sa,
                    direct_args,
                    d_per_pre_tokens_short, d_per_pre_count, n_short_ptr);
            return;
        }
#endif
#if !GBPE_USE_SLOT64_ONLY
        Slot128::Args sa{vocab.keys128, vocab.vals128, vocab.merge_table_mask};
        bpe_kernel<MAX_PRETOKEN_LEN_SHORT, Slot128>
            <<<grid, BLOCK_THREADS, 0, s>>>(
                d_short_bytes, d_short_lens, d_short_orig_idx, n,
                vocab.byte_to_id, sa, direct_args,
                d_per_pre_tokens_short, d_per_pre_count, n_short_ptr);
#endif
    };
    auto launch_long = [&](uint32_t grid, uint32_t n) {
#if GBPE_HAVE_FAMILY_SP
        if (sp_mode_) {
            Slot128::Args sa{vocab.keys128, vocab.vals128, vocab.merge_table_mask};
            bpe_kernel<MAX_PRETOKEN_LEN_LONG, Slot128, /*kFromIds=*/true>
                <<<grid, BLOCK_THREADS, 0, s>>>(
                    d_long_ids, d_long_lens, d_long_orig_idx, n,
                    vocab.byte_to_id, sa,
                    direct_args,
                    d_per_pre_tokens_long, d_per_pre_count, n_long_ptr);
            return;
        }
#endif
#if !GBPE_USE_SLOT128_ONLY
        if (vocab.kind == SlotKind::Slot64) {
            Slot64::Args sa{vocab.slots64, vocab.merge_table_mask};
            bpe_kernel<MAX_PRETOKEN_LEN_LONG, Slot64>
                <<<grid, BLOCK_THREADS, 0, s>>>(
                    d_long_bytes, d_long_lens, d_long_orig_idx, n,
                    vocab.byte_to_id, sa,
                    direct_args,
                    d_per_pre_tokens_long, d_per_pre_count, n_long_ptr);
            return;
        }
#endif
#if !GBPE_USE_SLOT64_ONLY
        Slot128::Args sa{vocab.keys128, vocab.vals128, vocab.merge_table_mask};
        bpe_kernel<MAX_PRETOKEN_LEN_LONG, Slot128>
            <<<grid, BLOCK_THREADS, 0, s>>>(
                d_long_bytes, d_long_lens, d_long_orig_idx, n,
                vocab.byte_to_id, sa, direct_args,
                d_per_pre_tokens_long, d_per_pre_count, n_long_ptr);
#endif
    };

#if GBPE_GPU_PRETOK
    auto launch_gpu_overflow = [&]() {
        if (!use_gpu_pretok) return;
#if GBPE_HAVE_FAMILY_SP && GBPE_GEMMA_GPU_PRETOK
        if (sp_mode_) {
            Slot128::Args sa{vocab.keys128, vocab.vals128, vocab.merge_table_mask};
            overflow_bpe_kernel<Slot128, /*kFromIds=*/true>
                <<<OVERFLOW_BPE_GRID, OVERFLOW_BPE_THREADS, 0, s>>>(
                    d_sp_flat_ids, d_overflow_start, d_overflow_len,
                    d_overflow_orig, d_n_overflow, vocab.byte_to_id, sa,
                    direct_args,
                    d_overflow_parts, d_overflow_prev, d_overflow_next,
                    d_overflow_tree, d_per_pre_count, d_pretok_errors);
            return;
        }
#endif
#if !GBPE_USE_SLOT128_ONLY
        if (vocab.kind == SlotKind::Slot64) {
            Slot64::Args sa{vocab.slots64, vocab.merge_table_mask};
            overflow_bpe_kernel<Slot64>
                <<<OVERFLOW_BPE_GRID, OVERFLOW_BPE_THREADS, 0, s>>>(
                    d_text_bytes, d_overflow_start, d_overflow_len,
                    d_overflow_orig, d_n_overflow, vocab.byte_to_id, sa,
                    direct_args,
                    d_overflow_parts, d_overflow_prev, d_overflow_next,
                    d_overflow_tree, d_per_pre_count, d_pretok_errors);
            return;
        }
#endif
#if !GBPE_USE_SLOT64_ONLY
        Slot128::Args sa{vocab.keys128, vocab.vals128, vocab.merge_table_mask};
        overflow_bpe_kernel<Slot128>
            <<<OVERFLOW_BPE_GRID, OVERFLOW_BPE_THREADS, 0, s>>>(
                d_text_bytes, d_overflow_start, d_overflow_len,
                d_overflow_orig, d_n_overflow, vocab.byte_to_id, sa,
                direct_args, d_overflow_parts, d_overflow_prev,
                d_overflow_next, d_overflow_tree, d_per_pre_count,
                d_pretok_errors);
#endif
    };
#endif

    // One flatten launch per bucket: short rows (32-wide) and long rows
    // (64-wide), each mapping bucket-local index -> orig_idx for the
    // count/scan/destination lookups.
    auto launch_flatten = [&](uint32_t short_blocks,
                              uint32_t long_blocks,
                              uint32_t short_rows,
                              uint32_t long_rows,
                              bool with_doc_offsets = false) {
#if GBPE_GPU_PRETOK
        // Byte-level GPU-pretok path: one partitioned launch for all stages.
        // The SP/Gemma front end keeps the separate launches.
        bool fused = use_gpu_pretok;
#if GBPE_HAVE_FAMILY_SP
        if (sp_mode_) fused = false;
#endif
        if (fused) {
            const uint32_t sb = (short_rows > 0) ? short_blocks : 0u;
            const uint32_t lb = (cap_long > 0 && long_rows > 0) ? long_blocks : 0u;
            const uint32_t ob = OVERFLOW_BPE_BLOCKS;
            const uint32_t db = (cap_input_bytes + flatten_threads - 1u) / flatten_threads;
            // One more partition covers cap_docs+1 documents, folding what was
            // a separate pretok_doc_token_offsets node into this launch.
            const uint32_t fb = with_doc_offsets
                ? (cap_docs + 1u + flatten_threads - 1u) / flatten_threads : 0u;
            flatten_all_kernel<MAX_PRETOKEN_LEN_SHORT, MAX_PRETOKEN_LEN_LONG>
                <<<sb + lb + ob + db + fb, flatten_threads, 0, s>>>(
                d_per_pre_tokens_short, d_short_orig_idx, d_short_lens, short_rows,
                n_short_ptr, sb,
                d_per_pre_tokens_long, d_long_orig_idx, d_long_lens, long_rows,
                n_long_ptr, lb,
                d_overflow_parts, d_overflow_start, d_overflow_orig, d_n_overflow, ob,
                d_cls, d_tok_orig_idx, d_ncp, d_direct_ids, cap_input_bytes, db,
                d_doc_byte_offsets, d_n_docs, cap_docs, d_cp_index, d_n_total,
                d_out_total, d_doc_token_offsets, fb,
                d_per_pre_count, d_scan_out, d_out_tokens);
            return;
        }
#endif
        if (short_blocks > 0 && short_rows > 0) {
            flatten_kernel_bucketed<MAX_PRETOKEN_LEN_SHORT>
                <<<short_blocks, flatten_threads, 0, s>>>(
                d_per_pre_tokens_short, d_short_orig_idx, d_short_lens,
                d_per_pre_count, d_scan_out, d_out_tokens,
                short_rows, n_short_ptr);
        }
        if (cap_long > 0 && long_blocks > 0 && long_rows > 0) {
            flatten_kernel_bucketed<MAX_PRETOKEN_LEN_LONG>
                <<<long_blocks, flatten_threads, 0, s>>>(
                d_per_pre_tokens_long, d_long_orig_idx, d_long_lens,
                d_per_pre_count, d_scan_out, d_out_tokens,
                long_rows, n_long_ptr);
        }
#if GBPE_HAVE_FAMILY_SP
        if (sp_mode_ && cap_overflow > 0) {
            flatten_overflow_kernel<<<cap_overflow, 64, 0, s>>>(
                d_per_pre_tokens_overflow, d_overflow_orig_idx, d_overflow_n,
                d_per_pre_count, d_scan_out, d_out_tokens);
        }
#endif
#if GBPE_GPU_PRETOK
        if (use_gpu_pretok) {
            flatten_gpu_overflow_kernel<<<OVERFLOW_BPE_BLOCKS, 64, 0, s>>>(
                d_overflow_parts, d_overflow_start, d_overflow_orig,
                d_n_overflow, d_per_pre_count, d_scan_out, d_out_tokens);
#if GBPE_HAVE_FAMILY_SP && GBPE_GEMMA_GPU_PRETOK
            if (!sp_mode_)
#endif
            {
                flatten_direct_added_kernel<<<config.direct_added_grid, 256, 0, s>>>(
                    d_cls, d_tok_orig_idx, d_ncp, d_direct_ids,
                    d_scan_out, cap_input_bytes, d_out_tokens);
            }
        }
#endif
    };

#if GBPE_GPU_PRETOK
    // GPU pre-tokenizer stage. Reads raw UTF-8 from d_text_bytes (+ d_text_len)
    // and produces the bucketed inputs (d_short_*/d_long_*) the host pretok
    // produces today. All counts are device-only; grids are cap-sized; tails
    // zeroed in encode(). Everything is graph-static (no alloc/D2H/sync).
    auto launch_pretok = [&]() {
        const uint32_t capb = cap_input_bytes;
        const uint32_t bgrid = config.pretok_grid;
        const bool fold_long_s = vocab.pretok_kind == PretokKind::Llama3 ||
                                 vocab.pretok_kind == PretokKind::Qwen25;
        // The digit-run-start scan is shared by the families that need a
        // digit cap (Llama 3-cap, DeepSeek pass1); its seed is written by the
        // gather kernel into d_is_short (free until k3; d_drun_seed carries
        // added-token IDs until then).
        const bool needs_digit_scan =
            vocab.pretok_kind == PretokKind::Llama3 ||
            vocab.pretok_kind == PretokKind::Qwen25 ||
            vocab.pretok_kind == PretokKind::DeepSeekV3;
        if (config.tiny_pretok) {
            const TinyPretokArgs ta{
                d_text_bytes, d_text_len, capb,
                vocab.added_root, vocab.added_node_edge_begin,
                vocab.added_node_edge_count, vocab.added_node_token_id,
                vocab.added_edges, vocab.added_trie_nodes, vocab.added_max_bytes,
                d_drun_seed, d_drun_start, d_is_long, d_is_short,
                d_tok_orig_idx, d_is_start, d_byte_class, d_cp_index,
                d_cls, d_cp_byte_pos, d_cp_byte0, d_ncp,
                d_cp_doc_start, d_is_boundary, d_short_local, d_long_local,
#if GBPE_HAVE_VOCAB_DEEPSEEK_V3
                d_byte_cp, d_cp_cp,
#else
                nullptr, nullptr,
#endif
                d_doc_byte_offsets, d_n_docs,
                d_n_overflow, d_pretok_errors,
                d_per_pre_count, d_direct_ids,
                d_short_bytes, d_short_lens, d_short_orig_idx,
                d_long_bytes, d_long_lens, d_long_orig_idx,
                d_overflow_start, d_overflow_len, d_overflow_orig,
                cap_short, cap_long, cap_pretokens,
                d_n_short, d_n_long, d_n_total,
                vocab.pretok_kind, fold_long_s, needs_digit_scan};
            pretok_tiny_kernel<<<1, TINY_PRETOK_THREADS,
                                 tiny_pretok_smem_bytes(capb), s>>>(ta);
            return;
        }
#if GBPE_HAVE_FAMILY_SP && GBPE_GEMMA_GPU_PRETOK
        // The SP path keeps its own reset; the byte-level path resets in k1.
        if (sp_mode_) CUDA_CHECK(cudaMemsetAsync(d_pretok_errors, 0, sizeof(uint32_t), s));
#endif
        added_match<<<bgrid, 256, 0, s>>>(
            d_text_bytes, d_text_len, capb, vocab.added_root,
            vocab.added_node_edge_begin, vocab.added_node_edge_count,
            vocab.added_node_token_id, vocab.added_edges,
            vocab.added_trie_nodes, vocab.added_max_bytes,
            d_drun_seed, d_is_long, d_tok_orig_idx,
            d_doc_byte_offsets, d_n_docs);
        added_expand_owners<<<bgrid, 256, 0, s>>>(
            d_is_long, capb, d_tok_orig_idx);
#if GBPE_HAVE_FAMILY_SP && GBPE_GEMMA_GPU_PRETOK
        if (sp_mode_) {
            const uint32_t bgrid1 = config.sp_pretok_grid;
            sp_added_run_seed<<<bgrid, 256, 0, s>>>(
                d_text_bytes, d_text_len, capb, d_drun_start);
            cub::DeviceScan::InclusiveScan(
                d_pt_scan_temp, pt_scan_temp_bytes,
                d_drun_start, d_is_short, gbpe::MaxOp{}, (int)capb, s);
            sp_pretok_analyze<<<bgrid1, 256, 0, s>>>(
                d_text_bytes, d_text_len, capb, vocab.sp_codepoint_to_id,
                d_tok_orig_idx, d_is_short, d_byte_class, d_is_boundary);
            U8ToU32Iterator emit_len_u32(d_byte_class, U8ToU32{});
            cub::DeviceScan::ExclusiveSum(d_pt_scan_temp, pt_scan_temp_bytes,
                emit_len_u32, d_cp_index, (int)(capb + 1), s);
            cub::DeviceScan::ExclusiveSum(d_pt_scan_temp, pt_scan_temp_bytes,
                d_is_boundary, d_tok_orig_idx, (int)capb, s);
            sp_pretok_emit_ids<<<bgrid, 256, 0, s>>>(
                d_text_bytes, d_text_len, capb, vocab.sp_codepoint_to_id,
                vocab.sp_byte_fallback, d_drun_seed, d_is_short,
                vocab.sp_lowbar_run_ids,
                d_byte_class, d_cp_index, d_sp_flat_ids);
            sp_pretok_segment_starts<<<bgrid, 256, 0, s>>>(
                d_is_boundary, d_tok_orig_idx, d_cp_index, capb,
                d_sp_segment_start, d_n_total);
            CUDA_CHECK(cudaMemsetAsync(d_n_overflow, 0, sizeof(uint32_t), s));
            sp_pretok_classify_segments<<<bgrid, 256, 0, s>>>(
                d_sp_segment_start, d_n_total, capb, d_is_short, d_is_long,
                d_per_pre_count, d_n_overflow);
            cub::DeviceScan::ExclusiveSum(d_pt_scan_temp, pt_scan_temp_bytes,
                d_is_short, d_short_local, (int)capb, s);
            cub::DeviceScan::ExclusiveSum(d_pt_scan_temp, pt_scan_temp_bytes,
                d_is_long, d_long_local, (int)capb, s);
            sp_pretok_materialize_buckets<<<bgrid, 256, 0, s>>>(
                d_sp_flat_ids, d_sp_segment_start, d_n_total,
                d_is_short, d_is_long, d_short_local, d_long_local,
                capb, cap_short, cap_long,
                d_short_ids, d_short_lens, d_short_orig_idx,
                d_long_ids, d_long_lens, d_long_orig_idx,
                d_overflow_start, d_overflow_len, d_overflow_orig,
                cap_pretokens, d_pretok_errors);
            pretok_extract_counts<<<1, 1, 0, s>>>(
                d_short_local, d_is_short, d_long_local, d_is_long,
                d_tok_orig_idx, d_is_boundary, capb,
                cap_short, cap_long, cap_pretokens,
                d_n_short, d_n_long, d_n_total, d_n_overflow,
                d_pretok_errors);
            return;
        }
#endif
        pretok_k1_classify<<<bgrid, 256, 0, s>>>(
            d_text_bytes, d_text_len, capb, d_tok_orig_idx,
            d_is_start, d_byte_class,
            d_n_overflow, d_pretok_errors, d_doc_byte_offsets, d_n_docs
#if GBPE_HAVE_VOCAB_DEEPSEEK_V3
            , d_byte_cp
#endif
            );
        cub::DeviceScan::ExclusiveSum(d_pt_scan_temp, pt_scan_temp_bytes,
            d_is_start, d_cp_index, (int)(capb + 1), s);
        pretok_gather_cp<<<bgrid, 256, 0, s>>>(
            d_text_bytes, d_text_len, capb, d_is_start, d_byte_class,
            d_cp_index,
            d_cls, d_cp_byte_pos, d_cp_byte0, d_ncp, fold_long_s,
            d_doc_byte_offsets, d_n_docs, d_cp_doc_start,
            needs_digit_scan ? d_is_short : nullptr
#if GBPE_HAVE_VOCAB_DEEPSEEK_V3
            , d_byte_cp, d_cp_cp
#endif
            );
        // Boundary stage — the per-family seam, dispatched on pretok_kind and
        // guarded so a single-vocab build compiles only its own kernel (mirrors
        // the Slot64/128 dispatch).
        if (needs_digit_scan) {
            cub::DeviceScan::InclusiveScan(d_pt_scan_temp, pt_scan_temp_bytes,
                d_is_short, d_drun_start, gbpe::MaxOp{}, (int)capb, s);
        }
        switch (vocab.pretok_kind) {
#if GBPE_HAVE_VOCAB_LLAMA3
            case PretokKind::Llama3:
                pretok_k2_llamaqwen<true><<<bgrid, 256, 0, s>>>(
                    d_cls, d_cp_byte0, d_drun_start, d_ncp, capb,
                    d_is_boundary, d_cp_doc_start);
                break;
#endif
#if GBPE_HAVE_VOCAB_QWEN25
            case PretokKind::Qwen25:
                pretok_k2_llamaqwen<false><<<bgrid, 256, 0, s>>>(
                    d_cls, d_cp_byte0, d_drun_start, d_ncp, capb,
                    d_is_boundary, d_cp_doc_start);
                break;
#endif
#if GBPE_HAVE_VOCAB_DEEPSEEK_V3
            case PretokKind::DeepSeekV3: {
                // Materialize the pass-1/pass-2 span start for every codepoint.
                // d_is_short is temporary until k3 and d_short_local is not
                // needed until the later bucket scan, so this stays allocation-
                // free while preserving d_drun_seed's added-token IDs.
                pretok_deepseek_span_seed<<<bgrid, 256, 0, s>>>(
                    d_cp_cp, d_cls, d_drun_start, d_ncp, capb, d_is_short,
                    d_cp_doc_start);
                cub::DeviceScan::InclusiveScan(
                    d_pt_scan_temp, pt_scan_temp_bytes,
                    d_is_short, d_short_local, gbpe::MaxOp{}, (int)capb, s);
                pretok_k2_deepseek<<<bgrid, 256, 0, s>>>(
                    d_ncp, capb, d_is_boundary, d_cp_cp, d_cls,
                    d_is_short, d_short_local, d_cp_doc_start);
                break;
            }
#endif
#if GBPE_HAVE_VOCAB_GPT2
            case PretokKind::GPT2:
                pretok_k2_boundaries<<<bgrid, 256, 0, s>>>(
                    d_cls, d_cp_byte0, d_ncp, capb, d_is_boundary,
                    d_cp_doc_start);
                break;
#endif
            default: break;  // SP/Gemma have no byte-level GPU boundary stage
        }
        cub::DeviceScan::ExclusiveSum(d_pt_scan_temp, pt_scan_temp_bytes,
            d_is_boundary, d_tok_orig_idx, (int)capb, s);
        pretok_k3_lenbucket<<<bgrid, 256, 0, s>>>(
            d_is_boundary, d_cp_byte_pos, d_cls, d_tok_orig_idx,
            d_drun_seed, d_ncp, d_text_len, capb,
            d_is_short, d_is_long, d_drun_start, d_n_overflow,
            d_per_pre_count, d_direct_ids,
            d_doc_byte_offsets, d_n_docs);
        // The short and long bucket offsets are two independent exclusive sums
        // over the same domain, both produced by k3 and both consumed by k4.
        // Scanning them as one pair-valued sequence costs one graph node
        // instead of two (a node is ~1.2 us of launch here, and each CUB scan
        // is really two nodes: a tile-status init plus the scan). The zipped
        // iterators read and write the existing separate arrays, so no buffer
        // changes and no extra pass.
        {
            ShortLongZipIn zin{d_is_short, d_is_long};
            ShortLongZipOut zout{d_short_local, d_long_local};
            cub::DeviceScan::ExclusiveSum(d_pt_scan_temp, pt_scan_temp_bytes,
                zin, zout, (int)capb, s);
        }
        pretok_k4_materialize<<<bgrid, 256, 0, s>>>(
            d_text_bytes, d_is_boundary, d_cp_byte_pos, d_tok_orig_idx,
            d_is_short, d_is_long, d_short_local, d_long_local,
            d_drun_start, d_cls, d_ncp, d_text_len,
            d_short_bytes, d_short_lens, d_short_orig_idx,
            d_long_bytes, d_long_lens, d_long_orig_idx,
            d_overflow_start, d_overflow_len, d_overflow_orig,
            cap_short, cap_long, cap_pretokens, d_pretok_errors,
            d_doc_byte_offsets, d_n_docs,
            // live counts (device-only) so bpe/flatten bound by length
            capb, cap_pretokens, d_n_short, d_n_long, d_n_total, d_n_overflow);
    };

    // Per-document token offsets. Byte-level (GPU-pretok) only: the SP front
    // end repurposes d_cp_index / d_tok_orig_idx, so the walk does not apply.
    auto launch_doc_offsets = [&]() {
        pretok_doc_token_offsets<<<config.doc_offsets_grid, 256, 0, s>>>(
            d_doc_byte_offsets, d_n_docs, cap_docs, cap_input_bytes,
            d_cp_index, d_tok_orig_idx, d_ncp, d_n_total,
            d_scan_out, d_out_total, d_doc_token_offsets);
    };
#endif

    // Warm up so module load + JIT happen outside graph capture (or eager
    // timing). Eager mode retains this one-time warmup but directly enqueues
    // the production pipeline below on every invocation.
    if (use_cuda_graph_ || !eager_warmed_) {
#if GBPE_GPU_PRETOK
    if (use_gpu_pretok) launch_pretok();
#endif
    launch_short(1, 0);
    if (cap_long > 0) launch_long(1, 0);
#if GBPE_GPU_PRETOK
    if (use_gpu_pretok) launch_gpu_overflow();
#endif
    cub::DeviceScan::ExclusiveSum(d_scan_temp, scan_temp_bytes,
        d_per_pre_count, d_scan_out, cap_pretokens + 1, s);
#if GBPE_GPU_PRETOK
    if (use_gpu_pretok) {
        if (emit_doc_offsets) launch_doc_offsets();
        else pretok_finalize_total<<<1, 1, 0, s>>>(d_scan_out, d_n_total, d_out_total);
    }
#endif
    launch_flatten(1, (cap_long > 0) ? 1 : 0,
                   1, (cap_long > 0) ? 1 : 0);
    CUDA_CHECK(cudaStreamSynchronize(s));
    }

    if (use_cuda_graph_) {
        CUDA_CHECK(cudaStreamBeginCapture(s, cudaStreamCaptureModeThreadLocal));
    }

#if GBPE_GPU_PRETOK
    if (use_gpu_pretok) launch_pretok();
#endif
    // The three BPE stages read disjoint bucket rows and write disjoint
    // per_pre_count / token rows, so fork them onto side streams: in the
    // graph they become parallel branches, and at large sizes the overflow
    // kernel overlaps the main one instead of serialising behind it.
    {
        cudaStream_t main = s;
        cudaStream_t aux0 = static_cast<cudaStream_t>(aux_stream_[0]);
        cudaStream_t aux1 = static_cast<cudaStream_t>(aux_stream_[1]);
        cudaEvent_t fork = static_cast<cudaEvent_t>(fork_ev_);
        CUDA_CHECK(cudaEventRecord(fork, main));
        CUDA_CHECK(cudaStreamWaitEvent(aux0, fork, 0));
        CUDA_CHECK(cudaStreamWaitEvent(aux1, fork, 0));
        launch_short(short_grid, cap_short);
        if (cap_long > 0) { s = aux0; launch_long(long_grid, cap_long); s = main; }
#if GBPE_GPU_PRETOK
        if (use_gpu_pretok) { s = aux1; launch_gpu_overflow(); s = main; }
#endif
        cudaEvent_t j0 = static_cast<cudaEvent_t>(join_ev_[0]);
        cudaEvent_t j1 = static_cast<cudaEvent_t>(join_ev_[1]);
        CUDA_CHECK(cudaEventRecord(j0, aux0));
        CUDA_CHECK(cudaEventRecord(j1, aux1));
        CUDA_CHECK(cudaStreamWaitEvent(main, j0, 0));
        CUDA_CHECK(cudaStreamWaitEvent(main, j1, 0));
    }

#if GBPE_GPU_PRETOK
    if (config.tiny_pretok) {
        const TinyTailArgs ta{
            d_per_pre_tokens_short, d_short_orig_idx, n_short_ptr,
            d_per_pre_tokens_long, d_long_orig_idx, cap_long > 0 ? n_long_ptr : nullptr,
            d_overflow_parts, d_overflow_start, d_overflow_orig, d_n_overflow,
            d_cls, d_tok_orig_idx, d_ncp, d_direct_ids, cap_input_bytes,
            d_doc_byte_offsets, d_n_docs, cap_docs, d_cp_index, d_n_total,
            d_out_total, d_doc_token_offsets,
            d_per_pre_count, d_scan_out, cap_pretokens, d_out_tokens};
        tiny_tail_kernel<MAX_PRETOKEN_LEN_SHORT, MAX_PRETOKEN_LEN_LONG>
            <<<1, TINY_PRETOK_THREADS, 0, s>>>(ta);
    } else
#endif
    {
        cub::DeviceScan::ExclusiveSum(d_scan_temp, scan_temp_bytes,
            d_per_pre_count, d_scan_out, cap_pretokens + 1, s);
        // The byte-level flatten launch absorbs the doc-offset walk as one more
        // partition (it reads only the finished count scan and writes buffers no
        // other partition touches), so that stage costs no node of its own there.
        bool doc_offsets_fused = false;
#if GBPE_GPU_PRETOK
        if (use_gpu_pretok) {
            doc_offsets_fused = emit_doc_offsets;
#if GBPE_HAVE_FAMILY_SP
            if (sp_mode_) doc_offsets_fused = false;
#endif
            // doc_token_offsets also finalizes d_out_total; the SP path (no doc
            // walk) and the unfused case still need their own node.
            if (doc_offsets_fused) {
                // handled inside launch_flatten below
            } else if (emit_doc_offsets) {
                launch_doc_offsets();
            } else {
                pretok_finalize_total<<<1, 1, 0, s>>>(d_scan_out, d_n_total, d_out_total);
            }
        }
#endif

        launch_flatten(short_flatten_grid, long_flatten_grid,
                       cap_short, cap_long, doc_offsets_fused);
    }

    if (!use_cuda_graph_) {
        eager_warmed_ = true;
        return;
    }

    cudaGraph_t graph;
    CUDA_CHECK(cudaStreamEndCapture(s, &graph));
    graph_ = graph;
    cudaGraphExec_t gex;
    CUDA_CHECK(cudaGraphInstantiate(&gex, graph, nullptr, nullptr, 0));
    graph_ex_ = gex;
    graph_built_ = true;
}

TokenizerCtx::~TokenizerCtx() {
    if (graph_ex_) cudaGraphExecDestroy(static_cast<cudaGraphExec_t>(graph_ex_));
    if (graph_)    cudaGraphDestroy(static_cast<cudaGraph_t>(graph_));
    if (stream_)   cudaStreamDestroy(static_cast<cudaStream_t>(stream_));
    if (ev_start_) cudaEventDestroy(static_cast<cudaEvent_t>(ev_start_));
    if (ev_stop_)  cudaEventDestroy(static_cast<cudaEvent_t>(ev_stop_));
    if (ev_wait_)  cudaEventDestroy(static_cast<cudaEvent_t>(ev_wait_));
    for (int k = 0; k < 2; ++k) {
        if (aux_stream_[k]) cudaStreamDestroy(static_cast<cudaStream_t>(aux_stream_[k]));
        if (join_ev_[k])    cudaEventDestroy(static_cast<cudaEvent_t>(join_ev_[k]));
    }
    if (fork_ev_) cudaEventDestroy(static_cast<cudaEvent_t>(fork_ev_));

#if GBPE_GPU_PRETOK
    cudaFree(d_text_bytes);
    cudaFree(d_text_len);
    cudaFree(d_is_start);
    cudaFree(d_byte_class);
    cudaFree(d_cp_index);
    cudaFree(d_cls);
    cudaFree(d_cp_byte_pos);
    cudaFree(d_cp_byte0);
    cudaFree(d_direct_ids);
    cudaFree(d_is_boundary);
    cudaFree(d_tok_orig_idx);
    cudaFree(d_is_short);
    cudaFree(d_is_long);
    cudaFree(d_short_local);
    cudaFree(d_long_local);
    cudaFree(d_ncp);
    cudaFree(d_pt_scan_temp);
    if (h_pin_text_bytes) cudaFreeHost(h_pin_text_bytes);
    if (h_pin_text_len)   cudaFreeHost(h_pin_text_len);
    // d_doc_byte_offsets / d_n_docs alias d_hdr (freed with d_text_len).
    if (d_doc_token_offsets) cudaFree(d_doc_token_offsets);
    if (d_cp_doc_start)      cudaFree(d_cp_doc_start);
    // h_pin_doc_byte_offsets / h_pin_n_docs alias h_pin_hdr (freed with h_pin_text_len).
    if (h_pin_out_tokens)        cudaFreeHost(h_pin_out_tokens);
    if (h_pin_status)            cudaFreeHost(h_pin_status);
    if (h_pin_doc_token_offsets) cudaFreeHost(h_pin_doc_token_offsets);
#endif
    cudaFree(d_short_bytes);
    cudaFree(d_short_lens);
    cudaFree(d_short_orig_idx);
    cudaFree(d_long_bytes);
    cudaFree(d_long_lens);
    cudaFree(d_long_orig_idx);
    cudaFree(d_per_pre_count);
    cudaFree(d_per_pre_tokens_short);
    cudaFree(d_per_pre_tokens_long);
    cudaFree(d_scan_out);
    cudaFree(d_out_tokens);
    cudaFree(d_scan_temp);
    cudaFreeHost(h_pin_short_bytes);
    cudaFreeHost(h_pin_short_lens);
    cudaFreeHost(h_pin_short_orig_idx);
    cudaFreeHost(h_pin_long_bytes);
    cudaFreeHost(h_pin_long_lens);
    cudaFreeHost(h_pin_long_orig_idx);

    // F5: Decode resources.
    if (dec_graph_ex_) cudaGraphExecDestroy(static_cast<cudaGraphExec_t>(dec_graph_ex_));
    if (dec_graph_)    cudaGraphDestroy(static_cast<cudaGraph_t>(dec_graph_));
    if (dec_ev_start_) cudaEventDestroy(static_cast<cudaEvent_t>(dec_ev_start_));
    if (dec_ev_stop_)  cudaEventDestroy(static_cast<cudaEvent_t>(dec_ev_stop_));
    if (d_dec_token_ids) cudaFree(d_dec_token_ids);
    if (d_dec_n_tokens)  cudaFree(d_dec_n_tokens);
    if (d_dec_lens)      cudaFree(d_dec_lens);
    if (d_dec_offsets)   cudaFree(d_dec_offsets);
    if (d_dec_out_bytes) cudaFree(d_dec_out_bytes);
    if (d_dec_scan_temp) cudaFree(d_dec_scan_temp);
    if (h_pin_dec_token_ids) cudaFreeHost(h_pin_dec_token_ids);

#if GBPE_HAVE_FAMILY_SP
    // F1b: SP-family resources.
    if (d_short_ids)     cudaFree(d_short_ids);
    if (d_long_ids)      cudaFree(d_long_ids);
    if (h_pin_short_ids) cudaFreeHost(h_pin_short_ids);
    if (h_pin_long_ids)  cudaFreeHost(h_pin_long_ids);
    if (d_per_pre_tokens_overflow) cudaFree(d_per_pre_tokens_overflow);
    if (d_overflow_orig_idx)       cudaFree(d_overflow_orig_idx);
    if (d_overflow_n)              cudaFree(d_overflow_n);
#endif
#if GBPE_GPU_PRETOK
    if (d_n_short)   cudaFree(d_n_short);
    if (d_n_long)    cudaFree(d_n_long);
    if (d_n_total)   cudaFree(d_n_total);
    // d_out_total / d_pretok_errors alias the mapped status block.
    if (d_n_overflow) cudaFree(d_n_overflow);
    if (d_overflow_start) cudaFree(d_overflow_start);
    if (d_overflow_len)   cudaFree(d_overflow_len);
    if (d_overflow_orig)  cudaFree(d_overflow_orig);
    if (d_overflow_parts) cudaFree(d_overflow_parts);
    if (d_overflow_prev)  cudaFree(d_overflow_prev);
    if (d_overflow_next)  cudaFree(d_overflow_next);
    if (d_overflow_tree)  cudaFree(d_overflow_tree);
#if GBPE_HAVE_FAMILY_SP && GBPE_GEMMA_GPU_PRETOK
    if (d_sp_flat_ids)      cudaFree(d_sp_flat_ids);
    if (d_sp_segment_start) cudaFree(d_sp_segment_start);
#endif
#if GBPE_HAVE_VOCAB_DEEPSEEK_V3
    if (d_byte_cp) cudaFree(d_byte_cp);
    if (d_cp_cp)   cudaFree(d_cp_cp);
#endif
    if (d_drun_seed)  cudaFree(d_drun_seed);
    if (d_drun_start) cudaFree(d_drun_start);
#endif
}

#if GBPE_GPU_PRETOK
void TokenizerCtx::wait_stream(void* stream)
{
    cudaStream_t s = static_cast<cudaStream_t>(stream);
    if (!blocking_wait_) {
        CUDA_CHECK(cudaStreamSynchronize(s));
        return;
    }
    cudaEvent_t w = static_cast<cudaEvent_t>(ev_wait_);
    CUDA_CHECK(cudaEventRecord(w, s));
    CUDA_CHECK(cudaEventSynchronize(w));
}

const uint32_t* TokenizerCtx::launch_graph_core(const uint8_t* const* docs,
                                                const uint32_t* lens,
                                                uint32_t n_docs,
                                                uint32_t* out_total,
                                                uint32_t* out_doc_offsets,
                                                void* stream)
{
#if GBPE_HAVE_FAMILY_SP && !GBPE_GEMMA_GPU_PRETOK
    if (sp_mode_) {
        throw std::logic_error(
            "launch_graph_core is byte-level only; SP uses host ID buckets");
    }
#endif
#if GBPE_HAVE_FAMILY_SP
    // Batched encode is byte-level only. SP contexts must be rejected for ANY
    // n_docs when per-document offsets are requested: the in-graph
    // pretok_doc_token_offsets node is not part of an SP graph (it walks
    // d_cp_index / d_tok_orig_idx, which the SP front end repurposes), so
    // d_doc_token_offsets would silently read back its zero-init value.
    if (sp_mode_ && (n_docs > 1 || out_doc_offsets != nullptr)) {
        throw std::logic_error(
            "launch_graph_core: batched encode is byte-level only; SP vocabs "
            "do not support encode_batch");
    }
#endif
    // Run on the caller-supplied stream (e.g. torch's current stream, so
    // producing the IDs is ordered with the consumer that reads them and the
    // next encode that reuses d_out_tokens). Used as-is — a null handle is the
    // CUDA default stream, which is a legitimate choice (it is also what torch's
    // current_stream resolves to when no stream context is active).
    cudaStream_t s = static_cast<cudaStream_t>(stream);

    if (n_docs < 1) {
        throw std::runtime_error("encode: n_docs must be >= 1");
    }
    if (n_docs > cap_docs) {
        std::fprintf(stderr, "encode: n_docs %u exceeds cap_docs %u\n",
                     n_docs, cap_docs);
        std::abort();
    }

    // Stage the batch: documents are concatenated back-to-back in the pinned
    // text buffer; doc_byte_offsets[d] is where document d starts.
    uint32_t total_bytes = 0;
    for (uint32_t d = 0; d < n_docs; ++d) {
        h_pin_doc_byte_offsets[d] = total_bytes;
        const uint32_t len = lens[d];
        if (len > cap_input_bytes - total_bytes) {
            std::fprintf(stderr,
                "encode: batch bytes exceed cap_input_bytes %u (doc %u len %u, "
                "staged %u)\n", cap_input_bytes, d, len, total_bytes);
            std::abort();
        }
        if (len) std::memcpy(h_pin_text_bytes + total_bytes, docs[d], len);
        total_bytes += len;
    }
    h_pin_doc_byte_offsets[n_docs] = total_bytes;
    *h_pin_text_len = total_bytes;
    *h_pin_n_docs   = n_docs;

    cudaEvent_t e0 = static_cast<cudaEvent_t>(ev_start_);
    cudaEvent_t e1 = static_cast<cudaEvent_t>(ev_stop_);

    // Two H2D copies: the text and the header (len, n_docs, offsets).
    CUDA_CHECK(cudaMemcpyAsync(d_text_bytes, h_pin_text_bytes, total_bytes,
                               cudaMemcpyHostToDevice, s));
    CUDA_CHECK(cudaMemcpyAsync(d_hdr, h_pin_hdr, (2 + n_docs + 1) * sizeof(uint32_t),
                               cudaMemcpyHostToDevice, s));

    if (use_cuda_graph_) {
        if (!graph_built_) capture_graph();
        CUDA_CHECK(cudaEventRecord(e0, s));
        CUDA_CHECK(cudaGraphLaunch(static_cast<cudaGraphExec_t>(graph_ex_), s));
    } else {
        // The initial direct launch warms modules/JIT outside the timed span.
        if (!eager_warmed_) capture_graph(s);
        CUDA_CHECK(cudaEventRecord(e0, s));
        capture_graph(s);
    }
    CUDA_CHECK(cudaEventRecord(e1, s));

    // Every token consumes at least one input byte, so total_bytes bounds the
    // output; copying that much now saves a second host wait later.
    last_d2h_done_ = false;
    if (speculative_d2h_) {
        const uint32_t bound = std::min(total_bytes, cap_output_tokens);
        if (bound) {
            CUDA_CHECK(cudaMemcpyAsync(h_pin_out_tokens, d_out_tokens,
                                       static_cast<size_t>(bound) * sizeof(uint32_t),
                                       cudaMemcpyDeviceToHost, s));
        }
        last_d2h_done_ = true;
    }

    // The graph writes the token total (scan_out[n_total]) and the error
    // count into the mapped status block, so the sync alone makes them
    // readable here; only batched doc offsets still need a D2H.
    if (out_doc_offsets) {
        CUDA_CHECK(cudaMemcpyAsync(h_pin_doc_token_offsets, d_doc_token_offsets,
                                   (n_docs + 1) * sizeof(uint32_t),
                                   cudaMemcpyDeviceToHost, s));
    }
    wait_stream(s);
    const uint32_t total = static_cast<volatile uint32_t*>(h_pin_status)[0];
    const uint32_t pretok_errors = static_cast<volatile uint32_t*>(h_pin_status)[1];
    if (out_doc_offsets) {
        std::memcpy(out_doc_offsets, h_pin_doc_token_offsets,
                    (n_docs + 1) * sizeof(uint32_t));
    }
    if (pretok_errors > 0) {
        throw std::runtime_error(
            "cuTokenize: pre-token bucket/descriptor capacity exceeded.");
    }
    if (total > cap_output_tokens) {
        throw std::runtime_error("cuTokenize: output token count exceeds "
                                 "the context's output capacity.");
    }
    if (out_total) *out_total = total;
    return d_out_tokens;
}

// Single-document encode is a batch of one; the offsets readback is skipped so
// the staged work is identical to what this path did before batching existed.
const uint32_t* TokenizerCtx::launch_encode_graph(const EncodeInput& in,
                                                  uint32_t* out_total, void* stream)
{
    const uint8_t* doc = in.raw_text;
    const uint32_t len = in.raw_len;
    return launch_graph_core(&doc, &len, /*n_docs=*/1, out_total,
                             /*out_doc_offsets=*/nullptr, stream);
}

const uint32_t* TokenizerCtx::encode_to_device(const EncodeInput& in,
                                               uint32_t* out_count,
                                               float* out_kernel_ms,
                                               float* out_e2e_ms,
                                               void* stream)
{
    // Same pipeline as encode(), minus the final token D2H: the IDs stay in
    // d_out_tokens and the caller gets a borrowed device pointer. Runs on the
    // caller's stream (`stream`) so the result is ordered with an async consumer
    // on that stream. The e2e clock covers H2D + graph + the count read.
    auto t_e2e_start = std::chrono::high_resolution_clock::now();

    uint32_t total = 0;
    const uint32_t* d_tokens = launch_encode_graph(in, &total, stream);

    auto t_e2e_end = std::chrono::high_resolution_clock::now();
    if (out_count) *out_count = total;
    if (out_kernel_ms) {
        float ms = 0.f;
        CUDA_CHECK(cudaEventElapsedTime(&ms,
            static_cast<cudaEvent_t>(ev_start_),
            static_cast<cudaEvent_t>(ev_stop_)));
        *out_kernel_ms = ms;
    }
    if (out_e2e_ms) {
        *out_e2e_ms =
            std::chrono::duration<float, std::milli>(t_e2e_end - t_e2e_start).count();
    }
    return d_tokens;
}

// Same staged H2D + one graph replay + one sync as encode_batch(), but the D2H
// lands directly in the caller's buffer.
//
// encode_batch() copies twice on the host: device -> h_pin_out_tokens ->
// std::vector, and a binding then copies a third time into whatever it hands
// back. Callers that already own the final allocation can skip both by passing
// it here. The pinned staging buffer is bypassed deliberately: a single large
// pageable D2H is not measurably worse than pinned-plus-memcpy, and it removes
// a full pass over the token block.
void TokenizerCtx::encode_batch_to_host(const BatchEncodeInput& in,
                                        uint32_t* out_tokens_host,
                                        uint32_t out_token_capacity,
                                        uint32_t* out_count,
                                        uint32_t* out_doc_offsets_host,
                                        float* out_kernel_ms,
                                        float* out_e2e_ms)
{
    auto t_e2e_start = std::chrono::high_resolution_clock::now();

    uint32_t total = 0;
    const uint32_t* d_tokens = launch_graph_core(in.docs, in.lens, in.n_docs,
                                                 &total, out_doc_offsets_host,
                                                 stream_);
    // Checked after the launch because the exact count is only known once the
    // device reports it; callers size from the byte count, which is an upper
    // bound, so this should never fire.
    if (total > out_token_capacity) {
        throw std::length_error("encode_batch_to_host: output buffer too small");
    }
    cudaStream_t s = static_cast<cudaStream_t>(stream_);
    if (total) {
        CUDA_CHECK(cudaMemcpyAsync(out_tokens_host, d_tokens,
                                   static_cast<size_t>(total) * sizeof(uint32_t),
                                   cudaMemcpyDeviceToHost, s));
        wait_stream(s);
    }
    auto t_e2e_end = std::chrono::high_resolution_clock::now();

    if (out_count) *out_count = total;
    if (out_kernel_ms) {
        float ms = 0.f;
        CUDA_CHECK(cudaEventElapsedTime(&ms,
            static_cast<cudaEvent_t>(ev_start_),
            static_cast<cudaEvent_t>(ev_stop_)));
        *out_kernel_ms = ms;
    }
    if (out_e2e_ms) {
        *out_e2e_ms =
            std::chrono::duration<float, std::milli>(t_e2e_end - t_e2e_start).count();
    }
}

void TokenizerCtx::encode_batch(const BatchEncodeInput& in,
                                std::vector<uint32_t>& out_tokens,
                                std::vector<uint32_t>& out_doc_offsets,
                                float* out_kernel_ms,
                                float* out_e2e_ms)
{
    // One staged H2D of the concatenated batch, ONE graph replay, ONE sync in
    // launch_graph_core, then a single D2H of the token block.
    auto t_e2e_start = std::chrono::high_resolution_clock::now();

    out_doc_offsets.resize(static_cast<size_t>(in.n_docs) + 1);
    uint32_t total = 0;
    const uint32_t* d_tokens = launch_graph_core(in.docs, in.lens, in.n_docs,
                                                 &total, out_doc_offsets.data(),
                                                 stream_);

    cudaStream_t s = static_cast<cudaStream_t>(stream_);
    out_tokens.resize(total);
    if (total) {
        if (!last_d2h_done_) {
            CUDA_CHECK(cudaMemcpyAsync(h_pin_out_tokens, d_tokens,
                                       total * sizeof(uint32_t),
                                       cudaMemcpyDeviceToHost, s));
            wait_stream(s);
        }
        std::memcpy(out_tokens.data(), h_pin_out_tokens, total * sizeof(uint32_t));
    }
    auto t_e2e_end = std::chrono::high_resolution_clock::now();

    if (out_kernel_ms) {
        float ms = 0.f;
        CUDA_CHECK(cudaEventElapsedTime(&ms,
            static_cast<cudaEvent_t>(ev_start_),
            static_cast<cudaEvent_t>(ev_stop_)));
        *out_kernel_ms = ms;
    }
    if (out_e2e_ms) {
        *out_e2e_ms =
            std::chrono::duration<float, std::milli>(t_e2e_end - t_e2e_start).count();
    }
}

const uint32_t* TokenizerCtx::encode_batch_to_device(const BatchEncodeInput& in,
                                                     uint32_t* out_count,
                                                     uint32_t* out_doc_offsets_host,
                                                     float* out_kernel_ms,
                                                     float* out_e2e_ms,
                                                     void* stream)
{
    // Same as encode_batch(), minus the token D2H: the IDs stay in d_out_tokens
    // (borrowed until the next encode on this context) and only the n_docs+1
    // offsets come back to the host.
    auto t_e2e_start = std::chrono::high_resolution_clock::now();

    uint32_t total = 0;
    const uint32_t* d_tokens = launch_graph_core(in.docs, in.lens, in.n_docs,
                                                 &total, out_doc_offsets_host,
                                                 stream);

    auto t_e2e_end = std::chrono::high_resolution_clock::now();
    if (out_count) *out_count = total;
    if (out_kernel_ms) {
        float ms = 0.f;
        CUDA_CHECK(cudaEventElapsedTime(&ms,
            static_cast<cudaEvent_t>(ev_start_),
            static_cast<cudaEvent_t>(ev_stop_)));
        *out_kernel_ms = ms;
    }
    if (out_e2e_ms) {
        *out_e2e_ms =
            std::chrono::duration<float, std::milli>(t_e2e_end - t_e2e_start).count();
    }
    return d_tokens;
}
#endif  // GBPE_GPU_PRETOK

uint32_t TokenizerCtx::encode_into(const EncodeInput& in, uint32_t* out, uint32_t max_out)
{
#if GBPE_GPU_PRETOK
#if GBPE_HAVE_FAMILY_SP
    if (!sp_mode_ || GBPE_GEMMA_GPU_PRETOK)
#endif
    {
        cudaStream_t s = static_cast<cudaStream_t>(stream_);
        uint32_t total = 0;
        const uint32_t* d_tokens = launch_encode_graph(in, &total, stream_);
        if (total > max_out) return UINT32_MAX;
        if (total) {
            // D2H into pinned staging (a pageable destination would make the
            // driver stage and block anyway), then one copy into `out`.
            CUDA_CHECK(cudaMemcpyAsync(h_pin_out_tokens, d_tokens,
                                       total * sizeof(uint32_t),
                                       cudaMemcpyDeviceToHost, s));
            wait_stream(s);
            std::memcpy(out, h_pin_out_tokens, total * sizeof(uint32_t));
        }
        return total;
    }
#endif
    std::vector<uint32_t> toks;
    encode(in, toks, nullptr, nullptr);
    if (toks.size() > max_out) return UINT32_MAX;
    std::memcpy(out, toks.data(), toks.size() * sizeof(uint32_t));
    return static_cast<uint32_t>(toks.size());
}

void TokenizerCtx::encode(const EncodeInput& in,
                          std::vector<uint32_t>& out_tokens,
                          float* out_kernel_ms,
                          float* out_e2e_ms)
{
    if (in.n_short > cap_short || in.n_long > cap_long || in.n_total > cap_pretokens) {
        std::fprintf(stderr,
            "encode: bucket overflow n_short=%u/%u n_long=%u/%u n_total=%u/%u\n",
            in.n_short, cap_short, in.n_long, cap_long, in.n_total, cap_pretokens);
        std::abort();
    }
    cudaStream_t s = static_cast<cudaStream_t>(stream_);

    // Copy into pinned host buffers (the caller's buffers may not be pinned).
#if GBPE_HAVE_FAMILY_SP
    const bool sp = sp_mode_;
#endif

#if GBPE_GPU_PRETOK
    // GPU pre-tokenizer mode: the host supplies only raw text. Bucketing
    // (short/long byte buffers, lens, orig_idx, all counts) is produced on the
    // device inside the captured graph by launch_encode_graph(); the only host
    // work added here is the final D2H of the emitted token IDs.
#if GBPE_HAVE_FAMILY_SP
    if (!sp || GBPE_GEMMA_GPU_PRETOK) {
#else
    {
#endif
        auto t_e2e_start = std::chrono::high_resolution_clock::now();

        uint32_t total = 0;
        const uint32_t* d_tokens = launch_encode_graph(in, &total, stream_);

        // D2H into pinned staging (a pageable destination would make the
        // driver stage and block anyway), then one host memcpy.
        out_tokens.resize(total);
        if (total) {
            if (!last_d2h_done_) {
                CUDA_CHECK(cudaMemcpyAsync(h_pin_out_tokens, d_tokens,
                                           total * sizeof(uint32_t),
                                           cudaMemcpyDeviceToHost, s));
                wait_stream(s);
            }
            std::memcpy(out_tokens.data(), h_pin_out_tokens, total * sizeof(uint32_t));
        }
        auto t_e2e_end = std::chrono::high_resolution_clock::now();

        if (out_kernel_ms) {
            float ms = 0.f;
            CUDA_CHECK(cudaEventElapsedTime(&ms,
                static_cast<cudaEvent_t>(ev_start_),
                static_cast<cudaEvent_t>(ev_stop_)));
            *out_kernel_ms = ms;
        }
        if (out_e2e_ms) {
            *out_e2e_ms =
                std::chrono::duration<float, std::milli>(t_e2e_end - t_e2e_start).count();
        }
        return;
    }
#endif

    // SPPreTokenizer populated short_ids/long_ids and their live counts; SP
    // contexts deliberately continue through the host-bucket staging path.
    if (in.n_short) {
        std::memcpy(h_pin_short_lens, in.short_lens,
                    static_cast<size_t>(in.n_short) * sizeof(uint16_t));
        std::memcpy(h_pin_short_orig_idx, in.short_orig_idx,
                    static_cast<size_t>(in.n_short) * sizeof(uint32_t));
#if GBPE_HAVE_FAMILY_SP
        if (sp) {
            std::memcpy(h_pin_short_ids, in.short_ids,
                static_cast<size_t>(in.n_short) * MAX_PRETOKEN_LEN_SHORT * sizeof(uint32_t));
        } else
#endif
        {
            std::memcpy(h_pin_short_bytes, in.short_bytes,
                        static_cast<size_t>(in.n_short) * MAX_PRETOKEN_LEN_SHORT);
        }
    }
    if (in.n_long) {
        std::memcpy(h_pin_long_lens, in.long_lens,
                    static_cast<size_t>(in.n_long) * sizeof(uint16_t));
        std::memcpy(h_pin_long_orig_idx, in.long_orig_idx,
                    static_cast<size_t>(in.n_long) * sizeof(uint32_t));
#if GBPE_HAVE_FAMILY_SP
        if (sp) {
            std::memcpy(h_pin_long_ids, in.long_ids,
                static_cast<size_t>(in.n_long) * MAX_PRETOKEN_LEN_LONG * sizeof(uint32_t));
        } else
#endif
        {
            std::memcpy(h_pin_long_bytes, in.long_bytes,
                        static_cast<size_t>(in.n_long) * MAX_PRETOKEN_LEN_LONG);
        }
    }

    // Zero the tails of len arrays so unused blocks early-exit (N=0 path).
    if (in.n_short < cap_short) {
        CUDA_CHECK(cudaMemsetAsync(
            d_short_lens + in.n_short, 0,
            static_cast<size_t>(cap_short - in.n_short) * sizeof(uint16_t), s));
    }
    if (cap_long > 0 && in.n_long < cap_long) {
        CUDA_CHECK(cudaMemsetAsync(
            d_long_lens + in.n_long, 0,
            static_cast<size_t>(cap_long - in.n_long) * sizeof(uint16_t), s));
    }
    // Zero d_per_pre_count for slots past n_total — the captured graph runs
    // ALL cap_short + cap_long blocks; out-of-range blocks early-return without
    // writing out_count, so we must pre-zero those output slots.
    if (in.n_total < cap_pretokens + 1) {
        CUDA_CHECK(cudaMemsetAsync(
            d_per_pre_count + in.n_total, 0,
            static_cast<size_t>(cap_pretokens + 1 - in.n_total) * sizeof(uint32_t), s));
    }

    cudaEvent_t e0 = static_cast<cudaEvent_t>(ev_start_);
    cudaEvent_t e1 = static_cast<cudaEvent_t>(ev_stop_);

    auto t_e2e_start = std::chrono::high_resolution_clock::now();

    // H2D both buckets.
    if (in.n_short) {
        CUDA_CHECK(cudaMemcpyAsync(d_short_lens, h_pin_short_lens,
            static_cast<size_t>(in.n_short) * sizeof(uint16_t),
            cudaMemcpyHostToDevice, s));
        CUDA_CHECK(cudaMemcpyAsync(d_short_orig_idx, h_pin_short_orig_idx,
            static_cast<size_t>(in.n_short) * sizeof(uint32_t),
            cudaMemcpyHostToDevice, s));
#if GBPE_HAVE_FAMILY_SP
        if (sp) {
            CUDA_CHECK(cudaMemcpyAsync(d_short_ids, h_pin_short_ids,
                static_cast<size_t>(in.n_short) * MAX_PRETOKEN_LEN_SHORT * sizeof(uint32_t),
                cudaMemcpyHostToDevice, s));
        } else
#endif
        {
            CUDA_CHECK(cudaMemcpyAsync(d_short_bytes, h_pin_short_bytes,
                static_cast<size_t>(in.n_short) * MAX_PRETOKEN_LEN_SHORT,
                cudaMemcpyHostToDevice, s));
        }
    }
    if (in.n_long) {
        CUDA_CHECK(cudaMemcpyAsync(d_long_lens, h_pin_long_lens,
            static_cast<size_t>(in.n_long) * sizeof(uint16_t),
            cudaMemcpyHostToDevice, s));
        CUDA_CHECK(cudaMemcpyAsync(d_long_orig_idx, h_pin_long_orig_idx,
            static_cast<size_t>(in.n_long) * sizeof(uint32_t),
            cudaMemcpyHostToDevice, s));
#if GBPE_HAVE_FAMILY_SP
        if (sp) {
            CUDA_CHECK(cudaMemcpyAsync(d_long_ids, h_pin_long_ids,
                static_cast<size_t>(in.n_long) * MAX_PRETOKEN_LEN_LONG * sizeof(uint32_t),
                cudaMemcpyHostToDevice, s));
        } else
#endif
        {
            CUDA_CHECK(cudaMemcpyAsync(d_long_bytes, h_pin_long_bytes,
                static_cast<size_t>(in.n_long) * MAX_PRETOKEN_LEN_LONG,
                cudaMemcpyHostToDevice, s));
        }
    }

#if GBPE_HAVE_FAMILY_SP
    // SP: splice pre-BPE'd overflow results directly into per_pre_tokens rows
    // and per_pre_count slots at orig_idx. The captured graph reads these
    // alongside the kernel's outputs.
    if (sp && in.n_overflow) {
        // Use synchronous copies — overflow is rare (a handful per million
        // pre-tokens) so the latency hit is negligible. cudaMemcpy waits for
        // the prior async H2Ds to complete on this stream implicitly only if
        // the stream is the default; here our stream is non-blocking, so we
        // synchronize explicitly before to avoid races.
        CUDA_CHECK(cudaStreamSynchronize(s));
        if (in.n_overflow > cap_overflow) {
            std::fprintf(stderr, "encode: n_overflow %u exceeds cap_overflow %u\n",
                         in.n_overflow, cap_overflow);
            std::abort();
        }
        for (uint32_t i = 0; i < in.n_overflow; ++i) {
            uint32_t oidx = in.overflow_orig_idx[i];
            uint32_t rlen = in.overflow_result_lens[i];
            const uint32_t* src = in.overflow_result_ids
                                  + static_cast<size_t>(i) * MAX_PRETOKEN_LEN_LONG;
            // Payload -> overflow buffer at LOCAL index i; flatten_overflow maps
            // i -> oidx via d_overflow_orig_idx for the count/scan/dest lookups.
            CUDA_CHECK(cudaMemcpy(
                d_per_pre_tokens_overflow + static_cast<size_t>(i) * MAX_PRETOKEN_LEN_LONG,
                src,
                static_cast<size_t>(rlen) * sizeof(uint32_t),
                cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(
                d_overflow_orig_idx + i, &oidx, sizeof(uint32_t),
                cudaMemcpyHostToDevice));
            // Count stays keyed by orig_idx (the scan is orig_idx-indexed).
            CUDA_CHECK(cudaMemcpy(
                d_per_pre_count + oidx, &rlen, sizeof(uint32_t),
                cudaMemcpyHostToDevice));
        }
        CUDA_CHECK(cudaMemcpy(d_overflow_n, &in.n_overflow, sizeof(uint32_t),
                              cudaMemcpyHostToDevice));
    } else if (sp && cap_overflow > 0) {
        // No overflow this call: zero the live count so flatten_overflow is a no-op.
        CUDA_CHECK(cudaMemsetAsync(d_overflow_n, 0, sizeof(uint32_t), s));
    }
#endif

    if (use_cuda_graph_) {
        if (!graph_built_) capture_graph();
        CUDA_CHECK(cudaEventRecord(e0, s));
        CUDA_CHECK(cudaGraphLaunch(static_cast<cudaGraphExec_t>(graph_ex_), s));
    } else {
        if (!eager_warmed_) capture_graph(s);
        CUDA_CHECK(cudaEventRecord(e0, s));
        capture_graph(s);
    }
    CUDA_CHECK(cudaEventRecord(e1, s));

    uint32_t total = 0;
    CUDA_CHECK(cudaMemcpyAsync(&total,
                               d_scan_out + cap_pretokens,
                               sizeof(uint32_t),
                               cudaMemcpyDeviceToHost, s));
    CUDA_CHECK(cudaStreamSynchronize(s));

    if (total > cap_output_tokens) {
        std::fprintf(stderr, "encode: total %u exceeds out cap %u\n",
                     total, cap_output_tokens);
        std::abort();
    }

    out_tokens.resize(total);
    CUDA_CHECK(cudaMemcpyAsync(out_tokens.data(), d_out_tokens,
                               total * sizeof(uint32_t),
                               cudaMemcpyDeviceToHost, s));
    CUDA_CHECK(cudaStreamSynchronize(s));

    auto t_e2e_end = std::chrono::high_resolution_clock::now();

    if (out_kernel_ms) {
        float ms = 0.f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, e0, e1));
        *out_kernel_ms = ms;
    }
    if (out_e2e_ms) {
        *out_e2e_ms = std::chrono::duration<float, std::milli>(t_e2e_end - t_e2e_start).count();
    }
}

// =============================================================================
//  Decode (F5): token IDs -> raw bytes. Separate captured CUDA graph.
//
//  (1) decode_lens_kernel  — per-token lookup of vocab_token_len[id]
//  (2) cub::DeviceScan::ExclusiveSum on lens -> per-token write offsets
//  (3) decode_gather_kernel — one block per token, threads cooperatively copy
//      vocab_token_bytes_concat[off .. off+len) to out_bytes[woff..)
// =============================================================================

__global__ void decode_lens_kernel(
    const uint32_t* __restrict__ token_ids,
    const uint32_t* __restrict__ n_tokens_ptr,
    uint32_t                     cap_tokens,
    const uint16_t* __restrict__ vocab_token_len,
    uint32_t* __restrict__ out_lens)
{
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= cap_tokens) return;
    uint32_t n = *n_tokens_ptr;
    if (i >= n) {
        out_lens[i] = 0;
        if (i == cap_tokens - 1) out_lens[cap_tokens] = 0;
        return;
    }
    uint32_t id = token_ids[i];
    out_lens[i] = static_cast<uint32_t>(vocab_token_len[id]);
    if (i == cap_tokens - 1) out_lens[cap_tokens] = 0;
}

__global__ void decode_gather_kernel(
    const uint32_t* __restrict__ token_ids,
    const uint32_t* __restrict__ n_tokens_ptr,
    const uint8_t*  __restrict__ vocab_bytes,
    const uint32_t* __restrict__ vocab_offset,
    const uint16_t* __restrict__ vocab_len,
    const uint32_t* __restrict__ write_offsets,
    uint8_t* __restrict__ out_bytes)
{
    uint32_t i = blockIdx.x;
    uint32_t n = *n_tokens_ptr;
    if (i >= n) return;
    uint32_t id = token_ids[i];
    uint32_t L = vocab_len[id];
    if (L == 0) return;
    uint32_t src = vocab_offset[id];
    uint32_t dst = write_offsets[i];
    const uint8_t* s = vocab_bytes + src;
    uint8_t* d = out_bytes + dst;
    for (uint32_t k = threadIdx.x; k < L; k += blockDim.x) {
        d[k] = s[k];
    }
}

#if GBPE_GPU_PRETOK
// ---- Benchmark helpers: isolate the captured graph at the Python level. ----
void TokenizerCtx::stage_input(const uint8_t* raw_text, uint32_t raw_len) {
    const uint8_t* docs[] = {raw_text};
    const uint32_t lens[] = {raw_len};
    const BatchEncodeInput in{docs, lens, 1};
    stage_batch(in);
}

void TokenizerCtx::stage_batch(const BatchEncodeInput& in) {
    if (in.n_docs < 1) {
        throw std::runtime_error("stage_batch: n_docs must be >= 1");
    }
    if (in.n_docs > cap_docs) {
        throw std::runtime_error("stage_batch: n_docs exceeds cap_docs");
    }

    cudaStream_t s = static_cast<cudaStream_t>(stream_);
    uint32_t total_bytes = 0;
    for (uint32_t d = 0; d < in.n_docs; ++d) {
        h_pin_doc_byte_offsets[d] = total_bytes;
        const uint32_t len = in.lens[d];
        if (len > cap_input_bytes - total_bytes) {
            throw std::runtime_error("stage_batch: input exceeds cap_input_bytes");
        }
        if (len) std::memcpy(h_pin_text_bytes + total_bytes, in.docs[d], len);
        total_bytes += len;
    }
    h_pin_doc_byte_offsets[in.n_docs] = total_bytes;
    *h_pin_text_len = total_bytes;
    *h_pin_n_docs = in.n_docs;
    staged_batch_n_docs_ = in.n_docs;
    staged_batch_pending_ = true;
    staged_batch_replayed_ = false;

    CUDA_CHECK(cudaMemcpyAsync(d_text_bytes, h_pin_text_bytes, total_bytes,
                               cudaMemcpyHostToDevice, s));
    CUDA_CHECK(cudaMemcpyAsync(d_hdr, h_pin_hdr, (2 + in.n_docs + 1) * sizeof(uint32_t),
                               cudaMemcpyHostToDevice, s));
    if (use_cuda_graph_) {
        if (!graph_built_) capture_graph();
    } else if (!eager_warmed_) {
        capture_graph(s);
    }
    CUDA_CHECK(cudaStreamSynchronize(s));   // all setup (H2D + capture) done here
}

float TokenizerCtx::launch_graph_sync() {
    cudaStream_t s = static_cast<cudaStream_t>(stream_);
    cudaEvent_t e0 = static_cast<cudaEvent_t>(ev_start_);
    cudaEvent_t e1 = static_cast<cudaEvent_t>(ev_stop_);
    // Bracket the replay with events (recorded outside any capture region) so
    // the returned time is the on-device graph duration, not the host launch
    // path. No D2H — token IDs stay on the device (d_out_tokens).
    CUDA_CHECK(cudaEventRecord(e0, s));
    if (use_cuda_graph_) {
        CUDA_CHECK(cudaGraphLaunch(static_cast<cudaGraphExec_t>(graph_ex_), s));
    } else {
        capture_graph(s);
    }
    CUDA_CHECK(cudaEventRecord(e1, s));
    CUDA_CHECK(cudaStreamSynchronize(s));   // block until the graph completes
    if (staged_batch_pending_) staged_batch_replayed_ = true;
    float ms = 0.f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, e0, e1));
    return ms;
}

void TokenizerCtx::read_staged_batch(std::vector<uint32_t>& out_tokens,
                                     std::vector<uint32_t>& out_doc_offsets) {
    if (!staged_batch_pending_ || staged_batch_n_docs_ == 0) {
        throw std::logic_error(
            "read_staged_batch: call stage_batch() before readback");
    }
    if (!staged_batch_replayed_) {
        throw std::logic_error(
            "read_staged_batch: call launch_graph_sync() before readback");
    }

    cudaStream_t s = static_cast<cudaStream_t>(stream_);
    uint32_t total = 0;
    uint32_t pretok_errors = 0;
    CUDA_CHECK(cudaMemcpyAsync(&total, d_out_total, sizeof(uint32_t),
                               cudaMemcpyDeviceToHost, s));
    CUDA_CHECK(cudaMemcpyAsync(&pretok_errors, d_pretok_errors, sizeof(uint32_t),
                               cudaMemcpyDeviceToHost, s));
    CUDA_CHECK(cudaMemcpyAsync(
        h_pin_doc_token_offsets, d_doc_token_offsets,
        static_cast<size_t>(staged_batch_n_docs_ + 1) * sizeof(uint32_t),
        cudaMemcpyDeviceToHost, s));
    CUDA_CHECK(cudaStreamSynchronize(s));

    if (pretok_errors > 0) {
        throw std::runtime_error(
            "cuTokenize: pre-token bucket/descriptor capacity exceeded.");
    }
    if (total > cap_output_tokens) {
        throw std::runtime_error(
            "cuTokenize: output token count exceeds the context's output capacity.");
    }

    out_doc_offsets.assign(
        h_pin_doc_token_offsets,
        h_pin_doc_token_offsets + static_cast<size_t>(staged_batch_n_docs_ + 1));
    if (out_doc_offsets.front() != 0 || out_doc_offsets.back() != total) {
        throw std::runtime_error(
            "read_staged_batch: invalid device-produced document offsets");
    }

    out_tokens.resize(total);
    if (total) {
        CUDA_CHECK(cudaMemcpyAsync(out_tokens.data(), d_out_tokens,
                                   static_cast<size_t>(total) * sizeof(uint32_t),
                                   cudaMemcpyDeviceToHost, s));
        CUDA_CHECK(cudaStreamSynchronize(s));
    }
}

void TokenizerCtx::stage_decode(const uint32_t* token_ids_host, uint32_t n_tokens) {
    if (cap_decode_tokens == 0) throw std::runtime_error("stage_decode: no decode capacity");
    if (n_tokens > cap_decode_tokens) throw std::runtime_error("stage_decode: n_tokens exceeds cap");
    cudaStream_t s = static_cast<cudaStream_t>(stream_);
    if (n_tokens) std::memcpy(h_pin_dec_token_ids, token_ids_host,
                              static_cast<size_t>(n_tokens) * sizeof(uint32_t));
    if (n_tokens) CUDA_CHECK(cudaMemcpyAsync(d_dec_token_ids, h_pin_dec_token_ids,
        static_cast<size_t>(n_tokens) * sizeof(uint32_t), cudaMemcpyHostToDevice, s));
    CUDA_CHECK(cudaMemcpyAsync(d_dec_n_tokens, &n_tokens, sizeof(uint32_t),
        cudaMemcpyHostToDevice, s));
    if (!dec_graph_built_) capture_decode_graph();
    CUDA_CHECK(cudaStreamSynchronize(s));   // setup (H2D + capture) done here
}

float TokenizerCtx::launch_decode_sync() {
    cudaStream_t s = static_cast<cudaStream_t>(stream_);
    cudaEvent_t e0 = static_cast<cudaEvent_t>(dec_ev_start_);
    cudaEvent_t e1 = static_cast<cudaEvent_t>(dec_ev_stop_);
    CUDA_CHECK(cudaEventRecord(e0, s));
    CUDA_CHECK(cudaGraphLaunch(static_cast<cudaGraphExec_t>(dec_graph_ex_), s));
    CUDA_CHECK(cudaEventRecord(e1, s));
    CUDA_CHECK(cudaStreamSynchronize(s));   // block until decode completes; no D2H
    float ms = 0.f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, e0, e1));
    return ms;
}
#endif

void TokenizerCtx::allocate_decode() {
    CUDA_CHECK(cudaMalloc(&d_dec_token_ids,
                          static_cast<size_t>(cap_decode_tokens) * sizeof(uint32_t)));
    CUDA_CHECK(cudaMemset(d_dec_token_ids, 0,
                          static_cast<size_t>(cap_decode_tokens) * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_dec_n_tokens, sizeof(uint32_t)));
    CUDA_CHECK(cudaMemset(d_dec_n_tokens, 0, sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_dec_lens,
                          static_cast<size_t>(cap_decode_tokens + 1) * sizeof(uint32_t)));
    CUDA_CHECK(cudaMemset(d_dec_lens, 0,
                          static_cast<size_t>(cap_decode_tokens + 1) * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_dec_offsets,
                          static_cast<size_t>(cap_decode_tokens + 1) * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_dec_out_bytes,
                          static_cast<size_t>(cap_decode_bytes)));
    CUDA_CHECK(cudaMallocHost(&h_pin_dec_token_ids,
                              static_cast<size_t>(cap_decode_tokens) * sizeof(uint32_t)));
    {
        size_t bytes = 0;
        cub::DeviceScan::ExclusiveSum(nullptr, bytes,
            d_dec_lens, d_dec_offsets,
            cap_decode_tokens + 1, /*stream=*/0);
        dec_scan_temp_bytes = bytes;
        CUDA_CHECK(cudaMalloc(&d_dec_scan_temp, dec_scan_temp_bytes));
    }
    cudaEvent_t e0, e1;
    CUDA_CHECK(cudaEventCreate(&e0));
    CUDA_CHECK(cudaEventCreate(&e1));
    dec_ev_start_ = e0;
    dec_ev_stop_  = e1;

    GBPE_LOG(
        "[ctx-dec] cap_tokens=%u cap_bytes=%u scan_temp=%zu B\n",
        cap_decode_tokens, cap_decode_bytes, dec_scan_temp_bytes);
}

void TokenizerCtx::capture_decode_graph() {
    cudaStream_t s = static_cast<cudaStream_t>(stream_);

    const uint32_t lens_block = 256;
    const uint32_t lens_grid  = (cap_decode_tokens + lens_block - 1) / lens_block;

    // Warmup outside capture (module load, JIT).
    decode_lens_kernel<<<lens_grid, lens_block, 0, s>>>(
        d_dec_token_ids, d_dec_n_tokens, cap_decode_tokens,
        vocab.vocab_token_len, d_dec_lens);
    cub::DeviceScan::ExclusiveSum(d_dec_scan_temp, dec_scan_temp_bytes,
        d_dec_lens, d_dec_offsets, cap_decode_tokens + 1, s);
    decode_gather_kernel<<<cap_decode_tokens, 32, 0, s>>>(
        d_dec_token_ids, d_dec_n_tokens,
        vocab.vocab_token_bytes_concat, vocab.vocab_token_offset,
        vocab.vocab_token_len, d_dec_offsets, d_dec_out_bytes);
    CUDA_CHECK(cudaStreamSynchronize(s));

    CUDA_CHECK(cudaStreamBeginCapture(s, cudaStreamCaptureModeThreadLocal));

    decode_lens_kernel<<<lens_grid, lens_block, 0, s>>>(
        d_dec_token_ids, d_dec_n_tokens, cap_decode_tokens,
        vocab.vocab_token_len, d_dec_lens);
    cub::DeviceScan::ExclusiveSum(d_dec_scan_temp, dec_scan_temp_bytes,
        d_dec_lens, d_dec_offsets, cap_decode_tokens + 1, s);
    decode_gather_kernel<<<cap_decode_tokens, 32, 0, s>>>(
        d_dec_token_ids, d_dec_n_tokens,
        vocab.vocab_token_bytes_concat, vocab.vocab_token_offset,
        vocab.vocab_token_len, d_dec_offsets, d_dec_out_bytes);

    cudaGraph_t g;
    CUDA_CHECK(cudaStreamEndCapture(s, &g));
    dec_graph_ = g;
    cudaGraphExec_t gex;
    CUDA_CHECK(cudaGraphInstantiate(&gex, g, nullptr, nullptr, 0));
    dec_graph_ex_ = gex;
    dec_graph_built_ = true;
}

void TokenizerCtx::decode(const uint32_t* token_ids_host,
                          uint32_t n_tokens,
                          std::vector<uint8_t>& out_bytes,
                          float* out_kernel_ms,
                          float* out_e2e_ms)
{
    if (cap_decode_tokens == 0 || cap_decode_bytes == 0) {
        std::fprintf(stderr,
            "decode: TokenizerCtx was constructed without decode capacity. "
            "Pass max_decode_tokens and max_decode_bytes to the constructor.\n");
        std::abort();
    }
    if (n_tokens > cap_decode_tokens) {
        std::fprintf(stderr,
            "decode: n_tokens=%u exceeds cap_decode_tokens=%u\n",
            n_tokens, cap_decode_tokens);
        std::abort();
    }
    cudaStream_t s = static_cast<cudaStream_t>(stream_);

    if (n_tokens) {
        std::memcpy(h_pin_dec_token_ids, token_ids_host,
                    static_cast<size_t>(n_tokens) * sizeof(uint32_t));
    }

    auto t_e2e_start = std::chrono::high_resolution_clock::now();

    if (n_tokens > 0) {
        CUDA_CHECK(cudaMemcpyAsync(d_dec_token_ids, h_pin_dec_token_ids,
            static_cast<size_t>(n_tokens) * sizeof(uint32_t),
            cudaMemcpyHostToDevice, s));
    }
    CUDA_CHECK(cudaMemcpyAsync(d_dec_n_tokens, &n_tokens, sizeof(uint32_t),
        cudaMemcpyHostToDevice, s));

    if (!dec_graph_built_) {
        capture_decode_graph();
    }

    cudaEvent_t e0 = static_cast<cudaEvent_t>(dec_ev_start_);
    cudaEvent_t e1 = static_cast<cudaEvent_t>(dec_ev_stop_);
    CUDA_CHECK(cudaEventRecord(e0, s));
    CUDA_CHECK(cudaGraphLaunch(static_cast<cudaGraphExec_t>(dec_graph_ex_), s));
    CUDA_CHECK(cudaEventRecord(e1, s));

    uint32_t total = 0;
    CUDA_CHECK(cudaMemcpyAsync(&total,
                               d_dec_offsets + n_tokens,
                               sizeof(uint32_t),
                               cudaMemcpyDeviceToHost, s));
    CUDA_CHECK(cudaStreamSynchronize(s));

    if (total > cap_decode_bytes) {
        std::fprintf(stderr, "decode: total %u exceeds cap_decode_bytes %u\n",
                     total, cap_decode_bytes);
        std::abort();
    }

    out_bytes.resize(total);
    if (total > 0) {
        CUDA_CHECK(cudaMemcpyAsync(out_bytes.data(), d_dec_out_bytes,
                                   total, cudaMemcpyDeviceToHost, s));
        CUDA_CHECK(cudaStreamSynchronize(s));
    }

    auto t_e2e_end = std::chrono::high_resolution_clock::now();
    if (out_kernel_ms) {
        float ms = 0.f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, e0, e1));
        *out_kernel_ms = ms;
    }
    if (out_e2e_ms) {
        *out_e2e_ms = std::chrono::duration<float, std::milli>(t_e2e_end - t_e2e_start).count();
    }
}

}  // namespace gbpe
