// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 the GraphTok contributors
//
// Single-block prefix scan for small captured graphs.
//
// `cub::DeviceScan` costs two graph nodes per scan: a tile-status init kernel
// plus the scan itself. At 4 KiB the six pre-tokenizer scans measured 24.4 us
// of a 74 us replay, 8.2 us of it in init kernels that do no useful work.
//
// When a context's capacity is small enough that one block can cover the whole
// input, a single block scans it tile by tile in shared memory: one node, no
// init kernel, and -- unlike a decoupled-lookback scan -- no __threadfence, no
// atomics and no inter-block spin. (A lookback version was tried and was
// slower at every size; see the commit history.) Above that capacity CUB still
// wins, so the choice is made once at capture time from the capacity, which
// keeps the graph static.
#pragma once
#include <cstdint>
#include <cuda_runtime.h>

namespace gbpe {
namespace smallscan {

constexpr int kThreads = 512;
constexpr int kItems   = 8;
constexpr int kTile    = kThreads * kItems;   // 4096 elements per pass
constexpr int kWarps   = kThreads / 32;

// Largest n this kernel is used for. One block streams the input in kTile
// passes, so past this the serial pass count outweighs CUB's parallelism.
constexpr uint32_t kMaxElems = 64u * 1024u;

struct SumOp {
    __device__ __forceinline__ uint32_t operator()(uint32_t a, uint32_t b) const { return a + b; }
    static constexpr uint32_t identity = 0u;
};
struct MaxOp {
    __device__ __forceinline__ uint32_t operator()(uint32_t a, uint32_t b) const { return a > b ? a : b; }
    static constexpr uint32_t identity = 0u;
};

// One block, one graph node. `s_carry` holds the running prefix of everything
// before the current tile; each tile adds its own total to it at the end.
template<typename Op, bool kInclusive>
__global__ void __launch_bounds__(kThreads) scan_kernel(
    const uint32_t* __restrict__ in, uint32_t* __restrict__ out, uint32_t n)
{
    __shared__ uint32_t s_warp_excl[kWarps];   // exclusive prefix of each warp
    __shared__ uint32_t s_tile_total;
    __shared__ uint32_t s_carry;
    Op op;
    const uint32_t lane = threadIdx.x & 31u, warp = threadIdx.x >> 5;
    if (threadIdx.x == 0) s_carry = Op::identity;
    __syncthreads();

    for (uint32_t base = 0; base < n; base += kTile) {
        // Per-thread serial scan over kItems consecutive elements.
        uint32_t incl[kItems];
        uint32_t acc = Op::identity;
#pragma unroll
        for (int k = 0; k < kItems; ++k) {
            const uint32_t i = base + threadIdx.x * kItems + k;
            acc = op(acc, (i < n) ? in[i] : Op::identity);
            incl[k] = acc;
        }
        // Warp-inclusive scan of the per-thread aggregates. Shuffles are
        // unconditional: a sync shuffle inside a lane-divergent branch never
        // converges and hangs the warp.
        uint32_t x = acc;
#pragma unroll
        for (int off = 1; off < 32; off <<= 1) {
            const uint32_t y = __shfl_up_sync(0xFFFFFFFFu, x, off);
            if (lane >= static_cast<uint32_t>(off)) x = op(y, x);
        }
        const uint32_t x_up = __shfl_up_sync(0xFFFFFFFFu, x, 1);
        const uint32_t thread_excl_in_warp = lane ? x_up : Op::identity;
        if (lane == 31) s_warp_excl[warp] = x;     // warp total, for now
        __syncthreads();

        // Thread 0 turns the warp totals into warp exclusive prefixes and
        // records the tile total. Only kWarps (16) elements, so serial is fine.
        if (threadIdx.x == 0) {
            uint32_t run = Op::identity;
#pragma unroll
            for (int w = 0; w < kWarps; ++w) {
                const uint32_t total = s_warp_excl[w];
                s_warp_excl[w] = run;
                run = op(run, total);
            }
            s_tile_total = run;
        }
        __syncthreads();

        const uint32_t tile_base = op(s_carry, s_warp_excl[warp]);
        const uint32_t excl = op(tile_base, thread_excl_in_warp);
#pragma unroll
        for (int k = 0; k < kItems; ++k) {
            const uint32_t i = base + threadIdx.x * kItems + k;
            if (i < n) {
                // Inclusive: prefix through element i. Exclusive: through i-1,
                // which for k == 0 is just the thread's own exclusive base.
                out[i] = kInclusive ? op(excl, incl[k])
                                    : (k == 0 ? excl : op(excl, incl[k - 1]));
            }
        }
        __syncthreads();   // all reads of s_carry done before thread 0 bumps it
        if (threadIdx.x == 0) s_carry = op(s_carry, s_tile_total);
        __syncthreads();
    }
}

}  // namespace smallscan

// True when a context of this capacity should use the single-block scans.
inline bool use_small_scan(uint32_t capacity_elems) {
    return capacity_elems <= smallscan::kMaxElems;
}

// Exclusive sum of n uint32 values in one graph node.
inline void small_exclusive_sum(const uint32_t* in, uint32_t* out, uint32_t n,
                                cudaStream_t s) {
    smallscan::scan_kernel<smallscan::SumOp, false>
        <<<1, smallscan::kThreads, 0, s>>>(in, out, n);
}

// Inclusive max-scan of n uint32 values in one graph node.
inline void small_inclusive_max(const uint32_t* in, uint32_t* out, uint32_t n,
                                cudaStream_t s) {
    smallscan::scan_kernel<smallscan::MaxOp, true>
        <<<1, smallscan::kThreads, 0, s>>>(in, out, n);
}

}  // namespace gbpe
