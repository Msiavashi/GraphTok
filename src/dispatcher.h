// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 the GraphTok contributors
//
// Load-adaptive dispatch for a serving frontend that tokenizes many
// independent requests from many threads at once.
//
// A graph replay has a fixed cost (~70-120 us) that barely depends on how many
// documents it carries, and with a blocking host wait it costs the CPU only
// the staging and launch around it. The CPU engine (gigatoken) is faster for
// one small request but spends a core for the whole encode. The dispatcher
// uses each where it is cheapest:
//
//   * a request the CPU route accepts (CpuRoute::takes: small, content-aware
//     limits) runs on the caller's thread on a CPU engine IF one of the
//     `cpu_workers` engines is free -- so an idle system gets the CPU engine's
//     latency, and tokenization never occupies more than `cpu_workers` cores;
//   * everything else joins the GPU queue. Queued requests are combined into
//     one graph replay, self-clocked: whichever caller finds no batch in
//     flight becomes the leader and launches everything queued, and requests
//     that arrive while that batch runs form the next one. There is no timer
//     and no background thread; a lone request is a batch of one.
//
// Token IDs are identical on every route (each route is gated bit-exact).
#pragma once
#include "cpu_route.h"
#include "tokenizer.cuh"
#include "vocab.h"

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <exception>
#include <memory>
#include <mutex>
#include <string>
#include <string_view>
#include <vector>

namespace gbpe {

enum class DispatchPolicy : uint8_t {
    Adaptive = 0,   // CPU engine when one is free and the input qualifies, else GPU batch
    GpuOnly  = 1,   // every request through the GPU batcher (coalescing only)
    CpuOnly  = 2,   // every request that qualifies waits for a CPU engine
};

struct DispatchStats {
    uint64_t cpu_requests = 0;     // encoded on a CPU engine
    uint64_t gpu_requests = 0;     // encoded in a GPU batch
    uint64_t gpu_batches  = 0;     // graph replays launched by the batcher
    uint64_t gpu_large    = 0;     // requests above the batch cap (own replay)
    uint64_t gpu_batch_ns = 0;     // wall time inside batch replays (staging..results)
};

class Dispatcher {
public:
    Dispatcher(const std::string& vocab_path, uint32_t cpu_workers,
               uint32_t max_batch_bytes, uint32_t max_batch_docs,
               DispatchPolicy policy, const std::string& cpu_backend);
    ~Dispatcher();
    Dispatcher(const Dispatcher&) = delete;
    Dispatcher& operator=(const Dispatcher&) = delete;

    // Thread-safe; blocks until the request's tokens are in `out` (replaced).
    void encode(std::string_view raw, std::vector<uint32_t>& out);

    DispatchStats stats() const;
    const char* cpu_backend() const;
    uint32_t cpu_workers() const { return static_cast<uint32_t>(cpu_.size()); }
    uint32_t vocab_size() const { return vp_.vocab_size; }

private:
    struct Req {
        std::string_view text;          // normalized bytes
        std::vector<uint32_t>* out;
        bool done = false;              // results (or err) delivered
        bool lead = false;              // handed the batcher's lead
        std::exception_ptr err;
        std::condition_variable cv;     // wakes exactly this request's thread
    };
    struct Ctx {
        uint32_t cap_input = 0;
        std::unique_ptr<TokenizerCtx> ctx;
    };

    int try_acquire_cpu();
    int acquire_cpu_blocking();
    void release_cpu(int i);
    void gpu_encode(std::string_view norm, std::vector<uint32_t>& out);
    void run_batch(const std::vector<Req*>& batch);
    void run_large(std::string_view norm, std::vector<uint32_t>& out);
    TokenizerCtx* batch_ctx(uint32_t total_bytes);

    HostVocab hv_;
    VocabPack vp_{};
    DispatchPolicy policy_;
    uint32_t max_batch_bytes_;
    uint32_t max_batch_docs_;

    // CPU engines, one per worker slot; busy_[i] guards cpu_[i].
    std::vector<std::unique_ptr<CpuRoute>> cpu_;
    std::unique_ptr<std::atomic<bool>[]> busy_;
    std::mutex cpu_wait_mu_;
    std::condition_variable cpu_wait_cv_;

    // GPU batcher.
    std::mutex q_mu_;
    std::deque<Req*> queue_;
    bool leader_active_ = false;
    std::vector<Ctx> ladder_;          // batch contexts by byte class (leader only)
    std::mutex large_mu_;
    std::vector<Ctx> large_;           // single-document contexts above the cap

    std::atomic<uint64_t> n_cpu_{0}, n_gpu_{0}, n_batches_{0}, n_large_{0}, batch_ns_{0};
};

}  // namespace gbpe
