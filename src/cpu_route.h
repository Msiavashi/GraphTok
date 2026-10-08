// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 the cuTokenize contributors
//
// CPU route for inputs too small to amortize a graph replay (~70-120 us of
// fixed GPU latency). Shared by the Python bindings and the C ABI so both pick
// the same engine and the same threshold.
//
// Two engines can serve it:
//   * gigatoken (rs-gigatoken, MIT, linked statically through rust/gtok_cpu
//     when the build finds a nightly Rust toolchain): measured 1.0-2.4x
//     faster than our HostEncoder on 128-2k-token inputs, and it covers
//     DeepSeek-V3 as well;
//   * HostEncoder (host_encode.h): our own encoder, used when gigatoken is not
//     built, fails to load this tokenizer.json, or disagrees with HostEncoder
//     on the construction-time canary.
//
// Selection: the constructor's `requested` string, else GTOK_CPU_BACKEND,
// else "auto" (gigatoken, then HostEncoder, then off).
//
// Threshold. An input of raw byte length n (before normalization) takes the
// CPU route when n <= max_bytes_multilingual(); never when n > 8 *
// max_bytes(); in between it depends on the text:
//   * static rule: CPU if the text is not multilingual (at most 1/64 of its
//     bytes start a non-ASCII character, typographic punctuation such as
//     curly quotes not counted) and n <= max_bytes();
//   * measured rule (default-limit routes only): once the GPU has been timed
//     on this content class and size class, the route whose recent cost is
//     lower (GPU: EWMA; CPU: the second smallest of its last 4 per-byte samples,
//     explored on 1 call in 4 until it has them). The static limits are the crossovers on
//     never-repeated text (cpu_thresholds.h);
//     the CPU engine's cost moves with how often it has seen the words, so
//     the real crossover moves with the traffic, and this follows it.
// Callers feed back GPU-route timings with observe_gpu() -- never one that
// included a graph capture; the CPU route times itself. An explicit max_bytes (constructor, setter, or
// GTOK_CPU_MAX_BYTES) sets both limits and turns the measured rule off, as
// does GTOK_CPU_ADAPT=0.
#pragma once
#include "host_encode.h"
#include "vocab.h"

#include <cstddef>
#include <cstdint>
#include <atomic>
#include <memory>
#include <mutex>
#include <string>
#include <string_view>
#include <vector>

namespace gbpe {

enum class CpuBackend : uint8_t { Off = 0, Host = 1, Gigatoken = 2 };

// Pass as max_bytes to use the measured per-family defaults.
constexpr uint32_t kCpuMaxBytesAuto = 0xFFFFFFFFu;

const char* cpu_backend_name(CpuBackend b);

// True when this binary links the gigatoken engine.
bool cpu_gigatoken_linked();

class CpuRoute {
public:
    CpuRoute(const HostVocab& hv, const std::string& vocab_path,
             const std::string& requested, uint32_t max_bytes);
    ~CpuRoute();
    CpuRoute(const CpuRoute&) = delete;
    CpuRoute& operator=(const CpuRoute&) = delete;

    CpuBackend backend() const { return backend_; }
    const char* backend_name() const { return cpu_backend_name(backend_); }
    // Why the backend was chosen (or another was rejected); for diagnostics.
    const std::string& note() const { return note_; }

    uint32_t max_bytes() const { return max_ascii_; }
    uint32_t max_bytes_multilingual() const { return max_ml_; }
    void set_max_bytes(uint32_t v) {
        set_limits(v, v);
        adaptive_ = false;
    }
    void set_limits(uint32_t ascii, uint32_t multilingual) {
        const bool off = backend_ == CpuBackend::Off;
        max_ascii_ = off ? 0u : ascii;
        max_ml_    = off ? 0u : (multilingual < ascii ? multilingual : ascii);
    }
    bool adaptive() const { return adaptive_; }

    // Whether `n` raw bytes at `p` should take the CPU route.
    bool takes(const char* p, size_t n) const {
        if (n <= max_ml_) return max_ascii_ != 0u;
        if (n > band_top()) return false;
        const int cls = multilingual(p, n) ? 1 : 0;
        if (adaptive_) {
            const uint64_t gpu_ns = gpu_ns_[cls][size_class(n)].load(std::memory_order_relaxed);
            if (gpu_ns) {
                // Until the CPU has been timed kCpuWindow times on this
                // content, try it on 1 call in 4 so the rule can learn.
                if (cpu_samples_[cls].load(std::memory_order_relaxed) < kCpuWindow)
                    return explore_.fetch_add(1, std::memory_order_relaxed) % 4 == 0;
                const uint64_t cpu_ns = cpu_ps_estimate(cls) * n / 1000;
                if (cpu_ns < gpu_ns) return true;
                // A close call: the CPU estimate may be stale or extrapolated
                // from smaller inputs, so re-measure it on 1 call in 16.
                return cpu_ns < 2 * gpu_ns &&
                       explore_.fetch_add(1, std::memory_order_relaxed) % 16 == 0;
            }
        }
        return cls == 0 && n <= max_ascii_;
    }
    // Feedback from a GPU-route encode of `n` raw bytes at `p` taking `ns`.
    void observe_gpu(const char* p, size_t n, uint64_t ns);
    // The measured rule's state, for diagnostics: CPU ps/byte per content
    // class, then GPU ns per (class, size class); 0 = not yet observed.
    std::vector<uint64_t> cost_model() const {
        std::vector<uint64_t> v = {cpu_ps_estimate(0), cpu_ps_estimate(1)};
        for (int c = 0; c < 2; ++c)
            for (int k = 0; k < kSizeClasses; ++k) v.push_back(gpu_ns_[c][k].load());
        return v;
    }
    // Several documents routed together (a batch): judged on their total.
    bool takes_all(const char* const* ps, const size_t* ns, size_t count) const;

    // Bare encode (no BOS/EOS) of raw, un-normalized UTF-8. `out` is
    // replaced. Throws std::runtime_error if the engine fails; callers then
    // take the GPU route. Serialized internally: one engine per route.
    void encode(std::string_view raw, std::vector<uint32_t>& out);

private:
    void encode_locked(std::string_view raw, std::vector<uint32_t>& out);

    // More than 1/64 of the bytes start a non-ASCII character other than
    // General Punctuation (U+2000-U+207F: curly quotes, dashes, ellipses).
    static bool multilingual(const char* p, size_t n);
    // multilingual() for bytes just classified on this thread, without
    // rescanning them.
    static bool multilingual_again(const char* p, size_t n);
    // The measured rule's band: (max_bytes_multilingual, 8 * max_bytes].
    size_t band_top() const { return 8 * static_cast<size_t>(max_ascii_); }
    static constexpr int kSizeClasses = 8;   // 4 KiB .. 512 KiB, powers of two
    static int size_class(size_t n) {
        int c = 0;
        for (size_t v = n >> 13; v && c < kSizeClasses - 1; v >>= 1) ++c;
        return c;
    }
    // EWMA with weight 1/4; a sample over 4x the estimate (a cold first
    // sight, a preempted thread) moves it as if it were 4x.
    static void ewma(std::atomic<uint64_t>& a, uint64_t x) {
        const uint64_t old = a.load(std::memory_order_relaxed);
        if (old && x > 4 * old) x = 4 * old;
        a.store(old ? old - old / 4 + x / 4 : x, std::memory_order_relaxed);
    }

    const HostVocab& hv_;
    CpuBackend backend_ = CpuBackend::Off;
    uint32_t max_ascii_ = 0;
    uint32_t max_ml_ = 0;
    bool adaptive_ = true;
    // GPU cost per (class, size class): EWMA (1/4), 0 = not yet observed.
    // CPU cost per byte: the last kCpuWindow samples per class; the estimate
    // is their second smallest. A CPU engine's first sight of text is slow
    // (its pre-token cache is cold), and on traffic that repeats text the
    // other samples show the warm cost; on never-repeated text most samples
    // are first sights and the estimate stays at that cost.
    static constexpr int kCpuWindow = 4;
    std::atomic<uint64_t> cpu_ring_[2][kCpuWindow] = {};
    std::atomic<uint32_t> cpu_samples_[2] = {};
    uint64_t cpu_ps_estimate(int cls) const {
        // Second smallest of the filled samples (the only one if just one):
        // one cold first sight or descheduled call cannot inflate it, and one
        // lucky re-encode cannot deflate it on traffic that is mostly new.
        uint64_t lo = 0, second = 0;
        for (int i = 0; i < kCpuWindow; ++i) {
            const uint64_t v = cpu_ring_[cls][i].load(std::memory_order_relaxed);
            if (!v) continue;
            if (!lo || v < lo) { second = lo; lo = v; }
            else if (!second || v < second) second = v;
        }
        return second ? second : lo;
    }
    std::atomic<uint64_t> gpu_ns_[2][kSizeClasses] = {};
    mutable std::atomic<uint32_t> explore_{0};
    std::string note_;
    std::unique_ptr<HostEncoder> host_;
    void* giga_ = nullptr;          // gtokcpu handle
    std::string norm_;              // HostEncoder normalization scratch
    std::mutex mu_;
};

}  // namespace gbpe
