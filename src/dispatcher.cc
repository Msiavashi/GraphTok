// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 the GraphTok contributors
#include "dispatcher.h"

#include "pretokenize.h"

#include <algorithm>
#include <chrono>
#include <stdexcept>

namespace gbpe {

namespace {

// Byte class of a batch context: a power of two with the same 1 KiB slack the
// single-document ladders use, so a class covers every batch that maps to it.
uint32_t byte_class(uint64_t bytes) {
    uint64_t p = 4096;
    while (p < bytes + 1024) p <<= 1;
    if (p > 0x80000000ull) throw std::length_error("dispatcher: batch too large");
    return static_cast<uint32_t>(p);
}

constexpr size_t kMaxClasses = 8;

}  // namespace

Dispatcher::Dispatcher(const std::string& vocab_path, uint32_t cpu_workers,
                       uint32_t max_batch_bytes, uint32_t max_batch_docs,
                       DispatchPolicy policy, const std::string& cpu_backend)
    : policy_(policy),
      max_batch_bytes_(std::max<uint32_t>(max_batch_bytes, 4096)),
      max_batch_docs_(std::max<uint32_t>(max_batch_docs, 1))
{
    if (!load_hf_tokenizer_json(vocab_path, hv_)) {
        throw std::runtime_error("dispatcher: failed to load " + vocab_path);
    }
    vp_ = build_vocab_pack(hv_);
    cpu_.reserve(cpu_workers);
    for (uint32_t i = 0; i < cpu_workers; ++i) {
        cpu_.push_back(std::make_unique<CpuRoute>(hv_, vocab_path, cpu_backend,
                                                  kCpuMaxBytesAuto));
        if (cpu_.back()->backend() == CpuBackend::Off) {   // no engine for this family
            cpu_.clear();
            break;
        }
    }
    busy_ = std::make_unique<std::atomic<bool>[]>(cpu_.size() ? cpu_.size() : 1);
    for (size_t i = 0; i < cpu_.size(); ++i) busy_[i].store(false);
}

Dispatcher::~Dispatcher() {
    ladder_.clear();
    large_.clear();
    cpu_.clear();
    free_vocab_pack(vp_);
}

const char* Dispatcher::cpu_backend() const {
    return cpu_.empty() ? "off" : cpu_.front()->backend_name();
}

DispatchStats Dispatcher::stats() const {
    DispatchStats s;
    s.cpu_requests = n_cpu_.load();
    s.gpu_requests = n_gpu_.load();
    s.gpu_batches  = n_batches_.load();
    s.gpu_large    = n_large_.load();
    s.gpu_batch_ns = batch_ns_.load();
    return s;
}

int Dispatcher::try_acquire_cpu() {
    for (size_t i = 0; i < cpu_.size(); ++i) {
        bool expected = false;
        if (busy_[i].compare_exchange_strong(expected, true, std::memory_order_acquire)) {
            return static_cast<int>(i);
        }
    }
    return -1;
}

int Dispatcher::acquire_cpu_blocking() {
    int i = try_acquire_cpu();
    if (i >= 0) return i;
    std::unique_lock<std::mutex> lk(cpu_wait_mu_);
    cpu_wait_cv_.wait(lk, [&] { return (i = try_acquire_cpu()) >= 0; });
    return i;
}

void Dispatcher::release_cpu(int i) {
    busy_[i].store(false, std::memory_order_release);
    if (policy_ == DispatchPolicy::CpuOnly) {
        std::lock_guard<std::mutex> lk(cpu_wait_mu_);
        cpu_wait_cv_.notify_one();
    }
}

void Dispatcher::encode(std::string_view raw, std::vector<uint32_t>& out) {
    if (!cpu_.empty() && policy_ != DispatchPolicy::GpuOnly &&
        cpu_.front()->takes(raw.data(), raw.size())) {
        const int i = policy_ == DispatchPolicy::CpuOnly ? acquire_cpu_blocking()
                                                         : try_acquire_cpu();
        if (i >= 0) {
            try {
                cpu_[i]->encode(raw, out);
                release_cpu(i);
                n_cpu_.fetch_add(1, std::memory_order_relaxed);
                return;
            } catch (const std::runtime_error&) {
                release_cpu(i);          // engine failed: take the GPU route
            }
        }
    }
    std::string storage;
    std::string_view norm = raw;
    if (normalize_view_for_vocab(hv_, raw, storage)) norm = storage;
    if (norm.size() > max_batch_bytes_) {
        run_large(norm, out);
        n_large_.fetch_add(1, std::memory_order_relaxed);
        return;
    }
    gpu_encode(norm, out);
    n_gpu_.fetch_add(1, std::memory_order_relaxed);
}

// Leader/follower combining. The first caller to find no batch in flight
// leads: it launches everything queued (up to the caps) as one replay and
// hands out the results. Requests that arrived meanwhile form the next batch:
// the leader passes the lead straight to the oldest of them (waking only that
// thread) and returns. Each request waits on its own condition variable, so a
// finished batch wakes exactly its own callers and one new leader.
void Dispatcher::gpu_encode(std::string_view norm, std::vector<uint32_t>& out) {
    Req r;
    r.text = norm;
    r.out = &out;
    std::unique_lock<std::mutex> lk(q_mu_);
    queue_.push_back(&r);
    if (leader_active_) {
        r.cv.wait(lk, [&] { return r.done || r.lead; });
    } else {
        leader_active_ = true;
        r.lead = true;
    }
    while (!r.done) {                   // this thread leads until it is served
        std::vector<Req*> batch;
        uint64_t bytes = 0;
        while (!queue_.empty() && batch.size() < max_batch_docs_) {
            Req* q = queue_.front();
            if (!batch.empty() && bytes + q->text.size() > max_batch_bytes_) break;
            bytes += q->text.size();
            batch.push_back(q);
            queue_.pop_front();
        }
        lk.unlock();
        try {
            run_batch(batch);
        } catch (...) {
            for (Req* q : batch) q->err = std::current_exception();
        }
        lk.lock();
        for (Req* q : batch) q->done = true;
        Req* next = nullptr;
        if (r.done) {
            // Served: pass the lead on BEFORE waking this batch's callers, so
            // the next replay is not delayed by a string of wakeups.
            if (queue_.empty()) {
                leader_active_ = false;
            } else {
                next = queue_.front();
                next->lead = true;
            }
        }
        lk.unlock();
        if (next) next->cv.notify_one();
        for (Req* q : batch)
            if (q != &r) q->cv.notify_one();
        lk.lock();
    }
    lk.unlock();
    if (r.err) std::rethrow_exception(r.err);
}

TokenizerCtx* Dispatcher::batch_ctx(uint32_t total_bytes) {
    const uint32_t want = byte_class(total_bytes);
    for (Ctx& c : ladder_)
        if (c.cap_input == want) return c.ctx.get();
    if (ladder_.size() >= kMaxClasses) {
        for (Ctx& c : ladder_)
            if (c.cap_input >= want) return c.ctx.get();
        ladder_.erase(ladder_.begin());
    }
    Ctx nc;
    nc.cap_input = want;
    nc.ctx = std::make_unique<TokenizerCtx>(
        vp_, estimate_max_pretokens(want), estimate_max_long_pretokens(want),
        want, want, /*max_decode_tokens*/ 0, /*max_decode_bytes*/ 0, max_batch_docs_);
    // The batcher's whole point is to free the host while the GPU works, and
    // with a sleeping wait one wait per batch matters.
    nc.ctx->set_blocking_wait(true);
    nc.ctx->set_speculative_d2h(true);
    auto pos = std::lower_bound(ladder_.begin(), ladder_.end(), want,
        [](const Ctx& c, uint32_t v) { return c.cap_input < v; });
    return ladder_.insert(pos, std::move(nc))->ctx.get();
}

void Dispatcher::run_batch(const std::vector<Req*>& batch) {
    const auto t0 = std::chrono::steady_clock::now();
    std::vector<const uint8_t*> ptrs(batch.size());
    std::vector<uint32_t> lens(batch.size());
    uint64_t total = 0;
    for (size_t i = 0; i < batch.size(); ++i) {
        ptrs[i] = reinterpret_cast<const uint8_t*>(batch[i]->text.data());
        lens[i] = static_cast<uint32_t>(batch[i]->text.size());
        total += lens[i];
    }
    TokenizerCtx* ctx = batch_ctx(static_cast<uint32_t>(total));
    BatchEncodeInput in{};
    in.docs = ptrs.data();
    in.lens = lens.data();
    in.n_docs = static_cast<uint32_t>(batch.size());
    std::vector<uint32_t> toks, offs;
    ctx->encode_batch(in, toks, offs);
    for (size_t i = 0; i < batch.size(); ++i) {
        batch[i]->out->assign(toks.begin() + offs[i], toks.begin() + offs[i + 1]);
    }
    n_batches_.fetch_add(1, std::memory_order_relaxed);
    batch_ns_.fetch_add(static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                            std::chrono::steady_clock::now() - t0).count()),
                        std::memory_order_relaxed);
}

void Dispatcher::run_large(std::string_view norm, std::vector<uint32_t>& out) {
    std::lock_guard<std::mutex> lk(large_mu_);
    const uint32_t want = byte_class(norm.size());
    TokenizerCtx* ctx = nullptr;
    for (Ctx& c : large_)
        if (c.cap_input == want) ctx = c.ctx.get();
    if (!ctx) {
        if (large_.size() >= kMaxClasses) large_.erase(large_.begin());
        Ctx nc;
        nc.cap_input = want;
        nc.ctx = std::make_unique<TokenizerCtx>(
            vp_, estimate_max_pretokens(want), estimate_max_long_pretokens(want),
            want, want, 0, 0, 1);
        nc.ctx->set_blocking_wait(true);
        nc.ctx->set_speculative_d2h(true);
        ctx = nc.ctx.get();
        large_.push_back(std::move(nc));
    }
    EncodeInput in{};
    in.raw_text = reinterpret_cast<const uint8_t*>(norm.data());
    in.raw_len = static_cast<uint32_t>(norm.size());
    ctx->encode(in, out, nullptr, nullptr);
}

}  // namespace gbpe
