// C-ABI shim around GraphTok for ctypes-based integration (e.g. vLLM).
// Exposes a minimal, stable C interface; all C++/CUDA stays internal.
//
// Build: see build_shim.sh -> libgtok.so
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <memory>
#include <string>
#include <string_view>
#include <vector>
#include <cuda_runtime.h>
#include "vocab.h"
#include "pretokenize.h"
#include "tokenizer.cuh"
#include "cpu_route.h"
#include "dispatcher.h"

namespace {
// One captured graph per power-of-two input-byte class, for single-document
// encodes (gtok_encode / gtok_encode_to_device). Every kernel of the captured
// graph is sized by its context's CAPACITY, not by the current input, so a
// context sized for the handle's byte cap (8 MiB in Dynamo) makes a 0.5 MB
// request pay for 8 MiB: measured on an H100 NVL, a 121k-token Qwen request
// took 1.55 ms that way against 0.61 ms in a context sized to its class. Same
// policy as the Python bindings' ladder: replay the EXACT class of the input
// (not merely the smallest that fits, which would bring the tax back), build
// a class on first use, keep at most kMaxClasses, then reuse the tightest fit.
struct SizeClass {
    uint32_t cap_input = 0;
    std::unique_ptr<gbpe::TokenizerCtx> ctx;
};
constexpr size_t kMaxClasses = 8;

struct Handle {
    gbpe::HostVocab  hv;
    gbpe::VocabPack  vp;
    gbpe::TokenizerCtx* ctx = nullptr;   // batch context, built on first batch
    uint32_t cap_bytes = 0;
    uint32_t cap_docs  = 0;   // max documents a single gtok_encode_batch can take
    std::vector<SizeClass> classes;      // increasing cap_input
    // CPU route for inputs below the graph's fixed-cost crossover (see
    // cpu_route.h); built after hv is loaded, destroyed before it.
    std::unique_ptr<gbpe::CpuRoute> cpu;
    std::vector<uint32_t> cpu_out;       // reused CPU-route output
};

uint32_t size_class(uint64_t bytes) {
    uint64_t p = 4096;
    while (p < bytes + 1024) p <<= 1;
    return static_cast<uint32_t>(p);
}

// Context for a single document of `len` (normalized) bytes, or nullptr if
// one could not be built.
gbpe::TokenizerCtx* single_ctx(Handle* h, uint32_t len) {
    const uint32_t want = size_class(len);
    for (SizeClass& c : h->classes)
        if (c.cap_input == want) return c.ctx.get();
    if (h->classes.size() >= kMaxClasses) {
        for (SizeClass& c : h->classes)          // tightest existing fit
            if (c.cap_input >= want) return c.ctx.get();
        h->classes.erase(h->classes.begin());   // input beyond every class held
    }
    SizeClass nc;
    nc.cap_input = want;
    try {
        nc.ctx = std::make_unique<gbpe::TokenizerCtx>(
            h->vp, gbpe::estimate_max_pretokens(want),
            gbpe::estimate_max_long_pretokens(want), want, want,
            /*max_decode_tokens*/ 0, /*max_decode_bytes*/ 0, /*max_batch_docs*/ 1);
    } catch (...) {
        return nullptr;
    }
    auto pos = std::lower_bound(h->classes.begin(), h->classes.end(), want,
        [](const SizeClass& c, uint32_t v) { return c.cap_input < v; });
    return h->classes.insert(pos, std::move(nc))->ctx.get();
}

// Initial batch-document capacity. Sized generously (the per-doc device state is
// a handful of uint32 arrays, so this is kilobytes, not megabytes) so that
// typical serving batches never trigger a graph re-capture. Larger batches grow
// the capacity in place by doubling — see ensure_docs().
constexpr uint32_t kInitialCapDocs = 256;

// Rebuild the context with at least `n_docs` document capacity, keeping the
// byte caps where they are. Growth is monotonic (doubling). Returns false if
// the rebuild failed - in that case the handle keeps its EXISTING context and
// caps, so a failed grow costs only this call and never poisons the handle for
// later gtok_encode/gtok_encode_batch calls. The new context is constructed
// BEFORE the old one is released, which briefly doubles the workspace; that is
// the price of not losing a working context on an allocation failure.
bool ensure_docs(Handle* h, uint32_t n_docs) {
    if (h->ctx && n_docs <= h->cap_docs) return true;
    uint32_t want = h->cap_docs ? h->cap_docs : kInitialCapDocs;
    while (want < n_docs) {
        if (want > (0xFFFFFFFFu >> 1)) return false;
        want <<= 1;
    }
    gbpe::TokenizerCtx* fresh = nullptr;
    try {
        fresh = new gbpe::TokenizerCtx(h->vp, h->cap_bytes, h->cap_bytes/2 + 1,
                                       h->cap_bytes, h->cap_bytes,
                                       /*max_decode_tokens*/ 0,
                                       /*max_decode_bytes*/ 0,
                                       /*max_batch_docs*/ want);
    } catch (...) {
        delete fresh;
        return false;                  // old ctx + caps left intact
    }
    delete h->ctx;
    h->ctx = fresh;
    h->cap_docs = want;
    return true;
}

// Apply the tokenizer.json normalizer before the graph, exactly as the Python
// bindings do (normalize_input_view). Byte-level Qwen-2.5/3 declare NFC, and
// skipping it is not bit-exact on decomposed multilingual text. NFC is the
// identity on ASCII, so the common case returns a view with no copy. The
// result can differ in length from the input: check caps on what it returns.
std::string_view normalized(const Handle* h, const uint8_t* text, uint32_t len,
                            std::string& storage) {
    std::string_view v(reinterpret_cast<const char*>(text), len);
    // Already normalized (always true without a normalizer, and for all
    // ASCII): hand the caller's bytes through with no copy. Only text that
    // normalization would change is copied, once, into `storage`.
    return gbpe::normalize_view_for_vocab(h->hv, v, storage) ? std::string_view(storage) : v;
}
}

extern "C" {

// Create a tokenizer handle from an HF tokenizer.json. max_bytes is the largest
// input the context is sized for (buffers grow by re-create if exceeded).
// Returns nullptr on failure.
void* gtok_create(const char* vocab_json_path, uint32_t max_bytes) {
    Handle* h = new Handle();
    if (!gbpe::load_hf_tokenizer_json(vocab_json_path, h->hv)) { delete h; return nullptr; }
    h->vp = gbpe::build_vocab_pack(h->hv);
    h->cap_bytes = max_bytes < 4096 ? 4096 : max_bytes;
    try {
        h->cpu = std::make_unique<gbpe::CpuRoute>(h->hv, vocab_json_path, std::string(),
                                                  gbpe::kCpuMaxBytesAuto);
    } catch (...) {
        gbpe::free_vocab_pack(h->vp);
        delete h;
        return nullptr;
    }
    // short/long/out caps generously sized off byte cap (>=1 token/byte upper bound).
    // Contexts are built on first use: single documents get one per size
    // class (single_ctx), batches one sized to the byte cap (ensure_docs).
    return h;
}

// Encode raw UTF-8 bytes -> token IDs written into out[0..max_out). Returns the
// number of tokens, or -1 on error / -2 if out buffer too small / -3 if input
// exceeds the context byte cap (caller should re-create with a larger cap).
int32_t gtok_encode(void* handle, const uint8_t* text, uint32_t len,
                    uint32_t* out, uint32_t max_out) {
    if (!handle) return -1;
    Handle* h = static_cast<Handle*>(handle);
    if (h->cpu->takes(reinterpret_cast<const char*>(text), len)) {
        try {
            h->cpu->encode(std::string_view(reinterpret_cast<const char*>(text), len),
                           h->cpu_out);
            if (h->cpu_out.size() > max_out) return -2;
            if (!h->cpu_out.empty())
                std::memcpy(out, h->cpu_out.data(), h->cpu_out.size() * sizeof(uint32_t));
            return (int32_t)h->cpu_out.size();
        } catch (...) {
            // fall through to the GPU route
        }
    }
    std::string storage;
    const std::string_view norm = normalized(h, text, len, storage);
    if (norm.size() > h->cap_bytes) return -3;
    gbpe::TokenizerCtx* ctx = single_ctx(h, static_cast<uint32_t>(norm.size()));
    if (!ctx) return -1;
    const bool captured = !ctx->graph_captured();   // this call captures
    gbpe::EncodeInput in{};   // value-init: zero all host-pretok fields
    in.raw_text = reinterpret_cast<const uint8_t*>(norm.data());
    in.raw_len  = static_cast<uint32_t>(norm.size());
    uint32_t n = 0;
    const auto t0 = std::chrono::steady_clock::now();
    try {
        n = ctx->encode_into(in, out, max_out);   // straight into `out`
    } catch (...) { return -1; }
    if (!captured) {   // a first-use graph capture is not a cost sample
        h->cpu->observe_gpu(reinterpret_cast<const char*>(text), len, static_cast<uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::steady_clock::now() - t0).count()));
    }
    if (n == UINT32_MAX) return -2;
    return (int32_t)n;
}

// Encode raw UTF-8 -> token IDs written DEVICE-TO-DEVICE into out_dev (a
// caller-provided device buffer, e.g. a torch CUDA tensor's data_ptr()).
// Only the token COUNT crosses to the host (4 bytes). Returns the count, or
// -1 on error, -2 if out_dev capacity (max_out) is too small, -3 if the input
// exceeds the context byte cap (caller should re-create with a larger cap).
int32_t gtok_encode_to_device(void* handle, const uint8_t* text, uint32_t len,
                              void* out_dev, uint32_t max_out) {
    if (!handle || !out_dev) return -1;
    Handle* h = static_cast<Handle*>(handle);
    std::string storage;
    const std::string_view norm = normalized(h, text, len, storage);
    if (norm.size() > h->cap_bytes) return -3;
    gbpe::TokenizerCtx* ctx = single_ctx(h, static_cast<uint32_t>(norm.size()));
    if (!ctx) return -1;
    gbpe::EncodeInput in{};   // value-init: zero all host-pretok fields
    in.raw_text = reinterpret_cast<const uint8_t*>(norm.data());
    in.raw_len  = static_cast<uint32_t>(norm.size());
    try {
        uint32_t n = 0;
        float k = 0, e = 0;
        const uint32_t* d_tokens = ctx->encode_to_device(in, &n, &k, &e);
        if (n > max_out) return -2;
        if (n > 0 &&
            cudaMemcpy(out_dev, d_tokens, n * sizeof(uint32_t),
                       cudaMemcpyDeviceToDevice) != cudaSuccess) return -1;
        return (int32_t)n;
    } catch (...) {
        return -1;
    }
}

// Batched encode: `n_docs` documents supplied as one concatenated byte blob
// plus an [n_docs+1] table of byte offsets into it. All documents are encoded
// in ONE graph replay with a single host sync; the result is bit-identical to
// calling gtok_encode() per document and concatenating.
//
// out_tokens receives the flat token stream (out_cap entries of room);
// out_doc_offsets receives n_docs+1 running token counts, so document i is
// out_tokens[out_doc_offsets[i] .. out_doc_offsets[i+1]).
//
// Like gtok_encode, each document is first passed through the vocabulary's
// normalizer (NFC for Qwen-2.5/3; none for GPT-2/Llama-3/DeepSeek-V3).
//
// Returns the total token count, or:
//   -1 error, -2 out_tokens capacity too small, -3 total input bytes exceed the
//   context byte cap (caller should re-create with a larger cap), -4 n_docs
//   exceeds the context's document capacity and it could not be grown.
int32_t gtok_encode_batch(void* handle,
                          const uint8_t* concat_bytes,
                          const uint32_t* doc_byte_offsets, /* [n_docs+1] */
                          uint32_t n_docs,
                          uint32_t* out_tokens, uint32_t out_cap,
                          uint32_t* out_doc_offsets /* [n_docs+1] */) {
    if (!handle || !concat_bytes || !doc_byte_offsets || !out_tokens ||
        !out_doc_offsets || n_docs == 0) return -1;
    Handle* h = static_cast<Handle*>(handle);
    if (doc_byte_offsets[n_docs] >= doc_byte_offsets[0] &&
        h->cpu->takes(reinterpret_cast<const char*>(concat_bytes) + doc_byte_offsets[0],
                      doc_byte_offsets[n_docs] - doc_byte_offsets[0])) {
        // Small batch: the CPU route, document by document (a graph replay's
        // fixed cost would dominate).
        try {
            uint32_t total = 0;
            out_doc_offsets[0] = 0;
            for (uint32_t i = 0; i < n_docs; ++i) {
                if (doc_byte_offsets[i + 1] < doc_byte_offsets[i]) return -1;
                h->cpu->encode(std::string_view(
                    reinterpret_cast<const char*>(concat_bytes + doc_byte_offsets[i]),
                    doc_byte_offsets[i + 1] - doc_byte_offsets[i]), h->cpu_out);
                if (total + h->cpu_out.size() > out_cap) return -2;
                if (!h->cpu_out.empty())
                    std::memcpy(out_tokens + total, h->cpu_out.data(),
                                h->cpu_out.size() * sizeof(uint32_t));
                total += static_cast<uint32_t>(h->cpu_out.size());
                out_doc_offsets[i + 1] = total;
            }
            return (int32_t)total;
        } catch (const std::runtime_error&) {
            // fall through to the GPU route
        }
    }
    // ensure_docs() leaves the existing ctx intact on failure, so a -4 here is
    // recoverable: smaller batches still work through the current context.
    if (!ensure_docs(h, n_docs)) return -4;

    std::vector<const uint8_t*> ptrs(n_docs);
    std::vector<uint32_t>       lens(n_docs);
    std::vector<std::string>    storage(n_docs);   // only filled when normalized
    uint64_t total = 0;
    for (uint32_t i = 0; i < n_docs; ++i) {
        if (doc_byte_offsets[i + 1] < doc_byte_offsets[i]) return -1;
        const std::string_view norm = normalized(
            h, concat_bytes + doc_byte_offsets[i],
            doc_byte_offsets[i + 1] - doc_byte_offsets[i], storage[i]);
        ptrs[i] = reinterpret_cast<const uint8_t*>(norm.data());
        lens[i] = static_cast<uint32_t>(norm.size());
        total += norm.size();
    }
    if (total > h->cap_bytes) return -3;
    gbpe::BatchEncodeInput in{};
    in.docs   = ptrs.data();
    in.lens   = lens.data();
    in.n_docs = n_docs;

    std::vector<uint32_t> toks, offs;
    float k = 0, e = 0;
    try {
        h->ctx->encode_batch(in, toks, offs, &k, &e);
    } catch (...) { return -1; }
    if ((uint32_t)toks.size() > out_cap) return -2;
    if (offs.size() != (size_t)n_docs + 1) return -1;
    if (!toks.empty())
        std::memcpy(out_tokens, toks.data(), toks.size()*sizeof(uint32_t));
    std::memcpy(out_doc_offsets, offs.data(), offs.size()*sizeof(uint32_t));
    return (int32_t)toks.size();
}

// CPU route controls. Inputs of at most gtok_cpu_max_bytes() raw bytes (the
// multilingual limit if over 1/32 of them are non-ASCII) are encoded on the
// CPU (gigatoken, else the host encoder); setting it sets both limits, 0
// disables the route. Defaults: the measured crossovers in cpu_thresholds.h,
// or GTOK_CPU_BACKEND / GTOK_CPU_MAX_BYTES at gtok_create.
uint32_t gtok_cpu_max_bytes(void* handle) {
    return handle ? static_cast<Handle*>(handle)->cpu->max_bytes() : 0u;
}
void gtok_set_cpu_max_bytes(void* handle, uint32_t max_bytes) {
    if (handle) static_cast<Handle*>(handle)->cpu->set_max_bytes(max_bytes);
}
// "gigatoken", "host" or "off" (static string).
const char* gtok_cpu_backend(void* handle) {
    return handle ? static_cast<Handle*>(handle)->cpu->backend_name() : "off";
}

// ---------------------------------------------------------------------------
// Load-adaptive dispatcher (dispatcher.h): ONE object shared by every request
// thread of a serving frontend. gtok_dispatch_encode is thread-safe and
// blocking; concurrent calls are combined into GPU batches, and small inputs
// run on one of `cpu_workers` CPU engines when one is free.
//   policy: 0 adaptive, 1 GPU batches only, 2 CPU engines only (for inputs
//   the CPU route accepts). cpu_workers = 0 disables the CPU engines.
// Returns nullptr on failure.
void* gtok_dispatcher_create(const char* vocab_json_path, uint32_t cpu_workers,
                             uint32_t max_batch_bytes, uint32_t max_batch_docs,
                             uint32_t policy) {
    if (!vocab_json_path || policy > 2) return nullptr;
    try {
        return new gbpe::Dispatcher(vocab_json_path, cpu_workers, max_batch_bytes,
                                    max_batch_docs,
                                    static_cast<gbpe::DispatchPolicy>(policy), std::string());
    } catch (...) {
        return nullptr;
    }
}

// Same contract as gtok_encode: token count, -1 error, -2 out too small.
int32_t gtok_dispatch_encode(void* dispatcher, const uint8_t* text, uint32_t len,
                             uint32_t* out, uint32_t max_out) {
    if (!dispatcher || (!text && len)) return -1;
    thread_local std::vector<uint32_t> ids;
    try {
        static_cast<gbpe::Dispatcher*>(dispatcher)->encode(
            std::string_view(reinterpret_cast<const char*>(text), len), ids);
    } catch (...) {
        return -1;
    }
    if (ids.size() > max_out) return -2;
    if (!ids.empty()) std::memcpy(out, ids.data(), ids.size() * sizeof(uint32_t));
    return (int32_t)ids.size();
}

// Counters since creation: [cpu_requests, gpu_requests, gpu_batches,
// gpu_large, gpu_batch_ns (wall time inside batch replays)].
void gtok_dispatcher_stats(void* dispatcher, uint64_t* out5) {
    if (!dispatcher || !out5) return;
    const gbpe::DispatchStats s = static_cast<gbpe::Dispatcher*>(dispatcher)->stats();
    out5[0] = s.cpu_requests;
    out5[1] = s.gpu_requests;
    out5[2] = s.gpu_batches;
    out5[3] = s.gpu_large;
    out5[4] = s.gpu_batch_ns;
}

void gtok_dispatcher_destroy(void* dispatcher) {
    delete static_cast<gbpe::Dispatcher*>(dispatcher);
}

uint32_t gtok_vocab_size(void* handle) {
    if (!handle) return 0;
    return static_cast<Handle*>(handle)->vp.vocab_size;
}

void gtok_destroy(void* handle) {
    if (!handle) return;
    Handle* h = static_cast<Handle*>(handle);
    delete h->ctx;
    h->classes.clear();              // before the vocab pack they reference
    h->cpu.reset();                  // references h->hv
    gbpe::free_vocab_pack(h->vp);   // device merge table + decode arrays (~17 MB)
    delete h;
}

} // extern "C"
