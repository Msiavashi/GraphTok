// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 the GraphTok contributors
//
// Python bindings for gpu_bpe_tokenizer. The Tokenizer class is a thin
// wrapper around HostVocab + VocabPack + TokenizerCtx that
// mirrors src/main.cc exactly so bit-exactness with the CLI is guaranteed.
//
// Bucketed capacity: one TokenizerCtx (and one captured CUDA graph) per
// power-of-two input-byte class, built on first use and kept. An encode
// replays the smallest class that fits the CURRENT input, so replay cost
// tracks the input rather than the largest input the object has ever seen.

#include <chrono>
#include <optional>
#include "vocab.h"
#include "pretokenize.h"
#include "host_encode.h"
#include "cpu_route.h"
#include "dispatcher.h"
#include "tokenizer.cuh"

// Batch encode writes the device's token block straight into the NumPy array
// that is returned. 0 selects the legacy device -> pinned -> vector -> NumPy
// form, kept only so the two can be A/B'd on one binary.
#ifndef GBPE_BATCH_DIRECT_NUMPY
#define GBPE_BATCH_DIRECT_NUMPY 1
#endif

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace py = pybind11;

namespace {

const char* regex_kind_name(gbpe::RegexKind k) {
    switch (k) {
        case gbpe::RegexKind::GPT2:       return "GPT2";
        case gbpe::RegexKind::Llama3:     return "Llama3";
        case gbpe::RegexKind::Qwen25:     return "Qwen25";
        case gbpe::RegexKind::DeepSeekV3: return "DeepSeekV3";
        case gbpe::RegexKind::Gemma3:     return "Gemma3";
    }
    return "Unknown";
}

// Round up to next power of two, with a small floor.
uint32_t next_pow2(uint64_t x, uint32_t floor_v = 1024) {
    if (x > 0x80000000ull) {
        throw std::length_error("cuTokenize: requested graph capacity exceeds 2^31");
    }
    if (x < floor_v) x = floor_v;
    uint32_t p = 1;
    while (p < x) p <<= 1;
    return p;
}

// The carrier type handed to torch.as_tensor(): a plain Python class created
// once, whose instances hold the __cuda_array_interface__ dict and a strong
// reference to the owning Tokenizer. Built with `type()` (not
// types.SimpleNamespace) so instances support both attribute assignment and
// weak references, which torch's tensor bookkeeping requires.
py::object device_buffer_holder_type() {
    static py::object cls = py::module_::import("builtins").attr("type")(
        py::str("_CuTokenizeDeviceBuffer"),
        py::make_tuple(),
        py::dict(py::arg("__doc__") =
                     py::str("Carries a cuTokenize device token buffer's "
                             "__cuda_array_interface__ and keeps its owning "
                             "Tokenizer alive.")));
    return cls;
}

// Wrap a device buffer of `count` token IDs at `dptr` as a torch.Tensor WITHOUT
// copying, by handing torch a throwaway object that carries the CUDA array
// interface (v2). The tensor aliases the buffer — see encode_to_device.
//
// dtype is advertised as int32, not uint32: token IDs are small non-negative
// values (<< 2^31) so the 4-byte pattern is identical (still zero-copy), and
// int32 is a type torch.nn.functional.embedding accepts as indices — torch
// rejects uint32 there.
//
// `owner` is the Tokenizer that owns the buffer. torch.as_tensor() keeps a
// STRONG reference to the object exposing __cuda_array_interface__, so stashing
// the owner on that carrier is what keeps the Tokenizer (and therefore the
// device allocation) alive for as long as the tensor is. The carrier must
// therefore be an instance of a class that can hold attributes AND be strongly
// referenced by torch — a `types.SimpleNamespace` is not weak-referenceable and
// left the tensor dangling once the Tokenizer was dropped, so we build a tiny
// dedicated holder type instead.
py::object wrap_device_tokens_as_torch(const uint32_t* dptr, uint32_t count,
                                       py::object owner = py::none()) {
    py::dict cai;
    cai["shape"]   = py::make_tuple(count);
    cai["typestr"] = py::str("<i4");                              // LE int32
    cai["data"]    = py::make_tuple(reinterpret_cast<uintptr_t>(dptr),
                                    /*read_only=*/false);
    cai["strides"] = py::none();
    cai["version"] = 2;

    // torch.as_tensor() consumes any object exposing __cuda_array_interface__
    // and shares its memory (zero-copy) on the pointer's device, retaining a
    // strong reference to this carrier (and through it, to `owner`).
    py::object holder = device_buffer_holder_type()();
    py::setattr(holder, "__cuda_array_interface__", cai);
    py::setattr(holder, "_cutokenize_owner", owner);
    return py::module_::import("torch").attr("as_tensor")(holder);
}

class Tokenizer {
    // One captured graph per power-of-two input-byte class; see the note above
    // ensure_capacity_for() for why the ladder exists.
    struct Bucket {
        uint32_t cap_input  = 0;
        uint32_t cap_output = 0;
        uint32_t cap_short  = 0;
        uint32_t cap_long   = 0;
        uint32_t cap_docs   = 0;
        // Decode workspaces live in the TokenizerCtx, so they are per-bucket
        // and can lag the ladder-wide high-water marks: a bucket built before
        // a decode ever ran carries none. ensure_decode_capacity_for() tests
        // THESE, not the ladder-wide caps, or a decode after switching back to
        // an older bucket would run against a context with no decode buffers.
        uint32_t cap_decode_tokens = 0;
        uint32_t cap_decode_bytes  = 0;
        std::unique_ptr<gbpe::TokenizerCtx> ctx;
    };

public:
    Tokenizer(const std::string& vocab_path,
              uint32_t max_input_chars,
              bool enable_decode,
              std::optional<uint32_t> host_max_bytes,
              const std::string& cpu_backend)
        : vocab_path_(vocab_path), enable_decode_(enable_decode)
    {
        if (!gbpe::load_hf_tokenizer_json(vocab_path, hv_)) {
            throw std::runtime_error("failed to load tokenizer.json: " + vocab_path);
        }
        vp_ = gbpe::build_vocab_pack(hv_);
        cpu_ = std::make_unique<gbpe::CpuRoute>(
            hv_, vocab_path, cpu_backend,
            host_max_bytes ? *host_max_bytes : gbpe::kCpuMaxBytesAuto);
#if GBPE_HAVE_FAMILY_SP
        sp_family_ = hv_.byte_fallback;
#if !GBPE_GEMMA_GPU_PRETOK
        if (sp_family_) {
            sp_vi_.id_by_token        = &hv_.id_by_token;
            sp_vi_.byte_fallback      = hv_.byte_fallback_id.data();
            sp_vi_.max_token_byte_len = hv_.max_token_byte_len;
            sp_vi_.host_merges        = &hv_.host_merges;
            sp_pretok_ = std::make_unique<gbpe::SPPreTokenizer>(sp_vi_);
        }
#endif
        // Byte-level families need no host pre-tokenizer — it runs on the GPU.
#endif

        // Pre-size to max_input_chars (default 4096) so the first encode
        // doesn't pay the graph-capture penalty for a long-lived Tokenizer.
        // Larger inputs get their own power-of-two size class, captured once
        // on first use and kept; a later smaller input replays the smallest
        // class that fits it, so one big prompt does not tax the small ones
        // that follow (see the note above ensure_capacity_for).
        ensure_capacity_for(std::max<uint32_t>(max_input_chars, 256));
    }

    ~Tokenizer() {
        ctx_ = nullptr;
        buckets_.clear();          // destroys every context (and its graph)
        gbpe::free_vocab_pack(vp_);
    }

    Tokenizer(const Tokenizer&) = delete;
    Tokenizer& operator=(const Tokenizer&) = delete;

    // Inputs at or below this many raw bytes take the CPU route (gigatoken or
    // the host encoder, see cpu_route.h). 0 disables it; it is forced to 0
    // when no CPU engine supports the family (Gemma).
    uint32_t host_max_bytes() const { return cpu_->max_bytes(); }
    uint32_t host_max_bytes_multilingual() const { return cpu_->max_bytes_multilingual(); }
    std::string cpu_backend() const { return cpu_->backend_name(); }
    std::string last_route() const { return last_route_; }
    std::vector<uint64_t> cpu_cost_model() const { return cpu_->cost_model(); }
    std::string cpu_backend_note() const { return cpu_->note(); }

    // The captured size classes currently held, smallest first (input-byte
    // capacities). Diagnostic: it is how a caller confirms that a small encode
    // after a large one replays a small graph rather than the large one.
    std::vector<uint32_t> graph_capacities() const {
        std::vector<uint32_t> v;
        v.reserve(buckets_.size());
        for (const Bucket& b : buckets_) v.push_back(b.cap_input);
        return v;
    }
    // Input-byte capacity of the graph the last encode replayed.
    uint32_t active_capacity() const { return cap_input_; }
    void set_host_max_bytes(uint32_t v) { cpu_->set_max_bytes(v); }

    // CPU route with the GIL released for inputs long enough that the
    // release pays for itself. False (after logging nothing) if the engine
    // failed, so the caller falls through to the GPU route.
    bool cpu_encode(std::string_view raw, std::vector<uint32_t>& tokens) {
        try {
            if (raw.size() >= kCpuReleaseGilBytes) {
                py::gil_scoped_release release;
                cpu_->encode(raw, tokens);
            } else {
                cpu_->encode(raw, tokens);
            }
            return true;
        } catch (const std::runtime_error&) {
            return false;
        }
    }

    // Borrow the caller's bytes without copying them. For an ordinary str this
    // is CPython's cached UTF-8 form (the object's own storage when it is
    // ASCII); for bytes it is the buffer itself. Anything else falls back to
    // pybind11's copying caster via `owned`, which must outlive the returned
    // view. See the note on the copy cost above encode_numpy().
    static std::string_view borrow_text(PyObject* obj, std::string& owned,
                                        const py::handle& h) {
        if (PyUnicode_CheckExact(obj)) {
            Py_ssize_t size = 0;
            const char* data = PyUnicode_AsUTF8AndSize(obj, &size);
            if (data == nullptr) throw py::error_already_set();
            return std::string_view(data, static_cast<size_t>(size));
        }
        if (PyBytes_CheckExact(obj)) {
            return std::string_view(PyBytes_AS_STRING(obj),
                                    static_cast<size_t>(PyBytes_GET_SIZE(obj)));
        }
        owned = py::reinterpret_borrow<py::object>(h).cast<std::string>();
        return std::string_view(owned);
    }

    // py::object, not const std::string&: the latter routes through
    // pybind11's std::string caster, which allocates and copies every input
    // byte before the encoder sees it. That copy measured 3.1-3.4 us per call
    // at 2 KiB on ASCII text and 11.0 us on multilingual (where CPython must
    // also transcode UCS-2 to UTF-8), against 0.34 us for the str's own
    // .encode('utf-8') -- i.e. it was a larger cost than most of what this
    // session optimised. encode_numpy() already avoided it; encode() did not.
    std::vector<uint32_t> encode(py::object text) {
        std::string owned;
        const std::string_view view = borrow_text(text.ptr(), owned, text);
        std::vector<uint32_t> tokens;
        encode_impl(view, tokens);
        return tokens;
    }

    // Takes py::object rather than const std::string& so the common case --
    // an ordinary str -- costs no copy. pybind11's std::string caster
    // allocates and copies every byte, which measured 4.8 us at 512 B and
    // 9.4 us at 8 KiB, most of the gap between our encoder's own throughput
    // and what a caller sees. PyUnicode_AsUTF8AndSize borrows CPython's
    // cached UTF-8 form instead (for an ASCII str, the object's own storage),
    // and bytes objects expose their buffer directly. Anything else falls
    // back to the copying caster.
    py::array_t<uint32_t> encode_numpy(py::object text) {
        const char* data = nullptr;
        Py_ssize_t size = 0;
        std::string owned;                     // only used by the fallback
        PyObject* obj = text.ptr();
        if (PyUnicode_CheckExact(obj)) {
            data = PyUnicode_AsUTF8AndSize(obj, &size);
            if (data == nullptr) throw py::error_already_set();
        } else if (PyBytes_CheckExact(obj)) {
            data = PyBytes_AS_STRING(obj);
            size = PyBytes_GET_SIZE(obj);
        } else {
            owned = text.cast<std::string>();
            data = owned.data();
            size = static_cast<Py_ssize_t>(owned.size());
        }
        // Reuse one buffer across calls. A fresh vector per call makes the
        // host path's push_back grow from zero every time (~1k tokens at
        // 4 KiB), which cost more than the encode itself; the capacity
        // settles after a few calls and is never given back.
        const std::string_view view(data, static_cast<size_t>(size));
#if GBPE_HAVE_FAMILY_SP && !GBPE_GEMMA_GPU_PRETOK
        const bool byte_level = !sp_family_;
#else
        const bool byte_level = true;
#endif
        // One routing decision per call: takes() may explore (see
        // cpu_route.h), so asking twice could contradict itself.
        const bool cpu_route = cpu_->takes(view.data(), view.size());
        if (cpu_route) {
            std::vector<uint32_t>& tokens = scratch_tokens_;
            tokens.clear();
            if (cpu_encode(view, tokens)) {
                last_route_ = "cpu";
                py::array_t<uint32_t> arr(static_cast<py::ssize_t>(tokens.size()));
                if (!tokens.empty())
                    std::memcpy(arr.mutable_data(), tokens.data(), tokens.size() * sizeof(uint32_t));
                return arr;
            }
        }
        if (byte_level) {
            // GPU route: device -> pinned staging -> this array, one host
            // copy. Every token covers at least one input byte, so the
            // normalized length bounds the count; shrink to it afterwards.
            last_route_ = "gpu";
            std::string normalized;
            const std::string_view input = normalize_input_view(view, normalized);
            py::array_t<uint32_t> arr(static_cast<py::ssize_t>(std::max<size_t>(input.size(), 1)));
            const uint32_t n = gpu_encode_view(view, input, nullptr, arr.mutable_data(),
                                               static_cast<uint32_t>(arr.size()));
            if (n == UINT32_MAX) throw std::runtime_error("cuTokenize: token count exceeds input bytes");
            arr.resize({static_cast<py::ssize_t>(n)}, /*refcheck=*/false);
            return arr;
        }
        std::vector<uint32_t>& tokens = scratch_tokens_;
        tokens.clear();
        encode_impl(view, tokens);   // SP vocabularies
        py::array_t<uint32_t> arr(static_cast<py::ssize_t>(tokens.size()));
        if (!tokens.empty()) {
            std::memcpy(arr.mutable_data(), tokens.data(), tokens.size() * sizeof(uint32_t));
        }
        return arr;
    }

    // Encode and leave the token IDs on the GPU, returned as a torch.Tensor
    // (int32, 1-D) that shares the device memory — no host round-trip, no copy
    // The tensor BORROWS the encoder's output buffer,
    // which is overwritten by the next encode call on this Tokenizer; clone it
    // (tensor.clone()) if you need to keep it across calls.
    // `self` is this Tokenizer's Python handle; it is stashed on the returned
    // tensor's CAI carrier so dropping the last Python reference to the
    // Tokenizer cannot free the device buffer the tensor points at.
    py::object encode_to_device(const std::string& text, py::object self) {
#if GBPE_GPU_PRETOK
#if GBPE_HAVE_FAMILY_SP && !GBPE_GEMMA_GPU_PRETOK
        if (sp_family_) {
            throw std::runtime_error(
                "encode_to_device() is byte-level only (no SP/Gemma support)");
        }
#endif
        std::string normalized;
        const std::string& input = normalize_input(text, normalized);
        ensure_capacity_for(static_cast<uint32_t>(input.size()));

        gbpe::EncodeInput ei{};
        ei.raw_text = reinterpret_cast<const uint8_t*>(input.data());
        ei.raw_len  = static_cast<uint32_t>(input.size());

        // Issue the encode on torch's CURRENT cuda stream so producing the IDs
        // is ordered with the consumer that reads them (e.g. the embedding
        // lookup) AND with the next encode that reuses the output buffer. On a
        // private stream those race once the result is consumed asynchronously.
        void* stream = reinterpret_cast<void*>(
            py::module_::import("torch").attr("cuda").attr("current_stream")()
                .attr("cuda_stream").cast<uintptr_t>());

        const uint32_t* dptr = nullptr;
        uint32_t count = 0;
        float k_ms = 0.f, e_ms = 0.f;
        {
            py::gil_scoped_release release;   // GPU encode touches no Python state
            dptr = ctx_->encode_to_device(ei, &count, &k_ms, &e_ms, stream);
        }
        return wrap_device_tokens_as_torch(dptr, count, self);
#else
        (void)text; (void)self;
        throw std::runtime_error("GPU pretok not built");
#endif
    }

    // Encode a list of documents in ONE graph replay. The result is identical
    // to encoding each string separately and concatenating: offsets[i]:offsets[i+1]
    // slices document i out of the flat token array.
    // Small batches take the host path too: the threshold applies to the
    // batch's total bytes, so a handful of short prompts avoids the replay.
    bool host_batch(const std::vector<std::string>& texts,
                    std::vector<uint32_t>& tokens,
                    std::vector<uint32_t>& offsets) {
        std::vector<const char*> ps(texts.size());
        std::vector<size_t> ns(texts.size());
        for (size_t i = 0; i < texts.size(); ++i) {
            ps[i] = texts[i].data();
            ns[i] = texts[i].size();
        }
        if (!cpu_->takes_all(ps.data(), ns.data(), texts.size())) return false;
        py::gil_scoped_release release;
        tokens.clear();
        offsets.assign(1, 0u);
        std::vector<uint32_t> doc;
        try {
            for (const auto& t : texts) {
                cpu_->encode(t, doc);
                tokens.insert(tokens.end(), doc.begin(), doc.end());
                offsets.push_back(static_cast<uint32_t>(tokens.size()));
            }
        } catch (const std::runtime_error&) {
            return false;
        }
        return true;
    }

    py::tuple encode_batch(const std::vector<std::string>& texts) {
#if GBPE_GPU_PRETOK
        if (texts.empty()) {
            // No GPU work for an empty batch.
            py::array_t<uint32_t> empty(static_cast<py::ssize_t>(0));
            py::array_t<uint32_t> offs(static_cast<py::ssize_t>(1));
            offs.mutable_data()[0] = 0;
            return py::make_tuple(empty, offs);
        }
#if GBPE_HAVE_FAMILY_SP
        const bool single_ok = !sp_family_;   // SP vocabs reject encode_batch
#else
        const bool single_ok = true;
#endif
        if (texts.size() == 1 && single_ok) {
            // A batch of one is a single-document encode: same routing
            // (including the measured rule) as encode_numpy.
            py::array_t<uint32_t> tok_arr = encode_numpy(py::bytes(texts[0]));
            py::array_t<uint32_t> offs(static_cast<py::ssize_t>(2));
            offs.mutable_data()[0] = 0;
            offs.mutable_data()[1] = static_cast<uint32_t>(tok_arr.size());
            return py::make_tuple(tok_arr, offs);
        }
        std::vector<uint32_t> tokens, offsets;
        if (host_batch(texts, tokens, offsets)) {
            py::array_t<uint32_t> tok_arr(static_cast<py::ssize_t>(tokens.size()));
            if (!tokens.empty()) {
                std::memcpy(tok_arr.mutable_data(), tokens.data(),
                            tokens.size() * sizeof(uint32_t));
            }
            py::array_t<uint32_t> off_arr(static_cast<py::ssize_t>(offsets.size()));
            if (!offsets.empty()) {
                std::memcpy(off_arr.mutable_data(), offsets.data(),
                            offsets.size() * sizeof(uint32_t));
            }
            return py::make_tuple(tok_arr, off_arr);
        }

#if !GBPE_BATCH_DIRECT_NUMPY
        {   // Legacy path, kept for A/B only: device -> pinned -> vector -> NumPy.
            std::vector<const uint8_t*> p2;
            std::vector<uint32_t>       l2;
            std::vector<std::string>    s2;
            gbpe::BatchEncodeInput b2 = prepare_batch(texts, p2, l2, s2);
            float k2 = 0.f, e2 = 0.f;
            {
                py::gil_scoped_release release;
                ctx_->encode_batch(b2, tokens, offsets, &k2, &e2);
            }
            py::array_t<uint32_t> t2(static_cast<py::ssize_t>(tokens.size()));
            if (!tokens.empty())
                std::memcpy(t2.mutable_data(), tokens.data(),
                            tokens.size() * sizeof(uint32_t));
            py::array_t<uint32_t> o2(static_cast<py::ssize_t>(offsets.size()));
            if (!offsets.empty())
                std::memcpy(o2.mutable_data(), offsets.data(),
                            offsets.size() * sizeof(uint32_t));
            return py::make_tuple(t2, o2);
        }
#endif
        // GPU path: allocate the NumPy array that will be returned and have the
        // D2H land straight in it. One byte emits at most one token, so the
        // total input size is an exact upper bound; once the device reports the
        // real count the same allocation is shrunk in place. This removes two
        // full passes over the token block that the std::vector form pays
        // (device -> pinned -> vector -> NumPy).
        std::vector<const uint8_t*> ptrs;
        std::vector<uint32_t>       lens;
        std::vector<std::string>    storage;   // keeps normalized copies alive
        gbpe::BatchEncodeInput bi = prepare_batch(texts, ptrs, lens, storage);

        uint64_t max_tokens64 = 0;
        for (uint32_t len : lens) max_tokens64 += len;
        if (max_tokens64 > static_cast<uint64_t>(PY_SSIZE_T_MAX)) {
            throw std::length_error(
                "cuTokenize: batch output exceeds Python array capacity");
        }
        py::array_t<uint32_t> tok_arr(static_cast<py::ssize_t>(max_tokens64));
        py::array_t<uint32_t> off_arr(static_cast<py::ssize_t>(texts.size() + 1));
        uint32_t count = 0;
        {
            float k_ms = 0.f, e_ms = 0.f;
            uint32_t* tok_ptr = tok_arr.mutable_data();
            uint32_t* off_ptr = off_arr.mutable_data();
            py::gil_scoped_release release;   // GPU encode touches no Python state
            ctx_->encode_batch_to_host(bi, tok_ptr,
                                       static_cast<uint32_t>(max_tokens64),
                                       &count, off_ptr, &k_ms, &e_ms);
        }
        // Shrink in place rather than reallocating, so the oversized backing
        // buffer is not retained by the returned array.
        tok_arr.resize({static_cast<py::ssize_t>(count)}, false);
        return py::make_tuple(tok_arr, off_arr);
#else
        (void)texts;
        throw std::runtime_error("GPU pretok not built");
#endif
    }

    // Batched encode that leaves the token IDs on the GPU. Same borrow
    // semantics as encode_to_device(): the returned tensor aliases the
    // encoder's output buffer and is invalidated by the next encode of any
    // kind on this Tokenizer.
    py::tuple encode_batch_to_device(const std::vector<std::string>& texts,
                                     py::object self) {
#if GBPE_GPU_PRETOK
        if (texts.empty()) {
            throw std::invalid_argument(
                "encode_batch_to_device() requires at least one document");
        }
        std::vector<const uint8_t*> ptrs;
        std::vector<uint32_t>       lens;
        std::vector<std::string>    storage;
        gbpe::BatchEncodeInput bi = prepare_batch(texts, ptrs, lens, storage);

        std::vector<uint32_t> offsets(texts.size() + 1, 0);

        // Issue on torch's CURRENT cuda stream — see encode_to_device().
        void* stream = reinterpret_cast<void*>(
            py::module_::import("torch").attr("cuda").attr("current_stream")()
                .attr("cuda_stream").cast<uintptr_t>());

        const uint32_t* dptr = nullptr;
        uint32_t count = 0;
        float k_ms = 0.f, e_ms = 0.f;
        {
            py::gil_scoped_release release;
            dptr = ctx_->encode_batch_to_device(bi, &count, offsets.data(),
                                                &k_ms, &e_ms, stream);
        }
        py::array_t<uint32_t> off_arr(static_cast<py::ssize_t>(offsets.size()));
        std::memcpy(off_arr.mutable_data(), offsets.data(),
                    offsets.size() * sizeof(uint32_t));
        return py::make_tuple(wrap_device_tokens_as_torch(dptr, count, self),
                              off_arr);
#else
        (void)texts; (void)self;
        throw std::runtime_error("GPU pretok not built");
#endif
    }

    // Decode token IDs back to text. Wraps the F5 GPU decode kernel (one
    // captured graph, separate from encode). The first call (or any call
    // with more tokens than the current capacity) triggers a re-capture.
    //
    // For SP-family vocabs (Gemma 3) the GPU gathers raw model-token
    // spellings. The serialized Replace + ByteFallback decoder runs below on
    // the host, with the original token IDs retained to preserve boundaries.
    py::bytes decode(const std::vector<uint32_t>& ids) {
        std::vector<uint8_t> out;
        decode_model_bytes(ids, out);
        // SP-family post-decoder: Replace(U+2581 -> " "), ByteFallback, Fuse.
        // Byte-level vocabs bypass this path unchanged.
        if (hv_.byte_fallback) {
            std::vector<uint8_t> decoded;
            gbpe::postprocess_decoded_bytes(
                hv_, ids.data(), static_cast<uint32_t>(ids.size()), out, decoded);
            return py::bytes(reinterpret_cast<const char*>(decoded.data()), decoded.size());
        }
        return py::bytes(reinterpret_cast<const char*>(out.data()), out.size());
    }

    // Convenience: decode and return str. For ByteFallback, malformed
    // contiguous fallback runs become the same U+FFFD sequence as HF.
    py::str decode_str(const std::vector<uint32_t>& ids) {
        std::vector<uint8_t> raw;
        decode_model_bytes(ids, raw);
        std::vector<uint8_t> utf8;
        if (hv_.byte_fallback) {
            gbpe::postprocess_decoded_utf8(
                hv_, ids.data(), static_cast<uint32_t>(ids.size()), raw, utf8);
        } else {
            utf8.swap(raw);
        }
        return py::reinterpret_steal<py::str>(
            PyUnicode_DecodeUTF8(reinterpret_cast<const char*>(utf8.data()),
                                 static_cast<Py_ssize_t>(utf8.size()), "strict"));
    }

    void decode_model_bytes(const std::vector<uint32_t>& ids,
                            std::vector<uint8_t>& out) {
        if (!enable_decode_) {
            throw std::runtime_error(
                "decode() is disabled; construct Tokenizer(..., enable_decode=True)");
        }
        ensure_decode_capacity_for(static_cast<uint32_t>(ids.size()));
        float k_ms = 0.f, e_ms = 0.f;
        py::gil_scoped_release release;
        ctx_->decode(ids.data(), static_cast<uint32_t>(ids.size()),
                     out, &k_ms, &e_ms);
    }

    uint32_t vocab_size() const { return vp_.vocab_size; }
    std::string regex_kind() const { return regex_kind_name(hv_.regex_kind); }
    std::string vocab_path() const { return vocab_path_; }
    bool gemma_gpu_pretok() const { return GBPE_GEMMA_GPU_PRETOK != 0; }

    // ---- Benchmark API: time the captured graph in isolation. ----
    // bench_stage(text): grow caps + capture graph + copy the prompt H2D + sync.
    //   All SETUP — call once, OUTSIDE timing.
    // bench_run(): launch the captured graph and block until done. No D2H; the
    //   token IDs stay on the device. This is the ONLY call to time.
    void bench_stage(const std::string& text) {
#if GBPE_GPU_PRETOK
#if GBPE_HAVE_FAMILY_SP && !GBPE_GEMMA_GPU_PRETOK
        if (sp_family_) throw std::runtime_error("bench API is byte-level only");
#endif
        std::string normalized;
        const std::string& input = normalize_input(text, normalized);
        ensure_capacity_for(static_cast<uint32_t>(input.size()));
        ctx_->stage_input(reinterpret_cast<const uint8_t*>(input.data()),
                          static_cast<uint32_t>(input.size()));
#else
        (void)text; throw std::runtime_error("GPU pretok not built");
#endif
    }
    // Returns the on-device graph time in ms (CUDA-event measured). Timing the
    // graph with device events — not a Python wall-clock around this call —
    // removes the ~50 us host-call/launch floor that otherwise dominates small
    // inputs and bends the low-end of the latency curve.
    double bench_run() {
#if GBPE_GPU_PRETOK
        py::gil_scoped_release release;
        return static_cast<double>(ctx_->launch_graph_sync());
#else
        throw std::runtime_error("GPU pretok not built");
#endif
    }
    // Batch graph-only benchmark setup. `prepare_batch` normalizes each
    // document, reserves batch capacity, and produces stable byte views;
    // stage_batch copies those views before this method returns. All staging
    // and graph-capture work stays outside the timing boundary.
    void bench_stage_batch(const std::vector<std::string>& texts) {
#if GBPE_GPU_PRETOK
        if (texts.empty()) {
            throw std::invalid_argument(
                "bench_stage_batch() requires at least one document");
        }
        std::vector<const uint8_t*> ptrs;
        std::vector<uint32_t> lens;
        std::vector<std::string> storage;
        gbpe::BatchEncodeInput in = prepare_batch(texts, ptrs, lens, storage);
        {
            py::gil_scoped_release release;
            ctx_->stage_batch(in);
        }
#else
        (void)texts;
        throw std::runtime_error("GPU pretok not built");
#endif
    }
    // Named batch counterpart to bench_run(). The preceding call must have
    // been bench_stage_batch(); this returns CUDA-event replay time only.
    double bench_run_batch() {
        return bench_run();
    }
    // Untimed batch-output readback for the exact context and graph replay
    // exercised by bench_stage_batch() / bench_run_batch().  This exists only
    // to make the graph-only benchmark's pre-timing exactness gate complete.
    py::tuple bench_readback_batch() {
#if GBPE_GPU_PRETOK
        std::vector<uint32_t> tokens, offsets;
        {
            py::gil_scoped_release release;
            ctx_->read_staged_batch(tokens, offsets);
        }
        py::array_t<uint32_t> tok_arr(static_cast<py::ssize_t>(tokens.size()));
        if (!tokens.empty()) {
            std::memcpy(tok_arr.mutable_data(), tokens.data(),
                        tokens.size() * sizeof(uint32_t));
        }
        py::array_t<uint32_t> off_arr(static_cast<py::ssize_t>(offsets.size()));
        if (!offsets.empty()) {
            std::memcpy(off_arr.mutable_data(), offsets.data(),
                        offsets.size() * sizeof(uint32_t));
        }
        return py::make_tuple(tok_arr, off_arr);
#else
        throw std::runtime_error("GPU pretok not built");
#endif
    }
    // Decode-side bench API (mirror of encode). bench_stage_decode: grow decode
    // caps + capture decode graph + stage IDs H2D + sync (SETUP, untimed).
    // bench_run_decode: launch the captured decode graph + sync, no D2H (timed).
    void bench_stage_decode(const std::vector<uint32_t>& ids) {
#if GBPE_GPU_PRETOK
        ensure_decode_capacity_for(static_cast<uint32_t>(ids.size()));
        ctx_->stage_decode(ids.data(), static_cast<uint32_t>(ids.size()));
#else
        (void)ids; throw std::runtime_error("GPU pretok not built");
#endif
    }
    // Returns the on-device decode-graph time in ms (CUDA-event measured).
    double bench_run_decode() {
#if GBPE_GPU_PRETOK
        py::gil_scoped_release release;
        return static_cast<double>(ctx_->launch_decode_sync());
#else
        throw std::runtime_error("GPU pretok not built");
#endif
    }

private:
    // One captured graph per power-of-two input-byte class. Every kernel node
    // in the captured graph is sized from the context's capacity, not from the
    // current input: the pre-tokenizer stages launch one thread per CAPACITY
    // byte (grid = cap_input_bytes/256) and the CUB scans sweep cap_input_bytes
    // / cap_pretokens elements with no length early-out. A single grow-only
    // context therefore taxes every later small encode with the largest input
    // ever seen -- measured on an H100 NVL (llama3/code, 16 KiB input): 63 us
    // of GPU work in a fresh context against 1090 us in one grown to 4 MiB,
    // with pretok_k2 at 21.8x, the zipped bucket scan at 21.9x and
    // pretok_gather_cp at 27.2x their fresh cost. The BPE kernels themselves
    // are grid-bounded per SM and did NOT change (1.44 us -> 1.34 us).
    //
    // Keeping one context per size class and replaying the smallest one that
    // fits makes replay cost track the CURRENT input again. Contexts are never
    // evicted: growth is monotone per class, so a mixed-size stream captures
    // each class once and thereafter only replays. The device memory cost is a
    // geometric series -- the ladder below the top class sums to under 1x the
    // top class itself. (Bucket itself is declared at the top of the class.)
    //
    // Sized buffers and a TokenizerCtx for the current cap.
    // `n_docs` is the batch size the next call needs (1 for single-doc encode).
    void ensure_capacity_for(uint32_t text_bytes, uint32_t n_docs = 1) {
        const uint32_t needed_input = next_pow2(static_cast<uint64_t>(text_bytes) + 1024u);
        // Single-doc encodes never need batch capacity; only encode_batch()
        // grows it (and then to at least 16 docs, in powers of two).
        const uint32_t needed_docs = (n_docs <= 1)
            ? 1u
            : next_pow2(std::max<uint32_t>(n_docs, 16u), /*floor*/ 16u);

        // Prefer the bucket for EXACTLY this size class. Selecting merely "the
        // smallest that fits" is not enough: after a 4 MiB encode that bucket
        // fits a 16 KiB input too, and picking it would reproduce the very tax
        // this ladder exists to remove.
        for (Bucket& b : buckets_) {
            if (b.cap_input == needed_input && b.cap_docs >= needed_docs) {
                select_bucket(b);
                return;
            }
        }

        // No exact class yet. Capturing a graph costs ~50 ms, so the ladder is
        // bounded: past kMaxBuckets distinct classes, fall back to the tightest
        // existing fit instead of capturing another. Eight classes cover the
        // whole prompt-length range a server sees. Once the ladder is full the
        // overshoot from reusing a neighbouring class is bounded by how densely
        // the held classes are spaced, not by a single power of two: a sparse
        // ladder can leave an input several classes from its own size.
        constexpr size_t kMaxBuckets = 8;
        if (buckets_.size() >= kMaxBuckets) {
            Bucket* best = nullptr;
            for (Bucket& b : buckets_) {
                if (b.cap_input >= needed_input && b.cap_docs >= needed_docs &&
                    (best == nullptr || b.cap_input < best->cap_input)) {
                    best = &b;
                }
            }
            if (best != nullptr) {
                select_bucket(*best);
                return;
            }
            // Nothing fits (the input is larger than every class held) -- a new
            // top class must be built regardless of the bound. Drop the
            // smallest class to stay within it; it is the cheapest to recapture
            // and the least likely to be the hot one at this point.
            buckets_.erase(buckets_.begin());
            ctx_ = nullptr;
        }

        // Build the bucket for this size class. All caps derive from
        // the CLASS size, not from this particular input, so every later input
        // that maps to the class fits without a recapture. estimate_max_* are
        // monotone in byte count, so a class-sized bound covers every member.
        Bucket nb;
        nb.cap_input  = needed_input;
        nb.cap_output = needed_input;
        // cap_input already includes the +1024 slack, so bound the pre-token
        // counts by the class's full byte capacity.
        nb.cap_short  = gbpe::estimate_max_pretokens(needed_input);
        nb.cap_long   = gbpe::estimate_max_long_pretokens(needed_input);
        // Batch capacity is shared across the ladder: a doc count that forced a
        // rebuild once should not force one again in another size class.
        cap_docs_ = std::max(cap_docs_, needed_docs);
        nb.cap_docs = std::max(cap_docs_, 1u);

        // A new bucket starts at the ladder-wide decode high-water marks, so a
        // decode that already grew them does not force this bucket to rebuild
        // the first time it serves one.
        nb.cap_decode_tokens = cap_decode_tokens_;
        nb.cap_decode_bytes  = cap_decode_bytes_;

        grow_host_staging(nb.cap_short, nb.cap_long);

        nb.ctx = std::make_unique<gbpe::TokenizerCtx>(
            vp_, nb.cap_short, nb.cap_long, nb.cap_input, nb.cap_output,
            nb.cap_decode_tokens, nb.cap_decode_bytes, nb.cap_docs);

        // Insert in increasing cap_input order so the scan above finds the
        // tightest fit first.
        auto pos = std::lower_bound(
            buckets_.begin(), buckets_.end(), nb.cap_input,
            [](const Bucket& b, uint32_t v) { return b.cap_input < v; });
        auto it = buckets_.insert(pos, std::move(nb));
        select_bucket(*it);
    }

    // Point the active-context handle and the mirrored caps at `b`. The caps
    // are what the staging code and the SP front end read, so they must track
    // whichever bucket is live.
    void select_bucket(Bucket& b) {
        ctx_        = b.ctx.get();
        cap_short_  = b.cap_short;
        cap_long_   = b.cap_long;
        cap_input_  = b.cap_input;
        cap_output_ = b.cap_output;
    }

    // Host staging is shared by every bucket, so it is sized to the largest one
    // and only ever grows. Callers index it through the live bucket's caps.
    void grow_host_staging(uint32_t need_short, uint32_t need_long) {
        if (need_short > host_stage_short_) {
            host_stage_short_ = need_short;
            h_short_bytes_.assign(static_cast<size_t>(need_short) * gbpe::MAX_PRETOKEN_LEN_SHORT, 0);
            h_short_lens_ .assign(need_short, 0);
            h_short_orig_ .assign(need_short, 0);
#if GBPE_HAVE_FAMILY_SP && !GBPE_GEMMA_GPU_PRETOK
            if (sp_family_) {
                h_short_ids_.assign(static_cast<size_t>(need_short) * gbpe::MAX_PRETOKEN_LEN_SHORT, 0);
            }
#endif
        }
        if (need_long > host_stage_long_) {
            host_stage_long_ = need_long;
            h_long_bytes_.assign(static_cast<size_t>(need_long) * gbpe::MAX_PRETOKEN_LEN_LONG, 0);
            h_long_lens_ .assign(need_long, 0);
            h_long_orig_ .assign(need_long, 0);
#if GBPE_HAVE_FAMILY_SP && !GBPE_GEMMA_GPU_PRETOK
            if (sp_family_) {
                h_long_ids_.assign(static_cast<size_t>(need_long) * gbpe::MAX_PRETOKEN_LEN_LONG, 0);
            }
#endif
        }
#if GBPE_HAVE_FAMILY_SP && !GBPE_GEMMA_GPU_PRETOK
        if (sp_family_ && h_overflow_orig_.empty()) {
            constexpr uint32_t kOverflowCap = 1024;
            h_overflow_orig_         .assign(kOverflowCap, 0);
            h_overflow_result_ids_   .assign(static_cast<size_t>(kOverflowCap) * gbpe::MAX_PRETOKEN_LEN_LONG, 0);
            h_overflow_result_lens_  .assign(kOverflowCap, 0);
        }
#endif
    }

    // For decode, the ctx needs separately-sized capacities. If a decode call
    // exceeds the current cap (or no decode cap was ever allocated), rebuild
    // the ctx with the new cap.
    void ensure_decode_capacity_for(uint32_t n_tokens) {
        // Upper bound on output bytes per token; matches main.cc's heuristic.
        uint64_t bytes64 = static_cast<uint64_t>(n_tokens) * 256ull + 1024ull;
        if (bytes64 > 0xFFFFFFFFull) bytes64 = 0xFFFFFFFFull;
        uint32_t needed_tok = next_pow2(std::max<uint32_t>(n_tokens, 1024u));
        uint32_t needed_bytes = next_pow2(static_cast<uint32_t>(bytes64));

        // Track the ladder-wide high-water marks so a bucket built later starts
        // already able to decode.
        cap_decode_tokens_ = std::max(cap_decode_tokens_, needed_tok);
        cap_decode_bytes_  = std::max(cap_decode_bytes_,  needed_bytes);

        Bucket* live = active_bucket();
        if (live == nullptr) {
            // Decode before any encode: materialise the floor bucket. It picks
            // up the caps just recorded above.
            ensure_capacity_for(1u);
            live = active_bucket();
        }

        // The decision to rebuild MUST test the live bucket's own decode caps,
        // not the ladder-wide ones. A bucket built before any decode ran
        // carries no decode workspace, and the ladder-wide marks say nothing
        // about it -- comparing against those let an encode switch back to such
        // a bucket and then take the early return here, handing the decode a
        // context with no decode buffers at all.
        if (needed_tok <= live->cap_decode_tokens &&
            needed_bytes <= live->cap_decode_bytes) {
            return;
        }

        // Decode workspaces live in the TokenizerCtx, so growing them needs a
        // rebuilt context. Rebuild only the ACTIVE bucket: the decode graph is
        // separate from the encode graph, so the ladder's other buckets keep
        // serving encodes at their own unchanged encode caps and need not each
        // carry a full-size decode workspace. They pick one up if and when a
        // decode is actually issued against them.
        live->cap_decode_tokens = std::max(live->cap_decode_tokens, needed_tok);
        live->cap_decode_bytes  = std::max(live->cap_decode_bytes,  needed_bytes);
        live->ctx.reset();
        live->ctx = std::make_unique<gbpe::TokenizerCtx>(
            vp_, live->cap_short, live->cap_long, live->cap_input,
            live->cap_output, live->cap_decode_tokens, live->cap_decode_bytes,
            std::max<uint32_t>(live->cap_docs, 1u));
        select_bucket(*live);
    }

    // The bucket whose ctx is currently selected, or nullptr before the first
    // ensure_capacity_for().
    Bucket* active_bucket() {
        for (Bucket& b : buckets_) {
            if (b.ctx.get() == ctx_) return &b;
        }
        return nullptr;
    }

    // View-based overloads for encode_numpy, which borrows Python's buffer
    // instead of copying it. The std::string forms below stay for the other
    // entry points (encode, encode_to_device, the batch paths).
    void encode_impl(std::string_view text, std::vector<uint32_t>& tokens) {
        encode_plain(text, tokens);
    }

    void encode_plain(std::string_view text, std::vector<uint32_t>& tokens) {
        if (cpu_->takes(text.data(), text.size()) && cpu_encode(text, tokens)) {
            last_route_ = "cpu";
            return;
        }
        last_route_ = "gpu";
        std::string normalized;
        const std::string_view input = normalize_input_view(text, normalized);
#if GBPE_HAVE_FAMILY_SP && !GBPE_GEMMA_GPU_PRETOK
        if (sp_family_) {   // the SP host front end takes a std::string
            encode_plain(std::string(input), tokens);
            return;
        }
#endif
        gpu_encode_view(text, input, &tokens, nullptr, 0);
    }

    // As normalize_input, but returns a view: no copy when the input needs no
    // normalization, which is every byte-level vocabulary and all-ASCII text.
    std::string_view normalize_input_view(std::string_view text,
                                          std::string& storage) const {
        if (hv_.normalizer_kind == gbpe::NormalizerKind::None) return text;
        // One pass, and a copy only when normalization changes something
        // (the same helper the C ABI uses).
        return gbpe::normalize_view_for_vocab(hv_, text, storage)
                   ? std::string_view(storage) : text;
    }

    void encode_impl(const std::string& text, std::vector<uint32_t>& tokens) {
        encode_plain(text, tokens);
    }

    void encode_plain(const std::string& text, std::vector<uint32_t>& tokens) {
        if (cpu_->takes(text.data(), text.size()) && cpu_encode(text, tokens)) {
            last_route_ = "cpu";
            return;
        }
        last_route_ = "gpu";
        std::string normalized;
        const std::string& input = normalize_input(text, normalized);
        ensure_capacity_for(static_cast<uint32_t>(input.size()));
        const bool captured = !ctx_->graph_captured();   // this call captures

#if GBPE_HAVE_FAMILY_SP && !GBPE_GEMMA_GPU_PRETOK
        if (sp_family_) {
            gbpe::SPResult pr{};
            pr.short_ids       = h_short_ids_.data();
            pr.short_lens      = h_short_lens_.data();
            pr.short_orig_idx  = h_short_orig_.data();
            pr.cap_short       = cap_short_;
            pr.long_ids        = h_long_ids_.data();
            pr.long_lens       = h_long_lens_.data();
            pr.long_orig_idx   = h_long_orig_.data();
            pr.cap_long        = cap_long_;
            pr.overflow_orig_idx    = h_overflow_orig_.data();
            pr.overflow_result_ids  = h_overflow_result_ids_.data();
            pr.overflow_result_lens = h_overflow_result_lens_.data();
            pr.cap_overflow         = static_cast<uint32_t>(h_overflow_orig_.size());

            gbpe::EncodeInput ei{};
            float k_ms = 0.f, e_ms = 0.f;
            {
                py::gil_scoped_release release;
                sp_pretok_->apply(input, pr);

                ei.short_ids        = h_short_ids_.data();
                ei.short_lens       = h_short_lens_.data();
                ei.short_orig_idx   = h_short_orig_.data();
                ei.n_short          = pr.n_short;
                ei.long_ids         = h_long_ids_.data();
                ei.long_lens        = h_long_lens_.data();
                ei.long_orig_idx    = h_long_orig_.data();
                ei.n_long           = pr.n_long;
                ei.overflow_orig_idx    = h_overflow_orig_.data();
                ei.overflow_result_ids  = h_overflow_result_ids_.data();
                ei.overflow_result_lens = h_overflow_result_lens_.data();
                ei.n_overflow       = pr.n_overflow;
                ei.n_total          = pr.n_total;
                ei.short_bytes      = h_short_bytes_.data();
                ei.long_bytes       = h_long_bytes_.data();
                ctx_->encode(ei, tokens, &k_ms, &e_ms);
            }
            return;
        }
#endif

        (void)captured;
        gpu_encode_view(text, input, &tokens, nullptr, 0);
    }

    // GPU route for normalized byte-level `input`, borrowing its bytes (no
    // copy before the pinned staging). `text` is the caller's raw text, which
    // keys the route feedback. Writes into `out` (room for max_out) when it is
    // given -- the caller's final buffer, one host copy -- else into `tokens`.
    // Returns the token count, UINT32_MAX if `out` is too small.
    uint32_t gpu_encode_view(std::string_view text, std::string_view input,
                             std::vector<uint32_t>* tokens, uint32_t* out,
                             uint32_t max_out) {
        ensure_capacity_for(static_cast<uint32_t>(input.size()));
        const bool captured = !ctx_->graph_captured();   // this call captures
        // Byte-level families and default Gemma builds pre-tokenize inside the
        // captured graph. The host supplies only raw UTF-8 bytes.
        gbpe::EncodeInput ei{};
        ei.raw_text = reinterpret_cast<const uint8_t*>(input.data());
        ei.raw_len  = static_cast<uint32_t>(input.size());
        uint32_t n = 0;
        const auto t0 = std::chrono::steady_clock::now();
        {
            py::gil_scoped_release release;   // GPU encode touches no Python state
            if (out) {
                n = ctx_->encode_into(ei, out, max_out);
            } else {
                float k_ms = 0.f, e_ms = 0.f;
                ctx_->encode(ei, *tokens, &k_ms, &e_ms);
                n = static_cast<uint32_t>(tokens->size());
            }
        }
        // Route feedback, keyed on the caller's text (what takes() sees); a
        // call that captured a new graph class is not a cost sample.
        if (!captured) {
            cpu_->observe_gpu(text.data(), text.size(), static_cast<uint64_t>(
                std::chrono::duration_cast<std::chrono::nanoseconds>(
                    std::chrono::steady_clock::now() - t0).count()));
        }
        return n;
    }

    // Normalize each document (normalization is per-doc semantics — normalizing
    // the concatenation is NOT equivalent), grow the caps for the total byte
    // count AND the document count, and build the BatchEncodeInput. `storage`
    // holds the normalized copies and must outlive the encode call; it is sized
    // up-front so the pointers pushed into `ptrs` are never invalidated.
    gbpe::BatchEncodeInput prepare_batch(const std::vector<std::string>& texts,
                                         std::vector<const uint8_t*>& ptrs,
                                         std::vector<uint32_t>& lens,
                                         std::vector<std::string>& storage) {
#if GBPE_HAVE_FAMILY_SP && !GBPE_GEMMA_GPU_PRETOK
        if (sp_family_) {
            throw std::runtime_error(
                "encode_batch() is byte-level only (no SP/Gemma support)");
        }
#endif
        storage.resize(texts.size());
        ptrs.reserve(texts.size());
        lens.reserve(texts.size());
        uint64_t total_bytes = 0;
        for (size_t i = 0; i < texts.size(); ++i) {
            const std::string& in = normalize_input(texts[i], storage[i]);
            if (&in != &storage[i]) storage[i] = in;   // keep a stable copy
            ptrs.push_back(reinterpret_cast<const uint8_t*>(storage[i].data()));
            lens.push_back(static_cast<uint32_t>(storage[i].size()));
            total_bytes += storage[i].size();
        }
        if (total_bytes > 0xFFFFFFFFull) {
            throw std::length_error("cuTokenize: batch exceeds 2^32 input bytes");
        }
        ensure_capacity_for(static_cast<uint32_t>(total_bytes),
                            static_cast<uint32_t>(texts.size()));

        gbpe::BatchEncodeInput bi{};
        bi.docs   = ptrs.data();
        bi.lens   = lens.data();
        bi.n_docs = static_cast<uint32_t>(texts.size());
        return bi;
    }

    const std::string& normalize_input(const std::string& text,
                                       std::string& storage) const {
        if (hv_.normalizer_kind == gbpe::NormalizerKind::None) {
            return text;
        }
        return gbpe::normalize_view_for_vocab(hv_, text, storage) ? storage : text;
    }

    std::string vocab_path_;
    gbpe::HostVocab hv_;
    gbpe::VocabPack vp_{};
    // Owning ladder of per-size-class contexts, ordered by cap_input.
    std::vector<Bucket> buckets_;
    // Non-owning handle to the bucket serving the current call; every existing
    // `ctx_->` call site is unchanged.
    gbpe::TokenizerCtx* ctx_ = nullptr;
    std::unique_ptr<gbpe::CpuRoute> cpu_;
    const char* last_route_ = "";
    // Below this, releasing and re-acquiring the GIL costs a measurable
    // fraction of a CPU-route encode (a few microseconds of pure C++).
    static constexpr size_t kCpuReleaseGilBytes = 2048;
    // Output staging for encode_numpy, reused across calls so the host path
    // does not regrow it from zero on every encode. Not thread-safe, which
    // matches the rest of this class (a Tokenizer is single-threaded; the GPU
    // path's workspace buffers are shared the same way).
    std::vector<uint32_t> scratch_tokens_;

    // Caps of the bucket currently selected by select_bucket(). Read by the
    // staging code and the SP front end; not authoritative (the bucket is).
    uint32_t cap_short_  = 0;
    uint32_t cap_long_   = 0;
    uint32_t cap_input_  = 0;
    uint32_t cap_output_ = 0;
    // Batch doc capacity is ladder-wide: once encode_batch() has needed N docs,
    // every newly built bucket is sized for N so no later call re-captures.
    uint32_t cap_docs_   = 1;
    // High-water marks of the shared host staging vectors.
    uint32_t host_stage_short_ = 0;
    uint32_t host_stage_long_  = 0;
    uint32_t cap_decode_tokens_ = 0;
    uint32_t cap_decode_bytes_  = 0;
    bool enable_decode_ = false;

    std::vector<uint8_t>  h_short_bytes_;
    std::vector<uint16_t> h_short_lens_;
    std::vector<uint32_t> h_short_orig_;
    std::vector<uint8_t>  h_long_bytes_;
    std::vector<uint16_t> h_long_lens_;
    std::vector<uint32_t> h_long_orig_;

#if GBPE_HAVE_FAMILY_SP
    bool sp_family_ = false;
    gbpe::SPVocabInfo sp_vi_{};
    std::unique_ptr<gbpe::SPPreTokenizer> sp_pretok_;
    std::vector<uint32_t> h_short_ids_;
    std::vector<uint32_t> h_long_ids_;
    std::vector<uint32_t> h_overflow_orig_;
    std::vector<uint32_t> h_overflow_result_ids_;
    std::vector<uint16_t> h_overflow_result_lens_;
#endif
};

}  // namespace

// Thread-safe encoder for many concurrent callers (see dispatcher.h). Every
// call releases the GIL for the whole encode so Python threads really overlap.
static py::array_t<uint32_t> dispatcher_encode(gbpe::Dispatcher& d, py::object text) {
    std::string owned;
    std::string_view view;
    PyObject* obj = text.ptr();
    if (PyUnicode_CheckExact(obj)) {
        Py_ssize_t size = 0;
        const char* data = PyUnicode_AsUTF8AndSize(obj, &size);
        if (data == nullptr) throw py::error_already_set();
        view = std::string_view(data, static_cast<size_t>(size));
    } else if (PyBytes_CheckExact(obj)) {
        view = std::string_view(PyBytes_AS_STRING(obj),
                                static_cast<size_t>(PyBytes_GET_SIZE(obj)));
    } else {
        owned = text.cast<std::string>();
        view = owned;
    }
    std::vector<uint32_t> out;
    {
        py::gil_scoped_release release;
        d.encode(view, out);
    }
    py::array_t<uint32_t> arr(static_cast<py::ssize_t>(out.size()));
    if (!out.empty()) std::memcpy(arr.mutable_data(), out.data(), out.size() * sizeof(uint32_t));
    return arr;
}

PYBIND11_MODULE(_gpu_bpe_tokenizer, m) {
    m.doc() = "gpu-bpe-tokenizer: GPU-accelerated BPE tokenizer Python bindings";

    py::class_<gbpe::Dispatcher>(m, "Dispatcher")
        .def(py::init([](const std::string& vocab_path, uint32_t cpu_workers,
                         uint32_t max_batch_bytes, uint32_t max_batch_docs,
                         const std::string& policy, const std::string& cpu_backend) {
                 gbpe::DispatchPolicy p;
                 if (policy == "adaptive") p = gbpe::DispatchPolicy::Adaptive;
                 else if (policy == "gpu") p = gbpe::DispatchPolicy::GpuOnly;
                 else if (policy == "cpu") p = gbpe::DispatchPolicy::CpuOnly;
                 else throw std::invalid_argument("policy must be adaptive, gpu or cpu");
                 return std::make_unique<gbpe::Dispatcher>(vocab_path, cpu_workers,
                                                           max_batch_bytes, max_batch_docs,
                                                           p, cpu_backend);
             }),
             py::arg("vocab_path"), py::arg("cpu_workers") = 1u,
             py::arg("max_batch_bytes") = 8u << 20, py::arg("max_batch_docs") = 256u,
             py::arg("policy") = std::string("adaptive"),
             py::arg("cpu_backend") = std::string(),
             "Thread-safe encoder for many concurrent requests. Small inputs run "
             "on one of `cpu_workers` CPU engines when one is free; everything "
             "else is combined with concurrent requests into one GPU graph "
             "replay (self-clocked: a batch launches as soon as the previous "
             "one finishes). policy='gpu' batches everything, policy='cpu' "
             "waits for a CPU engine for every input the CPU route accepts.")
        .def("encode_numpy", &dispatcher_encode, py::arg("text"),
             "Encode text to an np.ndarray[uint32]; releases the GIL. Thread-safe.")
        .def("stats", [](const gbpe::Dispatcher& d) {
                 const gbpe::DispatchStats s = d.stats();
                 py::dict r;
                 r["cpu_requests"] = s.cpu_requests;
                 r["gpu_requests"] = s.gpu_requests;
                 r["gpu_batches"]  = s.gpu_batches;
                 r["gpu_large"]    = s.gpu_large;
                 r["gpu_batch_ns"] = s.gpu_batch_ns;
                 return r;
             })
        .def_property_readonly("cpu_backend", &gbpe::Dispatcher::cpu_backend)
        .def_property_readonly("cpu_workers", &gbpe::Dispatcher::cpu_workers);

    py::class_<Tokenizer>(m, "Tokenizer")
        .def(py::init<const std::string&, uint32_t, bool, std::optional<uint32_t>, const std::string&>(),
             py::arg("vocab_path"),
             py::arg("max_input_chars") = 4096u,
             py::arg("enable_decode")   = true,
             // None: the measured per-family crossovers (cpu_thresholds.h,
             // CPU/GPU crossover sweep): 16 KiB for mostly-ASCII text
             // and 4 KiB when over 1/32 of the bytes are non-ASCII.
             py::arg("host_max_bytes")  = py::none(),
             py::arg("cpu_backend")     = std::string(),
             "Load an HF tokenizer.json and prepare a GPU encoder.\n\n"
             "max_input_chars pre-sizes the first captured graph at "
             "construction. Larger inputs get their own power-of-two size "
             "class, each captured once on first use (~50 ms) and then kept. "
             "A graph's per-call cost scales with its own capacity, but an "
             "encode always replays the smallest class that fits it, so a "
             "single large prompt does not slow the small prompts after it.\n\n"
             "Set enable_decode=False to skip decode workspaces (saves ~1 MB).\n\n"
             "cpu_backend picks the engine for inputs at or below "
             "host_max_bytes: 'auto' (default; gigatoken when linked, else the "
             "host encoder), 'gigatoken', 'host' or 'off'. '' reads "
             "GTOK_CPU_BACKEND.")
        .def("encode", &Tokenizer::encode, py::arg("text"),
             "Encode text to a list[int] of token IDs.")
        .def("encode_numpy", &Tokenizer::encode_numpy, py::arg("text"),
             "Encode text to an np.ndarray[uint32] of token IDs.")
        .def("encode_to_device",
             [](py::object self, const std::string& text) {
                 return self.cast<Tokenizer&>().encode_to_device(text, self);
             },
             py::arg("text"),
             // The returned tensor aliases this Tokenizer's GPU buffer; it holds
             // a strong reference to the Tokenizer through its CAI carrier, so
             // the buffer outlives any Python-side drop of the Tokenizer.
             "Encode text and return the token IDs as an int32 torch.Tensor that "
             "shares GPU memory with the encoder — no host copy (requires a "
             "GPU-pretokenizer build). The encode runs on torch's current "
             "CUDA stream. The tensor borrows the encoder's output buffer and is "
             "overwritten by the next encode call on this Tokenizer — call "
             ".clone() if you need to keep it across calls.")
        .def("encode_batch", &Tokenizer::encode_batch, py::arg("texts"),
             "Encode a list[str] of documents in ONE GPU graph replay.\n\n"
             "Returns (tokens, offsets): tokens is an np.ndarray[uint32] of all "
             "documents' IDs concatenated, offsets is an np.ndarray[uint32] of "
             "len(texts)+1 running sums — tokens[offsets[i]:offsets[i+1]] is "
             "document i. The result is IDENTICAL to encoding each string "
             "separately and concatenating. An empty list returns "
             "(empty array, array([0])) without touching the GPU. "
             "Byte-level vocabs only (no SP/Gemma).")
        .def("encode_batch_to_device",
             [](py::object self, const std::vector<std::string>& texts) {
                 return self.cast<Tokenizer&>().encode_batch_to_device(texts, self);
             },
             py::arg("texts"),
             // py::keep_alive<0,1> cannot be used here (the return is a tuple,
             // which is not weak-referenceable). The tensor instead keeps the
             // Tokenizer alive through the strong reference torch holds to its
             // __cuda_array_interface__ carrier.
             "Batched encode that leaves the IDs on the GPU. Returns "
             "(torch.Tensor int32, np.ndarray[uint32] offsets[len(texts)+1]) "
             "with the same per-document slicing contract as encode_batch(). "
             "The encode runs on torch's current CUDA stream. The tensor "
             "borrows the encoder's output buffer and is invalidated by the "
             "next encode of any kind on this Tokenizer — call .clone() if you "
             "need to keep it. The tensor keeps the Tokenizer itself alive, so "
             "dropping your last reference to the Tokenizer is safe. "
             "Byte-level vocabs only; raises on an empty list.")
        .def("decode", &Tokenizer::decode_str, py::arg("ids"),
             "Decode a sequence of token IDs to a str. For SP-family vocabs "
             "(Gemma 3), the ▁→space and ByteFallback post-decoder is applied "
             "to match HF tokenizers' output exactly.")
        .def("decode_bytes", &Tokenizer::decode, py::arg("ids"),
             "Decode a sequence of token IDs to raw bytes (no UTF-8 decode). "
             "For SP-family vocabs the ▁→space and ByteFallback substitutions "
             "are still applied.")
        .def("bench_stage", &Tokenizer::bench_stage, py::arg("text"),
             "Benchmark setup: capture graph + copy prompt H2D + sync. "
             "Call once OUTSIDE timing.")
        .def("bench_run", &Tokenizer::bench_run,
             "Benchmark: launch the captured graph and block until done. "
             "No device->host copy. Time THIS call.")
        .def("bench_stage_batch", &Tokenizer::bench_stage_batch, py::arg("texts"),
             "Batch benchmark setup: normalize documents, capture graph if needed, "
             "copy concatenated text and document offsets H2D, and synchronize. "
             "Call outside timing.")
        .def("bench_run_batch", &Tokenizer::bench_run_batch,
             "Batch benchmark: replay the batch staged by bench_stage_batch() and "
             "return CUDA-event graph time in milliseconds. No token D2H.")
        .def("bench_readback_batch", &Tokenizer::bench_readback_batch,
             "Untimed validation helper: read token IDs and document offsets from "
             "the context most recently exercised by bench_stage_batch() and "
             "bench_run_batch(). Raises unless that call order was used.")
        .def("bench_stage_decode", &Tokenizer::bench_stage_decode, py::arg("ids"),
             "Decode benchmark setup: capture decode graph + stage IDs H2D + "
             "sync. Call once OUTSIDE timing.")
        .def("bench_run_decode", &Tokenizer::bench_run_decode,
             "Decode benchmark: launch the captured decode graph + sync, no "
             "device->host copy. Time THIS call.")
        .def("__call__", &Tokenizer::encode, py::arg("text"))
        .def_property("host_max_bytes", &Tokenizer::host_max_bytes,
                      &Tokenizer::set_host_max_bytes,
                      "Inputs at or below this many raw bytes take the CPU route "
                      "(exactly the same token IDs; avoids the graph-replay floor); "
                      "inputs with over 1/32 non-ASCII bytes use "
                      "host_max_bytes_multilingual instead. Setting it sets both. "
                      "0 disables; forced to 0 when no CPU engine supports the family.")
        .def_property_readonly("host_max_bytes_multilingual",
             &Tokenizer::host_max_bytes_multilingual,
             "CPU-route limit for inputs with over 1/32 non-ASCII bytes.")
        .def_property_readonly("cpu_backend", &Tokenizer::cpu_backend,
             "Engine serving the CPU route: 'gigatoken', 'host' or 'off'.")
        .def_property_readonly("cpu_cost_model", &Tokenizer::cpu_cost_model,
             "Router state (diagnostic): [cpu ps/byte ascii, multilingual] + GPU ns "
             "per (class, size class 8-16K, 16-32K, ...); 0 = not observed.")
        .def_property_readonly("last_route", &Tokenizer::last_route,
             "Route the last single-document encode took: 'cpu' or 'gpu'.")
        .def_property_readonly("cpu_backend_note", &Tokenizer::cpu_backend_note,
             "Why cpu_backend was chosen (or another engine rejected).")
        .def_property_readonly("graph_capacities", &Tokenizer::graph_capacities,
             "Input-byte capacities of the captured size-class graphs held by "
             "this Tokenizer, smallest first. Diagnostic.")
        .def_property_readonly("active_capacity", &Tokenizer::active_capacity,
             "Input-byte capacity of the graph the last GPU encode replayed.")
        .def_property_readonly("vocab_size",  &Tokenizer::vocab_size)
        .def_property_readonly("regex_kind",  &Tokenizer::regex_kind)
        .def_property_readonly("vocab_path",  &Tokenizer::vocab_path)
        .def_property_readonly("gemma_gpu_pretok", &Tokenizer::gemma_gpu_pretok)
        .def("__repr__", [](const Tokenizer& t) {
            return "<gpu_bpe_tokenizer.Tokenizer vocab_size=" +
                   std::to_string(t.vocab_size()) +
                   " regex=" + t.regex_kind() + ">";
        });

#if defined(GBPE_HOST_STATS) && GBPE_HOST_STATS
    // Present only in a stats-instrumented build, so a benchmark script that
    // reads these cannot silently succeed against the shipped extension.
    m.def("host_cache_stats", []() {
        const gbpe::HostCacheStats s = gbpe::host_cache_stats();
        py::dict d;
        d["short_lookups"] = s.short_lookups;
        d["short_hits"] = s.short_hits;
        d["short_evictions"] = s.short_evictions;
        d["wide_lookups"] = s.wide_lookups;
        d["wide_hits"] = s.wide_hits;
        d["wide_evictions"] = s.wide_evictions;
        d["uncacheable"] = s.uncacheable;
        return d;
    }, "Pre-token cache counters (stats build only).");
    m.def("host_cache_stats_reset", &gbpe::host_cache_stats_reset,
          "Zero the pre-token cache counters (stats build only).");
#endif
    m.attr("has_host_stats") =
#if defined(GBPE_HOST_STATS) && GBPE_HOST_STATS
        true;
#else
        false;
#endif
}
