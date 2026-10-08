// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 the cuTokenize contributors
#include "cpu_route.h"
#include "cpu_thresholds.h"

#include <chrono>
#if defined(__x86_64__)
#include <immintrin.h>
#endif
#include <cstdlib>
#include <cstring>
#include <stdexcept>

#if GBPE_HAVE_GIGATOKEN
extern "C" {
void* gtokcpu_create(const char* tokenizer_json, uint64_t cache_bytes);
int32_t gtokcpu_encode(void* h, const uint8_t* text, size_t len,
                       const uint32_t** ids, size_t* n);
void gtokcpu_destroy(void* h);
const char* gtokcpu_last_error();
}
#endif

namespace gbpe {

namespace {

// Pre-token cache budget per gigatoken handle. The crate default (512 MiB per
// worker) suits a batch tokenizer that owns the machine, not a library that a
// serving frontend instantiates per context; GTOK_CPU_CACHE_BYTES overrides.
constexpr uint64_t kDefaultGigaCacheBytes = 64ull << 20;

const char* env_or_null(const char* name) {
    const char* v = std::getenv(name);
    return (v != nullptr && *v != '\0') ? v : nullptr;
}

// Inputs the two engines must agree on before gigatoken is trusted for this
// tokenizer.json: every pre-tokenizer rule family the regexes distinguish
// (contractions, digit runs, whitespace runs before letters and newlines,
// CR/LF, punctuation runs, non-Latin scripts), plus the vocabulary's own
// added tokens embedded in text.
std::vector<std::string> canary_inputs(const HostVocab& hv) {
    std::vector<std::string> v = {
        "Hello, world! It's 2026 -- don't we'll they've I'M 1234567 3.14159.",
        "def f(x):\n    return x  +  1\n\n\n\tif a<b: pass  \r\n  end",
        "  leading   and trailing spaces   \n\n  \n",
        "\xe4\xbd\xa0\xe5\xa5\xbd\xef\xbc\x8c\xe4\xb8\x96\xe7\x95\x8c "            // 你好，世界
        "\xd0\x9f\xd1\x80\xd0\xb8\xd0\xb2\xd0\xb5\xd1\x82 "                        // Привет
        "\xd9\x85\xd8\xb1\xd8\xad\xd8\xa8\xd8\xa7 "                                // مرحبا
        "\xe0\xa4\xa8\xe0\xa4\xae\xe0\xa4\xb8\xe0\xa5\x8d\xe0\xa4\xa4\xe0\xa5\x87 "  // नमस्ते
        "caf\xc3\xa9 na\xc3\xafve \xf0\x9f\x98\x80\xf0\x9f\x98\x80!!",               // café naïve 😀😀
        "x}\nfoo a\r\nb a\t  b ...!!!??? ---=== 000111222333",
    };
    std::string with_added = "prefix ";
    size_t used = 0;
    for (const auto& t : hv.added_tokens) {
        if (t.content.empty() || used == 3) continue;
        with_added += t.content + " mid " + t.content + t.content + "\n";
        ++used;
    }
    if (used) v.push_back(with_added);
    return v;
}

}  // namespace

const char* cpu_backend_name(CpuBackend b) {
    switch (b) {
        case CpuBackend::Gigatoken: return "gigatoken";
        case CpuBackend::Host:      return "host";
        default:                    return "off";
    }
}

bool cpu_gigatoken_linked() {
#if GBPE_HAVE_GIGATOKEN
    return true;
#else
    return false;
#endif
}

CpuRoute::CpuRoute(const HostVocab& hv, const std::string& vocab_path,
                   const std::string& requested_arg, uint32_t max_bytes)
    : hv_(hv)
{
    std::string requested = requested_arg;
    if (requested.empty()) {
        const char* e = env_or_null("GTOK_CPU_BACKEND");
        requested = e ? e : "auto";
    }
    if (requested != "auto" && requested != "gigatoken" &&
        requested != "host" && requested != "off") {
        throw std::invalid_argument(
            "cpu backend must be one of auto, gigatoken, host, off (got '" +
            requested + "')");
    }

    host_ = std::make_unique<HostEncoder>(hv);
    const bool host_ok = host_->supported();

    if (requested == "auto" || requested == "gigatoken") {
#if GBPE_HAVE_GIGATOKEN
        uint64_t cache = kDefaultGigaCacheBytes;
        if (const char* c = env_or_null("GTOK_CPU_CACHE_BYTES")) {
            cache = std::strtoull(c, nullptr, 10);
        }
        giga_ = gtokcpu_create(vocab_path.c_str(), cache);
        if (giga_ == nullptr) {
            note_ = std::string("gigatoken unavailable: ") + gtokcpu_last_error();
        } else if (host_ok) {
            // Canary: gigatoken must reproduce HostEncoder -- itself gated
            // bit-exact against the reference -- on every probe.
            std::vector<uint32_t> a, b;
            std::string norm;
            for (const std::string& s : canary_inputs(hv)) {
                const uint32_t* ids = nullptr;
                size_t n = 0;
                if (gtokcpu_encode(giga_, reinterpret_cast<const uint8_t*>(s.data()),
                                   s.size(), &ids, &n) != 0) {
                    note_ = std::string("gigatoken canary failed: ") + gtokcpu_last_error();
                    break;
                }
                a.assign(ids, ids + n);
                std::string_view in = s;
                if (normalize_view_for_vocab(hv, in, norm)) in = norm;
                host_->encode(reinterpret_cast<const uint8_t*>(in.data()), in.size(), b);
                if (a != b) {
                    note_ = "gigatoken disagrees with the host encoder on the canary";
                    break;
                }
            }
            if (note_.empty()) {
                backend_ = CpuBackend::Gigatoken;
                note_ = "gigatoken (canary matched the host encoder)";
            } else {
                gtokcpu_destroy(giga_);
                giga_ = nullptr;
            }
        } else {
            // No host reference for this family (DeepSeek-V3): trust the
            // loader's scheme detection; tests gate it against HF.
            backend_ = CpuBackend::Gigatoken;
            note_ = "gigatoken (no host encoder for this family; gated by tests)";
        }
#else
        note_ = "gigatoken not linked into this build";
#endif
        if (backend_ == CpuBackend::Off && requested == "gigatoken") {
            throw std::runtime_error("GTOK_CPU_BACKEND=gigatoken requested but " + note_);
        }
    }
    if (backend_ == CpuBackend::Off && (requested == "auto" || requested == "host")) {
        if (host_ok) {
            backend_ = CpuBackend::Host;
            note_ = note_.empty() ? "host encoder" : note_ + "; using the host encoder";
        } else if (requested == "host") {
            note_ = "host encoder does not support this tokenizer family";
        }
    }
    if (backend_ != CpuBackend::Host) host_.reset();

    if (const char* m = env_or_null("GTOK_CPU_MAX_BYTES")) {
        max_bytes = static_cast<uint32_t>(std::strtoul(m, nullptr, 10));
    }
    if (max_bytes == kCpuMaxBytesAuto) {
        const CpuThresholds t = cpu_default_thresholds(hv.regex_kind);
        set_limits(t.ascii, t.multilingual);
        const char* a = env_or_null("GTOK_CPU_ADAPT");
        adaptive_ = !(a && std::strcmp(a, "0") == 0);
    } else {
        set_max_bytes(max_bytes);
    }
}

namespace {

// lead = bytes >= 0xC0 (start of a multi-byte character); punct = E2 80 xx /
// E2 81 xx (General Punctuation U+2000-U+207F).
size_t nonpunct_leads_scalar(const unsigned char* b, size_t n) {
    size_t leads = 0, punct = 0;
    for (size_t i = 0; i < n; ++i) leads += b[i] >= 0xC0;
    for (size_t i = 0; i + 1 < n; ++i) punct += (b[i] == 0xE2) & ((b[i + 1] & 0xFE) == 0x80);
    return leads - punct;
}

#if defined(__x86_64__)
__attribute__((target("avx2,popcnt")))
size_t nonpunct_leads_avx2(const unsigned char* b, size_t n) {
    const __m256i c0 = _mm256_set1_epi8(static_cast<char>(0xC0));
    const __m256i e2 = _mm256_set1_epi8(static_cast<char>(0xE2));
    const __m256i fe = _mm256_set1_epi8(static_cast<char>(0xFE));
    const __m256i x80 = _mm256_set1_epi8(static_cast<char>(0x80));
    size_t leads = 0, punct = 0, i = 0;
    for (; i + 33 <= n; i += 32) {
        const __m256i v = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b + i));
        const __m256i w = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b + i + 1));
        // (v & 0xC0) == 0xC0  <=>  v >= 0xC0
        const __m256i lead = _mm256_cmpeq_epi8(_mm256_and_si256(v, c0), c0);
        const __m256i p = _mm256_and_si256(_mm256_cmpeq_epi8(v, e2),
                                           _mm256_cmpeq_epi8(_mm256_and_si256(w, fe), x80));
        leads += static_cast<size_t>(__builtin_popcount(
            static_cast<unsigned>(_mm256_movemask_epi8(lead))));
        punct += static_cast<size_t>(__builtin_popcount(
            static_cast<unsigned>(_mm256_movemask_epi8(p))));
    }
    // The vector loop counted every pair starting before i (its shifted load
    // reaches b[i]); the scalar tail counts the rest.
    return leads - punct + nonpunct_leads_scalar(b + i, n - i);
}
#endif

}  // namespace

namespace {
// takes() classifies the text and encode()/observe_gpu() need the same answer
// for the same bytes right after, on the same thread: remember it once.
thread_local const char* t_cls_p = nullptr;
thread_local size_t t_cls_n = 0;
thread_local bool t_cls = false;
}  // namespace

bool CpuRoute::multilingual(const char* p, size_t n) {
    const unsigned char* b = reinterpret_cast<const unsigned char*>(p);
#if defined(__x86_64__)
    static const bool avx2 = __builtin_cpu_supports("avx2");
    const size_t leads = avx2 ? nonpunct_leads_avx2(b, n) : nonpunct_leads_scalar(b, n);
#else
    const size_t leads = nonpunct_leads_scalar(b, n);
#endif
    t_cls_p = p;
    t_cls_n = n;
    t_cls = leads * 64 > n;
    return t_cls;
}

bool CpuRoute::multilingual_again(const char* p, size_t n) {
    const bool hit = p == t_cls_p && n == t_cls_n;
    const bool r = hit ? t_cls : multilingual(p, n);
    t_cls_p = nullptr;                 // one use: callers reuse buffers
    return r;
}

void CpuRoute::observe_gpu(const char* p, size_t n, uint64_t ns) {
    if (!adaptive_ || n <= max_ml_ || n > band_top()) return;
    ewma(gpu_ns_[multilingual_again(p, n) ? 1 : 0][size_class(n)], ns);
}

bool CpuRoute::takes_all(const char* const* ps, const size_t* ns, size_t count) const {
    // Batches use the static rule on their total.
    size_t total = 0;
    for (size_t i = 0; i < count; ++i) total += ns[i];
    if (total <= max_ml_) return max_ascii_ != 0u;
    if (total > max_ascii_) return false;
    for (size_t i = 0; i < count; ++i)
        if (multilingual(ps[i], ns[i])) return false;
    return true;
}

CpuRoute::~CpuRoute() {
#if GBPE_HAVE_GIGATOKEN
    if (giga_) gtokcpu_destroy(giga_);
#endif
}

void CpuRoute::encode(std::string_view raw, std::vector<uint32_t>& out) {
    std::lock_guard<std::mutex> lock(mu_);
    if (!adaptive_ || raw.size() < 1024 || raw.size() > band_top()) {
        encode_locked(raw, out);
        return;
    }
    const auto t0 = std::chrono::steady_clock::now();
    encode_locked(raw, out);
    const uint64_t ns = static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now() - t0).count());
    const int cls = multilingual_again(raw.data(), raw.size()) ? 1 : 0;
    const uint32_t k = cpu_samples_[cls].fetch_add(1, std::memory_order_relaxed);
    cpu_ring_[cls][k % kCpuWindow].store(ns * 1000 / raw.size(), std::memory_order_relaxed);
}

void CpuRoute::encode_locked(std::string_view raw, std::vector<uint32_t>& out) {
#if GBPE_HAVE_GIGATOKEN
    if (backend_ == CpuBackend::Gigatoken) {
        const uint32_t* ids = nullptr;
        size_t n = 0;
        if (gtokcpu_encode(giga_, reinterpret_cast<const uint8_t*>(raw.data()),
                           raw.size(), &ids, &n) != 0) {
            throw std::runtime_error(std::string("gigatoken encode failed: ") +
                                     gtokcpu_last_error());
        }
        out.assign(ids, ids + n);
        return;
    }
#endif
    if (backend_ == CpuBackend::Host) {
        std::string_view in = raw;
        if (normalize_view_for_vocab(hv_, raw, norm_)) in = norm_;
        host_->encode(reinterpret_cast<const uint8_t*>(in.data()), in.size(), out);
        return;
    }
    throw std::logic_error("CpuRoute::encode with the CPU route off");
}

}  // namespace gbpe
