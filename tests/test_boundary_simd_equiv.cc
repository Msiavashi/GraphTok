// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 the cuTokenize contributors
//
// Does the AVX2 boundary fast path agree with the scalar one, lane for lane?
// Random class/byte arrays drawn from an alphabet that exercises every branch
// (letters, space, CR, LF, digit, punctuation, apostrophe, CLS_ADDED), plus
// the real corpus. Any disagreement on a position the SIMD version claims to
// resolve, or any position it resolves differently, is a bug.
#include "pretok_boundary.h"
#include "pretok_boundary_simd.h"
#include "gbpe_class_table.h"
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <random>
#include <sstream>
#include <string>
#include <vector>
using namespace gbpe;

static long check(const std::vector<uint8_t>& cls, const std::vector<uint8_t>& b0,
                  const char* label, bool verbose) {
    const uint32_t n = (uint32_t)cls.size();
    if (n < 40) return 0;
    long bad = 0, resolved_simd = 0, resolved_scalar = 0;
    for (uint32_t base = 3; base + 33 <= n; base += 32) {
        uint32_t res = 0, bnd = 0;
        llamaqwen_fast_x32(cls.data(), b0.data(), base, &res, &bnd);
        for (int j = 0; j < 32; ++j) {
            const uint32_t c = base + j;
            const FastBoundary want = llamaqwen_fast(cls.data(), b0.data(), n, nullptr, c);
            const bool got_res = (res >> j) & 1u;
            const bool got_bnd = (bnd >> j) & 1u;
            resolved_simd += got_res;
            resolved_scalar += (want != kFastUnknown);
            if (want == kFastUnknown) {
                if (got_res && ++bad <= 8)
                    printf("  [%s] c=%u: SIMD resolved (%d) but scalar says unknown\n",
                           label, c, (int)got_bnd);
            } else {
                const bool want_b = (want == kFastYes);
                if (!got_res) {
                    if (++bad <= 8)
                        printf("  [%s] c=%u: SIMD unresolved, scalar says %d\n",
                               label, c, (int)want_b);
                } else if (got_bnd != want_b) {
                    if (++bad <= 8)
                        printf("  [%s] c=%u: SIMD %d, scalar %d\n",
                               label, c, (int)got_bnd, (int)want_b);
                }
            }
        }
    }
    if (verbose)
        printf("%-14s n=%-8u resolved simd=%ld scalar=%ld  mismatches=%ld\n",
               label, n, resolved_simd, resolved_scalar, bad);
    return bad;
}

int main(int argc, char** argv) {
    const char* corpus_dir = argc > 1 ? argv[1] : nullptr;
    long bad = 0;
#if !GBPE_HAVE_AVX2_BOUNDARY
    printf("AVX2 boundary path not compiled in\n");
    return 2;
#endif
    // 1. Random arrays over an adversarial alphabet.
    std::mt19937 rng(7);
    const char* ALPHA = "ab cd\r\n1.'sXY";
    const int A = (int)strlen(ALPHA);
    for (int trial = 0; trial < 3000; ++trial) {
        const uint32_t n = 64 + (rng() % 300);
        std::vector<uint8_t> cls(n), b0(n);
        for (uint32_t i = 0; i < n; ++i) {
            unsigned char ch = (unsigned char)ALPHA[rng() % A];
            b0[i] = ch;
            cls[i] = (ch == ' ' || ch == '\r' || ch == '\n') ? CLS_S
                   : (ch >= '0' && ch <= '9') ? CLS_N
                   : ((ch >= 'a' && ch <= 'z') || (ch >= 'A' && ch <= 'Z')) ? CLS_L
                   : CLS_O;
        }
        // Sprinkle CLS_ADDED, which the guard must reject.
        for (uint32_t i = 0; i < n; ++i) if ((rng() % 40) == 0) cls[i] = CLS_ADDED;
        bad += check(cls, b0, "random", false);
    }
    printf("random trials: %ld mismatches\n", bad);

    // 2. The real corpora, classified the way the host encoder does.
    // Corpora are optional: the random trials above are the real gate, and
    // they need no data files. Pass a directory to also scan real text.
    const char* names[] = {"corpus_1m_tokens.txt", "corpus_code_1m_tokens.txt",
                           "corpus_chat_multilingual_1m_tokens.txt"};
    for (const char* name : names) {
        if (!corpus_dir) break;
        const std::string f = std::string(corpus_dir) + "/" + name;
        std::ifstream in(f, std::ios::binary);
        if (!in) { printf("%-14s (not found, skipped)\n", name); continue; }
        std::stringstream ss; ss << in.rdbuf();
        std::string text = ss.str();
        if (text.size() > 400000) text.resize(400000);
        std::vector<uint8_t> cls, b0;
        const uint8_t* p = (const uint8_t*)text.data();
        size_t i = 0, N = text.size();
        while (i < N) {
            int clen = 1;
            uint32_t cp = decode_cp(p + i, (uint32_t)(N - i), &clen);
            b0.push_back(p[i]);
            cls.push_back(gbpe_classify(cp));
            i += clen;
        }
        bad += check(cls, b0, name, true);
    }
    // 3. The vector ASCII classifier must equal gbpe_classify on every ASCII
    // byte, in every lane position.
    {
        alignas(32) uint8_t in[32], out[32];
        long cbad = 0;
        for (int base = 0; base < 128; base += 32) {
            for (int j = 0; j < 32; ++j) in[j] = (uint8_t)(base + j);
            classify_ascii_32(in, out);
            for (int j = 0; j < 32; ++j) {
                const uint8_t want = gbpe_classify((uint32_t)in[j]);
                if (out[j] != want) {
                    if (++cbad <= 8)
                        printf("  CLASSIFY MISMATCH 0x%02X: simd=%u table=%u\n",
                               in[j], out[j], want);
                }
            }
        }
        printf("%-14s ASCII classify mismatches=%ld\n", "classify", cbad);
        bad += cbad;
    }

    printf(bad ? "SIMD EQUIVALENCE FAILED\n" : "SIMD EQUIVALENCE OK\n");
    return bad ? 1 : 0;
}
