// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 the GraphTok contributors
//
// Vectorised form of llamaqwen_fast() for the host encoder.
//
// The boundary scan is the single largest host cost (8.7 ns/byte of 19.8 at a
// 4 KiB window), and llamaqwen_fast() already resolves ~88% of prose positions
// from a 5-wide window over two byte arrays. Those are exactly the semantics
// AVX2 evaluates well: load 32 consecutive positions, compute every condition
// as a comparison mask, and emit two bitmaps -- one saying "this position is
// resolved", one giving the answer.
//
// This file computes the SAME predicate as llamaqwen_fast(); positions it
// leaves unresolved fall through to the scalar cascade exactly as before. The
// scalar version stays the reference, and a test compares the two over the
// corpora.
#pragma once
#include <cstdint>
#include "pretok_boundary.h"

// Compiled on any x86-64 with a GCC/Clang-compatible compiler: the kernel
// carries __attribute__((target("avx2"))) so the translation unit need not be
// built with -mavx2, and the caller checks __builtin_cpu_supports("avx2") once
// at construction. That keeps the shipped binary runnable on pre-AVX2 CPUs.
#if defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))
#  define GBPE_HAVE_AVX2_BOUNDARY 1
#  include <immintrin.h>
#  define GBPE_AVX2_TARGET __attribute__((target("avx2")))
#else
#  define GBPE_HAVE_AVX2_BOUNDARY 0
#  define GBPE_AVX2_TARGET
#endif

namespace gbpe {

#if GBPE_HAVE_AVX2_BOUNDARY

// Resolve positions [base, base+32) of a single document.
//
// Writes into *resolved a bit per position (1 = this lane answered) and into
// *is_boundary the answer for those lanes. Requires base >= 3 and
// base + 32 + 1 <= n so every lane's window c-3..c+1 is in range; the caller
// handles the head and tail scalar-wise.
//
// Preconditions match llamaqwen_fast's guarded body: single document (the
// caller checks cp_doc_start, which is null for the host's single-document
// encode), window in range, no CLS_ADDED and no apostrophe in the window.
// Lanes failing those tests are simply left unresolved.
GBPE_AVX2_TARGET static inline void llamaqwen_fast_x32(
    const uint8_t* __restrict cls, const uint8_t* __restrict cp_byte0,
    uint32_t base, uint32_t* __restrict resolved, uint32_t* __restrict is_boundary)
{
    const __m256i b_0  = _mm256_loadu_si256((const __m256i*)(cp_byte0 + base));
    const __m256i b_m1 = _mm256_loadu_si256((const __m256i*)(cp_byte0 + base - 1));
    const __m256i b_m2 = _mm256_loadu_si256((const __m256i*)(cp_byte0 + base - 2));
    const __m256i b_m3 = _mm256_loadu_si256((const __m256i*)(cp_byte0 + base - 3));
    const __m256i b_p1 = _mm256_loadu_si256((const __m256i*)(cp_byte0 + base + 1));
    const __m256i k_0  = _mm256_loadu_si256((const __m256i*)(cls + base));
    const __m256i k_m1 = _mm256_loadu_si256((const __m256i*)(cls + base - 1));
    const __m256i k_m2 = _mm256_loadu_si256((const __m256i*)(cls + base - 2));
    const __m256i k_p1 = _mm256_loadu_si256((const __m256i*)(cls + base + 1));

    const __m256i apos  = _mm256_set1_epi8(0x27);
    const __m256i space = _mm256_set1_epi8(0x20);
    const __m256i added = _mm256_set1_epi8((char)CLS_ADDED);
    const __m256i clsL  = _mm256_set1_epi8((char)CLS_L);
    const __m256i clsS  = _mm256_set1_epi8((char)CLS_S);

    // Window is usable: no apostrophe at c-3..c+1, no CLS_ADDED at c-2..c+1.
    __m256i bad = _mm256_cmpeq_epi8(b_0, apos);
    bad = _mm256_or_si256(bad, _mm256_cmpeq_epi8(b_m1, apos));
    bad = _mm256_or_si256(bad, _mm256_cmpeq_epi8(b_m2, apos));
    bad = _mm256_or_si256(bad, _mm256_cmpeq_epi8(b_m3, apos));
    bad = _mm256_or_si256(bad, _mm256_cmpeq_epi8(b_p1, apos));
    bad = _mm256_or_si256(bad, _mm256_cmpeq_epi8(k_0,  added));
    bad = _mm256_or_si256(bad, _mm256_cmpeq_epi8(k_m1, added));
    bad = _mm256_or_si256(bad, _mm256_cmpeq_epi8(k_m2, added));
    bad = _mm256_or_si256(bad, _mm256_cmpeq_epi8(k_p1, added));
    const uint32_t bad_m = (uint32_t)_mm256_movemask_epi8(bad);

    const __m256i clsN  = _mm256_set1_epi8((char)CLS_N);
    const __m256i clsO  = _mm256_set1_epi8((char)CLS_O);
    const __m256i cur_L  = _mm256_cmpeq_epi8(k_0,  clsL);
    const __m256i cur_S  = _mm256_cmpeq_epi8(k_0,  clsS);
    const __m256i cur_O  = _mm256_cmpeq_epi8(k_0,  clsO);
    const __m256i prev_L = _mm256_cmpeq_epi8(k_m1, clsL);
    const __m256i prev_S = _mm256_cmpeq_epi8(k_m1, clsS);
    const __m256i prev_N = _mm256_cmpeq_epi8(k_m1, clsN);
    const __m256i prev_O = _mm256_cmpeq_epi8(k_m1, clsO);
    const __m256i mm2_O  = _mm256_cmpeq_epi8(k_m2, clsO);
    const __m256i next_L = _mm256_cmpeq_epi8(k_p1, clsL);
    const __m256i b0_sp  = _mm256_cmpeq_epi8(b_0,  space);
    const __m256i bm1_sp = _mm256_cmpeq_epi8(b_m1, space);

    // (a) letter-run interior -> not a boundary
    const uint32_t a_m = (uint32_t)_mm256_movemask_epi8(
        _mm256_and_si256(cur_L, prev_L));
    // (b) lone space before a letter -> boundary. prev may be L, N or O; the
    // guard already excluded CLS_ADDED, so "not S" is the condition.
    const __m256i prev_not_S = _mm256_or_si256(_mm256_or_si256(prev_L, prev_N), prev_O);
    // c+1 only has to be non-whitespace for the run to be exactly [c, c+1).
    const __m256i next_N = _mm256_cmpeq_epi8(k_p1, clsN);
    const __m256i next_O = _mm256_cmpeq_epi8(k_p1, clsO);
    const __m256i next_not_S = _mm256_or_si256(_mm256_or_si256(next_L, next_N), next_O);
    const uint32_t b_m = (uint32_t)_mm256_movemask_epi8(
        _mm256_and_si256(_mm256_and_si256(cur_S, prev_not_S),
                         _mm256_and_si256(b0_sp, next_not_S)));
    // (c)+(c') letter after a space -> not a boundary, whatever c-2 is.
    //
    // (c) covers c-2 in {L, N, O} and (c') covers c-2 == S (a letter ending an
    // indentation run). CLS_ADDED at c-2 is already excluded by the window
    // guard above, so between them every surviving c-2 class is handled and the
    // c-2 term drops out of the vector form entirely -- one fewer comparison
    // chain than the scalar cascade needs.
    const uint32_t c_m = (uint32_t)_mm256_movemask_epi8(
        _mm256_and_si256(_mm256_and_si256(cur_L, prev_S), bm1_sp));
    // (d) punctuation straight after a letter or digit -> boundary
    const uint32_t d_m = (uint32_t)_mm256_movemask_epi8(
        _mm256_and_si256(cur_O, _mm256_or_si256(prev_L, prev_N)));
    // (e) letter straight after punctuation. Resolved either way: it attaches
    // (no boundary) unless c-2 is another O or a literal space.
    const __m256i e_sel = _mm256_and_si256(cur_L, prev_O);
    const uint32_t e_m = (uint32_t)_mm256_movemask_epi8(e_sel);
    const __m256i bm2_sp = _mm256_cmpeq_epi8(b_m2, space);
    const __m256i e_split = _mm256_or_si256(
        mm2_O, _mm256_and_si256(_mm256_cmpeq_epi8(k_m2, clsS), bm2_sp));
    const uint32_t e_yes = (uint32_t)_mm256_movemask_epi8(
        _mm256_and_si256(e_sel, e_split));

    // (h) whitespace-run interior -> not a boundary. Two homogeneous cases:
    // CR/LF after CR/LF, and a plain space between two plain spaces.
    const __m256i cr = _mm256_set1_epi8(0x0D), lf = _mm256_set1_epi8(0x0A);
    const __m256i c0_crlf = _mm256_or_si256(_mm256_cmpeq_epi8(b_0, cr),
                                            _mm256_cmpeq_epi8(b_0, lf));
    const __m256i cm1_crlf = _mm256_or_si256(_mm256_cmpeq_epi8(b_m1, cr),
                                             _mm256_cmpeq_epi8(b_m1, lf));
    const __m256i cp1_crlf = _mm256_or_si256(_mm256_cmpeq_epi8(b_p1, cr),
                                             _mm256_cmpeq_epi8(b_p1, lf));
    const __m256i both_S = _mm256_and_si256(cur_S, prev_S);
    const __m256i h_crlf = _mm256_and_si256(both_S,
        _mm256_and_si256(c0_crlf, cm1_crlf));
    const __m256i next_S = _mm256_cmpeq_epi8(k_p1, clsS);
    const __m256i h_space = _mm256_and_si256(both_S,
        _mm256_andnot_si256(_mm256_or_si256(_mm256_or_si256(c0_crlf, cm1_crlf), cp1_crlf),
                            next_S));
    const uint32_t h_m = (uint32_t)_mm256_movemask_epi8(
        _mm256_or_si256(h_crlf, h_space));

    // (k) the last space of a whitespace run, with a non-space after it --
    // the other end of an indentation run, and a boundary.
    // both_S, neither byte CR/LF, and c+1 not class S.
    const __m256i k_sel = _mm256_andnot_si256(
        _mm256_or_si256(_mm256_or_si256(c0_crlf, cm1_crlf), next_S),
        both_S);
    const uint32_t k_m = (uint32_t)_mm256_movemask_epi8(k_sel);

    // (g) punctuation after a lone space attaches to it -> not a boundary
    const __m256i mm2_S = _mm256_cmpeq_epi8(k_m2, clsS);
    const uint32_t g_m = (uint32_t)_mm256_movemask_epi8(
        _mm256_andnot_si256(mm2_S,
            _mm256_and_si256(_mm256_and_si256(cur_O, prev_S), bm1_sp)));

    // (g') punctuation ENDING an indentation run -> not a boundary, and
    // (n') a digit in the same position -> a boundary. Same selector, opposite
    // answers; see the scalar proofs.
    const __m256i after_ws_run =
        _mm256_and_si256(_mm256_and_si256(prev_S, bm1_sp), mm2_S);
    const uint32_t gp_m = (uint32_t)_mm256_movemask_epi8(
        _mm256_and_si256(after_ws_run, cur_O));
    const __m256i cur_N_v = _mm256_cmpeq_epi8(k_0, clsN);
    const uint32_t np_m = (uint32_t)_mm256_movemask_epi8(
        _mm256_and_si256(after_ws_run, cur_N_v));

    // (f) punctuation-run interior -> not a boundary
    const uint32_t f_m = (uint32_t)_mm256_movemask_epi8(
        _mm256_and_si256(cur_O, prev_O));

    // (i) a lone newline. Resolved either way: it attaches to a preceding
    // punctuation run (prev == O, the regex's trailing [\r\n]*), otherwise it
    // starts a token. prev != S and K[1] != S keep the run at exactly [c,c+1).
    const __m256i c0_lf = _mm256_cmpeq_epi8(b_0, lf);
    const __m256i prev_not_S2 = _mm256_or_si256(
        _mm256_or_si256(prev_L, prev_N), prev_O);
    // What follows the newline does not affect the answer (see the scalar
    // proof), so there is no next_S term here.
    const __m256i i_sel = _mm256_and_si256(
        _mm256_and_si256(cur_S, c0_lf), prev_not_S2);
    const uint32_t i_m = (uint32_t)_mm256_movemask_epi8(i_sel);
    // It is a boundary except when the previous class is O.
    const uint32_t i_yes = (uint32_t)_mm256_movemask_epi8(
        _mm256_andnot_si256(prev_O, i_sel));

    // (j) the character after a lone newline -> always a boundary.
    const __m256i cm1_lf = _mm256_cmpeq_epi8(b_m1, lf);
    // cur in {L, N, O}: the guard already excluded CLS_ADDED, so "not S" is
    // the same set. Built from the class byte directly since there is no
    // cur_N vector in scope.
    const __m256i cur_LNO = _mm256_or_si256(
        _mm256_or_si256(cur_L, cur_O),
        _mm256_cmpeq_epi8(k_0, clsN));
    const __m256i mm2_S2 = _mm256_cmpeq_epi8(k_m2, clsS);
    const uint32_t j_m = (uint32_t)_mm256_movemask_epi8(
        _mm256_andnot_si256(mm2_S2,
            _mm256_and_si256(_mm256_and_si256(prev_S, cm1_lf), cur_LNO)));

    // Disjoint by (cur, prev): (a) L,L  (b) S,non-S  (c) L,S  (d) O,{L,N}
    // (e) L,O  (f) O,O  (g) O,S  (h) S,S -- the cur_L cases differ in prev, as
    // do the cur_O ones, and (b) requires prev non-S so it cannot overlap (h).
    // The newline cases (i) S,non-S and (j) non-S,S overlap (b), (c) and (g)
    // by class, but those three all require the whitespace byte to be 0x20
    // while these require 0x0A, so no lane is claimed by both -- and a lane
    // claimed twice with the same answer would be harmless anyway, since the
    // masks are OR-ed.
    const uint32_t ok = ~bad_m;
    // (k) is S,S like (h) but with a non-S at c+1, so the two are disjoint.
    // (g') and (n') are O,S and N,S with c-2 class S, which (g) excludes by
    // requiring c-2 non-S and (j) excludes by requiring b[-1] == 0x0A; they
    // are disjoint from each other by cur.
    *resolved   = ok & (a_m | b_m | c_m | d_m | e_m | f_m | g_m | h_m
                        | i_m | j_m | k_m | gp_m | np_m);
    *is_boundary = ok & (b_m | d_m | e_yes | i_yes | j_m | k_m | np_m);
}

#endif  // GBPE_HAVE_AVX2_BOUNDARY

#if GBPE_HAVE_AVX2_BOUNDARY
// How many of the next 32 bytes are plain ASCII that start no added token?
// `ok[b]` is the caller's 256-entry table (1 = byte may be taken in bulk).
// Returns the length of the qualifying run starting at `p`, capped at 32, so
// the caller can copy and classify that many bytes without re-testing each.
//
// Reads 32 bytes at `p`, so the caller must have them; it checks the remaining
// length first.
// `roots` holds the ASCII bytes that begin an added token, padded to 4 by
// repeating one of them (or 0xFF when there are none, which matches no ASCII
// byte). Every vocabulary shipped here has exactly one such byte ('<'), and
// the caller falls back to the scalar loop when there are more than 4.
GBPE_AVX2_TARGET inline uint32_t ascii_run_32(const uint8_t* p,
                                              const uint8_t roots[4]) {
    const __m256i v = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(p));
    // A byte disqualifies the run if it is >= 0x80 (sign bit set) or is one of
    // the added-token root bytes. movemask gives the sign bits directly; the
    // root tests are four broadcast compares.
    __m256i bad = v;                                   // sign bit = byte >= 0x80
    bad = _mm256_or_si256(bad, _mm256_cmpeq_epi8(v, _mm256_set1_epi8((char)roots[0])));
    bad = _mm256_or_si256(bad, _mm256_cmpeq_epi8(v, _mm256_set1_epi8((char)roots[1])));
    bad = _mm256_or_si256(bad, _mm256_cmpeq_epi8(v, _mm256_set1_epi8((char)roots[2])));
    bad = _mm256_or_si256(bad, _mm256_cmpeq_epi8(v, _mm256_set1_epi8((char)roots[3])));
    const uint32_t m = static_cast<uint32_t>(_mm256_movemask_epi8(bad));
    return m ? static_cast<uint32_t>(__builtin_ctz(m)) : 32u;
}

// Unsigned range test: x in [lo,hi]  <=>  (x - lo) <= (hi - lo) unsigned.
GBPE_AVX2_TARGET inline __m256i byte_in_range(__m256i x, uint8_t lo, uint8_t hi) {
    const __m256i biased = _mm256_sub_epi8(x, _mm256_set1_epi8(static_cast<char>(lo)));
    const __m256i lim = _mm256_set1_epi8(static_cast<char>(static_cast<uint8_t>(hi - lo)));
    return _mm256_cmpeq_epi8(_mm256_min_epu8(biased, lim), biased);
}

// The ASCII half of gbpe_classify() as range comparisons: the class of a byte
// below 0x80 is a function of three contiguous ranges plus two literals, so it
// needs no table at all. 10.5x the per-byte table lookup, and a test checks it
// against gbpe_classify for all 128 ASCII bytes.
//
//   S: 0x09-0x0D, 0x20      N: 0x30-0x39
//   L: 0x41-0x5A, 0x61-0x7A O: everything else
GBPE_AVX2_TARGET inline void classify_ascii_32(const uint8_t* src, uint8_t* dst) {
    const __m256i v = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src));
    const __m256i is_S = _mm256_or_si256(byte_in_range(v, 0x09, 0x0D),
                                         _mm256_cmpeq_epi8(v, _mm256_set1_epi8(0x20)));
    const __m256i is_N = byte_in_range(v, 0x30, 0x39);
    const __m256i is_L = _mm256_or_si256(byte_in_range(v, 0x41, 0x5A),
                                         byte_in_range(v, 0x61, 0x7A));
    __m256i cls = _mm256_set1_epi8(static_cast<char>(CLS_O));
    cls = _mm256_blendv_epi8(cls, _mm256_set1_epi8(static_cast<char>(CLS_S)), is_S);
    cls = _mm256_blendv_epi8(cls, _mm256_set1_epi8(static_cast<char>(CLS_N)), is_N);
    cls = _mm256_blendv_epi8(cls, _mm256_set1_epi8(static_cast<char>(CLS_L)), is_L);
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(dst), cls);
}
#endif

// Does this CPU actually have AVX2? Checked once and cached; the vector path
// is only entered when this is true, so a binary built with the AVX2 kernel
// still runs on older hardware.
inline bool cpu_has_avx2() {
#if GBPE_HAVE_AVX2_BOUNDARY
    static const bool yes = __builtin_cpu_supports("avx2");
    return yes;
#else
    return false;
#endif
}

}  // namespace gbpe
