// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 the GraphTok contributors
//
// Byte-level pre-tokenization boundary predicates, shared verbatim between
// the GPU boundary kernels (pretok_k2_*) and the host small-input encoder.
// Everything here is pure over the dense per-codepoint arrays produced by the
// classify/gather stage: cls[] (CLS_*), b0[] (first byte, with the U+017F
// fold), drun_start[] (digit-run scan, Llama only) and doc[] (document start
// ordinal per codepoint). Keeping one definition is what makes the host path
// bit-exact by construction rather than by re-validation.
#pragma once
#include <cstdint>

#ifndef GBPE_HD
#  if defined(__CUDACC__)
#    define GBPE_HD __host__ __device__
#  else
#    define GBPE_HD
#  endif
#endif

// The local-window fast path must inline into its caller: its whole point is
// to answer without building the full predicate's stack frame.
#ifndef GBPE_FORCEINLINE
#  if defined(__CUDACC__)
#    define GBPE_FORCEINLINE __forceinline__
#  elif defined(__GNUC__) || defined(__clang__)
#    define GBPE_FORCEINLINE inline __attribute__((always_inline))
#  else
#    define GBPE_FORCEINLINE inline
#  endif
#endif

namespace gbpe {

constexpr uint8_t CLS_L = 0, CLS_N = 1, CLS_S = 2, CLS_O = 3,
                  CLS_CONT = 4, CLS_ADDED = 5;
constexpr uint32_t ADDED_MISS = 0xFFFFFFFFu;

// Decode the codepoint starting at lead byte `p` (avail bytes available).
// Returns the codepoint and sets *clen to its byte length. Callers that accept
// arbitrary bytes must validate first; this helper is exact for valid UTF-8.
GBPE_HD inline uint32_t decode_cp(const uint8_t* p, uint32_t avail, int* clen) {
    uint8_t b0 = p[0];
    if (b0 < 0x80)                  { *clen = 1; return b0; }
    if ((b0 & 0xE0) == 0xC0 && avail >= 2) {
        *clen = 2; return ((b0 & 0x1Fu) << 6) | (p[1] & 0x3Fu);
    }
    if ((b0 & 0xF0) == 0xE0 && avail >= 3) {
        *clen = 3; return ((b0 & 0x0Fu) << 12) | ((p[1] & 0x3Fu) << 6) | (p[2] & 0x3Fu);
    }
    if ((b0 & 0xF8) == 0xF0 && avail >= 4) {
        *clen = 4; return ((b0 & 0x07u) << 18) | ((p[1] & 0x3Fu) << 12)
                        | ((p[2] & 0x3Fu) << 6) | (p[3] & 0x3Fu);
    }
    *clen = 1; return b0;   // malformed/truncated: single byte (refined in #15)
}

namespace pretok {

// Document clamp. Every neighbour access made while deciding for a
// codepoint c must behave as if c's document were the whole input: a neighbour
// in a different document is indistinguishable from out-of-range. `doc[j]` is
// the cp ordinal of the first codepoint of j's document, so same-document is
// simply `doc[j] == my_doc`, and `my_doc` is itself the ordinal of the
// document's first codepoint (the "BOS" position for c). A null `doc` means
// a single document (the host encoder's case): the range test alone decides,
// with no per-access load.
GBPE_HD inline bool same_doc(
    const uint32_t* doc, int j, uint32_t n, uint32_t my_doc) {
    return j >= 0 && static_cast<uint32_t>(j) < n &&
           (doc == nullptr || doc[j] == my_doc);
}

// class at codepoint j, or -1 out of [0,n) / outside my document.
GBPE_HD inline int cls_at(const uint8_t* cls, int j, uint32_t n,
                          const uint32_t* doc, uint32_t my_doc) {
    if (!same_doc(doc, j, n, my_doc) || cls[j] == CLS_ADDED) return -1;
    return static_cast<int>(cls[j]);
}
// first byte of codepoint j, or -1 out of range / outside my document.
GBPE_HD inline int b0_at(const uint8_t* b0, int j, uint32_t n,
                         const uint32_t* doc, uint32_t my_doc) {
    return same_doc(doc, j, n, my_doc) ? (int)b0[j] : -1;
}
GBPE_HD inline int ascii_lower(int b) {
    return (b >= 'A' && b <= 'Z') ? b + 32 : b;
}
GBPE_HD inline bool is_crlf_byte(int b) { return b == 0x0D || b == 0x0A; }

// Contraction tail length if a contraction tail begins at codepoint j: 're/'ve/'ll
// -> 2, 's/'t/'m/'d -> 1, else 0. kCI = case-insensitive (Llama/Qwen) vs
// case-sensitive (GPT-2).
template<bool kCI>
GBPE_HD inline int contraction_tail_len(
    const uint8_t* b0, int j, uint32_t n,
    const uint32_t* doc, uint32_t my_doc) {
    int a = b0_at(b0, j, n, doc, my_doc), b = b0_at(b0, j + 1, n, doc, my_doc);
    if constexpr (kCI) { a = ascii_lower(a); b = ascii_lower(b); }
    if ((a=='r'&&b=='e') || (a=='v'&&b=='e') || (a=='l'&&b=='l')) return 2;
    if (a=='s'||a=='t'||a=='m'||a=='d') return 1;
    return 0;
}
// Does an apostrophe at codepoint k start a (fresh-token-start) contraction?
// Returns tail length (1/2) or 0. This shared free-function form is used only
// by after_contraction_end() below; the per-family predicates define their own
// starts_c()/starts_contraction() with additional cursor-availability rules.
template<bool kCI>
GBPE_HD inline int starts_contraction(
    const uint8_t* cls, const uint8_t* b0, int k, uint32_t n,
    const uint32_t* doc, uint32_t my_doc) {
    if (b0_at(b0, k, n, doc, my_doc) != 0x27) return 0;          // apostrophe
    int tl = contraction_tail_len<kCI>(b0, k + 1, n, doc, my_doc);
    if (!tl) return 0;
    if (k == (int)my_doc) return tl;
    int p = cls_at(cls, k - 1, n, doc, my_doc);
    if (p == CLS_L || p == CLS_N) return tl;
    if (p == CLS_S && b0_at(b0, k - 1, n, doc, my_doc) != 0x20) return tl;
    return 0;
}
// Is codepoint k the first codepoint AFTER a completed contraction (k-1 is the
// tail's last char)? The regex cursor sits at k then, so a letter at k starts a
// fresh pre-token instead of extending the tail.
template<bool kCI>
GBPE_HD inline bool after_contraction_end(
    const uint8_t* cls, const uint8_t* b0, int k, uint32_t n,
    const uint32_t* doc, uint32_t my_doc) {
    if (starts_contraction<kCI>(cls, b0, k - 2, n, doc, my_doc) == 1) return true;
    return starts_contraction<kCI>(cls, b0, k - 3, n, doc, my_doc) == 2;
}

}  // namespace pretok

// Tri-state result of the local-window fast paths below: the position is a
// boundary, is not, or needs the full predicate.
enum FastBoundary : int { kFastNo = 0, kFastYes = 1, kFastUnknown = 2 };


// Local-window fast path for the GPT-2 predicate, mirroring llamaqwen_fast().
// GPT-2's whitespace helpers are purely local -- ws_last_attaches(k) is just
// "k is a space, k+1 is a non-space in range" -- so the proofs are shorter
// than the Llama ones, but the cases and the window are the same.
//
// The window c-3..c+1 is validated once so the body reads `cls` and
// `cp_byte0` directly, and every case needs it apostrophe-free: in_tail(c) and
// starts_c(c) test c-2..c, after_contraction_end tests c-3..c, and the forward
// tests reach c+1. With no 0x27 there, each returns 0/false and the cascade
// falls through to the class rules.
GBPE_HD GBPE_FORCEINLINE FastBoundary gpt2_fast(
    const uint8_t* cls, const uint8_t* cp_byte0, uint32_t n,
    const uint32_t* cp_doc_start, uint32_t c)
{
    if (c < 3u || c + 1u >= n) return kFastUnknown;
    if (cp_doc_start != nullptr &&
        (cp_doc_start[c] != cp_doc_start[c - 3] ||
         cp_doc_start[c] != cp_doc_start[c + 1])) return kFastUnknown;
    const uint8_t* B = cp_byte0 + c;
    const uint8_t* K = cls + c;
    if (B[0] == 0x27 || B[-1] == 0x27 || B[-2] == 0x27 || B[-3] == 0x27 ||
        B[1] == 0x27) return kFastUnknown;
    if (K[0] == CLS_ADDED || K[1] == CLS_ADDED ||
        K[-1] == CLS_ADDED || K[-2] == CLS_ADDED) return kFastUnknown;
    const uint8_t cur = K[0], prev = K[-1];
    // (a) Letter-run interior. ws_last_attaches(c-1) needs CLS(c-1)==S, which
    // prev==L rules out, so the cascade reaches `cur == prev` and continues.
    if (cur == CLS_L && prev == CLS_L) return kFastNo;
    // (b) A lone space before a letter. cur==S, so the cascade reaches the
    // CLS_S branch: ws_split_before_last(c) needs CLS(c-1)==S, which prev!=S
    // rules out, so it is false, and then prev!=S gives continues=false --
    // a boundary.
    if (cur == CLS_S && prev != CLS_S && B[0] == 0x20 && K[1] == CLS_L)
        return kFastYes;
    // (c) The letter after that space. ws_last_attaches(c-1) holds: c-1 is a
    // space (CLS_S, byte 0x20) and c is a letter in range, so the third
    // cascade arm fires with cur==CLS_L and continues is true -- no boundary.
    if (cur == CLS_L && prev == CLS_S && B[-1] == 0x20) return kFastNo;
    // (d) Punctuation straight after a letter or digit. ws_last_attaches(c-1)
    // needs CLS(c-1)==S, ruled out; cur != prev so the same-class arm does not
    // fire; the final else gives continues=false -- a boundary.
    if (cur == CLS_O && (prev == CLS_L || prev == CLS_N)) return kFastYes;
    // (f) Punctuation-run interior -- "->", "==", ");", "});". 4.9% of code
    // positions, the single largest remaining class there. in_tail(c) and
    // starts_c(c) need an apostrophe in c-2..c, excluded by the window guard;
    // ws_last_attaches(c-1) needs CLS(c-1)==S, but prev is O. So the cascade
    // reaches the `cur == prev` arm, where in_tail(c-1) needs an apostrophe at
    // c-2 or c-3 (excluded), starts_c(c) is already false, and the CLS_L
    // clause cannot fire with cur==O. continues is true: no boundary.
    if (cur == CLS_O && prev == CLS_O) return kFastNo;
    // (i)/(j) Newlines. GPT-2's whitespace helpers are purely local and both
    // require the byte to be 0x20 (ws_last_attaches) or a run shape that a
    // lone LF cannot have, so both sides of a newline resolve to a boundary
    // -- and unlike Llama there is no [\r\n]* punctuation-run rule, so the
    // prev == CLS_O sub-case does not arise.
    //   (i) cur is S with byte LF: ws_split_before_last(c) needs
    //       CLS(c-1) == CLS_S, excluded by prev != CLS_S; the third arm needs
    //       ws_last_attaches(c-1), also needing CLS(c-1) == CLS_S. The cascade
    //       reaches the CLS_S branch, where ws_split_before_last is false and
    //       prev != CLS_S gives continues = false: a boundary.
    //   (j) prev is S with byte LF: ws_last_attaches(c-1) requires
    //       B0(c-1) == 0x20, which LF is not, so the third arm does not fire.
    //       cur != prev (prev is S, cur is not), so the same-class arm does
    //       not fire either, and the final else gives continues = false.
    // What follows does not matter here either: ws_split_before_last(c) needs
    // CLS(c-1) == CLS_S, which prev != CLS_S rules out, so the CLS_S branch
    // falls to `prev == CLS_S ? true : false` and gives a boundary.
    if (cur == CLS_S && B[0] == 0x0A && prev != CLS_S)
        return kFastYes;
    if (prev == CLS_S && B[-1] == 0x0A &&
        (cur == CLS_L || cur == CLS_N || cur == CLS_O))
        return kFastYes;
    // (k) The last space of a whitespace run, with a non-space after it.
    // ws_split_before_last(c) is exactly this condition -- c is class S, c+1
    // is a non-S class in range, c-1 is class S -- and the CLS_S branch makes
    // it `continues = false`: a boundary. The earlier arms cannot pre-empt it,
    // since in_tail/starts_c need an apostrophe in the window and
    // ws_last_attaches(c-1) would need CLS(c) != CLS_S.
    if (cur == CLS_S && prev == CLS_S && K[1] != CLS_S) return kFastYes;
    return kFastUnknown;
}

// GPT-2 boundary predicate: is codepoint c the first codepoint of a pre-token?
// Local window {c-2..c+2} + the two ws flags computed inline.
// Callers should try gpt2_fast() first; this handles the rest.
GBPE_HD inline bool gpt2_boundary(
    const uint8_t* cls, const uint8_t* cp_byte0, uint32_t n,
    const uint32_t* cp_doc_start, uint32_t c)
{
    if (cls[c] == CLS_ADDED) return true;
    // Every document start is a forced boundary (regex BOS for that document).
    const uint32_t my_doc = cp_doc_start ? cp_doc_start[c] : 0u;
    if (my_doc == c) return true;

    using namespace pretok;
    auto CLS = [&](int j) { return cls_at(cls, j, n, cp_doc_start, my_doc); };
    auto B0  = [&](int j) { return b0_at(cp_byte0, j, n, cp_doc_start, my_doc); };
    // ws flags (local): last char of a ws-run immediately before a non-space
    auto ws_split_before_last = [&](int k) -> bool {
        return CLS(k) == CLS_S && CLS(k + 1) >= 0 && CLS(k + 1) != CLS_S
            && CLS(k - 1) == CLS_S;
    };
    auto ws_last_attaches = [&](int k) -> bool {
        return CLS(k) == CLS_S && CLS(k + 1) >= 0 && CLS(k + 1) != CLS_S
            && B0(k) == 0x20;
    };
    // GPT-2 contractions are case-sensitive. The regex alternative can fire
    // whenever the apostrophe is at the current regex cursor: BOS, after an
    // L/N token, or after non-attaching whitespace. It cannot fire when the
    // apostrophe is consumed by a preceding O-run or by a leading literal space
    // of an O-run (" 's" -> [" '", "s"]).
    auto starts_c = [&](int k) -> int {
        if (B0(k) != 0x27) return 0;
        int tl = contraction_tail_len<false>(cp_byte0, k + 1, n,
                                             cp_doc_start, my_doc);
        if (!tl) return 0;
        if (k == (int)my_doc) return tl;      // document start == regex BOS
        int p = CLS(k - 1);
        if (p == CLS_O) return 0;
        if (p == CLS_S && B0(k - 1) == 0x20 &&
            ws_last_attaches(k - 1)) return 0;
        return tl;
    };
    auto in_tail = [&](int k) -> bool {
        if (starts_c(k - 1) >= 1) return true;
        return starts_c(k - 2) == 2;
    };

    int prev = CLS(c - 1), cur = CLS(c);
    // Letter-run interior. Every rule below that can split L,L is a
    // contraction rule (in_tail, starts_c, after_contraction_end), and each
    // of those needs an apostrophe at one of c-3..c; ws_last_attaches(c-1)
    // needs CLS(c-1)==S. With no apostrophe in that window the cascade
    // provably reaches `continues = true`, so answer directly.
    if (cur == CLS_L && prev == CLS_L &&
        B0(c) != 0x27 && B0(c - 1) != 0x27 && B0(c - 2) != 0x27 && B0(c - 3) != 0x27) {
        return false;
    }
    bool continues = false;
    if (in_tail(c)) {
        continues = true;
    } else if (starts_c(c)) {
        continues = false;
    } else if (ws_last_attaches(c - 1) && (cur == CLS_L || cur == CLS_N || cur == CLS_O)
               && !starts_c(c)) {
        continues = true;
    } else if (cur == CLS_S) {
        continues = ws_split_before_last(c) ? false : (prev == CLS_S ? true : false);
    } else if (cur == prev && (cur == CLS_L || cur == CLS_N || cur == CLS_O)) {
        continues = !in_tail(c - 1)
                 && !(cur == CLS_O && starts_c(c))
                 && !(cur == CLS_L
                      && after_contraction_end<false>(cls, cp_byte0, c, n,
                                                      cp_doc_start, my_doc));
    } else {
        continues = false;
    }
    return !continues;
}

// Local-window fast path for the Llama-3 / Qwen-2.5 predicate, covering ~88%
// of prose positions: the interior of a letter run, and the two positions
// bracketing a single space between two letter runs. Returns kFastUnknown for
// everything else, and the caller must then run llamaqwen_boundary().
//
// This is deliberately a separate, tiny, force-inlined function. The full
// predicate builds a frame for its lambda cascade on entry, so even a position
// that returns after three loads pays ~170 instructions inside it; answering
// here instead costs ~20 and never touches that frame. Both the host loop and
// the GPU kernel call this first, so host and device stay identical by
// construction.
//
// The window c-3..c+1 is validated once up front, so the body reads `cls` and
// `cp_byte0` directly: with every position in range, in the same document, and
// no CLS_ADDED, cls_at/b0_at would reduce to a plain load anyway.
//
// All three cases require no apostrophe in c-3..c+1, which rules out every
// contraction rule reachable from these positions: in_tail(c) and
// starts_contraction(c) test c-2..c, after_contraction_end tests c-3..c, and
// the forward tests reach c+1. With no 0x27 in that window each returns
// 0/false, so the cascade would fall through to the class rules below.
GBPE_HD GBPE_FORCEINLINE FastBoundary llamaqwen_fast(
    const uint8_t* cls, const uint8_t* cp_byte0, uint32_t n,
    const uint32_t* cp_doc_start, uint32_t c)
{
    if (c < 3u || c + 1u >= n) return kFastUnknown;
    if (cp_doc_start != nullptr &&
        (cp_doc_start[c] != cp_doc_start[c - 3] ||
         cp_doc_start[c] != cp_doc_start[c + 1])) return kFastUnknown;
    const uint8_t* B = cp_byte0 + c;
    const uint8_t* K = cls + c;
    if (B[0] == 0x27 || B[-1] == 0x27 || B[-2] == 0x27 || B[-3] == 0x27 ||
        B[1] == 0x27) return kFastUnknown;
    if (K[0] == CLS_ADDED || K[1] == CLS_ADDED ||
        K[-1] == CLS_ADDED || K[-2] == CLS_ADDED) return kFastUnknown;
    const uint8_t cur = K[0], prev = K[-1];
    // (a) Letter-run interior. crlf_in_orun(c) needs a CR/LF first byte, which
    // no letter has, so with the window apostrophe-free the cascade reaches
    // `continues = true`.
    if (cur == CLS_L && prev == CLS_L) return kFastNo;
    // (b) A lone space before a letter. cur==S and the space is not CR/LF, so
    // crlf_in_orun(c) is false and the cascade reaches ws_continue(c). The run
    // is exactly [c, c+1) because neither neighbour is whitespace -- in_ws_run
    // stops at any non-S class, so prev may be L, N or O -- giving a==c,
    // b==c+1, nl==-1 and r==a==c; ws_continue returns false at k==r, hence a
    // boundary.
    if (cur == CLS_S && prev != CLS_S && prev != CLS_ADDED && B[0] == 0x20 &&
        (K[1] == CLS_L || K[1] == CLS_N || K[1] == CLS_O))
        return kFastYes;
    // (c) A letter after a lone space. cur==L and prev==S, so the cascade
    // reaches `prev == CLS_S && ws_last_attaches(c - 1)`. What makes the run
    // exactly [c-1, c) is that c-2 is not itself whitespace -- in_ws_run stops
    // the backward walk at any non-S class -- so L, N and O all qualify, not
    // just L ("word. Next" is as common as "word next"). With that run:
    // last == c-1 == r, CLS(b)==CLS_L is not EOF, the char is 0x20 and not
    // CR/LF, so it attaches forward and continues is true: no boundary.
    if (cur == CLS_L && prev == CLS_S && B[-1] == 0x20 &&
        (K[-2] == CLS_L || K[-2] == CLS_N || K[-2] == CLS_O))
        return kFastNo;
    // (c') The same, but with c-2 itself whitespace -- a letter ending an
    // indentation run, "\n    def". (c) above excludes it because it requires
    // c-2 to be L, N or O, so on source code every indented line declined into
    // the cascade: 24.1% of all declines on the code corpus, the second-largest
    // class there and by itself 1.09% of positions.
    //
    // c is a letter, so the cascade reaches `prev == CLS_S &&
    // ws_last_attaches(c - 1)`. Let the run containing c-1 be [a, b). The
    // forward walk stops at c (a letter is not class S), so b == c and
    // last == c-1, i.e. the position being asked about IS the run's last
    // character. ws_last_attaches then tests `k != last || last < r`: the first
    // half is false since k == last == c-1, so it reduces to `last < r`, i.e.
    // does a CR/LF appear at or after c-1?
    //
    // c-1 is a literal 0x20 and so is not CR/LF, hence the run's last CR/LF is
    // at or before c-2, giving r <= c-1 == last. So `last < r` is false, the
    // guard does not fire, and the function goes on to ask whether the run's
    // last character attaches forward to what follows -- which for CLS(b) ==
    // CLS_L, a space, and not CR/LF, it does. continues is true: no boundary.
    //
    // Crucially none of that depends on where the run starts or on what is at
    // c-2 beyond it being whitespace, which is why widening the guard is sound.
    // Verified exhaustively against llamaqwen_boundary over the full books,
    // code and multilingual corpora: 29,168 positions, zero disagreements.
    if (cur == CLS_L && prev == CLS_S && B[-1] == 0x20 && K[-2] == CLS_S)
        return kFastNo;
    // (d) Punctuation straight after a letter or digit -- "word," "word." and
    // friends, ~2.9% of prose positions. crlf_in_orun(c) needs is_crlf(c), but
    // CR and LF are class S, so an O position's first byte is never one; with
    // the window apostrophe-free the cascade reaches the CLS_O branch, where
    // prev is neither CLS_O nor CLS_S, so continues stays false: a boundary.
    if (cur == CLS_O && (prev == CLS_L || prev == CLS_N))
        return kFastYes;
    // (e) A letter straight after punctuation -- "(word", "-word". The cascade
    // reaches `prev == CLS_O && o_attaches_letter(c - 1)`, which is entirely
    // local: c-1 is class O and c is a letter, both given, so it turns on c-2
    // alone. It attaches (no boundary) unless c-2 is another O, or is a
    // literal space. The window guard already validated c-2.
    if (cur == CLS_L && prev == CLS_O) {
        const bool attaches =
            K[-2] != CLS_O && !(K[-2] == CLS_S && B[-2] == 0x20);
        return attaches ? kFastNo : kFastYes;
    }
    // (g) Punctuation after a lone space -- " ." and friends. The cascade
    // reaches the CLS_O branch's `prev == CLS_S && B0(c-1) == 0x20 &&
    // ws_last_attaches(c - 1)`. With c-2 not whitespace the run is exactly
    // [c-1, c): last == c-1 == r, CLS(b) == CLS_O is not EOF, and the char is
    // 0x20, so the `nxt == CLS_O && ch == 0x20` arm makes it attach --
    // continues is true, so no boundary.
    if (cur == CLS_O && prev == CLS_S && B[-1] == 0x20 && K[-2] != CLS_S)
        return kFastNo;
    // (g') Punctuation ENDING an indentation run -- "\n    }", "\n  #". Same
    // shape as (g) but with c-2 itself whitespace, which (g) excludes; on code
    // this is the largest class (g) leaves behind.
    //
    // The proof mirrors (c'). c is class O, so the forward walk stops at c,
    // making c-1 the run's last character; ws_last_attaches' guard
    // `k != last || last < r` reduces to `last < r`, and c-1 is a literal 0x20
    // so it is not a CR/LF, putting the run's last CR/LF at or before c-2 and
    // giving r <= c-1 == last. The guard does not fire, and the run's last
    // char then attaches forward via the `nxt == CLS_O && ch == 0x20` arm:
    // continues is true, so no boundary. Where the run starts is irrelevant.
    //
    // (n') The same position with a DIGIT instead: a boundary, not an attach.
    // ws_last_attaches' forward-attach arms accept a following CLS_O (above)
    // or a CLS_L, but a digit takes neither, so continues stays false. That
    // asymmetry is the regex's: \p{N}{1,3} opens a fresh token, while
    // punctuation joins the whitespace token that precedes it.
    //
    // Both verified exhaustively against llamaqwen_boundary over the full
    // books, code and multilingual corpora -- 16,367 O positions and 794 N
    // positions, zero disagreements.
    if (prev == CLS_S && B[-1] == 0x20 && K[-2] == CLS_S) {
        if (cur == CLS_O) return kFastNo;
        if (cur == CLS_N) return kFastYes;
    }
    // (f) Punctuation-run interior -- "->", "==", ");", "});". At 4.9% of
    // positions this is the largest remaining class on the code corpus.
    // in_tail(c) and starts_contraction(c) need an apostrophe in c-2..c,
    // excluded by the window guard. crlf_in_orun(c) needs is_crlf(c), but CR
    // and LF are class S and cur is O. So the cascade reaches the CLS_O
    // branch's `prev == CLS_O && !o_attaches_letter(c)`, and
    // o_attaches_letter(c) returns false on its own `CLS(c-1) == CLS_O` test
    // before reading anything else -- so continues is true unconditionally,
    // with no dependence on c+1 or c-2.
    if (cur == CLS_O && prev == CLS_O) return kFastNo;
    // (h) Whitespace-run interior. The full predicate already short-circuits
    // these, but only after building its lambda frame; answering here skips
    // that entirely. Indentation makes them 6.3% of positions in code.
    // Two homogeneous cases, exactly as the predicate's own shortcut states:
    // a CR/LF following a CR/LF, and a non-CR/LF space with a non-CR/LF space
    // on each side. Both continue the run, so neither is a boundary.
    if (cur == CLS_S && prev == CLS_S) {
        const bool c_crlf = (B[0] == 0x0D || B[0] == 0x0A);
        const bool p_crlf = (B[-1] == 0x0D || B[-1] == 0x0A);
        if (c_crlf && p_crlf) return kFastNo;
        if (!c_crlf && !p_crlf && K[1] == CLS_S &&
            B[1] != 0x0D && B[1] != 0x0A) {
            return kFastNo;
        }
        // (k) The last space of a whitespace run, with a non-space after it.
        // This is the other end of an indentation run -- (h) above covers its
        // interior -- and on code it is most of the residual S,S positions.
        // The forward walk stops at c, so b == c+1 and last == c. Neither c
        // nor c-1 is CR/LF, so if the run holds a CR/LF at all its last one is
        // at or before c-2, giving r == nl+1 <= c-1 < b. ws_continue then
        // reaches `k == last` and returns false: a boundary. (The `k > a &&
        // k <= nl` arm cannot fire because k == last > nl.)
        if (!c_crlf && !p_crlf && K[1] != CLS_S) return kFastYes;
    }
    // (i) A lone newline. Cases (b)/(c)/(g) all require the whitespace byte to
    // be exactly 0x20, so until now every newline entered the cascade twice --
    // once here and once at the character after it. Newlines drive 98.8% of
    // the remaining cascade entries on prose and about half on code.
    //
    // cur is class S with first byte LF, and neither neighbour is whitespace
    // (prev != CLS_S stops in_ws_run's backward walk, K[1] != CLS_S its
    // forward walk), so the run is exactly [c, c+1). Two sub-cases, split by
    // whether the LF is claimed by a preceding punctuation run's trailing
    // [\r\n]*:
    //   prev == CLS_O: crlf_in_orun(c) walks back over contiguous CR/LF from
    //     c-1 -- c-1 is class O, so not CR/LF -- and finds CLS(c-1) == CLS_O,
    //     so it returns true and the cascade sets continues = true: no
    //     boundary, the LF attaches to the punctuation token.
    //   prev is L or N: crlf_in_orun(c) finds a non-O immediately, so it is
    //     false and the cascade reaches ws_continue(c). With the run [c, c+1):
    //     a == c, b == c+1, nl == c (the LF is the run's last CR/LF) and
    //     r == nl+1 == c+1. The `nl >= 0 && k > a` arm fails because k == a,
    //     then `k < r` holds, so ws_continue returns false: a boundary.
    // What follows the newline does not matter. The backward walk stops at
    // c (prev is not whitespace), so a == c; c is itself CR/LF, so the run's
    // last CR/LF is at or after c, giving nl >= c and r == nl+1 > c. With
    // k == a the `nl >= 0 && k > a` arm cannot fire, and k < r then returns
    // false. So the run may extend forwards over indentation -- as it does on
    // every line of source code -- without changing the answer.
    if (cur == CLS_S && B[0] == 0x0A && prev != CLS_S) {
        return prev == CLS_O ? kFastNo : kFastYes;
    }
    // (j) The character after a lone newline. For cur == CLS_L the cascade
    // reaches `prev == CLS_S && ws_last_attaches(c - 1)`; that run is
    // [c-1, c) with nl == c-1 and r == nl+1 == c, so its `k != last ||
    // last < r` guard fires (last == c-1 < r == c) and it returns false.
    // For cur == CLS_O the CLS_O branch needs B0(c-1) == 0x20, which LF is
    // not. For cur == CLS_N the digit branch gives continues = false because
    // prev != CLS_N. In every case continues stays false: a boundary.
    // K[-2] != CLS_S keeps the run at [c-1, c) as the proof assumes.
    if (prev == CLS_S && B[-1] == 0x0A && K[-2] != CLS_S &&
        (cur == CLS_L || cur == CLS_N || cur == CLS_O)) {
        return kFastYes;
    }
    return kFastUnknown;
}

// Llama-3 / Qwen-2.5 boundary predicate. Local window {k-2..k+1} + the
// digit-run scan (drun_start) for the 3-cap + in-run walks for whitespace and
// O-run CRLF. kDigitCap: Llama true (\p{N}{1,3}); Qwen false (single \p{N}).
// Callers should try llamaqwen_fast() first; this handles the rest.
template<bool kDigitCap>
GBPE_HD inline bool llamaqwen_boundary(
    const uint8_t* cls, const uint8_t* cp_byte0, const uint32_t* drun_start,
    uint32_t n, const uint32_t* cp_doc_start, uint32_t c)
{
    if (cls[c] == CLS_ADDED) return true;
    // Document start == regex BOS == forced boundary.
    const uint32_t my_doc = cp_doc_start ? cp_doc_start[c] : 0u;
    if (my_doc == c) return true;

    using namespace pretok;
    // All neighbour reads below go through these two clamped accessors, so
    // every run walk / EOF test (CLS(b) < 0) stops at the document edge.
    auto CLS = [&](int j) { return cls_at(cls, j, n, cp_doc_start, my_doc); };
    auto B0  = [&](int j) { return b0_at(cp_byte0, j, n, cp_doc_start, my_doc); };
    auto is_crlf = [&](int j) { return is_crlf_byte(B0(j)); };
    // crlf_in_orun(k): k is class S and a CR/LF, immediately following an O-run
    // (the run of O chars directly before this CR/LF, no intervening space).
    auto crlf_in_orun = [&](int k) -> bool {
        if (!is_crlf(k)) return false;
        // walk back over contiguous CR/LF then require an O char immediately before
        int j = k - 1;
        while (j >= (int)my_doc && is_crlf(j)) --j;
        return j >= (int)my_doc && CLS(j) == CLS_O;
    };
    auto in_ws_run = [&](int k) -> bool {
        return CLS(k) == CLS_S && !crlf_in_orun(k);
    };

    // O-char-before-letter attach (word rule single leading char).
    auto o_attaches_letter = [&](int k) -> bool {
        if (CLS(k) != CLS_O) return false;
        if (!(CLS(k + 1) == CLS_L)) return false;
        if (k == (int)my_doc) return true;
        int pc = CLS(k - 1);
        if (pc == CLS_O) return false;
        if (pc == CLS_S && B0(k - 1) == 0x20) return false;
        return true;
    };

    // Whitespace-run role for codepoint k (cls==S).
    // Returns: continues (k continues the previous token).
    auto ws_continue = [&](int k) -> bool {
        // Find run [a,b). The backward walk must NOT cross CR/LF chars claimed
        // by a preceding punct-run's trailing [\r\n]* (crlf_in_orun, folded
        // into in_ws_run): those belong to the O token, so the whitespace run
        // restarts after them.
        const int WALK_CAP = static_cast<int>(n) + 1;
        int a = k; int aw = 0; while (a - 1 >= (int)my_doc && in_ws_run(a - 1) && ++aw <= WALK_CAP) --a;
        int b = k; int bw = 0; while ((uint32_t)b < n && in_ws_run(b) && ++bw <= WALK_CAP) ++b;
        // last CR/LF in run
        int nl = -1;
        for (int j = b - 1; j >= a; --j) { if (is_crlf(j)) { nl = j; break; } }
        int r = (nl >= 0) ? nl + 1 : a;
        // [a, nl]: the \s*[\r\n]+ token; interior (a, nl] continues
        if (nl >= 0 && k > a && k <= nl) return true;
        if (k < r) return false;          // shouldn't happen (k in [a,b))
        if (r >= b) return false;         // run fully consumed by newline token
        // remainder [r, b)
        if (k == r) return false;         // remainder starts fresh
        if (CLS(b) < 0) {
            return k > r;                 // whole remainder one token
        }
        int last = b - 1;
        if (k == last) return false;      // boundary before last char
        return k > r && k < last;         // prefix interior
    };
    // does the last char of k's run attach forward to the run at b?
    auto ws_last_attaches = [&](int k) -> bool {
        const int WALK_CAP = static_cast<int>(n) + 1;
        int a = k; int aw = 0; while (a - 1 >= (int)my_doc && in_ws_run(a - 1) && ++aw <= WALK_CAP) --a;
        int b = k; int bw = 0; while ((uint32_t)b < n && in_ws_run(b) && ++bw <= WALK_CAP) ++b;
        if (CLS(b) < 0) return false;     // logical span EOF: no attach
        int nl = -1;
        for (int j = b - 1; j >= a; --j) { if (is_crlf(j)) { nl = j; break; } }
        int r = (nl >= 0) ? nl + 1 : a;
        int last = b - 1;
        if (k != last || last < r) return false;  // only the remainder's last char
        int nxt = CLS(b), ch = B0(last);
        if (nxt == CLS_L && !is_crlf(last)) return true;
        if (nxt == CLS_O && ch == 0x20) return true;
        return false;
    };
    // Llama/Qwen contractions are case-insensitive. Same cursor-availability
    // rule as GPT-2: after L/N or non-attaching whitespace is valid; after an
    // O-run or a leading literal space of an O-run is not.
    auto starts_contraction = [&](int k) -> int {
        if (B0(k) != 0x27) return 0;
        int tl = contraction_tail_len<true>(cp_byte0, k + 1, n,
                                            cp_doc_start, my_doc);
        if (!tl) return 0;
        if (k == (int)my_doc) return tl;      // document start == regex BOS
        int p = CLS(k - 1);
        if (p == CLS_O) return 0;
        if (p == CLS_S && B0(k - 1) == 0x20 &&
            ws_last_attaches(k - 1)) return 0;
        return tl;
    };
    auto in_tail = [&](int k) -> bool {
        if (starts_contraction(k - 1) >= 1) return true;
        return starts_contraction(k - 2) == 2;
    };

    int prev = CLS(c - 1), cur = CLS(c);
    // The common cases are handled by llamaqwen_fast() before this function is
    // ever entered (see its contract); reaching here means it returned
    // kFastUnknown, so only the edge cases below remain. The letter-run
    // interior is retried through the clamped accessors because
    // llamaqwen_fast() declines the whole window near a document edge.
    if (cur == CLS_L && prev == CLS_L &&
        B0(c) != 0x27 && B0(c - 1) != 0x27 && B0(c - 2) != 0x27 && B0(c - 3) != 0x27) {
        return false;
    }
    // Homogeneous whitespace interiors have no regex decision at their exact
    // run bounds. Bypass the general run reconstruction so a million spaces or
    // newlines stays linear instead of making every lane walk the whole run.
    if (cur == CLS_S && prev == CLS_S) {
        if (is_crlf(c) && is_crlf(c - 1)) {
            return false;
        }
        if (!is_crlf(c) && !is_crlf(c - 1) &&
            CLS(c + 1) == CLS_S && !is_crlf(c + 1)) {
            return false;
        }
    }
    bool continues = false;
    if (in_tail(c)) {
        continues = true;
    } else if (starts_contraction(c)) {
        continues = false;
    } else if (crlf_in_orun(c)) {
        continues = true;
    } else if (cur == CLS_S) {
        continues = ws_continue(c);
    } else if (cur == CLS_L) {
        if (prev == CLS_L && !in_tail(c - 1))
            continues = !after_contraction_end<true>(cls, cp_byte0, c, n,
                                                     cp_doc_start, my_doc);
        else if (prev == CLS_S && ws_last_attaches(c - 1)) continues = true;
        else if (prev == CLS_O && o_attaches_letter(c - 1)) continues = true;
        else continues = false;
    } else if (cur == CLS_N) {
        if (kDigitCap) {
            uint32_t off = c - drun_start[c];   // scan-based; correct for long runs
            continues = (prev == CLS_N && (off % 3u) != 0u);
        } else {
            continues = false;                  // Qwen: every digit its own token
        }
    } else if (cur == CLS_O) {
        if (prev == CLS_O && !o_attaches_letter(c)) continues = true;
        else if (prev == CLS_S && B0(c - 1) == 0x20 && ws_last_attaches(c - 1)) continues = true;
        else continues = false;
    } else {
        continues = false;
    }
    return !continues;
}

}  // namespace gbpe
