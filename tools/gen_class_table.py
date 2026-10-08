#!/usr/bin/env python3
"""Generate a compact GPU-loadable codepoint classification table.

SOURCE OF TRUTH = tiktoken's regex engine (`import regex`), NOT PCRE2. The GPU
k1 kernel cannot call Python regex at runtime, so we bake the L/N/\\s
classification into a 2-level table generated here at build time. This mirrors
the reference classify() (precedence L, then N, then S, else
OTHER), which is the bit-exact gate engine.

Layout (compact 2-level):
  * For each codepoint cp we store 2 bits:
        bit0 = isL  (\\p{L} fullmatch)
        bit1 = isN  (\\p{N} fullmatch)
        bit2 = isS  (\\s    fullmatch)
    Precedence collapses these to a 2-bit class at lookup time: L<N<S<O.
  * Level 2 "leaves": each leaf covers a 256-codepoint block, 256 entries x
    2 bits = 64 bytes. Identical leaves are deduplicated (most blocks are
    uniformly OTHER).
  * Level 1 "block index": (cp>>8) -> leaf id (uint16). 0x1100 blocks.
  * ASCII fast-path: cp<128 is the first half of leaf for block 0; callers may
    short-circuit, but the table lookup is correct for ASCII too.

Run:  python3 tools/gen_class_table.py
"""
import os
import sys

import regex as _re

# ---- Classification engine (matches pretok_ref_difftest.classify precedence) ----
_RL = _re.compile(r"\p{L}")
_RN = _re.compile(r"\p{N}")
_RS = _re.compile(r"\s")
_RM = _re.compile(r"\p{M}")   # marks (NEW, for DeepSeek-V3)
_RP = _re.compile(r"\p{P}")   # punctuation (NEW)
_RSY = _re.compile(r"\p{S}")  # symbols (NEW)

MAX_CP = 0x10FFFF
SUR_LO, SUR_HI = 0xD800, 0xDFFF  # surrogates -> OTHER

# Class encoding for the packed 2-bit slot AND the helper return value.
L, N, S, O = 0, 1, 2, 3

# ---- WIDE table bit layout (per-codepoint byte, regex-derived bits only) ----
# Only the 6 properties that need the `regex` engine are stored in the table.
# cjk / apl / apc are pure hardcoded codepoint ranges and are computed in the
# inline helpers from `cp` directly (no stored bits, smaller leaves, DRYer).
WB_L   = 1 << 0   # \p{L}
WB_N   = 1 << 1   # \p{N}
WB_WS  = 1 << 2   # \s   (whitespace; the narrow "S" class)
WB_M   = 1 << 3   # \p{M}
WB_P   = 1 << 4   # \p{P}
WB_SY  = 1 << 5   # \p{S}  (symbol)
# bits 6,7 unused/reserved.

# Hardcoded ranges (NOT regex-derived; computed in helpers and validated here).
_CJK_RANGES = ((0x4E00, 0x9FA5), (0x3040, 0x309F), (0x30A0, 0x30FF))
_APC_CHARS = frozenset(ord(c) for c in "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~")


def is_cjk(cp):
    return any(lo <= cp <= hi for lo, hi in _CJK_RANGES)


def is_apl(cp):
    return (0x41 <= cp <= 0x5A) or (0x61 <= cp <= 0x7A)


def is_apc(cp):
    return cp in _APC_CHARS


def wide_bits_of(cp):
    """Regex-derived property byte for a codepoint (bits per WB_* layout)."""
    if SUR_LO <= cp <= SUR_HI or cp > MAX_CP:
        return 0
    ch = chr(cp)
    b = 0
    if _RL.fullmatch(ch):
        b |= WB_L
    if _RN.fullmatch(ch):
        b |= WB_N
    if _RS.fullmatch(ch):
        b |= WB_WS
    if _RM.fullmatch(ch):
        b |= WB_M
    if _RP.fullmatch(ch):
        b |= WB_P
    if _RSY.fullmatch(ch):
        b |= WB_SY
    return b


def class_of(cp):
    """2-bit class of a codepoint per the regex engine, with L<N<S precedence."""
    if SUR_LO <= cp <= SUR_HI:
        return O
    ch = chr(cp)
    if _RL.fullmatch(ch):
        return L
    if _RN.fullmatch(ch):
        return N
    if _RS.fullmatch(ch):
        return S
    return O


def build():
    n_blocks = (MAX_CP + 1 + 255) // 256  # 0x1100 blocks for cp 0..0x10FFFF
    # Build each block as a 64-byte leaf (256 * 2 bits, 4 entries per byte).
    leaf_pool = []          # list of bytes objects, each 64 bytes
    leaf_index = {}         # bytes -> leaf id
    block_to_leaf = []      # list of leaf ids (uint16), len = n_blocks

    for blk in range(n_blocks):
        base = blk << 8
        leaf = bytearray(64)
        for i in range(256):
            cp = base + i
            cls = class_of(cp) if cp <= MAX_CP else O
            # pack 2 bits at position i: byte i>>2, shift (i&3)*2
            leaf[i >> 2] |= (cls & 0x3) << ((i & 3) * 2)
        leaf = bytes(leaf)
        lid = leaf_index.get(leaf)
        if lid is None:
            lid = len(leaf_pool)
            leaf_index[leaf] = lid
            leaf_pool.append(leaf)
        block_to_leaf.append(lid)

    assert len(leaf_pool) <= 0xFFFF, "leaf id overflows uint16"
    return n_blocks, leaf_pool, block_to_leaf


def build_wide():
    """Build the WIDE table: one byte (regex-derived bits) per codepoint.

    Same 2-level dedup-trie structure as build(), but each leaf is 256 bytes
    (one byte per codepoint, no bit-packing across codepoints) so the property
    bits stay directly addressable. Identical leaves are deduplicated.
    """
    n_blocks = (MAX_CP + 1 + 255) // 256
    leaf_pool = []          # list of bytes objects, each 256 bytes
    leaf_index = {}
    block_to_leaf = []

    for blk in range(n_blocks):
        base = blk << 8
        leaf = bytearray(256)
        for i in range(256):
            cp = base + i
            leaf[i] = wide_bits_of(cp) if cp <= MAX_CP else 0
        leaf = bytes(leaf)
        lid = leaf_index.get(leaf)
        if lid is None:
            lid = len(leaf_pool)
            leaf_index[leaf] = lid
            leaf_pool.append(leaf)
        block_to_leaf.append(lid)

    assert len(leaf_pool) <= 0xFFFF, "wide leaf id overflows uint16"
    return n_blocks, leaf_pool, block_to_leaf


def emit_wide_lines(w, n_blocks, leaf_pool, block_to_leaf):
    """Append the WIDE table arrays + accessors to the header line list `w`."""
    n_leaves = len(leaf_pool)
    leaf_bytes = n_leaves * 256
    block_bytes = n_blocks * 2
    total = leaf_bytes + block_bytes

    w("")
    w("// ===========================================================================")
    w("// WIDE property table (DeepSeek-V3). One byte per codepoint packing the")
    w("// regex-derived booleans below; cjk/apl/apc are pure hardcoded codepoint")
    w("// ranges computed in the helpers (no stored bits). Same dedup-trie layout.")
    w("// This is ADDITIVE: the narrow gbpe_class_leaves/blocks/gbpe_classify above")
    w("// are byte-identical to the 2-bit build and untouched, so existing kernels")
    w("// (GPT-2/Llama/Qwen) are unaffected.")
    w("// ===========================================================================")
    w("")
    w("#define GBPE_WB_L  0x01u  // \\p{L}")
    w("#define GBPE_WB_N  0x02u  // \\p{N}")
    w("#define GBPE_WB_WS 0x04u  // \\s")
    w("#define GBPE_WB_M  0x08u  // \\p{M}")
    w("#define GBPE_WB_P  0x10u  // \\p{P}")
    w("#define GBPE_WB_SY 0x20u  // \\p{S} (symbol)")
    w("")
    w(f"#define GBPE_WIDE_NUM_BLOCKS {n_blocks}u")
    w(f"#define GBPE_WIDE_NUM_LEAVES {n_leaves}u")
    w(f"#define GBPE_WIDE_LEAF_BYTES 256u   // 256 entries * 1 byte")
    w("")
    w(f"// Level-2 wide leaf pool: {n_leaves} leaves * 256 bytes = {leaf_bytes} bytes.")
    w(f"static const uint8_t gbpe_wide_leaves[{n_leaves} * 256] = {{")
    flat = b"".join(leaf_pool)
    for i in range(0, len(flat), 16):
        chunk = flat[i:i + 16]
        w("  " + "".join(f"0x{b:02x}," for b in chunk))
    w("};")
    w("")
    w(f"// Level-1 wide block index: (cp>>8) -> leaf id. {n_blocks} * 2 = "
      f"{block_bytes} bytes.")
    w(f"static const uint16_t gbpe_wide_blocks[{n_blocks}] = {{")
    for i in range(0, n_blocks, 16):
        chunk = block_to_leaf[i:i + 16]
        w("  " + "".join(f"{v}," for v in chunk))
    w("};")
    w("")
    w("// Wide accessor: returns the packed regex-derived property byte (GBPE_WB_*).")
    w("static inline GBPE_HD uint8_t gbpe_class_bits(uint32_t cp) {")
    w("  uint32_t over = (uint32_t)(cp > GBPE_CLASS_MAX_CP);")
    w("  uint32_t blk = (cp >> 8) & 0x1FFFu;")
    w("  uint32_t leaf = gbpe_wide_blocks[blk * (1u - over)];")
    w("  uint32_t lo = cp & 0xFFu;")
    w("  uint8_t bits = gbpe_wide_leaves[leaf * 256u + lo];")
    w("  return (uint8_t)(bits * (1u - over));  // 0 when out of range")
    w("}")
    w("")
    w("// Per-property inline helpers (regex-derived from the table).")
    w("static inline GBPE_HD int gbpe_is_L(uint32_t cp)  { return (gbpe_class_bits(cp) & GBPE_WB_L)  != 0; }")
    w("static inline GBPE_HD int gbpe_is_N(uint32_t cp)  { return (gbpe_class_bits(cp) & GBPE_WB_N)  != 0; }")
    w("static inline GBPE_HD int gbpe_is_ws(uint32_t cp) { return (gbpe_class_bits(cp) & GBPE_WB_WS) != 0; }")
    w("static inline GBPE_HD int gbpe_is_M(uint32_t cp)  { return (gbpe_class_bits(cp) & GBPE_WB_M)  != 0; }")
    w("static inline GBPE_HD int gbpe_is_P(uint32_t cp)  { return (gbpe_class_bits(cp) & GBPE_WB_P)  != 0; }")
    w("static inline GBPE_HD int gbpe_is_S(uint32_t cp)  { return (gbpe_class_bits(cp) & GBPE_WB_SY) != 0; }")
    w("")
    w("// Hardcoded-range helpers (NOT regex-derived; computed straight from cp).")
    w("static inline GBPE_HD int gbpe_is_cjk(uint32_t cp) {")
    w("  return (cp >= 0x4E00u && cp <= 0x9FA5u) ||")
    w("         (cp >= 0x3040u && cp <= 0x309Fu) ||")
    w("         (cp >= 0x30A0u && cp <= 0x30FFu);")
    w("}")
    w("static inline GBPE_HD int gbpe_is_apl(uint32_t cp) {")
    w("  return (cp >= 0x41u && cp <= 0x5Au) || (cp >= 0x61u && cp <= 0x7Au);")
    w("}")
    w("static inline GBPE_HD int gbpe_is_apc(uint32_t cp) {")
    w("  // ASCII punctuation: the 22 chars !\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~")
    w("  return (cp >= 0x21u && cp <= 0x2Fu) || (cp >= 0x3Au && cp <= 0x40u) ||")
    w("         (cp >= 0x5Bu && cp <= 0x60u) || (cp >= 0x7Bu && cp <= 0x7Eu);")
    w("}")
    w("")
    w("// DeepSeek 'gap'/C-category: derived, NOT stored. True when none of")
    w("// L | N | ws | M | P | S(symbol) | cjk holds.")
    w("static inline GBPE_HD int gbpe_is_gap(uint32_t cp) {")
    w("  uint8_t b = gbpe_class_bits(cp);")
    w("  uint8_t any = b & (GBPE_WB_L|GBPE_WB_N|GBPE_WB_WS|GBPE_WB_M|GBPE_WB_P|GBPE_WB_SY);")
    w("  return !any && !gbpe_is_cjk(cp);")
    w("}")
    return total, n_leaves


def emit_header(path, n_blocks, leaf_pool, block_to_leaf, wide):
    n_leaves = len(leaf_pool)
    leaf_bytes = n_leaves * 64
    block_bytes = n_blocks * 2  # uint16
    total = leaf_bytes + block_bytes

    lines = []
    w = lines.append
    w("// AUTO-GENERATED by tools/gen_class_table.py — DO NOT EDIT.")
    w("// Codepoint L/N/\\s classification table, byte-identical to tiktoken's")
    w("// `regex` engine (the bit-exact gate). Deterministic output.")
    w("//")
    w("// class encoding (2 bits): 0=L (\\p{L}), 1=N (\\p{N}), 2=S (\\s), 3=OTHER.")
    w("// precedence at classify: L < N < S < OTHER (matches classify() in")
    w("// the reference classifier).")
    w("//")
    w("// ASCII FAST-PATH: codepoints < 128 live in block 0; a caller may special-")
    w("// case cp<128, but gbpe_classify() below is already correct for ASCII.")
    w("#ifndef GBPE_CLASS_TABLE_H")
    w("#define GBPE_CLASS_TABLE_H")
    w("")
    w("#include <stdint.h>")
    w("")
    w("#if defined(__CUDACC__)")
    w("#  define GBPE_HD __host__ __device__")
    w("#else")
    w("#  define GBPE_HD")
    w("#endif")
    w("")
    w("#define GBPE_CLASS_L 0u")
    w("#define GBPE_CLASS_N 1u")
    w("#define GBPE_CLASS_S 2u")
    w("#define GBPE_CLASS_O 3u")
    w("")
    w(f"#define GBPE_CLASS_MAX_CP 0x{MAX_CP:X}u")
    w(f"#define GBPE_CLASS_NUM_BLOCKS {n_blocks}u  // cp>>8 range")
    w(f"#define GBPE_CLASS_NUM_LEAVES {n_leaves}u")
    w(f"#define GBPE_CLASS_LEAF_BYTES 64u          // 256 entries * 2 bits")
    w("")
    # Leaf pool: flat uint8 array, n_leaves * 64 bytes.
    w(f"// Level-2 leaf pool: {n_leaves} leaves * 64 bytes = {leaf_bytes} bytes.")
    w(f"static const uint8_t gbpe_class_leaves[{n_leaves} * 64] = {{")
    flat = b"".join(leaf_pool)
    for i in range(0, len(flat), 16):
        chunk = flat[i:i + 16]
        w("  " + "".join(f"0x{b:02x}," for b in chunk))
    w("};")
    w("")
    # Block index: uint16 leaf id per 256-cp block.
    w(f"// Level-1 block index: (cp>>8) -> leaf id. {n_blocks} entries * 2 bytes "
      f"= {block_bytes} bytes.")
    w(f"static const uint16_t gbpe_class_blocks[{n_blocks}] = {{")
    for i in range(0, n_blocks, 16):
        chunk = block_to_leaf[i:i + 16]
        w("  " + "".join(f"{v}," for v in chunk))
    w("};")
    w("")
    # Helper: branch-light, integer-only 2-level lookup + precedence.
    w("// Returns 0=L, 1=N, 2=S, 3=OTHER. Integer ops only, branch-light.")
    w("static inline GBPE_HD uint8_t gbpe_classify(uint32_t cp) {")
    w("  // Out-of-range / above plane 16 -> OTHER.")
    w("  uint32_t over = (uint32_t)(cp > GBPE_CLASS_MAX_CP);")
    w("  uint32_t blk = (cp >> 8) & 0x1FFFu;            // block index")
    w("  uint32_t leaf = gbpe_class_blocks[blk * (1u - over)];")
    w("  uint32_t lo = cp & 0xFFu;                       // index within block")
    w("  uint32_t byte = gbpe_class_leaves[leaf * 64u + (lo >> 2)];")
    w("  uint32_t cls = (byte >> ((lo & 3u) * 2u)) & 0x3u;")
    w("  // Force OTHER when out of range (over==1).")
    w("  return (uint8_t)(cls | (over * 3u));")
    w("}")

    # ---- Append the WIDE table (additive; narrow section above is untouched). ----
    wn_blocks, wleaf_pool, wblock_to_leaf = wide
    wide_total, wide_leaves = emit_wide_lines(w, wn_blocks, wleaf_pool, wblock_to_leaf)

    w("")
    w("#endif  // GBPE_CLASS_TABLE_H")

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    return total, n_leaves, wide_total, wide_leaves


def lookup_from_table(cp, n_blocks, leaf_pool, block_to_leaf):
    """Python mirror of gbpe_classify, used for self-validation."""
    if cp > MAX_CP:
        return O
    leaf = leaf_pool[block_to_leaf[(cp >> 8)]]
    lo = cp & 0xFF
    byte = leaf[lo >> 2]
    return (byte >> ((lo & 3) * 2)) & 0x3


def wide_lookup_from_table(cp, leaf_pool, block_to_leaf):
    """Python mirror of gbpe_class_bits."""
    if cp > MAX_CP:
        return 0
    return leaf_pool[block_to_leaf[(cp >> 8)]][cp & 0xFF]


def main():
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out = os.path.join(repo, "generated", "gbpe_class_table.h")

    n_blocks, leaf_pool, block_to_leaf = build()
    wide = build_wide()
    total, n_leaves, wide_total, wide_leaves = emit_header(
        out, n_blocks, leaf_pool, block_to_leaf, wide)

    # ---- Narrow validation: must equal the regex engine for ALL codepoints. ----
    mism = 0
    count = 0
    for cp in range(0, MAX_CP + 1):
        count += 1
        ref = class_of(cp)
        got = lookup_from_table(cp, n_blocks, leaf_pool, block_to_leaf)
        if ref != got:
            mism += 1
            if mism <= 10:
                print(f"  MISMATCH cp=U+{cp:04X}: ref={ref} table={got}")
    if mism:
        print(f"FAIL: {mism} narrow mismatches", file=sys.stderr)
        return 1

    print(f"Header: {out}")
    print(f"Packed size: {total} bytes "
          f"({n_leaves} leaves * 64 = {n_leaves*64} + {n_blocks} blocks * 2 = "
          f"{n_blocks*2})  (~{total/1024:.1f} KB, L2-resident)")
    print(f"TABLE EXACT: {count} codepoints, {n_leaves} leaves, {total} bytes")

    # ---- Wide validation: exhaustive over ALL codepoints 0..0x10FFFF. ----
    wn_blocks, wleaf_pool, wblock_to_leaf = wide
    per_class = {k: 0 for k in
                 ("L", "N", "ws", "M", "P", "S", "cjk", "apl", "apc")}
    wmism = 0
    wcount = 0
    for cp in range(0, MAX_CP + 1):
        wcount += 1
        bits = wide_lookup_from_table(cp, wleaf_pool, wblock_to_leaf)
        is_sur = SUR_LO <= cp <= SUR_HI
        ch = chr(cp)
        # regex-derived bits: gate the TABLE lookup directly against the regex
        # engine (NOT the builder) — this is the M/P/S churn gate per req #4.
        for name, mask, rx in (
            ("L", WB_L, _RL), ("N", WB_N, _RN), ("ws", WB_WS, _RS),
            ("M", WB_M, _RM), ("P", WB_P, _RP), ("S", WB_SY, _RSY),
        ):
            got = (bits & mask) != 0
            want = (not is_sur) and (rx.fullmatch(ch) is not None)
            if got != want:
                per_class[name] += 1
                wmism += 1
        # Hardcoded-range properties: validate the emitted C-helper RANGE logic
        # against the authoritative Python definitions (the C helpers compute
        # these from cp, not from the table — so we mirror their ranges here).
        c_cjk = ((0x4E00 <= cp <= 0x9FA5) or (0x3040 <= cp <= 0x309F)
                 or (0x30A0 <= cp <= 0x30FF))
        c_apl = (0x41 <= cp <= 0x5A) or (0x61 <= cp <= 0x7A)
        c_apc = ((0x21 <= cp <= 0x2F) or (0x3A <= cp <= 0x40)
                 or (0x5B <= cp <= 0x60) or (0x7B <= cp <= 0x7E))
        for name, c_val, ref_fn in (("cjk", c_cjk, is_cjk),
                                    ("apl", c_apl, is_apl),
                                    ("apc", c_apc, is_apc)):
            if bool(c_val) != bool(ref_fn(cp)):
                per_class[name] += 1
                wmism += 1

    if wmism:
        print(f"FAIL: {wmism} wide mismatches  {per_class}", file=sys.stderr)
        return 1

    print(f"WIDE TABLE EXACT: {wcount} codepoints, {wide_leaves} leaves, "
          f"{wide_total} bytes, 0 mismatches  (~{wide_total/1024:.1f} KB)")
    print(f"  per-class mismatches: {per_class}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
