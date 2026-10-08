"""Adversarial seam corpus for the encode_batch pre-tokenization proof.

Each case is a *named batch*: a list of 2+ documents (Python `str`) that, when
concatenated into one buffer and pre-tokenized as a single document (the
current GPU behavior), is expected to produce different pre-token boundaries
than pre-tokenizing each document separately and concatenating the results
(the required `encode_batch` semantics).

Cases here are restricted to `str` because the repo's reference
pre-tokenizers (`ref_pretokenize_gpt2`, `ref_pretokenize_llama3`,
`ref_pretokenize_qwen25`, `ref_pretokenize_deepseek*`) are str-only: they
build codepoint lists via `ord(c) for c in text` and cannot accept malformed
UTF-8 (a Python `str` can never contain a lone/invalid UTF-8 byte sequence —
by construction it's valid Unicode codepoints only).

Malformed / split-multibyte-UTF-8 cases (4-byte emoji split at a byte
boundary that is NOT a codepoint boundary, lone lead bytes, lone
continuation bytes, and the U+017F byte-split variant) are therefore kept
separately in `GPU_ONLY_BYTE_CASES` below as raw `bytes` documents. They
cannot be run through the Python str-based refs at all (encoding a split
would either fail or silently "repair" itself back into valid UTF-8, which
would defeat the point) — they are recorded here as ground truth for the
later GPU-side byte-level difftest (Task 2+), not exercised by
`pretok_batch_difftest.py`.
"""

# ---------------------------------------------------------------------------
# str-only corpus: consumed by the batch tests against all
# four families (gpt2, llama3, qwen25, deepseek). Each entry:
#   (case_name, [doc0, doc1, ...])
# ---------------------------------------------------------------------------
SEAM_CASES = [
    # ---- Digit runs split at seams ----------------------------------------
    ("digits_2_3", ["12", "345"]),
    ("digits_7_2", ["1234567", "89"]),          # Llama-3 3-digit cap boundary; Qwen control (no cap)
    # Seam at 5 digits in (not a multiple of DeepSeek's 3-digit cap, unlike a
    # 6|3 split which coincidentally realigns with the cap and shows SAME).
    ("digits_deepseek_run", ["12345", "67"]),

    # ---- Whitespace ---------------------------------------------------------
    ("ws_hello_world", ["hello ", "world"]),
    ("ws_a_space_b", ["a", " b"]),
    ("ws_long_run_split", ["x" + " " * 20, " " * 20 + "y"]),  # long space run split mid-run
    ("ws_newline_seam", ["x\n", "\ny"]),
    ("ws_crlf_seam", ["--\r", "\n\t\r\n"]),      # CRLF-in-whitespace-run split at seam

    # ---- Contractions ---------------------------------------------------------
    ("contraction_say_s", ["say", "'s"]),
    ("contraction_it_apostrophe_s", ["it'", "s"]),
    ("contraction_bang_s", ["!", "'s"]),
    ("contraction_newline_towhead", ["\n", "'towhead"]),
    ("contraction_upper_S", ["say", "'S"]),      # uppercase variant for llama/qwen (case-sensitive tails)
    ("u017f_whole_in_one_doc", ["long-s ſ whole", " codepoint stays together"]),

    # ---- Added / special tokens (must NOT merge across the seam) ------------
    ("special_endoftext_split", ["<|endo", "ftext|>"]),
    ("special_endoftext_prefix", ["x", "<|endoftext|>y"]),
    ("special_endoftext_exact_doc", ["a", "<|endoftext|>", "b"]),

    # ---- DeepSeek: CJK / non-CJK transitions and punctuation runs -----------
    ("deepseek_cjk_transition", ["アント", "ニ・ガウディ"]),  # アント | ニ・ガウディ
    ("deepseek_punct_run_split", ["!!!", "???"]),
    ("deepseek_ws_gap_run_split", ["a   ", "   b"]),
    ("deepseek_ascii_punct_letter_seam", ["foo.", "bar"]),

    # ---- Structure ------------------------------------------------------------
    ("structure_leading_empty", ["", "abc"]),
    ("structure_interior_empty", ["abc", "", "def"]),
    ("structure_trailing_empty", ["abc", ""]),
    ("structure_all_empty", ["", "", ""]),
    ("structure_one_byte_docs", ["a", "b", "c"]),
    ("structure_single_doc", ["only one document here"]),

    # ---- Multi-doc mixed cases (3+ docs combining several categories) -------
    ("mixed_digits_ws_contraction", ["12", "345 it'", "s ok", "6789"]),
    ("mixed_special_cjk_ws", ["<|endo", "ftext|>アント ", "ニ・ end"]),
    ("mixed_empty_digits_punct", ["", "42", "", "!!! 43", ""]),
]


# ---------------------------------------------------------------------------
# GPU-only byte-level cases: NOT consumable by the str-based Python refs.
# Recorded here as documentation of intent / ground truth targets for the
# later GPU byte-level difftest. Each doc is `bytes`.
# ---------------------------------------------------------------------------
_EMOJI = "\U0001F600".encode("utf-8")  # 4-byte UTF-8 sequence (F0 9F 98 80)
_U017F_UTF8 = "ſ".encode("utf-8")  # 2-byte UTF-8 sequence (C5 BF)

GPU_ONLY_BYTE_CASES = [
    ("byte_emoji_split_1_3", [_EMOJI[:1], _EMOJI[1:]]),
    ("byte_emoji_split_3_1", [_EMOJI[:3], _EMOJI[3:]]),
    ("byte_doc_ends_lone_lead", [b"abc" + _EMOJI[:1], _EMOJI[1:] + b"xyz"]),
    ("byte_doc_starts_lone_continuation", [b"abc", _EMOJI[1:] + b"xyz"]),
    ("byte_u017f_split", [_U017F_UTF8[:1], _U017F_UTF8[1:]]),
]


if __name__ == "__main__":
    print(f"SEAM_CASES: {len(SEAM_CASES)} cases")
    for name, docs in SEAM_CASES:
        print(f"  {name}: {docs!r}")
    print(f"\nGPU_ONLY_BYTE_CASES: {len(GPU_ONLY_BYTE_CASES)} cases (byte-level, not run by pretok_batch_difftest.py)")
    for name, docs in GPU_ONLY_BYTE_CASES:
        print(f"  {name}: {docs!r}")
