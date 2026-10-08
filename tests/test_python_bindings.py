#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 the cuTokenize contributors
"""Bit-exactness tests: gpu_bpe_tokenizer vs tiktoken and HF tokenizers."""

from __future__ import annotations

import os
import sys
import json

import numpy as np

import gpu_bpe_tokenizer as gbpe

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CORPUS = os.path.join(REPO, "data/corpus/corpus_100k_tokens.txt")


def test_gpt2() -> None:
    import tiktoken

    text = open(CORPUS).read()
    g = gbpe.Tokenizer(os.path.join(REPO, "data/hf_gpt2_tokenizer.json"))
    t = tiktoken.get_encoding("gpt2")

    gpu_ids = g.encode(text)
    ref_ids = t.encode(text)

    print(f"  [gpt2] gpu_len={len(gpu_ids)} ref_len={len(ref_ids)} "
          f"vocab_size={g.vocab_size} regex={g.regex_kind}")
    assert gpu_ids == ref_ids, "GPT-2 divergence"

    special_text = "left  <|endoftext|>'s right<|endoftext|>"
    assert g.encode(special_text) == t.encode(
        special_text, allowed_special={"<|endoftext|>"}), \
        "GPT-2 GPU AddedVocabulary divergence"

    # encode_numpy parity
    arr = g.encode_numpy("hello world")
    assert isinstance(arr, np.ndarray)
    assert arr.dtype == np.uint32
    assert arr.tolist() == g.encode("hello world")
    print("  [gpt2] PASS")


def test_llama3() -> None:
    from tokenizers import Tokenizer as HFTok

    text = open(CORPUS).read()
    vocab = os.path.join(REPO, "data/llama3_tokenizer.json")
    g = gbpe.Tokenizer(vocab)
    t = HFTok.from_file(vocab)

    gpu_ids = g.encode(text)
    ref_ids = t.encode(text, add_special_tokens=False).ids

    print(f"  [llama3] gpu_len={len(gpu_ids)} ref_len={len(ref_ids)} "
          f"vocab_size={g.vocab_size} regex={g.regex_kind}")
    assert gpu_ids == ref_ids, "Llama-3 divergence"
    for long_text in ("a" * 65, "b" * 257, " " * 193):
        assert g.encode(long_text) == t.encode(
            long_text, add_special_tokens=False).ids, "Llama-3 overflow divergence"
    special_text = (
        "left  <|end_of_text|>'s right"
        "<|begin_of_text|><|end_of_text|>tail"
    )
    assert g.encode(special_text) == t.encode(
        special_text, add_special_tokens=False).ids, \
        "Llama-3 GPU AddedVocabulary divergence"
    print("  [llama3] PASS")


def test_deepseek_v3() -> None:
    from tokenizers import Tokenizer as HFTok

    vocab = os.path.join(REPO, "data/vocabs/deepseek_v3.json")
    if not os.path.exists(vocab):
        import pytest
        pytest.skip("deepseek_v3 vocab not present")
    text = open(CORPUS).read()
    g = gbpe.Tokenizer(vocab)
    t = HFTok.from_file(vocab)
    assert g.encode(text) == t.encode(text, add_special_tokens=False).ids, "DeepSeek-V3 divergence"

    # Mixed short-CJK (each pre-token <=64B): the GPU pretok must handle CJK.
    cjk = "中文 cd 世界 123 hello"
    assert g.encode(cjk) == t.encode(cjk, add_special_tokens=False).ids, "DeepSeek-V3 CJK divergence"
    for long_text in ("a" * 129, " " * 129):
        assert g.encode(long_text) == t.encode(
            long_text, add_special_tokens=False).ids, "DeepSeek-V3 overflow divergence"
    special_text = (
        "left  <｜end▁of▁sentence｜>'s right"
        "<｜begin▁of▁sentence｜><｜end▁of▁sentence｜>tail"
    )
    assert g.encode(special_text) == t.encode(
        special_text, add_special_tokens=False).ids, \
        "DeepSeek-V3 GPU AddedVocabulary divergence"
    print("  [deepseek_v3] PASS")


def test_gemma3() -> None:
    """Both Gemma presets must remain bit-exact to the HF tokenizer."""
    import pytest
    from tokenizers import Tokenizer as HFTok

    vocab = os.path.join(REPO, "data/vocabs/gemma3.json")
    if not os.path.exists(vocab):
        pytest.skip("gemma3 vocab not present")

    g = gbpe.Tokenizer(vocab)
    t = HFTok.from_file(vocab)
    texts = [
        "",
        "hello world",
        " leading  and trailing spaces ",
        "literal ▁ and ▁▁ repeated metaspace",
        "héllo 世界 — Καλημέρα 👋",
        # The GPU path decomposes ordinary Gemma BPE at merge-table-proven
        # space/newline/tab components. Preserve the sole crossing-space merge
        # (`>` + `▁</`) and raw-U+2581 adjacency while exercising those cuts.
        "alpha beta  gamma\n\ndelta\t\tepsilon",
        "> </tag> but >  </tag> and ▁ next",
        # Gemma AddedVocabulary extraction precedes normalization and BPE.
        # Exercise compact overlapping run entries plus direct-only literals,
        # including adjacency to ordinary spans and to each other.
        "\n" * 63 + "x" + "\t" * 35 + "▁" * 33,
        "a<strong>bold</strong><html><body>x</body></html>z",
        "<unused123>plain<start_of_turn>model<end_of_turn>",
    ]
    if os.path.exists(CORPUS):
        texts.append(open(CORPUS).read())

    for text in texts:
        gpu_ids = g.encode(text)
        ref_ids = t.encode(text, add_special_tokens=False).ids
        assert gpu_ids == ref_ids, f"Gemma 3 divergence on {text[:80]!r}"

    with open(vocab) as f:
        model_vocab = json.load(f)["model"]["vocab"]

    overflow = "a" * 96
    if g.gemma_gpu_pretok:
        # The GPU path accepts raw bytes. Every malformed standalone byte must
        # survive as its corresponding Gemma <0xXX> fallback token rather than
        # being dropped or reinterpreted as a Unicode scalar.
        for byte in (0x80, 0xBF, 0xC0, 0xC1, 0xC2, 0xE2, 0xF0, 0xF5, 0xFF):
            assert g.encode(bytes([byte])) == [model_vocab[f"<0x{byte:02X}>"]]

        assert g.encode(overflow) == t.encode(
            overflow, add_special_tokens=False).ids
    else:
        assert g.encode(overflow) == t.encode(
            overflow, add_special_tokens=False).ids

    # Decoder order is Replace(U+2581 -> " ") then ByteFallback then Fuse.
    # The GPU gather receives literal vocabulary spellings such as "<0x80>",
    # so this confirms that the ID-aware host post-step restores bytes and the
    # str API uses HF's replacement behavior for invalid UTF-8.
    fallback = lambda byte: model_vocab[f"<0x{byte:02X}>"]
    fallback_cases = [
        ([fallback(0x00)], b"\x00"),
        ([fallback(0x0D)], b"\x0D"),
        ([fallback(0xC2), fallback(0xA2)], b"\xC2\xA2"),
        ([fallback(0xE2), fallback(0x82), fallback(0xAC)], b"\xE2\x82\xAC"),
        ([fallback(0x80)], b"\x80"),
    ]
    for ids, raw_bytes in fallback_cases:
        assert g.decode_bytes(ids) == raw_bytes
        assert g.decode(ids) == t.decode(ids, skip_special_tokens=False)
    malformed_run = bytes((0x97, 0xC8, 0x0F, 0xE0, 0xDE, 0x00, 0xB6))
    malformed_ids = [fallback(byte) for byte in malformed_run]
    assert g.decode_bytes(malformed_ids) == malformed_run
    assert g.decode(malformed_ids) == t.decode(
        malformed_ids, skip_special_tokens=False)
    ordinary = t.encode("a b", add_special_tokens=False).ids
    mixed = ordinary + [fallback(0xC2), fallback(0xA2)] + ordinary
    assert g.decode_bytes(mixed) == b"a b\xC2\xA2a b"
    assert g.decode(mixed) == t.decode(mixed, skip_special_tokens=False)

    assert g.regex_kind == "Gemma3"
    print("  [gemma3] PASS")


def test_short_inputs() -> None:
    """Short inputs and reuse don't break the graph or produce wrong output."""
    import tiktoken

    g = gbpe.Tokenizer(os.path.join(REPO, "data/hf_gpt2_tokenizer.json"))
    t = tiktoken.get_encoding("gpt2")
    for s in ["hello world", "", "a", "The quick brown fox", "abc def ghi 123"]:
        assert g.encode(s) == t.encode(s), f"divergence on {s!r}"
    print("  [short] PASS")


def test_long_pretokens() -> None:
    """>64-byte pre-tokens use the graph-resident generic BPE path."""
    import tiktoken

    g = gbpe.Tokenizer(os.path.join(REPO, "data/hf_gpt2_tokenizer.json"))
    t = tiktoken.get_encoding("gpt2")
    texts = [
        "a" * 65,
        "b" * 96,
        "c" * 257,
        "d" * 1024,
        "g" * 1025,
        " " * 257,
        "prefix " + "e" * 193 + " suffix",
        "a" * 129 + " " + "b" * 257,
    ]
    for text in texts:
        assert g.encode(text) == t.encode(text), \
            f"long-pretoken divergence at {len(text)} bytes"

    # Replay the same graph with changing overflow counts to catch stale
    # descriptors/counts leaking between calls.
    assert g.encode("short again") == t.encode("short again")
    assert g.encode("f" * 129) == t.encode("f" * 129)
    print("  [long-pretokens] PASS")


def test_lazy_growth() -> None:
    """Encoding inputs much larger than the initial cap triggers re-allocation."""
    import tiktoken

    g = gbpe.Tokenizer(os.path.join(REPO, "data/hf_gpt2_tokenizer.json"),
                       max_input_chars=512)
    t = tiktoken.get_encoding("gpt2")
    text = open(CORPUS).read()  # >> 512 chars
    assert g.encode(text) == t.encode(text), "lazy growth diverged"
    # Same instance, second call.
    assert g.encode("hello again") == t.encode("hello again")
    print("  [lazy-grow] PASS")


def main() -> int:
    test_gpt2()
    test_llama3()
    test_deepseek_v3()
    test_gemma3()
    test_short_inputs()
    test_long_pretokens()
    test_lazy_growth()
    print("\nALL PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
