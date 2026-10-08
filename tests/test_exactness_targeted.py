#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 the cuTokenize contributors
"""Targeted, reference-backed exactness cases for GPU BPE boundaries.

These cases complement the corpus gate. They deliberately exercise the
short/long initial-symbol cutovers, left-most equal-rank selection, and
replayed contexts whose overflow and storage requirements change between
calls. Every assertion is against tiktoken (GPT-2) or Hugging Face
``tokenizers`` (the other families), not another cuTokenize path.
"""

from __future__ import annotations

import json
from functools import lru_cache
from itertools import product
from pathlib import Path

import pytest

import gpu_bpe_tokenizer as gbpe


REPO = Path(__file__).resolve().parent.parent
VOCAB_FILES = {
    "gpt2": REPO / "data" / "hf_gpt2_tokenizer.json",
    "llama3": REPO / "data" / "llama3_tokenizer.json",
    "qwen25": REPO / "data" / "vocabs" / "qwen25.json",
    "deepseek_v3": REPO / "data" / "vocabs" / "deepseek_v3.json",
    "gemma3": REPO / "data" / "vocabs" / "gemma3.json",
}

# The ``all`` CMake preset represents the four paper configurations. Qwen is
# included too: its standalone preset and a wheel compiled with Qwen receive
# the same checks, while the default all wheel reports its skip explicitly.
TARGETED_VOCABS = tuple(VOCAB_FILES)
BYTE_LEVEL_VOCABS = ("gpt2", "llama3", "qwen25", "deepseek_v3")
INITIAL_SYMBOL_COUNTS = (31, 32, 33, 63, 64, 65)


@lru_cache(maxsize=None)
def reference_encoder(vocab: str):
    if vocab == "gpt2":
        import tiktoken

        encoding = tiktoken.get_encoding("gpt2")
        return lambda text: encoding.encode(text, allowed_special="all")

    from tokenizers import Tokenizer as HFTok

    encoding = HFTok.from_file(str(VOCAB_FILES[vocab]))
    return lambda text: encoding.encode(text, add_special_tokens=False).ids


def tokenizer(vocab: str, *, max_input_chars: int = 256):
    path = VOCAB_FILES[vocab]
    if not path.is_file():
        pytest.skip(f"{vocab} tokenizer asset is not present")
    try:
        return gbpe.Tokenizer(str(path), max_input_chars=max_input_chars)
    except RuntimeError as error:
        pytest.skip(f"{vocab} is not compiled into this Python extension: {error}")


def assert_exact(gpu_tokenizer, vocab: str, text: str, label: str) -> None:
    observed = gpu_tokenizer.encode(text)
    expected = reference_encoder(vocab)(text)
    assert observed == expected, f"[{vocab}] {label} diverged"


def initial_symbol_text(count: int) -> str:
    """Return one ASCII-letter pre-token with exactly ``count`` input symbols."""
    alphabet = "qzxjvkwymfpbghdcrltns"
    return (alphabet * ((count + len(alphabet) - 1) // len(alphabet)))[:count]


@pytest.mark.parametrize(
    ("vocab", "initial_symbols"),
    tuple(product(TARGETED_VOCABS, INITIAL_SYMBOL_COUNTS)),
    ids=lambda value: str(value),
)
def test_initial_symbol_boundaries(vocab: str, initial_symbols: int) -> None:
    """31/32/33 and 63/64/65 initial-symbol paths match the external oracle."""
    g = tokenizer(vocab)
    text = initial_symbol_text(initial_symbols)
    assert len(text) == initial_symbols
    assert_exact(g, vocab, text, f"{initial_symbols} initial symbols")


def test_gemma_raw_lowbar_added_vocabulary_runs() -> None:
    """Gemma greedily extracts raw U+2581 AddedVocabulary runs on the GPU.

    Test every short table entry, the one-scalar remainder after a 31-symbol
    chunk, and several multi-chunk cases in one reusable context.
    """
    g = tokenizer("gemma3")
    for count in range(1, 129):
        assert_exact(g, "gemma3", "▁" * count,
                     f"raw U+2581 AddedVocabulary run of {count} symbols")


@pytest.mark.parametrize("vocab", BYTE_LEVEL_VOCABS)
def test_equal_rank_leftmost_merge_selection(vocab: str) -> None:
    """Repeated ``a a`` candidates require the BPE tie-breaker to pick leftmost.

    At the first BPE iteration every adjacent pair in this pre-token is the
    same ``a,a`` merge and therefore has the same rank. An odd symbol count
    makes an incorrect first position observably affect later overlapping
    merges. The raw text is confirmed not to be an ``ignore_merges`` direct
    vocabulary entry, so the merge kernel is necessarily exercised.
    """
    path = VOCAB_FILES[vocab]
    model = json.loads(path.read_text(encoding="utf-8"))["model"]
    merges = model.get("merges", [])
    merge_strings = {
        item if isinstance(item, str) else " ".join(item)
        for item in merges
    }
    assert "a a" in merge_strings, f"{vocab} asset no longer has the a,a merge witness"

    text = "a" * 33
    assert text not in model["vocab"], "witness must not bypass BPE via ignore_merges"
    assert_exact(tokenizer(vocab), vocab, text, "equal-rank leftmost a,a merge")


def overflow_text(count: int) -> str:
    """``count`` separately pre-tokenized runs, each beyond the 64-symbol path."""
    return " ".join(initial_symbol_text(65) for _ in range(count))


@pytest.mark.parametrize("vocab", TARGETED_VOCABS)
def test_changing_overflow_counts_and_capacity_growth(vocab: str) -> None:
    """One reusable context sees 0 -> 1 -> 3 -> 2 -> 0 -> 4 long runs.

    Starting below the first requested size forces allocation growth. The
    decreasing steps then catch stale overflow descriptors or graph inputs
    left behind by the previous invocation.
    """
    g = tokenizer(vocab, max_input_chars=32)
    calls = [
        ("short-31", initial_symbol_text(31)),
        ("overflow-1", overflow_text(1)),
        ("overflow-3", overflow_text(3)),
        ("overflow-2", overflow_text(2)),
        ("short-33", initial_symbol_text(33)),
        ("overflow-4", overflow_text(4)),
        ("short-replay", "short again"),
    ]
    for label, text in calls:
        assert_exact(g, vocab, text, label)
