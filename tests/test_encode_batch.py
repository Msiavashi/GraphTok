#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 the GraphTok contributors
"""Bit-exactness tests for the batched encode path (`encode_batch`).

The contract under test: `encode_batch(docs)` must return exactly what you get
by encoding every document on its own and concatenating the IDs, with
`offsets` giving the per-document slice boundaries. That has to hold even when
the seam between two documents falls in the middle of something the
pre-tokenizer would otherwise have merged (a digit run, a whitespace run, a
contraction, an added-token literal) -- see `tests/seam_corpus.py`.

Every case is additionally checked against the reference tokenizer (tiktoken
for GPT-2, HuggingFace `tokenizers` for the rest) per document, so a bug that
corrupted BOTH the batched and the single-document path would still be caught.

Vocabs not compiled into the installed wheel are skipped. Qwen-2.5 is normally
absent (the `all` preset excludes it: host-side NFC), and Gemma 3 is SP-family
so `encode_batch` raises there by design -- that's asserted, not skipped.
"""

from __future__ import annotations

import gc
import os
import sys

import numpy as np
import pytest

import gpu_bpe_tokenizer as gbpe

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CORPUS = os.path.join(REPO, "data/corpus/corpus_100k_tokens.txt")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from seam_corpus import SEAM_CASES  # noqa: E402

VOCAB_FILES = {
    "gpt2": "data/hf_gpt2_tokenizer.json",
    "llama3": "data/llama3_tokenizer.json",
    "qwen25": "data/vocabs/qwen25.json",
    "deepseek_v3": "data/vocabs/deepseek_v3.json",
}
BYTE_LEVEL_VOCABS = list(VOCAB_FILES)


# ---------------------------------------------------------------------------
# fixtures / helpers
# ---------------------------------------------------------------------------

_TOK_CACHE: dict[str, object] = {}
_REF_CACHE: dict[str, object] = {}


def _tokenizer(name: str):
    """Load (once) the GPU tokenizer, or skip if this wheel lacks the vocab."""
    if name not in _TOK_CACHE:
        path = os.path.join(REPO, VOCAB_FILES[name])
        if not os.path.exists(path):
            pytest.skip(f"{name} vocab file not present")
        try:
            _TOK_CACHE[name] = gbpe.Tokenizer(path)
        except RuntimeError as exc:  # vocab not compiled into this build
            pytest.skip(f"{name} not available in this build: {exc}")
    return _TOK_CACHE[name]


def _reference(name: str):
    """The oracle: tiktoken for GPT-2, HF `tokenizers` otherwise."""
    if name not in _REF_CACHE:
        if name == "gpt2":
            import tiktoken

            enc = tiktoken.get_encoding("gpt2")
            _REF_CACHE[name] = lambda s: enc.encode(
                s, allowed_special={"<|endoftext|>"})
        else:
            from tokenizers import Tokenizer as HFTok

            enc = HFTok.from_file(os.path.join(REPO, VOCAB_FILES[name]))
            _REF_CACHE[name] = (
                lambda s: enc.encode(s, add_special_tokens=False).ids)
    return _REF_CACHE[name]


@pytest.fixture(scope="module", params=BYTE_LEVEL_VOCABS)
def vocab(request) -> str:
    return request.param


def check_batch(name: str, docs: list[str], *, ref: bool = True) -> None:
    """Assert encode_batch(docs) == concat(encode(d)) == concat(reference(d))."""
    g = _tokenizer(name)
    tokens, offsets = g.encode_batch(docs)

    assert isinstance(tokens, np.ndarray) and tokens.dtype == np.uint32
    assert isinstance(offsets, np.ndarray) and offsets.dtype == np.uint32
    assert len(offsets) == len(docs) + 1
    assert offsets[0] == 0
    assert offsets[-1] == len(tokens)
    assert list(offsets) == sorted(offsets), "offsets must be non-decreasing"

    reference = _reference(name) if ref else None
    for i, doc in enumerate(docs):
        got = tokens[offsets[i]:offsets[i + 1]].tolist()
        single = g.encode(doc)
        assert got == single, (
            f"[{name}] batch/single divergence on doc {i} {doc[:60]!r}: "
            f"{got[:16]} vs {single[:16]}")
        if reference is not None:
            assert got == reference(doc), (
                f"[{name}] batch/reference divergence on doc {i} {doc[:60]!r}")


# ---------------------------------------------------------------------------
# 1. adversarial seams (the whole point of the feature)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("case", SEAM_CASES, ids=[c[0] for c in SEAM_CASES])
def test_seam_cases(vocab: str, case) -> None:
    _name, docs = case
    check_batch(vocab, list(docs))


# ---------------------------------------------------------------------------
# 2. structural edge cases
# ---------------------------------------------------------------------------

def test_single_doc_matches_encode(vocab: str) -> None:
    g = _tokenizer(vocab)
    for text in ("hello world", "a", "12345", " ", "it's 42\n\nnext"):
        tokens, offsets = g.encode_batch([text])
        assert list(offsets) == [0, len(tokens)]
        assert tokens.tolist() == g.encode(text)


def test_empty_list(vocab: str) -> None:
    g = _tokenizer(vocab)
    tokens, offsets = g.encode_batch([])
    assert tokens.tolist() == []
    assert offsets.tolist() == [0]


def test_empty_docs_positions(vocab: str) -> None:
    check_batch(vocab, ["", "alpha beta", "gamma"])          # leading
    check_batch(vocab, ["alpha beta", "", "gamma"])          # interior
    check_batch(vocab, ["alpha beta", "gamma", ""])          # trailing
    check_batch(vocab, ["", "", ""])                         # all empty
    check_batch(vocab, ["", "", "x", "", "", "y", ""])       # runs of empties


def test_one_byte_docs(vocab: str) -> None:
    check_batch(vocab, list("abc 123!'\n\t"))


def test_all_empty_offsets_are_all_zero(vocab: str) -> None:
    g = _tokenizer(vocab)
    tokens, offsets = g.encode_batch(["", "", ""])
    assert tokens.tolist() == []
    assert offsets.tolist() == [0, 0, 0, 0]


# ---------------------------------------------------------------------------
# 3. capacity growth (cap_docs pow2 growth, then shrink on the same instance)
# ---------------------------------------------------------------------------

def test_cap_docs_growth_then_shrink(vocab: str) -> None:
    _tokenizer(vocab)  # Load once so every call below reuses the same context.
    small = ["seed one", "seed two"]
    check_batch(vocab, small)

    # Exercise every allocation transition in the batch-document descriptor:
    # cap_docs is floored at 16 and then grows 16 -> 32 -> 64 -> 128.  The
    # values immediately either side of each boundary are intentional; a
    # single 40/70-doc probe can miss stale descriptors at a transition.
    for count in (16, 17, 32, 33, 64, 65):
        docs = [
            f"document number {i} with 12{i} digits and it's text"
            for i in range(count)
        ]
        check_batch(vocab, docs)

    # Back down on the SAME instance: stale per-document state from the large
    # batch must not leak into the small one.
    check_batch(vocab, small)
    check_batch(vocab, ["only one"])


def test_large_docs_and_growth(vocab: str) -> None:
    g = _tokenizer(vocab)
    if not os.path.exists(CORPUS):
        pytest.skip("corpus not present")
    text = open(CORPUS).read()
    chunk = len(text) // 8
    docs = [text[i * chunk:(i + 1) * chunk] for i in range(8)]
    tokens, offsets = g.encode_batch(docs)
    for i, doc in enumerate(docs):
        assert tokens[offsets[i]:offsets[i + 1]].tolist() == g.encode(doc), \
            f"[{vocab}] corpus-chunk divergence on doc {i}"


# ---------------------------------------------------------------------------
# 4. graph-only benchmark surface for changing batches
# ---------------------------------------------------------------------------

def test_bench_batch_graph_api_replays_staged_batch(vocab: str) -> None:
    """The benchmark path accepts a real multi-document descriptor, not text."""
    g = _tokenizer(vocab)
    docs = ["alpha 123", "beta's gamma", "", "δelta 456"]
    check_batch(vocab, docs)

    g.bench_stage_batch(docs)
    assert g.bench_run_batch() > 0.0
    tokens, offsets = g.bench_readback_batch()
    expected_tokens, expected_offsets = g.encode_batch(docs)
    assert tokens.tolist() == expected_tokens.tolist()
    assert offsets.tolist() == expected_offsets.tolist()


# ---------------------------------------------------------------------------
# 5. device-tensor variant
# ---------------------------------------------------------------------------

def test_encode_batch_to_device_matches_host(vocab: str) -> None:
    torch = pytest.importorskip("torch")
    g = _tokenizer(vocab)
    docs = ["12", "345 it'", "s ok", "", "6789", "hello  world\n"]

    host_tokens, host_offsets = g.encode_batch(docs)
    dev_tokens, dev_offsets = g.encode_batch_to_device(docs)

    assert dev_tokens.is_cuda and dev_tokens.dtype == torch.int32
    assert list(dev_offsets) == list(host_offsets)
    assert dev_tokens.cpu().numpy().astype(np.uint32).tolist() == host_tokens.tolist()
    for i in range(len(docs)):
        sl = dev_tokens[dev_offsets[i]:dev_offsets[i + 1]]
        assert sl.cpu().numpy().astype(np.uint32).tolist() == g.encode(docs[i])


def test_encode_batch_to_device_empty_raises(vocab: str) -> None:
    pytest.importorskip("torch")
    g = _tokenizer(vocab)
    with pytest.raises(Exception):
        g.encode_batch_to_device([])


def test_device_tensor_outlives_tokenizer(vocab: str) -> None:
    """The returned tensor keeps its owning Tokenizer alive (Task 5 fix)."""
    pytest.importorskip("torch")
    path = os.path.join(REPO, VOCAB_FILES[vocab])
    _tokenizer(vocab)  # skip early if this vocab is unavailable

    throwaway = gbpe.Tokenizer(path)
    docs = ["alpha 123", "beta's gamma", ""]
    expected = [throwaway.encode(d) for d in docs]
    tokens, offsets = throwaway.encode_batch_to_device(docs)

    del throwaway
    gc.collect()

    # Must not be a use-after-free: the buffer is still owned and readable.
    got = tokens.cpu().numpy().astype(np.uint32)
    for i, exp in enumerate(expected):
        assert got[offsets[i]:offsets[i + 1]].tolist() == exp


# ---------------------------------------------------------------------------
# 6. SP family is rejected, not silently wrong
# ---------------------------------------------------------------------------

def test_gemma3_batch_rejected() -> None:
    path = os.path.join(REPO, "data/vocabs/gemma3.json")
    if not os.path.exists(path):
        pytest.skip("gemma3 vocab not present")
    try:
        g = gbpe.Tokenizer(path)
    except RuntimeError as exc:
        pytest.skip(f"gemma3 not available in this build: {exc}")
    with pytest.raises(RuntimeError):
        g.encode_batch(["a", "b"])
    with pytest.raises(RuntimeError):
        g.encode_batch(["only one"])
    # Single-document encode still works on the SP path.
    assert g.encode("hello world")


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
