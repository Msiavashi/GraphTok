"""A Tokenizer's encode cost must track the CURRENT input, not its largest ever.

Every kernel node in the captured graph is sized from the owning context's
capacity, not from the input: the pre-tokenizer stages launch one thread per
capacity byte and the CUB scans sweep every capacity element with no length
early-out. A single grow-only context therefore made each later small encode
pay for the biggest input the object had ever seen -- measured at 4.7x on a
16 KiB encode after one 4 MiB encode.

The fix is a ladder of per-size-class contexts, each with its own captured
graph, of which an encode replays the class matching its own size. These tests
pin both halves of that contract: the right class is selected (cost), and the
token IDs never depend on which class served them (exactness).
"""
from __future__ import annotations

from pathlib import Path

import pytest

import gpu_bpe_tokenizer as gbpe

REPO = Path(__file__).resolve().parent.parent
VOCAB = REPO / "data" / "llama3_tokenizer.json"
CORPUS = REPO / "data" / "corpus" / "corpus_code_100k_tokens.txt"

KIB = 1024


@pytest.fixture(scope="module")
def raw() -> bytes:
    if not CORPUS.exists() or not VOCAB.exists():
        pytest.skip("corpus or vocab not present")
    return CORPUS.read_bytes()


def chunk(raw: bytes, nbytes: int) -> str:
    return (raw * (nbytes // len(raw) + 1))[:nbytes].decode("utf-8", "ignore")


def make_tok() -> gbpe.Tokenizer:
    # host_max_bytes=0 forces the GPU path so the graph ladder is exercised.
    return gbpe.Tokenizer(str(VOCAB), host_max_bytes=0)


def test_small_encode_after_large_uses_a_small_graph(raw: bytes) -> None:
    """The defect itself: a 16 KiB encode after a 4 MiB one must not replay the
    4 MiB graph."""
    tok = make_tok()
    small = chunk(raw, 16 * KIB)

    tok.encode_numpy(small)
    fresh_cap = tok.active_capacity

    tok.encode_numpy(chunk(raw, 4 * KIB * KIB))
    assert tok.active_capacity > fresh_cap        # the big one got a big graph

    tok.encode_numpy(small)
    assert tok.active_capacity == fresh_cap, (
        "a small encode after a large one replayed the large graph: "
        f"{tok.active_capacity} != {fresh_cap}"
    )


def test_each_size_class_is_captured_once(raw: bytes) -> None:
    """Repeating a size must not add graphs, and revisiting a class must reuse
    the graph already captured for it."""
    tok = make_tok()
    sizes = [8 * KIB, 64 * KIB, 8 * KIB, 256 * KIB, 64 * KIB, 8 * KIB]
    for s in sizes:
        tok.encode_numpy(chunk(raw, s))
    caps = list(tok.graph_capacities)
    assert caps == sorted(caps), "ladder is not ordered by capacity"
    assert len(caps) == len(set(caps)), "a size class was captured twice"
    # ctor class + the three distinct sizes above.
    assert len(caps) <= 4

    before = list(tok.graph_capacities)
    for s in sizes:
        tok.encode_numpy(chunk(raw, s))
    assert list(tok.graph_capacities) == before, "replaying captured classes grew the ladder"


def test_ladder_is_bounded(raw: bytes) -> None:
    """Capturing a graph is expensive, so the number of retained classes is
    capped even under a stream of many distinct sizes."""
    tok = make_tok()
    for s in (2 * KIB, 8 * KIB, 32 * KIB, 128 * KIB, 512 * KIB,
              KIB * KIB, 2 * KIB * KIB, 4 * KIB * KIB, 3 * KIB * KIB,
              700 * KIB, 5 * KIB, 100 * KIB):
        tok.encode_numpy(chunk(raw, s))
    assert len(tok.graph_capacities) <= 8


def test_decode_works_after_switching_back_to_an_older_class(raw: bytes) -> None:
    """Decode workspaces live in the per-class context, so they can lag the
    ladder-wide high-water marks. A decode issued after an encode switched back
    to a class built before any decode ran must still work."""
    tok = make_tok()
    big = chunk(raw, 512 * KIB)
    small = chunk(raw, 4 * KIB)

    # Build the big class and decode on it: that grows the ladder-wide decode
    # caps but only rebuilds the big class's context.
    big_ids = tok.encode_numpy(big)
    assert tok.decode(big_ids.tolist()) == big

    # Switch to a different (smaller) class, whose context was built before the
    # decode caps existed, and decode there.
    small_ids = tok.encode_numpy(small)
    assert tok.decode(small_ids.tolist()) == small

    # And back to the big one, to confirm neither rebuild clobbered the other.
    assert tok.decode(tok.encode_numpy(big).tolist()) == big


@pytest.mark.parametrize("size", [4 * KIB, 16 * KIB, 100 * KIB, 300 * KIB])
def test_ids_are_independent_of_the_serving_class(raw: bytes, size: int) -> None:
    """Bit-exactness gate: the IDs an input encodes to must be identical whether
    it is served by its own class, by a fresh object, or by an object whose
    ladder was built in an adversarial order."""
    text = chunk(raw, size)

    fresh = make_tok()
    expected = fresh.encode_numpy(text).tolist()
    del fresh

    tok = make_tok()
    # Big first (so an oversized class exists), then small, then back up.
    for warm in (4 * KIB * KIB, KIB * KIB, 8 * KIB, 512 * KIB):
        tok.encode_numpy(chunk(raw, warm))
        assert tok.encode_numpy(text).tolist() == expected
