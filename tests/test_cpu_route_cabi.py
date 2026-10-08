"""libgtok.so's CPU route (gtok_encode / gtok_encode_batch below the
threshold) must return the same token IDs as its GPU route and as HF.

Needs a built libgtok.so: GTOK_LIB=<path> [GTOK_VOCAB=<tokenizer.json>]
pytest tests/test_cpu_route_cabi.py. The vocabulary must be one the library
was built for (default: Llama-3).
"""
from __future__ import annotations

import ctypes
import os
import random
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
LIB = os.environ.get("GTOK_LIB")
VOCAB = os.environ.get("GTOK_VOCAB", str(REPO / "data" / "vocabs" / "llama3_tokenizer.json"))
CORPUS = REPO / "data" / "corpus" / "corpus_chat_multilingual_100k_tokens.txt"

pytestmark = pytest.mark.skipif(not LIB, reason="set GTOK_LIB to a built libgtok.so")


@pytest.fixture(scope="module")
def lib():
    L = ctypes.CDLL(LIB)
    L.gtok_create.restype = ctypes.c_void_p
    L.gtok_create.argtypes = [ctypes.c_char_p, ctypes.c_uint32]
    L.gtok_encode.restype = ctypes.c_int32
    L.gtok_encode.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_uint32,
                              ctypes.POINTER(ctypes.c_uint32), ctypes.c_uint32]
    L.gtok_encode_batch.restype = ctypes.c_int32
    L.gtok_encode_batch.argtypes = [ctypes.c_void_p, ctypes.c_char_p,
                                    ctypes.POINTER(ctypes.c_uint32), ctypes.c_uint32,
                                    ctypes.POINTER(ctypes.c_uint32), ctypes.c_uint32,
                                    ctypes.POINTER(ctypes.c_uint32)]
    L.gtok_cpu_backend.restype = ctypes.c_char_p
    L.gtok_cpu_backend.argtypes = [ctypes.c_void_p]
    L.gtok_cpu_max_bytes.restype = ctypes.c_uint32
    L.gtok_cpu_max_bytes.argtypes = [ctypes.c_void_p]
    L.gtok_set_cpu_max_bytes.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
    L.gtok_destroy.argtypes = [ctypes.c_void_p]
    return L


@pytest.fixture(scope="module")
def handle(lib):
    h = lib.gtok_create(VOCAB.encode(), 1 << 22)
    assert h, "gtok_create failed"
    yield h
    lib.gtok_destroy(h)


@pytest.fixture(scope="module")
def ref():
    from tokenizers import Tokenizer
    hf = Tokenizer.from_file(VOCAB)
    return lambda t: hf.encode(t, add_special_tokens=False).ids


def encode(lib, h, text: str) -> list[int]:
    b = text.encode()
    cap = len(b) + 16
    out = (ctypes.c_uint32 * cap)()
    n = lib.gtok_encode(h, b, len(b), out, cap)
    assert n >= 0, n
    return list(out[:n])


def encode_batch(lib, h, docs: list[str]) -> list[list[int]]:
    bs = [d.encode() for d in docs]
    blob = b"".join(bs)
    offs = [0]
    for b in bs:
        offs.append(offs[-1] + len(b))
    cap = len(blob) + 16 * len(docs) + 16
    out = (ctypes.c_uint32 * cap)()
    tok_offs = (ctypes.c_uint32 * (len(docs) + 1))()
    n = lib.gtok_encode_batch(h, blob, (ctypes.c_uint32 * len(offs))(*offs), len(docs),
                              out, cap, tok_offs)
    assert n >= 0, n
    return [list(out[tok_offs[i]:tok_offs[i + 1]]) for i in range(len(docs))]


def test_backend_reported(lib, handle):
    assert lib.gtok_cpu_backend(handle).decode() in ("gigatoken", "host", "off")


def test_cpu_and_gpu_routes_match_reference(lib, handle, ref):
    text = CORPUS.read_text(encoding="utf-8") if CORPUS.is_file() else "hello world " * 5000
    rng = random.Random(3)
    cases = [text[a:a + rng.choice((1, 17, 300, 2000, 9000, 40000))]
             for a in rng.sample(range(len(text) - 41000), 30)]
    cases += ["", " ", "don't 123456 \r\n\t x}\nfoo", "é café"]
    default = lib.gtok_cpu_max_bytes(handle)
    for t in cases:
        want = ref(t)
        lib.gtok_set_cpu_max_bytes(handle, default)
        assert encode(lib, handle, t) == want
        lib.gtok_set_cpu_max_bytes(handle, 0)           # force the GPU route
        assert encode(lib, handle, t) == want
    lib.gtok_set_cpu_max_bytes(handle, default)


def test_batch_routes_match_reference(lib, handle, ref):
    docs = ["short one", "", "multi\nline\r\n doc 42", "café 日本", "x" * 100]
    want = [ref(d) for d in docs]
    default = lib.gtok_cpu_max_bytes(handle)
    assert encode_batch(lib, handle, docs) == want      # CPU route (small batch)
    lib.gtok_set_cpu_max_bytes(handle, 0)
    assert encode_batch(lib, handle, docs) == want      # GPU route
    lib.gtok_set_cpu_max_bytes(handle, default)
