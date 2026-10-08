"""The load-adaptive dispatcher must return the reference token IDs for every
request, from many threads at once, under every policy -- and must actually
combine concurrent GPU requests into shared replays."""
from __future__ import annotations

import random
import threading
from functools import lru_cache
from pathlib import Path

import pytest

import gpu_bpe_tokenizer as gbpe

REPO = Path(__file__).resolve().parent.parent
VOCAB = REPO / "data" / "vocabs" / "llama3_tokenizer.json"
CORPORA = [REPO / "data" / "corpus" / n for n in (
    "corpus_100k_tokens.txt", "corpus_code_100k_tokens.txt",
    "corpus_chat_multilingual_100k_tokens.txt")]


@lru_cache(maxsize=None)
def reference():
    from tokenizers import Tokenizer
    hf = Tokenizer.from_file(str(VOCAB))
    return lambda t: hf.encode(t, add_special_tokens=False).ids


def docs(n: int, seed: int) -> list[str]:
    texts = [p.read_text(encoding="utf-8") for p in CORPORA if p.is_file()]
    if not texts:
        pytest.skip("corpora missing")
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        t = rng.choice(texts)
        size = rng.choice((0, 1, 40, 700, 3000, 9000, 30000, 120000))
        a = rng.randrange(0, max(1, len(t) - size))
        out.append(t[a:a + size])
    out += ["<|begin_of_text|>hi<|eot_id|>", "é x}\nfoo 1234567"]
    return out


def hammer(d, items: list[str], threads: int) -> list:
    results = [None] * len(items)
    idx = iter(range(len(items)))
    lock = threading.Lock()
    start = threading.Barrier(threads)

    def worker():
        start.wait()
        while True:
            with lock:
                i = next(idx, None)
            if i is None:
                return
            results[i] = d.encode_numpy(items[i]).tolist()

    ts = [threading.Thread(target=worker) for _ in range(threads)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    return results


@pytest.mark.parametrize("policy", ("adaptive", "gpu", "cpu"))
def test_concurrent_requests_match_reference(policy: str) -> None:
    d = gbpe.Dispatcher(str(VOCAB), cpu_workers=2, max_batch_bytes=1 << 20,
                        max_batch_docs=64, policy=policy)
    items = docs(300, seed=hash(policy) & 0xFFFF)
    got = hammer(d, items, threads=16)
    ref = reference()
    for i, (text, ids) in enumerate(zip(items, got)):
        assert ids == ref(text), f"request {i} ({len(text)} chars) differs"
    s = d.stats()
    assert s["cpu_requests"] + s["gpu_requests"] + s["gpu_large"] == len(items)
    if policy == "gpu":
        assert s["cpu_requests"] == 0
        # 16 threads against one batcher must share replays
        assert s["gpu_batches"] < s["gpu_requests"]


def test_idle_small_request_uses_cpu_engine() -> None:
    d = gbpe.Dispatcher(str(VOCAB), cpu_workers=1)
    if d.cpu_backend == "off":
        pytest.skip("no CPU engine")
    d.encode_numpy("hello world")
    s = d.stats()
    assert s["cpu_requests"] == 1 and s["gpu_requests"] == 0
