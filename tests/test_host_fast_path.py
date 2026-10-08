"""The CPU route must produce exactly the GPU path's token IDs.

Every case is run for each CPU engine (the host encoder and gigatoken) three
ways: forced CPU (host_max_bytes huge), forced GPU (host_max_bytes=0), and the
external reference (tiktoken for GPT-2, HF tokenizers otherwise). All three
must agree. DeepSeek-V3 has no host encoder, so it runs for gigatoken only.
"""
from __future__ import annotations

import random
from functools import lru_cache
from pathlib import Path

import pytest

import gpu_bpe_tokenizer as gbpe

REPO = Path(__file__).resolve().parent.parent
VOCAB_FILES = {
    "gpt2": REPO / "data" / "hf_gpt2_tokenizer.json",
    "llama3": REPO / "data" / "llama3_tokenizer.json",
    "qwen25": REPO / "data" / "vocabs" / "qwen25.json",
    "deepseek_v3": REPO / "data" / "vocabs" / "deepseek_v3.json",
}
BACKENDS = ("host", "gigatoken")
CORPORA = {
    "books": REPO / "data" / "corpus" / "corpus_100k_tokens.txt",
    "code": REPO / "data" / "corpus" / "corpus_code_100k_tokens.txt",
    "multilingual": REPO / "data" / "corpus" / "corpus_chat_multilingual_100k_tokens.txt",
}
SIZES = (0, 1, 2, 3, 7, 16, 64, 128, 300, 1024, 4096, 16384, 65536)
FORCE_HOST = 1 << 30


@lru_cache(maxsize=None)
def pair(vocab: str, backend: str = "host"):
    path = VOCAB_FILES[vocab]
    if not path.is_file():
        pytest.skip(f"{vocab} asset missing")
    try:
        host = gbpe.Tokenizer(str(path), host_max_bytes=FORCE_HOST, cpu_backend=backend)
        gpu = gbpe.Tokenizer(str(path), host_max_bytes=0)
    except RuntimeError as error:
        pytest.skip(f"{vocab}/{backend} unavailable: {error}")
    if host.host_max_bytes == 0 or host.cpu_backend != backend:
        pytest.skip(f"{vocab} has no {backend} CPU route ({host.cpu_backend_note})")
    return host, gpu


@lru_cache(maxsize=None)
def reference(vocab: str):
    if vocab == "gpt2":
        import tiktoken

        enc = tiktoken.get_encoding("gpt2")
        return lambda t: enc.encode(t, allowed_special="all")
    from tokenizers import Tokenizer as HFTok

    hf = HFTok.from_file(str(VOCAB_FILES[vocab]))
    return lambda t: hf.encode(t, add_special_tokens=False).ids


def check(vocab: str, text: str, label: str, backend: str = "host") -> None:
    host, gpu = pair(vocab, backend)
    h = host.encode(text)
    g = gpu.encode(text)
    r = reference(vocab)(text)
    assert h == g, f"[{vocab}] {label}: host != gpu"
    assert h == r, f"[{vocab}] {label}: host != reference"
    assert host.encode_numpy(text).tolist() == h


def corpus_text(name: str) -> str:
    p = CORPORA[name]
    if not p.is_file():
        pytest.skip(f"{name} corpus missing")
    return p.read_text(encoding="utf-8")


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("vocab", tuple(VOCAB_FILES))
@pytest.mark.parametrize("corpus", tuple(CORPORA))
def test_corpus_prefixes(vocab: str, corpus: str, backend: str) -> None:
    text = corpus_text(corpus)
    for size in SIZES:
        check(vocab, text[:size], f"{corpus}[:{size}]", backend)
    # random windows exercise every start context, not just BOS
    rng = random.Random(size_seed := 1234)
    for _ in range(40):
        a = rng.randrange(0, max(1, len(text) - 600))
        check(vocab, text[a:a + rng.randrange(1, 600)], f"{corpus} window@{a}", backend)


SPECIALS = {
    "gpt2": ["<|endoftext|>"],
    "llama3": ["<|begin_of_text|>", "<|end_of_text|>", "<|start_header_id|>",
               "<|end_header_id|>", "<|eot_id|>", "<|reserved_special_token_5|>"],
    "qwen25": ["<|im_start|>", "<|im_end|>", "<|endoftext|>", "<|object_ref_start|>",
               "<tool_call>", "</tool_call>"],
    "deepseek_v3": ["<｜begin▁of▁sentence｜>", "<｜end▁of▁sentence｜>",
                    "<｜User｜>", "<｜Assistant｜>"],
}


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("vocab", tuple(VOCAB_FILES))
def test_added_tokens_and_edges(vocab: str, backend: str) -> None:
    sp = SPECIALS[vocab]
    cases = [
        sp[0],
        f"left {sp[0]} right",
        f"{sp[0]}{sp[-1]}",
        f"no space{sp[0]}glued",
        f"prefix<|not_a_token|>{sp[0]}",
        "<|", "<|end", sp[0][:-1],
        f"{sp[0]}'s it's don't THEY'RE we'll",
        "  leading and trailing  ",
        "\n\n\r\n \t\n",
        "123456789 12 3 4567 12345",
        "é café ſpecial ſ",
        "日本語 한국어 Ελληνικά Русский עברית 🚀",
        "--\r\n\t\r\n foo\r\nbar",
        "'twas 'Twas \"'s\" ('s)  'll'd're",
        "a" * 33, "a" * 65, " " * 70, "1" * 70, "!" * 70,
    ]
    for i, text in enumerate(cases):
        check(vocab, text, f"case{i}", backend)


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("vocab", tuple(VOCAB_FILES))
def test_fuzz(vocab: str, backend: str) -> None:
    sp = SPECIALS[vocab]
    pool = list("abcXYZ 019.,'\"-!?\n\r\t<|>/") + [
        "é", "ſ", "́", "日", "한", "Ω", "🚀", "'s", "'LL", "  ",
        "\r\n", "<|", "|>",
    ] + sp
    rng = random.Random(7)
    for _ in range(1500):
        text = "".join(rng.choice(pool) for _ in range(rng.randrange(0, 48)))
        check(vocab, text, repr(text), backend)


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("vocab", tuple(VOCAB_FILES))
def test_batch_host_matches_gpu(vocab: str, backend: str) -> None:
    host, gpu = pair(vocab, backend)
    text = corpus_text("books")
    rng = random.Random(99)
    docs = [text[a:a + rng.randrange(0, 200)] for a in rng.sample(range(len(text) - 300), 24)]
    docs += ["", SPECIALS[vocab][0], "x", "  ", "é"]
    ht, ho = host.encode_batch(docs)
    gt, go = gpu.encode_batch(docs)
    assert ht.tolist() == gt.tolist() and ho.tolist() == go.tolist()
    for i, d in enumerate(docs):
        assert ht[ho[i]:ho[i + 1]].tolist() == reference(vocab)(d), f"doc {i}"
    # repeated pre-tokens exercise the result cache; results must not change
    rep = ["the cat sat on the mat. " * 5] * 8
    assert host.encode_batch(rep)[0].tolist() == gpu.encode_batch(rep)[0].tolist()
    assert host.encode(rep[0]) == host.encode(rep[0]) == reference(vocab)(rep[0])


def test_routing_threshold() -> None:
    vocab = "llama3"
    host, _ = pair(vocab)
    t = gbpe.Tokenizer(str(VOCAB_FILES[vocab]), host_max_bytes=16)
    assert t.host_max_bytes == 16
    small, big = "hello world", "hello world " * 10
    assert t.encode(small) == host.encode(small)
    assert t.encode(big) == host.encode(big)
    t.host_max_bytes = 0
    assert t.host_max_bytes == 0
    assert t.encode(small) == host.encode(small)


def test_unsupported_family_disables_host() -> None:
    path = REPO / "data" / "vocabs" / "deepseek_v3.json"
    if not path.is_file():
        pytest.skip("deepseek asset missing")
    try:
        t = gbpe.Tokenizer(str(path), host_max_bytes=FORCE_HOST, cpu_backend="host")
    except RuntimeError:
        pytest.skip("deepseek not compiled in")
    assert t.cpu_backend == "off" and t.host_max_bytes == 0


def test_cpu_backend_selection() -> None:
    path = VOCAB_FILES["llama3"]
    off = gbpe.Tokenizer(str(path), host_max_bytes=FORCE_HOST, cpu_backend="off")
    assert off.cpu_backend == "off" and off.host_max_bytes == 0
    host = gbpe.Tokenizer(str(path), host_max_bytes=FORCE_HOST, cpu_backend="host")
    assert host.cpu_backend == "host"
    auto = gbpe.Tokenizer(str(path), host_max_bytes=FORCE_HOST)
    assert auto.cpu_backend in ("gigatoken", "host")
    with pytest.raises(ValueError):
        gbpe.Tokenizer(str(path), cpu_backend="nonsense")


def test_nfc_next_to_added_tokens() -> None:
    """HF normalizes the segments BETWEEN added tokens; decomposed text glued
    to an added token must still match the reference on every route."""
    sp = SPECIALS["qwen25"]
    cases = [f"{sp[0]}e\u0301", f"e\u0301{sp[1]}", f"A\u030a{sp[0]}\u0301x",
             f"{sp[1]}\u0301\u0301 n\u0303", "Cafe\u0301 \u1100\u1161\u11a8"]
    for backend in BACKENDS:
        for i, text in enumerate(cases):
            check("qwen25", text, f"nfc-added{i}", backend)


def test_content_split_threshold() -> None:
    """Mostly-ASCII input up to host_max_bytes stays on the CPU; input with
    over 1/32 non-ASCII bytes above host_max_bytes_multilingual goes to the
    GPU (observable as a new captured size class)."""
    t = gbpe.Tokenizer(str(VOCAB_FILES["llama3"]))
    if t.cpu_backend == "off":
        pytest.skip("no CPU route")
    lo, hi = t.host_max_bytes_multilingual, t.host_max_bytes
    assert 0 < lo < hi
    size = (lo + hi) // 2
    ascii_text = ("the quick brown fox jumps over the lazy dog. " * 2000)[:size]
    before = list(t.graph_capacities)
    assert t.encode(ascii_text) == reference("llama3")(ascii_text)
    assert list(t.graph_capacities) == before, "ASCII input left the CPU route"
    ml = ("日本語のテキストと English words. " * 2000).encode()[:size].decode("utf-8", "ignore")
    assert t.encode(ml) == reference("llama3")(ml)
    assert list(t.graph_capacities) != before, "multilingual input stayed on the CPU"
