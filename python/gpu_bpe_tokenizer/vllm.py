# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 the GraphTok contributors
"""vLLM adapter: route vLLM's prompt tokenization through GraphTok (libgtok.so).

The tokenizer vLLM builds (``cached_tokenizer_from_config`` and the one handed
to ``AsyncMicrobatchTokenizer``) is wrapped so that ``encode`` / ``__call__`` on
plain strings run on the GPU; everything else (decode, vocab, special tokens,
chat templates, unsupported call shapes) stays on the Hugging Face tokenizer.
Single documents go through ``gtok_encode``; a list of strings goes through
``gtok_encode_batch`` (one graph replay per microbatch). All single-document
CUDA-graph size classes are captured when the encoder is built (server start).

Enable it in every vLLM process with the bundled general plugin::

    pip install "gpu_bpe_tokenizer[vllm]"
    GBPE_VLLM=1 vllm serve Qwen/Qwen3-32B

or call :func:`install` in the server process before the engine is built.

Environment variables:
  GBPE_VLLM=1              activate the plugin (otherwise it does nothing)
  GBPE_GTOK_LIB            path to libgtok.so (CMake target ``gtok``);
                           default: libgtok.so next to this module
  GBPE_GTOK_MAX_BYTES      byte cap of the GPU context (default 1 MiB; grown on demand)
  GBPE_GTOK_PYLIST_DIR     optional directory holding the ``gtok_pylist`` C
                           extension (builds the id list in C); numpy is used otherwise
"""
from __future__ import annotations

import ctypes
import importlib
import os
import sys
import threading

__all__ = ["GtokEncoder", "GtokProxy", "install", "register", "CuTokenizeError"]

SUPPORTED = ("llama-3", "llama3", "meta-llama/llama-3", "meta-llama/meta-llama-3",
             "qwen2.5", "qwen3")   # Qwen3's tokenizer is Qwen2.5's

_WARM_WORDS = ("alpha beta gamma delta déjà vu naïve café 数据 東京 Привет "
               "мир 12345 hello, world! \n\tcode_block(x) -> y; ").split(" ")


class CuTokenizeError(RuntimeError):
    pass


def default_lib_path() -> str:
    return os.environ.get("GBPE_GTOK_LIB") or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "libgtok.so")


def _load_pylist():
    d = os.environ.get("GBPE_GTOK_PYLIST_DIR")
    if not d:
        return None
    sys.path.insert(0, d)
    try:
        import gtok_pylist
        return gtok_pylist
    except ImportError:
        return None
    finally:
        sys.path.remove(d)


def _warm_text(nbytes: int) -> str:
    """Text with enough non-ASCII bytes that the library's CPU route does not
    take the small size classes, so every GPU class gets captured."""
    unit = " ".join(_WARM_WORDS) + " "
    s = unit * (nbytes // len(unit.encode()) + 2)
    return s.encode()[:nbytes].decode("utf-8", errors="ignore")


class GtokEncoder:
    """ctypes wrapper over libgtok.so (one GPU context), thread-safe via a lock."""

    _WARM_MIN_CLASS = 8192
    _TLS_KEEP_ELEMS = 1 << 23

    def __init__(self, vocab_json: str, lib_path: str | None = None,
                 max_bytes: int | None = None):
        import numpy as np
        self._np = np
        lib_path = lib_path or default_lib_path()
        max_bytes = max_bytes or int(os.environ.get("GBPE_GTOK_MAX_BYTES", 1 << 20))
        self._lib = L = ctypes.CDLL(lib_path)
        L.gtok_create.restype = ctypes.c_void_p
        L.gtok_create.argtypes = [ctypes.c_char_p, ctypes.c_uint32]
        L.gtok_vocab_size.restype = ctypes.c_uint32
        L.gtok_vocab_size.argtypes = [ctypes.c_void_p]
        L.gtok_destroy.argtypes = [ctypes.c_void_p]
        self._encode_vp = L["gtok_encode"]
        self._encode_vp.restype = ctypes.c_int32
        self._encode_vp.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_uint32,
                                    ctypes.c_void_p, ctypes.c_uint32]
        if not hasattr(L, "gtok_encode_batch"):
            raise CuTokenizeError(f"{lib_path} has no gtok_encode_batch")
        self._batch_vp = L["gtok_encode_batch"]
        self._batch_vp.restype = ctypes.c_int32
        self._batch_vp.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_void_p,
                                   ctypes.c_uint32, ctypes.c_void_p, ctypes.c_uint32,
                                   ctypes.c_void_p]
        self._tls = threading.local()
        self._vocab_json = vocab_json
        top = 4096
        while top < max(max_bytes, 1 << 16) + 1024:
            top <<= 1
        self._max_bytes = top - 1024
        self._h = L.gtok_create(vocab_json.encode(), self._max_bytes)
        if not self._h:
            raise CuTokenizeError(f"gtok_create failed for {vocab_json}")
        self._lock = threading.Lock()
        self.vocab_size = int(L.gtok_vocab_size(self._h))
        self._pylist = _load_pylist()
        if self._pylist is not None:
            self._pylist.init(self.vocab_size)
        self._fn_addr = ctypes.cast(L.gtok_encode, ctypes.c_void_p).value
        self._out = np.empty(self._max_bytes + 64, dtype=np.uint32)
        self._out_addr = self._out.ctypes.data
        self._warm()

    def _warm(self):
        """Capture the 8 single-document size classes ending at the byte cap's
        class (two calls each: capture, then first replay)."""
        top = 4096
        while top < self._max_bytes + 1024:
            top <<= 1
        c = max(self._WARM_MIN_CLASS, top >> 7)
        while c <= top:
            b = _warm_text(c - 1024).encode("utf-8")
            for _ in range(2):
                r = self._encode_vp(self._h, b, len(b), self._out_addr, self._out.shape[0])
                if r < 0:
                    raise CuTokenizeError(f"warm-up encode error {r} ({c} B class)")
            c <<= 1

    def _tls_buf(self, slot: str, need: int):
        ent = getattr(self._tls, slot, None)
        if ent is not None and ent[0].shape[0] >= need:
            return ent
        size = 1 << max(need - 1, 4095).bit_length()
        buf = self._np.empty(size, dtype=self._np.uint32)
        ent = (buf, buf.ctypes.data)
        if size <= self._TLS_KEEP_ELEMS:
            setattr(self._tls, slot, ent)
        return ent

    def encode(self, text: str) -> list[int]:
        return self.encode_ids(text)

    def _encode_np(self, text, pre, suf, lim):
        b = text.encode("utf-8")
        r = self._encode_vp(self._h, b, len(b), self._out_addr, self._out.shape[0])
        if r < 0:
            return r
        ids = pre + self._out[:r].tolist() + suf
        return ids if lim < 0 else ids[:lim]

    def encode_ids(self, text: str, pre=(), suf=(), limit: int | None = None) -> list[int]:
        """(pre + encode(text) + suf)[:limit]."""
        pre = pre if type(pre) is list else list(pre)
        suf = suf if type(suf) is list else list(suf)
        lim = -1 if limit is None else limit
        with self._lock:
            for _ in range(2):     # -10 / -3: text exceeds the byte cap -> grow, retry
                if self._pylist is not None:
                    r = self._pylist.encode(self._fn_addr, self._h, text, pre, suf, lim,
                                            self._out_addr, self._out.shape[0], False)
                else:
                    r = self._encode_np(text, pre, suf, lim)
                if r.__class__ is list:
                    return r
                if r != -10 and r != -3:
                    break
                n = len(text.encode("utf-8"))
                self._grow(max(n, self._max_bytes + 1) if r == -10 else n + (n >> 1))
        if r == -2:
            raise CuTokenizeError("output buffer too small")
        raise CuTokenizeError(f"gtok_encode error {r}")

    def encode_batch_ids(self, docs, pre=(), suf=(), limit: int | None = None):
        """All documents in ONE graph replay; each (pre + ids + suf)[:limit]."""
        np = self._np
        if not docs:
            return []
        blobs = [d.encode("utf-8") if isinstance(d, str) else d for d in docs]
        n_docs = len(blobs)
        concat = b"".join(blobs)
        n = len(concat)
        byte_offs = np.zeros(n_docs + 1, dtype=np.uint32)
        byte_offs[1:] = np.cumsum(np.fromiter(map(len, blobs), dtype=np.int64, count=n_docs))
        tok_offs = np.zeros(n_docs + 1, dtype=np.uint32)
        cap = max(64, n + 16)
        out, out_ptr = self._tls_buf("out", cap)
        with self._lock:
            if n > self._max_bytes:
                self._grow(n)
            r = self._batch_vp(self._h, concat, byte_offs.ctypes.data, n_docs, out_ptr, cap,
                               tok_offs.ctypes.data)
        if r == -3:
            with self._lock:
                self._grow(n)
            return self.encode_batch_ids(docs, pre, suf, limit)
        if r == -2:
            raise CuTokenizeError("output buffer too small")
        if r == -4:
            raise CuTokenizeError(f"batch of {n_docs} docs exceeds doc capacity")
        if r < 0:
            raise CuTokenizeError(f"gtok_encode_batch error {r}")
        offs = tok_offs.tolist()
        pre, suf = list(pre), list(suf)
        res = []
        for i in range(n_docs):
            a, e = offs[i], offs[i + 1]
            if not pre and not suf and self._pylist is not None:
                if limit is not None and e - a > limit:
                    e = a + limit
                res.append(self._pylist.from_buffer(out_ptr, a, e - a))
                continue
            ids = pre + out[a:e].tolist() + suf
            res.append(ids if limit is None else ids[:limit])
        return res

    def _grow(self, need_bytes: int):
        """Re-create the GPU context with a larger byte cap (caller holds _lock)."""
        new = 1 << max(need_bytes, self._max_bytes + 1).bit_length()
        old, self._h = self._h, None
        if old:
            self._lib.gtok_destroy(old)
        h = self._lib.gtok_create(self._vocab_json.encode(), new)
        if not h:
            raise CuTokenizeError(f"gtok_create failed while growing to {new} bytes")
        self._h, self._max_bytes = h, new
        self._out = self._np.empty(new + 64, dtype=self._np.uint32)
        self._out_addr = self._out.ctypes.data
        self._warm()


_BatchEncoding = None


def _batch_encoding():
    global _BatchEncoding
    if _BatchEncoding is None:
        from transformers.tokenization_utils_base import BatchEncoding
        _BatchEncoding = BatchEncoding
    return _BatchEncoding


class GtokProxy:
    """HF tokenizer with its string encode path on GraphTok. truncation /
    max_length are honoured by slicing; GPU-side errors fall back to HF."""

    def __init__(self, hf_tokenizer, encoder: GtokEncoder):
        object.__setattr__(self, "_hf", hf_tokenizer)
        object.__setattr__(self, "_gpu", encoder)
        probe = "cuTokenize probe 123"      # specials HF adds (Llama-3: [BOS], [])
        with_sp = hf_tokenizer.encode(probe, add_special_tokens=True)
        bare = encoder.encode(probe)
        pre, suf = [], []
        if with_sp[len(with_sp) - len(bare):] == bare:
            pre = with_sp[: len(with_sp) - len(bare)]
        elif with_sp[: len(bare)] == bare:
            suf = with_sp[len(bare):]
        else:
            raise CuTokenizeError("cannot reconcile HF special tokens with GraphTok output")
        object.__setattr__(self, "_pre", pre)
        object.__setattr__(self, "_suf", suf)

    def __call__(self, text, text_pair=None, add_special_tokens=True, truncation=False,
                 max_length=None, **kwargs):
        BatchEncoding = _batch_encoding()
        if isinstance(text, (list, tuple)) and text_pair is None and not kwargs \
                and all(isinstance(t, str) for t in text):
            ids_list = self._encode_many(list(text), truncation, max_length, add_special_tokens)
            return BatchEncoding({"input_ids": ids_list,
                                  "attention_mask": [[1] * len(i) for i in ids_list]})
        if text_pair is not None or kwargs or not isinstance(text, str):
            return object.__getattribute__(self, "_hf")(
                text, text_pair=text_pair, add_special_tokens=add_special_tokens,
                truncation=truncation, max_length=max_length, **kwargs)
        ids = self.encode(text, truncation=truncation, max_length=max_length,
                          add_special_tokens=add_special_tokens)
        return BatchEncoding({"input_ids": ids, "attention_mask": [1] * len(ids)})

    def __getattr__(self, name):
        return getattr(object.__getattribute__(self, "_hf"), name)

    def __setattr__(self, name, value):
        setattr(object.__getattribute__(self, "_hf"), name, value)

    def __len__(self):
        return len(object.__getattribute__(self, "_hf"))

    def encode(self, text, truncation=None, max_length=None, add_special_tokens=True, **kwargs):
        gpu = object.__getattribute__(self, "_gpu")
        hf = object.__getattribute__(self, "_hf")
        if not isinstance(text, str) or kwargs:
            return hf.encode(text, add_special_tokens=add_special_tokens,
                             truncation=bool(truncation), max_length=max_length, **kwargs)
        lim = max_length if truncation and max_length is not None else None
        try:
            if add_special_tokens:
                return gpu.encode_ids(text, object.__getattribute__(self, "_pre"),
                                      object.__getattribute__(self, "_suf"), lim)
            return gpu.encode_ids(text, (), (), lim)
        except Exception:
            return hf.encode(text, add_special_tokens=add_special_tokens,
                             truncation=bool(truncation), max_length=max_length)

    def _encode_many(self, texts, truncation, max_length, add_special_tokens):
        """A microbatch in ONE gtok_encode_batch call (per-document on error)."""
        gpu = object.__getattribute__(self, "_gpu")

        def per_document():
            return [self.encode(t, truncation=truncation, max_length=max_length,
                                add_special_tokens=add_special_tokens) for t in texts]
        if len(texts) == 1:
            return per_document()
        limit = max_length if truncation and max_length is not None else None
        try:
            if add_special_tokens:
                return gpu.encode_batch_ids(texts, object.__getattribute__(self, "_pre"),
                                            object.__getattribute__(self, "_suf"), limit)
            return gpu.encode_batch_ids(texts, (), (), limit)
        except Exception:
            return per_document()


def _resolve_vocab_json(tokenizer) -> str | None:
    nvp = getattr(tokenizer, "name_or_path", None)
    if nvp and os.path.isfile(os.path.join(nvp, "tokenizer.json")):
        return os.path.join(nvp, "tokenizer.json")
    if nvp and "/" in nvp and not os.path.isdir(nvp):
        try:
            from huggingface_hub import hf_hub_download
            return hf_hub_download(nvp, "tokenizer.json")
        except Exception:
            return None
    return None


def _patch_tokenizer_factory(wrap) -> int:
    patched = 0
    for modname in ("vllm.renderers.registry", "vllm.multimodal.registry"):
        try:
            m = importlib.import_module(modname)
        except Exception:
            continue
        fn = getattr(m, "cached_tokenizer_from_config", None)
        if fn is None or hasattr(fn, "_gbpe_orig"):
            continue

        def inner(*a, _orig=fn, **k):
            return wrap(_orig(*a, **k))
        inner._gbpe_orig = fn
        m.cached_tokenizer_from_config = inner
        patched += 1
    return patched


def install(lib_path: str | None = None) -> None:
    """Wrap the tokenizers vLLM builds with GtokProxy. One encoder per
    process, built and warmed on first wrap (server start-up)."""
    lib_path = lib_path or default_lib_path()
    state = {"enc": None}

    def wrap(tok):
        if tok is None or isinstance(tok, GtokProxy):
            return tok
        nvp = (getattr(tok, "name_or_path", "") or "").lower()
        if not any(x in nvp for x in SUPPORTED):
            return tok              # not a GraphTok vocabulary: keep the HF tokenizer
        if state["enc"] is None:
            vj = _resolve_vocab_json(tok)
            if not vj:
                raise CuTokenizeError(f"no tokenizer.json for {nvp!r}")
            state["enc"] = GtokEncoder(vj, lib_path)
        return GtokProxy(tok, state["enc"])

    _patch_tokenizer_factory(wrap)
    au = importlib.import_module("vllm.utils.async_utils")
    cls = au.AsyncMicrobatchTokenizer
    if not getattr(cls.__init__, "_gbpe_gtok", False):
        orig_init = cls.__init__

        def new_init(self, tokenizer, *a, **k):
            orig_init(self, wrap(tokenizer), *a, **k)
        new_init._gbpe_gtok = True
        cls.__init__ = new_init


def register() -> None:
    """vLLM general-plugin entry point (``vllm.general_plugins``); active
    only when GBPE_VLLM=1."""
    if os.environ.get("GBPE_VLLM", "0") not in ("", "0"):
        install()
