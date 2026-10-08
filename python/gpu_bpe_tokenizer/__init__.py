# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 the cuTokenize contributors
"""gpu_bpe_tokenizer — GPU-accelerated BPE tokenizer.

Thin Python wrapper around the C++/CUDA encoder. Bit-exact with the CLI
binary (`gpu_bpe_tokenize`) on the same vocab + input.

Example
-------
    >>> import gpu_bpe_tokenizer as gbpe
    >>> tok = gbpe.Tokenizer("data/llama3_tokenizer.json")
    >>> tok.encode("hello world")
    [...]
    >>> tok.vocab_size, tok.regex_kind
    (128256, 'Llama3')
"""

from ._gpu_bpe_tokenizer import Dispatcher, Tokenizer  # noqa: F401

__all__ = ["Dispatcher", "Tokenizer"]
__version__ = "0.3.0"
