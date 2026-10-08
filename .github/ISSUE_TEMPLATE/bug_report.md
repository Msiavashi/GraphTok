---
name: Bug report
about: Report a correctness or build issue
title: ''
labels: bug
assignees: ''
---

## Summary

One sentence: what went wrong.

## Reproducer

```bash
./build/<preset>/gpu_bpe_tokenize --vocab ... --input ... --output ...
```

or the Python call:

```python
import gpu_bpe_tokenizer as gbpe
tok = gbpe.Tokenizer("...")
tok.encode("...")
```

## Expected vs actual

The reference output (`tiktoken` / Hugging Face `tokenizers`) and the GraphTok output. For a token-ID mismatch, include the first differing position and the surrounding text.

## Environment

- GraphTok version or commit:
- GPU and compute capability: <e.g. H100, sm_90>
- CUDA toolkit: <`nvcc --version`>
- Driver: <`nvidia-smi`>
- OS:
- CMake preset or install method: <e.g. `all`, `pip install .`>
- `tokenizer.json` source:

## Additional context

`compute-sanitizer --tool=synccheck` output, logs, or anything else relevant.
