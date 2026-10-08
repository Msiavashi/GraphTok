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
# exact CLI invocation
./build/<preset>/gpu_bpe_tokenize --vocab ... --input ... --output ... ...
```

If the issue is a bit-exactness divergence, include the input excerpt
around the byte offset the harness reports.

## Expected vs actual

What the tiktoken / HF tokenizers oracle produces, what the GPU produced.

## Environment

- GPU SKU and compute capability: <e.g. H100 NVL, sm_90>
- CUDA toolkit version: <`nvcc --version` first line>
- Driver version: <`nvidia-smi` top line>
- OS / glibc: <`lsb_release -d` and `ldd --version | head -1`>
- CMake preset used: <e.g. `all`, `llama3`, ...>
- `tokenizer.json` source: <URL or "vendored under data/...">

## Additional context

- `compute-sanitizer --tool=synccheck` output (if available)
- ncu report (if perf-related)
