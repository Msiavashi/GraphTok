# Contributing to GraphTok

Issues and pull requests are welcome at https://github.com/Msiavashi/graphtok.

## Reporting bugs

Include the exact CLI invocation or Python call, the input (or a sample), the
`tokenizer.json` or its source, the build preset and the GPU model. For a
token-ID mismatch against `tiktoken` or Hugging Face `tokenizers`, include the
byte offset where the outputs first differ and the surrounding text.

## Before opening a pull request

1. Exactness gate for the affected presets:
   ```bash
   scripts/test-all-presets.sh            # or: scripts/test-all-presets.sh gpt2 qwen25
   ```
2. Python tests:
   ```bash
   pip install '.[test]'
   pytest tests/
   ```
3. For any kernel change, `compute-sanitizer` must report 0 errors:
   ```bash
   compute-sanitizer --tool=synccheck ./build/all/gpu_bpe_tokenize \
       --vocab data/hf_gpt2_tokenizer.json --input small.txt \
       --output /tmp/out.bin --runs 1 --warmup 0
   ```

Changes to the pre-tokenizer, merge kernel, slot layouts or CPU route must keep
token IDs identical to the reference tokenizer.

## Code style

- C++/CUDA: 4-space indent, `snake_case` functions and variables,
  `PascalCase` types. Match the surrounding code.
- Warp-level operations run on all 32 lanes with an explicit
  `0xFFFFFFFFu` mask (see [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)).
- The captured graph must stay free of allocation and host synchronization.
- Comments explain why, not what.

## Adding a tokenizer

See [docs/ADDING_A_TOKENIZER.md](docs/ADDING_A_TOKENIZER.md).

## License

Contributions are licensed under Apache 2.0, the same as the project. See
[LICENSE](LICENSE).
