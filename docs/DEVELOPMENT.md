# Development

## Environment

- Linux, an NVIDIA GPU, CUDA toolkit 12.6 or newer (tested with 12.6, 12.8 and 13.1), CMake >= 3.24, a C++17 compiler, Python >= 3.10.
- `utf8proc` headers and library when Qwen is enabled.
- Optional: a nightly Rust `cargo` to link the gigatoken CPU engine (`-DGBPE_GIGATOKEN=AUTO|ON|OFF`).

```bash
git clone https://github.com/Msiavashi/graphtok.git && cd graphtok
git checkout develop
pip install '.[test]'
HF_TOKEN=hf_... python3 scripts/download-vocabs.py
```

## Build variants

Each CMake preset builds into `build/<preset>/`:

| Preset | Vocabularies |
|---|---|
| `all` | GPT-2, Llama 3, DeepSeek-V3, Gemma 3 |
| `gpt2`, `llama3`, `qwen25`, `deepseek_v3` | one vocabulary each |
| `bytelevel` | GPT-2, Llama 3, Qwen 2.5, DeepSeek-V3 |
| `gemma3`, `gemma3_cpu`, `gemma3_gpu` | Gemma 3 |

```bash
cmake --preset llama3 && cmake --build build/llama3 -j
cmake --preset all -DCMAKE_CUDA_ARCHITECTURES=89      # other GPU arch
CMAKE_ARGS="-DCMAKE_CUDA_ARCHITECTURES=89" pip install .
```

Equivalent `make` targets: `make build PRESET=<preset>`, `make clean PRESET=<preset>`. See [BUILD.md](BUILD.md) for all options.

## Tests

The Python tests need a GPU, an installed module (`pip install '.[test]'`) and the vocabulary files under `data/`. Tests whose vocabulary file is missing, or whose tokenizer is not compiled into the installed build, are skipped.

```bash
pytest tests/
pytest tests/test_python_bindings.py        # single file
make test-python DEVICE=0                   # bindings, batch and targeted exactness tests
```

C++ test executables (`test_encode_batch_smoke`, `test_cpu_route_classifier`, `test_boundary_simd_equiv`) are built into `build/<preset>/` when `GBPE_BUILD_TESTS=ON` (default).

## Exactness gate

Token IDs must match `tiktoken` (GPT-2) and Hugging Face `tokenizers` (all others).

```bash
scripts/test-all-presets.sh                 # all presets
scripts/test-all-presets.sh gpt2 qwen25     # selected presets
make test PRESET=llama3 DEVICE=0
make test-exactness                         # native + Python gate with JSON report
```

`GBPE_EXACTNESS_DEVICE` selects the GPU for `scripts/test-all-presets.sh`.

## Sanitizer

Run on any kernel change; it must report 0 errors:

```bash
compute-sanitizer --tool=synccheck ./build/all/gpu_bpe_tokenize \
    --vocab data/hf_gpt2_tokenizer.json --input small.txt \
    --output /tmp/out.bin --runs 1 --warmup 0
make sanitize INPUT=small.txt
```

## Generated character-class table

`generated/gbpe_class_table.h` is produced by `tools/gen_class_table.py` from the Python `regex` engine. Regenerate it after changing the classification rules and commit the result:

```bash
pip install regex
python3 tools/gen_class_table.py
```

## Adding a tokenizer

See [ADDING_A_TOKENIZER.md](ADDING_A_TOKENIZER.md).

## Debugging tips

- `--no-cuda-graph` runs the CLI without graph capture, which makes kernel errors easier to localize.
- `--runs 1 --warmup 0` gives a single pass for quick checks and sanitizer runs.
- Set `CUDA_LAUNCH_BLOCKING=1` to report CUDA errors at the failing launch.
- Build a single-vocabulary preset (for example `gpt2`) for faster iteration.
- For a token-ID mismatch, encode the same text with the reference tokenizer and find the first differing position.

## Code style

- C++/CUDA: 4-space indent, `snake_case` functions and variables, `PascalCase` types. Match the surrounding code.
- Warp-level operations run on all 32 lanes with an explicit `0xFFFFFFFFu` mask.
- The captured graph stays free of allocation and host synchronization.
- Python: PEP 8, 4-space indent.
- Comments explain why, not what.

## Release process

1. On `develop`, bump `version` in `pyproject.toml` and add a `CHANGELOG.md` entry.
2. Open a pull request from `develop` into `main` and merge it once the checks pass.
3. Tag the merge commit and push the tag:
   ```bash
   git tag vX.Y.Z && git push origin vX.Y.Z
   ```
4. Publish a GitHub release from the tag with the changelog entry.
