# Building GraphTok

## Requirements

- CUDA toolkit 12.x, CMake >= 3.24, a C++17 host compiler.
- `utf8proc` when Qwen is enabled.
- Optional nightly Rust `cargo` for the gigatoken CPU engine.
- Python >= 3.10 with `scikit-build-core` and `pybind11` for the Python module.

## Presets

```bash
cmake --preset <name> && cmake --build build/<name> -j
```

| Preset | Vocabularies |
|---|---|
| `all` | GPT-2, Llama 3, DeepSeek-V3, Gemma 3 |
| `gpt2` | GPT-2 (Slot64 only) |
| `llama3` | Llama 3 |
| `qwen25` | Qwen 2.5 / Qwen 3 |
| `deepseek_v3` | DeepSeek-V3 |
| `bytelevel` | GPT-2, Llama 3, Qwen 2.5, DeepSeek-V3 |
| `gemma3` | Gemma 3, GPU pre-tokenizer |
| `gemma3_cpu` | Gemma 3, host pre-tokenizer |
| `gemma3_gpu` | alias of `gemma3` |

All presets are Release builds in `build/<preset>/` with
`CMAKE_CUDA_ARCHITECTURES=80;90`.

Targets: `gpu_bpe_tokenize` (CLI), `gtok` (`libgtok.so`), and, with
`GBPE_BUILD_TESTS=ON`, `test_encode_batch_smoke`, `test_cpu_route_classifier`,
`test_boundary_simd_equiv` (byte-level builds only).

## CMake options

| Option | Default | Meaning |
|---|---|---|
| `GBPE_VOCABS` | `ALL` | `ALL` or a `;`-separated subset of `GPT2;LLAMA3;QWEN25;DEEPSEEK_V3;GEMMA3`. `ALL` excludes `QWEN25`. |
| `GBPE_GEMMA_GPU_PRETOK` | `ON` | graph-resident Gemma 3 pre-tokenizer |
| `GBPE_GIGATOKEN` | `AUTO` | link gigatoken as the CPU engine (`AUTO`/`ON`/`OFF`) |
| `GBPE_BUILD_TESTS` | `ON` | build the C++ gate executables |
| `GBPE_BUILD_PYTHON` | `OFF` | build the Python module (forced on by `pip install`) |
| `GBPE_HOST_STATS` | `OFF` | compile host-encoder cache counters |
| `CMAKE_CUDA_ARCHITECTURES` | `80;90` | target GPU architectures |

Example:

```bash
cmake -S . -B build/custom -DGBPE_VOCABS="GPT2;QWEN25" -DCMAKE_CUDA_ARCHITECTURES=90
cmake --build build/custom -j
```

## Python module

```bash
pip install .
pip install '.[test]'      # pytest, tiktoken, tokenizers
pip install '.[vllm]'      # vLLM plugin dependencies
```

`pip install` configures CMake with `-DGBPE_BUILD_PYTHON=ON -DGBPE_VOCABS=ALL`.
Extra CMake arguments go through `CMAKE_ARGS`, for example
`CMAKE_ARGS="-DCMAKE_CUDA_ARCHITECTURES=89" pip install .`.

## Makefile

`make help` lists the wrappers: `configure`, `build`, `clean`, `run`, `test`,
`test-all`, `python-install`, `test-python`, `test-exactness`, `sanitize`.
Variables: `PRESET` (default `all`), `VOCAB` (default `gpt2`), `DEVICE`,
`INPUT`, `OUTPUT`, `ARGS`.
