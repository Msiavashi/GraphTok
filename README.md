# GraphTok

[![Release](https://img.shields.io/github/v/release/Msiavashi/GraphTok?label=release)](https://github.com/Msiavashi/GraphTok/releases)
[![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![Python >=3.10](https://img.shields.io/badge/python-%E2%89%A53.10-3776AB.svg?logo=python&logoColor=white)](pyproject.toml)
[![CUDA 12.6 | 12.8 | 13.1](https://img.shields.io/badge/CUDA-12.6%20%7C%2012.8%20%7C%2013.1-76B900.svg?logo=nvidia&logoColor=white)](docs/BUILD.md)
[![Tested GPUs](https://img.shields.io/badge/tested%20GPUs-H100%20%7C%20A100%20%7C%20L40%20%7C%20RTX%205000%20Ada%20%7C%20RTX%20A6000-76B900.svg)](docs/BUILD.md)
[![Platform: Linux](https://img.shields.io/badge/platform-Linux-lightgrey.svg?logo=linux&logoColor=white)](docs/BUILD.md)
[![C++17](https://img.shields.io/badge/C%2B%2B-17-00599C.svg?logo=cplusplus&logoColor=white)](CMakeLists.txt)
[![Integrations: vLLM | Dynamo](https://img.shields.io/badge/integrations-vLLM%20%7C%20NVIDIA%20Dynamo-orange.svg)](#integrations)

**Website and docs:** https://msiavashi.github.io/GraphTok/

GraphTok is a BPE tokenizer that runs end to end on an NVIDIA GPU. Pre-tokenization, BPE merging and output assembly are captured as one replayable CUDA graph, and token IDs are bit-identical to the reference tokenizers (`tiktoken` for GPT-2, Hugging Face `tokenizers` for the rest). Small inputs can be routed to a CPU engine, and a load-adaptive dispatcher batches concurrent requests into single graph replays. Repository: https://github.com/Msiavashi/GraphTok.

- **Exact:** token IDs match the reference tokenizer bit for bit.
- **Built for long contexts:** encodes 100k to 1M-token prompts in one graph replay.
- **Drop-in for serving:** a Python module, a C library, a vLLM plugin and an NVIDIA Dynamo backend.

Names: Python module `gpu_bpe_tokenizer`, C library `libgtok`, CLI `gpu_bpe_tokenize`.

## News

- **2026-10** — v0.3.0, the first public release: GPU tokenizer, Python bindings, `libgtok` C ABI, vLLM plugin and NVIDIA Dynamo backend patch.
- **2026-10** — Paper "Taking Tokenization off the Host for Agentic LLM Serving" on arXiv (arXiv link coming soon).

## Supported tokenizers

| Tokenizer | CMake preset | Vocab file |
|---|---|---|
| GPT-2 | `gpt2` | `data/hf_gpt2_tokenizer.json` |
| Llama 3 | `llama3` | `data/llama3_tokenizer.json` |
| Qwen 2.5 / Qwen 3 | `qwen25` | `data/vocabs/qwen25.json` |
| DeepSeek-V3 | `deepseek_v3` | `data/vocabs/deepseek_v3.json` |
| Gemma 3 | `gemma3` | `data/vocabs/gemma3.json` |

The `all` preset (and `pip install .`) builds GPT-2, Llama 3, DeepSeek-V3 and Gemma 3. Build Qwen with the `qwen25` preset. See [docs/BUILD.md](docs/BUILD.md) for every preset and option.

## Quick start

Requirements: Linux, an NVIDIA GPU, CUDA toolkit 12.6 or newer (tested with 12.6, 12.8 and 13.1), CMake >= 3.24, a C++17 compiler, Python >= 3.10 (`utf8proc` when Qwen is enabled).

```bash
git clone https://github.com/Msiavashi/GraphTok.git && cd GraphTok

# Python module
pip install .                                   # or '.[test]' / '.[vllm]'

# CLI and libgtok
cmake --preset all && cmake --build build/all -j

# Vocabulary files (Llama 3 and Gemma 3 need a token with accepted licenses)
HF_TOKEN=hf_... python3 scripts/download-vocabs.py
```

Encode a file with the CLI:

```bash
./build/all/gpu_bpe_tokenize --vocab data/hf_gpt2_tokenizer.json \
    --input prompt.txt --output tokens.bin
```

Encode in Python:

```python
import gpu_bpe_tokenizer as gbpe

tok = gbpe.Tokenizer("data/hf_gpt2_tokenizer.json")
ids = tok.encode("Hello, world!")             # list[int]
text = tok.decode(ids)

tokens, offsets = tok.encode_batch(["first doc", "second doc"])
doc0 = tokens[offsets[0]:offsets[1]]

# thread-safe, load-adaptive encoder for many concurrent callers
d = gbpe.Dispatcher("data/hf_gpt2_tokenizer.json", cpu_workers=2)
arr = d.encode_numpy("Hello, world!")
```

## Useful commands

| Task | Command |
|---|---|
| Configure and build a preset | `cmake --preset <preset> && cmake --build build/<preset> -j` |
| Build for another GPU arch | `cmake --preset all -DCMAKE_CUDA_ARCHITECTURES=89` |
| Build only `libgtok` | `cmake --build build/all -j --target gtok` |
| Install the Python module | `pip install .` (tests: `pip install '.[test]'`) |
| Download vocabularies | `python3 scripts/download-vocabs.py` (`--force` re-downloads) |
| Encode a file | `./build/all/gpu_bpe_tokenize --vocab <json> --input <txt> --output <bin>` |
| Encode several files in one replay | `... --inputs a.txt,b.txt --output tokens.bin` |
| Chat template with BOS | `... --vocab data/llama3_tokenizer.json --chat chat.json --add-bos --output tokens.bin` |
| Decode IDs back to text | `... --decode --input tokens.bin --output text.txt` |
| CLI help | `./build/all/gpu_bpe_tokenize --help` |
| Python tests | `pytest tests/` |
| Exactness gate (all / selected presets) | `scripts/test-all-presets.sh` / `scripts/test-all-presets.sh gpt2 qwen25` |
| CUDA synccheck | `compute-sanitizer --tool=synccheck ./build/all/gpu_bpe_tokenize --vocab data/hf_gpt2_tokenizer.json --input small.txt --output /tmp/out.bin --runs 1 --warmup 0` |

The `Makefile` wraps the common workflows; `make help` lists them:

| Target | Does |
|---|---|
| `make build PRESET=llama3` | configure and build a preset |
| `make run INPUT=prompt.txt OUTPUT=/tmp/tokens.bin VOCAB=llama3` | tokenize a file |
| `make test PRESET=llama3` / `make test-all` | exactness gate for one / all presets |
| `make python-install` / `make test-python` | install the module / run the Python tests |
| `make test-exactness` | full native + Python exactness gate |
| `make sanitize INPUT=small.txt` | CUDA synccheck |
| `make clean` | clean the selected preset |

## Integrations

**vLLM.** The package registers a `vllm.general_plugins` entry point that routes string `encode` / `__call__` through `libgtok.so`.

```bash
pip install '.[vllm]'
cmake --preset qwen25 && cmake --build build/qwen25 -j --target gtok   # builds libgtok.so
GBPE_VLLM=1 GBPE_GTOK_LIB=$PWD/build/qwen25/libgtok.so vllm serve Qwen/Qwen3-32B
```

| Variable | Meaning |
|---|---|
| `GBPE_VLLM` | `1` activates the plugin |
| `GBPE_GTOK_LIB` | path to `libgtok.so` (default: next to the module) |
| `GBPE_GTOK_MAX_BYTES` | initial byte capacity of the GPU context (default 1 MiB, grown on demand) |
| `GBPE_GTOK_PYLIST_DIR` | optional directory with the `gtok_pylist` C extension |

**NVIDIA Dynamo.** See [integrations/dynamo/README.md](integrations/dynamo/README.md).

## Development

Development happens on the `develop` branch and pull requests target `develop`; `main` holds releases. See [docs/DEVELOPMENT.md](docs/DEVELOPMENT.md) for the environment, tests and release process, and [CONTRIBUTING.md](CONTRIBUTING.md) before opening a pull request.

## Documentation

- [docs/BUILD.md](docs/BUILD.md) — build presets and options
- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) — design of the GPU pipeline
- [docs/ADDING_A_TOKENIZER.md](docs/ADDING_A_TOKENIZER.md) — adding a vocabulary
- [docs/DEVELOPMENT.md](docs/DEVELOPMENT.md) — development workflow
- [CHANGELOG.md](CHANGELOG.md) — release history

## Citation

```bibtex
@misc{siavashi2026graphtok,
  title={Taking Tokenization off the Host for Agentic {LLM} Serving},
  author={Siavashi, Mohammad and Schwinte, Pascal and Banaei, Ali and Maguire Jr., Gerald Q. and Chiesa, Marco and Kosti{\'c}, Dejan},
  year={2026},
  note={arXiv preprint}
}
```

A machine-readable [`CITATION.cff`](CITATION.cff) is also provided.

## License

Apache License 2.0. See [LICENSE](LICENSE) and [NOTICE](NOTICE).
