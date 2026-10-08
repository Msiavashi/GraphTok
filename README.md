# GraphTok

GraphTok is a BPE tokenizer that runs end to end on an NVIDIA GPU. Pre-tokenization, BPE merging and output assembly are captured as one replayable CUDA graph, and token IDs are bit-identical to the reference tokenizers (`tiktoken` for GPT-2, Hugging Face `tokenizers` for the rest). Small inputs can be routed to a CPU engine, and a load-adaptive dispatcher batches concurrent requests into single graph replays.

Names: the Python module is `gpu_bpe_tokenizer`, the C library is `libgtok` (CMake target `gtok`), and the CLI binary is `gpu_bpe_tokenize`. Repository: https://github.com/Msiavashi/graphtok.

## Supported tokenizers

| Tokenizer | CMake preset | `GBPE_VOCABS` value |
|---|---|---|
| GPT-2 | `gpt2` | `GPT2` |
| Llama 3 | `llama3` | `LLAMA3` |
| Qwen 2.5 / Qwen 3 (same tokenizer) | `qwen25` | `QWEN25` |
| DeepSeek-V3 | `deepseek_v3` | `DEEPSEEK_V3` |
| Gemma 3 | `gemma3` | `GEMMA3` |

The `all` preset (and `pip install .`) builds GPT-2, Llama 3, DeepSeek-V3 and Gemma 3. Qwen is built with the `qwen25` preset or an explicit `GBPE_VOCABS` list that contains `QWEN25`. The `bytelevel` preset builds GPT-2, Llama 3, Qwen 2.5 and DeepSeek-V3.

## Requirements

- Linux, NVIDIA GPU, CUDA toolkit 12.x (developed with 12.6).
- Default CUDA architectures: `80;90` (A100, H100). Override with `CMAKE_CUDA_ARCHITECTURES`.
- CMake >= 3.24, a C++17 compiler (developed with gcc 11).
- Python >= 3.10, `scikit-build-core` and `pybind11` for the Python module.
- `utf8proc` (headers and library) when Qwen is enabled.
- Optional: a nightly Rust `cargo` to link the gigatoken CPU engine (`-DGBPE_GIGATOKEN=AUTO|ON|OFF`, default `AUTO`). Without it, the CPU route uses the built-in host encoder.

## Install

Python module:

```bash
pip install .                 # or: pip install '.[test]' / '.[vllm]'
```

CMake presets (binaries land in `build/<preset>/`):

```bash
cmake --preset all && cmake --build build/all -j
cmake --preset qwen25 && cmake --build build/qwen25 -j
```

Other architectures:

```bash
cmake --preset all -DCMAKE_CUDA_ARCHITECTURES=89
CMAKE_ARGS="-DCMAKE_CUDA_ARCHITECTURES=89" pip install .
```

See [docs/BUILD.md](docs/BUILD.md) for all build options. A `Makefile` wraps the common workflows (`make help`).

## Vocabulary files

```bash
HF_TOKEN=hf_... python3 scripts/download-vocabs.py      # --force re-downloads
```

This writes `data/hf_gpt2_tokenizer.json`, `data/llama3_tokenizer.json` and `data/vocabs/{qwen25,deepseek_v3,gemma3}.json`. Llama 3 and Gemma 3 are gated on Hugging Face: the token must belong to an account that has accepted their licenses. The token is read from `HF_TOKEN` or from `$HF_HOME/token`.

## Quickstart

### CLI

```bash
./build/all/gpu_bpe_tokenize --vocab data/hf_gpt2_tokenizer.json \
    --input prompt.txt --output tokens.bin

# several documents in one graph replay
./build/all/gpu_bpe_tokenize --vocab data/hf_gpt2_tokenizer.json \
    --inputs a.txt,b.txt --output tokens.bin

# chat template with BOS
./build/all/gpu_bpe_tokenize --vocab data/llama3_tokenizer.json \
    --chat chat.json --add-bos --output tokens.bin
```

Other flags: `--runs N`, `--warmup N`, `--csv`, `--no-cuda-graph`, `--add-eos`, `--no-generation-prompt`, `--decode`, `--help`.

### Python

```python
import gpu_bpe_tokenizer as gbpe

tok = gbpe.Tokenizer("data/hf_gpt2_tokenizer.json")
ids = tok.encode("Hello, world!")             # list[int]
arr = tok.encode_numpy("Hello, world!")       # np.ndarray[uint32]
text = tok.decode(ids)

tokens, offsets = tok.encode_batch(["first doc", "second doc"])
doc0 = tokens[offsets[0]:offsets[1]]          # byte-level vocabs only

# thread-safe, load-adaptive encoder for many concurrent callers
d = gbpe.Dispatcher("data/hf_gpt2_tokenizer.json", cpu_workers=2)
arr = d.encode_numpy("Hello, world!")
```

`Tokenizer(vocab_path, max_input_chars=4096, enable_decode=True, host_max_bytes=None, cpu_backend="")`. `cpu_backend` is `auto`, `gigatoken`, `host` or `off`; an empty string reads `GTOK_CPU_BACKEND`. Also exported: `encode_to_device`, `encode_batch_to_device` (return torch CUDA tensors), `decode_bytes`. `Dispatcher(vocab_path, cpu_workers=1, max_batch_bytes=8 MiB, max_batch_docs=256, policy="adaptive"|"gpu"|"cpu", cpu_backend="")` also provides `stats()`.

### C ABI (libgtok)

`libgtok.so` exports only `gtok_*` symbols (`src/gtok.map`); the functions are defined in `src/gtok_shim.cc`:

- `gtok_create`, `gtok_destroy`, `gtok_vocab_size`
- `gtok_encode`, `gtok_encode_to_device`, `gtok_encode_batch`
- `gtok_cpu_max_bytes`, `gtok_set_cpu_max_bytes`, `gtok_cpu_backend`
- `gtok_dispatcher_create`, `gtok_dispatch_encode`, `gtok_dispatcher_stats`, `gtok_dispatcher_destroy`

Build it with `cmake --build build/all -j --target gtok`.

## Serving integrations

**vLLM.** The package registers a `vllm.general_plugins` entry point. It wraps vLLM's tokenizer so that `encode` / `__call__` on strings run through `libgtok.so`; everything else stays on the Hugging Face tokenizer.

```bash
pip install '.[vllm]'
GBPE_VLLM=1 vllm serve Qwen/Qwen3-32B
```

| Variable | Meaning |
|---|---|
| `GBPE_VLLM` | `1` activates the plugin |
| `GBPE_GTOK_LIB` | path to `libgtok.so` (default: next to the module) |
| `GBPE_GTOK_MAX_BYTES` | initial byte capacity of the GPU context (default 1 MiB, grown on demand) |
| `GBPE_GTOK_PYLIST_DIR` | optional directory with the `gtok_pylist` C extension |

**NVIDIA Dynamo.** See [integrations/dynamo/README.md](integrations/dynamo/README.md).

## Testing

```bash
pip install '.[test]'
pytest tests/                               # needs a GPU and the vocab files
scripts/test-all-presets.sh                 # builds presets, runs exactness gates
scripts/test-all-presets.sh gpt2 qwen25     # selected presets
compute-sanitizer --tool=synccheck ./build/all/gpu_bpe_tokenize \
    --vocab data/hf_gpt2_tokenizer.json --input small.txt \
    --output /tmp/out.bin --runs 1 --warmup 0
```

`scripts/test-all-presets.sh` uses `GBPE_EXACTNESS_DEVICE` (default `2`) to choose the GPU. C++ gate executables (`test_encode_batch_smoke`, `test_cpu_route_classifier`, `test_boundary_simd_equiv`) are built into `build/<preset>/` when `GBPE_BUILD_TESTS=ON` (default).

## Project layout

```
src/                 C++/CUDA sources
  tokenizer.{cu,cuh}   GPU pre-tokenizer, BPE merge, assembly, graph capture
  vocab.{h,cc}         tokenizer.json loader and merge-table builder
  pretokenize.*, pretok_boundary*.h   host pre-tokenizer and boundary rules
  host_encode.*        host BPE encoder
  cpu_route.*, cpu_thresholds.h       CPU route (gigatoken / host encoder)
  dispatcher.*         load-adaptive dispatcher
  special_tokens.*     BOS/EOS and chat template
  main.cc              CLI
  python_bindings.cc   pybind11 module
  gtok_shim.cc, gtok.map              libgtok C ABI
python/gpu_bpe_tokenizer/   Python package and vLLM plugin
rust/gtok_cpu/       Rust wrapper that links gigatoken
third_party/         nlohmann/json, gigatoken source
generated/           generated character-class table
integrations/dynamo/ Dynamo patch
scripts/             vocab download, preset test runner
tools/               exactness collector and helpers
tests/               Python and C++ tests
docs/                architecture, build, adding a tokenizer
```

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
