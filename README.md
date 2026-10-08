# cuTokenize

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![CUDA 12.6 / 13.1](https://img.shields.io/badge/CUDA-12.6%20%7C%2013.1-76B900.svg)](https://developer.nvidia.com/cuda-toolkit)
[![C++17](https://img.shields.io/badge/C%2B%2B-17-blue.svg)](https://en.cppreference.com/w/cpp/17)
[![Tested on H100 / A100 / L40](https://img.shields.io/badge/GPUs-H100%20%7C%20A100%20%7C%20L40-76B900.svg)](docs/BUILD.md)

A CUDA-graph-compatible BPE tokenizer that runs end-to-end on the GPU, designed
for the 100k–1M token contexts typical of modern long-context LLMs. Pure
C++/CUDA on the hot path; bit-exact to `tiktoken` and HuggingFace `tokenizers`
on every supported vocab.

> **Names.** The project and repository are **cuTokenize**; the paper calls the
> system **GraphTok**. The Python module is `gpu_bpe_tokenizer` and the C library
> is `libgtok`.

## Supported tokenizers

| Vocab           | Vocab size | Merges  | Pre-tokenizer  | Slot policy |
| --------------- | ---------: | ------: | -------------- | ----------- |
| GPT-2           |     50,257 |  50,000 | single-pass    | Slot64      |
| Llama-3         |    128,000 | 280,147 | single-pass    | Slot128     |
| Qwen-2.5        |    151,643 | 151,387 | single-pass    | Slot128     |
| DeepSeek-V3     |    128,000 | 127,741 | **three-pass** | Slot128     |
| Gemma 3         |    262,144 | 514,906 | SP-family*     | Slot128     |

\* Gemma 3 uses a SentencePiece-derived BPE structure with byte fallback and a
`▁` normalizer. `gemma3` runs normalization, initial-ID lookup, fallback,
segmentation, and bucketing inside the CUDA graph. `gemma3_cpu` retains the
original CPU `SPPreTokenizer` for comparison. The GPU path decomposes ordinary
text at merge-table-proven space/newline/tab component boundaries so common
pieces stay in the 32/64-slot kernels; the generic tournament kernel handles
the remaining segments above 64 initial IDs.

## Quickstart

Run `make help` for the common build, test, and sanitizer workflows.
The Makefile is a convenience layer over the CMake presets and existing
harnesses; all commands remain available directly as documented below.

### Build

```bash
# Default tokenizers (GPT-2, Llama-3, DeepSeek-V3, Gemma 3; runtime dispatch between slot policies)
cmake --preset all
cmake --build build/all -j

# Single tokenizer for maximum optimization (smaller binary, fewer kernel
# instantiations, no runtime slot dispatch)
cmake --preset llama3
cmake --build build/llama3 -j

# Gemma 3 uses its GPU pre-tokenizer by default
cmake --preset gemma3
cmake --build build/gemma3 -j
```

Available presets: `all`, `bytelevel`, `gpt2`, `llama3`, `qwen25`,
`deepseek_v3`, `gemma3`, and the retained `gemma3_cpu` comparison build.
See [`docs/BUILD.md`](docs/BUILD.md) for the full matrix, dependencies, and
supported GPU SKUs.

### Tokenize a file

```bash
./build/all/gpu_bpe_tokenize \
    --vocab data/llama3_tokenizer.json \
    --input my_long_document.txt \
    --output tokens.bin \
    --runs 20 --warmup 3 --csv
```

Output: a flat binary array of `uint32_t` token IDs (`tokens.bin`) and a CSV
timing table on stdout.

### Batch-encode multiple documents in a single graph launch

```bash
./build/all/gpu_bpe_tokenize \
    --vocab data/hf_gpt2_tokenizer.json \
    --inputs doc_a.txt,doc_b.txt,doc_c.txt \
    --output batch.bin \
    --runs 10 --warmup 3
```

All documents are staged as one H2D copy and encoded in **one captured CUDA
graph replay** — not one launch per document. Document boundaries are isolated
*inside* the graph (forced pre-token boundaries at every document start plus
per-document clamps in the pre-tokenizer kernels), so a digit run, whitespace
run, contraction or added-token literal straddling a seam can never merge
across it.

**Contract.** `encode_batch(docs)` returns `(tokens, offsets)` with
`len(offsets) == len(docs) + 1`, `offsets[0] == 0`, `offsets[-1] ==
len(tokens)`. Document `d`'s IDs are `tokens[offsets[d]:offsets[d+1]]`, and
that slice is **bit-identical** to encoding document `d` on its own. Empty
documents produce empty slices (`offsets[d] == offsets[d+1]`).

| Surface | Call |
| --- | --- |
| C++ | `TokenizerCtx::encode_batch(BatchEncodeInput&, out_tokens, out_doc_offsets, ...)` |
| CLI | `--inputs a.txt,b.txt,c.txt` → `[n_docs:u32][doc_offsets:u32 x (n_docs+1)][flat tokens:u32 x total]` |
| Python | `tok.encode_batch(list[str])` → `(np.uint32 tokens, np.uint32 offsets)` |
| Python (no D2H) | `tok.encode_batch_to_device(list[str])` → `(torch.int32 cuda tensor, offsets)` |

Batch encoding runs on the byte-level vocabularies (GPT-2, Llama-3, Qwen-2.5,
DeepSeek-V3). For Gemma 3 the CLI's `--inputs` mode encodes each document in
turn on one reused context with the same output file layout.

### Decode (token IDs → bytes)

```bash
./build/all/gpu_bpe_tokenize --vocab data/llama3_tokenizer.json --decode \
    --input tokens.bin --output decoded.bin --runs 5 --warmup 2
```

Input is a flat binary `uint32_t` array (same format produced by the encoder).
Output is the decoded byte stream. The decoder is a separate captured CUDA
graph (length kernel → CUB exclusive-scan → gather kernel). Gemma receives an
ID-aware host post-step after that graph for `▁ → space` replacement and
`<0xHH>` byte-fallback fusion; this preserves token boundaries and matches the
serialized Hugging Face decoder. For valid UTF-8 sequences, CLI output is
bit-exact to `Tokenizer.decode(ids, skip_special_tokens=False).encode("utf-8")`
for HF vocabs and to `tiktoken.decode(ids).encode("utf-8")` for GPT-2.

### Run the repeatable exactness gate

```bash
scripts/test-all-presets.sh
# equivalent: make test-exactness DEVICE=2
```

This builds each stable CMake preset, checks every compiled vocabulary against
its external oracle, runs the C++ byte-level batch seam
gate, force-installs the Python extension in the project `uv` environment, and
runs the binding, batch, and targeted boundary suites. The `all` binary is
checked for GPT-2, Llama-3, DeepSeek-V3, and Gemma 3—not just GPT-2.

The collector writes a machine-readable local report to
`build/exactness/latest.json` (ignored by Git). Each test entry records
its pass/fail/skip state, configuration, source revision, executable or
extension hash, and relevant tokenizer asset hashes; top-level provenance also
records the full software and host configuration. Use a named path for a
reviewable artifact:

```bash
uv run --extra test python tools/collect_exactness.py --device 2 \
  --out exactness-h100.json
```

The `targeted_suite_complete_for_paper_configurations` field is the explicit
gate for the 31/32/33-, 63/64/65-symbol, equal-rank leftmost, and changing
overflow/capacity tests. `publication_exactness_gate_passed` additionally
requires every native and Python check to pass. Do not describe the suite as
complete without its corresponding JSON report.

The Gemma targeted newline/tab/metaspace case is part of
`tests/test_python_bindings.py::test_gemma3`; it compares GPU output directly
with Hugging Face and must pass before the collector can report a green gate.

### Special tokens and chat templates

The kernel emits bare BPE token IDs. A pure host-side post-step adds BOS/EOS
and applies chat templates, bit-exact to HuggingFace:

```bash
# add_special_tokens=True equivalent (BOS prepended for Llama-3, no-op for
# Qwen-2.5 / GPT-2 / DeepSeek-V3 — matches HF post_processor behavior).
./build/all/gpu_bpe_tokenize \
    --vocab data/llama3_tokenizer.json \
    --input prompt.txt --output ids.bin \
    --add-bos

# Chat template: pass a JSON array of {role, content}. Renders the canonical
# template for the vocab (Llama-3, Qwen-2.5, DeepSeek-V3), encodes the text
# pieces through the GPU pipeline, splices in special-token ids.
echo '[{"role":"user","content":"hi"}]' > chat.json
./build/all/gpu_bpe_tokenize \
    --vocab data/llama3_tokenizer.json \
    --chat chat.json --output ids.bin
```

Validated against `transformers.AutoTokenizer.apply_chat_template` on
Llama-3-8B-Instruct, Qwen2.5-7B-Instruct, and DeepSeek-V3. GPT-2 has no chat
template.

## Python bindings

```bash
pip install .              # builds the CUDA extension via scikit-build-core
```

```python
import gpu_bpe_tokenizer as gbpe

tok = gbpe.Tokenizer("data/llama3_tokenizer.json")
ids = tok.encode("hello world")          # list[int]
arr = tok.encode_numpy("hello world")    # np.ndarray[uint32]
print(tok.vocab_size, tok.regex_kind)    # 128000  Llama3

# Many documents, one graph replay. tokens[offsets[d]:offsets[d+1]] is doc d,
# bit-identical to tok.encode(docs[d]).
tokens, offsets = tok.encode_batch(["first doc", "", "third doc"])
```

The Python class is a thin wrapper around `HostVocab` + `VocabPack` +
`PreTokenizer` + `TokenizerCtx`; output is bit-exact with the CLI on the same
inputs. Buffers grow lazily to the next power of two on inputs that exceed
the current cap (one CUDA-graph re-capture, then steady state). The GIL is
released over the pre-tokenize + GPU encode hot path.
See `tests/test_python_bindings.py` for usage examples.

## Serving integrations

**vLLM.** `gpu_bpe_tokenizer.vllm` routes vLLM's prompt tokenization through
`libgtok.so` (build target `gtok`); decoding, chat templates and special tokens
stay on the Hugging Face tokenizer. It ships as a vLLM general plugin:

```bash
cmake --preset all && cmake --build build/all -j --target gtok
pip install ".[vllm]"
GBPE_VLLM=1 GBPE_GTOK_LIB=$PWD/build/all/libgtok.so vllm serve Qwen/Qwen3-32B
```

Supported models: Llama-3 and Qwen-2.5 / Qwen-3. Alternatively call
`gpu_bpe_tokenizer.vllm.install()` in the server process. See the module
docstring in [`python/gpu_bpe_tokenizer/vllm.py`](python/gpu_bpe_tokenizer/vllm.py).

**NVIDIA Dynamo.** [`integrations/dynamo/`](integrations/dynamo/) holds a patch
for Dynamo 1.3.0 that adds a `graphtok` frontend tokenizer backend. Apply it
with `git apply` on commit `bae0051`, then enable it with
`DYN_TOKENIZER=graphtok DYN_GRAPHTOK_DISPATCH=1 DYN_GRAPHTOK_LIB=.../libgtok.so`.
See [`integrations/dynamo/README.md`](integrations/dynamo/README.md).

## Architecture in one paragraph

One CUDA block per pre-token, one warp per block, two pre-tokens (= two
independent warps) packed per block to reach 100% theoretical occupancy on
Hopper. Each warp runs the canonical `_byte_pair_merge` algorithm (mirroring
`tiktoken`) with warp-cooperative argmin via `__shfl_xor_sync` and a 64-bit
ballot-derived alive mask. The merge table is a packed open-addressing hash:
**one 64-bit slot per entry** for GPT-2-family vocabs (vocab and rank fit in
16 bits each), **two 64-bit slots** for larger vocabs (Llama-3, Qwen, DeepSeek,
Gemma 3). For the byte-level vocabs (GPT-2, Llama-3, Qwen-2.5, DeepSeek-V3),
**pre-tokenization runs entirely on the GPU**: a classification kernel + a
parallel boundary predicate + CUB scans split the raw UTF-8 bytes into
pre-tokens on-device — no host regex. The full pipeline — pre-tokenize → BPE
kernel → `cub::DeviceScan::ExclusiveSum` → flatten — is captured into a single
`cudaGraph_t` and replayed via `cudaGraphLaunch`, with zero allocations and zero
D2H syncs inside the captured region. The host hands over raw bytes; token IDs
come back from device memory (or stay resident for a fully on-GPU pipeline).
Gemma 3 uses the captured GPU pre-tokenizer by default. The separate
`gemma3_cpu` preset retains the original host `SPPreTokenizer` for comparison.

See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for the full design — the
GPU pre-tokenizer pipeline, the BPE merge kernel, and the per-vocab seam.

## Project layout

```
src/                      C++/CUDA implementation
  build_config.h            compile-time vocab/slot guards
  tokenizer.{cuh,cu}        GPU pre-tokenizer + BPE kernels + CUDA-graph capture/replay
  vocab.{h,cc}              tokenizer.json loader; device merge-table + class-table builder
  pretokenize.{h,cc}        host SP-family (Gemma 3) pre-tokenizer; estimate_max_pretokens
  special_tokens.{h,cc}     BOS/EOS + chat-template (host post-step)
  python_bindings.cc        pybind11 Tokenizer; main.cc — standalone CLI
rust/                     CPU route engine (gtok_cpu)
generated/                gbpe_class_table.h (codepoint→class lookup table)
python/gpu_bpe_tokenizer/ Python package (vllm.py: vLLM adapter)
integrations/dynamo/      NVIDIA Dynamo 1.3.0 GraphTok backend patch
tools/                    gen_class_table.py, collect_exactness.py (exactness gate)
tests/                    C++ and Python library tests
docs/                     ARCHITECTURE, BUILD, ADDING_A_TOKENIZER
scripts/                  download-vocabs.py, test-all-presets.sh
third_party/              vendored nlohmann/json and rs-gigatoken
CMakePresets.json         one preset per vocab + multi-vocab
LICENSE                   Apache 2.0; NOTICE — third-party attributions
```

## Requirements

- NVIDIA GPU with **compute capability ≥ 8.0** (A100 / H100 / L40 verified).
  The kernel uses `__shfl_*_sync`, `__ballot_sync`, and `cudaStreamBeginCapture`.
- **CUDA Toolkit ≥ 12.0** (12.6 used in development).
- **gcc 11+** and **CMake 3.24+**.
- **utf8proc** headers/library for builds that include Qwen-2.5 NFC normalization.
- For bit-exactness checks: Python 3.10+ with `tiktoken` and `tokenizers`
  (`pip install '.[test]'`).

## Contributing

See [`CONTRIBUTING.md`](CONTRIBUTING.md). New tokenizer vocabs follow the
checklist in [`docs/ADDING_A_TOKENIZER.md`](docs/ADDING_A_TOKENIZER.md).

## License

Apache 2.0 — see [`LICENSE`](LICENSE) and [`NOTICE`](NOTICE).

## Citation

If you use cuTokenize, please cite:

```bibtex
@misc{siavashi2026graphtok,
  title={Taking Tokenization off the Host for Agentic {LLM} Serving},
  author={Siavashi, Mohammad and Schwinte, Pascal and Banaei, Ali and Maguire Jr., Gerald Q. and Chiesa, Marco and Kosti{\'c}, Dejan},
  year={2026},
  note={arXiv preprint}
}
```

A machine-readable [`CITATION.cff`](CITATION.cff) is also provided.
