# Changelog

All notable changes to this project. Format loosely follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/). Dates are in
ISO 8601 (`YYYY-MM-DD`). Numbered releases are tagged; the Milestones
section describes development milestones tied to commit boundaries.

## [Unreleased] — roadmap

Planned features tracked as task IDs in the engineering log. Not yet
implemented:

- **F1b** — Gemma 3 (SentencePiece-derived BPE). In progress on `agent/f1b`.
- **F8** — Final integration sweep + writeup.

## [0.2.0] — 2026-08-01

GPU-side batched encode: `encode_batch()` tokenizes N documents in a single
CUDA-graph replay with one host sync per batch and per-document token
offsets, bit-exact with per-document encodes for all byte-level vocab
families (GPT-2, Llama-3, Qwen-2.5, DeepSeek-V3). Exposed via C++
(`TokenizerCtx::encode_batch`), the CLI (`--inputs`), and Python
(`encode_batch` / `encode_batch_to_device`). Measured on H100: 256×4 KiB
batches run 10.4× faster end-to-end than a serial per-document loop;
single-document kernel medians at 1M tokens change by +0.73 % (GPT-2) /
+1.65 % (Llama-3), within the ≤2 % budget. Details in the milestone below.

## Milestones

### GPU-side batched encode — 2026-08-01 (feature/encode-batch)

`encode_batch()` re-implemented on top of the graph-resident GPU
pre-tokenizer. **Supersedes the host-side F3 batch interface below**, whose
`PreTokenizer::apply_batch` aggregation no longer exists (the host PCRE2
byte-level pipeline was removed when pre-tokenization moved onto the GPU).

- **Document isolation inside the graph.** Document start offsets are uploaded
  with the input bytes; `d_cp_doc_start` carries the same boundaries into the
  codepoint domain. Every pre-tokenizer kernel clamps its neighbour reads and
  its byte extents to the owning document, and document starts are forced
  pre-token boundaries. A digit run, whitespace run, contraction or
  added-token literal straddling a document seam therefore cannot merge across
  it. Bit-exact with per-document `encode()` on every byte-level family.
- **One graph replay per batch.** All documents are staged in a single H2D
  copy and encoded by one replay of the same captured graph used for single
  documents; `n_docs == 1` degenerates to exactly the old single-document
  behaviour. No host-side loop, no per-document launch.
- **Per-document offsets computed on device** (`pretok_doc_token_offsets`), so
  the only D2H is the offsets array plus the tokens.
- **Surfaces:** C++ `TokenizerCtx::encode_batch`, CLI `--inputs`, Python
  `encode_batch` / `encode_batch_to_device` (device tensor keeps its owning
  `Tokenizer` alive).
- **Byte-level families:** GPT-2, Llama-3, Qwen-2.5 and DeepSeek-V3. For
  Gemma 3 the CLI's `--inputs` encodes each document in turn on one reused
  context.
- **Gates:** `test_encode_batch_smoke` promoted to a CMake target
  (`test_encode_batch_smoke`, built by every byte-level preset) and wired into
  `scripts/test-all-presets.sh`; new `tests/test_encode_batch.py`. `compute-sanitizer` synccheck+memcheck
  clean on all four byte-level vocabs and both CLI batch paths.
- **Cost:** the per-document clamps also execute in the `n_docs == 1` path.
  Measured single-document kernel-median regression on H100 NVL at the 1M-token
  corpus is **+0.73 % (GPT-2)** and **+1.65 % (Llama-3)**. Batched throughput at
  256 × 4 KiB documents is **10.4× e2e / 7.9× kernel** versus a serial loop.

### Batch interface (F3) — 2026-05-28 (agent/f3) — SUPERSEDED

> Superseded by the GPU-side batched encode above. The host-side
> `apply_batch` aggregation described here has been removed along with the
> host byte-level pre-tokenizer; only the `encode_batch` entry point, the
> `--inputs` CLI flag and the output layout survive, with a different (and now
> seam-exact) implementation.

N independent documents per `encode_batch()` call, single CUDA graph launch.
No kernel change — all the work is host-side aggregation in the
pre-tokenizer plus a doc-offset table built from `d_scan_out`.

- **`PreTokenizer::apply_batch(texts, r, pre_start_per_doc)`** — runs the
  regex pipeline once per document (no matches span doc boundaries, so
  per-doc pre-token bytes are bit-identical to single-doc `apply`). The
  result `r` accumulates all docs into the same short/long buckets with
  globally-monotonic `orig_idx`. `pre_start_per_doc` (length `n_docs+1`)
  records the orig_idx range of each doc.
- **`TokenizerCtx::encode_batch(BatchEncodeInput&, ...)`** — reuses the
  same captured graph as single-doc `encode()` (graph operates on
  `cap_short + cap_long`, so any `n_total <= cap_pretokens` works). After
  the graph launch, D2Hs `d_scan_out[0 .. n_total+1]` and indexes it at
  doc boundaries to produce `out_doc_offsets[n_docs+1]` — each doc's
  tokens are `out_tokens[out_doc_offsets[d] .. out_doc_offsets[d+1]]`.
- **CLI: `--inputs file1,file2,...`** — batch-encodes the listed files in
  one graph launch.
- Single-doc `encode()` is bit-unchanged; all 6 CMake presets still pass.

### F4 — Special-token insertion (BOS/EOS/chat-template) — 2026-05-28 (agent/f4)

Host-side post-step that brings the encoder's bare BPE-id output into parity
with what real LLM inference pipelines expect. No kernel changes.

- **`src/vocab.{h,cc}`**: parse `added_tokens` and `post_processor` from
  tokenizer.json. `HostVocab` now carries `added_tokens`,
  `special_id_by_content`, and `bos_ids`/`eos_ids` resolved from the
  `TemplateProcessing.single` template (including nested under `Sequence`).
- **`EncodeOptions { add_bos, add_eos }`** + `apply_encode_options()` —
  bit-exact match to HF `tokenizers.Tokenizer.encode(text,
  add_special_tokens=True)` on all four vocabs (Llama-3 prepends
  `<|begin_of_text|>`; Qwen-2.5 / GPT-2 / DeepSeek-V3 are no-ops, matching
  HF behavior).
- **`src/special_tokens.{h,cc}`**: hardcoded chat-template applicators for
  Llama-3, Qwen-2.5 (with default-system injection), and DeepSeek-V3,
  emitting a tagged piece stream (raw-text chunks + pre-resolved special-ids)
  that the CLI runs through the existing BPE pipeline piecewise. Bit-exact
  to `transformers.AutoTokenizer.apply_chat_template(..., tokenize=True,
  add_generation_prompt=True)` on all three vocabs across user-only,
  system+user, and multi-turn cases. GPT-2 rejects `--chat` with a clear
  error.
- **CLI**: `--add-bos`, `--add-eos`, `--chat <chat.json>`,
  `--no-generation-prompt`.
- **No regression**: `scripts/test-all-presets.sh` still 6/6 PASS.

### F5 — GPU decode (token IDs → bytes) — 2026-05-28 (agent/f5)

Decoder runs entirely on the GPU as a separate captured CUDA graph, mirroring
the encoder's H2D → graph-launch → D2H pattern.

- **`src/vocab.{h,cc}`**: `build_vocab_pack()` now also concatenates the
  per-token raw bytes (`vocab_token_bytes_concat`) and builds device-side
  `vocab_token_offset` (uint32) and `vocab_token_len` (uint16) lookup tables
  from `HostVocab::token_bytes`. ~0.3–1.0 MB of extra device memory per vocab.
- **`src/tokenizer.{cuh,cu}`**: new public method `TokenizerCtx::decode()`,
  parameterised on `max_decode_tokens` / `max_decode_bytes` constructor caps.
  Implementation is three kernels captured into a dedicated `cudaGraph_t`:
  1. `decode_lens_kernel` — per-token `vocab_token_len[id]` lookup; uses a
     device-resident counter (`d_dec_n_tokens`) so the graph handles any
     `n_tokens ≤ cap` without re-instantiation (tail blocks write len=0).
  2. `cub::DeviceScan::ExclusiveSum` — converts lens to per-token byte offsets.
  3. `decode_gather_kernel` — one block per token, threads cooperatively copy
     `vocab_token_bytes_concat[off..off+len)` to `out_bytes[woff..)`.
- **`src/main.cc`**: new `--decode` flag.
- Measured on H100 NVL (1M-token corpus, 3.83 MB output):

  | Vocab        | kernel   | e2e      | GB/s | Mtok/s | bit-exact |
  | ------------ | -------: | -------: | ---: | -----: | :-------: |
  | GPT-2        | 0.68 ms  | 1.23 ms  | 5.66 |   1476 | ✓         |
  | Llama-3      | 0.62 ms  | 1.45 ms  | 6.19 |   1474 | ✓         |
  | Qwen-2.5     | 0.62 ms  | 1.19 ms  | 6.18 |   1474 | ✓         |
  | DeepSeek-V3  | 0.62 ms  | 1.47 ms  | 6.15 |   1473 | ✓         |

- Gemma 3's `byte_fallback` decoding remains with `F1b`.

### F6 — Python bindings (pybind11) — 2026-05-28 (agent/f6)

- New Python extension module `gpu_bpe_tokenizer` exposing a `Tokenizer`
  class that wraps `HostVocab` + `VocabPack` + `PreTokenizer` +
  `TokenizerCtx`. `encode()` returns `list[int]`, `encode_numpy()` returns
  `np.ndarray[uint32]`. GIL is released over pre-tokenize + GPU encode.
- Buffers and CUDA graph are sized to the next power of two of the input
  and grown lazily when an input exceeds the current cap (one graph
  re-capture, then steady state).
- `pip install .` drives the build via `scikit-build-core`; the CLI build
  is unaffected (`GBPE_BUILD_PYTHON` is OFF by default, ON when SKBUILD
  is set).
- Bit-exact with `tiktoken` (GPT-2) and HF `tokenizers` (Llama-3) on
  `corpus_100k_tokens.txt` (verified by `tests/test_python_bindings.py`).
- Throughput on H100 (1M-token corpus): ~10–14 Mtok/s end-to-end including
  H2D + D2H + Python overhead; 3.5× faster than tiktoken (GPT-2) and 24×
  faster than HF tokenizers (Llama-3) measured from Python.

### Compile-time vocab guards — 2026-05-28 (commit `9ff3c69`)

Per-tokenizer-style optimized builds.

- **`CMakePresets.json`**: `all`, `bytelevel`, `gpt2`, `llama3`, `qwen25`,
  `deepseek_v3`. Single-vocab presets compile only one regex set, drop the
  unused slot policy, and produce binaries ~20% smaller than the all-in
  build.
- **`src/build_config.h`**: derives `GBPE_USE_SLOT64_ONLY` /
  `GBPE_USE_SLOT128_ONLY` / `GBPE_USE_BOTH_SLOTS` and per-vocab `GBPE_HAVE_VOCAB_*`
  from the CMake-supplied flags.
- **`scripts/test-all-presets.sh`**: builds every preset, runs the binary
  against its matching `tokenizer.json`, and bit-exact-checks the output.
- Specialized builds reject incompatible `tokenizer.json` at load time with
  a clear reconfiguration message.

### Qwen-2.5 + DeepSeek-V3 — 2026-05-28 (commit `522f9b6`)

Two more BPE-family vocabs, plus a multi-pass pre-tokenizer for DeepSeek-V3.

- `PreTokenizer` now holds a `vector<pcre2_code*>` and applies the regexes in
  sequence. Single-pass (GPT-2 / Llama-3 / Qwen-2.5) and three-pass
  (DeepSeek-V3) share the same code path.
- New `RegexKind` variants: `Qwen25`, `DeepSeekV3`. Auto-detected by inspecting
  the `tokenizer.json` `pre_tokenizer.Sequence` structure and pattern
  fingerprints.

### Llama-3 — 2026-05-27 (commit `73e909c`)

Slot policy abstraction; the kernel now templates on `<int LEN, typename Slot>`.

- `Slot64` (1 × `uint64` per probe): GPT-2 path (vocab ≤ 65k, merges ≤ 65k).
- `Slot128` (2 × `uint64` per probe): Llama-3 and larger vocabs.
- Bit-exact match to `tokenizers.Tokenizer.encode(text, add_special_tokens=False).ids`
  on 100k / 500k / 1M corpora across L40 / A100 / H100.
- H100 NVL, 1M tokens: 2.86 ms kernel / 4.38 ms E2E / 208 Mtok/s / 537×
  vs HF tokenizers.

### Length bucketing — 2026-05-27 (commit `154bf34`)

Two-bucket design: pre-tokens ≤ 32 bytes go through a `SLOTS_PER_THREAD=1`
kernel; longer pre-tokens go through the `SLOTS_PER_THREAD=2` kernel.

- Pre-token length profiling on the 1M corpus shows
  p99=12 bytes, p99.99=16 bytes, max=42 bytes. 99.9999% of pre-tokens land in
  the short bucket.
- Templated kernel on `int LEN`; the compiler `constexpr`-eliminates row-1
  code in `LEN=32` builds. Registers/thread: 28 → 25.
- H100 1M E2E: 5.67 → 4.36 ms (−23%). 56× → 72× vs `tiktoken`.

### Two warps per block + packed 64-bit merge slot — 2026-05-27 (commit `44e52e0`)

H100-focused optimization pass driven by `ncu` profiling.

- **2 warps per block** removes the `Block Limit Barriers = 32` occupancy cap
  observed in v1. Theoretical occupancy 50% → 100%, achieved 36% → 61%.
- **Packed 64-bit merge-table slot** `[left:16 | right:16 | new_id:16 | rank:16]`
  for GPT-2 (vocab and rank both ≤ 16 bits). One 64-bit load per probe, was
  two. `Stall Long Scoreboard` dropped 26%.
- H100 1M kernel: 3.45 → 2.87 ms. 91× → 110× vs `tiktoken` (kernel-only).

### First working pure C++/CUDA implementation — 2026-05-27 (commit `f697a04`)

The end-to-end kernel, host-side CUDA-graph capture, PCRE2-JIT pre-tokenizer,
and `cub::DeviceScan::ExclusiveSum`-based output flattening. Bit-exact to
`tiktoken` on the 100k / 500k / 1M corpora. H100 NVL: 91× kernel-only speedup
vs `tiktoken`.
