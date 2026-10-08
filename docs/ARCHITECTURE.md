# GraphTok architecture

GraphTok encodes text with a single CUDA graph per input-size class. The graph
contains GPU pre-tokenization, BPE merging and output assembly, so a call is
one H2D copy, one graph replay and one D2H copy. Small inputs can instead take
a CPU route, and a dispatcher chooses between the two under concurrent load.

## 1. Host preparation (once per vocabulary)

`src/vocab.{h,cc}` loads a Hugging Face `tokenizer.json`, detects the
pre-tokenizer family (`detect_regex_kind`) and builds the device-side
`VocabPack` (`build_vocab_pack`): the byte-to-ID table, the merge table as a
packed open-addressing hash, and added/special tokens. The hash slot layout is
`Slot64` when vocab and merge counts fit in 16 bits (GPT-2) and `Slot128`
otherwise (`SlotKind` in `src/tokenizer.cuh`). Single-vocab builds compile only
the matching layout (`src/build_config.h`).

## 2. The captured graph (`src/tokenizer.cu`)

All workspace buffers are allocated up front for the capacity of the size
class; the captured region performs no allocation and no host synchronization.

### 2.1 Pre-tokenization

Pre-token boundaries are computed in parallel as a predicate over a bounded
codepoint window, not by a left-to-right regex scan:

- `pretok_k1_classify` classifies each codepoint with the generated table
  (`generated/gbpe_class_table.h`).
- A per-family boundary kernel (`pretok_k2_boundaries`,
  `pretok_k2_llamaqwen`, `pretok_k2_deepseek`) evaluates `is_boundary[k]`,
  using segmented run scans for constructs such as digit runs. The shared
  rules live in `src/pretok_boundary.h` (`namespace pretok`).
- Prefix scans (`cub::DeviceScan`, `src/small_scan.cuh`) turn the boundary
  flags into pre-token offsets; `pretok_k3_lenbucket` and
  `pretok_k4_materialize` bucket pre-tokens by length and lay them out for the
  merge kernel.
- Added and special tokens are matched on the GPU (`added_match`,
  `added_expand_owners`).

Gemma 3 (SentencePiece family) uses its own graph-resident pre-tokenizer
(`sp_pretok_*` kernels) when built with `GBPE_GEMMA_GPU_PRETOK=ON` (default);
with `OFF` it uses the host pre-tokenizer in `src/pretokenize.cc`.

Qwen 2.5/3 declare NFC normalization. It runs on the host (utf8proc) before
the input is copied to the device (`normalize_input_view` in
`src/python_bindings.cc` and `src/gtok_shim.cc`).

### 2.2 BPE merge

`bpe_kernel<LEN>` assigns one warp per pre-token, two warps per block, for the
two length buckets (`LEN` = 32 and 64 bytes). Each step, the warp looks up the
rank of every adjacent pair in the merge hash, takes the minimum with a
warp-cooperative reduction (`__shfl_xor_sync`) and applies that merge, which
reproduces the reference merge order. Warp rules: every warp-level op runs on
all 32 lanes with mask `0xFFFFFFFF`; inactive lanes carry dead tokens instead
of returning early.

Pre-tokens longer than the buckets go through `overflow_bpe_kernel`.

### 2.3 Assembly

An exclusive scan over per-pre-token output counts gives write offsets, and
the flatten kernels (`flatten_all_kernel`, `flatten_direct_added_kernel`,
`flatten_gpu_overflow_kernel`) write the final IDs contiguously. Batched
encodes (`encode_batch`) additionally produce per-document offsets.

### 2.4 Decode

Decode uses `decode_lens_kernel` and `decode_gather_kernel`, followed by the
SentencePiece post-decoder for Gemma 3.

## 3. Graph size classes

A graph's cost scales with its capacity, so the Python bindings and libgtok
keep one captured graph per power-of-two input-byte class. A class is captured
on first use and kept; each encode replays the smallest class that fits.

## 4. CPU route (`src/cpu_route.{h,cc}`, `src/cpu_thresholds.h`)

Inputs at or below a byte threshold are encoded on the CPU. The threshold is
content-aware (separate limits for mostly-ASCII and non-ASCII text). The engine
is gigatoken (`rust/gtok_cpu`, linked when a nightly cargo is found) or the
built-in host encoder (`src/host_encode.{h,cc}`). It is selected with
`cpu_backend` / `GTOK_CPU_BACKEND`: `auto`, `gigatoken`, `host`, `off`.

## 5. Dispatcher (`src/dispatcher.{h,cc}`)

`Dispatcher` is a thread-safe encoder for many concurrent callers. A request
the CPU route accepts runs on the caller's thread if one of `cpu_workers` CPU
engines is free. Everything else joins a GPU queue; the first caller that finds
no batch in flight launches everything queued as one batched graph replay, and
requests arriving meanwhile form the next batch. There is no timer or
background thread. Policies: `adaptive`, `gpu`, `cpu`.

## 6. Interfaces

- CLI: `src/main.cc` (`gpu_bpe_tokenize`).
- Python: `src/python_bindings.cc` (`Tokenizer`, `Dispatcher`) and
  `python/gpu_bpe_tokenizer/`.
- C ABI: `src/gtok_shim.cc` (`libgtok.so`, exports `gtok_*` only via
  `src/gtok.map`), used by the vLLM plugin and the Dynamo backend.
- Special tokens and chat templates: `src/special_tokens.{h,cc}`.
