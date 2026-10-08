# Architecture

cuTokenize tokenizes **byte-level BPE** vocabs (GPT-2, Llama-3, Qwen-2.5,
DeepSeek-V3) entirely on the GPU. Pre-tokenization, BPE merging, and output
assembly all run inside a single captured CUDA graph: raw UTF-8 bytes go in,
token IDs come out — no host pre-tokenizer on the hot path.

Bit-exactness with `tiktoken` (GPT-2) and HuggingFace `tokenizers` (the rest) is
a hard gate.

## Pipeline

Everything between the H2D and D2H copies is one `cudaGraph_t`, replayed per call
with `cudaGraphLaunch`. No allocations, no D2H, no host control flow inside it.

```
                 ┌──────────────────────── captured CUDA graph ───────────────────────┐
 raw bytes ─H2D─►│  ADDED VOCAB ─► PRE-TOKENIZE           BPE BACK-END (shared)        │─D2H─► token IDs
 (d_text_bytes)  │  trie match      k1 classify/gather ─► bpe_kernel ─► scan ─►         │  (d_out_tokens)
                 │  + ownership     k2 boundary/k4         flatten + direct IDs          │
                 └────────────────────────────────────────────────────────────────────┘
```

### Stage detail

```
    added_match             1 thread/byte  longest literal from compact device trie
    added_expand_owners     1 thread/match hide literal bytes; retain one synthetic direct-ID entry
k1  pretok_k1_classify     1 thread/byte   UTF-8 decode + class (L/N/space/other/added) via lookup table
    ExclusiveSum(is_start)                 → codepoint ordinals (CUB)
    pretok_gather_cp        1 thread/byte  → dense per-codepoint arrays (class, byte-pos, cp)
k2  pretok_k2_<family>      1 thread/cp    boundary predicate → is_boundary[]   ◄── the only per-vocab kernel
    ExclusiveSum × 3                       → orig_idx (doc order), short/long bucket indices
k4  pretok_k4_materialize   1 block/token  scatter bytes into bucketed rows + lens + orig_idx
─────────────────────────────────────────  k1–k4 produce the bucketed pre-tokens the BPE stage consumes
bpe bpe_kernel<32|64>       1 warp/token   register-resident fast merge path
    overflow_bpe_kernel     1 warp/token   global-workspace fallback for >64 elements
    ExclusiveSum(counts)                   → output offsets
    flatten_kernel_bucketed 1 warp/token   bounded grid-stride gather into contiguous output
    flatten_direct_added    1 thread/entry write AddedVocabulary IDs at scanned offsets
```

## The two design ideas

**1. Parallelism is *across* pre-tokens, never within one.** A 1M-token document
is ~270k independent pre-tokens. Each pre-token's merge sequence is strictly
sequential (left-to-right tie-breaking is contractual), but pre-tokens are
independent → one warp per pre-token. A fixed captured grid processes the full
device-resident active count with a grid-stride loop, preserving sequence
parallelism without admitting a capacity-sized stream of mostly idle CTAs.

**2. The per-vocab seam is `is_boundary[]`.** Pre-tokenizer regexes differ only
in *where* pre-token boundaries fall. So the only per-family code is the k2
kernel that fills `is_boundary[]`; classification (k1), bucketing (k3/k4), the
BPE back-end, and output assembly are byte-identical across all families. Boundary
decisions are *local* (a bounded codepoint window + a few parallel run-scans), so
every codepoint decides independently — no serial scan of the text.

## Component model

```
┌───────────────┐   builds    ┌──────────────┐   feeds    ┌────────────────────┐
│  HostVocab    │────────────►│  VocabPack   │───────────►│  TokenizerCtx      │
│ (tokenizer.   │             │ (device:     │            │ (allocates device  │
│  json loader, │             │  merge table,│            │  buffers, captures │
│  vocab.{h,cc})│             │  class table,│            │  & replays graph)  │
└───────────────┘             │  pretok_kind)│            │  tokenizer.{cuh,cu}│
                              └──────────────┘            └─────────┬──────────┘
                                                                    │ exposed by
                                              ┌─────────────────────┴───────────────────┐
                                              ▼                                          ▼
                                       main.cc (CLI)                    python_bindings.cc (Tokenizer)
```

`TokenizerCtx` is the engine: one context per (max-bytes, max-tokens) shape; it
allocates all device workspace once and captures the graph on first use, then
every `encode()` is just H2D + graph launch + D2H.

AddedVocabulary matching is shared by every family. `vocab.cc` converts literal
entries into a sparse byte trie once; the graph matches raw input before regex
pre-tokenization and interleaves direct IDs without replaying text fragments on
the host. Prefix literals at the same byte use longest-match semantics. The
loader currently requires non-overlapping literal starts and rejects
`single_word`, strip flags, and unproven normalizer combinations explicitly.
Gemma's structured newline/tab/U+2581 runs remain a small family extension.

## The BPE merge kernel (`bpe_kernel`)

The fast path uses one warp (32 threads) per pre-token, two pre-tokens per
block. It mirrors the reference greedy algorithm (`tiktoken`'s
`_byte_pair_merge`):

```
parts[i]  = byte-id of each input byte (≤64 per pre-token, 2 slots/thread)
ranks[i]  = merge_table[(parts[i], parts[i+1])].rank        # parallel across i
loop (bounded by length):
    (min_rank, min_idx) = warp_argmin(ranks)                # __shfl_xor_sync butterfly
    if min_rank == NONE: break                              # data-dependent exit, graph-legal
    parts[min_idx] = merge_table[...].new_id ; kill parts[min_idx+1]
    rebuild the two ranks adjacent to the merge                 (find_prev/next alive via __ballot_sync)
```

Tie-breaking uses `(rank<<32)|idx` so the lowest index wins on ties — matching
`tiktoken`/HF/llama.cpp left-to-right. The **merge table** is an open-addressing
hash, L2-resident: `Slot64` (one 64-bit load/probe) for vocabs ≤64k tokens
(GPT-2), `Slot128` (two loads) for larger vocabs.

Pre-tokens over 64 elements use a graph-resident correctness fallback. One warp
builds a power-of-two tournament tree of `(rank,index)` keys in preallocated
global memory. The root supplies the next exact greedy merge; afterward only
the merged-left, killed-right, and predecessor leaf paths are rebuilt. This is
`O(L + M log L)` for length `L` and `M` merges, and removes the length failure
boundary without adding register pressure to the measured 32/64 fast paths.
Each tree is padded to a power of two and placed at `4 * input_start`; a shared
`4 * max_input_bytes` uint64 workspace therefore guarantees disjoint storage
without allocation or a device-dependent graph shape.

## Hard constraints (enforced)

- **Graph-capturable:** no alloc / D2H / sync / host-branch-on-device-data inside
  capture. Per-token counts are device-only; fixed capture-time grids consume
  active rows with device-bounded grid-stride loops (no host-known sizes
  mid-graph).
- **Bit-exact:** any change to the merge kernel, bucket policy, or boundary
  predicate must reproduce the reference token IDs on the validation corpus +
  the differential edge set. See [BUILD.md](BUILD.md) for the gate commands.

## Compile-time decoupling

A single-vocab build compiles only its own kernels. `GBPE_VOCABS=GPT2` →
`Slot64` + the GPT-2 boundary kernel only; multi-vocab/`ALL` → runtime dispatch
on `VocabPack::pretok_kind` and `kind`. See `src/build_config.h`.

## Batched encode (`encode_batch`)

N documents in **one** graph replay, bit-exact with per-document `encode()`.
Byte-level families only; SP-family contexts (Gemma 3) throw.

- **Upload.** The documents are concatenated into the pinned staging buffer and
  copied H2D once. Alongside them go `d_doc_byte_offsets[n_docs+1]` — the running-sum
  byte offsets of each document, uploaded in the same staged copy. `n_docs`
  itself is uploaded H2D into the device scalar `d_n_docs` before launch and
  read in-kernel (`*n_docs_ptr`); every kernel bounds its grid/loop against
  that device scalar rather than a host-known count, which is exactly what
  lets the graph be captured once and replayed for any live document count up
  to `cap_docs`.
- **Codepoint domain.** `pretok_gather_cp` writes `d_cp_doc_start`, a dense
  per-codepoint array (sized `cap_input_bytes`): `cp_doc_start[c]` holds the
  codepoint ordinal of the first codepoint of `c`'s document. A codepoint `c`
  is therefore a doc-start iff `cp_doc_start[c] == c`, and two codepoints
  belong to the same document iff their `cp_doc_start` values are equal.
  Everything downstream of the gather clamps against `d_cp_doc_start` rather
  than re-deriving document boundaries from bytes.
- **Forced boundaries.** Every document start is unconditionally a pre-token
  boundary. That alone is not sufficient — a boundary predicate that *looks
  backwards* would still see the previous document's last codepoint — so each
  family's `pretok_k2_*` kernel clamps its neighbour reads to the owning
  document. A codepoint at a document start behaves exactly as if it were at
  index 0 of a standalone input: no digit-run continuation, no whitespace-run
  continuation, no contraction tail, no added-token literal spanning the seam.
  `added_match` is clamped in the byte domain for the same reason.
- **The load-bearing subtlety: k3/k4 byte extents.** `pretok_k3_lenbucket` and
  `pretok_k4_materialize` convert a pre-token's codepoint range back into a
  byte range to bucket it by length and to copy its bytes. Without a clamp,
  the byte extent of the *last* pre-token of a document runs to the start of
  the next document's first pre-token, silently pulling foreign bytes into the
  pre-token. Both kernels therefore binary-search the owning document and clamp
  the extent to that document's byte range. This is the one place where
  omitting a clamp produces wrong tokens rather than merely wrong boundaries.
- **Offsets.** `pretok_doc_token_offsets` runs inside the graph and turns the
  per-pre-token token counts into `n_docs+1` running sums, so the only host
  transfers are the tokens and that small offsets array — one D2H, one sync,
  regardless of `n_docs`.
- **`n_docs == 1` degenerates.** With a single document, `d_doc_byte_offsets` is
  `{0, len}` and every clamp resolves to the full input, so the batched path is
  the single-document path. That is why single-document `encode()` and
  `encode_batch({doc})` share one captured graph and one code path; it is also
  why the clamps cost single-document throughput (~+0.7 % GPT-2 / ~+1.7 %
  Llama-3 kernel median at 1M tokens on H100).

## Gemma long components

- Gemma 3 defaults to the captured GPU pre-tokenizer (`gemma3`). Because its
  Replace normalizer makes the serialized Split pre-tokenizer a no-op, the GPU
  path partitions ordinary text at merge-table-proven independent
  space/newline/tab components. Most components then use the 32/64-slot fast
  kernels; the graph-resident tournament kernel handles any remainder above
  64 initial IDs. The retained `gemma3_cpu` preset remains a comparison path.

## Source map

| File | Role |
|---|---|
| `tokenizer.{cuh,cu}` | GPU pre-tokenizer + BPE kernels + graph capture/replay; `TokenizerCtx` |
| `vocab.{h,cc}` | `tokenizer.json` loader; device merge-table + class-table builder |
| `pretokenize.{h,cc}` | host SP-family (Gemma) pre-tokenizer; `estimate_max_pretokens` |
| `special_tokens.{h,cc}` | BOS/EOS + chat-template (host post-step) |
| `build_config.h` | compile-time vocab/slot guards |
| `main.cc` / `python_bindings.cc` | CLI / Python `Tokenizer` |
| `generated/gbpe_class_table.h` | codepoint→class table (generated by `tools/gen_class_table.py`) |
