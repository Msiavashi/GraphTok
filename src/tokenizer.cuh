// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 the GraphTok contributors

#pragma once
// gpu-bpe-tokenizer — public host-side API of the encoder.
// Pure C++/CUDA. No PyTorch, no Python.

#include "build_config.h"
#include <cstddef>
#include <cstdint>
#include <vector>
#include <string>

namespace gbpe {

// Compile-time tuning. Justified by pre-token length profiling on the corpus:
//   p99=12, p99.99=16, absolute max=42 bytes on the measured corpus.
// Two-bucket design (post-bucketing optimization):
//   SHORT kernel: pre-tokens with len <= 32 bytes  — 1 slot per thread, 1-warp argmin
//   LONG  kernel: pre-tokens with len <= 64 bytes  — 2 slots per thread, 2-row argmin
//
// Empirical justification (data/corpus/corpus_1m_tokens.txt):
//   99.9999% of pre-tokens are <= 32 bytes; the long bucket exists only to
//   handle the rare outlier (e.g. long whitespace runs).
constexpr int MAX_PRETOKEN_LEN_SHORT = 32; // bytes per short-bucket pre-token
constexpr int MAX_PRETOKEN_LEN_LONG  = 64; // bytes per long-bucket pre-token
// Back-compat alias for the fast long-bucket cap. The GPU pre-tokenizer has a
// separate generic path for larger pre-tokens.
constexpr int MAX_PRETOKEN_LEN       = MAX_PRETOKEN_LEN_LONG;

constexpr int WARP_SIZE       = 32;
constexpr int WARPS_PER_BLOCK = 2;   // 2 pre-tokens per block (1 warp each, independent).
constexpr int BLOCK_THREADS   = WARP_SIZE * WARPS_PER_BLOCK; // 64 threads / block

// Sentinel rank meaning "no merge available for this adjacent pair".
// In the 16-bit packed slot, real ranks are < 50000 < 0xFFFF. The 32-bit
// RANK_NONE we use throughout the kernel matches the 16-bit field via
// extraction so we can keep the same value semantics.
constexpr uint32_t RANK_NONE   = 0xFFFFFFFFu;
constexpr uint16_t RANK_NONE16 = 0xFFFFu;
// Sentinel token id meaning "this slot is dead after a merge".
constexpr uint32_t TOKEN_DEAD  = 0xFFFFFFFFu;

// Vocabulary + merge table laid out for GPU consumption.
//
// Two slot layouts are supported, chosen at vocab-load time based on whether
// (vocab_size, merges_count) fit in 16 bits:
//
//   Slot64 (GPT-2 path, fits when vocab <= 65535 and merges <= 65535):
//     One uint64 per slot, packed
//       [right:16 | left:16 | new_id:16 | rank:16]
//     One 64-bit load per probe.
//
//   Slot128 (Llama-3 path, needed when vocab > 65535 or merges > 65535):
//     Two uint64 arrays.
//       keys[i] = (left:32 | right:32)
//       vals[i] = (new_id:32 | rank:32),  rank == 0xFFFFFFFFu marks empty
//     Two 64-bit loads per probe.
//
// The kernel is templated on a SlotPolicy that abstracts the probe; the rest
// of the kernel body is identical for both layouts.
enum class SlotKind : uint8_t { Slot64 = 0, Slot128 = 1 };

// Pre-tokenizer family, for GPU-pretok boundary-stage dispatch. Mirrors
// RegexKind (pretokenize.h) without pulling it into this CUDA-light header.
enum class PretokKind : uint8_t { GPT2 = 0, Llama3 = 1, Qwen25 = 2, DeepSeekV3 = 3, Gemma3 = 4 };

struct VocabPack {
    SlotKind  kind;
    PretokKind pretok_kind = PretokKind::GPT2;  // GPU-pretok boundary-stage selector
    // SP-family (Gemma 3) vocab: pre-tokens come in as uint32 ids instead of
    // bytes; the kernel skips the byte_to_id translation.
    bool      sp_family = false;
    // Slot64 path
    uint64_t* slots64;      // length = capacity, valid iff kind == Slot64
    // Slot128 path
    uint64_t* keys128;      // length = capacity, valid iff kind == Slot128
    uint64_t* vals128;      // length = capacity, valid iff kind == Slot128

    uint32_t  merge_table_capacity;
    uint32_t  merge_table_mask;

    uint32_t* byte_to_id;   // length=256
    uint32_t  vocab_size;

    // Shared AddedVocabulary literal matcher. The compact trie is built from
    // tokenizer.json once at vocabulary load time and is consumed before the
    // family-specific normalizer/pre-tokenizer. root[byte] is the node reached
    // after consuming the first byte; UINT32_MAX means that byte cannot start
    // a literal. Edges for each node are sorted by byte and packed as
    // (next_node << 8) | byte.
    uint32_t* added_root            = nullptr;  // [256]
    uint32_t* added_node_edge_begin = nullptr;  // [added_trie_nodes]
    uint16_t* added_node_edge_count = nullptr;  // [added_trie_nodes]
    uint32_t* added_node_token_id   = nullptr;  // [added_trie_nodes]
    uint64_t* added_edges           = nullptr;  // [added_trie_edges]
    uint32_t  added_trie_nodes      = 0;
    uint32_t  added_trie_edges      = 0;
    uint32_t  added_max_bytes       = 0;

    // Optional direct-vocabulary lookup for BPE model.ignore_merges. The table
    // is keyed by a raw-byte hash and byte-verified via decode-side arrays.
    bool      ignore_merges = false;
    uint64_t* direct_token_hashes = nullptr;
    uint32_t* direct_token_ids    = nullptr;  // TOKEN_DEAD marks an empty slot
    uint32_t  direct_token_mask   = 0;

#if GBPE_HAVE_FAMILY_SP && GBPE_GEMMA_GPU_PRETOK
    // Gemma GPU initial-token lookup. Direct Unicode-scalar indexing;
    // UINT32_MAX means emit byte-fallback IDs.
    uint32_t* sp_codepoint_to_id;  // [0x110000]
    uint32_t* sp_byte_fallback;    // [256]
    // Gemma AddedVocabulary IDs for raw U+2581 runs. Index 0 and 1 are
    // unused; indices 2..31 contain the corresponding greedy run token.
    uint32_t* sp_lowbar_run_ids;   // [32], UINT32_MAX means absent

#endif

    // Decode-side packed arrays (token id -> raw model-token bytes). Built
    // from HostVocab::token_bytes. TokenizerCtx::decode() gathers these bytes;
    // SP ByteFallback post-processing remains an ID-aware host step.
    //   vocab_token_bytes_concat[ vocab_token_offset[id] .. +vocab_token_len[id] ]
    //     -> raw bytes for token `id` (post byte-level decoding).
    uint8_t*  vocab_token_bytes_concat;  // length = vocab_total_bytes
    uint32_t* vocab_token_offset;        // length = vocab_size
    uint16_t* vocab_token_len;           // length = vocab_size  (per-token byte length)
    uint32_t  vocab_total_bytes;
};

// One graph context per (max_pretokens, max_input_bytes) shape bucket.
// Allocates all device workspaces *once*; encode() does only the H2D + graph launch + D2H.
class TokenizerCtx {
public:
    // capacity in pre-tokens per bucket and total output tokens.
    TokenizerCtx(const VocabPack& vp,
                 uint32_t max_short_pretokens,
                 uint32_t max_long_pretokens,
                 uint32_t max_input_bytes,
                 uint32_t max_output_tokens,
                 uint32_t max_decode_tokens = 0,
                 uint32_t max_decode_bytes  = 0,
                 uint32_t max_batch_docs    = 1,
                 bool use_cuda_graph        = true);
    ~TokenizerCtx();

    // Run the GPU encoder. Host-pretokenizer builds consume bucketed inputs;
    // GPU-pretokenizer builds consume EncodeInput::raw_text.
    // `out_tokens` is resized to the actual emitted count.
    void encode(const struct EncodeInput& in,
                std::vector<uint32_t>& out_tokens,
                float* out_kernel_ms,
                float* out_e2e_ms);

    // Like encode(), but writes the token IDs straight into the caller's
    // buffer out[0..max_out) instead of a std::vector: no per-call allocation
    // and one host copy fewer. Returns the token count, or UINT32_MAX if it
    // exceeds max_out (nothing is written then).
    uint32_t encode_into(const struct EncodeInput& in, uint32_t* out, uint32_t max_out);

    // Host wait mode for this context's syncs: false spins (lowest latency,
    // one host core busy for the whole GPU time), true sleeps on a
    // blocking-sync event. Initial value from GBPE_GPU_WAIT=block|spin.
    void set_blocking_wait(bool on) { blocking_wait_ = on; }
    // Speculative D2H: copy an upper bound of the output (one token per input
    // byte at most) in the same stream submission as the replay, so a
    // host-returning encode needs ONE host wait instead of two. Worth it when
    // each wait is a sleep/wake (blocking wait) and inputs are small.
    void set_speculative_d2h(bool on) { speculative_d2h_ = on; }
    // Whether the encode graph has been captured yet (the first launch of a
    // context captures it, which costs tens of milliseconds).
    bool graph_captured() const { return graph_built_; }

#if GBPE_GPU_PRETOK
    // Raw-text encode that leaves the token IDs ON THE DEVICE. Available to
    // byte-level vocabs and to Gemma when GBPE_GEMMA_GPU_PRETOK is enabled.
    // Returns a pointer into d_out_tokens and writes the emitted count to
    // *out_count; no host token copy is performed. The returned pointer aliases
    // an internal workspace buffer that is OVERWRITTEN by the next encode call
    // on this context — it is a borrow valid only until then, never freed by the
    // caller. Arbitrarily long individual pre-tokens are handled by a slower
    // graph-resident overflow path; throws only on capacity/output overflow.
    // `stream` is an opaque cudaStream_t (use the consumer's stream, e.g. torch's
    // current stream, to keep production ordered with an async consumer that
    // reads the returned buffer). A null handle means the CUDA default stream.
    const uint32_t* encode_to_device(const struct EncodeInput& in,
                                     uint32_t* out_count,
                                     float* out_kernel_ms,
                                     float* out_e2e_ms,
                                     void* stream = nullptr);

    // ---- Batched encode. Concatenates n_docs documents into ONE staged input
    // and runs a SINGLE graph replay, with exactly one host synchronization.
    // Per-document token ranges come back in `out_doc_offsets` (n_docs+1
    // entries, running sums; out_doc_offsets[n_docs] == total token count).
    //
    // Seam semantics: documents are concatenated in the byte domain, but each
    // document start is a FORCED pre-token boundary and every pre-tokenizer
    // kernel clamps its neighbour reads and byte extents to the owning
    // document (via d_doc_byte_offsets / d_cp_doc_start). Nothing — digit runs,
    // whitespace runs, contractions, added-token literals, multi-byte UTF-8
    // sequences — can merge across a seam. The result is therefore BIT-EXACT
    // with encoding each document separately and concatenating, for every
    // byte-level family (GPT-2, Llama-3, Qwen-2.5, DeepSeek-V3), on arbitrary
    // seams. Gated by tests/test_encode_batch_smoke.cc.
    //
    // SentencePiece-family contexts (Gemma 3) have no batch path and THROW
    // here for any n_docs, including n_docs == 1; callers that need batching
    // on those vocabs must loop over single-document encode() themselves.
    void encode_batch(const struct BatchEncodeInput& in,
                      std::vector<uint32_t>& out_tokens,
                      std::vector<uint32_t>& out_doc_offsets,
                      float* out_kernel_ms = nullptr,
                      float* out_e2e_ms = nullptr);

    // Device-resident variant of encode_batch(): the token IDs stay in
    // d_out_tokens (same borrow semantics as encode_to_device) and only the
    // n_docs+1 offsets are copied to the host, into `out_doc_offsets_host`.
    const uint32_t* encode_batch_to_device(const struct BatchEncodeInput& in,
                                           uint32_t* out_count,
                                           uint32_t* out_doc_offsets_host,
                                           float* out_kernel_ms = nullptr,
                                           float* out_e2e_ms = nullptr,
                                           void* stream = nullptr);

    // Host-output variant for callers that already own the destination buffer
    // (a NumPy array, say). Writes exactly *out_count tokens straight into
    // out_tokens_host, so the D2H copy lands in the caller's final allocation
    // instead of an intermediate std::vector that then has to be copied again.
    // `out_token_capacity` is in uint32_t entries; the call throws rather than
    // overrunning if the device produces more. Same bit-exactness and seam
    // guarantees as encode_batch().
    void encode_batch_to_host(const struct BatchEncodeInput& in,
                              uint32_t* out_tokens_host,
                              uint32_t out_token_capacity,
                              uint32_t* out_count,
                              uint32_t* out_doc_offsets_host,
                              float* out_kernel_ms = nullptr,
                              float* out_e2e_ms = nullptr);

    // Device-resident results of the last encode, for zero-copy consumers that
    // want to feed token IDs straight into an inference engine's device input
    // buffer with no D2H round trip at all. The count lives on the device too,
    // so reading it costs a copy the caller can choose to skip.
    //
    // Borrowed, not owned: valid until the next encode or launch on this
    // context, which overwrites both buffers.
    const uint32_t* device_tokens_ptr() const { return d_out_tokens; }
    const uint32_t* device_count_ptr()  const { return d_out_total;  }
    void*           device_stream()     const { return stream_;      }
#endif

#if GBPE_GPU_PRETOK
    // Benchmark helpers for measuring the graph in isolation at the Python level.
    // stage_input(): copy raw bytes H2D into the device buffer + capture the
    //   graph if needed + sync. All of this is SETUP — call it OUTSIDE timing.
    // launch_graph_sync(): launch the captured graph and block until it
    //   completes (no D2H, tokens stay on device). This is the ONLY thing to
    //   time. Returns the on-device graph time in milliseconds, measured with
    //   CUDA events bracketing the replay (immune to Python/launch host jitter
    //   and the host-call floor that dominates small inputs).
    void stage_input(const uint8_t* raw_text, uint32_t raw_len);
    // Batch equivalent of stage_input(). Concatenates already-normalized
    // documents into pinned staging, uploads their offsets, captures if needed,
    // and synchronizes. This is benchmark SETUP; call launch_graph_sync()
    // afterwards to obtain the CUDA-event graph time.
    void stage_batch(const struct BatchEncodeInput& in);
    float launch_graph_sync();
    // Untimed correctness readback for the exact batch context most recently
    // staged by stage_batch() and replayed by launch_graph_sync().  Copies the
    // emitted IDs and per-document offsets only after the timed replay has
    // completed; it must never be called inside a benchmark timing sample.
    void read_staged_batch(std::vector<uint32_t>& out_tokens,
                           std::vector<uint32_t>& out_doc_offsets);
    // Decode-side equivalents (separate captured graph): stage token IDs H2D +
    // capture (setup, untimed); launch decode graph + sync, no D2H (timed).
    // Returns the on-device decode-graph time in milliseconds.
    void stage_decode(const uint32_t* token_ids_host, uint32_t n_tokens);
    float launch_decode_sync();
#endif

    // Decode (F5): token IDs -> raw model-token bytes. Runs as its own
    // captured CUDA graph (separate from encode). Callers apply the
    // ID-aware SP ByteFallback post-decoder when HostVocab::byte_fallback is
    // true; byte-level vocabularies need no post-step.
    void decode(const uint32_t* token_ids_host,
                uint32_t n_tokens,
                std::vector<uint8_t>& out_bytes,
                float* out_kernel_ms,
                float* out_e2e_ms);

private:
    void allocate();
    struct LaunchConfig {
        uint32_t short_grid = 0;
        uint32_t long_grid = 0;
        uint32_t flatten_threads = 0;
        uint32_t short_flatten_grid = 0;
        uint32_t long_flatten_grid = 0;
#if GBPE_GPU_PRETOK
        bool use_gpu_pretok = false;
        bool emit_doc_offsets = false;
        uint32_t pretok_grid = 0;
        uint32_t sp_pretok_grid = 0;
        uint32_t doc_offsets_grid = 0;
        uint32_t direct_added_grid = 0;
        // Byte-level pre-tokenizer as one single-CTA kernel (small classes).
        bool tiny_pretok = false;
        const uint32_t* n_short_ptr = nullptr;
        const uint32_t* n_long_ptr = nullptr;
#endif
    };
    // Builds the capacity-specific direct-launch plan once.  In particular,
    // cudaGetDevice/cudaDeviceGetAttribute, environment parsing, and bounded
    // grid selection must not recur on every eager benchmark replay.
    void prepare_launch_config();
    // In graph mode, records the fixed pipeline once. In eager mode, enqueues
    // that identical pipeline directly on `eager_stream` (or stream_ when
    // null); this is used only to measure graph-launch savings.
    void capture_graph(void* eager_stream = nullptr);
    void allocate_decode();
    void capture_decode_graph();

#if GBPE_GPU_PRETOK
    // Shared launch core for the byte-level (raw_text) path used by both
    // encode() and encode_to_device(): stage text H2D, launch the captured
    // graph, read the output count + capacity guard, and validate. Records the
    // kernel-time events on the
    // stream and synchronizes; the caller owns the e2e clock and any D2H copy.
    // Returns d_out_tokens and writes *out_total. `stream` (opaque cudaStream_t)
    // selects the stream all work is issued on.
    const uint32_t* launch_encode_graph(const struct EncodeInput& in,
                                        uint32_t* out_total, void* stream);

    // Batched form of the above, and the single implementation both the
    // single-doc and the batched entry points go through (single-doc is a
    // batch of 1). Stages `n_docs` documents back-to-back into the pinned text
    // buffer, uploads the per-doc byte offsets + doc count, replays the graph
    // and performs ONE synchronization. `out_doc_offsets`, when non-null, is
    // filled with n_docs+1 token offsets read back from d_doc_token_offsets.
    const uint32_t* launch_graph_core(const uint8_t* const* docs,
                                      const uint32_t* lens,
                                      uint32_t n_docs,
                                      uint32_t* out_total,
                                      uint32_t* out_doc_offsets,
                                      void* stream);
#endif

    const VocabPack vocab;
    uint32_t cap_pretokens;       // total max pre-tokens (n_short + n_long bucketed)
    uint32_t cap_short;           // max short-bucket pre-tokens
    uint32_t cap_long;            // max long-bucket pre-tokens
    uint32_t cap_input_bytes;
    uint32_t cap_output_tokens;
    uint32_t cap_docs;            // max documents per batched encode (>= 1)

#if GBPE_GPU_PRETOK
    // ---- GPU pre-tokenizer stage (fused into the graph). Reads raw UTF-8 bytes
    // from d_text_bytes and PRODUCES the same bucketed inputs the host pretok
    // produces today (d_short_bytes/lens/orig_idx, d_long_*). All counts are
    // device-only; grids are cap-sized; tails zeroed in-graph. GPT-2 first.
    uint8_t*  d_text_bytes   = nullptr;   // [cap_input_bytes] raw UTF-8 (device input)
    uint32_t* d_text_len     = nullptr;   // device scalar: live byte count
    // per-byte workspaces [cap_input_bytes]
    uint32_t* d_is_start     = nullptr;   // 1 if byte i starts a codepoint & i<len
    uint8_t*  d_byte_class   = nullptr;   // class at codepoint starts; CONT elsewhere
    uint32_t* d_cp_index     = nullptr;   // exclusive scan of is_start -> cp ordinal
    // per-codepoint dense arrays [cap_input_bytes] (ncp <= nbytes)
    uint8_t*  d_cls          = nullptr;   // class per codepoint
    uint32_t* d_cp_byte_pos  = nullptr;   // byte offset of each codepoint
    uint8_t*  d_cp_byte0     = nullptr;   // first byte of each codepoint (ASCII tests)
#if GBPE_HAVE_VOCAB_DEEPSEEK_V3
    // DeepSeek retains the decoded scalar across the prefix scan.
    uint32_t* d_byte_cp      = nullptr;   // [cap] scalar at each lead byte
    uint32_t* d_cp_cp        = nullptr;   // [cap] dense per-codepoint codepoint value
#endif
    // Digit-run-start scan (Llama \p{N}{1,3} 3-cap + DeepSeek pass1). digit_off
    // = c - digit_run_start[c] is a recurrence to RUN START, NOT a local window
    // — must be a scan, not an in-kernel bounded walk (a long ID is one run but
    // many short pre-tokens). dseed[c] = (cls[c]!=N) ? (c+1) : 0; InclusiveScan
    // (Max) -> drun_start[c] = run start; offset = c - drun_start.
    uint32_t* d_drun_seed    = nullptr;   // added IDs by byte until direct-ID scatter
    uint32_t* d_drun_start   = nullptr;   // digit/run scan output, then overflow locals
    uint32_t* d_is_boundary  = nullptr;   // 1 if a pre-token starts at codepoint c
    uint32_t* d_tok_orig_idx = nullptr;   // Gemma owners, then boundary scan -> order
    uint32_t* d_is_short     = nullptr;   // scan seeds, then byte/ID len <=32
    uint32_t* d_is_long      = nullptr;   // Gemma match lengths, then ID len 33..64
    uint32_t* d_short_local  = nullptr;   // DeepSeek span starts, then scan of is_short
    uint32_t* d_long_local   = nullptr;   // exclusive scan of is_long
    uint32_t* d_ncp          = nullptr;   // device scalar: codepoint count
    // Live bucket counts (device-only). Let bpe_kernel + flatten bound themselves
    // BY LENGTH (pre_idx >= *n) instead of relying on zeroed-tail sentinels — so
    // no per-prompt memset of lens/count arrays is needed.
    uint32_t* d_n_short      = nullptr;   // device scalar: live short pre-tokens
    uint32_t* d_n_long       = nullptr;   // device scalar: live long pre-tokens
    uint32_t* d_n_total      = nullptr;   // device scalar: live pre-tokens total
    uint32_t* d_out_total    = nullptr;   // device scalar: final token count (= scan_out[n_total])
    // Generic GPU overflow path for pre-tokens with more than 64 initial
    // elements. Descriptors are bucket-local; state is indexed by the
    // pre-token's raw-byte/flat-ID start, so total workspace is bounded by the
    // already-fixed input capacity rather than a per-pre-token limit.
    uint32_t* d_n_overflow       = nullptr; // device scalar: live >64 pre-tokens
    uint32_t* d_overflow_start   = nullptr; // [cap_pretokens]
    uint32_t* d_overflow_len     = nullptr; // [cap_pretokens]
    uint32_t* d_overflow_orig    = nullptr; // [cap_pretokens]
    uint32_t* d_overflow_parts   = nullptr; // [cap_input_bytes]
    uint32_t* d_overflow_prev    = nullptr; // [cap_input_bytes]
    uint32_t* d_overflow_next    = nullptr; // [cap_input_bytes]
    uint64_t* d_overflow_tree    = nullptr; // [4 * cap_input_bytes]
    uint32_t* d_pretok_errors    = nullptr; // device scalar: bucket/descriptor capacity errors
#if GBPE_HAVE_FAMILY_SP && GBPE_GEMMA_GPU_PRETOK
    // Gemma uses d_byte_class as uint8 emission lengths; d_cp_index holds their
    // scan offsets. d_drun_start/d_is_short carry the raw-U+2581 run scan, and
    // d_tok_orig_idx carries direct owners until the later boundary scan.
    uint32_t* d_sp_flat_ids      = nullptr; // [cap_input_bytes]
    uint32_t* d_sp_segment_start = nullptr; // [cap_input_bytes + 1]
#endif
    // class table (uploaded once, outside capture)
    void*     d_class_blocks = nullptr;   // uint16 block-index
    void*     d_class_leaves = nullptr;   // packed 2-bit leaves
    // extra CUB scan temps for the pretok scans (sized once)
    void*     d_pt_scan_temp = nullptr;
    size_t    pt_scan_temp_bytes = 0;
    uint8_t*  h_pin_text_bytes = nullptr; // pinned staging for raw text H2D
    // One pinned + one device header block carries the length scalar, the
    // document count and the doc byte offsets, so the per-call H2D is two
    // copies (text + header) instead of four. The named pointers below alias
    // into these blocks and are not freed separately.
    uint32_t* h_pin_hdr = nullptr;        // [2 + cap_docs + 1]
    uint32_t* d_hdr     = nullptr;        // [2 + cap_docs + 1]
    uint32_t* h_pin_text_len   = nullptr; // = h_pin_hdr + 0
    // Mapped pinned status block written by the graph itself: [0] = final
    // token count, [1] = pretok error count. Read on the host after the
    // stream sync with no D2H copy. d_out_total / d_pretok_errors alias it.
    uint32_t* h_pin_status = nullptr;
    uint32_t* d_status     = nullptr;
    uint32_t* h_pin_out_tokens = nullptr; // pinned staging for the token D2H

    // ---- Batched-encode plumbing. The staged text buffer is the concatenation
    // of the batch's documents; these describe where each document lives.
    uint32_t* d_doc_byte_offsets = nullptr; // [cap_docs+1] byte start of each doc
    uint32_t* d_n_docs           = nullptr; // device scalar: live document count
    // Dense per-codepoint doc-start ordinal: cp_doc_start[c] is the codepoint
    // ordinal of the first codepoint of c's document (c is a doc-start iff
    // cp_doc_start[c] == c). Allocated (and zeroed) here; written by
    // pretok_gather_cp and consumed by the doc-boundary clamps.
    uint32_t* d_cp_doc_start     = nullptr; // [cap_input_bytes]
    uint32_t* d_doc_token_offsets = nullptr; // [cap_docs+1] token start of each doc
    uint32_t* h_pin_doc_byte_offsets = nullptr;  // pinned [cap_docs+1]
    uint32_t* h_pin_n_docs           = nullptr;  // pinned scalar
    uint32_t* h_pin_doc_token_offsets = nullptr; // pinned [cap_docs+1]
    uint32_t staged_batch_n_docs_ = 0;
    bool staged_batch_pending_ = false;
    bool staged_batch_replayed_ = false;
#endif

    // Per-bucket device input buffers.
    uint8_t*  d_short_bytes    = nullptr;  // [cap_short, MAX_PRETOKEN_LEN_SHORT]
    uint16_t* d_short_lens     = nullptr;  // [cap_short]
    uint32_t* d_short_orig_idx = nullptr;  // [cap_short]
    uint8_t*  d_long_bytes     = nullptr;  // [cap_long, MAX_PRETOKEN_LEN_LONG]
    uint16_t* d_long_lens      = nullptr;  // [cap_long]
    uint32_t* d_long_orig_idx  = nullptr;  // [cap_long]

    // Shared output workspaces (gather pattern: indexed by orig_idx).
    uint32_t* d_per_pre_count  = nullptr;  // [cap_pretokens + 1]
    // Per-bucket output payloads (gather by bucket-local index; flatten maps
    // local->orig_idx). Short rows are 32-wide, long rows 64-wide — halving the
    // footprint vs the old unified [cap_pretokens, 64] buffer.
    uint32_t* d_per_pre_tokens_short = nullptr;  // [cap_short, MAX_PRETOKEN_LEN_SHORT]
    uint32_t* d_per_pre_tokens_long  = nullptr;  // [cap_long,  MAX_PRETOKEN_LEN_LONG]
#if GBPE_HAVE_FAMILY_SP
    // SP-family overflow payloads (pre-BPE'd splices), 64-wide, keyed by a
    // dedicated local index; flatten_overflow maps local->orig_idx.
    uint32_t* d_per_pre_tokens_overflow = nullptr;  // [cap_overflow, MAX_PRETOKEN_LEN_LONG]
    uint32_t* d_overflow_orig_idx       = nullptr;  // [cap_overflow]
    uint32_t* d_overflow_n              = nullptr;  // device scalar: live overflow count
    uint32_t  cap_overflow              = 0;
#endif
    uint32_t* d_scan_out       = nullptr;  // [cap_pretokens + 1]
    uint32_t* d_out_tokens     = nullptr;  // [cap_output_tokens]
    // Direct AddedVocabulary IDs keyed by original pre-token index. Only
    // synthetic CLS_ADDED entries read this buffer during final flattening.
    uint32_t* d_direct_ids     = nullptr;  // [cap_pretokens]

    // CUB scan temp.
    void*   d_scan_temp     = nullptr;
    size_t  scan_temp_bytes = 0;

    // Stream + graph.
    void* stream_   = nullptr;   // cudaStream_t (opaque to keep .cuh CUDA-light)
    void* graph_    = nullptr;   // cudaGraph_t
    void* graph_ex_ = nullptr;   // cudaGraphExec_t
    void* ev_start_ = nullptr;   // cudaEvent_t
    void* ev_stop_  = nullptr;   // cudaEvent_t
    // Host wait for the stream. Spinning (cudaStreamSynchronize under the
    // default schedule) costs one host core for the whole GPU time; with
    // blocking_wait_ the caller sleeps on a blocking-sync event instead.
    // Default from GBPE_GPU_WAIT=block|spin (spin when unset).
    void* ev_wait_  = nullptr;   // cudaEvent_t (BlockingSync, no timing)
    bool  blocking_wait_ = false;
    bool  speculative_d2h_ = false;
    bool  last_d2h_done_ = false;   // set by launch_graph_core
    void  wait_stream(void* stream);
    void* aux_stream_[2] = {nullptr, nullptr};  // cudaStream_t, BPE fork branches
    void* fork_ev_ = nullptr;                   // cudaEvent_t (no timing)
    void* join_ev_[2] = {nullptr, nullptr};     // cudaEvent_t (no timing)
    bool  graph_built_ = false;
    bool  use_cuda_graph_ = true;
    bool  eager_warmed_ = false;
    LaunchConfig launch_config_{};
    bool launch_config_ready_ = false;

    // Pinned host staging per bucket.
    uint8_t*  h_pin_short_bytes    = nullptr;
    uint16_t* h_pin_short_lens     = nullptr;
    uint32_t* h_pin_short_orig_idx = nullptr;
    uint8_t*  h_pin_long_bytes     = nullptr;
    uint16_t* h_pin_long_lens      = nullptr;
    uint32_t* h_pin_long_orig_idx  = nullptr;

    // ---- F5: Decode workspaces (separate captured graph). ----
    uint32_t  cap_decode_tokens = 0;
    uint32_t  cap_decode_bytes  = 0;
    uint32_t* d_dec_token_ids   = nullptr;
    uint32_t* d_dec_n_tokens    = nullptr;
    uint32_t* d_dec_lens        = nullptr;
    uint32_t* d_dec_offsets     = nullptr;
    uint8_t*  d_dec_out_bytes   = nullptr;
    void*     d_dec_scan_temp   = nullptr;
    size_t    dec_scan_temp_bytes = 0;
    uint32_t* h_pin_dec_token_ids = nullptr;
    void* dec_graph_     = nullptr;
    void* dec_graph_ex_  = nullptr;
    void* dec_ev_start_  = nullptr;
    void* dec_ev_stop_   = nullptr;
    bool  dec_graph_built_ = false;

#if GBPE_HAVE_FAMILY_SP
    // ---- F1b: SP-family path (Gemma 3). Per-pre-token inputs are uint32_t
    // initial token ids instead of raw bytes. The kernel reads `_ids` arrays
    // directly and skips the byte_to_id translation.
    uint32_t* d_short_ids   = nullptr;
    uint32_t* d_long_ids    = nullptr;
    uint32_t* h_pin_short_ids = nullptr;
    uint32_t* h_pin_long_ids  = nullptr;
    bool      sp_mode_ = false;
#endif
};

// Input to TokenizerCtx::encode(). The caller fills these views; the
// context copies them into pinned host buffers and stages H2D.
//
// The default Gemma build consumes raw_text through the GPU pre-tokenizer and
// ignores the host bucket fields. The retained CPU comparison build fills
// short_ids/long_ids plus an overflow list instead.
struct EncodeInput {
    const uint8_t*  short_bytes;
    const uint16_t* short_lens;
    const uint32_t* short_orig_idx;
    uint32_t        n_short;
    const uint8_t*  long_bytes;
    const uint16_t* long_lens;
    const uint32_t* long_orig_idx;
    uint32_t        n_long;
    uint32_t        n_total;   // n_short + n_long (+ n_overflow for SP)

#if GBPE_GPU_PRETOK
    // GPU pre-tokenizer mode: the host supplies only the raw UTF-8 bytes; the
    // bucketing (short/long byte buffers, lens, orig_idx, all counts) is
    // produced on the device inside the captured graph. The byte-bucket fields
    // above are ignored in this mode.
    const uint8_t*  raw_text = nullptr;
    uint32_t        raw_len  = 0;
#endif

#if GBPE_HAVE_FAMILY_SP
    // SP-family alternate inputs (used iff TokenizerCtx was constructed with
    // an SP-family vocab). short_bytes/long_bytes are ignored in that case.
    const uint32_t* short_ids = nullptr;   // [n_short, MAX_PRETOKEN_LEN_SHORT]
    const uint32_t* long_ids  = nullptr;   // [n_long,  MAX_PRETOKEN_LEN_LONG]
    // Overflow: pre-BPE'd results to splice into per_pre_tokens device rows.
    const uint32_t* overflow_orig_idx    = nullptr;
    const uint32_t* overflow_result_ids  = nullptr;  // [n_overflow, MAX_PRETOKEN_LEN_LONG]
    const uint16_t* overflow_result_lens = nullptr;  // [n_overflow]
    uint32_t        n_overflow           = 0;
#endif
};

// Input to TokenizerCtx::encode_batch(). The documents are already normalized
// UTF-8 byte views owned by the caller; the context concatenates them into its
// pinned staging buffer.
struct BatchEncodeInput {
    const uint8_t* const* docs;   // n_docs pointers to normalized UTF-8 doc bytes
    const uint32_t*       lens;   // [n_docs]
    uint32_t              n_docs; // >= 1
};

}  // namespace gbpe
