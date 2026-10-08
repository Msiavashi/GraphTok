// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 the cuTokenize contributors

#pragma once
// gpu-bpe-tokenizer — compile-time configuration derived from CMake flags.
//
// The CMake build sets these macros via target_compile_definitions:
//   GBPE_HAVE_VOCAB_GPT2          1 / 0   GPT-2 vocab support compiled in
//   GBPE_HAVE_VOCAB_LLAMA3        1 / 0
//   GBPE_HAVE_VOCAB_QWEN25        1 / 0
//   GBPE_HAVE_VOCAB_DEEPSEEK_V3   1 / 0
//   GBPE_HAVE_VOCAB_GEMMA3        1 / 0
//   GBPE_HAVE_FAMILY_BYTELEVEL    1 / 0   any of GPT-2/Llama-3/Qwen/DeepSeek
//   GBPE_HAVE_FAMILY_SP           1 / 0   Gemma 3 (SP-derived BPE) — when added
//   GBPE_SLOT_ONLY                64 / 128 / undefined
//       Defined only in single-vocab builds where the slot policy is fixed.
//       Lets the kernel be non-templated on Slot, saving template instantiations
//       and letting nvcc inline the probe across the kernel body.
//
// Defaults (so that "compile this file" alone works in IDEs etc.): treat as ALL.

#if !defined(GBPE_HAVE_VOCAB_GPT2)
#  define GBPE_HAVE_VOCAB_GPT2 1
#endif
#if !defined(GBPE_HAVE_VOCAB_LLAMA3)
#  define GBPE_HAVE_VOCAB_LLAMA3 1
#endif
#if !defined(GBPE_HAVE_VOCAB_QWEN25)
#  define GBPE_HAVE_VOCAB_QWEN25 1
#endif
#if !defined(GBPE_HAVE_VOCAB_DEEPSEEK_V3)
#  define GBPE_HAVE_VOCAB_DEEPSEEK_V3 1
#endif
#if !defined(GBPE_HAVE_VOCAB_GEMMA3)
#  define GBPE_HAVE_VOCAB_GEMMA3 0   /* off until F1b lands */
#endif

#if !defined(GBPE_HAVE_FAMILY_BYTELEVEL)
#  if GBPE_HAVE_VOCAB_GPT2 || GBPE_HAVE_VOCAB_LLAMA3 || GBPE_HAVE_VOCAB_QWEN25 || GBPE_HAVE_VOCAB_DEEPSEEK_V3
#    define GBPE_HAVE_FAMILY_BYTELEVEL 1
#  else
#    define GBPE_HAVE_FAMILY_BYTELEVEL 0
#  endif
#endif
#if !defined(GBPE_HAVE_FAMILY_SP)
#  if GBPE_HAVE_VOCAB_GEMMA3
#    define GBPE_HAVE_FAMILY_SP 1
#  else
#    define GBPE_HAVE_FAMILY_SP 0
#  endif
#endif

// GBPE_SLOT_ONLY is intentionally left undefined in multi-vocab builds.
// Helpers:
//   GBPE_USE_SLOT64_ONLY    -> 1 if compiled as Slot64-only (GPT-2)
//   GBPE_USE_SLOT128_ONLY   -> 1 if compiled as Slot128-only (Llama-3 / Qwen / DS / Gemma)
//   GBPE_USE_BOTH_SLOTS     -> 1 if both compiled in (runtime dispatch)
#if defined(GBPE_SLOT_ONLY) && (GBPE_SLOT_ONLY == 64)
#  define GBPE_USE_SLOT64_ONLY   1
#  define GBPE_USE_SLOT128_ONLY  0
#  define GBPE_USE_BOTH_SLOTS    0
#elif defined(GBPE_SLOT_ONLY) && (GBPE_SLOT_ONLY == 128)
#  define GBPE_USE_SLOT64_ONLY   0
#  define GBPE_USE_SLOT128_ONLY  1
#  define GBPE_USE_BOTH_SLOTS    0
#else
#  define GBPE_USE_SLOT64_ONLY   0
#  define GBPE_USE_SLOT128_ONLY  0
#  define GBPE_USE_BOTH_SLOTS    1
#endif

// Convenience: total count of vocabs compiled in.
#define GBPE_VOCAB_COUNT ( \
    GBPE_HAVE_VOCAB_GPT2          + \
    GBPE_HAVE_VOCAB_LLAMA3        + \
    GBPE_HAVE_VOCAB_QWEN25        + \
    GBPE_HAVE_VOCAB_DEEPSEEK_V3   + \
    GBPE_HAVE_VOCAB_GEMMA3)

#if GBPE_VOCAB_COUNT == 0
#  error "No vocabs enabled — at least one of GBPE_HAVE_VOCAB_* must be 1"
#endif

// GPU pre-tokenizer stage (fused into the capture graph). This is now the ONLY
// byte-level pre-tokenization path — the legacy host PCRE2 byte-level pipeline
// has been removed. Always on for byte-level families. (Kept as a macro so the
// pretok device buffers/kernels stay clearly scoped. SP/Gemma uses the device
// front-end by default; GBPE_GEMMA_GPU_PRETOK=0 retains the host fallback.)
#if !defined(GBPE_GPU_PRETOK)
#  define GBPE_GPU_PRETOK 1
#endif

// Gemma defaults to captured normalization, initial-ID lookup/fallback, and
// bucketing. Set to zero to retain the original host SPPreTokenizer path.
#if !defined(GBPE_GEMMA_GPU_PRETOK)
#  define GBPE_GEMMA_GPU_PRETOK 1
#endif

// ---------------------------------------------------------------------------
// Diagnostic logging. Informational/setup lines ([vocab] loaded / device table
// / decode arrays, [ctx] capacities) and the pre_tokenizer WARNING are OFF by
// default; they print only when the environment variable GBPE_VERBOSE is set.
// FATAL messages and recoverable runtime errors are NOT routed through this and
// always print. Use GBPE_LOG(...) exactly like std::fprintf(stderr, ...).
#ifdef __cplusplus
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
inline bool gbpe_verbose() {
    static const bool v = (std::getenv("GBPE_VERBOSE") != nullptr);
    return v;
}
inline void gbpe_log(const char* fmt, ...) {
    if (!gbpe_verbose()) return;
    std::va_list ap;
    va_start(ap, fmt);
    std::vfprintf(stderr, fmt, ap);
    va_end(ap);
}
#define GBPE_LOG(...) gbpe_log(__VA_ARGS__)
#endif  // __cplusplus
