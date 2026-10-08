// SPDX-License-Identifier: Apache-2.0
// GENERATED from a CPU/GPU crossover sweep
//   (2026-09-24 measurement).
// (NVIDIA H100 NVL; rule: largest size up to which cpu p50 <= gpu p50; ascii = worse of books/code, multilingual = multilingual). Do not edit by hand.
#pragma once
#include "pretokenize.h"
#include <cstdint>

namespace gbpe {

// Default CPU-route thresholds (raw input bytes) per pre-tokenizer family:
// .ascii for text with little non-ASCII, .multilingual for the rest.
struct CpuThresholds { uint32_t ascii; uint32_t multilingual; };

inline CpuThresholds cpu_default_thresholds(RegexKind k) {
    switch (k) {
        case RegexKind::GPT2: return {16384u, 4096u};
        case RegexKind::Llama3: return {16384u, 4096u};
        case RegexKind::DeepSeekV3: return {16384u, 4096u};
        case RegexKind::Qwen25: return {16384u, 4096u};
        default: return {4096u, 4096u};
    }
}

}  // namespace gbpe
