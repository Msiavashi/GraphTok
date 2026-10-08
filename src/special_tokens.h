// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 the GraphTok contributors

#pragma once
// F4: Host-side chat-template applicator.
//
// HuggingFace tokenizer.json does NOT carry the Jinja2 chat_template field —
// that lives in the sibling `tokenizer_config.json`. Rather than ship a
// Jinja parser, we hardcode the canonical templates for the four vocabs
// gpu-bpe-tokenizer supports. Renderers produce a tagged "piece" stream:
// alternating raw-text chunks (to be sent through the BPE pipeline) and
// pre-resolved special-token ids (spliced into the output untouched).

#include "vocab.h"
#include <string>
#include <vector>
#include <cstdint>

namespace gbpe {

// Which hardcoded chat template to use. Auto-detected from HostVocab by
// looking for distinctive special-token contents.
enum class ChatTemplate : uint8_t {
    None = 0,        // GPT-2 — no chat template; encode_chat() will fail
    Llama3,          // <|begin_of_text|>...<|start_header_id|>...
    Qwen25,          // <|im_start|>...<|im_end|>...
    DeepSeekV3,      // <｜begin▁of▁sentence｜><｜User｜>...<｜Assistant｜>
};

ChatTemplate detect_chat_template(const HostVocab& hv);

struct ChatTurn {
    std::string role;     // "system" | "user" | "assistant"
    std::string content;
};

// A piece of the rendered chat: either a raw text chunk (to be BPE-encoded),
// or a special-token id (to be spliced in as-is).
struct ChatPiece {
    enum Kind : uint8_t { Text = 0, Id = 1 } kind;
    std::string text;   // kind == Text
    uint32_t    id;     // kind == Id
};

// Render a chat into pieces. Throws std::runtime_error on unsupported
// templates (e.g. GPT-2) or unknown roles. For Qwen-2.5, if no system turn is
// provided and `add_generation_prompt` is true, the canonical default
// "You are Qwen, created by Alibaba Cloud..." system message is injected to
// match `transformers.apply_chat_template` exactly.
std::vector<ChatPiece> render_chat(const HostVocab& hv,
                                   ChatTemplate tmpl,
                                   const std::vector<ChatTurn>& turns,
                                   bool add_generation_prompt);

// Convenience: render and concatenate the canonical text form (special-token
// contents inlined). Mirrors `apply_chat_template(tokenize=False)`.
std::string render_chat_text(const HostVocab& hv,
                             ChatTemplate tmpl,
                             const std::vector<ChatTurn>& turns,
                             bool add_generation_prompt);

}  // namespace gbpe
