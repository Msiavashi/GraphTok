// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 the GraphTok contributors

#include "special_tokens.h"

#include <stdexcept>
#include <sstream>

namespace gbpe {

namespace {

bool has_content(const HostVocab& hv, const char* c) {
    return hv.special_id_by_content.find(c) != hv.special_id_by_content.end();
}

uint32_t require_id(const HostVocab& hv, const char* c) {
    auto it = hv.special_id_by_content.find(c);
    if (it == hv.special_id_by_content.end()) {
        std::ostringstream os;
        os << "[special_tokens] required special token '" << c
           << "' not present in vocab";
        throw std::runtime_error(os.str());
    }
    return it->second;
}

// DeepSeek-V3 uses fullwidth/Unicode special-token delimiters:
//   "<｜begin▁of▁sentence｜>"  U+FF5C, U+2581
constexpr const char* DS_BOS  = "<\xef\xbd\x9c" "begin" "\xe2\x96\x81" "of" "\xe2\x96\x81" "sentence" "\xef\xbd\x9c>";
constexpr const char* DS_EOS  = "<\xef\xbd\x9c" "end" "\xe2\x96\x81" "of" "\xe2\x96\x81" "sentence" "\xef\xbd\x9c>";
constexpr const char* DS_USER = "<\xef\xbd\x9cUser\xef\xbd\x9c>";
constexpr const char* DS_ASST = "<\xef\xbd\x9c" "Assistant" "\xef\xbd\x9c>";

// --- Piece-stream builder ---
struct Builder {
    std::vector<ChatPiece> out;
    std::string text_buf;
    void push_text(const std::string& s) { text_buf += s; }
    void flush_text() {
        if (!text_buf.empty()) {
            ChatPiece p; p.kind = ChatPiece::Text; p.text = std::move(text_buf);
            out.push_back(std::move(p));
            text_buf.clear();
        }
    }
    void push_id(uint32_t id) {
        flush_text();
        ChatPiece p; p.kind = ChatPiece::Id; p.id = id;
        out.push_back(p);
    }
};

void render_llama3(const HostVocab& hv,
                   const std::vector<ChatTurn>& turns,
                   bool add_generation_prompt,
                   Builder& b) {
    uint32_t BOS  = require_id(hv, "<|begin_of_text|>");
    uint32_t SHS  = require_id(hv, "<|start_header_id|>");
    uint32_t EHS  = require_id(hv, "<|end_header_id|>");
    uint32_t EOT  = require_id(hv, "<|eot_id|>");

    b.push_id(BOS);
    for (const auto& t : turns) {
        b.push_id(SHS);
        b.push_text(t.role);
        b.push_id(EHS);
        b.push_text("\n\n");
        b.push_text(t.content);
        b.push_id(EOT);
    }
    if (add_generation_prompt) {
        b.push_id(SHS);
        b.push_text("assistant");
        b.push_id(EHS);
        b.push_text("\n\n");
    }
}

void render_qwen25(const HostVocab& hv,
                   const std::vector<ChatTurn>& turns,
                   bool add_generation_prompt,
                   Builder& b) {
    uint32_t IM_S = require_id(hv, "<|im_start|>");
    uint32_t IM_E = require_id(hv, "<|im_end|>");

    // Match `transformers.apply_chat_template` default system injection for
    // Qwen-2.5-Instruct exactly.
    bool has_system = false;
    for (const auto& t : turns) if (t.role == "system") { has_system = true; break; }

    auto emit_turn = [&](const std::string& role, const std::string& content) {
        b.push_id(IM_S);
        b.push_text(role);
        b.push_text("\n");
        b.push_text(content);
        b.push_id(IM_E);
        b.push_text("\n");
    };

    if (!has_system) {
        emit_turn("system",
                  "You are Qwen, created by Alibaba Cloud. "
                  "You are a helpful assistant.");
    }
    for (const auto& t : turns) {
        if (t.role != "system" && t.role != "user" && t.role != "assistant") {
            throw std::runtime_error("[special_tokens] Qwen-2.5: unknown role '" + t.role + "'");
        }
        emit_turn(t.role, t.content);
    }
    if (add_generation_prompt) {
        b.push_id(IM_S);
        b.push_text("assistant\n");
    }
}

void render_deepseek_v3(const HostVocab& hv,
                        const std::vector<ChatTurn>& turns,
                        bool add_generation_prompt,
                        Builder& b) {
    uint32_t BOS  = require_id(hv, DS_BOS);
    uint32_t USER = require_id(hv, DS_USER);
    uint32_t ASST = require_id(hv, DS_ASST);
    uint32_t EOS  = require_id(hv, DS_EOS);

    b.push_id(BOS);
    // Optional leading system message is rendered inline before the first
    // <｜User｜>, with no special separator (matches DeepSeek-V3's official
    // chat template).
    std::string sys;
    for (const auto& t : turns) if (t.role == "system") sys = t.content;
    if (!sys.empty()) b.push_text(sys);

    for (const auto& t : turns) {
        if (t.role == "system") continue;
        if (t.role == "user") {
            b.push_id(USER);
            b.push_text(t.content);
        } else if (t.role == "assistant") {
            b.push_id(ASST);
            b.push_text(t.content);
            b.push_id(EOS);
        } else {
            throw std::runtime_error("[special_tokens] DeepSeek-V3: unknown role '" + t.role + "'");
        }
    }
    if (add_generation_prompt) {
        b.push_id(ASST);
    }
}

}  // namespace

ChatTemplate detect_chat_template(const HostVocab& hv) {
    if (has_content(hv, "<|begin_of_text|>") && has_content(hv, "<|start_header_id|>"))
        return ChatTemplate::Llama3;
    if (has_content(hv, "<|im_start|>") && has_content(hv, "<|im_end|>"))
        return ChatTemplate::Qwen25;
    if (has_content(hv, DS_BOS) && has_content(hv, DS_USER))
        return ChatTemplate::DeepSeekV3;
    return ChatTemplate::None;
}

std::vector<ChatPiece> render_chat(const HostVocab& hv,
                                   ChatTemplate tmpl,
                                   const std::vector<ChatTurn>& turns,
                                   bool add_generation_prompt) {
    Builder b;
    switch (tmpl) {
        case ChatTemplate::Llama3:
            render_llama3(hv, turns, add_generation_prompt, b); break;
        case ChatTemplate::Qwen25:
            render_qwen25(hv, turns, add_generation_prompt, b); break;
        case ChatTemplate::DeepSeekV3:
            render_deepseek_v3(hv, turns, add_generation_prompt, b); break;
        case ChatTemplate::None:
        default:
            throw std::runtime_error(
                "[special_tokens] no chat template for this vocab "
                "(GPT-2 / unknown). Supported: Llama-3, Qwen-2.5, DeepSeek-V3.");
    }
    b.flush_text();
    return std::move(b.out);
}

std::string render_chat_text(const HostVocab& hv,
                             ChatTemplate tmpl,
                             const std::vector<ChatTurn>& turns,
                             bool add_generation_prompt) {
    auto pieces = render_chat(hv, tmpl, turns, add_generation_prompt);
    std::string out;
    for (const auto& p : pieces) {
        if (p.kind == ChatPiece::Text) {
            out += p.text;
        } else {
            // Look up id -> content (linear scan over added_tokens; small).
            for (const auto& at : hv.added_tokens) {
                if (at.id == p.id) { out += at.content; break; }
            }
        }
    }
    return out;
}

}  // namespace gbpe
