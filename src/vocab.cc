// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 the GraphTok contributors

#include "vocab.h"
#include "build_config.h"

#include <nlohmann/json.hpp>
#include <cuda_runtime.h>
#if GBPE_HAVE_VOCAB_QWEN25
#include <utf8proc.h>
#endif

#include <algorithm>
#include <string_view>
#include <unordered_map>
#if defined(__x86_64__)
#include <immintrin.h>
#endif
#include <array>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <functional>
#include <iostream>
#include <stdexcept>

namespace gbpe {

namespace {

// GPT-2 byte-to-unicode table. Identical to HF ByteLevel and OpenAI's gpt-2.
// Bytes in [0x21..0x7E], [0xA1..0xAC], [0xAE..0xFF] map to themselves;
// the remaining 256 bytes map sequentially into [0x100..0x1FF].
struct ByteLevelTables {
    std::array<uint32_t, 256> b2u;
    std::array<int32_t, 0x200> u2b;
    ByteLevelTables() {
        u2b.fill(-1);
        std::array<int, 256> taken{};
        // build the "natural" mapping list, then fill the rest at 256+n.
        std::vector<int> bs;
        for (int b = 0x21; b <= 0x7E; ++b) bs.push_back(b);
        for (int b = 0xA1; b <= 0xAC; ++b) bs.push_back(b);
        for (int b = 0xAE; b <= 0xFF; ++b) bs.push_back(b);
        std::vector<int> cs = bs;
        int n = 0;
        for (int b = 0; b < 256; ++b) {
            bool in = false;
            for (int x : bs) if (x == b) { in = true; break; }
            if (!in) {
                bs.push_back(b);
                cs.push_back(256 + n);
                ++n;
            }
        }
        for (size_t i = 0; i < bs.size(); ++i) {
            b2u[bs[i]] = static_cast<uint32_t>(cs[i]);
            u2b[cs[i]] = bs[i];
            taken[bs[i]] = 1;
        }
    }
};

const ByteLevelTables& tables() {
    static ByteLevelTables t;
    return t;
}

// Decode a tokenizer.json string (which uses the GPT-2 byte-encoded alphabet)
// back to the raw bytes the merge table reasons over.
std::vector<uint8_t> decode_token_str(const std::string& s) {
    const auto& u2b = tables().u2b;
    std::vector<uint8_t> out;
    out.reserve(s.size());
    size_t i = 0;
    while (i < s.size()) {
        unsigned char c0 = static_cast<unsigned char>(s[i]);
        uint32_t cp = 0;
        size_t adv = 0;
        // UTF-8 decode (tokenizer.json strings are UTF-8 JSON text).
        if (c0 < 0x80) { cp = c0; adv = 1; }
        else if ((c0 & 0xE0) == 0xC0 && i + 1 < s.size()) {
            cp = ((c0 & 0x1F) << 6) | (static_cast<unsigned char>(s[i+1]) & 0x3F);
            adv = 2;
        }
        else if ((c0 & 0xF0) == 0xE0 && i + 2 < s.size()) {
            cp = ((c0 & 0x0F) << 12) |
                 ((static_cast<unsigned char>(s[i+1]) & 0x3F) << 6) |
                 (static_cast<unsigned char>(s[i+2]) & 0x3F);
            adv = 3;
        }
        else if ((c0 & 0xF8) == 0xF0 && i + 3 < s.size()) {
            cp = ((c0 & 0x07) << 18) |
                 ((static_cast<unsigned char>(s[i+1]) & 0x3F) << 12) |
                 ((static_cast<unsigned char>(s[i+2]) & 0x3F) << 6) |
                 (static_cast<unsigned char>(s[i+3]) & 0x3F);
            adv = 4;
        } else {
            // malformed — skip one byte
            adv = 1;
        }
        if (cp < u2b.size() && u2b[cp] >= 0) {
            out.push_back(static_cast<uint8_t>(u2b[cp]));
        } else {
            // not a GPT-2-byte-encoded codepoint — should not happen in vocab tokens.
            // Push the raw codepoint low byte as a last resort.
            out.push_back(static_cast<uint8_t>(cp & 0xFF));
        }
        i += adv;
    }
    return out;
}

}  // namespace

const uint32_t* gpt2_byte_to_unicode() { return tables().b2u.data(); }
const int32_t*  gpt2_unicode_to_byte() { return tables().u2b.data(); }

// Exact tokenizer pipelines supported by the CUDA boundary kernels. These are
// copied byte-for-byte from the original tokenizer.json assets. Detection must
// be structural: accepting a merely similar regex and silently dispatching it
// to an existing kernel violates the project's bit-exactness contract.
constexpr const char* kLlama3Pattern =
    "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|"
    "\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|"
    "\\s+(?!\\S)|\\s+";
constexpr const char* kQwen25Pattern =
    "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|"
    "\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|"
    "\\s+(?!\\S)|\\s+";
constexpr const char* kDeepSeekDigitPattern = "\\p{N}{1,3}";
constexpr const char* kDeepSeekCjkPattern = "[一-龥぀-ゟ゠-ヿ]+";
constexpr const char* kDeepSeekMainPattern =
    "[!\"#$%&'()*+,\\-./:;<=>?@\\[\\\\\\]^_`{|}~][A-Za-z]+|"
    "[^\r\n\\p{L}\\p{P}\\p{S}]?[\\p{L}\\p{M}]+|"
    " ?[\\p{P}\\p{S}]+[\r\n]*|\\s*[\r\n]+|\\s+(?!\\S)|\\s+";

static bool string_field_is(const nlohmann::json& obj, const char* key,
                            const char* expected) {
    auto it = obj.find(key);
    return it != obj.end() && it->is_string() && it->get<std::string>() == expected;
}

static bool bool_field_is(const nlohmann::json& obj, const char* key,
                          bool expected, bool default_value) {
    auto it = obj.find(key);
    if (it == obj.end()) return default_value == expected;
    return it->is_boolean() && it->get<bool>() == expected;
}

static bool is_regex_split(const nlohmann::json& step, const char* pattern) {
    if (!step.is_object() || !string_field_is(step, "type", "Split") ||
        !string_field_is(step, "behavior", "Isolated") ||
        !bool_field_is(step, "invert", false, false)) {
        return false;
    }
    auto pit = step.find("pattern");
    return pit != step.end() && pit->is_object() &&
           string_field_is(*pit, "Regex", pattern);
}

static bool is_bytelevel(const nlohmann::json& step, bool trim_offsets,
                         bool use_regex, bool use_regex_default) {
    return step.is_object() && string_field_is(step, "type", "ByteLevel") &&
           bool_field_is(step, "add_prefix_space", false, false) &&
           bool_field_is(step, "trim_offsets", trim_offsets, true) &&
           bool_field_is(step, "use_regex", use_regex, use_regex_default);
}

static bool is_empty_sequence(const nlohmann::json& norm) {
    auto it = norm.find("normalizers");
    return norm.is_object() && string_field_is(norm, "type", "Sequence") &&
           it != norm.end() && it->is_array() && it->empty();
}

static bool model_options_supported(const nlohmann::json& model, RegexKind kind) {
    auto null_or_absent = [&](const char* key) {
        auto it = model.find(key);
        return it == model.end() || it->is_null();
    };
    auto empty_or_absent = [&](const char* key) {
        auto it = model.find(key);
        return it == model.end() || it->is_null() ||
               (it->is_string() && it->get<std::string>().empty());
    };
    if (!null_or_absent("dropout") ||
        !empty_or_absent("continuing_subword_prefix") ||
        !empty_or_absent("end_of_word_suffix")) {
        return false;
    }
    const bool gemma = kind == RegexKind::Gemma3;
    const bool llama = kind == RegexKind::Llama3;
    if (!bool_field_is(model, "byte_fallback", gemma, false) ||
        !bool_field_is(model, "fuse_unk", gemma, false) ||
        !bool_field_is(model, "ignore_merges", llama, false)) {
        return false;
    }
    if (gemma) {
        return string_field_is(model, "unk_token", "<unk>");
    }
    return null_or_absent("unk_token");
}

static bool kind_is_compiled(RegexKind kind) {
    switch (kind) {
        case RegexKind::GPT2:       return GBPE_HAVE_VOCAB_GPT2;
        case RegexKind::Llama3:     return GBPE_HAVE_VOCAB_LLAMA3;
        case RegexKind::Qwen25:     return GBPE_HAVE_VOCAB_QWEN25;
        case RegexKind::DeepSeekV3: return GBPE_HAVE_VOCAB_DEEPSEEK_V3;
        case RegexKind::Gemma3:     return GBPE_HAVE_VOCAB_GEMMA3;
    }
    return false;
}

static const char* regex_kind_name(RegexKind kind) {
    switch (kind) {
        case RegexKind::GPT2:       return "GPT-2";
        case RegexKind::Llama3:     return "Llama-3";
        case RegexKind::Qwen25:     return "Qwen2.5";
        case RegexKind::DeepSeekV3: return "DeepSeek-V3";
        case RegexKind::Gemma3:     return "Gemma 3 (SP)";
    }
    return "unknown";
}

static bool detect_regex_kind(const nlohmann::json& j, RegexKind& out) {
    const auto& pt = j.value("pre_tokenizer", nlohmann::json(nullptr));
    const auto& norm = j.value("normalizer", nlohmann::json(nullptr));
    const auto& model = j["model"];
    bool matched = false;

    // GPT-2: the ByteLevel component owns the canonical GPT-2 regex internally.
    if (norm.is_null() && is_bytelevel(pt, /*trim_offsets=*/true,
                                      /*use_regex=*/true,
                                      /*default=*/true)) {
        out = RegexKind::GPT2;
        matched = true;
    }

    if (!matched && pt.is_object() && string_field_is(pt, "type", "Sequence")) {
        auto sit = pt.find("pretokenizers");
        if (sit != pt.end() && sit->is_array()) {
            const auto& seq = *sit;
            if (norm.is_null() && seq.size() == 2 &&
                is_regex_split(seq[0], kLlama3Pattern) &&
                is_bytelevel(seq[1], true, false, true)) {
                out = RegexKind::Llama3;
                matched = true;
            } else if (norm.is_object() && string_field_is(norm, "type", "NFC") &&
                       seq.size() == 2 &&
                       is_regex_split(seq[0], kQwen25Pattern) &&
                       is_bytelevel(seq[1], false, false, true)) {
                out = RegexKind::Qwen25;
                matched = true;
            } else if (is_empty_sequence(norm) && seq.size() == 4 &&
                       is_regex_split(seq[0], kDeepSeekDigitPattern) &&
                       is_regex_split(seq[1], kDeepSeekCjkPattern) &&
                       is_regex_split(seq[2], kDeepSeekMainPattern) &&
                       is_bytelevel(seq[3], true, false, true)) {
                out = RegexKind::DeepSeekV3;
                matched = true;
            }
        }
    }

    // Gemma 3: Replace literal space with U+2581, followed by the exact literal
    // Split component serialized in the original asset.
    if (!matched && norm.is_object() && string_field_is(norm, "type", "Replace") &&
        string_field_is(norm, "content", "▁")) {
        auto npat = norm.find("pattern");
        auto ppat = pt.find("pattern");
        if (npat != norm.end() && npat->is_object() &&
            string_field_is(*npat, "String", " ") && pt.is_object() &&
            string_field_is(pt, "type", "Split") &&
            ppat != pt.end() && ppat->is_object() &&
            string_field_is(*ppat, "String", " ") &&
            string_field_is(pt, "behavior", "MergedWithPrevious") &&
            bool_field_is(pt, "invert", false, false)) {
            out = RegexKind::Gemma3;
            matched = true;
        }
    }

    if (!matched) {
        std::fprintf(stderr,
            "load_hf_tokenizer_json: unsupported pre_tokenizer/normalizer pipeline; "
            "only the exact GPT-2, Llama-3, Qwen-2.5, DeepSeek-V3, and Gemma-3 "
            "assets are supported\n");
        return false;
    }
    if (!model_options_supported(model, out)) {
        std::fprintf(stderr,
            "load_hf_tokenizer_json: tokenizer matches %s preprocessing but uses "
            "unsupported BPE model options\n", regex_kind_name(out));
        return false;
    }
    if (!kind_is_compiled(out)) {
        std::fprintf(stderr,
            "load_hf_tokenizer_json: tokenizer is %s, but this build does not "
            "include it; reconfigure GBPE_VOCABS and rebuild\n",
            regex_kind_name(out));
        return false;
    }
    return true;
}

bool load_hf_tokenizer_json(const std::string& path, HostVocab& out) {
    std::ifstream f(path);
    if (!f.is_open()) {
        std::fprintf(stderr, "load_hf_tokenizer_json: cannot open %s\n", path.c_str());
        return false;
    }
    nlohmann::json j;
    try { f >> j; }
    catch (const std::exception& e) {
        std::fprintf(stderr, "load_hf_tokenizer_json: JSON parse error: %s\n", e.what());
        return false;
    }
    if (!j.contains("model")) {
        std::fprintf(stderr, "load_hf_tokenizer_json: top-level 'model' missing\n");
        return false;
    }
    const auto& model = j["model"];
    // Some tokenizer.json files omit model.type when the format is unambiguous
    // (vocab + merges => BPE). Treat missing type as BPE; reject other types.
    std::string type = model.value("type", "BPE");
    if (type != "BPE") {
        std::fprintf(stderr, "load_hf_tokenizer_json: unsupported model.type='%s'\n", type.c_str());
        return false;
    }
    if (!model.contains("vocab") || !model.contains("merges")) {
        std::fprintf(stderr, "load_hf_tokenizer_json: model.vocab or model.merges missing\n");
        return false;
    }
    const auto& vocab = model["vocab"];
    const auto& merges = model["merges"];

    // Detect SP family up-front: model.byte_fallback == true means the vocab
    // strings are raw UTF-8, not GPT-2 byte-level-encoded. We must skip the
    // gpt2-style decode in that case (it produces junk for non-mapped
    // codepoints like '▁' = U+2581).
    const bool sp_family = model.value("byte_fallback", false);
    out.byte_fallback = sp_family;
    out.byte_fallback_id.clear();
    out.byte_fallback_byte_by_id.clear();
    if (sp_family) {
        out.unk_token = model.value("unk_token", std::string("<unk>"));
    }
    out.ignore_merges = model.contains("ignore_merges")
        && model["ignore_merges"].is_boolean()
        && model["ignore_merges"].get<bool>();

    out.normalizer_kind = NormalizerKind::None;
    if (j.contains("normalizer") && !j["normalizer"].is_null()) {
        // Byte-level vocabularies supported here either have no normalizer,
        // NFC (Qwen), or a no-op Sequence (DeepSeek). Parse the latter rather
        // than rejecting it just because tokenizer.json serializes an empty
        // normalizer list instead of null.
        std::function<bool(const nlohmann::json&, NormalizerKind&)> parse_normalizer;
        parse_normalizer = [&](const nlohmann::json& norm,
                               NormalizerKind& kind) -> bool {
            const std::string type = norm.value("type", std::string{});
            if (type == "NFC") {
                if (kind != NormalizerKind::None && kind != NormalizerKind::NFC) {
                    return false;
                }
                kind = NormalizerKind::NFC;
                return true;
            }
            if (type == "Sequence") {
                const auto& items = norm.value("normalizers", nlohmann::json::array());
                if (!items.is_array()) return false;
                for (const auto& item : items) {
                    if (!parse_normalizer(item, kind)) return false;
                }
                return true;
            }
            return false;
        };
        if (!sp_family && !parse_normalizer(j["normalizer"], out.normalizer_kind)) {
            const std::string norm_type = j["normalizer"].value("type", std::string{});
            std::fprintf(stderr,
                "load_hf_tokenizer_json: unsupported byte-level normalizer '%s'\n",
                norm_type.c_str());
            return false;
        }
    }

    // vocab: map token-string -> id
    out.tokens_by_id.clear();
    out.token_bytes.clear();
    out.id_by_token.clear();
    out.tokens_by_id.resize(vocab.size());
    out.token_bytes.resize(vocab.size());
    out.model_vocab_size = static_cast<uint32_t>(vocab.size());

    for (auto it = vocab.begin(); it != vocab.end(); ++it) {
        const std::string& token_str = it.key();
        uint32_t id = it.value().get<uint32_t>();
        if (id >= out.tokens_by_id.size()) {
            out.tokens_by_id.resize(id + 1);
            out.token_bytes.resize(id + 1);
        }
        out.tokens_by_id[id] = token_str;
        if (sp_family) {
            // Raw UTF-8: token's "bytes" are simply its UTF-8 string bytes.
            out.token_bytes[id].assign(token_str.begin(), token_str.end());
        } else {
            out.token_bytes[id] = decode_token_str(token_str);
        }
        out.id_by_token.emplace(token_str, id);
    }

    if (!sp_family) {
        // Initial 256 single-byte tokens. Look them up by their byte-encoded form.
        out.byte_to_id.assign(256, 0xFFFFFFFFu);
        const auto& b2u = tables().b2u;
        for (int b = 0; b < 256; ++b) {
            uint32_t cp = b2u[b];
            // encode codepoint as UTF-8 string and look up
            char buf[5] = {0};
            if (cp < 0x80) { buf[0] = static_cast<char>(cp); }
            else if (cp < 0x800) {
                buf[0] = static_cast<char>(0xC0 | (cp >> 6));
                buf[1] = static_cast<char>(0x80 | (cp & 0x3F));
            } else {
                buf[0] = static_cast<char>(0xE0 | (cp >> 12));
                buf[1] = static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
                buf[2] = static_cast<char>(0x80 | (cp & 0x3F));
            }
            auto it = out.id_by_token.find(buf);
            if (it == out.id_by_token.end()) {
                std::fprintf(stderr, "load_hf_tokenizer_json: missing single-byte token for byte 0x%02x (codepoint U+%04x '%s')\n", b, cp, buf);
                return false;
            }
            out.byte_to_id[b] = it->second;
        }
    } else {
        // SP family: build byte_fallback_id[256] from "<0xXX>" tokens.
        out.byte_fallback_id.assign(256, 0xFFFFFFFFu);
        out.byte_fallback_byte_by_id.assign(out.tokens_by_id.size(), -1);
        for (int b = 0; b < 256; ++b) {
            char buf[8];
            std::snprintf(buf, sizeof(buf), "<0x%02X>", b);
            auto it = out.id_by_token.find(buf);
            if (it == out.id_by_token.end()) {
                std::fprintf(stderr,
                    "load_hf_tokenizer_json: SP family but byte_fallback token '%s' missing\n",
                    buf);
                return false;
            }
            out.byte_fallback_id[b] = it->second;
            out.byte_fallback_byte_by_id[it->second] =
                static_cast<int16_t>(b);
        }
        // Track max token byte length for longest-prefix probing.
        for (const auto& kv : out.id_by_token) {
            if (kv.first.size() > out.max_token_byte_len) {
                out.max_token_byte_len = static_cast<uint32_t>(kv.first.size());
            }
        }
        // byte_to_id table is unused for SP; leave empty (the GPU kernel takes
        // a different code path).
        out.byte_to_id.assign(256, 0);
    }

    // Merges: list of "left right" strings (tokenizer.json v1 uses strings;
    // some newer versions use ["left","right"] arrays).
    out.merges.clear();
    out.merges.reserve(merges.size());
    for (const auto& m : merges) {
        std::string l, r;
        if (m.is_string()) {
            const std::string& s = m.get_ref<const std::string&>();
            auto sp = s.find(' ');
            if (sp == std::string::npos) {
                std::fprintf(stderr, "load_hf_tokenizer_json: malformed merge entry: '%s'\n", s.c_str());
                return false;
            }
            l = s.substr(0, sp);
            r = s.substr(sp + 1);
        } else if (m.is_array() && m.size() == 2) {
            l = m[0].get<std::string>();
            r = m[1].get<std::string>();
        } else {
            std::fprintf(stderr, "load_hf_tokenizer_json: unexpected merge entry type\n");
            return false;
        }
        auto li = out.id_by_token.find(l);
        auto ri = out.id_by_token.find(r);
        if (li == out.id_by_token.end() || ri == out.id_by_token.end()) {
            std::fprintf(stderr, "load_hf_tokenizer_json: merge references unknown token: '%s' or '%s'\n", l.c_str(), r.c_str());
            return false;
        }
        // new token = left + right (concat in the encoded alphabet)
        std::string combined = l + r;
        auto ci = out.id_by_token.find(combined);
        if (ci == out.id_by_token.end()) {
            std::fprintf(stderr, "load_hf_tokenizer_json: merge product '%s' missing from vocab\n", combined.c_str());
            return false;
        }
        out.merges.push_back({li->second, ri->second, ci->second});
    }

    // ---- F4: added_tokens + post_processor specials ----
    out.added_tokens.clear();
    out.special_id_by_content.clear();
    if (j.contains("added_tokens") && j["added_tokens"].is_array()) {
        for (const auto& at : j["added_tokens"]) {
            if (at.value("single_word", false) || at.value("lstrip", false) ||
                at.value("rstrip", false)) {
                std::fprintf(stderr,
                    "load_hf_tokenizer_json: added token uses unsupported "
                    "single_word/lstrip/rstrip matching\n");
                return false;
            }
            HostVocab::SpecialToken s;
            s.id      = at.value("id", 0u);
            s.content = at.value("content", std::string{});
            s.special = at.value("special", true);
            s.normalized = at.value("normalized", false);
            if (s.normalized && out.normalizer_kind != NormalizerKind::None) {
                std::fprintf(stderr,
                    "load_hf_tokenizer_json: normalized added tokens combined "
                    "with a non-empty normalizer are not supported\n");
                return false;
            }
            if (!s.content.empty()) {
                const auto existing_content = out.special_id_by_content.find(s.content);
                if (existing_content != out.special_id_by_content.end() &&
                    existing_content->second != s.id) {
                    std::fprintf(stderr,
                        "load_hf_tokenizer_json: added token content is assigned "
                        "to both ids %u and %u\n", existing_content->second, s.id);
                    return false;
                }
                out.added_tokens.push_back(s);
                out.special_id_by_content.emplace(s.content, s.id);
                if (s.id >= out.tokens_by_id.size()) {
                    out.tokens_by_id.resize(static_cast<size_t>(s.id) + 1);
                    out.token_bytes.resize(static_cast<size_t>(s.id) + 1);
                }
                if (out.tokens_by_id[s.id].empty()) {
                    out.tokens_by_id[s.id] = s.content;
                    out.token_bytes[s.id].assign(s.content.begin(), s.content.end());
                } else if (out.tokens_by_id[s.id] != s.content) {
                    std::fprintf(stderr,
                        "load_hf_tokenizer_json: added token id %u conflicts "
                        "with model-vocab content\n", s.id);
                    return false;
                }
            }
        }
    }
    out.bos_ids.clear();
    out.eos_ids.clear();
    auto resolve_template = [&](const nlohmann::json& tp) {
        if (!tp.contains("single") || !tp["single"].is_array()) return;
        auto resolve_id = [&](const std::string& content) -> int64_t {
            if (tp.contains("special_tokens") && tp["special_tokens"].is_object()) {
                auto it = tp["special_tokens"].find(content);
                if (it != tp["special_tokens"].end() && it->contains("ids")
                    && it.value()["ids"].is_array() && !it.value()["ids"].empty()) {
                    return it.value()["ids"][0].get<int64_t>();
                }
            }
            auto it = out.special_id_by_content.find(content);
            if (it != out.special_id_by_content.end()) return it->second;
            return -1;
        };
        bool seen_sequence = false;
        for (const auto& tok : tp["single"]) {
            if (tok.contains("SpecialToken")) {
                std::string id_str = tok["SpecialToken"].value("id", std::string{});
                int64_t tid = resolve_id(id_str);
                if (tid >= 0) {
                    if (!seen_sequence) out.bos_ids.push_back(static_cast<uint32_t>(tid));
                    else                out.eos_ids.push_back(static_cast<uint32_t>(tid));
                }
            } else if (tok.contains("Sequence")) {
                seen_sequence = true;
            }
        }
    };
    if (j.contains("post_processor") && !j["post_processor"].is_null()) {
        const auto& pp = j["post_processor"];
        std::string pt = pp.value("type", "");
        if (pt == "TemplateProcessing") {
            resolve_template(pp);
        } else if (pt == "Sequence" && pp.contains("processors")) {
            for (const auto& p : pp["processors"]) {
                if (p.value("type", "") == "TemplateProcessing") {
                    resolve_template(p);
                }
            }
        }
    }

    // ---- F1b: SP family: build host_merges lookup keyed on packed (left:32|right:32).
    if (sp_family) {
        out.host_merges.reserve(out.merges.size() * 2);
        for (uint32_t rank = 0; rank < out.merges.size(); ++rank) {
            const auto& mm = out.merges[rank];
            uint64_t k = (static_cast<uint64_t>(mm.left_id) << 32)
                       | static_cast<uint64_t>(mm.right_id);
            uint64_t v = (static_cast<uint64_t>(mm.new_id) << 32)
                       | static_cast<uint64_t>(rank);
            out.host_merges.emplace(k, v);
        }
    }

    if (!detect_regex_kind(j, out.regex_kind)) return false;
    GBPE_LOG(
        "[vocab] loaded: vocab_size=%zu merges=%zu regex=%s\n",
        out.tokens_by_id.size(), out.merges.size(), regex_kind_name(out.regex_kind));
    return true;
}

static inline uint32_t next_pow2(uint32_t x) {
    if (x < 2) return 2;
    --x;
    x |= x >> 1;
    x |= x >> 2;
    x |= x >> 4;
    x |= x >> 8;
    x |= x >> 16;
    return x + 1;
}

#if GBPE_HAVE_FAMILY_SP && GBPE_GEMMA_GPU_PRETOK
constexpr uint32_t kGemmaAddedRunMax = 31;

// Gemma carries compact AddedVocabulary entries for homogeneous newline, tab,
// and U+2581 runs. Keep these out of the literal trie: raw U+2581 runs are
// materialized by the dedicated device path, while newline/tab runs are
// reconstructed by the validated merge table.
bool gemma_added_run(const std::string& content,
                     uint32_t& kind, uint32_t& length) {
    if (!content.empty() && content.size() <= kGemmaAddedRunMax &&
        std::all_of(content.begin(), content.end(),
                    [](char c) { return c == '\n'; })) {
        kind = 0;
        length = static_cast<uint32_t>(content.size());
        return true;
    }
    if (!content.empty() && content.size() <= kGemmaAddedRunMax &&
        std::all_of(content.begin(), content.end(),
                    [](char c) { return c == '\t'; })) {
        kind = 1;
        length = static_cast<uint32_t>(content.size());
        return true;
    }

    constexpr std::array<uint8_t, 3> kLowbar = {0xE2u, 0x96u, 0x81u};
    if (!content.empty() && content.size() % kLowbar.size() == 0) {
        const size_t count = content.size() / kLowbar.size();
        if (count <= kGemmaAddedRunMax) {
            bool matches = true;
            for (size_t i = 0; i < content.size(); ++i) {
                if (static_cast<uint8_t>(content[i]) != kLowbar[i % 3]) {
                    matches = false;
                    break;
                }
            }
            if (matches) {
                kind = 2;
                length = static_cast<uint32_t>(count);
                return true;
            }
        }
    }
    return false;
}
#endif

#define CUDA_CHECK(call) do {                                       \
    cudaError_t err__ = (call);                                     \
    if (err__ != cudaSuccess) {                                     \
        std::fprintf(stderr, "CUDA error %s:%d: %s\n",              \
                     __FILE__, __LINE__, cudaGetErrorString(err__));\
        std::abort();                                               \
    }                                                               \
} while (0)

VocabPack build_vocab_pack(const HostVocab& hv) {
    VocabPack vp{};
    // bpe_kernel packs a merge rank and a slot index into one 32-bit key
    // (rank << 6 | slot) for its warp argmin.
    if (hv.merges.size() >= (size_t{1} << 26)) {
        throw std::runtime_error("cuTokenize: more than 2^26 merges is not supported");
    }
    vp.vocab_size = static_cast<uint32_t>(hv.tokens_by_id.size());
    vp.sp_family = hv.byte_fallback;
    vp.ignore_merges = hv.ignore_merges && !vp.sp_family;
    if (!vp.sp_family && hv.normalizer_kind != NormalizerKind::None) {
        // Qwen currently applies NFC on the host before the captured graph.
        // Raw AddedVocabulary extraction can commute with that normalization
        // only for ASCII literals: their leading/trailing ccc=0 scalars are
        // normalization barriers and every literal byte is NFC-stable.
        for (const auto& token : hv.added_tokens) {
            if (!std::all_of(token.content.begin(), token.content.end(),
                             [](unsigned char byte) { return byte < 0x80u; })) {
                std::fprintf(stderr,
                    "[vocab] FATAL: non-ASCII raw added token id %u cannot be "
                    "matched after a non-identity host normalizer; add a device "
                    "raw-match/normalize pipeline for this tokenizer\n", token.id);
                std::abort();
            }
        }
    }
    // GPU-pretok boundary-stage selector (mirrors RegexKind).
    switch (hv.regex_kind) {
        case RegexKind::GPT2:       vp.pretok_kind = PretokKind::GPT2;       break;
        case RegexKind::Llama3:     vp.pretok_kind = PretokKind::Llama3;     break;
        case RegexKind::Qwen25:     vp.pretok_kind = PretokKind::Qwen25;     break;
        case RegexKind::DeepSeekV3: vp.pretok_kind = PretokKind::DeepSeekV3; break;
        case RegexKind::Gemma3:     vp.pretok_kind = PretokKind::Gemma3;     break;
    }

    // Open-addressing table. Load factor target ~0.5.
    uint32_t cap = next_pow2(static_cast<uint32_t>(hv.merges.size()) * 2);
    if (cap < 4) cap = 4;
    vp.merge_table_capacity = cap;
    vp.merge_table_mask = cap - 1;

    [[maybe_unused]] const bool fits_64 =
        (vp.vocab_size <= 65535) && (hv.merges.size() <= 65535);

#if GBPE_USE_SLOT64_ONLY
    if (!fits_64) {
        std::fprintf(stderr,
            "[vocab] FATAL: vocab=%u merges=%zu exceeds 16-bit limits, but this build "
            "is compiled Slot64-only. Reconfigure with a larger -DGBPE_VOCABS=....\n",
            vp.vocab_size, (unsigned)hv.merges.size());
        std::abort();
    }
    const bool use_slot64 = true;
#elif GBPE_USE_SLOT128_ONLY
    const bool use_slot64 = false;  // forced Slot128 layout
#else
    const bool use_slot64 = fits_64;
#endif

    if (use_slot64) {
        // ---- Slot64 path (GPT-2) ----
        vp.kind = SlotKind::Slot64;
        constexpr uint64_t EMPTY_SLOT = 0x000000000000FFFFULL;  // rank field == 0xFFFF
        std::vector<uint64_t> h_slots(cap, EMPTY_SLOT);
        for (uint32_t rank = 0; rank < hv.merges.size(); ++rank) {
            const auto& m = hv.merges[rank];
            uint64_t slot = pack_slot64(m.left_id, m.right_id, m.new_id, rank);
            uint32_t k32 = pack_key32(m.left_id, m.right_id);
            uint32_t idx = merge_hash_key32(k32, vp.merge_table_mask);
            while ((h_slots[idx] & 0xFFFFu) != 0xFFFFu) {
                idx = (idx + 1) & vp.merge_table_mask;
            }
            h_slots[idx] = slot;
        }
        CUDA_CHECK(cudaMalloc(&vp.slots64, sizeof(uint64_t) * cap));
        CUDA_CHECK(cudaMemcpy(vp.slots64, h_slots.data(),
                              sizeof(uint64_t) * cap, cudaMemcpyHostToDevice));
        GBPE_LOG(
            "[vocab] device table (Slot64): cap=%u (%.2f MB, 8B/slot), load=%.2f\n",
            cap, cap * 8.0 / (1024.0 * 1024.0),
            (double)hv.merges.size() / (double)cap);
    } else {
        // ---- Slot128 path (Llama-3) ----
        vp.kind = SlotKind::Slot128;
        constexpr uint32_t EMPTY_RANK = 0xFFFFFFFFu;
        // empty slot: any key (we use 0), val with rank=EMPTY_RANK
        std::vector<uint64_t> h_keys(cap, 0);
        std::vector<uint64_t> h_vals(cap, static_cast<uint64_t>(EMPTY_RANK));
        for (uint32_t rank = 0; rank < hv.merges.size(); ++rank) {
            const auto& m = hv.merges[rank];
            uint64_t k64 = pack_key64(m.left_id, m.right_id);
            uint64_t v64 = pack_val64(m.new_id, rank);
            uint32_t idx = merge_hash_key64(k64, vp.merge_table_mask);
            while ((h_vals[idx] & 0xFFFFFFFFu) != EMPTY_RANK) {
                idx = (idx + 1) & vp.merge_table_mask;
            }
            h_keys[idx] = k64;
            h_vals[idx] = v64;
        }
        CUDA_CHECK(cudaMalloc(&vp.keys128, sizeof(uint64_t) * cap));
        CUDA_CHECK(cudaMalloc(&vp.vals128, sizeof(uint64_t) * cap));
        CUDA_CHECK(cudaMemcpy(vp.keys128, h_keys.data(),
                              sizeof(uint64_t) * cap, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(vp.vals128, h_vals.data(),
                              sizeof(uint64_t) * cap, cudaMemcpyHostToDevice));
        GBPE_LOG(
            "[vocab] device table (Slot128): cap=%u (%.2f MB, 16B/slot), load=%.2f\n",
            cap, cap * 16.0 / (1024.0 * 1024.0),
            (double)hv.merges.size() / (double)cap);
    }

    CUDA_CHECK(cudaMalloc(&vp.byte_to_id, sizeof(uint32_t) * 256));
    CUDA_CHECK(cudaMemcpy(vp.byte_to_id, hv.byte_to_id.data(),
                          sizeof(uint32_t) * 256, cudaMemcpyHostToDevice));

    // Build the tokenizer-independent AddedVocabulary trie. Device storage is
    // sparse: current assets have very low node degree, so a dense 256-way
    // table wastes orders of magnitude more memory (notably for Gemma).
    auto build_added_pack = [&](const std::vector<const HostVocab::SpecialToken*>& literals) {
        constexpr uint32_t kMiss = 0xFFFFFFFFu;
        if (literals.empty()) return;

        std::array<bool, 256> root_bytes{};
        for (const auto* token : literals) {
            root_bytes[static_cast<uint8_t>(token->content.front())] = true;
        }
        // The sparse owner-expansion kernel is race-free only when a literal
        // cannot contain the start of another literal. Prefixes at the same
        // byte are fine: the trie matcher keeps the longest terminal.
        for (const auto* token : literals) {
            for (size_t i = 1; i < token->content.size(); ++i) {
                if (root_bytes[static_cast<uint8_t>(token->content[i])]) {
                    std::fprintf(stderr,
                        "[vocab] FATAL: added-token literal id %u can overlap "
                        "a later literal start; general GPU arbitration is not "
                        "implemented for this asset\n", token->id);
                    std::abort();
                }
            }
        }

        struct BuildNode {
            std::unordered_map<uint8_t, uint32_t> next;
            uint32_t token_id = 0xFFFFFFFFu;
        };
        std::vector<BuildNode> nodes(1);
        uint32_t max_bytes = 0;
        for (const auto* token : literals) {
            uint32_t node = 0;
            for (uint8_t byte : token->content) {
                auto [it, inserted] = nodes[node].next.emplace(
                    byte, static_cast<uint32_t>(nodes.size()));
                if (inserted) nodes.emplace_back();
                node = it->second;
            }
            if (nodes[node].token_id != kMiss && nodes[node].token_id != token->id) {
                std::fprintf(stderr,
                    "[vocab] FATAL: duplicate added-token literal for ids %u and %u\n",
                    nodes[node].token_id, token->id);
                std::abort();
            }
            nodes[node].token_id = token->id;
            max_bytes = std::max<uint32_t>(
                max_bytes, static_cast<uint32_t>(token->content.size()));
        }

        std::vector<uint32_t> root(256, kMiss);
        for (const auto& edge : nodes.front().next) root[edge.first] = edge.second;
        std::vector<uint32_t> edge_begin(nodes.size());
        std::vector<uint16_t> edge_count(nodes.size());
        std::vector<uint32_t> terminal(nodes.size(), kMiss);
        std::vector<uint64_t> edges;
        for (size_t i = 0; i < nodes.size(); ++i) {
            std::vector<std::pair<uint8_t, uint32_t>> sorted(
                nodes[i].next.begin(), nodes[i].next.end());
            std::sort(sorted.begin(), sorted.end(),
                      [](const auto& a, const auto& b) { return a.first < b.first; });
            edge_begin[i] = static_cast<uint32_t>(edges.size());
            edge_count[i] = static_cast<uint16_t>(sorted.size());
            terminal[i] = nodes[i].token_id;
            for (const auto& edge : sorted) {
                edges.push_back((static_cast<uint64_t>(edge.second) << 8) |
                                static_cast<uint64_t>(edge.first));
            }
        }

        vp.added_trie_nodes = static_cast<uint32_t>(nodes.size());
        vp.added_trie_edges = static_cast<uint32_t>(edges.size());
        vp.added_max_bytes = max_bytes;
        CUDA_CHECK(cudaMalloc(&vp.added_root, root.size() * sizeof(uint32_t)));
        CUDA_CHECK(cudaMemcpy(vp.added_root, root.data(),
                              root.size() * sizeof(uint32_t), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMalloc(&vp.added_node_edge_begin,
                              edge_begin.size() * sizeof(uint32_t)));
        CUDA_CHECK(cudaMemcpy(vp.added_node_edge_begin, edge_begin.data(),
                              edge_begin.size() * sizeof(uint32_t), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMalloc(&vp.added_node_edge_count,
                              edge_count.size() * sizeof(uint16_t)));
        CUDA_CHECK(cudaMemcpy(vp.added_node_edge_count, edge_count.data(),
                              edge_count.size() * sizeof(uint16_t), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMalloc(&vp.added_node_token_id,
                              terminal.size() * sizeof(uint32_t)));
        CUDA_CHECK(cudaMemcpy(vp.added_node_token_id, terminal.data(),
                              terminal.size() * sizeof(uint32_t), cudaMemcpyHostToDevice));
        if (!edges.empty()) {
            CUDA_CHECK(cudaMalloc(&vp.added_edges, edges.size() * sizeof(uint64_t)));
            CUDA_CHECK(cudaMemcpy(vp.added_edges, edges.data(),
                                  edges.size() * sizeof(uint64_t), cudaMemcpyHostToDevice));
        }
        GBPE_LOG(
            "[vocab] GPU added-token matcher: literals=%zu nodes=%u edges=%u "
            "max_bytes=%u storage=%.2f MB\n",
            literals.size(), vp.added_trie_nodes, vp.added_trie_edges,
            vp.added_max_bytes,
            (root.size() * sizeof(uint32_t) +
             edge_begin.size() * sizeof(uint32_t) +
             edge_count.size() * sizeof(uint16_t) +
             terminal.size() * sizeof(uint32_t) +
             edges.size() * sizeof(uint64_t)) / (1024.0 * 1024.0));
    };

    if (!vp.sp_family) {
        std::vector<const HostVocab::SpecialToken*> literals;
        literals.reserve(hv.added_tokens.size());
        for (const auto& token : hv.added_tokens) literals.push_back(&token);
        build_added_pack(literals);
    }

#if GBPE_HAVE_FAMILY_SP && GBPE_GEMMA_GPU_PRETOK
    if (vp.sp_family) {
        constexpr uint32_t kUnicodeScalars = 0x110000u;
        constexpr uint32_t kMiss = 0xFFFFFFFFu;
        std::vector<uint32_t> h_cp_to_id(kUnicodeScalars, kMiss);

        auto decode_one_scalar = [](const std::string& s, uint32_t& cp) -> bool {
            if (s.empty()) return false;
            const auto* p = reinterpret_cast<const unsigned char*>(s.data());
            const size_t n = s.size();
            uint32_t v = 0;
            size_t w = 0;
            if (p[0] < 0x80) { v = p[0]; w = 1; }
            else if ((p[0] & 0xE0) == 0xC0) { v = p[0] & 0x1F; w = 2; }
            else if ((p[0] & 0xF0) == 0xE0) { v = p[0] & 0x0F; w = 3; }
            else if ((p[0] & 0xF8) == 0xF0) { v = p[0] & 0x07; w = 4; }
            else return false;
            if (w != n) return false;
            for (size_t i = 1; i < w; ++i) {
                if ((p[i] & 0xC0) != 0x80) return false;
                v = (v << 6) | (p[i] & 0x3F);
            }
            if ((w == 2 && v < 0x80) || (w == 3 && v < 0x800) ||
                (w == 4 && v < 0x10000) || v > 0x10FFFFu ||
                (v >= 0xD800u && v <= 0xDFFFu)) return false;
            cp = v;
            return true;
        };

        for (uint32_t id = 0; id < hv.model_vocab_size; ++id) {
            uint32_t cp = 0;
            if (decode_one_scalar(hv.tokens_by_id[id], cp)) h_cp_to_id[cp] = id;
        }
        if (h_cp_to_id[0x2581u] == kMiss) {
            std::fprintf(stderr,
                "[vocab] FATAL: SP vocab has no initial token for U+2581\n");
            std::abort();
        }
        CUDA_CHECK(cudaMalloc(&vp.sp_codepoint_to_id,
                              kUnicodeScalars * sizeof(uint32_t)));
        CUDA_CHECK(cudaMemcpy(vp.sp_codepoint_to_id, h_cp_to_id.data(),
                              kUnicodeScalars * sizeof(uint32_t),
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMalloc(&vp.sp_byte_fallback, 256 * sizeof(uint32_t)));
        CUDA_CHECK(cudaMemcpy(vp.sp_byte_fallback, hv.byte_fallback_id.data(),
                              256 * sizeof(uint32_t), cudaMemcpyHostToDevice));

        std::vector<const HostVocab::SpecialToken*> direct_literals;
        direct_literals.reserve(hv.added_tokens.size());
        std::array<std::array<uint32_t, 32>, 3> run_ids{};
        for (auto& ids : run_ids) ids.fill(kMiss);
        uint32_t lowbar_min_run = kMiss;
        for (const auto& token : hv.added_tokens) {
            if (token.normalized) {
                std::fprintf(stderr,
                    "[vocab] FATAL: Gemma GPU AddedVocabulary does not support "
                    "normalized=true for token id %u\n", token.id);
                std::abort();
            }
            uint32_t run_kind = 0, run_length = 0;
            if (gemma_added_run(token.content, run_kind, run_length)) {
                if (run_ids[run_kind][run_length] != kMiss) {
                    std::fprintf(stderr,
                        "[vocab] FATAL: duplicate Gemma added-run token "
                        "(kind=%u length=%u)\n", run_kind, run_length);
                    std::abort();
                }
                run_ids[run_kind][run_length] = token.id;
                if (run_kind == 2) lowbar_min_run = std::min(
                    lowbar_min_run, run_length);
            } else {
                direct_literals.push_back(&token);
            }
        }
        if (lowbar_min_run != 2) {
            std::fprintf(stderr,
                "[vocab] FATAL: Gemma GPU fast path expects added U+2581 "
                "runs to begin at length 2 (found %u)\n", lowbar_min_run);
            std::abort();
        }
        for (uint32_t kind = 0; kind < 3; ++kind) {
            const uint32_t first = kind == 2 ? 2u : 1u;
            for (uint32_t length = first; length <= 31; ++length) {
                if (run_ids[kind][length] == kMiss) {
                    std::fprintf(stderr,
                        "[vocab] FATAL: Gemma GPU fast path requires complete "
                        "added-run lengths (kind=%u missing=%u)\n",
                        kind, length);
                    std::abort();
                }
            }
        }
        CUDA_CHECK(cudaMalloc(&vp.sp_lowbar_run_ids,
                              run_ids[2].size() * sizeof(uint32_t)));
        CUDA_CHECK(cudaMemcpy(vp.sp_lowbar_run_ids, run_ids[2].data(),
                              run_ids[2].size() * sizeof(uint32_t),
                              cudaMemcpyHostToDevice));

        // Newline and tab run tokens are safe to leave inside the surrounding
        // Gemma BPE segment only because this exact asset has no merge crossing
        // from either run alphabet into ordinary text. Enforce that invariant
        // rather than silently reintroducing AddedVocabulary boundary skew for
        // a future asset. Raw U+2581 runs are isolated on-device because U+2581
        // intentionally has many cross-word merges.
        auto is_homogeneous = [](const std::string& token, char byte) {
            return !token.empty() && std::all_of(
                token.begin(), token.end(),
                [byte](char value) { return value == byte; });
        };
        for (const auto& merge : hv.merges) {
            const auto& left = hv.tokens_by_id[merge.left_id];
            const auto& right = hv.tokens_by_id[merge.right_id];
            for (char byte : {'\n', '\t'}) {
                if (is_homogeneous(left, byte) != is_homogeneous(right, byte)) {
                    std::fprintf(stderr,
                        "[vocab] FATAL: Gemma merge crosses an added %s run; "
                        "the GPU fast path requires explicit isolation\n",
                        byte == '\n' ? "newline" : "tab");
                    std::abort();
                }
            }
        }

        // The device pre-tokenizer restores parallelism by cutting before a
        // maximal raw-space run (which the normalizer turns into U+2581).  A
        // first merge crossing such a cut must have a non-lowbar token on the
        // left and a token beginning with U+2581 on the right.  The supported
        // asset has exactly one shape of that kind, `>` + `▁</`; the kernel
        // recognizes its raw `> </` spelling and deliberately does not cut
        // there.  Reject any other crossing merge so a future tokenizer asset
        // cannot silently make this performance decomposition non-bit-exact.
        const std::string lowbar = "\xE2\x96\x81";
        auto is_lowbar_run = [&lowbar](const std::string& token) {
            if (token.empty() || token.size() % lowbar.size() != 0) return false;
            for (size_t i = 0; i < token.size(); ++i) {
                if (token[i] != lowbar[i % lowbar.size()]) return false;
            }
            return true;
        };
        auto starts_with_lowbar = [&lowbar](const std::string& token) {
            return token.size() >= lowbar.size() &&
                   token.compare(0, lowbar.size(), lowbar) == 0;
        };
        for (size_t rank = 0; rank < hv.merges.size(); ++rank) {
            const auto& merge = hv.merges[rank];
            const auto& left = hv.tokens_by_id[merge.left_id];
            const auto& right = hv.tokens_by_id[merge.right_id];
            if (!is_lowbar_run(left) && starts_with_lowbar(right) &&
                !(left == ">" && right == lowbar + "</")) {
                std::fprintf(stderr,
                    "[vocab] FATAL: Gemma merge crosses a normalized-space "
                    "component boundary (rank=%u left_id=%u right_id=%u)\n",
                    static_cast<uint32_t>(rank), merge.left_id, merge.right_id);
                std::abort();
            }
        }

        // Structured run entries are handled by the Gemma run path; all other
        // literals use the same compact device trie as byte-level tokenizers.
        for (const auto* token : direct_literals) {
            if (token->content.find('\n') != std::string::npos ||
                token->content.find('\t') != std::string::npos ||
                token->content.find("\xE2\x96\x81") != std::string::npos) {
                std::fprintf(stderr,
                    "[vocab] FATAL: Gemma added-token literal id %u overlaps "
                    "an added-token run\n", token->id);
                std::abort();
            }
        }
        build_added_pack(direct_literals);
    }
#endif

    // ---- Decode-side packed arrays. Concat all token_bytes, build offsets/lens.
    {
        const uint32_t V = vp.vocab_size;
        std::vector<uint32_t> h_off(V, 0);
        std::vector<uint16_t> h_len(V, 0);
        size_t total = 0;
        for (uint32_t i = 0; i < V; ++i) {
            size_t L = (i < hv.token_bytes.size()) ? hv.token_bytes[i].size() : 0;
            if (L > 0xFFFFu) {
                std::fprintf(stderr,
                    "[vocab] FATAL: token %u byte length %zu exceeds uint16_t\n",
                    i, L);
                std::abort();
            }
            h_off[i] = static_cast<uint32_t>(total);
            h_len[i] = static_cast<uint16_t>(L);
            total += L;
        }
        if (total > 0xFFFFFFFFu) {
            std::fprintf(stderr, "[vocab] FATAL: total token bytes %zu exceeds uint32_t\n", total);
            std::abort();
        }
        std::vector<uint8_t> h_concat(total);
        for (uint32_t i = 0; i < V; ++i) {
            if (i < hv.token_bytes.size()) {
                const auto& b = hv.token_bytes[i];
                std::memcpy(h_concat.data() + h_off[i], b.data(), b.size());
            }
        }
        vp.vocab_total_bytes = static_cast<uint32_t>(total);
        CUDA_CHECK(cudaMalloc(&vp.vocab_token_bytes_concat,
                              total > 0 ? total : 1));
        if (total > 0) {
            CUDA_CHECK(cudaMemcpy(vp.vocab_token_bytes_concat, h_concat.data(),
                                  total, cudaMemcpyHostToDevice));
        }
        CUDA_CHECK(cudaMalloc(&vp.vocab_token_offset, sizeof(uint32_t) * V));
        CUDA_CHECK(cudaMemcpy(vp.vocab_token_offset, h_off.data(),
                              sizeof(uint32_t) * V, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMalloc(&vp.vocab_token_len, sizeof(uint16_t) * V));
        CUDA_CHECK(cudaMemcpy(vp.vocab_token_len, h_len.data(),
                              sizeof(uint16_t) * V, cudaMemcpyHostToDevice));
        GBPE_LOG(
            "[vocab] decode arrays: vocab=%u total_bytes=%u (%.2f MB)\n",
            V, vp.vocab_total_bytes, total / (1024.0 * 1024.0));
    }

    // Llama-3's BPE model enables ignore_merges. It is not merely a speed
    // hint: a full raw pre-token can be a direct model-vocab item even when
    // normal greedy BPE would leave multiple parts (for example " Việt").
    if (vp.ignore_merges) {
        const uint32_t direct_cap = next_pow2(
            std::max<uint32_t>(4u, vp.vocab_size * 2u));
        const uint32_t direct_mask = direct_cap - 1;
        std::vector<uint64_t> h_hashes(direct_cap, 0);
        std::vector<uint32_t> h_ids(direct_cap, TOKEN_DEAD);
        for (uint32_t id = 0; id < hv.model_vocab_size; ++id) {
            const auto& bytes = hv.token_bytes[id];
            if (bytes.empty()) continue;
            const uint64_t hash = raw_token_hash64(
                bytes.data(), static_cast<uint32_t>(bytes.size()));
            uint32_t slot = raw_token_hash_slot(hash, direct_mask);
            while (h_ids[slot] != TOKEN_DEAD) {
                slot = (slot + 1u) & direct_mask;
            }
            h_hashes[slot] = hash;
            h_ids[slot] = id;
        }
        CUDA_CHECK(cudaMalloc(&vp.direct_token_hashes,
                              static_cast<size_t>(direct_cap) * sizeof(uint64_t)));
        CUDA_CHECK(cudaMalloc(&vp.direct_token_ids,
                              static_cast<size_t>(direct_cap) * sizeof(uint32_t)));
        CUDA_CHECK(cudaMemcpy(vp.direct_token_hashes, h_hashes.data(),
                              static_cast<size_t>(direct_cap) * sizeof(uint64_t),
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(vp.direct_token_ids, h_ids.data(),
                              static_cast<size_t>(direct_cap) * sizeof(uint32_t),
                              cudaMemcpyHostToDevice));
        vp.direct_token_mask = direct_mask;
        GBPE_LOG(
            "[vocab] ignore_merges direct-token table: cap=%u (%.2f MB)\n",
            direct_cap,
            direct_cap * (sizeof(uint64_t) + sizeof(uint32_t)) /
                (1024.0 * 1024.0));
    }

    return vp;
}

void free_vocab_pack(VocabPack& vp) {
    if (vp.slots64)    cudaFree(vp.slots64);
    if (vp.keys128)    cudaFree(vp.keys128);
    if (vp.vals128)    cudaFree(vp.vals128);
    if (vp.byte_to_id) cudaFree(vp.byte_to_id);
#if GBPE_HAVE_FAMILY_SP && GBPE_GEMMA_GPU_PRETOK
    if (vp.sp_codepoint_to_id) cudaFree(vp.sp_codepoint_to_id);
    if (vp.sp_byte_fallback)   cudaFree(vp.sp_byte_fallback);
    if (vp.sp_lowbar_run_ids)  cudaFree(vp.sp_lowbar_run_ids);
#endif
    if (vp.added_root)            cudaFree(vp.added_root);
    if (vp.added_node_edge_begin) cudaFree(vp.added_node_edge_begin);
    if (vp.added_node_edge_count) cudaFree(vp.added_node_edge_count);
    if (vp.added_node_token_id)   cudaFree(vp.added_node_token_id);
    if (vp.added_edges)           cudaFree(vp.added_edges);
    if (vp.vocab_token_bytes_concat) cudaFree(vp.vocab_token_bytes_concat);
    if (vp.vocab_token_offset)       cudaFree(vp.vocab_token_offset);
    if (vp.vocab_token_len)          cudaFree(vp.vocab_token_len);
    if (vp.direct_token_hashes)       cudaFree(vp.direct_token_hashes);
    if (vp.direct_token_ids)          cudaFree(vp.direct_token_ids);
    vp = {};
}

// ---- F4: prepend/append specials ----
namespace {
// Probe a list of candidate content strings; first one present in the vocab
// wins. Returns UINT32_MAX if none match.
uint32_t lookup_special(const HostVocab& hv,
                        std::initializer_list<const char*> candidates) {
    for (const char* c : candidates) {
        auto it = hv.special_id_by_content.find(c);
        if (it != hv.special_id_by_content.end()) return it->second;
    }
    return 0xFFFFFFFFu;
}
}  // namespace

void apply_encode_options(const HostVocab& hv,
                          const EncodeOptions& opts,
                          std::vector<uint32_t>& ids) {
    if (opts.add_bos && !hv.bos_ids.empty()) {
        ids.insert(ids.begin(), hv.bos_ids.begin(), hv.bos_ids.end());
    }
    if (opts.add_eos) {
        if (!hv.eos_ids.empty()) {
            ids.insert(ids.end(), hv.eos_ids.begin(), hv.eos_ids.end());
        } else {
            // Heuristic canonical EOS by content name across our four vocabs.
            uint32_t eos = lookup_special(hv, {
                "<|end_of_text|>",          // Llama-3
                "<|endoftext|>",            // GPT-2, Qwen-2.5
                "<\xef\xbd\x9c" "end" "\xe2\x96\x81" "of" "\xe2\x96\x81" "sentence" "\xef\xbd\x9c>", // DeepSeek-V3 (UTF-8)
            });
            if (eos != 0xFFFFFFFFu) ids.push_back(eos);
        }
    }
}

#if GBPE_HAVE_VOCAB_QWEN25
namespace {

// tokenizers 0.23.1's normalization data predates the Unicode 15.1/16/17
// additions present in recent utf8proc releases. Treat scalars involved
// in that exhaustively audited delta as normalization barriers: HF sees
// them as unassigned ccc=0 scalars and neither reorders nor composes across
// them. Normalization is stable for every previously assigned scalar.
bool is_hf_nfc_barrier(utf8proc_int32_t cp) {
    switch (cp) {
        case 0x0897:
        case 0x105D2: case 0x105DA:
        case 0x11382: case 0x11384: case 0x1138B: case 0x11390:
        case 0x113B8: case 0x113BB: case 0x113C2: case 0x113C9:
        case 0x11930: case 0x11935:
        case 0x1611E: case 0x1611F: case 0x16120: case 0x16129:
        case 0x16D63: case 0x16D67:
            return true;
        default:
            return (cp >= 0x1ACF && cp <= 0x1ADD) ||
                   (cp >= 0x1AE0 && cp <= 0x1AEB) ||
                   (cp >= 0x10D69 && cp <= 0x10D6D) ||
                   (cp >= 0x10EFA && cp <= 0x10EFB) ||
                   (cp >= 0x113CE && cp <= 0x113D0) ||
                   (cp >= 0x1E5EE && cp <= 0x1E5EF) ||
                   cp == 0x1E6E3 || cp == 0x1E6E6 ||
                   (cp >= 0x1E6EE && cp <= 0x1E6EF) || cp == 0x1E6F5;
    }
}

const utf8proc_option_t kNfcOpts =
    static_cast<utf8proc_option_t>(UTF8PROC_STABLE | UTF8PROC_COMPOSE);

// Memoized "is this scalar NFC-stable" (see the quick-check comment in
// is_normalized_for_vocab). One instance per call; memoization only.
struct StableCache {
    utf8proc_int32_t cp[64];
    bool stable[64];
    int n = 0;

    bool is_stable(utf8proc_int32_t cp, const utf8proc_property_t* p) {
        const auto opts = kNfcOpts;
        StableCache& cache = *this;
        // A scalar that can be the SECOND element of a canonical composition
        // (utf8proc: comb_index bit 15 set; e.g. Indic vowel signs such as
        // U+09BE, U+0B56, Myanmar U+102E) is unstable even at ccc=0, since
        // the preceding starter may compose with it. First elements
        // (comb_index < 0x8000) and non-composers (0xFFFF) are fine.
        if (p->comb_index != 0xFFFFu && (p->comb_index & 0x8000u)) return false;
        if (p->decomp_type != 0 || p->decomp_seqindex == 0xFFFFu) return true;
        for (int i = 0; i < cache.n; ++i) {
            if (cache.cp[i] == cp) return cache.stable[i];
        }
        utf8proc_int32_t tmp[8];
        const auto n = utf8proc_decompose_char(cp, tmp, 8, opts, nullptr);
        bool stable = false;
        if (n > 0 && n < 8) {
            const auto m = utf8proc_normalize_utf32(tmp, n, opts);
            stable = (m == 1 && tmp[0] == cp);
        }
        if (cache.n < 64) {
            cache.cp[cache.n] = cp;
            cache.stable[cache.n] = stable;
            ++cache.n;
        }
        return stable;
    }
};

// Pass-1 verdict for every BMP scalar, precomputed once: bit cp is set iff
// the scalar loop below would accept cp and continue (not a barrier-free
// scalar with ccc != 0, not Hangul, NFC-stable). Built by running that exact
// predicate over all 65,536 BMP scalars, so a lookup is equal to the loop by
// construction. Non-BMP scalars (4-byte UTF-8) still take the scalar path.
// Measured on 1M tokens of English books (2.9% non-ASCII bytes, mostly curly
// quotes): the per-scalar path cost ~1.8 ms of a ~5 ms encode.
bool scalar_passes(utf8proc_int32_t cp, StableCache& cache) {
    if (is_hf_nfc_barrier(cp)) return true;
    const utf8proc_property_t* p = utf8proc_get_property(cp);
    const bool hangul = (cp >= 0x1100 && cp <= 0x11FF) || (cp >= 0xAC00 && cp <= 0xD7A3);
    return p->combining_class == 0 && !hangul && cache.is_stable(cp, p);
}

const uint64_t* bmp_pass_bits() {
    static const std::vector<uint64_t> bits = [] {
        std::vector<uint64_t> b(65536 / 64, 0);
        for (utf8proc_int32_t cp = 0; cp < 0x10000; ++cp) {
            if (cp >= 0xD800 && cp <= 0xDFFF) continue;   // not scalars
            StableCache cache;   // memo only; the verdict does not depend on it
            if (scalar_passes(cp, cache)) b[cp >> 6] |= 1ull << (cp & 63);
        }
        return b;
    }();
    return bits.data();
}

// bmp_pass_bits() minus the barrier scalars: pass 2 copies a scalar in bulk
// only when it is stable AND not a barrier (barriers close spans explicitly).
const uint64_t* bmp_bulk_bits() {
    static const std::vector<uint64_t> bits = [] {
        const uint64_t* pass = bmp_pass_bits();
        std::vector<uint64_t> b(pass, pass + 65536 / 64);
        for (utf8proc_int32_t cp = 0; cp < 0x10000; ++cp)
            if (is_hf_nfc_barrier(cp)) b[cp >> 6] &= ~(1ull << (cp & 63));
        return b;
    }();
    return bits.data();
}

}  // namespace
#endif

#if GBPE_HAVE_VOCAB_QWEN25
namespace {

// Offset of the first byte >= 0x80 at or after `off`, or `n`. AVX2 when the
// CPU has it (32 bytes per step), else eight bytes per step.
#if defined(__x86_64__)
__attribute__((target("avx2")))
size_t skip_ascii_avx2(const uint8_t* b, size_t off, size_t n) {
    while (off + 32 <= n) {
        const __m256i v = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b + off));
        const unsigned m = static_cast<unsigned>(_mm256_movemask_epi8(v));
        if (m) return off + static_cast<size_t>(__builtin_ctz(m));
        off += 32;
    }
    while (off < n && b[off] < 0x80u) ++off;
    return off;
}
#endif

size_t skip_ascii_swar(const uint8_t* b, size_t off, size_t n) {
    while (off + 8 <= n) {
        uint64_t w;
        std::memcpy(&w, b + off, sizeof w);
        const uint64_t hi = w & 0x8080808080808080ull;
        if (hi) return off + static_cast<size_t>(__builtin_ctzll(hi) >> 3);
        off += 8;
    }
    while (off < n && b[off] < 0x80u) ++off;
    return off;
}

using SkipAsciiFn = size_t (*)(const uint8_t*, size_t, size_t);

// Chosen once per scan, so the per-scalar loops pay no dispatch check.
SkipAsciiFn skip_ascii_fn() {
#if defined(__x86_64__)
    static const SkipAsciiFn fn =
        __builtin_cpu_supports("avx2") ? &skip_ascii_avx2 : &skip_ascii_swar;
    return fn;
#else
    return &skip_ascii_swar;
#endif
}

// Width of the well-formed 2- or 3-byte (BMP, no overlong, no surrogate)
// sequence at b[off] -- exactly the sequences utf8proc_iterate accepts there
// -- with its scalar in *cp; 0 for anything else (4-byte, malformed).
inline int bmp_seq(const uint8_t* b, size_t off, size_t n, uint32_t* cp) {
    const uint8_t c0 = b[off];
    if (c0 >= 0xC2u && c0 <= 0xDFu && off + 1 < n && (b[off + 1] & 0xC0u) == 0x80u) {
        *cp = ((c0 & 0x1Fu) << 6) | (b[off + 1] & 0x3Fu);
        return 2;
    }
    if (c0 >= 0xE0u && c0 <= 0xEFu && off + 2 < n) {
        const uint8_t b1 = b[off + 1], b2 = b[off + 2];
        const bool b1_ok = c0 == 0xE0u ? (b1 >= 0xA0u && b1 <= 0xBFu)
                         : c0 == 0xEDu ? (b1 >= 0x80u && b1 <= 0x9Fu)
                                       : (b1 & 0xC0u) == 0x80u;
        if (b1_ok && (b2 & 0xC0u) == 0x80u) {
            *cp = ((c0 & 0x0Fu) << 12) | ((b1 & 0x3Fu) << 6) | (b2 & 0x3Fu);
            return 3;
        }
    }
    return 0;
}

// Whether the Hangul syllable U+AC00..D7A3 `c` whose bytes end at b[end]
// is NFC-stable where it stands. A syllable is ccc=0, NFC(syllable) is the
// syllable, and it is never the second element of a composition, so the only
// way NFC can change it is an LV syllable composing with an IMMEDIATELY
// following trailing jamo U+11A8..11C2 (for a ccc=0 second element any
// intervening scalar blocks composition). Anywhere else it is a safe
// boundary exactly like any other stable starter.
inline bool hangul_syllable_stable(uint32_t c, const uint8_t* b, size_t end, size_t n) {
    if ((c - 0xAC00u) % 28u != 0) return true;              // LVT: takes no T
    if (end + 2 >= n || b[end] != 0xE1u) return true;
    const uint32_t nx = ((b[end + 1] & 0x3Fu) << 6) | (b[end + 2] & 0x3Fu);   // U+1000 + nx
    return !((b[end + 1] & 0xC0u) == 0x80u && (b[end + 2] & 0xC0u) == 0x80u &&
             nx >= 0x1A8u && nx <= 0x1C2u);
}

// Pass 1: NFC quick check.// Pass 1: NFC quick check. A string is already NFC when every scalar is
// ccc=0 (nothing can reorder around it) and is "NFC-stable": either it has no
// canonical decomposition, or it is a precomposed character that NFC
// recomposes back to itself (decided once per distinct scalar, memoized).
// Hangul jamo (1100..11FF) and syllables (AC00..D7A3) can compose across
// scalar boundaries, so they always fail the check. Barrier scalars are ccc=0
// and unassigned to HF, so they pass. BMP scalars are answered from
// bmp_pass_bits(), built from this same predicate; ASCII is skipped in bulk.
//
// Returns n when the text is already NFC. Otherwise returns the byte offset
// of the scalar just before the first failing one (0 if it is the first):
// every scalar before that offset passes and none of them is inside a span
// pass 2 would normalize, so pass 2 may copy [0, offset) verbatim and resume
// there with exactly the state a scan from 0 would have. Malformed UTF-8
// fails the check, and pass 2 then reports it.
size_t nfc_scan(const uint8_t* bytes, size_t n) {
    const uint64_t* bmp_bits = bmp_pass_bits();
    const SkipAsciiFn skip_ascii = skip_ascii_fn();
    StableCache cache;
    size_t off = 0, prev = 0;   // prev: start of the previous (passing) scalar
    for (;;) {
        if (off < n && bytes[off] < 0x80u) {
            const size_t a = skip_ascii(bytes, off, n);
            prev = a - 1;
            off = a;
        }
        if (off >= n) return n;
        uint32_t c = 0;
        if (const int w = bmp_seq(bytes, off, n, &c)) {
            const bool ok = ((bmp_bits[c >> 6] >> (c & 63)) & 1) ||
                            (c >= 0xAC00u && c <= 0xD7A3u &&
                             hangul_syllable_stable(c, bytes, off + static_cast<size_t>(w), n));
            if (!ok) return prev;
            prev = off;
            off += static_cast<size_t>(w);
            continue;
        }
        utf8proc_int32_t cp = 0;
        const auto width = utf8proc_iterate(bytes + off,
                                            static_cast<utf8proc_ssize_t>(n - off), &cp);
        if (width < 0 || !scalar_passes(cp, cache)) return prev;
        prev = off;
        off += static_cast<size_t>(width);
    }
}

}  // namespace
#endif

bool is_normalized_for_vocab(const HostVocab& hv, std::string_view text) {
    if (hv.normalizer_kind == NormalizerKind::None) {
        return true;
    }
    if (hv.normalizer_kind != NormalizerKind::NFC) {
        throw std::runtime_error("cuTokenize: unsupported tokenizer normalizer");
    }
#if GBPE_HAVE_VOCAB_QWEN25
    return nfc_scan(reinterpret_cast<const uint8_t*>(text.data()), text.size()) == text.size();
#else
    throw std::runtime_error(
        "cuTokenize: NFC tokenizer loaded by a build without Qwen-2.5 support");
#endif
}

bool normalize_view_for_vocab(const HostVocab& hv, std::string_view text, std::string& out) {
    if (hv.normalizer_kind == NormalizerKind::None) {
        return false;
    }
    if (hv.normalizer_kind != NormalizerKind::NFC) {
        throw std::runtime_error("cuTokenize: unsupported tokenizer normalizer");
    }

#if GBPE_HAVE_VOCAB_QWEN25
    const auto opts = kNfcOpts;
    const auto* bytes = reinterpret_cast<const utf8proc_uint8_t*>(text.data());
    const auto n_bytes = static_cast<utf8proc_ssize_t>(text.size());
    const size_t restart = nfc_scan(bytes, text.size());
    if (restart == text.size()) return false;
    const uint64_t* bulk_bits = bmp_bulk_bits();
    const SkipAsciiFn skip_ascii = skip_ascii_fn();
    StableCache cache;
    auto scalar_is_stable = [&](utf8proc_int32_t cp, const utf8proc_property_t* p) {
        return cache.is_stable(cp, p);
    };

    // Pass 2: normalize only the spans that can change. A stable ccc=0 scalar
    // is a safe boundary: nothing reorders past it and nothing composes
    // across it (a starter only composes with what FOLLOWS it, and the
    // preceding starter is included in the span). Stable runs are copied
    // verbatim; each maximal unstable span, extended back to the previous
    // stable starter, goes through utf8proc_decompose (which reorders) and
    // utf8proc_reencode (which composes) in one reused int32 buffer.
    // Barrier scalars are stable and never inside a normalized span.
    // Everything before `restart` passed pass 1, so it is copied as is and
    // the scan resumes at `restart` (see nfc_scan).
    std::vector<utf8proc_int32_t> buf;
    std::string& normalized = out;
    normalized.clear();
    normalized.reserve(text.size() + 16);

    // Short spans (a base letter plus its marks) recur constantly in scripts
    // that write vowel signs as separate scalars, so their NFC is memoized for
    // this call, keyed by the span's own bytes. NFC of a span depends only on
    // those bytes, so a hit returns exactly what utf8proc would.
    constexpr size_t kMemoMaxSpan = 32;
    std::unordered_map<std::string_view, std::string> memo;
    auto normalize_span = [&](size_t begin, size_t end) {
        if (begin >= end) return;
        const std::string_view key(text.data() + begin, end - begin);
        if (key.size() <= kMemoMaxSpan) {
            if (auto it = memo.find(key); it != memo.end()) {
                normalized.append(it->second);
                return;
            }
        }
        const size_t out_before = normalized.size();
        const auto* seg = bytes + begin;
        const auto seg_len = static_cast<utf8proc_ssize_t>(end - begin);
        const size_t need = static_cast<size_t>(seg_len) * 4 + 1;
        if (buf.size() < need) buf.resize(need);
        auto n = utf8proc_decompose(seg, seg_len, buf.data(),
                                    static_cast<utf8proc_ssize_t>(buf.size() - 1),
                                    opts);
        if (n >= 0 && static_cast<size_t>(n) + 1 > buf.size()) {
            buf.resize(static_cast<size_t>(n) + 1);
            n = utf8proc_decompose(seg, seg_len, buf.data(),
                                   static_cast<utf8proc_ssize_t>(buf.size() - 1),
                                   opts);
        }
        if (n < 0) {
            throw std::runtime_error(
                std::string("cuTokenize: NFC normalization failed: ") +
                utf8proc_errmsg(n));
        }
        const auto m = utf8proc_reencode(buf.data(), n, opts);
        if (m < 0) {
            throw std::runtime_error(
                std::string("cuTokenize: NFC normalization failed: ") +
                utf8proc_errmsg(m));
        }
        normalized.append(reinterpret_cast<const char*>(buf.data()),
                          static_cast<size_t>(m));
        if (key.size() <= kMemoMaxSpan) memo.emplace(key, normalized.substr(out_before));
    };

    // copied_to: bytes [0, copied_to) are already in `normalized`.
    // span_begin: start of the current pending unstable span, or SIZE_MAX.
    // prev_stable: byte offset of the most recent stable starter scalar.
    constexpr size_t kNoSpan = static_cast<size_t>(-1);
    size_t copied_to = 0;
    size_t span_begin = kNoSpan;
    size_t prev_stable = restart;
    utf8proc_ssize_t offset = static_cast<utf8proc_ssize_t>(restart);
    while (offset < n_bytes) {
        if (span_begin == kNoSpan) {
            // Outside a span a stable scalar only moves prev_stable, so runs
            // of ASCII and of table-stable BMP scalars are consumed in bulk.
            // Barriers are excluded here: they take the branch below.
            if (bytes[offset] < 0x80u) {
                const size_t a = skip_ascii(bytes, static_cast<size_t>(offset), text.size());
                prev_stable = a - 1;
                offset = static_cast<utf8proc_ssize_t>(a);
                if (offset >= n_bytes) break;
            }
            uint32_t c = 0;
            if (const int w = bmp_seq(bytes, static_cast<size_t>(offset), text.size(), &c)) {
                if (((bulk_bits[c >> 6] >> (c & 63)) & 1) ||
                    (c >= 0xAC00u && c <= 0xD7A3u &&
                     hangul_syllable_stable(c, bytes, static_cast<size_t>(offset) + static_cast<size_t>(w),
                                            text.size()))) {
                    prev_stable = static_cast<size_t>(offset);
                    offset += w;
                    continue;
                }
            }
        }
        const size_t here = static_cast<size_t>(offset);
        const utf8proc_uint8_t b = bytes[offset];
        utf8proc_int32_t cp = 0;
        utf8proc_ssize_t width = 1;
        bool stable = true;
        if (b >= 0x80u) {
            width = utf8proc_iterate(bytes + offset, n_bytes - offset, &cp);
            if (width < 0) {
                throw std::runtime_error(
                    std::string("cuTokenize: NFC normalization failed: ") +
                    utf8proc_errmsg(width));
            }
            if (is_hf_nfc_barrier(cp)) {
                // HF treats these as unassigned ccc=0 scalars and never
                // normalizes across them, whatever this utf8proc's tables
                // say. Hard boundary: close any span, copy verbatim, and do
                // not let the barrier itself start the next span.
                if (span_begin != kNoSpan) {
                    normalized.append(text.data() + copied_to,
                                      span_begin - copied_to);
                    normalize_span(span_begin, here);
                    span_begin = kNoSpan;
                } else {
                    normalized.append(text.data() + copied_to, here - copied_to);
                }
                normalized.append(text.data() + here, static_cast<size_t>(width));
                copied_to = here + static_cast<size_t>(width);
                prev_stable = copied_to;
                offset += width;
                continue;
            }
            const utf8proc_property_t* p = utf8proc_get_property(cp);
            const bool hangul = (cp >= 0x1100 && cp <= 0x11FF) ||
                                (cp >= 0xAC00 && cp <= 0xD7A3);
            stable = p->combining_class == 0 && !hangul &&
                     scalar_is_stable(cp, p);
        }
        if (stable) {
            if (span_begin != kNoSpan) {
                // Close the pending span at this stable starter.
                normalized.append(text.data() + copied_to,
                                  span_begin - copied_to);
                normalize_span(span_begin, here);
                copied_to = here;
                span_begin = kNoSpan;
            }
            prev_stable = here;
        } else if (span_begin == kNoSpan) {
            // Open a span from the previous stable starter so a base letter
            // can absorb the following mark exactly as full NFC would.
            span_begin = prev_stable;
        }
        offset += width;
    }
    if (span_begin != kNoSpan) {
        normalized.append(text.data() + copied_to, span_begin - copied_to);
        normalize_span(span_begin, text.size());
    } else {
        normalized.append(text.data() + copied_to, text.size() - copied_to);
    }
    return true;
#else
    throw std::runtime_error(
        "cuTokenize: NFC tokenizer loaded by a build without Qwen-2.5 support");
#endif
}

std::string normalize_for_vocab(const HostVocab& hv, const std::string& text) {
    std::string out;
    return normalize_view_for_vocab(hv, text, out) ? out : text;
}

void postprocess_decoded_bytes(const HostVocab& hv,
                               const uint32_t* token_ids,
                               uint32_t n_tokens,
                               const std::vector<uint8_t>& raw_bytes,
                               std::vector<uint8_t>& decoded_bytes) {
    if (!hv.byte_fallback) {
        decoded_bytes = raw_bytes;
        return;
    }
    if (n_tokens != 0 && token_ids == nullptr) {
        throw std::invalid_argument(
            "postprocess_decoded_bytes: token_ids is null for a non-empty decode");
    }

    constexpr uint8_t kLowbar[] = {0xE2u, 0x96u, 0x81u};
    decoded_bytes.clear();
    decoded_bytes.reserve(raw_bytes.size());
    size_t raw_offset = 0;

    for (uint32_t index = 0; index < n_tokens; ++index) {
        const uint32_t id = token_ids[index];
        if (id >= hv.token_bytes.size()) {
            throw std::invalid_argument(
                "postprocess_decoded_bytes: token id exceeds vocabulary size");
        }
        const size_t token_size = hv.token_bytes[id].size();
        if (token_size > raw_bytes.size() - raw_offset) {
            throw std::logic_error(
                "postprocess_decoded_bytes: gathered decode bytes are truncated");
        }

        const int16_t fallback_byte =
            id < hv.byte_fallback_byte_by_id.size()
                ? hv.byte_fallback_byte_by_id[id]
                : -1;
        if (fallback_byte >= 0) {
            // HF ByteFallback operates on decoded token pieces, not an
            // arbitrary spelling in the concatenated output. The ID-driven
            // lookup deliberately preserves that boundary.
            decoded_bytes.push_back(static_cast<uint8_t>(fallback_byte));
        } else {
            const uint8_t* token = raw_bytes.data() + raw_offset;
            for (size_t byte = 0; byte < token_size;) {
                if (byte + sizeof(kLowbar) <= token_size &&
                    std::equal(std::begin(kLowbar), std::end(kLowbar),
                               token + byte)) {
                    decoded_bytes.push_back(static_cast<uint8_t>(' '));
                    byte += sizeof(kLowbar);
                } else {
                    decoded_bytes.push_back(token[byte]);
                    ++byte;
                }
            }
        }
        raw_offset += token_size;
    }

    if (raw_offset != raw_bytes.size()) {
        throw std::logic_error(
            "postprocess_decoded_bytes: gathered decode bytes have an unexpected tail");
    }
}

void postprocess_decoded_utf8(const HostVocab& hv,
                              const uint32_t* token_ids,
                              uint32_t n_tokens,
                              const std::vector<uint8_t>& raw_bytes,
                              std::vector<uint8_t>& decoded_utf8) {
    if (!hv.byte_fallback) {
        decoded_utf8 = raw_bytes;
        return;
    }
    if (n_tokens != 0 && token_ids == nullptr) {
        throw std::invalid_argument(
            "postprocess_decoded_utf8: token_ids is null for a non-empty decode");
    }

    constexpr uint8_t kLowbar[] = {0xE2u, 0x96u, 0x81u};
    constexpr uint8_t kReplacement[] = {0xEFu, 0xBFu, 0xBDu};
    auto valid_utf8 = [](const std::vector<uint8_t>& bytes) {
        for (size_t offset = 0; offset < bytes.size();) {
            const uint8_t lead = bytes[offset];
            if (lead <= 0x7Fu) {
                ++offset;
                continue;
            }
            size_t width = 0;
            if (lead >= 0xC2u && lead <= 0xDFu) width = 2;
            else if (lead >= 0xE0u && lead <= 0xEFu) width = 3;
            else if (lead >= 0xF0u && lead <= 0xF4u) width = 4;
            else return false;
            if (offset + width > bytes.size()) return false;
            for (size_t i = 1; i < width; ++i) {
                if ((bytes[offset + i] & 0xC0u) != 0x80u) return false;
            }
            const uint8_t second = bytes[offset + 1];
            if ((lead == 0xE0u && second < 0xA0u) ||
                (lead == 0xEDu && second > 0x9Fu) ||
                (lead == 0xF0u && second < 0x90u) ||
                (lead == 0xF4u && second > 0x8Fu)) {
                return false;
            }
            offset += width;
        }
        return true;
    };

    decoded_utf8.clear();
    decoded_utf8.reserve(raw_bytes.size());
    std::vector<uint8_t> fallback_run;
    auto flush_fallback_run = [&]() {
        if (fallback_run.empty()) return;
        if (valid_utf8(fallback_run)) {
            decoded_utf8.insert(decoded_utf8.end(),
                                fallback_run.begin(), fallback_run.end());
        } else {
            // Hugging Face's ByteFallback emits a replacement for every byte
            // in a malformed contiguous run, including ASCII bytes that were
            // adjacent to the malformed lead/continuation byte.
            for (size_t i = 0; i < fallback_run.size(); ++i) {
                decoded_utf8.insert(decoded_utf8.end(),
                                    std::begin(kReplacement),
                                    std::end(kReplacement));
            }
        }
        fallback_run.clear();
    };

    size_t raw_offset = 0;
    for (uint32_t index = 0; index < n_tokens; ++index) {
        const uint32_t id = token_ids[index];
        if (id >= hv.token_bytes.size()) {
            throw std::invalid_argument(
                "postprocess_decoded_utf8: token id exceeds vocabulary size");
        }
        const size_t token_size = hv.token_bytes[id].size();
        if (token_size > raw_bytes.size() - raw_offset) {
            throw std::logic_error(
                "postprocess_decoded_utf8: gathered decode bytes are truncated");
        }

        const int16_t fallback_byte =
            id < hv.byte_fallback_byte_by_id.size()
                ? hv.byte_fallback_byte_by_id[id]
                : -1;
        if (fallback_byte >= 0) {
            fallback_run.push_back(static_cast<uint8_t>(fallback_byte));
        } else {
            flush_fallback_run();
            const uint8_t* token = raw_bytes.data() + raw_offset;
            for (size_t byte = 0; byte < token_size;) {
                if (byte + sizeof(kLowbar) <= token_size &&
                    std::equal(std::begin(kLowbar), std::end(kLowbar),
                               token + byte)) {
                    decoded_utf8.push_back(static_cast<uint8_t>(' '));
                    byte += sizeof(kLowbar);
                } else {
                    decoded_utf8.push_back(token[byte]);
                    ++byte;
                }
            }
        }
        raw_offset += token_size;
    }
    flush_fallback_run();

    if (raw_offset != raw_bytes.size()) {
        throw std::logic_error(
            "postprocess_decoded_utf8: gathered decode bytes have an unexpected tail");
    }
}

}  // namespace gbpe
