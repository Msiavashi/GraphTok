// INTERNAL seam gate: TokenizerCtx::encode_batch() must be BIT-EXACT with the
// concatenation of per-document encode() calls, for every byte-level family.
//
// The first block checks the batch PLUMBING (one staged input, one graph
// replay, one sync, correct per-document token offsets) on "safe" seams. The
// second block is the adversarial seam gate (mirrors tests/seam_corpus.py):
// digit runs, whitespace/CRLF runs, contractions, added-token straddles,
// DeepSeek CJK/punct/gap transitions, empty and 1-byte documents, and
// malformed / split multi-byte UTF-8 payloads. Ground truth is always the
// runtime concatenation of the single-document encode() of each document on
// the same context, so the expectation is valid by construction for whatever
// vocabulary is passed on the command line.
//
// Build: this is a first-class CMake target (`test_encode_batch_smoke`), built
// by default alongside the CLI for every preset that includes a byte-level
// vocabulary. It shares gbpe_configure_target() with gpu_bpe_tokenize, so its
// -DGBPE_* macros can never drift from the shipping binary's.
//   cmake --preset all && cmake --build build/all -j
// Qwen-2.5 is not in the `all` preset (host NFC): use the qwen25 preset for it.
//   cmake --preset qwen25 && cmake --build build/qwen25 -j
// Set -DGBPE_BUILD_TESTS=OFF to skip building the gate executables.
//
// Run: ./build/all/test_encode_batch_smoke data/hf_gpt2_tokenizer.json
//      ./build/all/test_encode_batch_smoke data/vocabs/llama3_tokenizer.json
//      ./build/all/test_encode_batch_smoke data/vocabs/deepseek_v3.json
//      ./build/qwen25/test_encode_batch_smoke data/vocabs/qwen25.json
#include "tokenizer.cuh"
#include "vocab.h"
#include "pretokenize.h"
#include <cstdio>
#include <algorithm>
#include <cstdlib>
#include <string>
#include <vector>

static int g_fail = 0;

static void check(bool ok, const char* name) {
    std::printf("[%s] %s\n", ok ? "PASS" : "FAIL", name);
    if (!ok) g_fail = 1;
}

int main(int argc, char** argv) {
    const std::string vocab_path =
        (argc > 1) ? argv[1] : "data/hf_gpt2_tokenizer.json";

    gbpe::HostVocab hv;
    if (!gbpe::load_hf_tokenizer_json(vocab_path, hv)) {
        std::fprintf(stderr, "vocab load failed: %s\n", vocab_path.c_str());
        return 1;
    }
    gbpe::VocabPack vp = gbpe::build_vocab_pack(hv);

    // Four documents, each terminated by a single newline (and each starting
    // with a non-space) so every seam falls on a GPT-2 pre-token boundary.
    std::vector<std::string> texts = {
        "The quick brown fox jumps over the lazy dog.\n",
        "Hello, world! Tokenization on the GPU, batched.\n",
        "Lorem ipsum dolor sit amet, consectetur adipiscing elit; 1234 5678.\n",
        "def main():\n    return sum(x * 2 for x in range(10))\n",
    };

    size_t total_bytes = 0, max_bytes = 0;
    for (auto& t : texts) {
        total_bytes += t.size();
        if (t.size() > max_bytes) max_bytes = t.size();
    }

    const uint32_t cap_pre   = gbpe::estimate_max_pretokens(total_bytes + 1024);
    const uint32_t cap_short = cap_pre;
    const uint32_t cap_long  = std::max(1024u, cap_pre / 64u);
    const uint32_t cap_in    = static_cast<uint32_t>(total_bytes + 1024);
    const uint32_t cap_out   = static_cast<uint32_t>(total_bytes + 1024);

    gbpe::TokenizerCtx ctx(vp, cap_short, cap_long, cap_in, cap_out,
                           /*max_decode_tokens=*/0, /*max_decode_bytes=*/0,
                           /*max_batch_docs=*/8);

    // Per-document reference: single-doc encode() on the SAME context.
    std::vector<std::vector<uint32_t>> ref(texts.size());
    for (size_t i = 0; i < texts.size(); ++i) {
        gbpe::EncodeInput ei{};
        ei.raw_text = reinterpret_cast<const uint8_t*>(texts[i].data());
        ei.raw_len  = static_cast<uint32_t>(texts[i].size());
        float k = 0, e = 0;
        ctx.encode(ei, ref[i], &k, &e);
    }

    auto run_batch = [&](size_t n, std::vector<uint32_t>& toks,
                         std::vector<uint32_t>& offs) {
        std::vector<const uint8_t*> ptrs(n);
        std::vector<uint32_t> lens(n);
        for (size_t i = 0; i < n; ++i) {
            ptrs[i] = reinterpret_cast<const uint8_t*>(texts[i].data());
            lens[i] = static_cast<uint32_t>(texts[i].size());
        }
        gbpe::BatchEncodeInput bi{ptrs.data(), lens.data(),
                                  static_cast<uint32_t>(n)};
        float k = 0, e = 0;
        ctx.encode_batch(bi, toks, offs, &k, &e);
    };

    auto expect_concat = [&](size_t n) {
        std::vector<uint32_t> v;
        for (size_t i = 0; i < n; ++i)
            v.insert(v.end(), ref[i].begin(), ref[i].end());
        return v;
    };
    auto expect_offsets = [&](size_t n) {
        std::vector<uint32_t> v(n + 1, 0);
        for (size_t i = 0; i < n; ++i)
            v[i + 1] = v[i] + static_cast<uint32_t>(ref[i].size());
        return v;
    };

    // (a) + (b): 4 docs — tokens equal the concatenation, offsets the running sums.
    {
        std::vector<uint32_t> toks, offs;
        run_batch(4, toks, offs);
        check(toks == expect_concat(4), "4-doc batch tokens == concat of per-doc encode()");
        check(offs == expect_offsets(4), "4-doc batch offsets == running sums");
    }

    // (c): a second call on the SAME ctx with a different doc count (graph replay reuse).
    {
        std::vector<uint32_t> toks, offs;
        run_batch(2, toks, offs);
        check(toks == expect_concat(2), "2-doc batch on reused ctx: tokens correct");
        check(offs == expect_offsets(2), "2-doc batch on reused ctx: offsets correct");
    }

    // (d): n_docs == 1 matches encode() exactly.
    {
        std::vector<uint32_t> toks, offs;
        run_batch(1, toks, offs);
        check(toks == ref[0], "n_docs=1 batch == single-doc encode()");
        check(offs.size() == 2 && offs[0] == 0 && offs[1] == ref[0].size(),
              "n_docs=1 offsets == {0, ntokens}");
    }

    // ---------------------------------------------------------------------
    // Seam gate (Task 4). Ground truth is computed at runtime as the
    // concatenation of the single-doc encode() of each document, so the
    // expectation is valid by construction for whatever vocab was loaded.
    // Every case is asserted at FULL equality (tokens AND offsets): after the
    // codepoint-domain document clamps, encode_batch must be bit-exact with
    // per-document encodes for all four byte-level families.
    // ---------------------------------------------------------------------
    auto run_case = [&](const char* name, std::vector<std::string> docs) {
        std::vector<uint32_t> want;
        std::vector<uint32_t> want_offs(docs.size() + 1, 0);
        for (size_t i = 0; i < docs.size(); ++i) {
            gbpe::EncodeInput ei{};
            ei.raw_text = reinterpret_cast<const uint8_t*>(docs[i].data());
            ei.raw_len  = static_cast<uint32_t>(docs[i].size());
            std::vector<uint32_t> one;
            float k = 0, e = 0;
            ctx.encode(ei, one, &k, &e);
            want.insert(want.end(), one.begin(), one.end());
            want_offs[i + 1] = want_offs[i] + static_cast<uint32_t>(one.size());
        }
        std::vector<const uint8_t*> ptrs(docs.size());
        std::vector<uint32_t> lens(docs.size());
        for (size_t i = 0; i < docs.size(); ++i) {
            ptrs[i] = reinterpret_cast<const uint8_t*>(docs[i].data());
            lens[i] = static_cast<uint32_t>(docs[i].size());
        }
        gbpe::BatchEncodeInput bi{ptrs.data(), lens.data(),
                                  static_cast<uint32_t>(docs.size())};
        std::vector<uint32_t> got, got_offs;
        float k = 0, e = 0;
        ctx.encode_batch(bi, got, got_offs, &k, &e);
        const bool ok = (got == want && got_offs == want_offs);
        check(ok, name);
        if (!ok) {
            std::printf("      want[%zu]:", want.size());
            for (uint32_t t : want) std::printf(" %u", t);
            std::printf("\n      got [%zu]:", got.size());
            for (uint32_t t : got) std::printf(" %u", t);
            std::printf("\n");
        }
    };

    // ---- Digit runs split at seams (Llama 3-cap, DeepSeek pass-1 cap) ------
    run_case("digits_2_3 {\"12\",\"345\"}", {"12", "345"});
    run_case("digits_7_2 {\"1234567\",\"89\"}", {"1234567", "89"});
    run_case("digits_deepseek_run {\"12345\",\"67\"}", {"12345", "67"});
    run_case("digits_6_3 {\"123456\",\"789\"}", {"123456", "789"});
    run_case("digits_doc_starts_digit {\"a\",\"1\",\"23456\"}",
             {"a", "1", "23456"});

    // ---- Whitespace --------------------------------------------------------
    run_case("ws_hello_world", {"hello ", "world"});
    run_case("ws_a_space_b", {"a", " b"});
    run_case("ws_long_run_split",
             {"x" + std::string(20, ' '), std::string(20, ' ') + "y"});
    run_case("ws_newline_seam", {"x\n", "\ny"});
    run_case("ws_crlf_seam", {"--\r", "\n\t\r\n"});
    run_case("ws_trailing_double_newline", {"a\n\n", "b"});
    run_case("ws_only_spaces_docs", {"   ", "   "});

    // ---- Contractions ------------------------------------------------------
    run_case("contraction_say_s", {"say", "'s"});
    run_case("contraction_it_apostrophe_s", {"it'", "s"});
    run_case("contraction_bang_s", {"!", "'s"});
    run_case("contraction_newline_towhead", {"\n", "'towhead"});
    run_case("contraction_upper_S", {"say", "'S"});
    run_case("contraction_doc_starts_apostrophe", {"say", "'ve x"});
    run_case("u017f_whole_in_one_doc",
             {"long-s \xC5\xBF whole", " codepoint stays together"});

    // ---- Added / special tokens (must NOT merge across the seam) -----------
    run_case("special_endoftext_split", {"<|endo", "ftext|>"});
    run_case("special_endoftext_split_padded", {"x<|endo", "ftext|>y"});
    run_case("special_endoftext_prefix", {"x", "<|endoftext|>y"});
    run_case("special_endoftext_exact_doc", {"a", "<|endoftext|>", "b"});
    run_case("special_endoftext_prefix_at_end", {"a\n", "b<|endo"});
    {
        // The added literal still must not MATCH across the seam (GPT-2 only:
        // id 50256 is <|endoftext|>). Equality above already implies it, but
        // this pins the intent explicitly.
        std::vector<std::string> docs = {"x<|endo", "ftext|>y"};
        std::vector<const uint8_t*> ptrs(2);
        std::vector<uint32_t> lens(2);
        for (size_t i = 0; i < 2; ++i) {
            ptrs[i] = reinterpret_cast<const uint8_t*>(docs[i].data());
            lens[i] = static_cast<uint32_t>(docs[i].size());
        }
        gbpe::BatchEncodeInput bi{ptrs.data(), lens.data(), 2u};
        std::vector<uint32_t> got, got_offs;
        float k = 0, e = 0;
        ctx.encode_batch(bi, got, got_offs, &k, &e);
        check(std::find(got.begin(), got.end(), 50256u) == got.end(),
              "added straddle: id 50256 does not match across seam");
    }

    // ---- DeepSeek: CJK / non-CJK transitions, punct and gap runs -----------
    run_case("deepseek_cjk_transition",
             {"\xE3\x82\xA2\xE3\x83\xB3\xE3\x83\x88",
              "\xE3\x83\x8B\xE3\x83\xBB\xE3\x82\xAC\xE3\x82\xA6\xE3\x83\x87"
              "\xE3\x82\xA3"});
    run_case("deepseek_punct_run_split", {"!!!", "???"});
    run_case("deepseek_ws_gap_run_split", {"a   ", "   b"});
    run_case("deepseek_ascii_punct_letter_seam", {"foo.", "bar"});

    // ---- Structure ---------------------------------------------------------
    run_case("structure_leading_empty", {"", "abc"});
    run_case("structure_interior_empty", {"abc", "", "def"});
    run_case("structure_trailing_empty", {"abc", ""});
    run_case("structure_all_empty", {"", "", ""});
    run_case("structure_one_byte_docs", {"a", "b", "c"});
    run_case("structure_single_doc", {"only one document here"});

    // ---- Multi-doc mixed cases --------------------------------------------
    run_case("mixed_digits_ws_contraction", {"12", "345 it'", "s ok", "6789"});
    run_case("mixed_special_cjk_ws",
             {"<|endo", "ftext|>\xE3\x82\xA2\xE3\x83\xB3\xE3\x83\x88 ",
              "\xE3\x83\x8B\xE3\x83\xBB end"});
    run_case("mixed_empty_digits_punct", {"", "42", "", "!!! 43", ""});

    // ---- Malformed / split multi-byte UTF-8 (byte-level, GPU-only) ---------
    run_case("byte_lone_lead_E2 | \\x96\\x81", {"a\n\xE2", "\x96\x81y\n"});
    run_case("byte_emoji_split_1_3", {"a\n\xF0", "\x9F\x98\x80y\n"});
    run_case("byte_emoji_split_3_1", {"a\n\xF0\x9F\x98", "\x80y\n"});
    run_case("byte_emoji_split_bare_1_3", {"\xF0", "\x9F\x98\x80"});
    run_case("byte_doc_starts_lone_continuation", {"a\n", "\x80\x81y\n"});
    run_case("byte_u017f_split", {"a\n\xC5", "\xBFy\n"});
    run_case("byte_u017f_split_bare", {"\xC5", "\xBF"});

    std::printf("%s\n", g_fail ? "SEAM GATE FAILED" : "SEAM GATE PASSED");
    return g_fail;
}
