// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 the GraphTok contributors

// gpu_bpe_tokenize -- CLI driver for the GPU BPE tokenizer.
//
// Usage:
//   gpu_bpe_tokenize --vocab data/hf_gpt2_tokenizer.json
//                    --input data/corpus/corpus_1m_tokens.txt
//                    --output /tmp/tok_ids.bin
//                    --runs 20 --warmup 3 --csv
//
// Output:
//   - binary file `/tmp/tok_ids.bin` containing uint32_t token IDs.
//   - if --csv is set, prints one CSV line per benchmark run to stdout:
//       run,n_tokens_out,n_pretokens,pretok_ms,kernel_ms,e2e_ms,full_e2e_ms
//   `e2e_ms` is backend staging + graph + output copy. `full_e2e_ms` measures
//   the complete per-call decode path, including any required host decoder
//   post-processing (Gemma Replace + ByteFallback).
//   - if --csv is not set, prints a human-readable summary.

#include "vocab.h"
#include "pretokenize.h"
#include "tokenizer.cuh"
#include "special_tokens.h"
#include "build_config.h"
#include <nlohmann/json.hpp>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include <cuda_runtime.h>
#include <nvtx3/nvToolsExt.h>

namespace {

struct Args {
    std::string vocab_path;
    std::string input_path;
    std::string inputs_csv;        // comma-separated list of files for batch mode
    std::string benchmark_input_list;  // newline-separated files, rotated per run
    std::string output_path = "/tmp/tok_ids.bin";
    int runs   = 5;
    int warmup = 3;
    bool csv = false;
    bool no_cuda_graph = false;
    // Emit one NVTX host range around every measured single-input replay.
    // This is deliberately benchmark-only instrumentation; it is off by
    // default and is not part of normal timing or CUDA-graph capture.
    bool nvtx_stage_profile = false;
    // F4: special-token insertion.
    bool add_bos = false;
    bool add_eos = false;
    std::string chat_path;   // JSON file with [{role,content},...]
    bool no_gen_prompt = false;
    // F5: decode mode (input is uint32 IDs binary; output is raw bytes).
    bool decode = false;
};

std::vector<std::string> split_csv(const std::string& s) {
    std::vector<std::string> out;
    size_t i = 0;
    while (i < s.size()) {
        size_t j = s.find(',', i);
        if (j == std::string::npos) { out.emplace_back(s.substr(i)); break; }
        out.emplace_back(s.substr(i, j - i));
        i = j + 1;
    }
    return out;
}

void usage(const char* argv0) {
    std::fprintf(stderr,
        "usage: %s --vocab <tokenizer.json> "
        "(--input <text> | --inputs <csv> | --benchmark-input-list <file> | "
        "--chat <chat.json>) "
        "[--output <file.bin>] [--runs N] [--warmup N] [--csv] [--no-cuda-graph] "
        "[--nvtx-stage-profile] "
        "[--add-bos] [--add-eos] [--no-generation-prompt]\n"
        "       %s --vocab <tokenizer.json> --inputs <file1,file2,...> "
        "[--output <file.bin>] [--runs N] [--warmup N] [--csv]\n"
        "           (batch mode: one graph replay for all listed files; "
        "--output holds the concatenated token IDs for all documents, "
        "back to back in input order)\n",
        argv0, argv0);
}

bool parse_args(int argc, char** argv, Args& a) {
    for (int i = 1; i < argc; ++i) {
        std::string k = argv[i];
        auto next = [&]() -> const char* {
            if (i + 1 >= argc) { std::fprintf(stderr, "missing value for %s\n", k.c_str()); std::exit(2); }
            return argv[++i];
        };
        if (k == "--vocab")        a.vocab_path = next();
        else if (k == "--input")   a.input_path = next();
        else if (k == "--inputs")  a.inputs_csv = next();
        else if (k == "--benchmark-input-list") a.benchmark_input_list = next();
        else if (k == "--output")  a.output_path = next();
        else if (k == "--runs")    a.runs   = std::atoi(next());
        else if (k == "--warmup")  a.warmup = std::atoi(next());
        else if (k == "--csv")     a.csv = true;
        else if (k == "--no-cuda-graph") a.no_cuda_graph = true;
        else if (k == "--nvtx-stage-profile") a.nvtx_stage_profile = true;
        else if (k == "--add-bos") a.add_bos = true;
        else if (k == "--add-eos") a.add_eos = true;
        else if (k == "--chat")    a.chat_path = next();
        else if (k == "--no-generation-prompt") a.no_gen_prompt = true;
        else if (k == "--decode")  a.decode = true;
        else if (k == "-h" || k == "--help") { usage(argv[0]); std::exit(0); }
        else { std::fprintf(stderr, "unknown flag %s\n", k.c_str()); usage(argv[0]); return false; }
    }
    if (a.vocab_path.empty()) { usage(argv[0]); return false; }
    const int input_modes =
        !a.input_path.empty() + !a.inputs_csv.empty() +
        !a.benchmark_input_list.empty() + !a.chat_path.empty();
    if (input_modes != 1 || a.runs < 1 || a.warmup < 0) {
        usage(argv[0]);
        return false;
    }
    if (a.nvtx_stage_profile && a.input_path.empty()) {
        std::fprintf(stderr,
            "--nvtx-stage-profile is supported only with --input; "
            "it labels each measured single-input graph replay\\n");
        return false;
    }
    return true;
}

std::string read_file(const std::string& p) {
    std::ifstream f(p, std::ios::binary);
    if (!f) { std::fprintf(stderr, "cannot open %s\n", p.c_str()); std::exit(1); }
    std::ostringstream ss; ss << f.rdbuf(); return ss.str();
}

std::vector<std::string> read_input_list(const std::string& path) {
    std::istringstream lines(read_file(path));
    std::vector<std::string> paths;
    std::string line;
    while (std::getline(lines, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (!line.empty()) paths.push_back(line);
    }
    if (paths.empty()) {
        std::fprintf(stderr, "benchmark input list is empty: %s\n", path.c_str());
        std::exit(1);
    }
    return paths;
}

// Encode a single text string into BPE ids, allocating only what's needed.
// Used by the --chat path which has many small pieces.
std::vector<uint32_t> encode_one(const gbpe::HostVocab& hv,
                                 const gbpe::VocabPack& vp,
                                 const std::string& text) {
    if (text.empty()) return {};
    std::string normalized;
    const std::string* input = &text;
    if (hv.normalizer_kind != gbpe::NormalizerKind::None) {
        normalized = gbpe::normalize_for_vocab(hv, text);
        input = &normalized;
    }
    uint32_t max_pretokens = gbpe::estimate_max_pretokens(input->size());
    uint32_t max_short = std::max<uint32_t>(max_pretokens, 16);
    uint32_t max_long  = gbpe::estimate_max_long_pretokens(input->size(), 16);
    uint32_t max_input_bytes = gbpe::checked_byte_capacity(input->size());
    uint32_t max_output_tokens = gbpe::checked_byte_capacity(input->size());

    // GPU pre-tokenizer mode: pass raw bytes; the device buckets inside the
    // captured graph. No host byte-level pre-tokenizer.
    gbpe::TokenizerCtx ctx(vp, max_short, max_long, max_input_bytes, max_output_tokens);
    gbpe::EncodeInput ei{};
    ei.raw_text = reinterpret_cast<const uint8_t*>(input->data());
    ei.raw_len  = static_cast<uint32_t>(input->size());

    std::vector<uint32_t> out;
    float k_ms = 0, e2e_ms = 0;
    ctx.encode(ei, out, &k_ms, &e2e_ms);
    return out;
}

}  // namespace


int main(int argc, char** argv) {
    Args args;
    if (!parse_args(argc, argv, args)) return 2;

    // Load vocab + build device-side pack.
    gbpe::HostVocab hv;
    if (!gbpe::load_hf_tokenizer_json(args.vocab_path, hv)) return 1;
    gbpe::VocabPack vp = gbpe::build_vocab_pack(hv);

    // ---- F4: --chat <chat.json> path ----
    if (!args.chat_path.empty()) {
        std::string js = read_file(args.chat_path);
        nlohmann::json cj;
        try { cj = nlohmann::json::parse(js); }
        catch (const std::exception& e) {
            std::fprintf(stderr, "[main] --chat: JSON parse error: %s\n", e.what());
            return 1;
        }
        std::vector<gbpe::ChatTurn> turns;
        // Accept either {"messages":[...]} or a bare array of {role,content}.
        const nlohmann::json* arr = nullptr;
        if (cj.is_array()) arr = &cj;
        else if (cj.contains("messages") && cj["messages"].is_array()) arr = &cj["messages"];
        else { std::fprintf(stderr, "[main] --chat: expected array or {messages:[...]}\n"); return 1; }
        for (const auto& m : *arr) {
            gbpe::ChatTurn t;
            t.role    = m.value("role", std::string{});
            t.content = m.value("content", std::string{});
            turns.push_back(std::move(t));
        }
        gbpe::ChatTemplate tmpl = gbpe::detect_chat_template(hv);
        bool add_gen = !args.no_gen_prompt;
        std::vector<gbpe::ChatPiece> pieces;
        try { pieces = gbpe::render_chat(hv, tmpl, turns, add_gen); }
        catch (const std::exception& e) {
            std::fprintf(stderr, "[main] --chat: %s\n", e.what());
            return 1;
        }
        std::vector<uint32_t> tokens;
        for (const auto& p : pieces) {
            if (p.kind == gbpe::ChatPiece::Id) {
                tokens.push_back(p.id);
            } else {
                auto ids = encode_one(hv, vp, p.text);
                tokens.insert(tokens.end(), ids.begin(), ids.end());
            }
        }
        std::fprintf(stderr, "[main] chat: %zu pieces -> %zu tokens\n",
                     pieces.size(), tokens.size());
        std::ofstream o(args.output_path, std::ios::binary);
        if (!o) { std::fprintf(stderr, "cannot write %s\n", args.output_path.c_str()); return 1; }
        o.write(reinterpret_cast<const char*>(tokens.data()),
                static_cast<std::streamsize>(tokens.size() * sizeof(uint32_t)));
        gbpe::free_vocab_pack(vp);
        return 0;
    }

    // ---- Unified benchmark mode: rotate one exact-sized input per run. ----
    //
    // The input-list file contains one path per line. A single path is reused
    // for fixed-text mode. Otherwise warmups consume the first `warmup` paths
    // and timed runs consume the following `runs` paths. Vocabulary loading,
    // allocations, and graph capture are shared across the complete sequence.
    if (!args.benchmark_input_list.empty()) {
#if GBPE_HAVE_FAMILY_SP && !GBPE_GEMMA_GPU_PRETOK
        if (hv.byte_fallback) {
            std::fprintf(stderr,
                "--benchmark-input-list requires the graph-resident Gemma "
                "pre-tokenizer; this binary uses the retained host fallback\n");
            gbpe::free_vocab_pack(vp);
            return 1;
        }
#endif
        std::vector<std::string> paths =
            read_input_list(args.benchmark_input_list);
        if (paths.size() != 1 &&
            paths.size() < static_cast<size_t>(args.warmup + args.runs)) {
            std::fprintf(stderr,
                "benchmark input list has %zu path(s), but %d warmup + %d "
                "timed inputs are required (or provide one fixed path)\n",
                paths.size(), args.warmup, args.runs);
            gbpe::free_vocab_pack(vp);
            return 1;
        }

        std::vector<std::string> raw_texts;
        raw_texts.reserve(paths.size());
        size_t max_backend_bytes = 0;
        for (const auto& path : paths) {
            raw_texts.push_back(read_file(path));
            if (hv.normalizer_kind == gbpe::NormalizerKind::None) {
                max_backend_bytes =
                    std::max(max_backend_bytes, raw_texts.back().size());
            } else {
                std::string normalized =
                    gbpe::normalize_for_vocab(hv, raw_texts.back());
                max_backend_bytes =
                    std::max(max_backend_bytes, normalized.size());
            }
        }
        std::fprintf(stderr,
            "[main] benchmark inputs: %zu text(s), max backend bytes=%zu\n",
            raw_texts.size(), max_backend_bytes);

        uint32_t max_pretokens =
            gbpe::estimate_max_pretokens(max_backend_bytes);
        uint32_t max_short = max_pretokens;
        uint32_t max_long =
            gbpe::estimate_max_long_pretokens(max_backend_bytes);
        uint32_t max_input_bytes =
            gbpe::checked_byte_capacity(max_backend_bytes);
        uint32_t max_output_tokens =
            gbpe::checked_byte_capacity(max_backend_bytes);
        gbpe::TokenizerCtx ctx(
            vp, max_short, max_long, max_input_bytes, max_output_tokens,
            0, 0, 1, !args.no_cuda_graph);

        std::vector<uint32_t> tokens;
        std::vector<uint32_t> flat_tokens;
        std::vector<uint32_t> output_offsets;
        output_offsets.reserve(static_cast<size_t>(args.runs) + 1);
        output_offsets.push_back(0);

        auto path_index = [&](int iteration, bool timed) -> size_t {
            if (raw_texts.size() == 1) return 0;
            const int base = timed ? args.warmup : 0;
            return static_cast<size_t>(base + iteration);
        };

        auto encode_input = [&](size_t index, float& kernel_ms,
                                float& backend_e2e_ms, float& full_e2e_ms,
                                size_t& backend_bytes) {
            const std::string& raw = raw_texts[index];
            auto full_start = std::chrono::high_resolution_clock::now();
            std::string normalized;
            const std::string* backend_text = &raw;
            if (hv.normalizer_kind != gbpe::NormalizerKind::None) {
                normalized = gbpe::normalize_for_vocab(hv, raw);
                backend_text = &normalized;
            }
            backend_bytes = backend_text->size();

            gbpe::EncodeInput input{};
            input.raw_text = reinterpret_cast<const uint8_t*>(
                backend_text->data());
            input.raw_len = static_cast<uint32_t>(backend_text->size());
            ctx.encode(
                input, tokens, &kernel_ms, &backend_e2e_ms);
            auto full_stop = std::chrono::high_resolution_clock::now();
            full_e2e_ms = std::chrono::duration<float, std::milli>(
                full_stop - full_start).count();
        };

        for (int i = 0; i < args.warmup; ++i) {
            float kernel_ms = 0.f;
            float backend_e2e_ms = 0.f;
            float full_e2e_ms = 0.f;
            size_t backend_bytes = 0;
            encode_input(
                path_index(i, false), kernel_ms, backend_e2e_ms,
                full_e2e_ms, backend_bytes);
        }

        if (args.csv) {
            std::printf(
                "run,input_index,n_bytes_raw,n_bytes_backend,n_tokens_out,"
                "n_pretokens,pretok_ms,kernel_ms,e2e_ms,full_e2e_ms\n");
        }
        for (int i = 0; i < args.runs; ++i) {
            const size_t index = path_index(i, true);
            float kernel_ms = 0.f;
            float backend_e2e_ms = 0.f;
            float full_e2e_ms = 0.f;
            size_t backend_bytes = 0;
            encode_input(
                index, kernel_ms, backend_e2e_ms, full_e2e_ms,
                backend_bytes);

            if (flat_tokens.size() + tokens.size() > 0xFFFFFFFFull) {
                std::fprintf(stderr,
                    "benchmark output exceeds uint32 offset capacity\n");
                gbpe::free_vocab_pack(vp);
                return 1;
            }
            flat_tokens.insert(
                flat_tokens.end(), tokens.begin(), tokens.end());
            output_offsets.push_back(
                static_cast<uint32_t>(flat_tokens.size()));

            if (args.csv) {
                std::printf(
                    "%d,%zu,%zu,%zu,%zu,0,0.0000,%.4f,%.4f,%.4f\n",
                    i, index, raw_texts[index].size(), backend_bytes,
                    tokens.size(), kernel_ms, backend_e2e_ms, full_e2e_ms);
            } else {
                std::fprintf(stderr,
                    "[run %2d] input=%zu raw_bytes=%zu backend_bytes=%zu "
                    "tokens=%zu graph=%.3f ms e2e=%.3f ms "
                    "full_e2e=%.3f ms\n",
                    i, index, raw_texts[index].size(), backend_bytes,
                    tokens.size(), kernel_ms, backend_e2e_ms, full_e2e_ms);
            }
        }

        // Packed correctness output:
        // [n_runs:u32][offsets:u32 x (n_runs+1)][flat token ids:u32].
        {
            std::ofstream output(args.output_path, std::ios::binary);
            if (!output) {
                std::fprintf(
                    stderr, "cannot write %s\n", args.output_path.c_str());
                gbpe::free_vocab_pack(vp);
                return 1;
            }
            uint32_t n_outputs = static_cast<uint32_t>(args.runs);
            output.write(
                reinterpret_cast<const char*>(&n_outputs),
                sizeof(uint32_t));
            output.write(
                reinterpret_cast<const char*>(output_offsets.data()),
                static_cast<std::streamsize>(
                    output_offsets.size() * sizeof(uint32_t)));
            output.write(
                reinterpret_cast<const char*>(flat_tokens.data()),
                static_cast<std::streamsize>(
                    flat_tokens.size() * sizeof(uint32_t)));
        }
        std::fprintf(stderr,
            "[main] benchmark wrote %d output stream(s), %zu total tokens "
            "to %s\n",
            args.runs, flat_tokens.size(), args.output_path.c_str());
        gbpe::free_vocab_pack(vp);
        return 0;
    }

    // -------- Batch mode: --inputs file1,file2,...  --------
    if (!args.inputs_csv.empty()) {
        std::vector<std::string> paths = split_csv(args.inputs_csv);
        std::vector<std::string> docs;
        docs.reserve(paths.size());
        size_t total_bytes = 0;
        for (const auto& p : paths) {
            docs.push_back(read_file(p));
            if (hv.normalizer_kind != gbpe::NormalizerKind::None) {
                docs.back() = gbpe::normalize_for_vocab(hv, docs.back());
            }
            total_bytes += docs.back().size();
        }
        std::fprintf(stderr, "[main] batch: %zu docs, %zu total bytes\n",
                     docs.size(), total_bytes);

        uint32_t max_pretokens = gbpe::estimate_max_pretokens(total_bytes);
        uint32_t max_short = max_pretokens;
        uint32_t max_long  = gbpe::estimate_max_long_pretokens(total_bytes);
        uint32_t max_input_bytes = gbpe::checked_byte_capacity(total_bytes);
        uint32_t max_output_tokens = gbpe::checked_byte_capacity(total_bytes);

        // Byte-level vocabs run ALL documents through a single encode_batch()
        // call on one TokenizerCtx: one staged H2D of the concatenated bytes,
        // one graph replay, one host sync. Document boundaries are isolated
        // inside the graph, so the result is bit-identical to encoding each doc
        // separately and concatenating. The flat-tokens + per-doc-offset output
        // layout is unchanged; doc_off comes straight from encode_batch().
        //
        // SP-family vocabs (Gemma 3) have no batch path — encode_batch() throws
        // on those contexts — so they keep the original serial per-doc encode()
        // loop on a reused ctx, which is what this mode always did. Same output
        // file layout and same CSV columns either way; only the timing
        // attribution differs (summed per-doc times vs one batch call).
        const bool batch_capable = !hv.byte_fallback;
        gbpe::TokenizerCtx ctx(vp, max_short, max_long, max_input_bytes, max_output_tokens,
                               /*max_decode_tokens*/ 0, /*max_decode_bytes*/ 0,
                               /*max_batch_docs*/ batch_capable
                                   ? static_cast<uint32_t>(paths.size()) : 1u,
                               /*use_cuda_graph*/ !args.no_cuda_graph);
        if (!batch_capable) {
            std::fprintf(stderr,
                "[main] batch: SP-family vocab has no GPU batch path; "
                "falling back to the serial per-document encode loop\n");
        }

        std::vector<const uint8_t*> doc_ptrs;
        std::vector<uint32_t>       doc_lens;
        doc_ptrs.reserve(docs.size());
        doc_lens.reserve(docs.size());
        for (const auto& d : docs) {
            doc_ptrs.push_back(reinterpret_cast<const uint8_t*>(d.data()));
            doc_lens.push_back(static_cast<uint32_t>(d.size()));
        }
        gbpe::BatchEncodeInput bi{};
        bi.docs   = doc_ptrs.data();
        bi.lens   = doc_lens.data();
        bi.n_docs = static_cast<uint32_t>(docs.size());

        std::vector<uint32_t> tokens;
        std::vector<uint32_t> doc_off;
        float k_ms = 0, e2e_ms = 0;
        auto encode_all = [&]() {
            if (batch_capable) {
                // doc_off comes back as n_docs+1 running sums with
                // doc_off[n_docs] == tokens.size() (checked below).
                ctx.encode_batch(bi, tokens, doc_off, &k_ms, &e2e_ms);
                return;
            }
            // SP fallback: encode each document on the reused ctx and build the
            // same running-sum offsets by hand.
            tokens.clear(); doc_off.clear();
            doc_off.push_back(0);
            float kt = 0, et = 0;
            for (const auto& d : docs) {
                gbpe::EncodeInput ei{};
                ei.raw_text = reinterpret_cast<const uint8_t*>(d.data());
                ei.raw_len  = static_cast<uint32_t>(d.size());
                std::vector<uint32_t> doc_tokens;
                float dk = 0, de = 0;
                ctx.encode(ei, doc_tokens, &dk, &de);
                kt += dk; et += de;
                tokens.insert(tokens.end(), doc_tokens.begin(), doc_tokens.end());
                doc_off.push_back(static_cast<uint32_t>(tokens.size()));
            }
            k_ms = kt; e2e_ms = et;
        };
        for (int i = 0; i < args.warmup; ++i) encode_all();
        if (args.csv) {
            std::printf("run,n_docs,n_tokens_out,kernel_ms,e2e_ms\n");
        }
        for (int i = 0; i < args.runs; ++i) {
            encode_all();
            if (args.csv) {
                std::printf("%d,%zu,%zu,%.4f,%.4f\n",
                            i, docs.size(), tokens.size(), k_ms, e2e_ms);
            } else {
                std::fprintf(stderr,
                    "[run %2d] docs=%zu tokens=%zu  kernel=%.3f ms  e2e=%.3f ms\n",
                    i, docs.size(), tokens.size(), k_ms, e2e_ms);
            }
        }

        // Output format: header [n_docs:u32][doc_off:u32 x (n_docs+1)] then flat tokens.
        {
            std::ofstream o(args.output_path, std::ios::binary);
            if (!o) { std::fprintf(stderr, "cannot write %s\n", args.output_path.c_str()); return 1; }
            uint32_t n_docs = static_cast<uint32_t>(docs.size());
            if (doc_off.size() != static_cast<size_t>(n_docs) + 1 ||
                doc_off.back() != static_cast<uint32_t>(tokens.size())) {
                std::fprintf(stderr,
                    "[main] batch: bad doc offsets (size=%zu expected=%u, "
                    "back=%u tokens=%zu)\n",
                    doc_off.size(), n_docs + 1,
                    doc_off.empty() ? 0u : doc_off.back(), tokens.size());
                return 1;
            }
            o.write(reinterpret_cast<const char*>(&n_docs), sizeof(uint32_t));
            o.write(reinterpret_cast<const char*>(doc_off.data()),
                    static_cast<std::streamsize>(doc_off.size() * sizeof(uint32_t)));
            o.write(reinterpret_cast<const char*>(tokens.data()),
                    static_cast<std::streamsize>(tokens.size() * sizeof(uint32_t)));
            std::fprintf(stderr,
                "[main] batch wrote n_docs=%u  total_tokens=%zu  to %s\n",
                n_docs, tokens.size(), args.output_path.c_str());
        }

        gbpe::free_vocab_pack(vp);
        return 0;
    }

    // ---- F5: --decode path (uint32 IDs binary -> raw bytes) ----
    if (args.decode) {
        std::string raw = read_file(args.input_path);
        if (raw.size() % sizeof(uint32_t) != 0) {
            std::fprintf(stderr,
                "[main] --decode: input size %zu is not a multiple of 4 bytes\n",
                raw.size());
            return 1;
        }
        uint32_t n_tokens = static_cast<uint32_t>(raw.size() / sizeof(uint32_t));
        const uint32_t* ids = reinterpret_cast<const uint32_t*>(raw.data());

        // Upper bound on output bytes: per-token max ~128, use 256 for headroom.
        uint32_t cap_tokens = n_tokens;
        uint64_t cap_bytes64 = static_cast<uint64_t>(n_tokens) * 256ull + 1024ull;
        if (cap_bytes64 > 0xFFFFFFFFull) cap_bytes64 = 0xFFFFFFFFull;
        uint32_t cap_bytes = static_cast<uint32_t>(cap_bytes64);

        // Decode-only ctx (minimal encode capacities).
        gbpe::TokenizerCtx ctx(vp,
            /*max_short*/ 1, /*max_long*/ 1,
            /*max_input_bytes*/ 1024,
            /*max_output_tokens*/ 1024,
            cap_tokens, cap_bytes);

        std::vector<uint8_t> raw_out, out;
        float k_ms = 0, backend_e2e_ms = 0, full_e2e_ms = 0;
        auto decode_once = [&]() {
            // Keep TokenizerCtx's existing backend timing separate from the
            // complete decode-call time. In particular, Gemma's serialized
            // Replace + ByteFallback post-decoder is required output work and
            // must not be omitted from an end-to-end measurement.
            const auto full_start = std::chrono::high_resolution_clock::now();
            if (hv.byte_fallback) {
                ctx.decode(ids, n_tokens, raw_out, &k_ms, &backend_e2e_ms);
                gbpe::postprocess_decoded_bytes(
                    hv, ids, n_tokens, raw_out, out);
            } else {
                // Preserve the former byte-level path: no extra host copy is
                // introduced merely to produce a full-E2E timing.
                ctx.decode(ids, n_tokens, out, &k_ms, &backend_e2e_ms);
            }
            const auto full_end = std::chrono::high_resolution_clock::now();
            full_e2e_ms = std::chrono::duration<float, std::milli>(
                full_end - full_start).count();
        };
        for (int i = 0; i < args.warmup; ++i) {
            decode_once();
        }
        if (args.csv) {
            std::printf(
                "run,n_tokens_in,n_bytes_out,kernel_ms,backend_e2e_ms,full_e2e_ms\n");
        }
        for (int i = 0; i < args.runs; ++i) {
            decode_once();
            if (args.csv) {
                std::printf("%d,%u,%zu,%.4f,%.4f,%.4f\n",
                    i, n_tokens, out.size(), k_ms, backend_e2e_ms, full_e2e_ms);
            } else {
                std::fprintf(stderr,
                    "[run %2d] decode tokens=%u bytes=%zu  kernel=%.3f ms  "
                    "backend_e2e=%.3f ms  full_e2e=%.3f ms\n",
                    i, n_tokens, out.size(), k_ms, backend_e2e_ms, full_e2e_ms);
            }
        }
        {
            std::ofstream o(args.output_path, std::ios::binary);
            if (!o) { std::fprintf(stderr, "cannot write %s\n", args.output_path.c_str()); return 1; }
            o.write(reinterpret_cast<const char*>(out.data()),
                    static_cast<std::streamsize>(out.size()));
            std::fprintf(stderr, "[main] decode wrote %zu bytes to %s\n",
                         out.size(), args.output_path.c_str());
        }
        gbpe::free_vocab_pack(vp);
        return 0;
    }

    // Read input text.
    std::string text = read_file(args.input_path);
    std::fprintf(stderr, "[main] input %zu bytes\n", text.size());
    if (hv.normalizer_kind != gbpe::NormalizerKind::None) {
        text = gbpe::normalize_for_vocab(hv, text);
        std::fprintf(stderr, "[main] normalized input %zu bytes\n", text.size());
    }

    // Size capacities. estimate_max_pretokens is an upper bound on the TOTAL
    // pre-token count; we split it across the two buckets generously since
    // the actual partition is data-dependent. The short bucket dominates (the
    // measured distribution shows ~99.9999% short on English prose) so we
    // give it the lion's share. The long bucket needs at least a handful of
    // slots to handle outliers without aborting.
    uint32_t max_pretokens = gbpe::estimate_max_pretokens(text.size());
    uint32_t max_short = max_pretokens;             // worst case: all short
    uint32_t max_long  = gbpe::estimate_max_long_pretokens(text.size());
    uint32_t max_input_bytes = gbpe::checked_byte_capacity(text.size());
    uint32_t max_output_tokens = gbpe::checked_byte_capacity(text.size());

    // Host pre-token buffers (per bucket).
    std::vector<uint8_t>  h_short_bytes(static_cast<size_t>(max_short) * gbpe::MAX_PRETOKEN_LEN_SHORT);
    std::vector<uint16_t> h_short_lens (max_short);
    std::vector<uint32_t> h_short_orig (max_short);
    std::vector<uint8_t>  h_long_bytes (static_cast<size_t>(max_long)  * gbpe::MAX_PRETOKEN_LEN_LONG);
    std::vector<uint16_t> h_long_lens  (max_long);
    std::vector<uint32_t> h_long_orig  (max_long);

#if GBPE_HAVE_FAMILY_SP && !GBPE_GEMMA_GPU_PRETOK
    const bool sp_family = hv.byte_fallback;
#endif

#if GBPE_HAVE_FAMILY_SP && !GBPE_GEMMA_GPU_PRETOK
    // SP-family extra buffers: uint32 id rows + overflow result rows.
    std::vector<uint32_t> h_short_ids;
    std::vector<uint32_t> h_long_ids;
    constexpr uint32_t kOverflowCap = 1024;
    std::vector<uint32_t> h_overflow_orig_idx;
    std::vector<uint32_t> h_overflow_result_ids;
    std::vector<uint16_t> h_overflow_result_lens;
    if (sp_family) {
        h_short_ids.assign(static_cast<size_t>(max_short) * gbpe::MAX_PRETOKEN_LEN_SHORT, 0);
        h_long_ids.assign(static_cast<size_t>(max_long)  * gbpe::MAX_PRETOKEN_LEN_LONG,  0);
        h_overflow_orig_idx.assign(kOverflowCap, 0);
        h_overflow_result_ids.assign(static_cast<size_t>(kOverflowCap) * gbpe::MAX_PRETOKEN_LEN_LONG, 0);
        h_overflow_result_lens.assign(kOverflowCap, 0);
    }
#endif

    gbpe::EncodeInput ei{};
    float pretok_ms = 0.f;
    uint32_t pretok_total = 0;

#if GBPE_HAVE_FAMILY_SP && !GBPE_GEMMA_GPU_PRETOK
    gbpe::SPVocabInfo sp_vi{};
    gbpe::SPResult sp_result{};
    std::unique_ptr<gbpe::SPPreTokenizer> sp_pretok;

    if (sp_family) {
        sp_vi.id_by_token = &hv.id_by_token;
        sp_vi.byte_fallback = hv.byte_fallback_id.data();
        sp_vi.max_token_byte_len = hv.max_token_byte_len;
        sp_vi.host_merges = &hv.host_merges;
        sp_pretok = std::make_unique<gbpe::SPPreTokenizer>(sp_vi);

        sp_result.short_ids       = h_short_ids.data();
        sp_result.short_lens      = h_short_lens.data();
        sp_result.short_orig_idx  = h_short_orig.data();
        sp_result.cap_short       = max_short;
        sp_result.long_ids        = h_long_ids.data();
        sp_result.long_lens       = h_long_lens.data();
        sp_result.long_orig_idx   = h_long_orig.data();
        sp_result.cap_long        = max_long;
        sp_result.overflow_orig_idx    = h_overflow_orig_idx.data();
        sp_result.overflow_result_ids  = h_overflow_result_ids.data();
        sp_result.overflow_result_lens = h_overflow_result_lens.data();
        sp_result.cap_overflow         = kOverflowCap;

        ei.short_ids        = h_short_ids.data();
        ei.short_lens       = h_short_lens.data();
        ei.short_orig_idx   = h_short_orig.data();
        ei.long_ids         = h_long_ids.data();
        ei.long_lens        = h_long_lens.data();
        ei.long_orig_idx    = h_long_orig.data();
        ei.overflow_orig_idx    = h_overflow_orig_idx.data();
        ei.overflow_result_ids  = h_overflow_result_ids.data();
        ei.overflow_result_lens = h_overflow_result_lens.data();
        // Byte-buffer pointers unused but set to non-null for safety.
        ei.short_bytes = h_short_bytes.data();
        ei.long_bytes  = h_long_bytes.data();
    } else
#endif
    {
        // GPU pre-tokenizer mode: skip the host PCRE2 pretok entirely; the
        // device produces the bucketed inputs inside the captured graph. Pass
        // the raw text bytes + length. This is now the only byte-level path
        // (the legacy host PCRE2 byte-level pre-tokenizer has been removed).
        ei.raw_text  = reinterpret_cast<const uint8_t*>(text.data());
        ei.raw_len   = static_cast<uint32_t>(text.size());
        pretok_total = 0;  // device-only; reported as 0 in this mode
        std::fprintf(stderr, "[main] gpu-pretok: %zu raw bytes (device bucketing)\n",
                     text.size());
        // Byte-buffer pointers unused but set non-null for safety.
        ei.short_bytes = h_short_bytes.data();
        ei.long_bytes  = h_long_bytes.data();
    }

#if GBPE_HAVE_FAMILY_SP && !GBPE_GEMMA_GPU_PRETOK
    // Repeat the host SP work for every benchmark iteration. Besides keeping
    // the ID buckets current, this makes full_e2e_ms a genuine raw-text timing
    // rather than adding one setup sample to a separately timed GPU replay.
    auto prepare_sp_input = [&]() -> float {
        auto t0 = std::chrono::high_resolution_clock::now();
        sp_pretok->apply(text, sp_result);
        auto t1 = std::chrono::high_resolution_clock::now();

        ei.n_short    = sp_result.n_short;
        ei.n_long     = sp_result.n_long;
        ei.n_overflow = sp_result.n_overflow;
        ei.n_total    = sp_result.n_total;
        pretok_total  = sp_result.n_total;
        return std::chrono::duration<float, std::milli>(t1 - t0).count();
    };

    if (sp_family) {
        pretok_ms = prepare_sp_input();
        std::fprintf(stderr,
            "[main] sp-pretok: %u pre-tokens (%u short + %u long + %u overflow) in %.2f ms\n",
            sp_result.n_total, sp_result.n_short, sp_result.n_long,
            sp_result.n_overflow, pretok_ms);
    }
#endif

    gbpe::TokenizerCtx ctx(vp, max_short, max_long, max_input_bytes, max_output_tokens,
                           0, 0, 1, !args.no_cuda_graph);

    std::vector<uint32_t> tokens;
    float k_ms = 0, e2e_ms = 0;

    // Warmup (builds graph on first encode).
    for (int i = 0; i < args.warmup; ++i) {
#if GBPE_HAVE_FAMILY_SP && !GBPE_GEMMA_GPU_PRETOK
        if (sp_family) pretok_ms = prepare_sp_input();
#endif
        ctx.encode(ei, tokens, &k_ms, &e2e_ms);
    }

    if (args.csv) {
        std::printf("run,n_tokens_out,n_pretokens,pretok_ms,kernel_ms,e2e_ms,full_e2e_ms\n");
    }

    for (int i = 0; i < args.runs; ++i) {
#if GBPE_HAVE_FAMILY_SP && !GBPE_GEMMA_GPU_PRETOK
        auto full_t0 = std::chrono::high_resolution_clock::now();
        if (sp_family) pretok_ms = prepare_sp_input();
#endif
        if (args.nvtx_stage_profile) {
            const std::string label =
                "cutokenize-stage-profile:replay=" + std::to_string(i);
            nvtxRangePushA(label.c_str());
        }
        ctx.encode(ei, tokens, &k_ms, &e2e_ms);
        if (args.nvtx_stage_profile) {
            nvtxRangePop();
        }
#if GBPE_HAVE_FAMILY_SP && !GBPE_GEMMA_GPU_PRETOK
        auto full_t1 = std::chrono::high_resolution_clock::now();
#endif
        float full_e2e_ms = e2e_ms;
#if GBPE_HAVE_FAMILY_SP && !GBPE_GEMMA_GPU_PRETOK
        if (sp_family) {
            full_e2e_ms = std::chrono::duration<float, std::milli>(
                full_t1 - full_t0).count();
        }
#endif
        if (args.csv) {
            std::printf("%d,%zu,%u,%.4f,%.4f,%.4f,%.4f\n",
                        i, tokens.size(), pretok_total, pretok_ms, k_ms,
                        e2e_ms, full_e2e_ms);
        } else {
            std::fprintf(stderr,
                "[run %2d] tokens=%zu  kernel=%.3f ms  e2e=%.3f ms  full_e2e=%.3f ms\n",
                i, tokens.size(), k_ms, e2e_ms, full_e2e_ms);
        }
    }
    // F4: optional BOS/EOS wrapping (mirrors HF add_special_tokens=True for
    // tokenizers whose post_processor carries a TemplateProcessing).
    if (args.add_bos || args.add_eos) {
        gbpe::EncodeOptions opts;
        opts.add_bos = args.add_bos;
        opts.add_eos = args.add_eos;
        size_t before = tokens.size();
        gbpe::apply_encode_options(hv, opts, tokens);
        std::fprintf(stderr,
            "[main] specials: add_bos=%d add_eos=%d  %zu -> %zu tokens\n",
            args.add_bos, args.add_eos, before, tokens.size());
    }

    // Write the final token output (last run's result).
    {
        std::ofstream o(args.output_path, std::ios::binary);
        if (!o) { std::fprintf(stderr, "cannot write %s\n", args.output_path.c_str()); return 1; }
        o.write(reinterpret_cast<const char*>(tokens.data()),
                static_cast<std::streamsize>(tokens.size() * sizeof(uint32_t)));
        std::fprintf(stderr, "[main] wrote %zu tokens (%zu bytes) to %s\n",
                     tokens.size(), tokens.size() * sizeof(uint32_t), args.output_path.c_str());
    }

    gbpe::free_vocab_pack(vp);
    return 0;
}
