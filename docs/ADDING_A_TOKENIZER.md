# Adding a Tokenizer

Two cases. Pick the one that matches your vocab.

## Case A — same pre-tokenizer family as an existing vocab

If your `tokenizer.json` uses a pre-tokenizer regex **identical** to GPT-2,
Llama-3, Qwen-2.5, or DeepSeek-V3 (most byte-level BPE models do), there is
**no boundary-kernel change**. Its AddedVocabulary entries also work without
new code when they are literal, have no `single_word`/strip flags, use an
identity normalizer (or the existing ASCII-stable NFC case), and no literal
start byte occurs inside another literal. Unsupported semantics fail during
vocabulary loading; add a shared matcher capability rather than embedding it
in the family regex kernel.

```bash
gpu_bpe_tokenize --vocab path/to/tokenizer.json --input text.txt --output ids.bin
```

The loader detects the family from the `pre_tokenizer` field
(`vocab.cc: detect_regex_kind`) and picks the merge-table slot layout from the
vocab size automatically. Verify bit-exactness against your reference:

```bash
scripts/test-all-presets.sh <name>
```

## Case B — a genuinely new pre-tokenizer regex

A new regex means a new **boundary kernel** (the only per-vocab GPU code — see
[ARCHITECTURE.md](ARCHITECTURE.md), "the per-vocab seam is `is_boundary[]`"). The
discipline below is mandatory because a wrong boundary corrupts output silently
on inputs the English corpus never exercises.

**1. Prove the predicate in Python first — before any CUDA.**
Write a Python reference module exporting `ORACLE_PAT` (the regex) and
`ref_pretokenize_<name>(text) -> [(start,end)]` (a *parallel* boundary predicate:
`is_boundary[k]` from a bounded codepoint window + segmented run-scans, never a
serial cursor). Diff it against the regex on a corpus plus an edge set until it matches.

Include long-run edge cases (long digit runs, CJK, CRLF) — these expose
window-collapse bugs the corpus hides.

**2. Extend the class table if you need new character classes.**
GPT-2/Llama/Qwen need only L/N/whitespace. DeepSeek added `\p{M}\p{P}\p{S}` + CJK
ranges. If yours needs more, extend `tools/gen_class_table.py`, regenerate
`generated/gbpe_class_table.h`, and confirm the exhaustive validation reports
`0 mismatches` vs the reference engine.

**3. Translate the predicate to a `pretok_k2_<name>` kernel.**
In `src/tokenizer.cu`, add the kernel guarded by `#if GBPE_HAVE_VOCAB_<NAME>`,
reusing the shared helpers in `namespace pretok` (`cls_at`, `b0_at`,
`starts_contraction`, the digit-run scan). Add the dispatch arm in
`launch_pretok` and a `PretokKind` value (`tokenizer.cuh`), set it in
`vocab.cc: build_vocab_pack`.

**4. Wire the build.** Add the vocab to `CMakeLists.txt` (`GBPE_KNOWN_VOCABS`)
and `CMakePresets.json`. Single-vocab presets must compile only your kernel.

**5. Gate it end-to-end** (not just the Python predicate):

```bash
cmake --preset <name> && cmake --build build/<name> -j
scripts/test-all-presets.sh <name>
compute-sanitizer --tool memcheck build/<name>/gpu_bpe_tokenize --vocab ... --input ... --output ... --runs 1 --warmup 0
```

Bit-exact on all size rungs + 0 sanitizer errors, or it doesn't land.
