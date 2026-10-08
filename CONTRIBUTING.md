# Contributing

Thanks for your interest. This project is small enough that there is no
formal review board — issues and PRs are read and triaged as they come in.

## Reporting bugs

File a GitHub issue. The two pieces of information that make a bug actionable:

1. **Reproducer**: the exact CLI invocation, input file (or a sample),
   tokenizer.json (or its source URL), and GPU SKU.
2. **What went wrong**: stderr output, the divergence point if the
   bit-exactness gate failed, `compute-sanitizer` output if you have it.

If the bug is "the GPU output diverges from `tiktoken` / HF tokenizers" — the
single highest-value piece of information is the byte offset of the input at
which the divergence starts and the surrounding text. The harness already
prints this when it fails.

## Submitting changes

### Before opening a PR

1. Run the preset smoke test:
   ```bash
   scripts/test-all-presets.sh
   ```
   All six presets should pass.

2. Run `compute-sanitizer --tool=synccheck` on any kernel change:
   ```bash
   compute-sanitizer --tool=synccheck \
       ./build/all/gpu_bpe_tokenize \
       --vocab data/hf_gpt2_tokenizer.json \
       --input /tmp/small.txt \
       --output /tmp/out.bin --runs 1 --warmup 0
   ```
   Must show `ERROR SUMMARY: 0 errors`. The kernel's warp-cooperation
   invariants (every `__shfl_*_sync` / `__ballot_sync` on all 32 lanes,
   mask `0xFFFFFFFF`) are not optional — see
   [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) §2.2.

3. Re-run the exactness gate:
   ```bash
   scripts/test-all-presets.sh
   ```

### Style

- C++: roughly the existing style. 4-space indent, `snake_case` for
  variables and functions, `PascalCase` for types and templates. We are not
  running a formatter; match the surrounding code.
- CUDA: `__device__ __forceinline__` for warp-cooperative helpers.
  Always specify the warp mask explicitly (`0xFFFFFFFFu`) — implicit
  full-warp masks have bitten us.
- Comments: explain the *why*, not the *what*. Especially: any place a
  reader might ask "isn't this just X?" — answer it inline. No comments that
  just restate the code (no trailing `// rotates` next to a shift instruction).

### Commit messages

Imperative mood, present tense, with a body when there's a "why" or a
notable measurement. Look at recent commits for the established shape —
they're prefixed with `feat:`, `perf:`, `notes:`, etc., and the body
includes before/after numbers for perf commits.

### Adding a tokenizer vocab

See [`docs/ADDING_A_TOKENIZER.md`](docs/ADDING_A_TOKENIZER.md). The checklist
is nine steps and gets you from "I have a new tokenizer.json" to a bit-exact
preset.

### Hot-path changes

Any change that touches:

- the merge kernel,
- the slot policies,
- the CUDA graph capture / launch path,

needs an `ncu --set full` profile before and after. Post the relevant
SOL/stall-state/occupancy lines in the PR. We have caught actual regressions
(packing optimizations that increased Stall Math Pipe Throttle) this way.

### What we accept readily

- New tokenizer vocabs with bit-exact validation.
- New CPU baselines (a `tiktoken`-rebuilt Llama-3 encoder would tighten the
  apples-to-apples story considerably).
- Smaller, sharper kernels with reproducible perf wins.

### What we push back on

- Optimizations that gain perf by losing bit-exactness. The equivalence gate
  is non-negotiable; speedup numbers without it are meaningless.
- Adding dependencies on the hot path (Python, large C++ frameworks). The
  kernel deliberately depends on no runtime other than CUDA and CUB.
- Premature generalization. The codebase has chosen specific shapes
  (warp-per-pre-token, two-bucket lengths, packed Slot64 for GPT-2) because
  they were measured to be the right shape. New abstractions should similarly
  be measured.

## Code of conduct

Be direct, be technically honest, and assume the other party is acting in
good faith. Disagreement on architecture is welcome; ad hominem is not. If
you would not say it in a face-to-face design review, do not say it in an
issue.

## License

By submitting a contribution you agree to license it under Apache 2.0, the
same as the rest of the project. See [`LICENSE`](LICENSE).
