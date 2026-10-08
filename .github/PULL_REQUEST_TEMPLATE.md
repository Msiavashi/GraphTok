<!--
Thanks for the PR. Quick checklist before submitting — see CONTRIBUTING.md.
-->

## Summary

What this PR does in one or two sentences.

## Motivation

Why this change. Link any issue it addresses.

## Validation

- [ ] `scripts/test-all-presets.sh` exits 0
- [ ] If kernel changed: `compute-sanitizer --tool=synccheck` reports 0 errors
- [ ] If perf-related: ncu profile attached (before + after)
- [ ] If bit-exactness affected: equivalence gate passes on 100k / 500k / 1M
- [ ] Any new tokenizer vocab follows `docs/ADDING_A_TOKENIZER.md`

## Numbers (if perf-related)

Before:
```
<paste relevant timing output, e.g. H100 1M kernel/E2E>
```

After:
```
<paste the same lines>
```
