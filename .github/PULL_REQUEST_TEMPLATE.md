<!-- Pull requests target `develop`. See CONTRIBUTING.md. -->

## Summary

What this PR does in one or two sentences.

## Motivation

Why this change. Link any issue it addresses (`Fixes #123`).

## Checklist

- [ ] Targets `develop`
- [ ] Affected presets compile (`cmake --preset <preset> && cmake --build build/<preset> -j`)
- [ ] `pytest tests/` passes on a GPU
- [ ] `scripts/test-all-presets.sh` exits 0 (token IDs identical to the reference tokenizers)
- [ ] Kernel changes: `compute-sanitizer --tool=synccheck` reports 0 errors
- [ ] New tokenizers follow `docs/ADDING_A_TOKENIZER.md`
- [ ] Documentation updated
- [ ] `CHANGELOG.md` entry added
