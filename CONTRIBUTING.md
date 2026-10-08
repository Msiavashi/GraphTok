# Contributing to GraphTok

Issues and pull requests are welcome at https://github.com/Msiavashi/graphtok. By participating you agree to the [Code of Conduct](CODE_OF_CONDUCT.md).

## Branch model

- `main` is protected and holds releases only.
- `develop` is the integration branch.
- Fork the repository, branch from `develop` (`feature/<name>`, `fix/<name>`), and open the pull request against `develop`.

See [docs/DEVELOPMENT.md](docs/DEVELOPMENT.md) for setup, builds and tests.

## Commit messages

- Short imperative subject line (about 72 characters), for example `Fix overflow path for long pre-tokens`.
- A body that explains why the change is needed when it is not obvious.
- Reference issues with `Fixes #123` where applicable.
- One logical change per commit.

## Pull request checklist

- [ ] The affected CMake presets compile (`cmake --preset <preset> && cmake --build build/<preset> -j`).
- [ ] `pytest tests/` passes on a GPU.
- [ ] Token IDs stay identical to the reference tokenizers: `scripts/test-all-presets.sh` exits 0.
- [ ] Kernel changes: `compute-sanitizer --tool=synccheck` reports 0 errors.
- [ ] Documentation is updated.
- [ ] `CHANGELOG.md` has an entry.

Changes to the pre-tokenizer, merge kernel, slot layouts or CPU route must keep token IDs identical to `tiktoken` (GPT-2) and Hugging Face `tokenizers`.

## Reporting bugs

Open an issue with the bug report template and include:

- The exact CLI invocation or Python call.
- The input, or a small sample that reproduces it.
- The `tokenizer.json` or its source.
- The build preset, GPU model, CUDA toolkit and driver versions.
- For a token-ID mismatch: the reference output, the GraphTok output, and the first differing position with the surrounding text.

Report security issues privately as described in [SECURITY.md](SECURITY.md).

## Adding a tokenizer

Follow [docs/ADDING_A_TOKENIZER.md](docs/ADDING_A_TOKENIZER.md). A new tokenizer needs a preset, a vocabulary download entry and an exactness check against its reference tokenizer.

## Code style

- C++/CUDA: 4-space indent, `snake_case` functions and variables, `PascalCase` types. Match the surrounding code.
- Warp-level operations run on all 32 lanes with an explicit `0xFFFFFFFFu` mask (see [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)).
- The captured graph must stay free of allocation and host synchronization.
- Python: PEP 8, 4-space indent.
- Comments explain why, not what.

## License

Contributions are licensed under the Apache License 2.0, the same as the project. See [LICENSE](LICENSE). A `Signed-off-by` line (`git commit -s`) is welcome.
