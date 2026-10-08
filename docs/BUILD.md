# Build

## Dependencies

Tested on Ubuntu 22.04 with:

| Component           | Minimum  | Tested      |
| ------------------- | -------- | ----------- |
| NVIDIA CUDA Toolkit | 12.0     | 12.6.85     |
| NVIDIA driver       | (CUDA-compatible) | 590.48.01 |
| CMake               | 3.24     | 4.3.2 (via `pip install --user cmake`) |
| g++                 | 11       | 11.4.0      |
| utf8proc (Qwen builds only) | 2.8 | 2.11.3 |
| Python (for tests)  | 3.10     | 3.10.17     |

Install on Ubuntu:

```bash
# CUDA: follow https://developer.nvidia.com/cuda-downloads
sudo apt install -y g++ python3 python3-pip
python3 -m pip install --user cmake tiktoken tokenizers regex requests

# Only when building qwen25 or bytelevel:
sudo apt install -y libutf8proc-dev
```

`libutf8proc-dev` is required only by presets containing Qwen-2.5 (`qwen25`
and `bytelevel`); the default `all` preset and the GPT-2, Llama-3, DeepSeek-V3,
and Gemma-only builds do not link it. Qwen's normalization includes a
compatibility barrier table for the Unicode normalization additions newer than
pinned Hugging Face
`tokenizers==0.23.1`; this is tested against utf8proc Unicode data versions 15
through 17. The `tokenizers` and `tiktoken` Python packages are not required to build the
binary; they are used by the bit-exactness gate. `nlohmann/json` ships
vendored in the repo (`third_party/nlohmann/json.hpp`) — nothing to fetch.

Qwen-2.5 is built by the `qwen25` and `bytelevel` presets, which apply the
`tokenizer.json` NFC normalizer through `utf8proc`.

CUDA 12.6 and 13.1 are both verified. (13.x renamed a CCCL functor; the code
handles both — no action needed.)

## Supported GPUs

Tested SKUs (all CC ≥ 8.0):

| GPU              | Compute capability | Status         |
| ---------------- | ------------------ | -------------- |
| NVIDIA H100 NVL  | sm_90              | **primary target** |
| NVIDIA A100 80GB | sm_80              | tested, supported  |
| NVIDIA L40       | sm_89              | tested, supported (not tuned) |

The CMake default targets both sm_80 and sm_90:

```cmake
set(CMAKE_CUDA_ARCHITECTURES "80;90")
```

Override per build:

```bash
cmake -DCMAKE_CUDA_ARCHITECTURES="80;89;90" -S . -B build/local
```

## Build presets

Single user-facing knob: `GBPE_VOCABS`. Family (byte-level / SP) and slot
policy (`Slot64` / `Slot128` / both) are derived. See
[`CMakePresets.json`](../CMakePresets.json) for the full set.

### Convenience Makefile

The root `Makefile` provides a thin, discoverable interface over CMake and the
test scripts. It does not define a second build system: CMake presets and
the existing Python/shell harnesses remain the source of truth.

```bash
make help                              # list workflows and variables
make build                             # configure + build PRESET=all
make build PRESET=llama3 JOBS=16       # specialized build
make test PRESET=llama3 DEVICE=0       # build + bit-exact smoke test
make test-all DEVICE=0                 # stable preset matrix
make test-python DEVICE=0              # installed Python binding tests
```

Common variables are `PRESET` (default `all`), `VOCAB` (default `gpt2`),
`DEVICE` (default `0`), `JOBS` (default `8`), and `ARGS` for harness-specific
arguments. `make run` additionally requires `INPUT` and `OUTPUT`; set
`VOCAB_FILE` explicitly when using a tokenizer file outside the built-in map.

### One-shot via `CMakePresets.json`

```bash
cmake --preset <preset-name>
cmake --build build/<preset-name> -j
```

Available presets:

| Preset        | Vocabs                                     | Slot policy            | Binary size (current) |
| ------------- | ------------------------------------------ | ---------------------- | --------------------: |
| `all`         | GPT-2 + Llama-3 + DeepSeek-V3 + Gemma 3    | Both (runtime dispatch)|                1.15 MB |
| `bytelevel`   | GPT-2 + Llama-3 + Qwen-2.5 + DeepSeek-V3   | Both                   |                1.15 MB |
| `gpt2`        | GPT-2 only                                 | **Slot64-only**        |              **901 KB** |
| `llama3`      | Llama-3 only                               | **Slot128-only**       |                  918 KB |
| `qwen25`      | Qwen-2.5 only                              | Slot128-only           |                  918 KB |
| `deepseek_v3` | DeepSeek-V3 only                           | Slot128-only           |                  918 KB |
| `gemma3`      | Gemma 3, GPU pre-tokenizer (default)       | Slot128-only           |                       — |
| `gemma3_cpu`  | Gemma 3, original CPU `SPPreTokenizer`     | Slot128-only           |                       — |

The Gemma presets intentionally produce separate build trees and binaries, so
the CPU and GPU pre-tokenizers are selected at build time.
`gemma3_gpu` remains available as a compatibility alias for `gemma3`.

Each specialized build:

- compiles only its vocab's regex set,
- has only one kernel slot policy (drops 2 of 4 instantiations),
- removes the runtime `if (vocab.kind == Slot64)` dispatch,
- **refuses** to load an incompatible `tokenizer.json` (errors with a clear
  "reconfigure with `-DGBPE_VOCABS=...`" message).

### Manually with `-DGBPE_VOCABS`

```bash
# Single vocab
cmake -DGBPE_VOCABS=GPT2 -S . -B build/gpt2-manual
cmake --build build/gpt2-manual -j

# Multi-vocab subset
cmake -DGBPE_VOCABS="GPT2;LLAMA3" -S . -B build/two
cmake --build build/two -j

# Default set (GPT-2, Llama-3, DeepSeek-V3, Gemma 3)
cmake -DGBPE_VOCABS=ALL -S . -B build
cmake --build build -j

# Gemma uses the GPU front-end by default
cmake -DGBPE_VOCABS=GEMMA3 -S . -B build/gemma3-manual
cmake --build build/gemma3-manual -j

# Retained CPU SPPreTokenizer comparison build
cmake -DGBPE_VOCABS=GEMMA3 -DGBPE_GEMMA_GPU_PRETOK=OFF \
  -S . -B build/gemma3-cpu-manual
cmake --build build/gemma3-cpu-manual -j
```

Valid vocab names: `GPT2`, `LLAMA3`, `QWEN25`, `DEEPSEEK_V3`, `GEMMA3`,
or the literal `ALL`. Unknown names cause a CMake-time `FATAL_ERROR`.

## Verifying a build

```bash
# Run the stable preset matrix (build + smoke-test each)
scripts/test-all-presets.sh

# Or a single preset
scripts/test-all-presets.sh llama3

# Explicitly test the retained Gemma CPU comparison path
scripts/test-all-presets.sh gemma3_cpu
```

The script delegates to `tools/collect_exactness.py`. It builds each preset,
checks every compiled vocabulary against tiktoken (GPT-2) or Hugging Face
`tokenizers` (the others), runs the byte-level batch seam tests, and includes
the Python binding, batch, and targeted boundary suites. It writes a JSON
summary to the ignored `build/exactness/latest.json`; use `--out` for
a report that should be retained. The `all` binary is checked for GPT-2,
Llama-3, DeepSeek-V3, and Gemma 3.

To check just one configuration manually:

```bash
./build/<preset>/gpu_bpe_tokenize \
    --vocab data/llama3_tokenizer.json \
    --input my_text.txt \
    --output /tmp/tok.bin \
    --runs 20 --warmup 3

python3 -c "
import struct, os
from tokenizers import Tokenizer
gpu = list(struct.unpack(f'{os.path.getsize(\"/tmp/tok.bin\")//4}I',
                          open('/tmp/tok.bin','rb').read()))
ref = Tokenizer.from_file('data/llama3_tokenizer.json').encode(
    open('my_text.txt').read(),
    add_special_tokens=False).ids
print('match:', gpu == ref)"
```

## Build types

The default `CMAKE_BUILD_TYPE` is `Release`. CUDA compile flags
(`src/CMakeLists.txt` lines 96–99):

```
-O3 --use_fast_math --expt-relaxed-constexpr --extended-lambda -lineinfo
```

`-lineinfo` is on by default so `ncu` source attribution works. Drop it via
`-DCMAKE_CUDA_FLAGS_RELEASE=...` if you need a slightly smaller binary.

## Common build issues

- **`cuda_runtime.h: No such file or directory`** — `CUDAToolkit` was not
  found by CMake. Make sure `CUDA_PATH` or `CUDACXX` points at your CUDA
  install, e.g. `export CUDACXX=/usr/local/cuda/bin/nvcc`.
- **`compute-sanitizer --tool=synccheck` reports errors** — almost certainly a
  regression in the merge kernel's warp-cooperation invariants: every
  `__shfl_*_sync` / `__ballot_sync` must run on all 32 lanes with mask
  `0xFFFFFFFF`. See [`ARCHITECTURE.md`](ARCHITECTURE.md).
