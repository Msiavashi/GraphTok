# GraphTok backend for NVIDIA Dynamo 1.3.0

`graphtok-dynamo-1.3.0.patch` adds a `graphtok` tokenizer backend to the
Dynamo frontend. Prompt encoding runs on the GPU through `libgtok.so` (the C
ABI of this repository); decoding stays with Hugging Face `tokenizers`.

It adds `lib/tokenizers/src/graphtok.rs` (backend, loaded with `libloading`),
`lib/tokenizers/tests/graphtok_dispatch.rs`, wires the backend into
`lib/llm/src/model_card.rs`, and adds `graphtok` to the frontend's
`--tokenizer-backend` choices.

## Build libgtok.so

From the root of this repository:

```bash
cmake --preset all && cmake --build build/all -j --target gtok
# -> build/all/libgtok.so
```

## Apply and build Dynamo

```bash
git clone https://github.com/ai-dynamo/dynamo.git && cd dynamo
git checkout bae0051            # Dynamo 1.3.0
git apply /path/to/graphtok/integrations/dynamo/graphtok-dynamo-1.3.0.patch
cargo build --release           # or rebuild the Python bindings: cd lib/bindings/python && maturin develop --release
cargo test -p dynamo-tokenizers --test graphtok_dispatch   # needs a GPU and DYN_GRAPHTOK_LIB
```

## Enable

```bash
export DYN_GRAPHTOK_LIB=/path/to/graphtok/build/all/libgtok.so
export DYN_TOKENIZER=graphtok          # or: python -m dynamo.frontend --tokenizer-backend graphtok
export DYN_GRAPHTOK_DISPATCH=1         # load-adaptive dispatcher (GPU batching + CPU route)
python -m dynamo.frontend ...
```

`libgtok.so` is located through `DYN_GRAPHTOK_LIB`; when unset, `libgtok.so`
is resolved by the dynamic loader (`LD_LIBRARY_PATH`, rpath, system paths).

Other knobs:

| Variable | Default | Meaning |
|---|---|---|
| `DYN_GRAPHTOK_CONTEXTS` | 4 | GPU contexts in the pool (without the dispatcher) |
| `DYN_GRAPHTOK_MAX_BYTES` | 2 MiB | largest input a context is sized for; larger inputs use Hugging Face |
| `DYN_GRAPHTOK_DISPATCH_CPU_WORKERS` | 1 | CPU engines used by the dispatcher for small inputs |

On construction the backend encodes a probe containing chat-template special
tokens with both GraphTok and Hugging Face and refuses to load if the ids
differ (the frontend then falls back to Hugging Face).
