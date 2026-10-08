# Changelog

## [0.3.0] - 2026-10 - First public release

- GPU BPE encoder captured as one CUDA graph per power-of-two input-size
  class: GPU pre-tokenization, warp-cooperative BPE merge with an overflow
  path for long pre-tokens, and assembly.
- Tokenizers: GPT-2, Llama 3, Qwen 2.5 / Qwen 3, DeepSeek-V3, Gemma 3.
  Token IDs match `tiktoken` (GPT-2) and Hugging Face `tokenizers`.
- Batched encode of many documents in one graph replay, and GPU decode.
- CPU route for small inputs (gigatoken or the built-in host encoder) and a
  load-adaptive dispatcher that batches concurrent requests.
- Interfaces: CLI `gpu_bpe_tokenize`, Python module `gpu_bpe_tokenizer`
  (`Tokenizer`, `Dispatcher`), C library `libgtok`.
- Serving integrations: vLLM general plugin and an NVIDIA Dynamo 1.3.0 patch.
