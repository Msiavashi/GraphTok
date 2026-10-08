#!/usr/bin/env bash
# Build the stable CMake matrix and run every exactness gate through the
# machine-readable collector. The all binary checks GPT-2, Llama-3,
# DeepSeek-V3, and Gemma 3; it is not reduced to a GPT-2-only probe.
#
# Usage: scripts/test-all-presets.sh [PRESET ...]
#        (no args = gpt2 llama3 qwen25 deepseek_v3 gemma3 bytelevel all)
#
# Set GBPE_EXACTNESS_DEVICE to choose the physical CUDA GPU (default: 2). The
# report defaults to the ignored build/exactness/latest.json; pass
# --out through to tools/collect_exactness.py to retain a named
# artifact.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

: "${GBPE_EXACTNESS_DEVICE:=2}"
export GBPE_EXACTNESS_DEVICE

exec uv run --extra test python tools/collect_exactness.py \
    --device "$GBPE_EXACTNESS_DEVICE" "$@"
