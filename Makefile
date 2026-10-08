# Convenience interface for common developer workflows. CMake remains the
# source of truth; every target here is a thin wrapper around a documented
# command.

SHELL := /bin/bash
.DEFAULT_GOAL := help

PRESET ?= all
VOCAB ?= gpt2
DEVICE ?= 0
JOBS ?= 8
PYTHON ?= python3
UV ?= uv
ARGS ?=

BUILD_DIR = build/$(PRESET)
BINARY = $(BUILD_DIR)/gpu_bpe_tokenize

VOCAB_FILE_gpt2 = data/hf_gpt2_tokenizer.json
VOCAB_FILE_llama3 = data/llama3_tokenizer.json
VOCAB_FILE_qwen25 = data/vocabs/qwen25.json
VOCAB_FILE_deepseek_v3 = data/vocabs/deepseek_v3.json
VOCAB_FILE_gemma3 = data/vocabs/gemma3.json
VOCAB_FILE ?= $(VOCAB_FILE_$(VOCAB))

.PHONY: help configure build clean run test test-all test-python test-exactness \
	python-install sanitize

help: ## Show this help
	@echo "cuTokenize developer commands"
	@echo
	@echo "Usage: make <target> [PRESET=all] [VOCAB=gpt2] [DEVICE=0] [ARGS='...']"
	@echo
	@echo "Core workflows:"
	@awk 'BEGIN {FS = ":.*## "} /^[a-zA-Z0-9_-]+:.*## / {printf "  %-20s %s\n", $$1, $$2}' $(MAKEFILE_LIST)
	@echo
	@echo "Examples:"
	@echo "  make build PRESET=llama3"
	@echo "  make test PRESET=llama3 DEVICE=0"
	@echo "  make run INPUT=prompt.txt OUTPUT=/tmp/tokens.bin VOCAB=llama3"

configure: ## Configure the selected CMake preset
	cmake --preset $(PRESET)

build: configure ## Build the selected preset
	cmake --build $(BUILD_DIR) -j $(JOBS)

clean: ## Clean the selected preset's build products
	cmake --build $(BUILD_DIR) --target clean

run: build ## Tokenize INPUT to OUTPUT with the selected vocab (optional ARGS)
	@test -n "$(INPUT)" || { echo "INPUT is required (for example INPUT=prompt.txt)" >&2; exit 2; }
	@test -n "$(OUTPUT)" || { echo "OUTPUT is required (for example OUTPUT=/tmp/tokens.bin)" >&2; exit 2; }
	@test -n "$(VOCAB_FILE)" || { echo "Unknown VOCAB='$(VOCAB)'; set VOCAB_FILE explicitly" >&2; exit 2; }
	CUDA_VISIBLE_DEVICES=$(DEVICE) "$(BINARY)" --vocab "$(VOCAB_FILE)" \
		--input "$(INPUT)" --output "$(OUTPUT)" $(ARGS)

test: ## Build and bit-exact smoke-test PRESET
	GBPE_EXACTNESS_DEVICE=$(DEVICE) scripts/test-all-presets.sh $(PRESET)

test-all: ## Build and bit-exact smoke-test the stable preset matrix
	GBPE_EXACTNESS_DEVICE=$(DEVICE) scripts/test-all-presets.sh

python-install: ## Install the Python extension and test dependencies
	$(PYTHON) -m pip install '.[test]'

test-python: ## Run reference-backed Python binding, batch, and targeted exactness tests
	CUDA_VISIBLE_DEVICES=$(DEVICE) $(UV) run --extra test python -m pytest \
		tests/test_python_bindings.py tests/test_encode_batch.py tests/test_exactness_targeted.py $(ARGS)

test-exactness: ## Build/check the complete native + Python exactness gate and write its JSON report
	GBPE_EXACTNESS_DEVICE=$(DEVICE) scripts/test-all-presets.sh $(ARGS)

sanitize: build ## Run CUDA synccheck; set INPUT for a custom small input
	@test -n "$(VOCAB_FILE)" || { echo "Unknown VOCAB='$(VOCAB)'; set VOCAB_FILE explicitly" >&2; exit 2; }
	@test -n "$(INPUT)" || { echo "INPUT is required and should be small" >&2; exit 2; }
	CUDA_VISIBLE_DEVICES=$(DEVICE) compute-sanitizer --tool=synccheck "$(BINARY)" \
		--vocab "$(VOCAB_FILE)" --input "$(INPUT)" --output /tmp/cutokenize-sanitize.bin \
		--runs 1 --warmup 0 $(ARGS)
