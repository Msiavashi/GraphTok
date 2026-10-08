#!/usr/bin/env python3
"""Download the tokenizer.json assets used by the build and tests."""

from __future__ import annotations

import argparse
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
VOCAB_DIR = DATA_DIR / "vocabs"
DOWNLOAD_RETRIES = 3
RETRY_DELAY_SECONDS = 2

VOCABS = (
    (
        "GPT-2",
        "openai-community/gpt2",
        "GPT2_REVISION",
        DATA_DIR / "hf_gpt2_tokenizer.json",
    ),
    (
        "Llama 3",
        "meta-llama/Meta-Llama-3-8B-Instruct",
        "LLAMA3_REVISION",
        DATA_DIR / "llama3_tokenizer.json",
    ),
    (
        "Qwen 2.5",
        "Qwen/Qwen2.5-7B-Instruct",
        "QWEN25_REVISION",
        VOCAB_DIR / "qwen25.json",
    ),
    (
        "DeepSeek V3",
        "deepseek-ai/DeepSeek-V3",
        "DEEPSEEK_V3_REVISION",
        VOCAB_DIR / "deepseek_v3.json",
    ),
    (
        "Gemma 3",
        "google/gemma-3-1b-it",
        "GEMMA3_REVISION",
        VOCAB_DIR / "gemma3.json",
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download tokenizer vocabularies from Hugging Face.",
        epilog=(
            "Set HF_TOKEN to a Hugging Face token with access to Meta Llama 3 "
            "and Gemma 3."
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="download and replace files that are already present",
    )
    return parser.parse_args()


def hugging_face_token() -> str:
    if token := os.environ.get("HF_TOKEN"):
        return token

    default_home = Path.home() / ".cache" / "huggingface"
    token_path = Path(os.environ.get("HF_HOME", default_home)) / "token"
    try:
        token = token_path.read_text(encoding="utf-8").splitlines()[0]
    except (FileNotFoundError, IndexError, OSError):
        token = ""

    if token:
        return token

    print(
        "Meta Llama 3 and Gemma 3 are gated upstream. Accept their licenses, "
        "then run:\n  HF_TOKEN=hf_... uv run scripts/download-vocabs.py",
        file=sys.stderr,
    )
    raise SystemExit(2)


def download(
    name: str,
    repository: str,
    revision: str,
    destination: Path,
    token: str,
    *,
    force: bool,
) -> None:
    if not force and destination.is_file() and destination.stat().st_size > 0:
        print(f"Already present: {destination.relative_to(REPO_ROOT)}")
        return

    temporary = destination.with_name(f"{destination.name}.part")
    url = (
        f"https://huggingface.co/{repository}/resolve/{revision}/"
        "tokenizer.json?download=true"
    )
    request = urllib.request.Request(
        url,
        headers={"Authorization": f"Bearer {token}"},
    )

    print(f"Downloading {name:<12} -> {destination.relative_to(REPO_ROOT)}")
    for attempt in range(DOWNLOAD_RETRIES + 1):
        try:
            with (
                urllib.request.urlopen(request) as response,
                temporary.open("wb") as output,
            ):
                while chunk := response.read(1024 * 1024):
                    output.write(chunk)
            if temporary.stat().st_size == 0:
                raise ValueError(f"Downloaded an empty tokenizer file for {name}")
            temporary.replace(destination)
            return
        except (OSError, urllib.error.URLError, ValueError) as error:
            temporary.unlink(missing_ok=True)
            if attempt == DOWNLOAD_RETRIES:
                raise RuntimeError(
                    f"Failed to download {name} from {repository} at revision "
                    f"{revision}: {error}"
                ) from error
            time.sleep(RETRY_DELAY_SECONDS)


def update_llama_symlink() -> None:
    link = VOCAB_DIR / "llama3_tokenizer.json"
    link.unlink(missing_ok=True)
    link.symlink_to(Path("..") / "llama3_tokenizer.json")


def main() -> None:
    args = parse_args()
    token = hugging_face_token()
    VOCAB_DIR.mkdir(parents=True, exist_ok=True)

    try:
        for name, repository, revision_variable, destination in VOCABS:
            download(
                name,
                repository,
                os.environ.get(revision_variable, "main"),
                destination,
                token,
                force=args.force,
            )
        update_llama_symlink()
    except RuntimeError as error:
        print(error, file=sys.stderr)
        raise SystemExit(1) from error

    print("Downloaded all tokenizer vocabularies.")


if __name__ == "__main__":
    main()
