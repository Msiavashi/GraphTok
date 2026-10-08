"""NFC normalization exactness for the Qwen-2.5 (NFC-normalizer) family.

Both branches of ``normalize_for_vocab`` are pinned here: the already-NFC
quick check must not skip inputs that NFC would change, and the full path
must canonically reorder and compose exactly as HuggingFace ``tokenizers``.
"""
from __future__ import annotations

from pathlib import Path

import pytest

import gpu_bpe_tokenizer as gbpe

REPO = Path(__file__).resolve().parent.parent
VOCAB = REPO / "data" / "vocabs" / "qwen25.json"
MULTILINGUAL = REPO / "data" / "corpus" / "corpus_chat_multilingual_100k_tokens.txt"


@pytest.fixture(scope="module")
def qwen():
    if not VOCAB.is_file():
        pytest.skip("qwen25 tokenizer asset is not present")
    try:
        return gbpe.Tokenizer(str(VOCAB))
    except RuntimeError as error:
        pytest.skip(f"qwen25 is not compiled into this extension: {error}")


@pytest.fixture(scope="module")
def reference():
    from tokenizers import Tokenizer as HFTok

    hf = HFTok.from_file(str(VOCAB))
    return lambda text: hf.encode(text, add_special_tokens=False).ids


CASES = {
    # quick-check path: already NFC, must be returned unchanged
    "precomposed_latin": "café naïve óíáú Ångström",
    "cjk_cyrillic_arabic": "日本語 Русский لا",
    "compat_not_nfc_relevant": "ﬁ ำ ½ Ω",
    # full path: NFC changes the bytes
    "nfd_latin": "éà café näive Å",
    "reorder_ccc": "ạ́ ạ́ Ḍ̇",
    "arabic_marks": "بَّ لا",
    "hangul_jamo_composes": "각 가 가",
    "singleton_ohm_angstrom": "Ω Å ẛ̣",
    "combining_after_ascii": "é ö ù",
    # ccc=0 composition second-elements: the quick check must not treat these
    # as stable, or NFC's composition is skipped (0B47+0B56 -> 0B48, etc.)
    "indic_second_element_pairs": (
        "ୈ ো ৌ ொ ஔ "
        "ೊ ൊ ේ ဦ"
    ),
    "indic_precomposed_forms": "ୈ ো ৌ ொ ஔ ೊ ൊ ේ ဦ",
    "indic_second_element_after_ascii": "aୖ bা cီ",
    # HF barrier scalar splits normalization segments
    "barrier_cp": "xࢗy éࢗé",
    "barrier_adjacent_to_marks": "ࢗ́ ́ࢗ éࢗ́e େࢗୖ",
    "ascii": "plain ascii only text 123",
    "empty": "",
}


@pytest.mark.parametrize("name", tuple(CASES), ids=str)
def test_nfc_cases_match_reference(qwen, reference, name: str) -> None:
    text = CASES[name]
    assert qwen.encode(text) == reference(text), f"[qwen25] {name} diverged"
    assert qwen.encode_numpy(text).tolist() == reference(text)


def test_multilingual_corpus_matches_reference(qwen, reference) -> None:
    if not MULTILINGUAL.is_file():
        pytest.skip("multilingual corpus asset is not present")
    text = MULTILINGUAL.read_text(encoding="utf-8")
    assert qwen.encode_numpy(text).tolist() == reference(text)
