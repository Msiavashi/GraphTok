#!/usr/bin/env python3
"""Collect the native and Python bit-exactness gates into one JSON report.

The corpus gate verifies every complete output sequence produced by the native
CLI. This orchestrator adds the targeted Python/batch cases and records both
successes and deliberate skips with enough provenance to audit later.

Examples:

    uv run --extra test python tools/collect_exactness.py --device 2

    uv run --extra test python tools/collect_exactness.py --preset all --no-build \\
      --measurement-tier final --preflight-profile /secure/h100.json \\
      --build-manifest build/all/benchmark-build-manifest.json \\
      --out /secure/results/exactness-h100.json
"""

from __future__ import annotations

import argparse
import importlib
import os
import subprocess
import sys
import tempfile
import time
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

try:
    from .provenance import (
        add_provenance_arguments,
        collect_provenance,
        file_metadata,
        finish_provenance,
        git_metadata,
    )
    from .result_json import format_result_json
except ImportError:  # Direct execution from tools/*.py.
    from provenance import (
        add_provenance_arguments,
        collect_provenance,
        file_metadata,
        finish_provenance,
        git_metadata,
    )
    from result_json import format_result_json


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PRESETS = ("gpt2", "llama3", "qwen25", "deepseek_v3", "gemma3", "bytelevel", "all")
PRESET_VOCABS = {
    "gpt2": ("gpt2",),
    "llama3": ("llama3",),
    "qwen25": ("qwen25",),
    "deepseek_v3": ("deepseek_v3",),
    "gemma3": ("gemma3",),
    "gemma3_cpu": ("gemma3",),
    "gemma3_gpu": ("gemma3",),
    "bytelevel": ("gpt2", "llama3", "qwen25", "deepseek_v3"),
    # This must remain explicit: the all preset excludes Qwen's host NFC path,
    # but must test all four configurations that it *does* compile.
    "all": ("gpt2", "llama3", "deepseek_v3", "gemma3"),
}
BYTE_LEVEL_VOCABS = frozenset({"gpt2", "llama3", "qwen25", "deepseek_v3"})
PAPER_CONFIGURATIONS = ("gpt2", "llama3", "deepseek_v3", "gemma3")
VOCAB_FILES = {
    "gpt2": REPO_ROOT / "data" / "hf_gpt2_tokenizer.json",
    "llama3": REPO_ROOT / "data" / "llama3_tokenizer.json",
    "qwen25": REPO_ROOT / "data" / "vocabs" / "qwen25.json",
    "deepseek_v3": REPO_ROOT / "data" / "vocabs" / "deepseek_v3.json",
    "gemma3": REPO_ROOT / "data" / "vocabs" / "gemma3.json",
}
EXTENSION_MODULE = "gpu_bpe_tokenizer._gpu_bpe_tokenizer"
PYTHON_TEST_PATHS = (
    "tests/test_python_bindings.py",
    "tests/test_encode_batch.py",
    "tests/test_exactness_targeted.py",
    # CPU route (host encoder and gigatoken) vs GPU route vs reference.
    "tests/test_host_fast_path.py",
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("presets", nargs="*", metavar="PRESET",
                        help="CMake presets to build/check (default: stable full matrix)")
    parser.add_argument("--preset", dest="preset_options", action="append", default=[],
                        help="add one CMake preset; may be repeated")
    parser.add_argument("--device", type=int, default=int(os.environ.get("GBPE_EXACTNESS_DEVICE", "2")),
                        help="physical CUDA GPU for native checks (default: 2)")
    parser.add_argument("--jobs", type=int, default=8, help="CMake build jobs (default: 8)")
    parser.add_argument("--no-build", action="store_true",
                        help="require existing native binaries instead of configuring/building")
    parser.add_argument("--native-only", action="store_true",
                        help="omit Python extension, binding, batch, and targeted suites")
    parser.add_argument("--skip-python-install", action="store_true",
                        help="do not force-reinstall the extension before Python tests")
    parser.add_argument(
        "--out", type=Path,
        default=REPO_ROOT / "build" / "exactness" / "latest.json",
        help="JSON report (default: ignored build/exactness/latest.json)",
    )
    add_provenance_arguments(parser)
    return parser.parse_args(argv)


def uv_python_command(*args: str) -> list[str]:
    """Run Python in the prepared lockfile-backed project environment.

    The extension is rebuilt immediately before the suites.  ``--no-sync``
    prevents a nested ``uv run`` from restoring the prior lockfile artifact
    over that freshly installed shared object.
    """
    return ["uv", "run", "--no-sync", "--extra", "test", "python", *args]


def uv_pip_command(*args: str) -> list[str]:
    """Install into the same uv environment without assuming pip is present."""
    return ["uv", "pip", "install", "--python", sys.executable, *args]


def command_result(command: list[str], *, environment: dict[str, str] | None = None) -> dict[str, Any]:
    start = time.monotonic()
    completed = subprocess.run(command, cwd=REPO_ROOT, text=True, capture_output=True,
                               check=False, env=environment)
    return {
        "argv": command,
        "returncode": completed.returncode,
        "duration_seconds": round(time.monotonic() - start, 3),
        "stdout_tail": completed.stdout[-8000:],
        "stderr_tail": completed.stderr[-8000:],
    }


def command_status(result: dict[str, Any]) -> str:
    return "passed" if result["returncode"] == 0 else "failed"


def concise_source(source: dict[str, Any]) -> dict[str, Any]:
    return {key: source.get(key) for key in ("commit", "branch", "working_tree_clean")}


def asset_hashes(vocabs: Iterable[str]) -> dict[str, str | None]:
    return {vocab: file_metadata(VOCAB_FILES[vocab], REPO_ROOT).get("sha256") for vocab in vocabs}


def test_record(
    *, suite: str, name: str, status: str, source: dict[str, Any],
    configuration: dict[str, Any], vocabs: Iterable[str], binary: Path | None = None,
    detail: str | None = None, command: dict[str, Any] | None = None,
) -> dict[str, Any]:
    binary_metadata = file_metadata(binary, REPO_ROOT) if binary is not None else None
    record: dict[str, Any] = {
        "suite": suite,
        "name": name,
        "status": status,
        "skip_reason": detail if status == "skipped" else None,
        "detail": detail if status != "skipped" else None,
        "configuration": configuration,
        "source_revision": concise_source(source),
        "binary_sha256": binary_metadata.get("sha256") if binary_metadata else None,
        "binary_path": binary_metadata.get("path") if binary_metadata else None,
        "tokenizer_asset_hashes": asset_hashes(vocabs),
    }
    if command is not None:
        record["command"] = command
    return record


def selected_presets(args: argparse.Namespace) -> list[str]:
    presets = [*args.preset_options, *args.presets] or list(DEFAULT_PRESETS)
    unknown = [preset for preset in presets if preset not in PRESET_VOCABS]
    if unknown:
        raise SystemExit(f"unknown preset(s): {', '.join(unknown)}")
    if args.jobs < 1:
        raise SystemExit("--jobs must be positive")
    return presets


def run_native_matrix(
    args: argparse.Namespace, presets: list[str], source: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Path]]:
    records: list[dict[str, Any]] = []
    binaries: dict[str, Path] = {}
    native_environment = os.environ.copy()
    native_environment["CUDA_VISIBLE_DEVICES"] = str(args.device)
    for preset in presets:
        print(f"[native] preset={preset}", flush=True)
        config = {"preset": preset, "device": args.device, "runner": "native-cli"}
        binary = REPO_ROOT / "build" / preset / "gpu_bpe_tokenize"
        if not args.no_build:
            configure = command_result(["cmake", "--preset", preset])
            records.append(test_record(
                suite="cmake", name=f"configure[{preset}]", status=command_status(configure),
                source=source, configuration=config, vocabs=PRESET_VOCABS[preset],
                detail=None if configure["returncode"] == 0 else configure["stderr_tail"], command=configure,
            ))
            if configure["returncode"] != 0:
                continue
            build = command_result(["cmake", "--build", f"build/{preset}", "--parallel", str(args.jobs)])
            records.append(test_record(
                suite="cmake", name=f"build[{preset}]", status=command_status(build), source=source,
                configuration=config, vocabs=PRESET_VOCABS[preset], binary=binary,
                detail=None if build["returncode"] == 0 else build["stderr_tail"], command=build,
            ))
            if build["returncode"] != 0:
                continue
        if not binary.is_file():
            records.append(test_record(
                suite="native", name=f"binary-present[{preset}]", status="error", source=source,
                configuration=config, vocabs=PRESET_VOCABS[preset], binary=binary,
                detail=f"required binary is absent: {binary}",
            ))
            continue
        binaries[preset] = binary

        for vocab in PRESET_VOCABS[preset]:
            print(f"[native] preset={preset} vocab={vocab}", flush=True)
            vocab_config = {**config, "vocab": vocab}
            smoke = REPO_ROOT / "build" / preset / "test_encode_batch_smoke"
            if vocab in BYTE_LEVEL_VOCABS:
                if smoke.is_file():
                    smoke_run = command_result(
                        [str(smoke), str(VOCAB_FILES[vocab])], environment=native_environment,
                    )
                    records.append(test_record(
                        suite="native-batch-smoke", name=f"{preset}/{vocab}",
                        status=command_status(smoke_run), source=source, configuration=vocab_config,
                        vocabs=(vocab,), binary=smoke,
                        detail=None if smoke_run["returncode"] == 0 else smoke_run["stderr_tail"],
                        command=smoke_run,
                    ))
                else:
                    records.append(test_record(
                        suite="native-batch-smoke", name=f"{preset}/{vocab}", status="error",
                        source=source, configuration=vocab_config, vocabs=(vocab,), binary=binary,
                        detail=f"expected batch smoke executable is absent: {smoke}",
                    ))
            else:
                records.append(test_record(
                    suite="native-batch-smoke", name=f"{preset}/{vocab}", status="skipped",
                    source=source, configuration=vocab_config, vocabs=(vocab,), binary=binary,
                    detail="Gemma/SP intentionally rejects the byte-level encode_batch API",
                ))

    return records, binaries


def installed_extension_path() -> Path | None:
    try:
        module = importlib.import_module(EXTENSION_MODULE)
    except Exception:
        return None
    path = getattr(module, "__file__", None)
    return Path(path).resolve() if path else None


def junit_records(
    junit_path: Path, *, source: dict[str, Any], device: int, extension: Path | None,
) -> list[dict[str, Any]]:
    try:
        root = ET.parse(junit_path).getroot()
    except (OSError, ET.ParseError) as error:
        return [test_record(
            suite="python-api", name="junit-report", status="error", source=source,
            configuration={"device": device, "runner": "uv-python-extension"}, vocabs=VOCAB_FILES,
            binary=extension, detail=f"cannot parse pytest JUnit report: {error}",
        )]
    records: list[dict[str, Any]] = []
    for case in root.iter("testcase"):
        status, detail = "passed", None
        for tag, mapped in (("failure", "failed"), ("error", "error"), ("skipped", "skipped")):
            child = case.find(tag)
            if child is not None:
                status = mapped
                detail = child.attrib.get("message") or (child.text or "").strip() or None
                break
        records.append(test_record(
            suite="python-api",
            name=f"{case.attrib.get('classname', '')}::{case.attrib.get('name', 'unnamed-test')}",
            status=status, source=source,
            configuration={"device": device, "runner": "uv-python-extension", "extension": "default-all"},
            vocabs=VOCAB_FILES, binary=extension, detail=detail,
        ))
    return records


def run_python_suites(
    args: argparse.Namespace, source: dict[str, Any],
) -> tuple[list[dict[str, Any]], Path | None]:
    records: list[dict[str, Any]] = []
    environment = os.environ.copy()
    # Python bindings select logical CUDA 0 after this physical-device remap.
    environment["CUDA_VISIBLE_DEVICES"] = str(args.device)
    if not args.skip_python_install:
        install = command_result(
            uv_pip_command("--no-deps", "--reinstall", "."),
            environment=environment,
        )
        records.append(test_record(
            suite="python-extension", name="install", status=command_status(install), source=source,
            configuration={"device": args.device, "runner": "uv"}, vocabs=VOCAB_FILES,
            detail=None if install["returncode"] == 0 else install["stderr_tail"], command=install,
        ))
        if install["returncode"] != 0:
            return records, None
    extension = installed_extension_path()
    if extension is None:
        records.append(test_record(
            suite="python-extension", name="import", status="error", source=source,
            configuration={"device": args.device, "runner": "uv-python-extension"}, vocabs=VOCAB_FILES,
            detail=f"cannot import {EXTENSION_MODULE} after installation",
        ))
        return records, None

    with tempfile.TemporaryDirectory(prefix="cutokenize_exactness_") as directory:
        junit_path = Path(directory) / "python-api.xml"
        pytest_run = command_result(
            uv_python_command("-m", "pytest", "-q", f"--junitxml={junit_path}", *PYTHON_TEST_PATHS),
            environment=environment,
        )
        records.extend(junit_records(junit_path, source=source, device=args.device, extension=extension))
        if pytest_run["returncode"] != 0 and not any(
            record["status"] in {"failed", "error"} for record in records
        ):
            records.append(test_record(
                suite="python-api", name="pytest-process", status="error", source=source,
                configuration={"device": args.device, "runner": "uv-python-extension"},
                vocabs=VOCAB_FILES, binary=extension,
                detail=pytest_run["stderr_tail"] or pytest_run["stdout_tail"], command=pytest_run,
            ))
    return records, extension


def targeted_suite_complete(records: Iterable[dict[str, Any]]) -> bool:
    """Require passes for targeted cases applicable to all four paper configs."""
    targeted = [record for record in records if "test_exactness_targeted" in record["name"]]
    if not targeted:
        return False
    for vocab in PAPER_CONFIGURATIONS:
        relevant = [record for record in targeted if vocab in record["name"]]
        if not relevant or any(record["status"] != "passed" for record in relevant):
            return False
    return True


def artifact_binary_names(binaries: dict[str, Path]) -> dict[str, Path]:
    """Match prepare_benchmark_build.py's manifest key for an all-only run."""
    if list(binaries) == ["all"]:
        return {"cutokenize_cli": binaries["all"]}
    return {f"cutokenize_cli:{preset}": binary for preset, binary in binaries.items()}


def final_preflight(
    args: argparse.Namespace, presets: list[str],
) -> dict[str, Any] | None:
    """Reject a non-reproducible final run before it can consume GPU time."""
    if args.measurement_tier != "final":
        return None
    if not args.no_build:
        raise SystemExit(
            "final exactness runs require --no-build; run prepare_benchmark_build.py first"
        )
    if presets != ["all"]:
        raise SystemExit("final exactness runs require exactly --preset all")
    binary = REPO_ROOT / "build" / "all" / "gpu_bpe_tokenize"
    extension = None if args.native_only else installed_extension_path()
    return collect_provenance(
        repo_root=REPO_ROOT,
        benchmark_script=Path(__file__),
        binary_artifacts={"cutokenize_cli": binary},
        tokenizer_assets=VOCAB_FILES,
        extension_modules=(EXTENSION_MODULE,) if extension is not None else (),
        gpu_devices=(args.device,),
        measurement_tier=args.measurement_tier,
        preflight_profile=args.preflight_profile,
        build_manifest=args.build_manifest,
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    presets = selected_presets(args)
    source = git_metadata(REPO_ROOT)
    provenance = final_preflight(args, presets)
    records: list[dict[str, Any]] = []
    native_records, binaries = run_native_matrix(args, presets, source)
    records.extend(native_records)
    extension: Path | None = None
    if not args.native_only:
        python_records, extension = run_python_suites(args, source)
        records.extend(python_records)

    # Final mode is intended for ``--preset all --no-build`` after the clean,
    # hash-pinned prepare_benchmark_build.py workflow. Development mode still
    # records all host deviations and dirty state in the JSON.
    if provenance is None:
        provenance = collect_provenance(
            repo_root=REPO_ROOT, benchmark_script=Path(__file__),
            binary_artifacts=artifact_binary_names(binaries), tokenizer_assets=VOCAB_FILES,
            extension_modules=(EXTENSION_MODULE,) if extension is not None else (),
            gpu_devices=(args.device,), measurement_tier=args.measurement_tier,
            preflight_profile=args.preflight_profile, build_manifest=args.build_manifest,
        )
    finish_provenance(provenance)

    counts = Counter(record["status"] for record in records)
    failures = counts["failed"] + counts["error"]
    payload = {
        "schema_version": 1,
        "kind": "cutokenize-exactness-gate",
        "configuration": {
            "presets": presets, "device": args.device, "native_only": args.native_only,
            "no_build": args.no_build,
            "python_tests": list(PYTHON_TEST_PATHS) if not args.native_only else [],
            "paper_configurations_in_all_binary": list(PAPER_CONFIGURATIONS),
        },
        "summary": {
            "status_counts": dict(sorted(counts.items())),
            "all_required_tests_passed": failures == 0,
            "targeted_suite_complete_for_paper_configurations": targeted_suite_complete(records),
            "publication_exactness_gate_passed": (
                failures == 0 and targeted_suite_complete(records)
            ),
            "note": (
                "Skipped tests are never converted into passes. Qwen Python-extension skips are "
                "expected with the default all wheel; native qwen25 remains independently checked."
            ),
        },
        "tests": records,
        "provenance": provenance,
    }
    output = args.out if args.out.is_absolute() else REPO_ROOT / args.out
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(format_result_json(payload) + "\n", encoding="utf-8")
    print(f"wrote {output}")
    print("status: " + ", ".join(f"{name}={count}" for name, count in sorted(counts.items())))
    print("targeted paper suite: " + (
        "COMPLETE" if payload["summary"]["targeted_suite_complete_for_paper_configurations"] else "INCOMPLETE"
    ))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
