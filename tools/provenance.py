"""Reproducible benchmark provenance and host preflight checks.

Collection scripts use this module at the start of a run and embed its result
unchanged in their JSON artifact.  It deliberately *observes* host state: a
benchmark must not silently alter CPU governors, turbo policy, GPU clocks, or
other users' processes.

``measurement_tier='final'`` is intentionally strict.  It accepts only a
clean source tree plus a manifest emitted by ``prepare_benchmark_build.py``
and a machine profile that pins the expected hardware allocation.  Ordinary
development runs use ``measurement_tier='development'`` and retain the same
metadata without turning environmental drift into a hard failure.
"""

from __future__ import annotations

import ctypes
import hashlib
import importlib
import importlib.metadata
import json
import os
import platform
import re
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


PROVENANCE_SCHEMA_VERSION = 1
RELEVANT_ENVIRONMENT_VARIABLES = (
    "CUDA_VISIBLE_DEVICES",
    "CUDA_DEVICE_ORDER",
    "CUDA_MODULE_LOADING",
    "CUDA_LAUNCH_BLOCKING",
    "CUDA_CACHE_DISABLE",
    "CUDA_CACHE_PATH",
    "TOKENIZERS_PARALLELISM",
    "RAYON_NUM_THREADS",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "KMP_AFFINITY",
    "GOMP_CPU_AFFINITY",
    "GBPE_BINARY",
    "GBPE_BPE_BLOCKS_PER_SM",
    "GBPE_FLATTEN_BLOCKS_PER_SM",
    "GBPE_FLATTEN_THREADS",
)


class PreflightError(RuntimeError):
    """Raised when a final benchmark does not meet its declared environment."""


def add_provenance_arguments(parser: Any) -> None:
    """Add the common collection-policy arguments to an argparse parser."""
    parser.add_argument(
        "--measurement-tier", choices=("development", "final"), default="development",
        help="development records deviations; final rejects a dirty/unpinned host",
    )
    parser.add_argument(
        "--preflight-profile", type=Path,
        help="JSON profile pinning the expected GPU and CPU allocation for final runs",
    )
    parser.add_argument(
        "--build-manifest", type=Path,
        help="manifest emitted by a prepared build for final runs",
    )


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    """Return the content SHA-256 of ``path`` without loading it into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _display_path(path: Path, repo_root: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(repo_root.resolve()))
    except ValueError:
        return str(resolved)


def file_metadata(path: Path, repo_root: Path) -> dict[str, Any]:
    """Identify one input artifact, including a missing-path diagnostic."""
    resolved = path.resolve()
    result: dict[str, Any] = {"path": _display_path(resolved, repo_root)}
    if not resolved.is_file():
        result.update({"present": False, "sha256": None, "bytes": None})
        return result
    result.update({
        "present": True,
        "sha256": sha256_file(resolved),
        "bytes": resolved.stat().st_size,
    })
    return result


def named_file_metadata(
    files: Mapping[str, Path] | None, repo_root: Path,
) -> dict[str, dict[str, Any]]:
    return {
        name: file_metadata(Path(path), repo_root)
        for name, path in sorted((files or {}).items())
    }


def _run(command: Sequence[str], *, cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        list(command), cwd=cwd, capture_output=True, text=True, check=False,
    )


def git_metadata(repo_root: Path) -> dict[str, Any]:
    """Capture the exact source identity before a result artifact is written."""
    commit = _run(["git", "rev-parse", "HEAD"], cwd=repo_root)
    status = _run(["git", "status", "--porcelain=v1", "--untracked-files=all"], cwd=repo_root)
    branch = _run(["git", "branch", "--show-current"], cwd=repo_root)
    if commit.returncode != 0 or status.returncode != 0:
        return {
            "available": False,
            "commit": None,
            "branch": None,
            "working_tree_clean": False,
            "working_tree_status": [],
            "error": (commit.stderr or status.stderr).strip() or "git unavailable",
        }
    entries = status.stdout.splitlines()
    return {
        "available": True,
        "commit": commit.stdout.strip(),
        "branch": branch.stdout.strip() if branch.returncode == 0 else None,
        "working_tree_clean": not entries,
        "working_tree_status": entries,
    }


def _parse_cpu_list(value: str) -> set[int]:
    result: set[int] = set()
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        if "-" in item:
            first, last = item.split("-", 1)
            result.update(range(int(first), int(last) + 1))
        else:
            result.add(int(item))
    return result


def cpu_list(values: Iterable[int]) -> str:
    """Render CPU ids as a compact, stable Linux cpulist string."""
    ordered = sorted(set(values))
    if not ordered:
        return ""
    ranges: list[str] = []
    start = end = ordered[0]
    for value in ordered[1:]:
        if value == end + 1:
            end = value
            continue
        ranges.append(str(start) if start == end else f"{start}-{end}")
        start = end = value
    ranges.append(str(start) if start == end else f"{start}-{end}")
    return ",".join(ranges)


def _read_text(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8").strip()
    except OSError:
        return None


def _cpu_model() -> str:
    info = _read_text(Path("/proc/cpuinfo")) or ""
    for line in info.splitlines():
        if line.lower().startswith("model name") and ":" in line:
            return line.split(":", 1)[1].strip()
    return platform.processor() or platform.machine() or "unknown"


def _turbo_state() -> dict[str, Any]:
    intel_no_turbo = _read_text(Path("/sys/devices/system/cpu/intel_pstate/no_turbo"))
    if intel_no_turbo is not None:
        return {
            "source": "intel_pstate/no_turbo",
            "raw": intel_no_turbo,
            "state": "disabled" if intel_no_turbo == "1" else "enabled",
        }
    cpufreq_boost = _read_text(Path("/sys/devices/system/cpu/cpufreq/boost"))
    if cpufreq_boost is not None:
        return {
            "source": "cpufreq/boost",
            "raw": cpufreq_boost,
            "state": "enabled" if cpufreq_boost == "1" else "disabled",
        }
    return {"source": None, "raw": None, "state": "unknown"}


def cpu_metadata() -> dict[str, Any]:
    try:
        affinity = set(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        affinity = set(range(os.cpu_count() or 1))
    online = _read_text(Path("/sys/devices/system/cpu/online"))
    governors: dict[str, list[int]] = {}
    for cpu in sorted(affinity):
        governor = _read_text(
            Path(f"/sys/devices/system/cpu/cpu{cpu}/cpufreq/scaling_governor")
        )
        if governor is not None:
            governors.setdefault(governor, []).append(cpu)
    return {
        "model": _cpu_model(),
        "architecture": platform.machine(),
        "logical_cores_os": os.cpu_count(),
        "online_cores": online,
        "affinity": cpu_list(affinity),
        "affinity_count": len(affinity),
        "governors": {
            governor: cpu_list(cpus) for governor, cpus in sorted(governors.items())
        },
        "turbo": _turbo_state(),
    }


def _cuda_runtime_metadata() -> dict[str, Any]:
    """Query the runtime actually loadable by this Python process, if any."""
    candidates = ("libcudart.so", "libcudart.so.12", "libcudart.so.11.0")
    for candidate in candidates:
        try:
            library = ctypes.CDLL(candidate)
            version = ctypes.c_int()
            status = library.cudaRuntimeGetVersion(ctypes.byref(version))
        except OSError:
            continue
        except AttributeError:
            continue
        if status != 0:
            return {"loaded_library": candidate, "version": "unknown", "status": status}
        numeric = version.value
        return {
            "loaded_library": candidate,
            "version": f"{numeric // 1000}.{(numeric % 1000) // 10}",
            "numeric_version": numeric,
            "status": status,
        }
    return {"loaded_library": None, "version": "unknown", "status": "unavailable"}


def cuda_metadata() -> dict[str, Any]:
    nvcc = _run(["nvcc", "--version"])
    toolkit = "unknown"
    if nvcc.returncode == 0:
        match = re.search(r"release\s+([^,\s]+)", nvcc.stdout)
        toolkit = match.group(1) if match else nvcc.stdout.strip().splitlines()[-1]
    return {"runtime": _cuda_runtime_metadata(), "toolkit": toolkit}


def gpu_metadata(device: int) -> dict[str, Any]:
    """Return physical-device state. ``nvidia-smi`` does not honor CUDA_VISIBLE_DEVICES."""
    fields = (
        "index", "uuid", "name", "driver_version", "pstate", "persistence_mode",
        "compute_mode", "clocks.current.sm", "clocks.current.memory",
        "clocks.applications.graphics", "clocks.applications.memory", "clocks.max.sm",
        "clocks.max.memory", "power.limit",
    )
    result = _run([
        "nvidia-smi", f"--id={device}", f"--query-gpu={','.join(fields)}",
        "--format=csv,noheader,nounits",
    ])
    if result.returncode != 0 or not result.stdout.strip():
        return {
            "physical_index": device,
            "available": False,
            "error": result.stderr.strip() or "nvidia-smi returned no GPU row",
        }
    values = [value.strip() for value in result.stdout.splitlines()[0].split(",")]
    raw = dict(zip(fields, values))
    return {
        "physical_index": device,
        "available": True,
        "uuid": raw.get("uuid"),
        "model": raw.get("name"),
        "driver_version": raw.get("driver_version"),
        "pstate": raw.get("pstate"),
        "persistence_mode": raw.get("persistence_mode"),
        "compute_mode": raw.get("compute_mode"),
        "clock_state_mhz": {
            "sm_current": raw.get("clocks.current.sm"),
            "memory_current": raw.get("clocks.current.memory"),
            "graphics_application": raw.get("clocks.applications.graphics"),
            "memory_application": raw.get("clocks.applications.memory"),
            "sm_max": raw.get("clocks.max.sm"),
            "memory_max": raw.get("clocks.max.memory"),
        },
        "power_limit_watts": raw.get("power.limit"),
    }


def gpu_processes(device: int) -> list[dict[str, str]]:
    result = _run([
        "nvidia-smi", f"--id={device}",
        "--query-compute-apps=pid,process_name,used_memory",
        "--format=csv,noheader,nounits",
    ])
    if result.returncode != 0:
        return [{"error": result.stderr.strip() or "nvidia-smi compute-process query failed"}]
    if not result.stdout.strip():
        return []
    records: list[dict[str, str]] = []
    for line in result.stdout.splitlines():
        values = [value.strip() for value in line.split(",")]
        if len(values) == 3:
            records.append({"pid": values[0], "process_name": values[1], "used_memory_mib": values[2]})
    return records


def package_versions() -> dict[str, str]:
    """Record the complete installed distribution set, independent of ``pip``."""
    versions: dict[str, str] = {}
    for distribution in importlib.metadata.distributions():
        name = distribution.metadata.get("Name") or distribution.metadata.get("name")
        if name:
            versions[name.lower()] = distribution.version
    return dict(sorted(versions.items()))


def extension_metadata(modules: Iterable[str], repo_root: Path) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for module_name in sorted(set(modules)):
        try:
            module = importlib.import_module(module_name)
            module_path = getattr(module, "__file__", None)
            if module_path is None:
                result[module_name] = {"loaded": False, "error": "module has no __file__"}
            else:
                result[module_name] = {"loaded": True, **file_metadata(Path(module_path), repo_root)}
        except Exception as error:  # A benchmark should record a load failure, not hide it.
            result[module_name] = {"loaded": False, "error": f"{type(error).__name__}: {error}"}
    return result


def _read_build_manifest(path: Path) -> dict[str, Any]:
    try:
        decoded = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise PreflightError(f"cannot read benchmark build manifest {path}: {error}") from error
    if decoded.get("schema_version") != 1:
        raise PreflightError(f"unsupported benchmark build manifest: {path}")
    return decoded


def _validate_build_manifest(
    manifest_path: Path,
    repo_root: Path,
    git: Mapping[str, Any],
    artifacts: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    manifest = _read_build_manifest(manifest_path)
    checks: list[dict[str, Any]] = []
    source = manifest.get("source", {})
    checks.append({
        "name": "build_manifest_commit_matches_source",
        "passed": source.get("commit") == git.get("commit"),
        "expected": source.get("commit"),
        "actual": git.get("commit"),
    })
    checks.append({
        "name": "build_manifest_was_clean",
        "passed": source.get("working_tree_clean") is True,
        "expected": True,
        "actual": source.get("working_tree_clean"),
    })
    expected_artifacts = manifest.get("artifacts", {})
    for name, actual in artifacts.items():
        expected = expected_artifacts.get(name)
        if expected is None:
            checks.append({
                "name": f"artifact_is_pinned_by_manifest:{name}",
                "passed": False,
                "expected": "artifact entry in build manifest",
                "actual": actual.get("sha256"),
            })
            continue
        checks.append({
            "name": f"artifact_hash_matches_manifest:{name}",
            "passed": expected.get("sha256") == actual.get("sha256") and actual.get("present", False),
            "expected": expected.get("sha256"),
            "actual": actual.get("sha256"),
        })
    return {
        "path": _display_path(manifest_path, repo_root),
        "sha256": sha256_file(manifest_path),
        "checks": checks,
        "all_checks_passed": all(check["passed"] for check in checks),
    }


def _load_profile(profile_path: Path) -> dict[str, Any]:
    try:
        profile = json.loads(profile_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise PreflightError(f"cannot read benchmark environment profile {profile_path}: {error}") from error
    if not isinstance(profile, dict):
        raise PreflightError(f"benchmark environment profile must be a JSON object: {profile_path}")
    return profile


def _profile_gpus(profile: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    plural = profile.get("gpus")
    if isinstance(plural, list):
        return [gpu for gpu in plural if isinstance(gpu, Mapping)]
    singular = profile.get("gpu")
    return [singular] if isinstance(singular, Mapping) else []


def _profile_checks(profile: Mapping[str, Any], *, gpus: Sequence[Mapping[str, Any]], cpu: Mapping[str, Any]) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    expected_gpus = _profile_gpus(profile)
    if expected_gpus:
        checks.append({
            "name": "expected_gpu_count",
            "passed": len(gpus) == len(expected_gpus),
            "expected": len(expected_gpus),
            "actual": len(gpus),
        })
        for index, expected_gpu in enumerate(expected_gpus):
            for key in ("uuid", "model"):
                if key in expected_gpu:
                    actual = gpus[index].get(key) if index < len(gpus) else None
                    checks.append({
                        "name": f"expected_gpu_{index}_{key}",
                        "passed": actual == expected_gpu[key],
                        "expected": expected_gpu[key],
                        "actual": actual,
                    })
    expected_cpu = profile.get("cpu", {})
    if "affinity" in expected_cpu:
        expected = cpu_list(_parse_cpu_list(str(expected_cpu["affinity"])))
        checks.append({"name": "expected_cpu_affinity", "passed": cpu.get("affinity") == expected,
                       "expected": expected, "actual": cpu.get("affinity")})
    if "governor" in expected_cpu:
        governors = cpu.get("governors", {})
        checks.append({"name": "expected_cpu_governor", "passed": set(governors) == {expected_cpu["governor"]},
                       "expected": expected_cpu["governor"], "actual": governors})
    if "turbo" in expected_cpu:
        actual_turbo = cpu.get("turbo", {}).get("state")
        checks.append({"name": "expected_turbo_state", "passed": actual_turbo == expected_cpu["turbo"],
                       "expected": expected_cpu["turbo"], "actual": actual_turbo})
    return checks


def collect_provenance(
    *,
    repo_root: Path,
    benchmark_script: Path,
    binary_artifacts: Mapping[str, Path] | None = None,
    tokenizer_assets: Mapping[str, Path] | None = None,
    corpus_files: Mapping[str, Path] | None = None,
    corpus_manifest: Path | None = None,
    extension_modules: Iterable[str] = (),
    gpu_devices: Iterable[int] = (),
    command: Sequence[str] | None = None,
    measurement_tier: str = "development",
    preflight_profile: Path | None = None,
    build_manifest: Path | None = None,
    environment_names: Iterable[str] = RELEVANT_ENVIRONMENT_VARIABLES,
) -> dict[str, Any]:
    """Collect source, binary, asset, software, and machine identity.

    The function is intended to run before timing starts.  In final tier it
    raises ``PreflightError`` on any failed prerequisite, preventing a result
    artifact from being presented as a final measurement.
    """
    if measurement_tier not in {"development", "final"}:
        raise ValueError("measurement_tier must be 'development' or 'final'")
    repo_root = repo_root.resolve()
    started_at = utc_now()
    git = git_metadata(repo_root)
    binaries = named_file_metadata(binary_artifacts, repo_root)
    extensions = extension_metadata(extension_modules, repo_root)
    artifact_hashes: dict[str, Mapping[str, Any]] = dict(binaries)
    artifact_hashes.update({f"extension:{name}": details for name, details in extensions.items()})
    devices = [int(device) for device in gpu_devices]
    gpus = [gpu_metadata(device) for device in devices]
    cpu = cpu_metadata()
    competing_processes = {str(device): gpu_processes(device) for device in devices}
    checks: list[dict[str, Any]] = [
        {"name": "source_tree_clean_before_run", "passed": bool(git.get("working_tree_clean")),
         "expected": True, "actual": git.get("working_tree_clean")},
        {"name": "selected_gpus_available", "passed": all(gpu.get("available") for gpu in gpus),
         "expected": True, "actual": [gpu.get("available") for gpu in gpus]},
        {"name": "no_competing_gpu_compute_process", "passed": not any(competing_processes.values()),
         "expected": [], "actual": competing_processes},
    ]
    profile_metadata: dict[str, Any] | None = None
    if preflight_profile is not None:
        profile_path = Path(preflight_profile).resolve()
        profile = _load_profile(profile_path)
        checks.extend(_profile_checks(profile, gpus=gpus, cpu=cpu))
        profile_metadata = {"path": _display_path(profile_path, repo_root), "sha256": sha256_file(profile_path)}
    manifest_metadata: dict[str, Any] | None = None
    if build_manifest is not None:
        manifest_metadata = _validate_build_manifest(
            Path(build_manifest).resolve(), repo_root, git, artifact_hashes,
        )
        checks.extend(manifest_metadata["checks"])
    if measurement_tier == "final":
        if preflight_profile is None:
            checks.append({"name": "final_tier_has_environment_profile", "passed": False,
                           "expected": "--preflight-profile", "actual": None})
        else:
            profile_gpus = _profile_gpus(profile)
            profile_cpu = profile.get("cpu", {})
            if devices:
                checks.append({
                    "name": "final_profile_pins_gpu_count",
                    "passed": len(profile_gpus) == len(devices),
                    "expected": len(devices),
                    "actual": len(profile_gpus),
                })
                for index, profile_gpu in enumerate(profile_gpus):
                    for key in ("uuid", "model"):
                        checks.append({
                            "name": f"final_profile_pins_gpu_{index}_{key}",
                            "passed": key in profile_gpu,
                            "expected": key,
                            "actual": profile_gpu.get(key),
                        })
            for key in ("affinity", "governor", "turbo"):
                checks.append({
                    "name": f"final_profile_pins_cpu_{key}",
                    "passed": key in profile_cpu,
                    "expected": key,
                    "actual": profile_cpu.get(key),
                })
        if build_manifest is None:
            checks.append({"name": "final_tier_has_build_manifest", "passed": False,
                           "expected": "--build-manifest", "actual": None})
    command_line = list(command) if command is not None else [sys.executable, *sys.argv]
    provenance = {
        "schema_version": PROVENANCE_SCHEMA_VERSION,
        "started_at_utc": started_at,
        "command": {"argv": command_line, "display": shlex.join(command_line), "cwd": str(Path.cwd())},
        "source": git,
        "artifacts": {
            "benchmark_script": file_metadata(benchmark_script, repo_root),
            "binaries": binaries,
            "python_extensions": extensions,
            "tokenizer_assets": named_file_metadata(tokenizer_assets, repo_root),
            "corpus_manifest": file_metadata(corpus_manifest, repo_root) if corpus_manifest else None,
            "corpus_files": named_file_metadata(corpus_files, repo_root),
            "build_manifest": manifest_metadata,
        },
        "hardware": {"gpus": gpus, "cpu": cpu},
        "software": {
            "python_executable": sys.executable,
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "cuda": cuda_metadata(),
            "packages": package_versions(),
        },
        "environment": {name: os.environ.get(name) for name in sorted(set(environment_names))},
        "preflight": {
            "measurement_tier": measurement_tier,
            "profile": profile_metadata,
            "competing_gpu_processes_before_run": competing_processes,
            "checks": checks,
            "passed": all(check["passed"] for check in checks),
        },
    }
    if measurement_tier == "final" and not provenance["preflight"]["passed"]:
        failed = ", ".join(check["name"] for check in checks if not check["passed"])
        raise PreflightError(f"final benchmark preflight failed: {failed}")
    return provenance


def finish_provenance(provenance: dict[str, Any]) -> None:
    """Append end time and a second GPU-process snapshot after collection."""
    provenance["finished_at_utc"] = utc_now()
    devices = [gpu["physical_index"] for gpu in provenance.get("hardware", {}).get("gpus", [])]
    provenance["preflight"]["competing_gpu_processes_after_run"] = {
        str(device): gpu_processes(device) for device in devices
    }
