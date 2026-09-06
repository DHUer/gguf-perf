"""Generate the ICASSP companion results appendix from frozen result files.

This script is deliberately separate from the analysis pipeline. It treats
the regenerated CSV exports as the row-level source of truth and fails loudly
if host/model/depth grids, protocol fields, or aggregate summaries disagree.

Run from anywhere in the repository with::

    python paper/icassp2027/generate_supplement.py
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RESULTS = ROOT / "results"

EXPECTED_HOSTS = {
    "lun-mac": {
        "label": "MacBook Pro M4 Max",
        "code": "MB",
        "colour": "#377eb8",
        "marker": "o",
        "linestyle": "-",
        "hatch": "",
        "measurements": "measurements_lun-mac.csv",
        "calibration": "calibration_lun-mac.json",
        "environment": "env_lun-mac.json",
        "provenance": {},
        "require_manifest_coverage": True,
    },
    "mac-studio-m4-max": {
        "label": "Mac Studio M4 Max",
        "code": "MS",
        "colour": "#4daf4a",
        "marker": "^",
        "linestyle": "-.",
        "hatch": "xx",
        "measurements": "measurements_mac-studio-m4-max.csv",
        "calibration": "calibration_mac-studio-m4-max.json",
        "environment": "env_mac-studio-m4-max.json",
        "provenance": {
            "system profile": "system_profile_mac-studio-m4-max.txt",
            "operating system": "os_mac-studio-m4-max.txt",
            "source commit": "source_commit_mac-studio-m4-max.txt",
            "measurement-source SHA-256": "measurement_source_tree_mac-studio-m4-max.sha256",
            "llama.cpp package": "llama_cpp_package_mac-studio-m4-max.txt",
            "llama-bench SHA-256": "llama_bench_mac-studio-m4-max.sha256",
            "model sources": "model_sources_mac-studio-m4-max.json",
            "model integrity": "model_integrity_mac-studio-m4-max.json",
            "Python packages": "python_packages_mac-studio-m4-max.txt",
            "Tectonic package": "tectonic_package_mac-studio-m4-max.txt",
            "Tectonic SHA-256": "tectonic_mac-studio-m4-max.sha256",
        },
        "require_manifest_coverage": True,
    },
    "rtx5080": {
        "label": "RTX 5080",
        "code": "RTX",
        "colour": "#e6550d",
        "marker": "s",
        "linestyle": "--",
        "hatch": "///",
        "measurements": "measurements_rtx5080.csv",
        "calibration": "calibration_rtx5080.json",
        "environment": "env_rtx5080.json",
        "provenance": {},
        "require_manifest_coverage": False,
    },
}

HOST_LABELS = {
    host: str(spec["label"]) for host, spec in EXPECTED_HOSTS.items()
}

DECLARED_DEPTHS = (0, 4096, 16384)
DECLARED_GRID = frozenset(
    (phase, depth)
    for phase in ("decode", "prefill")
    for depth in DECLARED_DEPTHS
)
DECLARED_ENV_PROTOCOL = {
    "flash_attn": "on (pinned)",
    "cache_type_k": "f16",
    "cache_type_v": "f16",
    "repetitions": 5,
    "settle_seconds": 45.0,
    "warmup": "llama-bench default (enabled)",
}
DECLARED_NUMERIC_FIELDS = {
    "n_gpu_layers": 99,
    "repetitions": 5,
    "settle_s": 45.0,
    "max_cv_pct": 3.0,
    "n_batch": 2048,
    "n_ubatch": 512,
}
DECLARED_TEXT_FIELDS = {
    "type_k": "f16",
    "type_v": "f16",
}
QUALITY_FIELDS = ("load_before", "load_after", "attempts", "kept_cv_pct")


def host_label(value: object) -> str:
    """Human-readable label while preserving unknown host identifiers."""
    value = str(value)
    return HOST_LABELS.get(value, value)


def host_code(value: object) -> str:
    """Compact, unambiguous table code for a known host."""
    value = str(value)
    spec = EXPECTED_HOSTS.get(value)
    return str(spec["code"]) if spec else value


def english_join(values: list[str]) -> str:
    """Join display labels without assuming a fixed host count."""
    if not values:
        return ""
    if len(values) == 1:
        return values[0]
    if len(values) == 2:
        return " and ".join(values)
    return ", ".join(values[:-1]) + ", and " + values[-1]


def _numeric_column(frame: pd.DataFrame, name: str) -> pd.Series:
    if name not in frame.columns:
        return pd.Series(np.nan, index=frame.index, dtype=float)
    return pd.to_numeric(frame[name], errors="coerce")


def _text_column(frame: pd.DataFrame, name: str) -> pd.Series:
    if name not in frame.columns:
        return pd.Series("", index=frame.index, dtype="string")
    return frame[name].fillna("").astype(str).str.strip()


def environment_protocol_errors(environment: dict, host: str) -> list[str]:
    """Validate the frozen environment declaration for one paper host."""
    errors: list[str] = []
    if str(environment.get("host", "")) != host:
        errors.append(f"environment host is {environment.get('host')!r}")
    protocol = environment.get("protocol")
    if not isinstance(protocol, dict):
        return errors + ["environment protocol is missing or not an object"]
    if set(protocol) != set(DECLARED_ENV_PROTOCOL):
        errors.append(
            "environment protocol fields differ: "
            f"expected={sorted(DECLARED_ENV_PROTOCOL)}, found={sorted(protocol)}"
        )
    for field, expected in DECLARED_ENV_PROTOCOL.items():
        actual = protocol.get(field)
        if isinstance(expected, (int, float)):
            try:
                matches = float(actual) == float(expected)
            except (TypeError, ValueError):
                matches = False
        else:
            matches = str(actual) == expected
        if not matches:
            errors.append(
                f"environment protocol {field}={actual!r}, expected {expected!r}"
            )
    try:
        if int(environment.get("threads")) <= 0:
            errors.append("environment threads must be positive")
    except (TypeError, ValueError):
        errors.append("environment threads are missing or malformed")
    for field in ("platform", "git_commit", "llama_bench_version"):
        if not str(environment.get(field, "")).strip():
            errors.append(f"environment {field} is missing")
    return errors


def _row_identity_mask(
    frame: pd.DataFrame, environment: dict, host: str
) -> pd.Series:
    mask = _text_column(frame, "host").eq(host)
    for column, env_field in (
        ("platform", "platform"),
        ("git_commit", "git_commit"),
        ("llama_bench_version", "llama_bench_version"),
    ):
        mask &= _text_column(frame, column).eq(str(environment.get(env_field, "")))
    try:
        threads = int(environment["threads"])
    except (KeyError, TypeError, ValueError):
        threads = -1
    mask &= _numeric_column(frame, "n_threads").eq(threads)
    return mask.fillna(False).astype(bool)


def _selected_protocol_mask(
    frame: pd.DataFrame, environment: dict, host: str
) -> pd.Series:
    mask = _row_identity_mask(frame, environment, host)
    numeric = {
        field: _numeric_column(frame, field)
        for field in (*DECLARED_NUMERIC_FIELDS, *QUALITY_FIELDS)
    }
    for field, expected in DECLARED_NUMERIC_FIELDS.items():
        mask &= numeric[field].eq(expected)
    for field, expected in DECLARED_TEXT_FIELDS.items():
        mask &= _text_column(frame, field).str.lower().eq(expected)
    mask &= _text_column(frame, "flash_attn").str.lower().isin(
        {"1", "1.0", "true", "on", "yes"}
    )
    for field in QUALITY_FIELDS:
        mask &= np.isfinite(numeric[field])
    mask &= numeric["attempts"].ge(1)
    mask &= numeric["attempts"].eq(numeric["attempts"].round())
    mask &= numeric["kept_cv_pct"].ge(0)

    prompt = _numeric_column(frame, "n_prompt")
    generated = _numeric_column(frame, "n_gen")
    prefill = prompt.eq(512) & generated.eq(0)
    decode = prompt.eq(0) & generated.eq(128)
    mask &= prefill | decode
    mask &= _numeric_column(frame, "n_depth").isin(DECLARED_DEPTHS)

    def samples_match(value: object) -> bool:
        try:
            samples = json.loads(str(value))
            return (
                isinstance(samples, list)
                and len(samples) == DECLARED_NUMERIC_FIELDS["repetitions"]
                and all(math.isfinite(float(sample)) for sample in samples)
            )
        except (TypeError, ValueError, json.JSONDecodeError):
            return False

    if "samples_ts" not in frame.columns:
        mask &= False
    else:
        mask &= frame["samples_ts"].map(samples_match)
    return mask.fillna(False).astype(bool)


def _explicit_error_mask(
    frame: pd.DataFrame, environment: dict, host: str
) -> pd.Series:
    mask = _row_identity_mask(frame, environment, host)
    mask &= _text_column(frame, "error").ne("")
    for field in ("n_gpu_layers", "repetitions", "settle_s", "max_cv_pct"):
        mask &= _numeric_column(frame, field).eq(DECLARED_NUMERIC_FIELDS[field])
    mask &= _numeric_column(frame, "n_prompt").eq(512)
    mask &= _numeric_column(frame, "n_gen").eq(0)
    mask &= _numeric_column(frame, "n_depth").eq(DECLARED_DEPTHS[0])
    return mask.fillna(False).astype(bool)


def campaign_validation_errors(
    selected: pd.DataFrame,
    raw: pd.DataFrame,
    manifest: dict,
    environment: dict,
    host: str,
    *,
    require_manifest_coverage: bool,
) -> list[str]:
    """Return reasons a host is not a complete, exact paper campaign.

    A required model must contribute the six phase--depth cells selected by the
    analysis, or one exact-protocol error record and no selected partial grid.
    RTX deliberately measured a manifest subset, so callers can validate every
    attempted RTX model without requiring all 23 manifest entries.
    """
    errors = environment_protocol_errors(environment, host)
    if "host" not in selected.columns or "host" not in raw.columns:
        return errors + ["measurement rows have no host column"]

    raw_hosts = set(_text_column(raw, "host")) - {""}
    if raw_hosts != {host}:
        errors.append(
            f"measurement file records host(s) {sorted(raw_hosts)}, expected {host}"
        )
    selected_host = selected[_text_column(selected, "host").eq(host)].copy()
    raw_host = raw[_text_column(raw, "host").eq(host)].copy()
    manifest_models = set(map(str, manifest))
    attempted_models = set(_text_column(raw_host, "model_file")) - {""}
    selected_models = set(_text_column(selected_host, "model_file")) - {""}
    unknown = sorted((attempted_models | selected_models) - manifest_models)
    if unknown:
        errors.append("models absent from manifest: " + ", ".join(unknown))
    if not attempted_models:
        errors.append("no models were attempted")

    protocol_mask = _selected_protocol_mask(selected_host, environment, host)
    if len(selected_host) and not protocol_mask.all():
        errors.append(
            f"{int((~protocol_mask).sum())} selected row(s) violate the declared "
            "protocol or environment identity"
        )
    valid_selected = selected_host[protocol_mask]
    valid_errors = raw_host[_explicit_error_mask(raw_host, environment, host)]
    explicit_failures = set(_text_column(valid_errors, "model_file")) - {""}

    required_models = manifest_models if require_manifest_coverage else attempted_models
    missing_attempts = sorted(required_models - attempted_models)
    if missing_attempts:
        errors.append(
            f"{len(missing_attempts)} manifest model(s) were not attempted: "
            + ", ".join(missing_attempts)
        )

    incomplete: list[str] = []
    for model in sorted(required_models & manifest_models, key=str.casefold):
        rows = valid_selected[_text_column(valid_selected, "model_file").eq(model)]
        prompt = _numeric_column(rows, "n_prompt")
        generated = _numeric_column(rows, "n_gen")
        phases = np.where(generated.gt(0), "decode", "prefill")
        cells = set(zip(phases, _numeric_column(rows, "n_depth").astype(int)))
        if len(rows) == len(DECLARED_GRID) and cells == DECLARED_GRID:
            continue
        if not len(rows) and model in explicit_failures:
            continue
        missing_cells = sorted(DECLARED_GRID - cells)
        detail = f"{model} has {len(rows)}/6 selected rows"
        if missing_cells:
            detail += f" (missing {missing_cells})"
        if model in explicit_failures:
            detail += " plus an error record"
        incomplete.append(detail)
    if incomplete:
        errors.append("incomplete model grids: " + "; ".join(incomplete))
    return errors


def model_source_provenance_errors(
    document: dict, manifest: dict, manifest_sha256: str
) -> list[str]:
    """Validate frozen repository revisions and per-file LFS SHA-256 values."""
    errors: list[str] = []
    if document.get("schema_version") != 1:
        errors.append("model-source schema_version is not 1")
    if document.get("frozen_manifest_sha256") != manifest_sha256:
        errors.append("model-source manifest hash does not match model_manifest.json")
    revisions = document.get("repository_revisions")
    file_hashes = document.get("file_lfs_sha256")
    if not isinstance(revisions, dict) or not isinstance(file_hashes, dict):
        return errors + ["model-source revision/hash mappings are missing"]
    expected_repositories = {
        str(entry.get("repo_id", "")) for entry in manifest.values()
    } - {""}
    if set(revisions) != expected_repositories:
        errors.append("model-source repository coverage differs from the manifest")
    if set(file_hashes) != set(map(str, manifest)):
        errors.append("model-source file-hash coverage differs from the manifest")
    if any(not re.fullmatch(r"[0-9a-f]{40}", str(value))
           for value in revisions.values()):
        errors.append("model-source repository revision is not a 40-digit SHA-1")
    if any(not re.fullmatch(r"[0-9a-f]{64}", str(value))
           for value in file_hashes.values()):
        errors.append("model-source file identifier is not a 64-digit SHA-256")
    return errors


def model_integrity_provenance_errors(
    document: dict,
    manifest: dict,
    model_sources: dict,
    manifest_sha256: str,
    host: str,
) -> list[str]:
    """Validate the persisted local-GGUF integrity attestation."""
    if not isinstance(document, dict):
        return ["model-integrity report is not a JSON object"]

    errors: list[str] = []
    if document.get("schema_version") != 1:
        errors.append("model-integrity schema_version is not 1")
    if document.get("status") != "pass":
        errors.append("model-integrity status is not pass")
    if document.get("host") != host:
        errors.append(
            f"model-integrity host is {document.get('host')!r}, expected {host!r}"
        )
    if document.get("manifest_sha256") != manifest_sha256:
        errors.append("model-integrity manifest hash does not match model_manifest.json")
    if document.get("errors") != []:
        errors.append("model-integrity errors list is not empty")

    if not isinstance(manifest, dict):
        return errors + ["model-integrity manifest is not a JSON object"]
    try:
        expected_bytes = sum(
            int(entry["size_bytes"]) for entry in manifest.values()
        )
    except (KeyError, TypeError, ValueError):
        return errors + ["model-integrity manifest has invalid size_bytes entries"]
    expected_count = len(manifest)
    for field, expected in (
        ("expected_count", expected_count),
        ("verified_count", expected_count),
        ("expected_bytes", expected_bytes),
        ("verified_bytes", expected_bytes),
    ):
        if document.get(field) != expected:
            errors.append(
                f"model-integrity {field} is {document.get(field)!r}, "
                f"expected {expected}"
            )

    files = document.get("files")
    if not isinstance(files, dict):
        return errors + ["model-integrity files mapping is missing"]
    expected_files = set(map(str, manifest))
    actual_files = set(map(str, files))
    missing = sorted(expected_files - actual_files, key=str.casefold)
    unexpected = sorted(actual_files - expected_files, key=str.casefold)
    if missing:
        errors.append("model-integrity report omits file(s): " + ", ".join(missing))
    if unexpected:
        errors.append(
            "model-integrity report contains unexpected file(s): "
            + ", ".join(unexpected)
        )

    source_hashes = (
        model_sources.get("file_lfs_sha256")
        if isinstance(model_sources, dict) else None
    )
    if not isinstance(source_hashes, dict):
        return errors + ["model-integrity model-source hash mapping is missing"]

    for name in sorted(expected_files & actual_files, key=str.casefold):
        item = files[name]
        if not isinstance(item, dict):
            errors.append(f"model-integrity entry for {name} is not an object")
            continue
        expected_size = manifest[name].get("size_bytes")
        if item.get("size_bytes") != expected_size:
            errors.append(
                f"model-integrity size mismatch for {name}: "
                f"report={item.get('size_bytes')!r}, manifest={expected_size!r}"
            )
        source_hash = source_hashes.get(name)
        for field in ("expected_sha256", "actual_sha256"):
            if item.get(field) != source_hash:
                errors.append(
                    f"model-integrity {field} mismatch for {name}"
                )
        for field in ("sha256_ok", "header_ok", "metadata_ok"):
            if item.get(field) is not True:
                errors.append(
                    f"model-integrity {field} is not true for {name}"
                )
        if not isinstance(item.get("metadata"), dict):
            errors.append(f"model-integrity metadata is missing for {name}")
        if item.get("metadata_mismatches") != {}:
            errors.append(
                f"model-integrity metadata mismatches are not empty for {name}"
            )
    return errors


def binary_provenance_errors(
    package_text: str, checksum_text: str, package_key: str, label: str
) -> list[str]:
    """Cross-check a recorded binary digest against its package metadata."""
    package_values = {}
    for line in package_text.splitlines():
        if ":" in line:
            key, value = line.split(":", 1)
            package_values[key.strip()] = value.strip()
    checksum = checksum_text.strip().split(maxsplit=1)[0] if checksum_text.strip() else ""
    declared = package_values.get(package_key, "")
    errors: list[str] = []
    if not re.fullmatch(r"[0-9a-f]{64}", checksum):
        errors.append(f"{label} checksum file does not begin with a SHA-256")
    if not re.fullmatch(r"[0-9a-f]{64}", declared):
        errors.append(f"{label} package metadata lacks {package_key}")
    elif checksum and declared != checksum:
        errors.append(f"{label} package/checksum digests disagree")
    return errors


def measurement_source_provenance_errors(
    checksum_text: str, root: Path = ROOT
) -> list[str]:
    """Validate the exact source inventory used by the measurement pipeline.

    The inventory is deliberately repository-relative and limited to direct
    ``llmperf/*.py`` modules plus ``requirements.txt``. Computing the expected
    set from the current tree makes a newly added measurement module mandatory;
    rejecting extra entries keeps the checksum file from including itself or
    unrelated paper sources.
    """
    errors: list[str] = []
    root = root.resolve()
    source_dir = root / "llmperf"
    expected = {"requirements.txt"}
    if source_dir.is_dir():
        try:
            source_dir.resolve().relative_to(root)
        except (OSError, RuntimeError, ValueError):
            errors.append(
                "measurement source directory resolves outside the repository"
            )
        else:
            expected.update(
                path.relative_to(root).as_posix()
                for path in source_dir.glob("*.py")
            )
    else:
        errors.append("measurement source directory is missing: llmperf")

    listed: dict[str, str] = {}
    resolved_paths: dict[str, Path] = {}
    lines = checksum_text.splitlines()
    if not lines:
        errors.append("measurement-source checksum inventory is empty")

    for line_number, line in enumerate(lines, 1):
        match = re.fullmatch(r"([0-9a-f]{64})  (.+)", line)
        if not match:
            errors.append(
                f"measurement-source checksum line {line_number} is malformed"
            )
            continue
        declared, relative_name = match.groups()
        # Require a canonical POSIX repository-relative spelling. This rejects
        # traversal, absolute paths, duplicate separators, and Windows paths on
        # every platform before resolving or opening anything.
        parts = relative_name.split("/")
        if (
            relative_name != relative_name.strip()
            or relative_name.startswith("/")
            or "\\" in relative_name
            or re.match(r"^[A-Za-z]:", relative_name)
            or any(ord(character) < 32 or ord(character) == 127
                   for character in relative_name)
            or any(part in {"", ".", ".."} for part in parts)
        ):
            errors.append(
                f"measurement-source checksum path is unsafe: {relative_name!r}"
            )
            continue
        candidate = root.joinpath(*parts)
        try:
            resolved = candidate.resolve()
            resolved.relative_to(root)
        except (OSError, RuntimeError, ValueError):
            errors.append(
                "measurement-source checksum path resolves outside the repository: "
                f"{relative_name!r}"
            )
            continue
        if relative_name == (
            "results/measurement_source_tree_mac-studio-m4-max.sha256"
        ):
            errors.append(
                "measurement-source inventory must not checksum itself: "
                + relative_name
            )
            continue
        if relative_name in listed:
            errors.append(
                f"duplicate measurement-source checksum entry: {relative_name}"
            )
            continue
        listed[relative_name] = declared
        resolved_paths[relative_name] = resolved

    missing_entries = sorted(expected - set(listed), key=str.casefold)
    unexpected_entries = sorted(set(listed) - expected, key=str.casefold)
    if missing_entries:
        errors.append(
            "measurement-source inventory omits required file(s): "
            + ", ".join(missing_entries)
        )
    if unexpected_entries:
        errors.append(
            "measurement-source inventory contains unexpected file(s): "
            + ", ".join(unexpected_entries)
        )

    # Never open an unexpected path supplied by the inventory. Resolve and hash
    # only members of the exact expected set after coverage has been checked.
    for relative_name in sorted(expected & set(listed), key=str.casefold):
        resolved = resolved_paths[relative_name]
        try:
            is_file = resolved.is_file()
        except OSError:
            is_file = False
        if not is_file:
            errors.append(
                f"listed measurement source is missing: {relative_name}"
            )
            continue
        digest = hashlib.sha256()
        try:
            with resolved.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(chunk)
        except OSError as exc:
            errors.append(
                f"listed measurement source cannot be read: {relative_name}: {exc}"
            )
            continue
        actual = digest.hexdigest()
        declared = listed[relative_name]
        if actual != declared:
            errors.append(
                f"measurement-source checksum mismatch for {relative_name}: "
                f"declared={declared}, actual={actual}"
            )
    for relative_name in sorted(expected, key=str.casefold):
        try:
            required = (root / relative_name).resolve()
            required.relative_to(root)
            is_file = required.is_file()
        except (OSError, RuntimeError, ValueError):
            is_file = False
        if not is_file:
            errors.append(
                f"required measurement source is missing: {relative_name}"
            )
    return errors


def tex(value: object) -> str:
    """Escape ordinary text for LaTeX table cells."""
    s = str(value)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(ch, ch) for ch in s)


def path_cell(value: object) -> str:
    """Render a filename or repository ID with safe punctuation and breaks."""
    return r"\nolinkurl{" + str(value).replace("%", r"\%") + "}"


def hash_cell(value: object) -> str:
    """Render a long hexadecimal identifier with explicit line-break points."""
    value = str(value)
    chunks = [value[i : i + 8] for i in range(0, len(value), 8)]
    return r"\texttt{" + r"\allowbreak{}".join(chunks) + "}"


def predictor_label(value: object) -> str:
    """Use fit terminology: calibration measurements cancel in B1/B2/P1/P2."""
    value = str(value)
    value = value.replace("uncalibrated", "unfitted")
    value = value.replace("calibrated", "fitted")
    return tex(value)


def nfmt(value: object) -> str:
    return f"{int(value):,}"


def ffmt(value: object, digits: int = 2) -> str:
    x = float(value)
    if not math.isfinite(x):
        return "--"
    return f"{x:.{digits}f}"


def bool_value(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes"}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def git_head() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "unavailable"


def add_longtable(
    out: list[str],
    caption: str,
    label: str,
    colspec: str,
    headers: list[str],
    rows: list[list[str]],
    landscape: bool = True,
    lead: list[str] | None = None,
) -> None:
    if landscape:
        out.append(r"\begin{landscape}")
    if lead:
        out.extend(lead)
    out.extend(
        [
            r"\begingroup",
            r"\small",
            r"\setlength{\tabcolsep}{3.5pt}",
            r"\renewcommand{\arraystretch}{1.08}",
            rf"\begin{{longtable}}{{{colspec}}}",
            rf"\caption{{{caption}}}\label{{{label}}}\\",
            r"\toprule",
            " & ".join(headers) + r" \\",
            r"\midrule",
            r"\endfirsthead",
            rf"\caption[]{{{caption} (continued)}}\\",
            r"\toprule",
            " & ".join(headers) + r" \\",
            r"\midrule",
            r"\endhead",
            r"\midrule",
            rf"\multicolumn{{{len(headers)}}}{{r}}{{Continued on next page}}\\",
            r"\endfoot",
            r"\bottomrule",
            r"\endlastfoot",
        ]
    )
    out.extend(" & ".join(row) + r" \\" for row in rows)
    out.extend([r"\end{longtable}", r"\endgroup"])
    if landscape:
        out.append(r"\end{landscape}")


def group_predictions(
    frame: pd.DataFrame, bytes_col: str, prediction_bytes_col: str | None = None
) -> np.ndarray:
    train = frame[frame["split"] == "train"]
    eta = (
        train["avg_ts"] * train[bytes_col] / train["bw"]
    ).groupby([train["host"], train["quant"]]).median()
    host_median = eta.groupby(level=0).median()
    overall = float(eta.median())

    def lookup(host: str, quant: str) -> float:
        if (host, quant) in eta.index:
            return float(eta.loc[(host, quant)])
        if host in host_median.index:
            return float(host_median.loc[host])
        return overall

    denominator = frame[prediction_bytes_col or bytes_col].to_numpy(float)
    fitted = np.array([lookup(h, q) for h, q in zip(frame.host, frame["quant"])])
    return fitted * frame["bw"].to_numpy(float) / denominator


def ape(pred: np.ndarray, measured: np.ndarray) -> np.ndarray:
    return np.abs(pred - measured) / np.abs(measured) * 100.0


def make_summary_figure(dec: pd.DataFrame, prefill: pd.DataFrame) -> None:
    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "pdf.fonttype": 42,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.65), constrained_layout=True)

    ladder = dec[
        dec["model_file"].str.startswith("Qwen3.8-27B-")
        & (dec["n_depth"] == 0)
    ].copy()
    order = {
        "IQ2_XXS": 0,
        "Q2_K": 1,
        "IQ3_XXS": 2,
        "Q3_K": 3,
        "Q4_K": 4,
        "Q5_K": 5,
        "Q6_K": 6,
        "Q8_0": 7,
    }
    ladder["order"] = ladder["quant"].map(order)
    ladder = ladder.dropna(subset=["order"]).sort_values(["host", "order"])
    ladder["effective_gb_s"] = (
        ladder["avg_ts"] * ladder["bytes_active"] / 1e9
    )
    quants = [
        q for q, _ in sorted(order.items(), key=lambda item: item[1])
        if q in set(ladder["quant"])
    ]
    xpos = {q: i for i, q in enumerate(quants)}
    colours = {
        host: str(spec["colour"]) for host, spec in EXPECTED_HOSTS.items()
    }
    markers = {
        host: str(spec["marker"]) for host, spec in EXPECTED_HOSTS.items()
    }
    linestyles = {
        host: str(spec["linestyle"]) for host, spec in EXPECTED_HOSTS.items()
    }
    hatches = {
        host: str(spec["hatch"]) for host, spec in EXPECTED_HOSTS.items()
    }
    for host, group in ladder.groupby("host", sort=True):
        group = group.sort_values("order")
        x = group["quant"].map(xpos).to_numpy(float)
        axes[0].plot(
            x,
            group["effective_gb_s"],
            color=colours.get(host, "#555555"),
            marker=markers.get(host, "D"),
            linestyle=linestyles.get(host, ":"),
            lw=1.3,
            ms=4.5,
            label=host_label(host),
        )
    axes[0].set_xticks(np.arange(len(quants)), quants, rotation=35, ha="right")
    axes[0].set_ylabel("effective streamed GB/s")
    axes[0].set_title("(a) Cross-system Qwen3.8-27B ladder", loc="left")
    axes[0].legend(loc="best", frameon=False)

    test = prefill[prefill["split"] == "test"]
    depths = sorted(int(x) for x in test["n_depth"].unique())
    bx = np.arange(len(depths))
    hosts = sorted(test["host"].unique())
    width = 0.72 / max(len(hosts), 1)
    for index, host in enumerate(hosts):
        group = test[test["host"] == host]
        values = [group.loc[group.n_depth == d, "ape_P2"].mean() for d in depths]
        offset = (index - (len(hosts) - 1) / 2) * width
        bars = axes[1].bar(
            bx + offset,
            values,
            width,
            color=colours.get(host, "#555555"),
            edgecolor="#222222",
            linewidth=0.6,
            hatch=hatches.get(host, ".."),
            label=host_label(host),
        )
        # Put values inside the bars so a tall label does not collide with the
        # legend in the compact two-panel rendering.
        axes[1].bar_label(
            bars, fmt="%.1f", padding=-12, fontsize=8, color="white"
        )
    axes[1].set_xticks(bx, [f"{d:,}" for d in depths])
    axes[1].set_xlabel("existing-prefix depth (tokens)")
    axes[1].set_ylabel("P2 held-out MAPE (%)")
    axes[1].set_title("(b) Prefill error is host dependent", loc="left")
    axes[1].legend(loc="upper left", frameon=False)

    fig.savefig(HERE / "supplement_summary.pdf", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def main() -> None:
    paths: dict[str, Path] = {}
    for host, spec in EXPECTED_HOSTS.items():
        paths[f"measurements ({host})"] = RESULTS / str(spec["measurements"])
        paths[f"calibration ({host})"] = RESULTS / str(spec["calibration"])
        paths[f"environment ({host})"] = RESULTS / str(spec["environment"])
        for role, filename in dict(spec["provenance"]).items():
            paths[f"{role} ({host})"] = RESULTS / str(filename)
    paths.update({
        "decode predictions": RESULTS / "predictions.csv",
        "prefill predictions": RESULTS / "predictions_prefill.csv",
        "decode aggregate error table": RESULTS / "error_table.csv",
        "decode error table by host": RESULTS / "error_table_by_host.csv",
        "prefill aggregate error table": RESULTS / "error_table_prefill.csv",
        "prefill error table by host": RESULTS / "error_table_prefill_by_host.csv",
        "manifest": RESULTS / "model_manifest.json",
        "metadata": RESULTS / "model_metadata.json",
        "repeatability (lun-mac)": RESULTS / "repeatability_lun-mac.csv",
        "archived uncontrolled comparison": (
            RESULTS / "contaminated" / "measurements_UNCONTROLLED_lun-mac.csv.bak"
        ),
        "analysis implementation": ROOT / "llmperf" / "analyze.py",
        "refinement implementation": ROOT / "llmperf" / "refine.py",
        "metadata implementation": ROOT / "llmperf" / "common.py",
        "publication-figure wrapper": HERE / "generate_main_figures.py",
        "appendix generator": HERE / "generate_supplement.py",
    })
    missing = [str(p) for p in paths.values() if not p.is_file()]
    if missing:
        raise SystemExit("missing required input(s): " + ", ".join(missing))
    empty = [str(path) for path in paths.values() if path.stat().st_size == 0]
    if empty:
        raise SystemExit("empty required input(s): " + ", ".join(empty))

    dec = pd.read_csv(paths["decode predictions"])
    pf = pd.read_csv(paths["prefill predictions"])
    err = pd.read_csv(paths["decode error table by host"])
    pf_err = pd.read_csv(paths["prefill error table by host"])
    measurements_by_host = {
        host: pd.read_csv(paths[f"measurements ({host})"])
        for host in EXPECTED_HOSTS
    }
    repeat = pd.read_csv(paths["repeatability (lun-mac)"])
    calibrations = {
        host: json.loads(
            paths[f"calibration ({host})"].read_text(encoding="utf-8")
        )
        for host in EXPECTED_HOSTS
    }
    environments = {
        host: json.loads(
            paths[f"environment ({host})"].read_text(encoding="utf-8")
        )
        for host in EXPECTED_HOSTS
    }
    manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
    metadata_doc = json.loads(paths["metadata"].read_text(encoding="utf-8"))
    metadata = metadata_doc["models"]

    studio = "mac-studio-m4-max"
    if studio in EXPECTED_HOSTS:
        source_document = json.loads(
            paths[f"model sources ({studio})"].read_text(encoding="utf-8")
        )
        integrity_document = json.loads(
            paths[f"model integrity ({studio})"].read_text(encoding="utf-8")
        )
        manifest_digest = sha256(paths["manifest"])
        provenance_errors = model_source_provenance_errors(
            source_document, manifest, manifest_digest
        )
        provenance_errors.extend(
            model_integrity_provenance_errors(
                integrity_document,
                manifest,
                source_document,
                manifest_digest,
                studio,
            )
        )
        provenance_errors.extend(
            measurement_source_provenance_errors(
                paths[f"measurement-source SHA-256 ({studio})"].read_text(
                    encoding="utf-8"
                ),
                ROOT,
            )
        )
        provenance_errors.extend(
            binary_provenance_errors(
                paths[f"llama.cpp package ({studio})"].read_text(encoding="utf-8"),
                paths[f"llama-bench SHA-256 ({studio})"].read_text(encoding="utf-8"),
                "runner_sha256",
                "llama-bench",
            )
        )
        provenance_errors.extend(
            binary_provenance_errors(
                paths[f"Tectonic package ({studio})"].read_text(encoding="utf-8"),
                paths[f"Tectonic SHA-256 ({studio})"].read_text(encoding="utf-8"),
                "binary_sha256",
                "Tectonic",
            )
        )
        if provenance_errors:
            raise SystemExit(
                "Mac Studio provenance validation failed: "
                + "; ".join(provenance_errors)
            )

    dec = dec.copy()
    pf = pf.copy()
    for frame in (dec, pf):
        frame["n_depth"] = pd.to_numeric(frame["n_depth"])
        frame["avg_ts"] = pd.to_numeric(frame["avg_ts"])
        frame["stddev_ts"] = pd.to_numeric(frame["stddev_ts"], errors="coerce")
        frame["row_cv_pct"] = frame["stddev_ts"] / frame["avg_ts"] * 100.0

    expected_hosts = set(EXPECTED_HOSTS)
    decode_hosts = set(dec["host"].astype(str).unique())
    prefill_hosts = set(pf["host"].astype(str).unique())
    if decode_hosts != prefill_hosts:
        raise SystemExit("decode and prefill exports describe different hosts")
    if decode_hosts != expected_hosts:
        raise SystemExit(
            "prediction exports do not contain the expected hosts: "
            f"expected={sorted(expected_hosts)}, found={sorted(decode_hosts)}"
        )
    if (set(measurements_by_host) != expected_hosts
            or set(calibrations) != expected_hosts
            or set(environments) != expected_hosts):
        raise SystemExit("measurement, calibration, and environment hosts disagree")
    hosts = list(EXPECTED_HOSTS)
    for host in hosts:
        raw_hosts = set(
            measurements_by_host[host]["host"].dropna().astype(str).unique()
        )
        if raw_hosts != {host}:
            raise SystemExit(
                f"measurement file for {host} records host(s) {sorted(raw_hosts)}"
            )
        for kind, document in (
            ("calibration", calibrations[host]),
            ("environment", environments[host]),
        ):
            if str(document.get("host", "")) != host:
                raise SystemExit(
                    f"{kind} file for {host} records host "
                    f"{document.get('host')!r}"
                )
        for field in ("platform", "git_commit", "cpu"):
            if calibrations[host].get(field) != environments[host].get(field):
                raise SystemExit(
                    f"calibration/environment {field} mismatch for {host}: "
                    f"{calibrations[host].get(field)!r} vs "
                    f"{environments[host].get(field)!r}"
                )

    key_columns = ["host", "model_file", "n_depth"]
    if dec.duplicated(key_columns).any() or pf.duplicated(key_columns).any():
        raise SystemExit("row exports contain duplicate host/model/depth keys")
    decode_keys = set(dec[key_columns].itertuples(index=False, name=None))
    prefill_keys = set(pf[key_columns].itertuples(index=False, name=None))
    if decode_keys != prefill_keys:
        raise SystemExit("decode and prefill exports describe different host/model grids")
    depth_sets = {
        (host, model): tuple(sorted(group["n_depth"].astype(int).unique()))
        for (host, model), group in dec.groupby(["host", "model_file"])
    }
    if not depth_sets or len(set(depth_sets.values())) != 1:
        raise SystemExit(f"non-rectangular depth grids: {depth_sets}")
    depths = next(iter(depth_sets.values()))

    if not dec["quality_protocol"].map(bool_value).all():
        raise SystemExit("decode export contains a protocol-incomplete row")
    if not pf["quality_protocol"].map(bool_value).all():
        raise SystemExit("prefill export contains a protocol-incomplete row")

    scored = sorted(dec["model_file"].unique(), key=str.casefold)
    scored_pairs = set(
        dec[["host", "model_file"]].drop_duplicates().itertuples(index=False, name=None)
    )
    for name in scored:
        if name not in metadata or name not in manifest:
            raise SystemExit(f"missing manifest/metadata entry for {name}")
    for host, name in scored_pairs:
        row_splits = set(dec.loc[(dec.host == host) & (dec.model_file == name), "split"])
        if row_splits != {manifest[name]["split"]}:
            raise SystemExit(f"split mismatch for {host}/{name}: {row_splits}")

    loaded_by_host: dict[str, set[str]] = {}
    failed_by_host: dict[str, list[str]] = {}
    probes_by_host: dict[str, list[str]] = {}
    for host in hosts:
        raw = measurements_by_host[host]
        ok = raw["error"].fillna("").astype(str).str.strip().eq("")
        attempted = set(raw["model_file"].dropna().astype(str))
        loaded_by_host[host] = set(raw.loc[ok, "model_file"].astype(str))
        failed_by_host[host] = sorted(
            attempted - loaded_by_host[host], key=str.casefold
        )
        probes_by_host[host] = list(
            (calibrations[host].get("llm_ref") or {}).get("probe_models") or []
        )
        scored_here = {name for h, name in scored_pairs if h == host}
        if not scored_here.issubset(loaded_by_host[host]):
            missing_scored = sorted(scored_here - loaded_by_host[host])
            raise SystemExit(f"scored rows absent from raw {host} data: {missing_scored}")

    identified = sorted(
        set(scored).union(
            *(set(names) for names in probes_by_host.values())
        ),
        key=str.casefold,
    )
    ids = {name: f"M{i:02d}" for i, name in enumerate(identified, 1)}
    dec["id"] = dec["model_file"].map(ids)
    pf["id"] = pf["model_file"].map(ids)

    # Reconstruct B0/B1 solely for disaggregated audit tables; B2 is read from
    # the authoritative row export.
    b0 = dec["bw"].to_numpy(float) / dec["bytes_total"].to_numpy(float)
    b1 = group_predictions(dec, "bytes_total")
    b2 = dec["pred_ours"].to_numpy(float)
    measured = dec["avg_ts"].to_numpy(float)
    dec["pred_B0"] = b0
    dec["pred_B1"] = b1
    dec["ape_B0"] = ape(b0, measured)
    dec["ape_B1"] = ape(b1, measured)

    # Uniform-KV counterfactual, with its own training-only coefficient fit.
    dec["kv_uniform"] = (
        2
        * dec["n_layers"].to_numpy(float)
        * dec["n_kv_heads"].to_numpy(float)
        * dec["head_dim"].to_numpy(float)
        * dec["n_depth"].to_numpy(float)
        * 2
    )
    dec["bytes_uniform"] = (
        dec["file_bytes"].to_numpy(float) * dec["active_frac"].to_numpy(float)
        + dec["kv_uniform"].to_numpy(float)
    )
    pred_uniform = group_predictions(dec, "bytes_uniform")
    dec["ape_uniform"] = ape(pred_uniform, measured)

    # Negative output-projection experiment, recomputed from the authoritative
    # decode export and frozen vocabulary/model widths.
    sys.path.insert(0, str(ROOT))
    from llmperf.analyze import (
        add_features,
        apply_eta,
        calibration_probe_models,
        fit_eta,
        load_measurements,
        select_primary_measurements,
    )
    from llmperf.refine import (error_rows, fit_model,
                                leave_one_machine_out,
                                leave_one_machine_out_b2)

    refine = dec.copy()
    refine["vocab_size"] = refine["model_file"].map(
        lambda name: metadata[name]["vocab_size"]
    )
    refine["d_model"] = refine["model_file"].map(
        lambda name: metadata[name]["d_model"]
    )
    refine["out_flops"] = 2.0 * refine["vocab_size"] * refine["d_model"]
    refine["t_meas"] = 1.0 / refine["avg_ts"]
    refine_train = refine[refine["split"] == "train"]
    transfer_b2 = leave_one_machine_out_b2(refine)
    transfer_b2_test = leave_one_machine_out_b2(
        refine, target_split="test")
    transfer_two = leave_one_machine_out(
        refine, use_output_term=True, relative=False)
    target_fitted_all = {
        host: float(group["ape_B2"].mean())
        for host, group in dec.groupby("host", sort=False)
    }
    projection_rows: list[dict] = []
    for loss, relative in (("absolute time", False), ("relative time", True)):
        one = fit_model(refine_train, use_output_term=False, relative=relative)
        two = fit_model(refine_train, use_output_term=True, relative=relative)
        for host in [*hosts, "all hosts"]:
            evaluated = refine if host == "all hosts" else refine[refine.host == host]
            for row in error_rows("one term", one, evaluated) + error_rows(
                "two terms", two, evaluated
            ):
                projection_rows.append({"host": host, "loss": loss, **row})
    projection = pd.DataFrame(projection_rows)

    # Retain a diagnostic comparison with the archived, pre-protocol pass. It
    # is not mixed into the scored cohort. The raw append-only CSV contains one
    # complete legacy decode grid, so this comparison can be reconstructed
    # without selecting on prediction residuals.
    protocol_fields = [
        "load_before", "load_after", "attempts", "kept_cv_pct", "max_cv_pct"
    ]
    archived = measurements_by_host["lun-mac"].copy()
    archived["quality_protocol"] = archived[protocol_fields].notna().all(axis=1)
    numeric = [
        "avg_ts", "stddev_ts", "file_bytes", "n_params", "n_active_params",
        "n_layers", "n_kv_heads", "head_dim", "n_depth", "n_prompt",
        "n_gen", "n_gpu_layers", "n_expert",
    ]
    for column in numeric:
        archived[column] = pd.to_numeric(archived[column], errors="coerce")
    archived = archived[
        archived["error"].fillna("").astype(str).str.strip().eq("")
        & ~archived["quality_protocol"]
        & archived["n_gpu_layers"].eq(99)
    ].copy()
    calibration = calibrations["lun-mac"]
    calibration_by_host = {"lun-mac": calibration}
    split_map = {name: entry["split"] for name, entry in manifest.items()}
    archived = add_features(
        archived, calibration_by_host, split_map, metadata, ROOT / "models",
        prefer_live=False,
    )
    probe_by_host = calibration_probe_models(calibration_by_host)
    archived = archived[
        np.array(
            [
            name not in probe_by_host.get(host, set())
            for host, name in zip(archived["host"], archived["model_file"])
            ],
            dtype=bool,
        )
        & archived["phase"].eq("decode")
    ].reset_index(drop=True)
    mac_decode = dec[dec.host == "lun-mac"]
    archived_keys = set(
        archived[["host", "model_file", "n_depth"]]
        .itertuples(index=False, name=None)
    )
    mac_keys = set(
        mac_decode[["host", "model_file", "n_depth"]]
        .itertuples(index=False, name=None)
    )
    if archived_keys != mac_keys:
        raise SystemExit(
            "unexpected archived decode cohort: "
            f"archived keys={len(archived_keys)}, current Mac keys={len(mac_keys)}"
        )
    archived_train = archived[archived["split"] == "train"]
    archived_prediction = apply_eta(
        archived,
        fit_eta(archived_train, "bytes_active"),
        "bytes_active",
    )
    archived["ape_B2"] = ape(
        archived_prediction, archived["avg_ts"].to_numpy(float)
    )
    archived["row_cv_pct"] = (
        archived["stddev_ts"] / archived["avg_ts"] * 100.0
    )

    # Account for the complete final protocol cohort, including the reference
    # probes intentionally omitted from fitting and scoring.
    selected = select_primary_measurements(load_measurements(RESULTS))
    selected["phase"] = np.where(selected["n_gen"] > 0, "decode", "prefill")
    selected["row_cv_pct"] = (
        pd.to_numeric(selected["stddev_ts"], errors="coerce")
        / pd.to_numeric(selected["avg_ts"], errors="coerce")
        * 100.0
    )
    selected_probe_mask = np.array(
        [
            name in set(probes_by_host.get(host, []))
            for host, name in zip(selected["host"], selected["model_file"])
        ],
        dtype=bool,
    )
    probe_observations = selected[selected_probe_mask].copy()
    expected_selected = len(dec) + len(pf) + len(probe_observations)
    if len(selected) != expected_selected:
        raise SystemExit(
            "selected-cohort accounting mismatch: "
            f"selected={len(selected)}, scored={len(dec) + len(pf)}, "
            f"probes={len(probe_observations)}"
        )
    campaign_errors: list[str] = []
    for host in hosts:
        host_errors = campaign_validation_errors(
            selected,
            measurements_by_host[host],
            manifest,
            environments[host],
            host,
            require_manifest_coverage=bool(
                EXPECTED_HOSTS[host]["require_manifest_coverage"]
            ),
        )
        campaign_errors.extend(f"{host}: {error}" for error in host_errors)
    if campaign_errors:
        raise SystemExit(
            "campaign completeness/protocol validation failed: "
            + " | ".join(campaign_errors)
        )

    # Conservative sensitivity for every host with an LLM-reference probe
    # exclusion. The final exports use device-copy bandwidth, so these rows do
    # not set B0; quantify the effect of adding them back and refitting B2.
    selected_featured = add_features(
        selected.copy(), calibrations, split_map, metadata, ROOT / "models",
        prefer_live=False,
    )
    probe_sensitivity_rows: list[list[str]] = []
    for host in hosts:
        augmented = selected_featured[
            (selected_featured["host"] == host)
            & (selected_featured["phase"] == "decode")
            & selected_featured["bw"].notna()
        ].copy()
        probe_mask = augmented["is_calibration_probe"].map(bool_value)
        if augmented.empty or not probe_mask.any():
            continue
        augmented_train = augmented[augmented["split"] == "train"]
        if augmented_train.empty:
            raise SystemExit(f"no training rows for probe sensitivity on {host}")
        augmented_prediction = apply_eta(
            augmented,
            fit_eta(augmented_train, "bytes_active"),
            "bytes_active",
        )
        augmented["ape_probe_inclusion"] = ape(
            augmented_prediction, augmented["avg_ts"].to_numpy(float)
        )
        augmented_train = augmented[augmented["split"] == "train"]
        augmented_test = augmented[augmented["split"] == "test"]
        original_train = augmented_train[
            ~augmented_train["is_calibration_probe"].map(bool_value)
        ]
        probe_sensitivity_rows.append(
            [
                tex(host_label(host)),
                str(int(augmented.loc[probe_mask, "model_file"].nunique())),
                str(len(augmented_train)),
                ffmt(augmented_train["ape_probe_inclusion"].mean()),
                str(len(original_train)),
                ffmt(original_train["ape_probe_inclusion"].mean()),
                str(len(augmented_test)),
                ffmt(augmented_test["ape_probe_inclusion"].mean()),
            ]
        )

    cohort_flow_rows: list[list[str]] = []
    for host in hosts:
        raw = measurements_by_host[host]
        success = raw["error"].fillna("").astype(str).str.strip().eq("")
        protocol = success & raw[protocol_fields].notna().all(axis=1)
        selected_host = selected[selected.host == host]
        probes_host = probe_observations[probe_observations.host == host]
        cohort_flow_rows.append(
            [
                tex(host_label(host)),
                str(len(raw)),
                str(int((~success).sum())),
                str(int((success & ~protocol).sum())),
                str(int(protocol.sum())),
                str(int(protocol.sum() - len(selected_host))),
                str(len(selected_host)),
                str(len(probes_host)),
                str(len(selected_host) - len(probes_host)),
            ]
        )

    probe_observations["_phase_order"] = probe_observations["phase"].map(
        {"decode": 0, "prefill": 1}
    )
    probe_observation_rows = [
        [
            tex(host_label(row.host)),
            ids[str(row.model_file)],
            tex(row.phase),
            nfmt(row.n_depth),
            ffmt(row.avg_ts, 3),
            ffmt(row.stddev_ts, 3),
            ffmt(row.row_cv_pct, 3),
            str(int(row.attempts)),
        ]
        for row in probe_observations.sort_values(
            ["host", "model_file", "_phase_order", "n_depth"]
        ).itertuples()
    ]

    # Validate the row exports against the aggregate authoritative tables.
    for host in hosts:
        for split in ("train", "test"):
            actual = float(
                dec.loc[
                    (dec.host == host) & (dec.split == split), "ape_ours"
                ].mean()
            )
            recorded = float(
                err.loc[
                    (err.host == host)
                    & err["model"].str.startswith("B2")
                    & (err["split"] == split),
                    "MAPE_%",
                ].iloc[0]
            )
            if abs(actual - recorded) > 0.011:
                raise SystemExit(
                    f"decode summary mismatch for {host}/{split}: "
                    f"{actual} vs {recorded}"
                )
            primary = pf[
                (pf.host == host) & (pf.split == split) & (pf.n_depth == 0)
            ]
            for short in ("P1", "P2"):
                actual = float(primary[f"ape_{short}"].mean())
                recorded = float(
                    pf_err.loc[
                        (pf_err.host == host)
                        & pf_err["model"].str.startswith(short)
                        & (pf_err["split"] == split),
                        "MAPE_%",
                    ].iloc[0]
                )
                if abs(actual - recorded) > 0.011:
                    raise SystemExit(
                        f"prefill summary mismatch for {host}/{short}/{split}: "
                        f"{actual} vs {recorded}"
                    )

    make_summary_figure(dec, pf)

    configuration_rows: list[list[str]] = []
    quality_rows: list[list[str]] = []
    for host in hosts:
        dhost = dec[dec.host == host]
        phost = pf[pf.host == host]
        configurations = dhost[["model_file", "split"]].drop_duplicates()
        retry_cells = int(
            dhost.loc[pd.to_numeric(dhost.attempts) > 1, "model_file"].nunique()
        )
        configuration_rows.append(
            [
                tex(host_label(host)),
                tex(", ".join(sorted(dhost.backend.astype(str).unique()))),
                str(int((configurations.split == "train").sum())),
                str(int((configurations.split == "test").sum())),
                str(len(dhost)),
                str(len(phost)),
            ]
        )
        threshold = float(pd.to_numeric(dhost.max_cv_pct).iloc[0])
        quality_rows.append(
            [
                tex(host_label(host)),
                str(retry_cells),
                ffmt(pd.to_numeric(dhost.kept_cv_pct).median(), 2),
                ffmt(pd.to_numeric(dhost.kept_cv_pct).max(), 2),
                ffmt(phost.row_cv_pct.median(), 2),
                ffmt(phost.row_cv_pct.max(), 2),
                f"{int((phost.row_cv_pct > threshold).sum())}/{len(phost)}",
            ]
        )

    failures = [
        f"{tex(host_label(host))}: " + ", ".join(path_cell(name) for name in names)
        for host, names in failed_by_host.items()
        if names
    ]
    probe_descriptions = [
        f"{tex(host_label(host))}: "
        + (", ".join(path_cell(name) for name in names) if names else "none")
        for host, names in probes_by_host.items()
    ]
    n_configurations = len(scored_pairs)
    n_train_configurations = int(
        dec[["host", "model_file", "split"]]
        .drop_duplicates()["split"]
        .eq("train")
        .sum()
    )
    n_test_configurations = n_configurations - n_train_configurations
    noisiest_prefill = pf.loc[pf.row_cv_pct.idxmax()]
    host_names = [host_label(host) for host in hosts]
    target_fitted_summary = "; ".join(
        f"{target_fitted_all[host]:.2f}\\% on {host_label(host)}"
        for host in hosts
    )

    lines: list[str] = [
        "% Generated by generate_supplement.py; do not hand-edit.",
        r"\documentclass[10pt]{article}",
        r"\usepackage[letterpaper,margin=0.65in]{geometry}",
        r"\usepackage{newtxtext,newtxmath}",
        r"\usepackage{booktabs,longtable,array,pdflscape,graphicx,url}",
        r"\usepackage[hidelinks]{hyperref}",
        r"\newcolumntype{L}[1]{>{\raggedright\arraybackslash}p{#1}}",
        r"\newcolumntype{R}[1]{>{\raggedleft\arraybackslash}p{#1}}",
        r"\setlength{\parindent}{0pt}",
        r"\setlength{\parskip}{4pt}",
        r"\begin{document}",
        r"\begin{center}",
        r"{\Large\bfseries Companion Results Appendix: GGUF Throughput Prediction\par}",
        r"\vspace{4pt}",
        rf"{{\normalsize Protocol-complete {len(hosts)}-host evidence for the ICASSP manuscript\par}}",
        r"\end{center}",
        r"\textbf{Status.} This is a locally generated companion appendix. It does not claim publication, archival acceptance, or independent replication.",
        r"\section{Scope and cohort}",
        f"The source manifest contains {len(manifest)} GGUF files. Across "
        f"{len(hosts)} hosts ({english_join(host_names)}), "
        f"the scored data contain {len(scored)} unique files and {n_configurations} "
        f"host--file configurations ({n_train_configurations} training and "
        f"{n_test_configurations} held out). Each configuration has one decode and "
        f"one prefill observation at each of the {len(depths)} measured context depths, "
        f"for {len(dec)} decode and {len(pf)} prefill rows. The final protocol selector "
        f"retains {len(selected)} observations from {len(identified)} unique files in "
        f"total: {len(dec) + len(pf)} scored rows "
        f"and {len(probe_observations)} reference-probe rows excluded from prediction "
        "fits and scores. Probe exclusion is host-specific, so a file used as a probe "
        "on one host can be scored on another.",
        "All exported rows carry the declared protocol fields. The retry decision, however, "
        r"uses the worst within-run \emph{decode} CV in an invocation. Prefill rows inherit "
        "that invocation's protocol metadata but are not gated on their own CV. This "
        f"distinction matters on {host_label(noisiest_prefill.host)}, where the noisiest "
        f"prefill row has {noisiest_prefill.row_cv_pct:.2f}\\% CV even though the "
        f"associated decode-gate CV is {float(noisiest_prefill.kept_cv_pct):.2f}\\%.",
        r"\begin{center}",
        r"\small",
        r"\begin{tabular}{llrrrr}",
        r"\toprule",
        r"Host & Backend & Train cfg. & Test cfg. & Scored decode & Scored prefill\\",
        r"\midrule",
        *(" & ".join(row) + r" \\" for row in configuration_rows),
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{center}",
        r"\begin{center}",
        r"\small",
        r"\begin{tabular}{lrrrrrr}",
        r"\toprule",
        r"Host & Retried cells & Gate CV med. & Gate CV max & PP CV med. & PP CV max & PP $>3\%$\\",
        r"\midrule",
        *(" & ".join(row) + r" \\" for row in quality_rows),
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{center}",
        "Attempted configurations with no successful row are "
        + ("; ".join(failures) if failures else "none")
        + ". On the MacBook Pro M4 Max, two raw attempts of the one gpt-oss-120B "
        r"configuration report \texttt{failed to decode prompt batch, res=-3}; "
        "these are prompt-batch failures, not demonstrated model-load or "
        "out-of-memory failures. "
        "Host-specific calibration probes are "
        + "; ".join(probe_descriptions)
        + ".",
        r"\subsection{Cohort accounting}",
        "Raw records, superseded legacy rows, repeated protocol attempts, selected "
        "observations, and scoring exclusions are separated below. A protocol duplicate "
        "is a successful protocol row removed by the atomic host--model--phase--depth "
        "selector; it is not an additional scored observation.",
        r"\begin{center}",
        r"\small",
        r"\setlength{\tabcolsep}{2.8pt}",
        r"\begin{tabular}{lrrrrrrrr}",
        r"\toprule",
        r"Host & Raw & Failed & Legacy ok & Protocol ok & Dup. removed & Selected & Probes & Scored\\",
        r"\midrule",
        *(" & ".join(row) + r" \\" for row in cohort_flow_rows),
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{center}",
    ]

    add_longtable(
        lines,
        "Selected protocol-complete reference-probe observations excluded from prediction fitting and scoring. Measured throughput and SD are in tokens/s; IDs resolve through Table~\\ref{tab:model-id}.",
        "tab:probe-observations",
        r"@{}lllrrrrr@{}",
        ["Host", "ID", "Phase", "Depth", "Measured", "SD", r"CV \%", "Attempts"],
        probe_observation_rows,
        landscape=False,
        lead=[
            r"\subsection{Excluded reference-probe observations}",
            f"These rows complete the {len(selected)}-observation selected cohort and make the "
            "descriptive matched-host and quantization comparisons reproducible. They "
            "are reported only as measurements, never as prediction errors.",
        ],
    )

    if probe_sensitivity_rows:
        lines.extend(
            [
                r"\paragraph{Probe-exclusion sensitivity.}",
                "For each host with calibration probes, B2 is refit after adding "
                "those files back. The original-train column evaluates that refit "
                "only on non-probe training rows; test rows remain held out.",
                r"\begin{center}",
                r"\small",
                r"\setlength{\tabcolsep}{3.0pt}",
                r"\begin{tabular}{lrrrrrrr}",
                r"\toprule",
                r"Host & Probe cfg. & Aug. train $n$ & Aug. MAPE & Original $n$ & Original MAPE & Test $n$ & Test MAPE\\",
                r"\midrule",
                *(" & ".join(row) + r" \\" for row in probe_sensitivity_rows),
                r"\bottomrule",
                r"\end{tabular}",
                r"\end{center}",
            ]
        )

    lines.extend(
        [
        r"\section{Aggregate prediction results}",
        "Results are separated by host because fitted efficiency factors are host-specific; "
        "a pooled row would weight the unequal host cohorts and is not a cross-system score.",
        r"\begin{center}",
        r"\small",
        r"\begin{tabular}{lllrrrr}",
        r"\toprule",
        r"Host & Predictor & Split & $n$ & MAPE (\%) & Median (\%) & P90 / max (\%)\\",
        r"\midrule",
        ]
    )
    for _, row in err.iterrows():
        lines.append(
            f"{tex(host_label(row['host']))} & {predictor_label(row['model'])} & "
            f"{tex(row['split'])} & {int(row['n'])} & "
            f"{ffmt(row['MAPE_%'])} & {ffmt(row['median_APE_%'])} & "
            f"{ffmt(row['p90_APE_%'])} / {ffmt(row['max_APE_%'])} \\\\"
        )
    lines.extend(
        [
            r"\midrule",
        ]
    )
    for _, row in pf_err.iterrows():
        lines.append(
            f"{tex(host_label(row['host']))} & {predictor_label(row['model'])} & "
            f"{tex(row['split'])} & {int(row['n'])} & "
            f"{ffmt(row['MAPE_%'])} & {ffmt(row['median_APE_%'])} & "
            f"{ffmt(row['p90_APE_%'])} / {ffmt(row['max_APE_%'])} \\\\"
        )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{center}",
            "Decode summaries use all three context depths. Prefill headline rows use "
            "only depth 0, because the stated prefill model has no existing-prefix "
            "term; deeper rows are scope diagnostics below.",
        ]
    )

    transfer_rows: list[list[str]] = []
    for variant, frame in (("B2: all target rows", transfer_b2),
                           ("B2: target test rows", transfer_b2_test),
                           ("rejected two-term", transfer_two)):
        for _, row in frame.iterrows():
            transfer_rows.append(
                [
                    tex(variant),
                    tex(host_label(row.held_out_host)),
                    str(int(row.n_train)),
                    str(int(row.n_held)),
                    ffmt(row["MAPE_%"]),
                    ffmt(row["median_APE_%"]),
                    ffmt(row["max_APE_%"]),
                ]
            )
    lines.extend(
        [
            r"\subsection{Leave-one-host-out transfer}",
            "B2 fits median per-quantization coefficients on all non-target training "
            "hosts "
            "and substitutes the untouched target host's bandwidth. The target "
            "set contains all eligible rows on that host; because most files also "
            "occur on at least one source host, this isolates host/runtime transfer "
            "rather than a simultaneous model-and-hardware holdout. A target format "
            "absent from all source training rows uses the source-wide median fallback. "
            "The rejected two-term rows are the absolute-time output-projection extension.",
            r"\begin{center}",
            r"\small",
            r"\begin{tabular}{llrrrrr}",
            r"\toprule",
            r"Variant & Target host & Source $n$ & Target $n$ & MAPE & Median & Max\\",
            r"\midrule",
            *(" & ".join(row) + r" \\" for row in transfer_rows),
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{center}",
            "For context, target-fitted B2 gives all-row MAPE of "
            + target_fitted_summary
            + ". Unlike the "
            "untouched-target transfer rows, these comparators mix training "
            "rows used to fit the target coefficients with held-out rows.",
        ]
    )

    architecture_rows: list[list[str]] = []
    for host in hosts:
        for split in ("train", "test"):
            for moe, label in ((False, "dense"), (True, "MoE")):
                mask = (
                    (dec.host == host)
                    & (dec.split == split)
                    & (dec.is_moe.map(bool_value) == moe)
                )
                if not mask.any():
                    continue
                architecture_rows.append(
                    [
                        tex(host_label(host)),
                        tex(split),
                        tex(label),
                        str(int(mask.sum())),
                        ffmt(dec.loc[mask, "ape_B0"].mean()),
                        ffmt(dec.loc[mask, "ape_B1"].mean()),
                        ffmt(dec.loc[mask, "ape_ours"].mean()),
                    ]
                )
    lines.extend(
        [
            r"\begin{center}",
            r"\small",
            r"\begin{tabular}{lllrrrr}",
            r"\toprule",
            r"Host & Split & Architecture & $n$ & B0 & B1 & B2\\",
            r"\midrule",
            *(" & ".join(row) + r" \\" for row in architecture_rows),
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{center}",
        ]
    )

    uniform_rows: list[list[str]] = []
    for host in hosts:
        for split in ("train", "test"):
            m = (dec.host == host) & (dec.split == split)
            uniform_rows.append(
                [
                    tex(host_label(host)),
                    tex(split),
                    str(int(m.sum())),
                    ffmt(dec.loc[m, "ape_uniform"].mean()),
                    ffmt(dec.loc[m, "ape_ours"].mean()),
                ]
            )
    lines.extend(
        [
            r"\subsection{Per-layer KV correction}",
            "With each variant refit on training rows, replacing a uniform-global "
            "KV denominator by the frozen per-layer metadata produces:",
            r"\begin{center}",
            r"\small",
            r"\begin{tabular}{llrrr}",
            r"\toprule",
            r"Host & Split & $n$ & Uniform KV MAPE (\%) & Per-layer KV MAPE (\%)\\",
            r"\midrule",
            *(" & ".join(row) + r" \\" for row in uniform_rows),
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{center}",
        ]
    )

    depth_zero_rate = dec.loc[
        dec["n_depth"] == min(depths), ["host", "model_file", "avg_ts"]
    ].set_index(["host", "model_file"])["avg_ts"]
    dec["rate_vs_empty"] = [
        float(row.avg_ts) / float(depth_zero_rate.loc[(row.host, row.model_file)])
        for row in dec.itertuples()
    ]
    monotonic = all(
        np.all(np.diff(g.sort_values("n_depth")["avg_ts"].to_numpy(float)) <= 0)
        for _, g in dec.groupby(["host", "model_file"])
    )
    context_rows: list[list[str]] = []
    for host in hosts:
        for depth in depths:
            g = dec[(dec.host == host) & (dec["n_depth"] == depth)]
            context_rows.append(
                [
                    tex(host_label(host)),
                    nfmt(depth),
                    str(int((g["split"] == "train").sum())),
                    str(int((g["split"] == "test").sum())),
                    ffmt(g.loc[g["split"] == "train", "ape_ours"].mean()),
                    ffmt(g.loc[g["split"] == "test", "ape_ours"].mean()),
                    ffmt(g["rate_vs_empty"].median(), 3),
                ]
            )
    lines.extend(
        [
            r"\subsection{Decode context behavior}",
            (
                f"All {n_configurations} scored host--file configurations slow "
                f"monotonically from {min(depths):,} through {max(depths):,} tokens. "
                if monotonic else ""
            )
            + "The normalized rate is each model's throughput divided by its own "
            "depth-0 throughput.",
            r"\begin{center}",
            r"\small",
            r"\begin{tabular}{lrrrrrr}",
            r"\toprule",
            r"Host & Depth & Train $n$ & Test $n$ & Train B2 MAPE & Test B2 MAPE & Median normalized rate\\",
            r"\midrule",
            *(" & ".join(row) + r" \\" for row in context_rows),
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{center}",
        ]
    )

    scope_rows: list[list[str]] = []
    for host in hosts:
        for split in ("train", "test"):
            for depth in depths:
                g = pf[
                    (pf.host == host)
                    & (pf.split == split)
                    & (pf.n_depth == depth)
                ]
                scope_rows.append(
                    [
                        tex(host_label(host)),
                        tex(split),
                        nfmt(depth),
                        str(len(g)),
                        ffmt(g.ape_P1.mean()),
                        ffmt(g.ape_P2.mean()),
                        ffmt(g.ape_P2.median()),
                        ffmt(g.ape_P2.max()),
                    ]
                )
    lines.extend(
        [
            r"\subsection{Prefill scope diagnostic}",
            r"Depth 0 is the primary prefill fit and score. The same fitted coefficients are applied unchanged at larger existing-prefix depths. These errors are prediction errors, distinct from within-run prefill variability; results remain separated by host.",
            r"\begin{center}",
            r"\small",
            r"\begin{tabular}{llrrrrrr}",
            r"\toprule",
            r"Host & Split & Depth & $n$ & P1 MAPE & P2 MAPE & P2 median & P2 max\\",
            r"\midrule",
            *(" & ".join(row) + r" \\" for row in scope_rows),
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{center}",
            r"\begin{figure}[t]",
            r"\centering",
            r"\includegraphics[width=\textwidth]{supplement_summary.pdf}",
            r"\caption{Host-separated diagnostics. Left: effective streamed bandwidth for the scored Qwen3.8-27B quantizations at zero prefix; host coverage differs, and IQ2 has 64 layers/26.90B parameters versus 65 layers/27.32B for the other shown files, so the set is a family comparison rather than a strictly identical-network quantization ladder. Right: P2 held-out error when depth-0 coefficients are applied at each existing-prefix depth. Host results are not pooled.}",
            r"\end{figure}",
        ]
    )

    identity_rows: list[list[str]] = []
    geometry_rows: list[list[str]] = []
    for name in identified:
        m = metadata[name]
        man = manifest[name]
        split = (str(dec.loc[dec.model_file == name, "split"].iloc[0])
                 if name in scored else "probe only")
        measured_hosts = ", ".join(
            host_code(host)
            for host in hosts
            if ((dec.host == host) & (dec.model_file == name)).any()
        ) or "--"
        identity_rows.append(
            [
                ids[name],
                tex(split),
                tex(measured_hosts),
                path_cell(name),
                path_cell(man["repo_id"]),
                tex(m["arch"]),
                tex(m["quant"]),
                nfmt(m["file_bytes"]),
            ]
        )
        active_pct = 100.0 * m["n_active_params"] / m["n_params"]
        expert = f"{m['n_expert_used']}/{m['n_expert']}" if m["n_expert"] else "--"
        kv = m["kv_bytes_by_depth"]
        geometry_rows.append(
            [
                ids[name],
                nfmt(m["n_params"]),
                nfmt(m["n_active_params"]),
                ffmt(active_pct, 1),
                nfmt(m["n_layers"]),
                nfmt(m["d_model"]),
                nfmt(m["vocab_size"]),
                nfmt(m["n_kv_heads"]),
                nfmt(m["head_dim"]),
                expert,
                nfmt(kv["0"]),
                nfmt(kv["4096"]),
                nfmt(kv["16384"]),
            ]
        )

    add_longtable(
        lines,
        "Selected model identity, source, and file size. Sizes are exact bytes; "
        "host codes are "
        + ", ".join(
            f"{host_code(host)} ({host_label(host)})" for host in hosts
        )
        + ", and ``probe only'' denotes a file excluded from every prediction fit and score.",
        "tab:model-id",
        r"@{}l l L{0.75in} L{2.7in} L{2.45in} L{0.9in} l r@{}",
        [
            "ID", "Split", "Scored host(s)", "GGUF filename", "Repository",
            "Architecture", "Quant.", "Bytes",
        ],
        identity_rows,
        lead=[
            r"\section{Selected model metadata}",
            "Tables~\\ref{tab:model-id} and~\\ref{tab:model-geometry} jointly "
            f"reproduce every frozen metadata field used for the {len(identified)} "
            f"unique selected files ({len(scored)} scored and "
            f"{len(identified) - len(scored)} probe-only). Scored host coverage is "
            "explicit because the measurement cohorts are not identical.",
        ],
    )
    add_longtable(
        lines,
        "Selected model geometry. KV columns are exact bytes at the measured depths; $k/E$ is routed/total experts.",
        "tab:model-geometry",
        r"@{}l r r r r r r r r r r r r@{}",
        [
            "ID",
            "Total P",
            "Active P",
            r"Active \%",
            "Layers",
            "Width",
            "Vocab.",
            "KV heads",
            "Head dim.",
            "Used/all",
            "KV@0",
            "KV@4096",
            "KV@16384",
        ],
        geometry_rows,
    )

    decode_rows: list[list[str]] = []
    split_order = {"train": 0, "test": 1}
    dec_sorted = dec.assign(_split=dec.split.map(split_order)).sort_values(
        ["host", "_split", "id", "n_depth"]
    )
    for _, row in dec_sorted.iterrows():
        decode_rows.append(
            [
                tex(host_label(row.host)),
                row.id,
                tex(row.split),
                nfmt(row.n_depth),
                ffmt(row.avg_ts),
                ffmt(row.pred_ours),
                ffmt(row.ape_ours),
                ffmt(row.row_cv_pct),
                ffmt(row.load_before),
                str(int(float(row.attempts))),
            ]
        )
    add_longtable(
        lines,
        "Protocol-complete decode measurements and B2 predictions. The retry gate uses the worst decode CV in each host--file invocation. Pre-load is the macOS one-minute POSIX load average or, on Windows, busy-logical-CPU equivalents; it is meaningful within a host but not comparable across hosts.",
        "tab:decode-rows",
        r"@{}l l l r r r r r r r@{}",
        ["Host", "ID", "Split", "Depth", "Measured", "B2", r"APE \%", r"Row CV \%", "Load", "Attempts"],
        decode_rows,
        lead=[
            r"\section{All scored decode observations}",
            f"These are the {len(dec)} rows exported by "
            + path_cell("results/predictions.csv")
            + ". Throughput is tokens/s; APE is absolute percentage error. "
            "Rows are ordered by host, split, model ID, and depth.",
        ],
    )

    prefill_rows: list[list[str]] = []
    pf_sorted = pf.assign(_split=pf.split.map(split_order)).sort_values(
        ["host", "_split", "id", "n_depth"]
    )
    for _, row in pf_sorted.iterrows():
        scope = r"\textbf{primary}" if int(row.n_depth) == 0 else "diagnostic"
        prefill_rows.append(
            [
                tex(host_label(row.host)),
                row.id,
                tex(row.split),
                nfmt(row.n_depth),
                scope,
                ffmt(row.avg_ts),
                ffmt(row.pred_P1),
                ffmt(row.ape_P1),
                ffmt(row.pred_P2),
                ffmt(row.ape_P2),
                ffmt(row.row_cv_pct),
            ]
        )
    add_longtable(
        lines,
        "Protocol-complete prefill measurements and predictions. The retry decision was decode-gated, not prefill-gated; depth-0 rows are the primary evaluation and larger depths are scope diagnostics.",
        "tab:prefill-rows",
        r"@{}l l l r l r r r r r r@{}",
        ["Host", "ID", "Split", "Depth", "Scope", "Measured", "P1", "P1 APE", "P2", "P2 APE", r"Row CV \%"],
        prefill_rows,
        lead=[
            r"\section{All scored prefill observations}",
            f"These are the {len(pf)} rows exported by "
            + path_cell("results/predictions_prefill.csv")
            + ". P1 has one host coefficient; P2 has one host-by-quantization "
            "coefficient. Both are fit only at depth 0. The row CV column is "
            "reported rather than used as an inclusion filter.",
        ],
    )

    repeat_mean = float(repeat.decode_ts.mean())
    repeat_cv = float(repeat.decode_ts.std(ddof=1) / repeat_mean * 100)
    repeat_prefill_mean = float(repeat.prefill_ts.mean())
    repeat_prefill_cv = float(
        repeat.prefill_ts.std(ddof=1) / repeat_prefill_mean * 100
    )
    repeat_decode_span = float(
        (repeat.decode_ts.max() - repeat.decode_ts.min()) / repeat_mean * 100
    )
    repeat_prefill_span = float(
        (repeat.prefill_ts.max() - repeat.prefill_ts.min())
        / repeat_prefill_mean
        * 100
    )
    repeat_decode_trend = float(
        np.polyfit(repeat.run_index, repeat.decode_ts, 1)[0]
        / repeat_mean
        * 100
    )
    current_mac = dec[dec.host == "lun-mac"]
    current_test = current_mac[current_mac["split"] == "test"]
    archived_test = archived[archived["split"] == "test"]
    protocol_comparison = [
        [
            "Archived pre-protocol",
            str(len(archived)),
            str(len(archived_test)),
            ffmt(archived_test["ape_B2"].mean()),
            ffmt(archived_test["ape_B2"].median()),
            ffmt(np.percentile(archived_test["ape_B2"], 90)),
            ffmt(archived_test["ape_B2"].max()),
            ffmt(archived["row_cv_pct"].median()),
            ffmt(archived["row_cv_pct"].max()),
        ],
        [
            "Final Mac protocol-complete",
            str(len(current_mac)),
            str(len(current_test)),
            ffmt(current_test["ape_ours"].mean()),
            ffmt(current_test["ape_ours"].median()),
            ffmt(np.percentile(current_test["ape_ours"], 90)),
            ffmt(current_test["ape_ours"].max()),
            ffmt(current_mac["row_cv_pct"].median()),
            ffmt(current_mac["row_cv_pct"].max()),
        ],
    ]
    repeat_rows = [
        [
            str(int(row.run_index)),
            tex(row.timestamp),
            ffmt(row.settle_s, 0),
            ffmt(row.prefill_ts, 3),
            ffmt(row.decode_ts, 3),
            ffmt(row.decode_sd_within, 3),
        ]
        for _, row in repeat.sort_values("run_index").iterrows()
    ]
    lines.extend(
        [
            r"\section{Repeatability and calibration}",
            "A separate repeatability series exists only for the MacBook Pro M4 Max; "
            "variability on the other hosts is represented by the within-invocation "
            "repetitions in the row tables, not by equivalent across-run series. "
            f"The six Mac runs give mean decode throughput {repeat_mean:.3f} tokens/s "
            f"and sample between-run CV {repeat_cv:.3f}\\%; prefill is "
            f"{repeat_prefill_mean:.3f} tokens/s with CV {repeat_prefill_cv:.3f}\\%. "
            f"The max-minus-min spans are {repeat_decode_span:.2f}\\% and "
            f"{repeat_prefill_span:.2f}\\% of their respective means. Decode has "
            f"a descriptive linear trend of {repeat_decode_trend:.2f}\\% per run. "
            "These are single-cell diagnostics, not general detection thresholds.",
            r"\begin{center}",
            r"\small",
            r"\begin{tabular}{rlrrrr}",
            r"\toprule",
            r"Run & Timestamp & Settle (s) & Prefill & Decode & Decode within-run SD\\",
            r"\midrule",
            *(" & ".join(row) + r" \\" for row in repeat_rows),
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{center}",
            "The append-only Mac measurement file also preserves a complete "
            "pre-protocol decode grid. It is excluded from every fit and score "
            "above. The comparison is MacBook-only and is not pooled with other hosts.",
            r"\begin{center}",
            r"\small",
            r"\begin{tabular}{lrrrrrrrr}",
            r"\toprule",
            r"Pass & All $n$ & Test $n$ & Test MAPE & Median & P90 & Max & All-row CV med. & All-row CV max\\",
            r"\midrule",
            *(" & ".join(row) + r" \\" for row in protocol_comparison),
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{center}",
        ]
    )

    uncontrolled = pd.read_csv(paths["archived uncontrolled comparison"])
    u = uncontrolled[
        (uncontrolled.model_file == "Qwen3.5-4B-Q4_K_M.gguf")
        & (pd.to_numeric(uncontrolled.n_depth) == 0)
    ].copy()
    u["phase"] = np.where(pd.to_numeric(u.n_gen) > 0, "decode", "prefill")
    contamination_rows: list[list[str]] = []
    for phase in ("prefill", "decode"):
        vals = u.loc[u.phase == phase, "avg_ts"].to_numpy(float)
        high, low = float(vals.max()), float(vals.min())
        contamination_rows.append(
            [tex(phase), ffmt(high, 3), ffmt(low, 3), ffmt((high - low) / high * 100, 2)]
        )
    lines.extend(
        [
            "For comparison, the retained historical uncontrolled file contains two "
            r"depth-0 observations of the same Qwen3.5-4B Q4\_K\_M cell:",
            r"\begin{center}",
            r"\small",
            r"\begin{tabular}{lrrr}",
            r"\toprule",
            r"Phase & Higher throughput & Lower throughput & Decrease (\%)\\",
            r"\midrule",
            *(" & ".join(row) + r" \\" for row in contamination_rows),
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{center}",
        ]
    )

    runtime_rows: list[list[str]] = []
    calibration_rows: list[list[str]] = []
    probe_rows: list[list[str]] = []
    for host in hosts:
        calibration = calibrations[host]
        environment = environments[host]
        gpu = calibration["torch_gpu"]
        ref = calibration.get("llm_ref") or {}
        protocol = environment.get("protocol") or {}
        runtime_rows.append(
            [
                tex(host_label(host)),
                tex(environment["platform"]),
                tex(gpu["name"]),
                str(environment["threads"]),
                str(protocol.get("repetitions", "--")),
                ffmt(protocol.get("settle_seconds", math.nan), 0),
                tex(", ".join(str(int(x)) for x in sorted(
                    pd.to_numeric(dec.loc[dec.host == host, "n_gpu_layers"]).unique()
                ))),
            ]
        )
        calibration_rows.extend(
            [
                [
                    tex(host_label(host)),
                    "CPU STREAM-style triad",
                    f"{calibration['cpu_triad']['gb_s']:.3f} GB/s",
                    f"{calibration['cpu_triad']['reps']} repetitions",
                ],
                [
                    tex(host_label(host)),
                    "CPU FP32 matrix multiply",
                    f"{calibration['cpu_matmul']['gflops'] / 1000:.3f} TFLOP/s",
                    f"{calibration['cpu_matmul']['reps']} repetitions",
                ],
                [
                    tex(host_label(host)),
                    f"{tex(str(gpu['backend']).upper())} device copy",
                    f"{gpu['copy_gb_s']:.3f} GB/s",
                    tex(gpu["name"]),
                ],
                [
                    tex(host_label(host)),
                    f"{tex(str(gpu['backend']).upper())} FP16 matrix multiply",
                    f"{gpu['fp16_tflops']:.3f} TFLOP/s",
                    tex(gpu["name"]),
                ],
            ]
        )
        if ref.get("available"):
            calibration_rows.append(
                [
                    tex(host_label(host)),
                    "LLM reference ceiling",
                    f"{ref['eff_bw_gb_s']:.3f} GB/s",
                    path_cell(ref["anchor_model"]),
                ]
            )
            for row in ref.get("probes", []):
                probe_rows.append(
                    [
                        tex(host_label(host)),
                        path_cell(row["model"]),
                        ffmt(row["file_gb"], 2),
                        ffmt(row["decode_tok_s"], 3),
                        ffmt(row["stddev_tok_s"], 3),
                        ffmt(row["stddev_tok_s"] / row["decode_tok_s"] * 100, 2),
                        ffmt(row["eff_bw_gb_s"], 3),
                    ]
                )
    lines.extend(
        [
            r"\subsection{Host settings and calibration}",
            r"\begin{center}",
            r"\small",
            r"\begin{tabular}{lllrrrr}",
            r"\toprule",
            r"Host & Platform & Accelerator & Threads & Reps & Settle (s) & Requested ngl\\",
            r"\midrule",
            *(" & ".join(row) + r" \\" for row in runtime_rows),
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{center}",
            r"The requested \texttt{n\_gpu\_layers=99} setting asks the runtime to "
            "offload as many layers as it can; it is not direct telemetry proving "
            "that every tensor and KV allocation remained resident on the accelerator. "
            "No device-memory trace was recorded, so the appendix does not label these "
            "observations as confirmed fully resident.",
            r"\begin{center}",
            r"\small",
            r"\begin{tabular}{llrl}",
            r"\toprule",
            r"Host & Calibration & Value & Detail\\",
            r"\midrule",
            *(" & ".join(row) + r" \\" for row in calibration_rows),
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{center}",
            "For the fitted B1/B2 and P1/P2 models, the measured bandwidth or "
            "FLOP/s scale cancels algebraically after fitting on this same host. "
            "It sets the scale of B0 and supports roofline interpretation; it does "
            "not by itself produce the reported fitted-model accuracy. The exported "
            "row-level bandwidth field uses the device-copy result on all hosts. "
            "The LLM references are retained as diagnostics and are not the B0 "
            "bandwidth value used in the current exports.",
        ]
    )
    if probe_rows:
        lines.extend(
            [
                r"\begin{center}",
                r"\small",
                r"\begin{tabular}{llrrrrr}",
                r"\toprule",
                r"Host & Probe & File GB & Decode & SD & CV (\%) & Effective GB/s\\",
                r"\midrule",
                *(" & ".join(row) + r" \\" for row in probe_rows),
                r"\bottomrule",
                r"\end{tabular}",
                r"\end{center}",
            ]
        )

    projection_table: list[list[str]] = []
    for _, row in projection.iterrows():
        projection_table.append(
            [
                tex(host_label(row.host) if row.host != "all hosts" else row.host),
                tex(row.loss),
                tex(row.model),
                tex(row.split),
                str(int(row.n)),
                ffmt(row["MAPE_%"]),
                ffmt(row["median_APE_%"]),
                ffmt(row["p90_APE_%"]),
                ffmt(row["max_APE_%"]),
            ]
        )
    lines.extend(
        [
            r"\section{Output-projection sensitivity}",
            "The additive output-projection term is reported as a "
            "sensitivity analysis. It is not selected: held-out improvement is not "
            "consistent across loss functions and hosts, and the term partly recharges "
            "weights already included in the streamed-byte term. Host rows and the "
            "unequally weighted all-host aggregate are shown separately. The one-term "
            "rows in this section are refit under the listed time-domain losses; they "
            "are not the median-ratio B2 headline and need not reproduce its MAPE.",
            r"\begin{center}",
            r"\small",
            r"\begin{tabular}{llllrrrrr}",
            r"\toprule",
            r"Host & Loss & Model & Split & $n$ & MAPE & Median & P90 & Max\\",
            r"\midrule",
            *(" & ".join(row) + r" \\" for row in projection_table),
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{center}",
        ]
    )

    provenance_rows = [
        [tex(label), path_cell(path.relative_to(ROOT).as_posix()), hash_cell(sha256(path))]
        for label, path in paths.items()
    ]
    source = metadata_doc.get("source") or {}
    rtx_rows = dec[dec.host == "rtx5080"]
    rtx_largest = rtx_rows.loc[rtx_rows.bytes_total.astype(float).idxmax()]
    rtx_vram_gb = float(calibrations["rtx5080"]["torch_gpu"]["vram_gb"])
    heldout_by_host = {
        host: int(
            dec.loc[(dec.host == host) & (dec.split == "test"), "model_file"].nunique()
        )
        for host in hosts
    }
    heldout_summary = english_join(
        [
            f"{heldout_by_host[host]} configurations on {host_label(host)}"
            for host in hosts
        ]
    )
    heldout_quants = sorted(
        dec.loc[dec["split"] == "test", "quant"].astype(str).unique()
    )
    prefill_gate_summary = english_join(
        [
            (
                f"{int((group.row_cv_pct > pd.to_numeric(group.max_cv_pct)).sum())} "
                f"of {len(group)} on {host_label(host)}"
            )
            for host in hosts
            for group in [pf[pf.host == host]]
        ]
    )
    probe_count_summary = english_join(
        [
            f"{len(probes_by_host[host])} on {host_label(host)}"
            for host in hosts
        ]
    )
    add_longtable(
        lines,
        "Exact inputs used to generate this appendix.",
        "tab:hashes",
        r"@{}L{1.35in} L{2.55in} L{5.5in}@{}",
        ["Role", "Path", "SHA-256"],
        provenance_rows,
        lead=[
            r"\section{Provenance and explicit limits}",
            "The on-disk inputs used for this appendix have the following SHA-256 "
            "digests. The repository HEAD at generation was "
            + hash_cell(git_head())
            + "; the generator does not assert a clean working tree, so the file "
            "digests, rather than this commit alone, identify the inputs.",
        ],
    )
    lines.extend(
        [
            r"\begin{itemize}",
            f"\\item {english_join(host_names)} contributed unequal, overlapping "
            "cohorts. Headline coefficients are fit separately by measured host. "
            "Each leave-one-host-out audit fits on all remaining measured hosts and "
            "does not establish universal transfer to new runtime stacks or formats.",
            r"\item Every scored row records a requested \texttt{n\_gpu\_layers=99}, but no device-memory trace or runtime-reported resident-layer count was retained. This is a maximal-offload request, not proof of full accelerator residency; no partial-offload sweep or offload-cliff result exists.",
            f"\\item The largest RTX modeled file-plus-KV footprint is "
            f"{float(rtx_largest.bytes_total) / 1e9:.3f} GB for "
            + path_cell(rtx_largest.model_file)
            + f" at depth {int(rtx_largest.n_depth):,}, while calibration reports "
            f"{rtx_vram_gb:.3f} GB of device memory. The differing accounting "
            "conventions further preclude a residency claim.",
            r"\item Runtime provenance is host-specific and is reproduced in the environment and auxiliary provenance files. Placeholder commits, project commits, and usage-text version output are not interpreted as immutable runtime-binary revisions.",
            f"\\item The Studio acquisition records immutable repository revisions and "
            f"per-file LFS SHA-256 identifiers for all {len(manifest)} manifest entries. "
            "These identify the remote artifacts; they are not retrospective local-file "
            "hashes for the earlier MacBook and RTX copies.",
            r"\item Derived parameter counts and per-depth KV bytes are consumed from the frozen metadata snapshot during appendix generation. This reproduces the analysis but is not a fresh, independent metadata extraction.",
            f"\\item Held-out coverage is small: {heldout_summary}. Its observed "
            "quantization formats are "
            + ", ".join(tex(value) for value in heldout_quants)
            + ". No learning curve establishes how many reference configurations suffice.",
            r"\item Decode MAPE treats three depths from each host--file configuration as observations; those rows are correlated. No narrow confidence claim is warranted.",
            f"\\item Prefill's primary result is restricted to depth 0. Larger-depth "
            "predictions are scope diagnostics because the equation has no existing-prefix "
            "term. The retry rule is decode-only; the counts of prefill rows above "
            f"each host's declared CV gate are {prefill_gate_summary}.",
            f"\\item Calibration-probe exclusions are host-specific ({probe_count_summary}). "
            "Failed configurations and probe identities are listed above; a recorded "
            "prompt-batch failure is not relabeled as an out-of-memory failure without "
            "supporting telemetry.",
            r"\item Qwen3.8-27B IQ2 has 64 layers and 26.90B parameters, whereas the other shown Qwen3.8-27B files have 65 layers and 27.32B parameters. The quantization plot is therefore a near-family comparison, not a controlled bit-format substitution for one identical network.",
            r"\end{itemize}",
            "The metadata snapshot additionally records these parser/source Git object IDs:",
            r"\begin{center}",
            r"\small",
            r"\begin{tabular}{ll}",
            r"\toprule",
            r"Object & Identifier\\",
            r"\midrule",
            "Code commit & " + hash_cell(source.get("code_commit", "unavailable")) + r" \\",
            "Measurement blob & " + hash_cell(source.get("measurement_blob", "unavailable")) + r" \\",
            "Predictions blob & " + hash_cell(source.get("predictions_blob", "unavailable")) + r" \\",
            "Shape-log blob & " + hash_cell(source.get("shape_log_blob", "unavailable")) + r" \\",
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{center}",
            r"\end{document}",
            "",
        ]
    )

    output = HERE / "supplement.tex"
    output.write_text("\n".join(lines), encoding="utf-8", newline="\n")
    print(f"wrote {output}")
    print(f"wrote {HERE / 'supplement_summary.pdf'}")


if __name__ == "__main__":
    main()
