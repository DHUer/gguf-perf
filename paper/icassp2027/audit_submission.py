"""Audit the generated ICASSP paper against the supplied submission rules.

Run from anywhere with::

    python paper/icassp2027/audit_submission.py

The author identity check intentionally remains blocking until the real author
name and affiliation are supplied.  All other checks are deterministic and
operate on the packaged PDFs, their build products, and the LaTeX source.
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import pymupdf


ROOT = Path(__file__).resolve().parents[2]
PAPER = ROOT / "paper" / "icassp2027"
MAIN_PDF = PAPER / "gguf-throughput-icassp2027.pdf"
SUPPLEMENT_PDF = PAPER / "gguf-throughput-supplement.pdf"
MAIN_TEX = PAPER / "main.tex"
SUPPLEMENT_TEX = PAPER / "supplement.tex"
STUDIO_INPUTS = (
    ROOT / "results" / "measurements_mac-studio-m4-max.csv",
    ROOT / "results" / "calibration_mac-studio-m4-max.json",
    ROOT / "results" / "env_mac-studio-m4-max.json",
)
STUDIO_PROVENANCE_INPUTS = {
    "system profile": ROOT / "results" / "system_profile_mac-studio-m4-max.txt",
    "operating system": ROOT / "results" / "os_mac-studio-m4-max.txt",
    "source commit": ROOT / "results" / "source_commit_mac-studio-m4-max.txt",
    "measurement-source SHA-256": (
        ROOT / "results" / "measurement_source_tree_mac-studio-m4-max.sha256"
    ),
    "llama.cpp package": ROOT / "results" / "llama_cpp_package_mac-studio-m4-max.txt",
    "llama-bench SHA-256": ROOT / "results" / "llama_bench_mac-studio-m4-max.sha256",
    "model sources": ROOT / "results" / "model_sources_mac-studio-m4-max.json",
    "model integrity": ROOT / "results" / "model_integrity_mac-studio-m4-max.json",
    "Python packages": ROOT / "results" / "python_packages_mac-studio-m4-max.txt",
    "Tectonic package": ROOT / "results" / "tectonic_package_mac-studio-m4-max.txt",
    "Tectonic SHA-256": ROOT / "results" / "tectonic_mac-studio-m4-max.sha256",
}

STALE_THREE_HOST_PATTERNS = (
    ("two-system title", r"ACROSS\s+TWO\s+SYSTEMS"),
    ("two-host appendix subtitle", r"protocol-complete\s+two-host\s+evidence"),
    ("two-host cohort", r"\bacross\s+two\s+hosts\b"),
    ("planned Studio has no rows", r"planned\s+Mac\s+Studio\s+has\s+no\s+rows"),
    ("two-system transfer limit", r"uses\s+only\s+two\s+systems"),
    ("two-stack comparison", r"across\s+the\s+two\s+host/runtime\s+stacks"),
    ("missing model revisions", r"model\s+revisions\s+and\s+full\s+hashes\s+were\s+not\s+frozen"),
    ("missing immutable model IDs", r"not\s+immutable\s+repository\s+revisions\s+or\s+full-file\s+hashes"),
)

failures: list[str] = []
blockers: list[str] = []


def report(name: str, passed: bool, detail: str, *, blocker: bool = False) -> None:
    mark = "PASS" if passed else ("BLOCK" if blocker else "FAIL")
    print(f"[{mark:5}] {name}: {detail}")
    if not passed:
        (blockers if blocker else failures).append(name)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def word_count_latex(fragment: str) -> int:
    # A percent sign starts a TeX comment only when it is not escaped.  The
    # abstract contains literal percentages (``\%``), which still count as
    # visible prose rather than deleting the rest of their source lines.
    text = re.sub(r"(?<!\\)%.*", "", fragment)
    text = re.sub(r"\\(?:textbf|textit|texttt|emph)\{([^{}]*)\}", r"\1", text)
    text = re.sub(r"\\[%&_#]", " ", text)
    text = re.sub(r"\\[A-Za-z]+", " ", text)
    text = re.sub(r"[${}~]", " ", text)
    # Count visible lexical tokens.  Decimal numbers, source-level TeX dashes,
    # and ordinary hyphenated words each render as one whitespace-delimited
    # word in the abstract (for example, ``13.1`` and ``phase--depth``).
    token = r"[A-Za-z0-9]+(?:(?:--|[-.'’])[A-Za-z0-9]+)*"
    return len(re.findall(token, text))


def font_audit(doc: pymupdf.Document) -> tuple[int, int, list[str]]:
    fonts: dict[int, tuple] = {}
    for page in doc:
        for font in page.get_fonts(full=True):
            fonts[font[0]] = font
    missing: list[str] = []
    unsubset: list[str] = []
    for xref, _ext, _kind, basefont, *_rest in fonts.values():
        if xref <= 0:
            continue
        if not doc.extract_font(xref)[3]:
            missing.append(basefont)
        if not re.match(r"^[A-Z]{6}\+", basefont):
            unsubset.append(basefont)
    return len(fonts), len(missing), sorted(set(unsubset))


def printed_bounds(page: pymupdf.Page) -> tuple[float, float, float, float]:
    """Return the bounds of marks actually visible after PDF clipping.

    Vector hatch definitions can extend far beyond an axes rectangle before a
    PDF clipping path is applied.  Auditing raw path coordinates would therefore
    report invisible marks outside the page margins.  A 2x grayscale render
    measures the final composited page instead.
    """
    scale = 2
    pix = page.get_pixmap(matrix=pymupdf.Matrix(scale, scale), alpha=False)
    samples = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
        pix.height, pix.width, pix.n)
    gray = samples[:, :, :3].min(axis=2)
    rows, columns = np.where(gray < 248)
    return (
        float(columns.min()) / scale,
        float(rows.min()) / scale,
        float(columns.max() + 1) / scale,
        float(rows.max() + 1) / scale,
    )


def three_host_content_errors(documents: dict[str, str]) -> list[str]:
    """Find stale two-host prose or missing Studio coverage in sources/PDFs."""
    errors: list[str] = []
    for document_name, text in documents.items():
        for label, pattern in STALE_THREE_HOST_PATTERNS:
            if re.search(pattern, text, re.IGNORECASE):
                errors.append(f"{label} in {document_name}")

    for document_name in ("main source", "main PDF"):
        text = documents.get(document_name, "")
        if not re.search(r"ACROSS\s+THREE\s+SYSTEMS", text, re.IGNORECASE):
            errors.append(f"three-system title missing from {document_name}")
        if "Mac Studio M4 Max" not in text:
            errors.append(f"Studio missing from {document_name}")
    for document_name in ("supplement source", "supplement PDF"):
        if "Mac Studio M4 Max" not in documents.get(document_name, ""):
            errors.append(f"Studio missing from {document_name}")
    return errors


def _pdf_text(path: Path) -> str:
    with pymupdf.open(path) as document:
        return "\n".join(page.get_text() for page in document)


def audit_three_host_freshness(main_source: str) -> None:
    """Activate three-host content checks only after a complete Studio campaign."""
    missing = [path.name for path in STUDIO_INPUTS if not path.is_file()]
    if missing:
        print(
            "[SKIP ] three-host manuscript freshness: waiting for "
            + ", ".join(missing)
        )
        return

    try:
        sys.path.insert(0, str(ROOT))
        from llmperf.analyze import load_measurements, select_primary_measurements
        from paper.icassp2027.generate_supplement import (
            binary_provenance_errors,
            campaign_validation_errors,
            measurement_source_provenance_errors,
            model_integrity_provenance_errors,
            model_source_provenance_errors,
        )

        results = ROOT / "results"
        raw = pd.read_csv(STUDIO_INPUTS[0])
        calibration = json.loads(STUDIO_INPUTS[1].read_text(encoding="utf-8"))
        environment = json.loads(STUDIO_INPUTS[2].read_text(encoding="utf-8"))
        manifest = json.loads(
            (results / "model_manifest.json").read_text(encoding="utf-8")
        )
        selected = select_primary_measurements(load_measurements(results))
        campaign_errors = campaign_validation_errors(
            selected,
            raw,
            manifest,
            environment,
            "mac-studio-m4-max",
            require_manifest_coverage=True,
        )
        for field in ("host", "platform", "git_commit", "cpu"):
            if calibration.get(field) != environment.get(field):
                campaign_errors.append(
                    f"calibration/environment {field} mismatch: "
                    f"{calibration.get(field)!r} vs {environment.get(field)!r}"
                )
        missing_provenance = [
            label
            for label, path in STUDIO_PROVENANCE_INPUTS.items()
            if not path.is_file() or path.stat().st_size == 0
        ]
        if missing_provenance:
            campaign_errors.append(
                "missing or empty Studio provenance: "
                + ", ".join(missing_provenance)
            )
        else:
            model_sources = json.loads(
                STUDIO_PROVENANCE_INPUTS["model sources"].read_text(
                    encoding="utf-8"
                )
            )
            model_integrity = json.loads(
                STUDIO_PROVENANCE_INPUTS["model integrity"].read_text(
                    encoding="utf-8"
                )
            )
            manifest_digest = sha256(results / "model_manifest.json")
            campaign_errors.extend(
                model_source_provenance_errors(
                    model_sources,
                    manifest,
                    manifest_digest,
                )
            )
            campaign_errors.extend(
                model_integrity_provenance_errors(
                    model_integrity,
                    manifest,
                    model_sources,
                    manifest_digest,
                    "mac-studio-m4-max",
                )
            )
            campaign_errors.extend(
                measurement_source_provenance_errors(
                    STUDIO_PROVENANCE_INPUTS[
                        "measurement-source SHA-256"
                    ].read_text(encoding="utf-8"),
                    ROOT,
                )
            )
            campaign_errors.extend(
                binary_provenance_errors(
                    STUDIO_PROVENANCE_INPUTS["llama.cpp package"].read_text(
                        encoding="utf-8"
                    ),
                    STUDIO_PROVENANCE_INPUTS["llama-bench SHA-256"].read_text(
                        encoding="utf-8"
                    ),
                    "runner_sha256",
                    "llama-bench",
                )
            )
            campaign_errors.extend(
                binary_provenance_errors(
                    STUDIO_PROVENANCE_INPUTS["Tectonic package"].read_text(
                        encoding="utf-8"
                    ),
                    STUDIO_PROVENANCE_INPUTS["Tectonic SHA-256"].read_text(
                        encoding="utf-8"
                    ),
                    "binary_sha256",
                    "Tectonic",
                )
            )
    except Exception as exc:
        campaign_errors = [f"could not validate Studio campaign: {exc}"]

    report(
        "Studio campaign completeness",
        not campaign_errors,
        (
            "all manifest models have a six-row grid or an explicit failure"
            if not campaign_errors
            else "; ".join(campaign_errors)
        ),
    )
    if campaign_errors:
        print(
            "[SKIP ] three-host manuscript freshness: Studio campaign is not complete"
        )
        return

    supplement_source = (
        SUPPLEMENT_TEX.read_text(encoding="utf-8")
        if SUPPLEMENT_TEX.is_file()
        else ""
    )
    documents = {
        "main source": main_source,
        "supplement source": supplement_source,
        "main PDF": _pdf_text(MAIN_PDF) if MAIN_PDF.is_file() else "",
        "supplement PDF": (
            _pdf_text(SUPPLEMENT_PDF) if SUPPLEMENT_PDF.is_file() else ""
        ),
    }
    issues = three_host_content_errors(documents)
    report(
        "three-host manuscript freshness",
        not issues,
        (
            "main and supplement identify the completed three-host cohort"
            if not issues
            else "stale or missing content: " + ", ".join(issues)
        ),
    )


def main() -> int:
    source = MAIN_TEX.read_text(encoding="utf-8")
    abstract_match = re.search(
        r"\\begin\{abstract\}(.*?)\\end\{abstract\}", source, re.DOTALL)
    abstract_words = word_count_latex(abstract_match.group(1)) if abstract_match else 0
    report("abstract", 100 <= abstract_words <= 150,
           f"{abstract_words} words; required range is about 100--150")

    title_match = re.search(r"\\title\{([^{}]*)\}", source)
    title = title_match.group(1) if title_match else ""
    report("title style", bool(title) and title == title.upper(),
           "title is present and fully capitalized")
    report("nine-point body", r"\ninept" in source,
           "source selects the template's nine-point body style")
    report("full justification", r"\raggedright" not in source,
           "no ragged-right override is present")
    report("page numbering", not re.search(r"\\(?:page|pagenumbering)\b", source),
           "source does not add page numbers")
    audit_three_host_freshness(source)

    placeholder_tokens = ("Author Name", "Affiliation", "author@example.com")
    placeholders = [token for token in placeholder_tokens if token in source]
    report("author identity", not placeholders,
           "replace " + ", ".join(placeholders) if placeholders else "verified author block present",
           blocker=True)

    main_doc = pymupdf.open(MAIN_PDF)
    report("main page count", main_doc.page_count == 5,
           f"{main_doc.page_count} pages; the fifth is the allowed reference page")
    sizes = {(round(page.rect.width, 1), round(page.rect.height, 1))
             for page in main_doc}
    report("main page size", sizes == {(612.0, 792.0)},
           f"page boxes {sorted(sizes)} points (US Letter)")
    report("main encryption", not main_doc.is_encrypted, "PDF is not encrypted")

    page5 = main_doc[4].get_text()
    body_headings = ("ABSTRACT", "INTRODUCTION", "EXPERIMENTAL METHOD",
                     "RESULTS", "LIMITATIONS AND CONCLUSION")
    refs_only = "REFERENCES" in page5 and not any(h in page5 for h in body_headings)
    report("reference-only fifth page", refs_only,
           "page 5 contains the bibliography and no technical-body section")

    font_count, missing_count, unsubset = font_audit(main_doc)
    report("main font embedding", missing_count == 0,
           f"{font_count} font resources; {missing_count} missing programs")
    report("main font subsetting", not unsubset,
           "every base-font name has a six-letter subset prefix" if not unsubset
           else "unsubset fonts: " + ", ".join(unsubset))

    raster_images = sum(len(page.get_images(full=True)) for page in main_doc)
    report("main graphics", raster_images == 0,
           f"{raster_images} raster image objects; charts remain vector graphics")

    bounds = [printed_bounds(page) for page in main_doc]
    # The template's 7 x 9 inch box is x=54..558 and y=72..720 points.
    # A four-point tolerance admits glyph/stroke overshoot around baselines.
    in_box = all(x0 >= 50 and x1 <= 562 and y1 <= 724 and
                 (index == 0 or y0 >= 68)
                 for index, (x0, y0, x1, y1) in enumerate(bounds))
    detail = "; ".join(
        f"p{i + 1}=({x0:.1f},{y0:.1f})--({x1:.1f},{y1:.1f})"
        for i, (x0, y0, x1, y1) in enumerate(bounds))
    report("print-area bounds", in_box, detail)

    title_blocks = [b for b in main_doc[0].get_text("blocks")
                    if "GGUF-METADATA PREDICTION" in b[4]]
    title_top = title_blocks[0][1] if title_blocks else float("nan")
    report("title position", bool(title_blocks) and 90 <= title_top <= 105,
           f"title glyph box begins at {title_top:.1f} pt; nominal line is 99.4 pt")

    first_page_blocks = main_doc[0].get_text("blocks")
    abstract_labels = [b for b in first_page_blocks if b[4].strip() == "ABSTRACT"]
    index_terms = [b for b in first_page_blocks if b[4].lstrip().startswith("Index Terms")]
    abstract_blocks = []
    if abstract_labels and index_terms:
        abstract_blocks = [
            b for b in first_page_blocks
            if b[1] >= abstract_labels[0][3] and b[3] <= index_terms[0][1]
            and b[4].strip()
        ]
    abstract_height = (max(b[3] for b in abstract_blocks)
                       - min(b[1] for b in abstract_blocks)
                       if abstract_blocks else float("nan"))
    report("abstract height", bool(abstract_blocks) and abstract_height <= 225,
           f"{abstract_height:.1f} pt; maximum is 225 pt (3.125 inches)")

    page1_blocks = main_doc[0].get_text("blocks")
    left_boxes = [b for b in page1_blocks if 50 <= b[0] <= 60 and 290 <= b[2] <= 305]
    right_boxes = [b for b in page1_blocks if 310 <= b[0] <= 320 and 550 <= b[2] <= 562]
    left_width = float(np.median([b[2] - b[0] for b in left_boxes]))
    right_width = float(np.median([b[2] - b[0] for b in right_boxes]))
    column_gap = float(np.median([b[0] for b in right_boxes]) -
                       np.median([b[2] for b in left_boxes]))
    columns_ok = (240 <= left_width <= 248 and 240 <= right_width <= 248 and
                  14 <= column_gap <= 20)
    report("two-column geometry", columns_ok,
           f"widths {left_width / 72:.2f}/{right_width / 72:.2f} in, "
           f"gap {column_gap / 72:.2f} in")

    footer_text = [span["text"].strip() for page in main_doc
                   for block in page.get_text("dict")["blocks"] if "lines" in block
                   for line in block["lines"] for span in line["spans"]
                   if span["bbox"][1] >= 744 and span["text"].strip()]
    report("visible page numbers", not footer_text,
           "no text appears in the footer/page-number region" if not footer_text
           else "footer text: " + ", ".join(footer_text[:5]))

    text_sizes = [span["size"] for page in main_doc[:4]
                  for block in page.get_text("dict")["blocks"] if "lines" in block
                  for line in block["lines"] for span in line["spans"]]
    body_mode = Counter(round(size, 2) for size in text_sizes).most_common(1)[0][0]
    report("effective body font", body_mode >= 8.9,
           f"modal extracted size is {body_mode:.2f} pt (nominal 9 pt)")

    figure_spans = [span for page in main_doc[1:4]
                    for block in page.get_text("dict")["blocks"] if "lines" in block
                    for line in block["lines"] for span in line["spans"]
                    if ("TimesNewRoman" in span["font"] or
                        "DejaVu" in span["font"])]
    base_figure_sizes = [span["size"] for span in figure_spans
                         if span["size"] >= 8.8]
    reduced_scripts = [span for span in figure_spans if span["size"] < 8.8]
    min_figure_base = min(base_figure_sizes) if base_figure_sizes else 0.0
    report("figure-label font", min_figure_base >= 8.9,
           f"smallest base label is {min_figure_base:.2f} pt; "
           f"{len(reduced_scripts)} smaller spans are math super/subscripts")

    for label, packaged, built in (
        ("main package", MAIN_PDF, PAPER / "build" / "main.pdf"),
        ("supplement package", SUPPLEMENT_PDF,
         PAPER / "supplement-build" / "supplement.pdf"),
    ):
        same = packaged.is_file() and built.is_file() and sha256(packaged) == sha256(built)
        report(label, same, "packaged PDF matches the latest build byte-for-byte")

    supplement = pymupdf.open(SUPPLEMENT_PDF)
    supplement_sizes = {(round(page.rect.width, 1), round(page.rect.height, 1))
                        for page in supplement}
    valid_letter = supplement_sizes <= {(612.0, 792.0), (792.0, 612.0)}
    report("supplement geometry", supplement.page_count > 0 and valid_letter,
           f"{supplement.page_count} Letter portrait/landscape pages")
    report("supplement encryption", not supplement.is_encrypted,
           "PDF is not encrypted")
    sfont_count, smissing, sunsubset = font_audit(supplement)
    report("supplement fonts", smissing == 0 and not sunsubset,
           f"{sfont_count} font resources, all embedded and subsetted")

    forbidden_log = re.compile(
        r"undefined|Overfull \\[hv]box|Package balance Warning", re.IGNORECASE)
    log_hits: list[str] = []
    for log in (PAPER / "build" / "main.log",
                PAPER / "supplement-build" / "supplement.log"):
        for line in log.read_text(encoding="utf-8", errors="replace").splitlines():
            if forbidden_log.search(line):
                log_hits.append(f"{log.name}: {line.strip()}")
    report("LaTeX diagnostics", not log_hits,
           "no undefined references/citations or overfull boxes" if not log_hits
           else "; ".join(log_hits[:5]))

    print()
    if failures:
        print("Submission audit failed: " + ", ".join(failures))
        return 1
    if blockers:
        print("All machine-verifiable checks pass; waiting on: " + ", ".join(blockers))
        return 2
    print("Submission audit passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
