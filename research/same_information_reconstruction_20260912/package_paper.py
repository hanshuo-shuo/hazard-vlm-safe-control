"""Quest-only delivery audit and isolated build of the manuscript source bundle.

This performs file-integrity and document checks, not new scientific fitting.
The visual-review statement is specific to the already inspected final PDF.
"""
import argparse
import hashlib
import json
import re
import shutil
import subprocess
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import pymupdf

REVIEWED_PDF_SHA256 = "0716d4c96b688706656c37f8274159dd910ab6864deab5b7653a1bf1c625ece4"
SOURCE_FILES = [
    "main.tex", "evidence.tex", "comparison.tex", "discussion.tex", "appendix.tex",
    "case_table.tex", "references.bib", "iclr2027_conference.sty",
    "iclr2027_conference.bst", "fancyhdr.sty", "natbib.sty", "math_commands.tex",
    "build.sh", "README.md", "figures/task_scene.png",
    "figures/oracle_confirmation.png", "figures/same_information_tradeoff.png",
]
PERSONAL = re.compile(rb"hanshuo|shv7753|/Users/|/projects/|p33100|hazard-vlm-safe-control", re.I)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def main(root, build, out):
    out.mkdir(exist_ok=False)
    paper = root / "paper/iclr2027"
    source = root / "research/same_information_reconstruction_20260912"
    report = {"at_utc": datetime.now(timezone.utc).isoformat(), "status": "CHECKING"}
    stage_counts = {}
    for stage in ("development", "appended", "report", "verification", "publication", "paper_figures"):
        manifest = read(root / stage / "MANIFEST.json")
        for name, expected in manifest.items():
            assert digest(root / stage / name) == expected, (stage, name)
        stage_counts[stage] = len(manifest)
    report["stage_manifest_entries_checked"] = stage_counts
    old_inputs = {}
    for stage in ("development", "appended"):
        for name, expected in read(root / stage / "AUDIT.json")["source_hashes"].items():
            if name in old_inputs:
                assert old_inputs[name] == expected
            old_inputs[name] = expected
    for name, expected in old_inputs.items():
        assert digest(name) == expected, name
    report["old_input_hashes_checked"] = len(old_inputs)
    for name, expected in read(root / "SOURCE_FREEZE.json")["files"].items():
        assert digest(source / name) == expected, name
    for name, expected in read(root / "OPERATOR_FREEZE.json")["implementation_hashes"].items():
        assert digest(source / name) == expected, name
    report["source_and_operator_freezes_unchanged"] = True

    assert digest(build / "main.pdf") == REVIEWED_PDF_SHA256
    qa = read(build / "PDF_QA.json")
    assert qa["pdf_sha256"] == REVIEWED_PDF_SHA256
    assert qa["pages"] == 13 and qa["main_text_pages"] == 9
    for key in ("personal_identifier_hits", "text_outside_page", "overfull_boxes", "undefined_warnings"):
        assert not qa[key], key
    report["final_pdf_qa"] = qa
    report["visual_review"] = {
        "reviewer": "Codex visual inspection; not a claim of completed human author review",
        "pdf_sha256": REVIEWED_PDF_SHA256,
        "pages_inspected": list(range(1, 14)),
        "result": "All final rendered pages inspected; figures, tables, equations, citations and page breaks readable; no clipping or overlap observed",
    }
    template = read(paper / "TEMPLATE_SOURCE.json")
    assert digest(paper / "official_style.zip") == template["sha256"]
    with zipfile.ZipFile(paper / "official_style.zip") as archive:
        for name in ("iclr2027_conference.sty", "iclr2027_conference.bst"):
            matches = [x for x in archive.namelist() if Path(x).name == name and "__MACOSX" not in x]
            assert len(matches) == 1
            assert archive.read(matches[0]) == (paper / name).read_bytes(), name
    report["official_style_unchanged"] = template

    bundle_dir = out / "manuscript_source"
    bundle_dir.mkdir()
    file_hashes = {}
    for name in SOURCE_FILES:
        blob = (paper / name).read_bytes()
        assert not PERSONAL.search(blob), name
        destination = bundle_dir / name
        destination.parent.mkdir(exist_ok=True, parents=True)
        destination.write_bytes(blob)
        file_hashes[name] = digest(destination)
    bundle_zip = out / "anonymous_manuscript_source.zip"
    with zipfile.ZipFile(bundle_zip, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name in SOURCE_FILES:
            archive.write(bundle_dir / name, arcname="manuscript_source/" + name)
    report["manuscript_files"] = file_hashes
    report["anonymous_source_scope"] = "Manuscript only; full anonymous experimental supplement remains to be prepared"
    # Build the exact extracted ZIP in an independent directory, not the live tree.
    test_dir = out / "isolated_zip_test"
    with zipfile.ZipFile(bundle_zip) as archive:
        archive.extractall(test_dir)
    with (out / "portable_build.log").open("w") as stream:
        subprocess.run(["bash", "build.sh"], cwd=test_dir / "manuscript_source", stdout=stream,
                       stderr=subprocess.STDOUT, check=True)
    rebuilt = test_dir / "manuscript_source/build/main.pdf"
    original_doc, rebuilt_doc = pymupdf.open(build / "main.pdf"), pymupdf.open(rebuilt)
    assert len(original_doc) == len(rebuilt_doc) == 13
    for i, (a, b) in enumerate(zip(original_doc, rebuilt_doc)):
        assert a.get_text() == b.get_text(), ("page text", i + 1)
        assert a.get_pixmap(dpi=110).samples == b.get_pixmap(dpi=110).samples, ("page raster", i + 1)
    rebuilt_log = (rebuilt.parent / "main.log").read_text()
    assert "FIRST-STATEMENTS-PAGE=10" in rebuilt_log
    assert "Overfull" not in rebuilt_log and "undefined" not in rebuilt_log
    report["portable_build"] = {
        "passed": True, "pages": 13, "main_text_pages": 9,
        "all_page_text_and_110dpi_rasters_identical": True,
        "binary_pdf_identity_claimed": False,
    }
    shutil.copy2(build / "main.pdf", out / "perfect_coarse_predictions_iclr2027.pdf")
    shutil.copy2(build / "PDF_QA.json", out / "PDF_QA.json")
    logs = out / "diagnostic_build_logs"
    for previous in sorted(paper.glob("build_*")):
        if not previous.is_dir():
            continue
        target = logs / previous.name
        target.mkdir(parents=True)
        for item in previous.iterdir():
            if item.is_file() and item.suffix in (".log", ".blg", ".txt", ".json"):
                shutil.copy2(item, target / item.name)
    report["delivery_hashes"] = {
        name: digest(out / name)
        for name in ("perfect_coarse_predictions_iclr2027.pdf", "anonymous_manuscript_source.zip")
    }
    report["status"] = "PASS"
    report["scope"] = "No new data, fit, training, certificate or submission; document packaging and integrity audit only"
    (out / "DELIVERY.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: report[key] for key in ("status", "stage_manifest_entries_checked", "old_input_hashes_checked", "portable_build", "delivery_hashes")}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--build", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    main(args.root, args.build, args.out)
