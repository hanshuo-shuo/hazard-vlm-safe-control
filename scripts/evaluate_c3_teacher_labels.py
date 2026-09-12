#!/usr/bin/env python3
"""Offline renderer-oracle diagnostics for real teacher labels; no model calls."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from c3_safe.artifacts import file_sha, finish_run, new_run, run_cli, write_json
from c3_safe.geometry import CHANNELS, GroundProjection, measure_exposure
from c3_safe.teacher_io import inside, parse_response, rasterize_response


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--labels", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    label_manifest = json.loads((args.labels / "MANIFEST.json").read_text())
    for name, sha in label_manifest["artifacts"].items():
        if file_sha(inside(args.labels, name)) != sha:
            raise ValueError(f"teacher artifact integrity failure: {name}")
    spec = json.loads((args.labels / "RUN_SPEC.json").read_text())
    if spec["source_dataset_manifest_sha256"] != file_sha(args.dataset / "MANIFEST.json"):
        raise ValueError("teacher labels belong to a different source dataset")
    rows = [json.loads(line) for line in (args.dataset / "TRANSITIONS.jsonl").read_text().splitlines()]
    by_image = {row["image_sha256"]: row for row in rows}
    records, fields, known_masks = {}, {}, {}
    for path in sorted((args.labels / "records").glob("*.json")):
        record = json.loads(path.read_text())
        sha = record["image_sha256"]
        if sha not in by_image or sha in records:
            raise ValueError("unknown or repeated teacher image identity")
        if record["teacher_model"] != spec["model_id"] or record["teacher_revision"] != spec["model_revision"] or record["prompt_sha256"] != spec["prompt_sha256"]:
            raise ValueError("teacher identity or prompt changed")
        if hashlib.sha256(record["raw_response"].encode()).hexdigest() != record["response_sha256"]:
            raise ValueError("teacher raw response hash mismatch")
        records[sha] = record
        if record["status"] != "ACCEPTED":
            continue
        p = inside(args.labels, record["field_path"])
        if file_sha(p) != record["field_sha256"]:
            raise ValueError("teacher field hash mismatch")
        with np.load(p, allow_pickle=False) as stored:
            field, known = stored["requirement_field"], stored["known_mask"]
        reconstructed = rasterize_response(parse_response(record["raw_response"]), record["width"], record["height"])
        if not np.array_equal(field, reconstructed[0]) or not np.array_equal(known, reconstructed[1]):
            raise ValueError("stored label cannot be reconstructed from raw teacher response")
        fields[sha], known_masks[sha] = field, known
    if len(records) != spec["request_count"]:
        raise ValueError("incomplete teacher run")
    output = new_run(args.output)
    metrics, per_image, exposures = {}, [], []
    for split in ("train", "validation", "test"):
        images = [(sha, row) for sha, row in by_image.items() if row["split"] == split]
        counts = np.zeros((3, 3), dtype=np.int64)
        known_pixels, pixels, invalid = 0, 0, 0
        for sha, row in images:
            if sha not in fields:
                invalid += 1
                continue
            if file_sha(args.dataset / row["image_path"]) != sha or file_sha(args.dataset / row["requirement_field_path"]) != row["field_sha256"]:
                raise ValueError("oracle comparison source drift")
            with np.load(args.dataset / row["requirement_field_path"], allow_pickle=False) as saved:
                truth = saved["requirement_field"] > .5
            field, known = fields[sha] > .5, known_masks[sha]
            if field.shape != truth.shape or known.shape != truth.shape[:2]:
                raise ValueError("teacher/oracle shape mismatch")
            tp = ((field & truth) & known[..., None]).sum(axis=(0,1))
            fp = ((field & ~truth) & known[..., None]).sum(axis=(0,1))
            fn = ((~field & truth) & known[..., None]).sum(axis=(0,1))
            counts += np.stack((tp,fp,fn), axis=1)
            known_pixels += int(known.sum()); pixels += known.size
            per_image.append({"image_sha256": sha, "split": split, "image_path": row["image_path"],
                              "known_fraction": float(known.mean()),
                              "water_known_iou": float(tp[0] / (tp[0]+fp[0]+fn[0])) if tp[0]+fp[0]+fn[0] else None})
        split_exposure = []
        for row in rows:
            if row["split"] != split:
                continue
            result = {"transition_id": row["transition_id"], "split": split, "status": "invalid_teacher_response"}
            sha = row["image_sha256"]
            if sha in fields:
                c = row["camera_calibration"]
                projection = GroundProjection(c["width"], c["height"], tuple(map(tuple,c["matrix"])), c["source"])
                exposure, _ = measure_exposure(fields[sha], row["motion_samples"], row["robot_radius"], projection,
                                              visibility=known_masks[sha])
                result["status"] = exposure.validity
                if exposure.validity == "valid":
                    result.update(teacher_water_exposure=exposure.values[0], oracle_water_exposure=row["exposure"]["values"][0])
            split_exposure.append(result)
        exposures.extend(split_exposure)
        valid = [e for e in split_exposure if e["status"] == "valid"]
        y = np.array([e["oracle_water_exposure"] for e in valid])
        p = np.array([e["teacher_water_exposure"] for e in valid])
        channels = {}
        for channel, (tp, fp, fn) in zip(CHANNELS, counts.tolist()):
            channels[channel] = {"tp": tp, "fp": fp, "fn": fn,
                                 "iou_on_known_pixels": tp / (tp+fp+fn) if tp+fp+fn else None}
        metrics[split] = {"unique_images": len(images), "invalid_responses": invalid,
                          "known_pixel_fraction_among_accepted": known_pixels / pixels if pixels else None,
                          "channels": channels, "sampled_transitions": len(split_exposure),
                          "valid_exposure_count": len(valid),
                          "unknown_projection_count": sum(e["status"] == "unknown_projection" for e in split_exposure),
                          "exposure_mae_on_valid_only": float(np.abs(y-p).mean()) if len(valid) else None,
                          "high_exposure_count_on_valid_only": int((y>.5).sum()),
                          "false_negative_count_on_valid_only": int(((y>.5)&(p<.5)).sum()),
                          "low_exposure_count_on_valid_only": int((y<=.5).sum()),
                          "false_positive_count_on_valid_only": int(((y<=.5)&(p>=.5)).sum())}
    write_json(output / "PER_IMAGE.json", per_image)
    write_json(output / "EXPOSURE_COMPARISON.json", exposures)
    board = Image.new("RGB", (4*264, 3*286), "#f4f7fb")
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", 14)
    except OSError:
        font = ImageFont.load_default()
    draw = ImageDraw.Draw(board)
    for i, split in enumerate(("train", "validation", "test")):
        row = next(r for r in rows if r["split"] == split)
        sha = row["image_sha256"]
        with Image.open(args.dataset / row["image_path"]) as im:
            rgb = np.asarray(im.convert("RGB"))
        with np.load(args.dataset / row["requirement_field_path"], allow_pickle=False) as oracle:
            truth = oracle["requirement_field"][...,0]
        arrays = [rgb, truth, fields[sha][...,0] if sha in fields else np.zeros(truth.shape),
                  known_masks[sha] if sha in fields else np.zeros(truth.shape)]
        for j, (title, array) in enumerate(zip(("RGB / "+split, "Oracle water", "Teacher water", "Known pixels (white)"), arrays)):
            if array.ndim == 2:
                array = np.repeat((array.astype(float)*255).astype(np.uint8)[...,None], 3, axis=2)
            board.paste(Image.fromarray(array).resize((248,248), Image.Resampling.NEAREST), (j*264+8,i*286+28))
            draw.text((j*264+8,i*286+7), title, font=font, fill="#172b42")
    board.save(output / "teacher_fields.png")
    summary = {"status": "REAL_TEACHER_RENDERER_ORACLE_DEVELOPMENT_DIAGNOSTIC", "model_id": spec["model_id"],
               "revision": spec["model_revision"], "accepted": len(fields), "invalid_responses": len(records)-len(fields),
               "source_dataset_manifest_sha256": file_sha(args.dataset / "MANIFEST.json"),
               "source_teacher_manifest_sha256": file_sha(args.labels / "MANIFEST.json"), "metrics": metrics,
               "human_review": "NOT_PERFORMED", "scientific_gates_passed": [],
               "limits": ["water-only development data", "unknown pixels excluded explicitly", "metrics conditional on reported coverage",
                          "simulator masks are geometric comparison only, not human-reviewed semantic truth", "no teacher/student or policy superiority claim"]}
    finish_run(output, summary, source_paths=["scripts/evaluate_c3_teacher_labels.py"])
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    run_cli(main)
