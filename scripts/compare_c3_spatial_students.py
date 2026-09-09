#!/usr/bin/env python3
"""Re-evaluate saved students; report ranking, calibration and threshold errors."""
from pathlib import Path
import argparse
import json
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

from c3_safe.artifacts import finish_run, file_sha, new_run, run_cli, write_json
from scripts.train_c3_spatial_baseline import evaluate, load_images


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset",required=True,type=Path)
    parser.add_argument("--field-only",required=True,type=Path)
    parser.add_argument("--with-exposure",required=True,type=Path)
    parser.add_argument("--output",required=True,type=Path)
    args = parser.parse_args()
    import torch
    from c3_safe.spatial_student import SpatialRequirementNet
    torch.set_num_threads(2)
    rows = [json.loads(l) for l in (args.dataset/"TRANSITIONS.jsonl").read_text().splitlines()]
    images = load_images(args.dataset,rows)
    output = new_run(args.output)
    models,results = {},{}
    for name,source in (("field_only",args.field_only),("with_exposure",args.with_exposure)):
        saved = torch.load(source,map_location="cpu",weights_only=True)
        if saved["source_dataset_manifest_sha256"] != file_sha(args.dataset/"MANIFEST.json"):
            raise ValueError("checkpoint was trained against a different dataset manifest")
        model = SpatialRequirementNet().eval()
        model.load_state_dict(saved["state_dict"])
        models[name] = model
        folder = output/name
        folder.mkdir()
        metrics = evaluate(model,images,rows,folder)
        records = json.loads((folder/"EXPOSURE_PREDICTIONS.json").read_text())
        for split in ("train","validation","test"):
            rr = [r for r in records if r["split"]==split]
            truth = np.array([r["oracle_water_exposure"]>.5 for r in rr])
            prediction = np.array([r["predicted_water_exposure"]>=.5 for r in rr])
            metrics[split]["exposure_threshold_0_5"] = {
                "positive_count":int(truth.sum()),"negative_count":int((~truth).sum()),
                "false_negatives":int((truth & ~prediction).sum()),
                "false_positives":int((~truth & prediction).sum()),
                "false_negative_rate":float((truth & ~prediction).sum()/max(1,truth.sum())),
                "false_positive_rate":float((~truth & prediction).sum()/max(1,(~truth).sum())),
            }
        results[name] = {"checkpoint_sha256":file_sha(source),"exposure_weight":saved.get("exposure_weight",0.),"metrics":metrics}
    examples = [next(r for r in rows if r["split"]=="test" and not r["oracle_contact_violation"]),
                next(r for r in rows if r["split"]=="test" and r["oracle_contact_violation"])]
    board = Image.new("RGB",(4*280,2*310),"#f4f7fb")
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf",16)
    except OSError:
        font = ImageFont.load_default()
    for i,row in enumerate(examples):
        item = images[row["image_sha256"]]
        rgb = torch.from_numpy(item["rgb"].transpose(2,0,1)).unsqueeze(0)
        arrays = [(item["rgb"]*255).astype(np.uint8),item["field"][...,0]]
        titles = ["RGB observation","Oracle water field","Field-only prediction","With exposure supervision"]
        for model in models.values():
            arrays.append(model.predict_field(rgb)[0,0].numpy())
        for j,array in enumerate(arrays):
            if array.ndim==2:
                array = (np.stack((array,.25+array*.5,1-array*.65),axis=-1)*255).clip(0,255).astype(np.uint8)
            tile = Image.fromarray(array).resize((256,256),Image.Resampling.NEAREST)
            board.paste(tile,(j*280+12,i*310+28))
            draw = ImageDraw.Draw(board)
            draw.text((j*280+12,i*310+5),titles[j],font=font,fill="#172b42")
            if j==0:
                draw.text((j*280+12,i*310+288),f"Example {i+1}; contact={row['oracle_contact_violation']}",font=font,fill="#172b42")
    board.save(output/"student_fields.png")
    summary = {"status":"POSTHOC_DEVELOPMENT_DIAGNOSTIC_NO_NEW_TRAINING",
               "source_dataset_manifest_sha256":file_sha(args.dataset/"MANIFEST.json"),
               "threshold":.5,"results":results,
               "limits":"single seed; water only; development data; exposure accuracy is not closed-loop safety"}
    finish_run(output,summary,source_paths=["scripts/compare_c3_spatial_students.py"])
    print(json.dumps({k:v["metrics"]["test"] for k,v in results.items()},indent=2))


if __name__ == "__main__":
    run_cli(main)
