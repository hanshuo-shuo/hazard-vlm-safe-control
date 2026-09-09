#!/usr/bin/env python3
"""Train the RGB-only non-VLM student against existing simulator-oracle masks.

This is a small, single-seed development baseline on water scenes. It is not
teacher distillation, a learned controller, or an OOD/generalization claim.
"""
from pathlib import Path
import argparse
import json
import sys

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

from c3_safe.artifacts import file_sha, finish_run, new_run, run_cli, write_json
from c3_safe.data import audit_splits, load_spec
from c3_safe.geometry import CHANNELS, GroundProjection, measure_exposure


def average_precision(labels, scores):
    labels = np.asarray(labels,dtype=bool).ravel()
    scores = np.asarray(scores,dtype=float).ravel()
    if not labels.any():
        return None
    order = np.argsort(-scores,kind="stable")
    y, s = labels[order], scores[order]
    ends = np.r_[np.nonzero(np.diff(s))[0],len(s)-1]
    tp = np.cumsum(y)[ends]
    recall = tp/labels.sum()
    precision = tp/(ends+1)
    return float(np.sum(np.diff(np.r_[0.,recall])*precision))


def roc_auc(labels, scores):
    y, s = np.asarray(labels,dtype=bool), np.asarray(scores,dtype=float)
    if not y.any() or y.all():
        return None
    p,n = s[y],s[~y]
    return float(np.mean((p[:,None]>n[None,:]) + .5*(p[:,None]==n[None,:])))


def load_images(dataset, rows):
    by_hash = {}
    for row in rows:
        sha = row["image_sha256"]
        if sha in by_hash:
            if by_hash[sha]["field_sha256"] != row["field_sha256"]:
                raise ValueError("identical image has inconsistent spatial labels")
            continue
        image_path,field_path = dataset/row["image_path"],dataset/row["requirement_field_path"]
        if file_sha(image_path)!=sha or file_sha(field_path)!=row["field_sha256"]:
            raise ValueError("dataset image/field integrity failure")
        if row["label_source"] != "SIMULATOR_ORACLE_NOT_VLM":
            raise ValueError("this baseline expects simulator-only labels")
        with Image.open(image_path) as im:
            rgb = np.asarray(im.convert("RGB"),dtype=np.float32)/255.
        with np.load(field_path,allow_pickle=False) as data:
            field = data["requirement_field"].astype(np.float32)
        if field.shape!=rgb.shape or not np.isfinite(field).all() or (field<0).any() or (field>1).any():
            raise ValueError("invalid spatial field")
        by_hash[sha] = {"split":row["split"],"rgb":rgb,"field":field,"field_sha256":row["field_sha256"]}
    return by_hash


def calibration_error(labels, scores, bins=10):
    y, p = np.asarray(labels,dtype=float), np.asarray(scores,dtype=float)
    total = 0.
    for b in range(bins):
        mask = (p >= b/bins) & (p < (b+1)/bins if b < bins-1 else p <= 1.)
        if mask.any():
            total += float(mask.mean()*abs(p[mask].mean()-y[mask].mean()))
    return total


def evaluate(model, image_data, rows, output=None):
    import torch
    predictions = {}
    model.eval()
    with torch.no_grad():
        for sha,item in image_data.items():
            rgb = torch.from_numpy(item["rgb"].transpose(2,0,1)).unsqueeze(0)
            predictions[sha] = model.predict_field(rgb)[0].permute(1,2,0).numpy()
    results = {}
    exposure_records = []
    for split in ("train","validation","test"):
        items = [(sha,r) for sha,r in image_data.items() if r["split"]==split]
        metrics = {}
        for c,channel in enumerate(CHANNELS):
            truth = np.concatenate([r["field"][...,c].ravel() for _,r in items])>0.5
            scores = np.concatenate([predictions[sha][...,c].ravel() for sha,_ in items])
            if not truth.any():
                metrics[channel] = {"status":"NOT_EVALUATED_NO_POSITIVE_REGIONS"}
                continue
            prediction = scores >= .5
            intersection = int((prediction & truth).sum())
            union = int((prediction | truth).sum())
            metrics[channel] = {"iou":intersection/max(1,union),"auprc":average_precision(truth,scores),
                                "positive_pixels":int(truth.sum()),"pixels":len(truth)}
        predicted,truth,contact = [],[],[]
        for row in rows:
            if row["split"]!=split or row["exposure"]["validity"]!="valid":
                continue
            spec = row["camera_calibration"]
            projection = GroundProjection(spec["width"],spec["height"],tuple(map(tuple,spec["matrix"])),spec["source"])
            e,_ = measure_exposure(predictions[row["image_sha256"]],row["motion_samples"],row["robot_radius"],projection)
            if e.validity != "valid":
                raise RuntimeError("prediction altered projection validity")
            predicted.append(e.values[0]); truth.append(row["exposure"]["values"][0])
            contact.append(row["oracle_contact_violation"])
            exposure_records.append({"transition_id":row["transition_id"],"split":split,
                "image_path":row["image_path"],"predicted_water_exposure":e.values[0],
                "oracle_water_exposure":row["exposure"]["values"][0],
                "oracle_contact_violation":row["oracle_contact_violation"]})
        results[split] = {"unique_frames":len(items),"channels":metrics,
            "water_q95_exposure_mae":float(np.mean(np.abs(np.array(predicted)-truth))),
            "water_q95_exposure_auroc":roc_auc(np.array(truth)>.5,predicted),
            "water_contact_auroc_from_q95":roc_auc(contact,predicted),
            "water_exposure_ece":calibration_error(truth,predicted),
            "evaluated_transitions":len(predicted)}
    if output is not None:
        write_json(output/"EXPOSURE_PREDICTIONS.json",exposure_records)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset",required=True,type=Path)
    parser.add_argument("--output",required=True,type=Path)
    parser.add_argument("--epochs",type=int,default=100)
    parser.add_argument("--seed",type=int,default=20260908)
    parser.add_argument("--exposure-weight",type=float,default=0.,
                        help="Predeclared diagnostic ablation: 0 is field-only; 1 adds motion-exposure BCE")
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error("epochs must be positive")
    if not np.isfinite(args.exposure_weight) or args.exposure_weight < 0:
        parser.error("exposure-weight must be finite and nonnegative")
    import torch
    from c3_safe.spatial_student import SpatialRequirementNet
    torch.set_num_threads(2)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    dataset = args.dataset.resolve()
    rows = [json.loads(line) for line in (dataset/"TRANSITIONS.jsonl").read_text().splitlines()]
    split_audit = audit_splits(rows,load_spec())
    images = load_images(dataset,rows)
    train_ids = [sha for sha,item in images.items() if item["split"]=="train"]
    train = [images[sha] for sha in train_ids]
    if not train or not split_audit["all_declared_splits_present"]:
        raise ValueError("all splits and nonempty training images are required")
    x = torch.from_numpy(np.stack([i["rgb"].transpose(2,0,1) for i in train]))
    y = torch.from_numpy(np.stack([i["field"].transpose(2,0,1) for i in train]))
    output = new_run(args.output)
    model = SpatialRequirementNet()
    optimizer = torch.optim.Adam(model.parameters(),lr=.003)
    positives = y.sum((0,2,3))
    negatives = y.shape[0]*y.shape[2]*y.shape[3]-positives
    weights = torch.where(positives>0,(negatives/positives.clamp_min(1)).clamp(max=20),torch.ones_like(positives))
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=weights[:,None,None])
    exposure_targets = {sha:[] for sha in train_ids}
    if args.exposure_weight:
        for row in rows:
            if row["split"] != "train" or row["exposure"]["validity"] != "valid":
                continue
            camera = row["camera_calibration"]
            projection = GroundProjection(camera["width"],camera["height"],tuple(map(tuple,camera["matrix"])),camera["source"])
            truth,mask = measure_exposure(images[row["image_sha256"]]["field"],row["motion_samples"],row["robot_radius"],projection)
            exposure_targets[row["image_sha256"]].append((torch.from_numpy(mask),torch.tensor(truth.values,dtype=torch.float32)))
    history = []
    # Fixed final epoch, fixed learning rate, no checkpoint selection on test.
    model.train()
    for epoch in range(args.epochs):
        losses = []
        for batch in torch.randperm(len(x)).split(16):
            optimizer.zero_grad()
            logits = model(x[batch])
            loss = loss_fn(logits,y[batch])
            if args.exposure_weight:
                probabilities = logits.sigmoid()
                terms = []
                for local_index,index in enumerate(batch.tolist()):
                    for mask,target in exposure_targets[train_ids[index]]:
                        estimate = torch.quantile(probabilities[local_index][:,mask],.95,dim=1)
                        terms.append(torch.nn.functional.binary_cross_entropy(estimate.clamp(1e-6,1-1e-6),target))
                if terms:
                    loss = loss + args.exposure_weight*torch.stack(terms).mean()
            if not torch.isfinite(loss):
                raise RuntimeError("non-finite training loss")
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        history.append({"epoch":epoch+1,"training_loss":float(np.mean(losses))})
        if epoch==0 or (epoch+1)%20==0:
            print(json.dumps(history[-1]),flush=True)
    metrics = evaluate(model,images,rows,output)
    checkpoint = output/"spatial_student.pt"
    torch.save({"state_dict":model.state_dict(),"network_revision":model.revision,
                "training_label_source":"SIMULATOR_ORACLE_NOT_VLM","seed":args.seed,
                "exposure_weight":args.exposure_weight,
                "source_dataset_manifest_sha256":file_sha(dataset/"MANIFEST.json")},checkpoint)
    # Verify the saved model is usable independently and does not require VLM
    # weights, an optimizer, simulator objects, or future-state input.
    loaded = SpatialRequirementNet()
    loaded.load_state_dict(torch.load(checkpoint,map_location="cpu",weights_only=True)["state_dict"])
    loaded.eval()
    probe = x[:1]
    if not torch.equal(loaded.predict_field(probe),model.predict_field(probe)):
        raise RuntimeError("checkpoint round-trip changed field predictions")
    write_json(output/"TRAINING.json",history)
    summary = {"status":"SINGLE_SEED_NON_VLM_SPATIAL_BASELINE_DEVELOPMENT_ONLY",
        "seed":args.seed,"epochs":args.epochs,"torch_version":torch.__version__,
        "training_images":len(train),"optimizer":"Adam","learning_rate":.003,"batch_size":16,
        "exposure_weight":args.exposure_weight,
        "training_objective":"weighted_field_BCE + exposure_weight * motion_exposure_BCE",
        "parameter_count":sum(p.numel() for p in model.parameters()),"model_input":"RGB only",
        "label_source":"SIMULATOR_ORACLE_NOT_VLM","teacher_calls":0,
        "split_audit":split_audit,"metrics":metrics,"checkpoint_sha256":file_sha(checkpoint),
        "source_dataset_manifest_sha256":file_sha(dataset/"MANIFEST.json"),
        "limitations":["water only","single optimization seed","development splits already used for infrastructure debugging",
                       "no controller training","no VLM comparison","no counterfactual learning loss"]}
    finish_run(output,summary,source_paths=["scripts/train_c3_spatial_baseline.py"])
    print(json.dumps(summary,indent=2))


if __name__ == "__main__":
    run_cli(main)
