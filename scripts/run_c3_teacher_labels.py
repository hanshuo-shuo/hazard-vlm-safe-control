#!/usr/bin/env python3
"""Run the pinned Qwen teacher on a verified RGB/prompt-only bundle on one GPU."""
import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import socket
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from c3_safe.teacher_io import file_hash, inside, load_bundle, parse_response, rasterize_response, save_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", required=True, type=Path)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    args = parser.parse_args()
    if args.max_new_tokens < 1:
        parser.error("max-new-tokens must be positive")
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("use a fresh output directory; never overwrite teacher attempts")
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "records").mkdir()
    (args.output / "fields").mkdir()
    save_json(args.output / "RUN_STATUS.json", {"status": "STARTING", "scientific_validation": False})
    started = time.monotonic()
    completed, accepted = 0, 0
    try:
        bundle, requests = load_bundle(args.bundle)
        verified_path = args.model / "MODEL_VERIFIED.json"
        verified = json.loads(verified_path.read_text())
        if verified["status"] != "VERIFIED" or verified["model_id"] != "Qwen/Qwen3-VL-8B-Instruct":
            raise ValueError("requires a verified Qwen3-VL-8B checkpoint")
        for name, item in verified["files"].items():
            path = args.model / name
            if path.stat().st_size != item["size"]:
                raise ValueError(f"checkpoint size changed: {name}")
            if not name.endswith(".safetensors") and file_hash(path) != item["sha256"]:
                raise ValueError(f"checkpoint config changed: {name}")
        source_hashes = {}
        for relative in ("c3_safe/__init__.py", "c3_safe/geometry.py", "c3_safe/teacher_io.py", "scripts/run_c3_teacher_labels.py"):
            target = args.output / "source_snapshot" / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / relative, target)
            source_hashes[relative] = file_hash(target)
        save_json(args.output / "SOURCE_SNAPSHOT.json", source_hashes)
        import numpy as np
        from PIL import Image
        import torch
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
        if not os.environ.get("SLURM_JOB_ID") or not torch.cuda.is_available():
            raise RuntimeError("teacher inference requires a scheduled GPU job")
        torch.set_num_threads(min(4, int(os.environ.get("SLURM_CPUS_PER_TASK", "4"))))
        torch.manual_seed(20260911)
        torch.cuda.set_device(0)
        runtime = {"python": sys.version, "hostname": socket.gethostname(), "slurm_job_id": os.environ["SLURM_JOB_ID"],
                   "gpu": torch.cuda.get_device_name(0), "gpu_bytes": torch.cuda.get_device_properties(0).total_memory,
                   "cuda": torch.version.cuda,
                   "dependencies": {name: importlib.metadata.version(name) for name in
                       ("torch", "torchvision", "transformers", "huggingface-hub", "numpy", "Pillow", "safetensors")}}
        spec = {"protocol_version": "c3-local-teacher-diagnostic-v1", "model_id": verified["model_id"],
                "model_revision": verified["revision"], "model_inventory_sha256": file_hash(verified_path),
                "bundle_manifest_sha256": file_hash(args.bundle / "BUNDLE_MANIFEST.json"),
                "source_dataset_manifest_sha256": bundle["source_dataset_manifest_sha256"],
                "prompt_sha256": bundle["prompt_sha256"], "request_count": len(requests),
                "generation": {"do_sample": False, "max_new_tokens": args.max_new_tokens, "seed": 20260911},
                "dtype": "bfloat16", "attention": "sdpa", "min_pixels": 65536, "max_pixels": 262144,
                "runtime": runtime, "source_files": source_hashes,
                "label_interpretation": "binary property field; confidence separate; unknown has known_mask=false",
                "rasterizer": "normalized-pixel-centers-v1", "provider_calls": 0,
                "scientific_validation": False}
        save_json(args.output / "RUN_SPEC.json", spec)
        print(json.dumps({"stage": "loading_model", **runtime}), flush=True)
        processor = AutoProcessor.from_pretrained(args.model, local_files_only=True, trust_remote_code=False,
                                                 min_pixels=65536, max_pixels=262144)
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            args.model, local_files_only=True, trust_remote_code=False, dtype=torch.bfloat16,
            device_map={"": "cuda:0"}, attn_implementation="sdpa").eval()
        model.generation_config.do_sample = False
        model.generation_config.temperature = None
        model.generation_config.top_p = None
        model.generation_config.top_k = None
        eos = model.generation_config.eos_token_id
        eos = {eos} if isinstance(eos, int) else set(eos or [])
        save_json(args.output / "RUN_STATUS.json", {"status": "RUNNING", "request_count": len(requests)})
        for i, request in enumerate(requests):
            sha = request["image_sha256"]
            record_path = args.output / "records" / f"{sha}.json"
            record = {"schema_version": "c3-real-teacher-record-v1", "index": i,
                      "image_sha256": sha, "prompt_sha256": request["prompt_sha256"],
                      "teacher_model": verified["model_id"], "teacher_revision": verified["revision"],
                      "status": "STARTED", "started_utc": datetime.now(timezone.utc).isoformat()}
            save_json(record_path, record)
            tick = time.monotonic()
            with Image.open(inside(args.bundle, request["image_path"])) as source:
                image = source.convert("RGB")
            # Each call is independent and receives exactly one RGB and the
            # frozen property prompt; no cards, motion, evaluator or history.
            messages = [{"role": "user", "content": [{"type": "image", "image": image},
                        {"type": "text", "text": request["prompt"]}]}]
            inputs = processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                                    return_dict=True, return_tensors="pt")
            record["processor_input_shape"] = {key: list(value.shape) for key, value in inputs.items() if hasattr(value, "shape")}
            record["input_ids"] = inputs["input_ids"][0].tolist()
            if "image_grid_thw" in inputs:
                record["image_grid_thw"] = inputs["image_grid_thw"].tolist()
            inputs = inputs.to("cuda:0")
            with torch.inference_mode():
                output = model.generate(**inputs, do_sample=False, max_new_tokens=args.max_new_tokens)
            generated = output[0, inputs["input_ids"].shape[-1]:].cpu().tolist()
            raw = processor.decode(generated, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            import hashlib
            record.update(raw_response=raw, response_sha256=hashlib.sha256(raw.encode()).hexdigest(),
                          generated_token_ids=generated, generated_tokens=len(generated),
                          latency_seconds=time.monotonic() - tick, width=image.width, height=image.height)
            truncated = len(generated) >= args.max_new_tokens and (not generated or generated[-1] not in eos)
            try:
                if truncated:
                    raise ValueError("generation reached token cap without an end token")
                parsed = parse_response(raw)
                field, known = rasterize_response(parsed, image.width, image.height)
                relative = f"fields/{sha}.npz"
                np.savez_compressed(args.output / relative, requirement_field=field, known_mask=known)
                record.update(status="ACCEPTED", field_path=relative, field_sha256=file_hash(args.output / relative),
                              parsed_response=parsed, known_pixel_fraction=float(known.mean()))
                accepted += 1
            except (ValueError, TypeError, KeyError) as exc:
                record.update(status="INVALID_RESPONSE", parse_error=str(exc), truncated=truncated)
            completed += 1
            save_json(record_path, record)
            print(json.dumps({"image": i + 1, "total": len(requests), "status": record["status"],
                              "seconds": round(record["latency_seconds"], 3), "tokens": len(generated)}), flush=True)
        for relative, sha in source_hashes.items():
            if file_hash(ROOT / relative) != sha:
                raise RuntimeError("source changed during labeling")
        summary = {"status": "REAL_LOCAL_VLM_LABELS_DEVELOPMENT_ONLY", "model_id": verified["model_id"],
                   "revision": verified["revision"], "model_generations": completed, "accepted": accepted,
                   "invalid_responses": completed - accepted, "provider_calls": 0,
                   "elapsed_seconds": time.monotonic() - started,
                   "peak_gpu_allocated_bytes": torch.cuda.max_memory_allocated(),
                   "source_dataset_manifest_sha256": bundle["source_dataset_manifest_sha256"],
                   "human_review": "NOT_PERFORMED", "scientific_gates_passed": []}
        save_json(args.output / "SUMMARY.json", summary)
        save_json(args.output / "RUN_STATUS.json", {"status": "COMPLETE", "scientific_validation": False})
        save_json(args.output / "MANIFEST.json", {"run_spec_sha256": file_hash(args.output / "RUN_SPEC.json"),
                  "artifacts": {str(p.relative_to(args.output)): file_hash(p) for p in sorted(args.output.rglob("*")) if p.is_file()}})
        print(json.dumps(summary), flush=True)
    except BaseException as exc:
        save_json(args.output / "RUN_STATUS.json", {"status": "FAILED", "completed_generations": completed,
                  "accepted": accepted, "error_type": type(exc).__name__, "detail": str(exc)})
        raise


if __name__ == "__main__":
    main()
