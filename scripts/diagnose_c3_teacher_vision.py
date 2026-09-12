#!/usr/bin/env python3
"""Five action-free visual input checks, separate from frozen teacher labels."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import socket
import time


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, data):
    Path(path).write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bundle", type=Path, required=True)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        p.error("use a fresh diagnostic directory")
    args.output.mkdir(parents=True, exist_ok=True)
    write(args.output / "RUN_STATUS.json", {"status": "STARTING"})
    try:
        import numpy as np
        from PIL import Image
        import torch
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
        if not os.environ.get("SLURM_JOB_ID") or not torch.cuda.is_available():
            raise RuntimeError("requires a scheduled GPU")
        requests = json.loads((args.bundle / "REQUESTS.json").read_text())
        if len(requests) != 5:
            raise ValueError("this diagnostic is frozen to five requests")
        inventory = json.loads((args.model / "MODEL_VERIFIED.json").read_text())
        if inventory["status"] != "VERIFIED":
            raise ValueError("unverified model")
        source = Path(__file__).resolve()
        shutil.copyfile(source, args.output / source.name)
        torch.set_num_threads(4)
        torch.manual_seed(20260911)
        processor = AutoProcessor.from_pretrained(args.model, local_files_only=True, trust_remote_code=False,
                                                 min_pixels=65536, max_pixels=262144)
        model = Qwen3VLForConditionalGeneration.from_pretrained(args.model, local_files_only=True, trust_remote_code=False,
                        dtype=torch.bfloat16, device_map={"": "cuda:0"}, attn_implementation="sdpa").eval()
        model.generation_config.temperature = None
        model.generation_config.top_p = None
        model.generation_config.top_k = None
        results = []
        for request in requests:
            image_path = (args.bundle / request["image_path"]).resolve()
            if not image_path.is_relative_to(args.bundle.resolve()) or sha(image_path) != request["image_sha256"]:
                raise ValueError("diagnostic image identity mismatch")
            with Image.open(image_path) as original:
                image = original.convert("RGB")
            messages = [{"role": "user", "content": [{"type": "image", "image": image},
                        {"type": "text", "text": request["prompt"]}]}]
            inputs = processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                                    return_dict=True, return_tensors="pt")
            pixels = inputs["pixel_values"].float().numpy()
            record = {**request, "processor_pixel_sha256": hashlib.sha256(pixels.tobytes()).hexdigest(),
                      "processor_pixel_std": float(pixels.std()), "image_grid_thw": inputs["image_grid_thw"].tolist(),
                      "source_mean_rgb": np.asarray(image).mean(axis=(0,1)).tolist()}
            inputs = inputs.to("cuda:0")
            tick = time.monotonic()
            with torch.inference_mode():
                generated = model.generate(**inputs, do_sample=False, max_new_tokens=256)
            tokens = generated[0, inputs["input_ids"].shape[-1]:].cpu().tolist()
            raw = processor.decode(tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            record.update(raw_response=raw, response_sha256=hashlib.sha256(raw.encode()).hexdigest(),
                          generated_tokens=tokens, seconds=time.monotonic()-tick)
            results.append(record)
            write(args.output / "RESPONSES.json", results)
            print(json.dumps({"case": request["case"], "response": raw}), flush=True)
        summary = {"status": "VISION_INPUT_DIAGNOSTIC_NOT_TEACHER_LABELS", "model_generations": len(results),
                   "model_id": inventory["model_id"], "revision": inventory["revision"],
                   "gpu": torch.cuda.get_device_name(0), "hostname": socket.gethostname(),
                   "slurm_job_id": os.environ["SLURM_JOB_ID"], "provider_calls": 0,
                   "request_sha256": sha(args.bundle / "REQUESTS.json"), "source_sha256": sha(source),
                   "scientific_gates_passed": []}
        write(args.output / "SUMMARY.json", summary)
        write(args.output / "RUN_STATUS.json", {"status": "COMPLETE"})
        write(args.output / "MANIFEST.json", {"artifacts": {p.name: sha(p) for p in args.output.iterdir() if p.is_file()}})
    except BaseException as exc:
        write(args.output / "RUN_STATUS.json", {"status": "FAILED", "error_type": type(exc).__name__, "detail": str(exc)})
        raise


if __name__ == "__main__":
    main()
