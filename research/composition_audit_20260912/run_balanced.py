#!/usr/bin/env python3
"""Post-discovery repair diagnostic; preserves the frozen first-stage audit."""
from pathlib import Path
from types import SimpleNamespace
import argparse
import json
import numpy as np
import run_audit as audit

BASE = Path(__file__).resolve().parent
ORIGINAL_TRAIN = audit.train_candidate


def balance(scenes, *, seed):
    # Exactly half the original layouts are swapped; image count and geometry
    # identities stay fixed. This is input/target intervention, not a new loss.
    rng = np.random.default_rng(seed)
    chosen = set(rng.permutation(len(scenes))[:len(scenes) // 2].tolist())
    values = []
    for index, item in enumerate(scenes):
        fields = item.scene.fields[::-1].copy() if index in chosen else item.scene.fields
        values.append(SimpleNamespace(
            rgb=audit.render_requirement_rgb(fields, family=item.family, image_size=64,
                                             render_seed=item.render_seed),
            field_target=audit.lowres_field(fields, 16),
        ))
    assert sum(not np.array_equal(a.rgb, b.rgb) for a, b in zip(scenes, values)) == len(chosen)
    return values


def balanced_train(train, validation, **kwargs):
    return ORIGINAL_TRAIN(balance(train, seed=2026091251),
                          balance(validation, seed=2026091252), **kwargs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed-index", type=int, required=True, choices=range(3))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    protocol_path = BASE / "balanced_repair" / "protocol.json"
    protocol = json.loads(protocol_path.read_text())
    audit.HERE = protocol_path.parent
    audit.train_candidate = balanced_train
    audit.run(args.seed_index, args.output.resolve(), protocol)
    audit.write_json(args.output / "REPAIR_PROVENANCE.json", {
        "wrapper_sha256": audit.sha(__file__), "base_runner_sha256": audit.sha(audit.__file__),
        "selection_train_seed": 2026091251, "selection_validation_seed": 2026091252,
        "note": "Fixed no-RGB controls still use the original reference training distribution. Only the learned visual model receives balanced inputs/targets."
    })
    audit.write_json(args.output / "MANIFEST.json", {
        f.name: audit.sha(f) for f in args.output.iterdir() if f.is_file() and f.name != "MANIFEST.json"
    })
