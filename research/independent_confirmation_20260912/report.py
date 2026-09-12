"""Numerical addendum and standard scientific plots, built on Quest."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from common import protocol, write_json, manifest


LABELS = {"legacy": "Original", "balanced": "Left/right balanced", "independent": "Independent geometry"}
COLORS = {"legacy": "#b44b4b", "balanced": "#cc9a32", "independent": "#247a91"}


def main(root):
    p = protocol()
    out = root / "reports"
    s = json.loads((out / "SUMMARY.json").read_text())
    v = json.loads((out / "VERIFICATION.json").read_text())
    if v["status"] != "PASS":
        raise ValueError("Independent verification required before report")
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "pdf.fonttype": 42})
    splits = ["random_iid", "random_appearance", "shape_appearance"]
    names = ["Random geometry\nIID appearance", "Random geometry\nNew appearance", "New shapes\nNew appearance"]
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.1))
    for ax, metric, title in zip(axes, ["regret", "paired_correct", "false_safe_micro"],
                                 ["Decision regret (lower is better)", "Both interventions correct", "False-safe among dangerous entries"]):
        for i, recipe in enumerate(p["recipes"]):
            stats = [s["splits"][split]["recipes"][recipe]["uncertainty"][metric] for split in splits]
            means = np.array([x["mean"] for x in stats])
            # Percentile intervals can lie to one side of a finite-sample point estimate.
            lower = np.array([x["crossed_ci95"][0] for x in stats])
            upper = np.array([x["crossed_ci95"][1] for x in stats])
            x = np.arange(3) + (i - 1) * .19
            ax.plot(x, means, "o", color=COLORS[recipe], label=LABELS[recipe], ms=5)
            ax.vlines(x, lower, upper, color=COLORS[recipe], lw=1.6)
        ax.set_xticks(range(3), names, fontsize=8)
        ax.set_title(title, fontsize=10)
        ax.grid(axis="y", alpha=.2)
        ax.set_ylim(bottom=0)
    axes[1].set_ylim(0, 1.04)
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("Five new training seeds; intervals resample seeds and matched scene pairs", fontsize=12)
    fig.tight_layout()
    for ext in ["png", "pdf"]:
        fig.savefig(out / f"confirmation_results.{ext}", dpi=190, bbox_inches="tight")
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    for ax, split in zip(axes, ["random_iid", "random_appearance"]):
        for recipe in p["recipes"]:
            for name, style in [("raw", "-"), ("calibrated", "--")]:
                curve = s["splits"][split]["recipes"][recipe]["coverage"][name]["curve"]
                points = [x for x in curve if x["conditional_unsafe_rate"] is not None]
                ax.plot([x["coverage"] for x in points], [x["conditional_unsafe_rate"] for x in points],
                        style, marker=".", color=COLORS[recipe], label=f"{LABELS[recipe]} / {name}", lw=1.3)
        ax.set(xlabel="Fraction of decisions accepted", ylabel="Unsafe fraction among accepted decisions",
               title=split.replace("_", " "), xlim=(-.01, 1.01), ylim=(-.01, 1.01))
        ax.grid(alpha=.2)
    axes[0].legend(frameon=False, fontsize=7)
    fig.suptitle("Fixed threshold sweep; undefined risk at zero coverage is omitted", fontsize=12)
    fig.tight_layout()
    for ext in ["png", "pdf"]:
        fig.savefig(out / f"risk_coverage.{ext}", dpi=190, bbox_inches="tight")
    plt.close(fig)

    primary = s["primary"]
    effect = primary["regret_reduction"]
    lines = ["# Prospective follow-up: independent geometry, five seeds, and risk calibration", "",
        "Companion to *Compositional Visual Risk Under Property Interventions*. Executed entirely on Quest on 2026-09-12.", "",
        "## Study status and primary result", "",
        f"The predeclared joint confirmation claim **{'passes' if primary['joint_claim_pass'] else 'does not pass'}**. "
        f"On random geometry with new appearance families, original minus left/right-balanced regret is {effect['mean_left_minus_right']:.6f} "
        f"(crossed seed-and-scene 95% interval [{effect['crossed_ci95'][0]:.6f}, {effect['crossed_ci95'][1]:.6f}]). "
        f"The upper interval endpoint for the balanced model's false-safe increase is {primary['false_safe_increase_upper_ci95']:.4f}; "
        "the preregistered maximum increase was 0.01. The claim required both positive regret reduction and this guard.", "",
        "This is a prospectively frozen procedural component study, following an adaptive pilot. It is not an independently administered "
        "robotics benchmark or a formal C3-Safe gate. Secondary contrasts below are descriptive, without multiplicity correction.", "",
        "## Generator and controls", "",
        "Each random scene has two anonymous, nonoverlapping visible patches. Position, size, rotation, and shape are sampled before "
        "a separate random property assignment. Four exchangeable paths use an independent random stream, common endpoints on opposite "
        "boundary sides, and two random interior waypoints. No rejection condition examines exposure, candidate rank, or desired action. "
        "A separate stream permutes candidate order. Paths can avoid both properties, and target ties remain in the dataset. "
        "The property intervention swaps visible property textures and target channels while retaining geometry, footprints, family, "
        "background noise, and cards. This samples factors independently subject to explicit nonoverlap/visibility constraints; it does "
        "not establish causal independence of learned features.", "",
        "New masks and footprints are rasterized at 64x64 and area-pooled to 16x16; the RGB input is 64x64. Target cost uses mean "
        "exposure on these low-resolution arrays. The separate legacy-regression set retains the original 48x48 generator. Neither "
        "target is a physical-contact or q95 safety measure. The held-out shapes are diamonds and superellipses; training shapes are "
        "ellipses and boxes. The two new appearance families are amber_ice_v1 and rose_moss_v1, using the existing procedural renderer.", "",
        "Three visual recipes use the same captured ResNet-18, loss, augmentation, optimizer, model seeds, 800 training and 100 validation "
        "images, and 24-epoch ceiling. The balanced recipe swaps half the original train and validation images; the independent recipe "
        "changes both training and validation geometry/property distributions. These are data-distribution comparisons, not isolated "
        "loss ablations. Early stopping is allowed by the shared recipe, so actual optimizer steps are recorded separately.", "",
        "The learned relation heads reuse frozen historical checkpoints and receive exactly the formula arm's predicted exposures. "
        "The no-RGB spatial mean is estimated from each recipe's training targets and pooled along the legitimate query footprint. "
        "Cards-only assigns all candidates the same per-card value. Image shuffle uses the next scene's predicted field with the "
        "queried scene's footprints. All visual training is simulator supervised; no teacher call or actor training was performed.", "",
        "## Metrics and sampling units", "",
        "Primary regret and false-safe metrics average the nontrivial cards A/C/D; the zero-cost B card remains in raw arrays but is "
        "excluded from reported coverage and primary aggregates. This differs from the pilot's all-four-card regret. False-safe uses "
        "target cost >0.02 and predicted cost <=0.02. Pair correctness requires a unique optimum separated by at least 0.002 in each "
        "variant, and a changed optimal candidate. Ties in predictions are resolved uniformly in expectation. All unqualified pairs "
        "still contribute to ordinary regret, false-safe, and coverage.", "",
        "Each of the three random tests contains 600 scene pairs; legacy regression contains 200. These are 2,000 pairs and 4,000 "
        "unique test images, evaluated by 15 models, not 60,000 independent scenes. Calibration has 300 separate scene pairs. "
        "The bootstrap independently resamples five paired training seeds and scene pairs 5,000 times, preserving both variants, "
        "all cards/candidates, and recipe pairing. Conditional scene-only intervals and individual seeds are also retained. Five "
        "seeds give limited precision for the distribution of newly trained models.", "",
        "## Main results", "",
        "| Test | Recipe | mIoU | Regret A/C/D | False-safe | Both correct on eligible pairs | Eligible card pairs |",
        "|---|---|---:|---:|---:|---:|---:|"]
    for split in p["test_splits"]:
        for recipe in p["recipes"]:
            r = s["splits"][split]["recipes"][recipe]
            a = r["arms"]["formula"]
            lines.append(f"| {split} | {LABELS[recipe]} | {r['field_miou']:.4f} | {a['regret']:.6f} | {a['false_safe_micro']:.2%} | {a['paired_correct']:.2%} | {a['paired_eligible']:.0f} |")
    lines += ["", "![Results and crossed bootstrap intervals](confirmation_results.png)", "",
        "Eligible counts describe the common scenes per model and are not multiplied by five. Counts and conditional metrics "
        "are retained for every seed in SUMMARY.json. Legacy-regression entries above average original and swapped layouts; "
        "the following table isolates the original layout to expose regression.", "",
        "| Recipe | Original legacy-layout mIoU | Original legacy-layout regret |", "|---|---:|---:|"]
    for recipe in p["recipes"]:
        r = s["splits"]["legacy_regression"]["recipes"][recipe]
        lines.append(f"| {LABELS[recipe]} | {r['original_field_miou']:.4f} | {r['arms']['formula']['original_regret']:.6f} |")
    lines += ["", "## Same-perception composition and nonvisual controls", "",
        "The following values use random_appearance. They assess this known synthetic cost equation, not the necessity of learning "
        "relations for unknown physical interactions.", "",
        "| Recipe | Formula regret | Learned no-CF regret | Learned CF regret | No-RGB mean regret | Shuffled RGB regret | Cards-only regret |",
        "|---|---:|---:|---:|---:|---:|---:|"]
    for recipe in p["recipes"]:
        arms = s["splits"]["random_appearance"]["recipes"][recipe]["arms"]
        lines.append("| " + LABELS[recipe] + " | " + " | ".join(f"{arms[a]['regret']:.6f}" for a in
            ["formula", "learned_no_cf", "learned_cf", "mean_field_no_rgb", "shuffled_rgb", "cards_only"]) + " |")
    lines += ["", "## Calibration and risk-coverage", "",
        "After exporting each checkpoint, we compute one positive underprediction score per calibration pair, maximizing over "
        "both images, all four actions and A/C/D. A fixed 5% level uses order statistic ceil(301*0.95)=286 of 300 scores. "
        "The resulting nonnegative offset is added to each predicted nontrivial cost before clipping to [0,1]. Calibration is "
        "persisted before opening test arrays. This is a split-conformal construction applied to a pair-block maximum score; its "
        "marginal coverage interpretation requires exchangeability with calibration, and is only intended for random_iid. "
        "No coverage guarantee is asserted for appearance shifts, new shapes, or legacy layouts. The construction follows "
        "[Angelopoulos and Bates](https://arxiv.org/html/2107.07511v6); the pair-block score is our application of that framework.", "",
        "A decision is accepted if its minimum estimated upper cost is no more than the displayed threshold. At threshold 0.02, "
        "conditional unsafe rate measures target cost >0.02 among accepted decisions. Zero acceptance produces an undefined "
        "conditional rate, never a zero-risk success. Adding a common offset ordinarily preserves cost ranking; its purpose "
        "is to change acceptance. Clipping-induced ties use the same uniform rule.", "",
        "| Test | Recipe | Raw coverage at .02 | Raw conditional unsafe | Calibrated coverage at .02 | Calibrated conditional unsafe | Pair upper-bound failure |",
        "|---|---|---:|---:|---:|---:|---:|"]
    def rate(value):
        return "undefined (no acceptance)" if value is None else f"{value:.2%}"
    for split in splits:
        for recipe in p["recipes"]:
            r = s["splits"][split]["recipes"][recipe]["coverage"]
            raw = next(x for x in r["raw"]["curve"] if x["threshold"] == .02)
            cal = next(x for x in r["calibrated"]["curve"] if x["threshold"] == .02)
            lines.append(f"| {split} | {LABELS[recipe]} | {raw['coverage']:.2%} | {rate(raw['conditional_unsafe_rate'])} | "
                         f"{cal['coverage']:.2%} | {rate(cal['conditional_unsafe_rate'])} | {r['calibrated']['pair_bound_failure_rate']:.2%} |")
    lines += ["", "![Risk versus coverage](risk_coverage.png)", "",
        "## Per-seed results and provenance", "",
        "| Recipe | Seed | Epochs | Steps | Train seconds | Calibration offset | New-appearance regret | New-appearance false-safe |",
        "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for recipe in p["recipes"]:
        vals = s["splits"]["random_appearance"]["recipes"][recipe]["arms"]["formula"]["per_seed"]
        for t, r in zip(s["training"][recipe], vals):
            lines.append(f"| {LABELS[recipe]} | {t['seed']} | {t['epochs']} | {t['optimizer_steps']} | {t['seconds']:.1f} | {t['offset']:.6f} | {r['regret']:.6f} | {r['false_safe_micro']:.2%} |")
    lines += ["", f"All 15 checkpoints and {s['checked_manifest_files']} manifest-listed data/run files were verified. "
        "An independent implementation reconstructed regret, false-safe counts, paired correctness, formula costs and coverage "
        f"counts for every saved arm. Quest CPU replay checked {v['replayed_images']} images across all checkpoints; "
        "GPU predictions remain the authoritative metric arrays. The preflight failure and retry logs are retained; both "
        "occurred before the prospective freeze and before study data generation.", "",
        f"Quest root: `{root}`. Protocol identity is retained in `FREEZE.json`; training histories, timestamps, calibration scores, "
        "raw fields, cost predictions and per-scene arrays are retained under each recipe/seed. The study does not modify protected "
        "historical sources, substitute simulated labels for VLM labels, or turn static candidate results into closed-loop safety claims.", ""]
    (out / "PAPER_ADDENDUM.md").write_text("\n".join(lines))
    write_json(out / "MANIFEST.json", manifest(out))
    print("Paper addendum and scientific plots written to", out, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    main(parser.parse_args().root)
