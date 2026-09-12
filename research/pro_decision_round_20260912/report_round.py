"""Evidence-driven decision sheet and paper addendum; no test-time tuning."""
import argparse
from pathlib import Path
import numpy as np
from common import *
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def percent(v):
    return "undefined" if v is None else f"{v:.2%}"

def pooled(summaries):
    n=sum(r["n"] for r in summaries)
    a=sum(r["accepted"] for r in summaries)
    u=sum(r["unsafe_accepted"] for r in summaries)
    safe=sum(r["safe_accepted"] for r in summaries)
    oracle=sum(r["oracle_safe"] for r in summaries)
    return {"coverage":a/n,"conditional_unsafe":u/a if a else None,"eta":safe/oracle if oracle else None,
            "oracle_coverage":oracle/n,"accepted":a,"unsafe":u,"safe":safe,"oracle":oracle,"model_decisions":n}

def main(root):
    p=protocol()
    verify_freeze(root,"REPORT")
    e1=json.loads((root/"e1_report/SUMMARY.json").read_text())
    e2=json.loads((root/"e2_report/SUMMARY.json").read_text())
    e3=json.loads((root/"e3_report/SUMMARY.json").read_text())
    out=root/"final_report"
    verification=json.loads((out/"VERIFICATION.json").read_text())
    assert verification["status"]=="PASS"
    e1_table={}
    oracle_exact={}
    for split in ["random_iid","random_appearance","shape_appearance"]:
        d=load_npz(root/"e1_data"/f"{split}.npz")
        oracle_exact[split]={}
        for truth_name,truth in [("low16",d["truth"]),("reference",d["truth_ref512"])]:
            allowed=truth.min(2)[...,CARDS]<=.02
            oracle_exact[split][truth_name]={"safe_decisions":int(allowed.sum()),"total_variant_card_decisions":int(allowed.size),
                "coverage":float(allowed.mean()),"margin_ge_005_coverage":float((truth.min(2)[...,CARDS]<=.015).mean())}
            for post in p["e1"]["postprocessing"]:
                e1_table[f"{split}__{truth_name}__{post}"]=pooled([r["splits"][split][post][truth_name]["raw"] for r in e1["per_seed"]])
    e2_table={}
    for case,runs in e2["per_seed"].items():
        e2_table[case]={}
        for split in ["iid","appearance"]:
            e2_table[case][split]={mode:{"regret":float(np.mean([r["splits"][split]["restored_zero"][mode]["mean_acd_regret"] for r in runs])),
                "false_safe":float(np.mean([r["splits"][split]["restored_zero"][mode]["false_safe"] for r in runs])),
                "deployment":pooled([r["splits"][split]["restored_zero"][mode]["deployment"] for r in runs])}
                for mode in ["common","native"]}
    decisions={"simple_postprocessing_fix":e1["simple_postprocessing_suffices_diagnostically"],
               "measurement_interface_warning":e1["measurement_warning"],"geometry_effect_supported":e2["geometry_primary_pass"],
               "iid_selective_gate_pass":e3["primary_gate_pass"],
               "target_adaptation_gate_pass":e3["scenarios"]["target__target_adapted"]["gate_pass"],
               "new_algorithm_claim_supported":False,"vlm_or_rl_expansion_authorized_by_results":False,
               "research_decision":"Keep a measurement/postprocessing/selection-unit diagnostic study. No novel calibration or public-relation learning claim; require independent-task evidence before further method expansion."}
    if not e3["primary_gate_pass"]:
        decisions["research_decision"]="Stop adding VLM/relation/RL modules after this frozen round; the declared risk/usefulness gate failed. Repair task/measurement or move to independently observed physical outcomes."
    if not e2["geometry_primary_pass"]:
        decisions["geometry_interpretation"]="Do not retain the declared meaningful geometry advantage on both rasters; use intervals to distinguish reversal, equivalence and uncertainty."
    source_test_bounds={}
    for scenario,s in e3["scenarios"].items():
        source_test_bounds[scenario]={arm:[cp_upper(v["unsafe"],v["accepted"],.05) for v in a["per_seed_counts"]]
                                     for arm,a in s["arms"].items()}
    write_json(out/"DECISION.json",{"at_utc":now(),"decisions":decisions,"e1":e1_table,"exact_oracle":oracle_exact,
                                   "e2":e2_table,"e3":e3["scenarios"],"test_pointwise_binomial_upper95":source_test_bounds})
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":10,"axes.spines.top":False,"axes.spines.right":False})
    fig,axes=plt.subplots(1,2,figsize=(10.5,4))
    for j,(metric,title) in enumerate([("conditional_unsafe","Unsafe among accepted"),("eta","Safe opportunity efficiency")]):
        for i,post in enumerate(p["e1"]["postprocessing"]):
            vals=[e1_table[f"{split}__reference__{post}"][metric] for split in ["random_iid","random_appearance","shape_appearance"]]
            axes[j].plot(np.arange(3),vals,"o-",label=post.replace("_"," "),color=["#c2644b","#237f96"][i])
        axes[j].set_xticks(range(3),["IID","Appearance","Shape + appearance"])
        axes[j].set_title(title)
        axes[j].set_ylim(bottom=0)
        axes[j].axhline(.05 if metric=="conditional_unsafe" else .5,color="gray",ls=":",lw=1)
        axes[j].grid(axis="y",alpha=.2)
    axes[0].legend(frameon=False,fontsize=8)
    fig.suptitle("E1: same frozen weights, direct 512 reference, consumed diagnostic scenes")
    fig.tight_layout()
    for ext in ["png","svg"]:
        fig.savefig(out/f"e1_postprocessing.{ext}",dpi=190,bbox_inches="tight")
    plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    for ax,split in zip(axes,["iid","appearance"]):
        for geom,color in [("balanced_anchor","#bd9435"),("independent","#237f96")]:
            for i in range(5):
                vals=[e2["per_seed"][f"{geom}_{r}"][i]["splits"][split]["restored_zero"]["common"]["mean_acd_regret"] for r in [48,64]]
                ax.plot([48,64],vals,"o-",alpha=.55,color=color,label=geom.replace("_"," ") if i==0 else None)
        ax.set(title=split,xlabel="Training supervision source raster",ylabel="A/C/D regret on common 512 truth")
        ax.set_xticks([48,64]);ax.grid(alpha=.2);ax.set_ylim(bottom=0)
    axes[0].legend(frameon=False,fontsize=8)
    fig.suptitle("E2: paired seeds, identical RGB across rasters, exactly 960 updates")
    fig.tight_layout()
    for ext in ["png","svg"]:
        fig.savefig(out/f"e2_factorial.{ext}",dpi=190,bbox_inches="tight")
    plt.close(fig)
    fig,axes=plt.subplots(2,3,figsize=(13.5,7))
    labels=["Raw -2","Raw 0","Pair bound","Selective","Oracle"]
    for col,scenario in enumerate(["iid__source","target__source","target__target_adapted"]):
        for row,(metric,title) in enumerate([("conditional_unsafe","Unsafe / accepted"),("eta","Safe opportunity efficiency")]):
            ax=axes[row,col]
            vals=[e3["scenarios"][scenario]["arms"][arm][metric] for arm in p["e3"]["arms"]]
            for x,v in enumerate(vals):
                if v["mean"] is None:
                    ax.text(x,.02,"N/A",ha="center",fontsize=8,rotation=90)
                    continue
                ax.plot(x,v["mean"],"o",color=["#c2644b","#ca9a37","#888888","#237f96","#486b3b"][x])
                if v["ci95"] is not None:
                    ax.vlines(x,*v["ci95"],color="black",lw=1)
            ax.set_xticks(range(5),labels,rotation=30,ha="right",fontsize=8)
            ax.set_title(scenario.replace("__"," / ").replace("_"," "),fontsize=10)
            ax.set_ylabel(title);ax.set_ylim(bottom=-.015)
            ax.axhline(.05 if row==0 else .5,color="gray",ls=":",lw=1)
            ax.grid(axis="y",alpha=.2)
    fig.suptitle("E3: source certification, shift stress test, and explicitly labeled target adaptation")
    fig.tight_layout()
    for ext in ["png","svg"]:
        fig.savefig(out/f"e3_acceptance.{ext}",dpi=190,bbox_inches="tight")
    plt.close(fig)
    text=["# From visual ranking to safe acceptance: a three-experiment decision round", "",
          "Pro-directed follow-up, executed on Quest on 2026-09-12. This addendum tests measurement and deployment alignment; it proposes no new risk-control algorithm.","",
          "## Decision", "", decisions["research_decision"],"",
          f"The no-training audit's simple-postprocessing flag is **{decisions['simple_postprocessing_fix']}**. "
          f"The matched geometry claim passes its predeclared effect criterion: **{decisions['geometry_effect_supported']}**. "
          f"The IID risk/usefulness screening gate passes: **{decisions['iid_selective_gate_pass']}**.","",
          "The raw restored-bias model is retained as a strong baseline. Any improvement supplied by known exact-binomial selective risk certification is an application of existing methods, not evidence of a new calibration mechanism. No VLM call, relation-head training, actor or critic training was performed.","",
          "## E1: oracle feasibility, postprocessing and calibration scope", "",
          "Five previously frozen independent-geometry checkpoints are reevaluated without training. Inherited minus-two logits and restored-zero logits come from the same forward pass. We keep the old low16 costs and reconstruct direct 256/512 mean-exposure references from saved continuous shapes and paths. This remains spatial mean exposure, not contact or q95. The reference gate requires at most 1% dangerous-label disagreement and 95th-percentile cost difference at most 0.002 between 256 and 512. All diagnostic splits pass.","",
          "The old low16 versus direct-reference danger labels differ in roughly 2.6%-3.3% of entries. This measurement change must not be described as a model improvement or silently pooled with old scores. The exact oracle table enumerates both variants and all A/C/D cards; the acceptance table uses one preregistered deployment draw per scene, shared across models and postprocessing arms.","",
          "| Split | Oracle low16 | Oracle direct 512 | Direct-512 oracle margin >=0.005 |","|---|---:|---:|---:|"]
    for split,v in oracle_exact.items():
        text.append(f"| {split} | {v['low16']['coverage']:.2%} | {v['reference']['coverage']:.2%} | {v['reference']['margin_ge_005_coverage']:.2%} |")
    text += ["","| Split | Postprocessing | Acceptance | Unsafe among accepted | Safe opportunity efficiency |","|---|---|---:|---:|---:|"]
    for split in ["random_iid","random_appearance","shape_appearance"]:
        for post in p["e1"]["postprocessing"]:
            r=e1_table[f"{split}__reference__{post}"]
            text.append(f"| {split} | {post} | {r['coverage']:.2%} | {percent(r['conditional_unsafe'])} | {percent(r['eta'])} |")
    text += ["","![Same weights, changed postprocessing](e1_postprocessing.png)","",
        "Nested residual calibration uses exactly 300 scene scores for each scope: the pair's 24 nontrivial entries, the current image/card's four candidates, or the fixed selected action. The current image/card is sampled once per independent scene; the smaller scopes do not gain sample size by flattening correlated entries. Separate scores are recomputed for each postprocessing and truth convention. Selected-action conformal bounds remain a diagnostic and do not automatically certify conditional unsafe rate.","",
        "| Restored model seed index | Pair q | Current four-candidate q | Selected-action q |","|---|---:|---:|---:|"]
    for i,r in enumerate(e1["per_seed"]):
        q=r["splits"]["calibration"]["restored_zero"]["reference"]["calibrated"]
        text.append(f"| {i} | {q['pair_24_entries']['q']:.6f} | {q['current_four_candidates']['q']:.6f} | {q['fixed_selected_action']['q']:.6f} |")
    text += ["","A one-pixel boundary-band oracle replacement is retained as a privileged mechanism diagnostic, with boundary/interior positive deficits in E1 raw records. It is not a deployment method. E1 uses consumed scenes and supplies diagnostic evidence only.","",
        "## E2: matched geometry by source-raster experiment","",
        "The 2x2 design uses balanced-anchor versus independent geometry and 48 versus 64 source supervision. Each continuous scene, property assignment, path and 64x64 RGB is fixed before source rasterization. Geometry acceptance uses continuous bounding-circle/in-frame checks, not minimum pixel counts or cost/optimum rejection. Both schemes use anonymous independent paths. Exactly half the train/validation property assignments are swapped. Each model trains on variant zero only: 800 images, 24 epochs, batches of 20, exactly 960 optimizer updates, and the final checkpoint. Validation is diagnostic and never selects a checkpoint. All five seeds are paired; 20 models are retained.","",
        "Common scoring uses the same 512-derived query footprints for every arm and direct-512 mean-exposure truth. Native scoring uses each source raster's pooled masks and footprints. The main geometry effect must exceed 0.001 regret reduction at both source rasters, using crossed seed/scene 95% intervals. Raster equivalence requires the entire interval inside +/-0.0005; failure to reject zero is not equivalence.","",
        "| Geometry/source | Common-reference appearance regret | Native-reference appearance regret | Common-reference false-safe |","|---|---:|---:|---:|"]
    for case,v in e2_table.items():
        a=v["appearance"]
        text.append(f"| {case} | {a['common']['regret']:.6f} | {a['native']['regret']:.6f} | {a['common']['false_safe']:.2%} |")
    text += ["","| Appearance contrast | Mean regret difference | Crossed 95% interval |","|---|---:|---|"]
    for name,v in e2["contrasts"]["appearance"].items():
        text.append(f"| {name} | {v['mean']:.6f} | [{v['ci95'][0]:.6f}, {v['ci95'][1]:.6f}] |")
    text += ["","![Matched factorial](e2_factorial.png)","",
        "The geometry scheme still jointly changes position, size, rotation and shape. This matched experiment removes the RGB/source-raster coupling; it does not isolate position as a single causal factor. The balanced-anchor generator is a continuous legacy-like layout with anonymous paths, not an exact rerun of the earlier raster-rejection generator.","",
        "## E3: fixed-selector risk certification","",
        "The five independent/source-64 final checkpoints are chosen a priori. Restored-zero formula scores select an action first; stored independent uniforms resolve ties at 1e-9. Acceptance follows selection, and rejection never triggers reselection. Each independent certification scene draws one of two image variants and one of A/C/D, supplying one Bernoulli outcome. The guarantee concerns that mixture, not each card or appearance subgroup.","",
        "Danger remains true mean-exposure cost >tau=0.02. Epsilon=0.05 is the maximum conditional unsafe rate; delta=0.05 is the family certification failure probability; alpha=0.05 is used separately for residual upper bounds. The development, 1,600-scene calibration and 2,000-scene test banks are disjoint. A finite 12-threshold acceptance bank is frozen before calibration. Exact one-sided binomial bounds use delta/(5 models x 2 calibration domains x 12 rules)=0.05/120. The highest-coverage certified rule is selected; an empty valid set rejects all and is not certified. Certificates are persisted before opening final tests.","",
        "This finite-bank selective-risk construction applies the exact-binomial and multiple-testing ideas in [Learn then Test, Section 3.2](https://arxiv.org/html/2110.01052v5). It is not a new algorithm. A source-certified rule tested on new appearance is a shift stress test. Independently labeled target calibration is explicit adaptation. Neither provides a guarantee on unknown future shifts.","",
        "| Scenario | Arm | Acceptance | Unsafe among accepted | Eta | Eta crossed 95% interval |","|---|---|---:|---:|---:|---|"]
    for scenario,v in e3["scenarios"].items():
        for arm,a in v["arms"].items():
            interval=a["eta"]["ci95"]
            text.append(f"| {scenario} | {arm} | {percent(a['coverage']['mean'])} | {percent(a['conditional_unsafe']['mean'])} | {percent(a['eta']['mean'])} | [{interval[0]:.4f}, {interval[1]:.4f}] |")
    text += ["","![Risk and safe acceptance](e3_acceptance.png)","",
        "Eta counts accepted AND actually safe decisions divided by oracle-safe opportunities; unsafe accepts never contribute. Zero coverage has undefined conditional risk. Crossed intervals resample model seeds and independent scenes, conditional on the fixed calibration banks. Exact calibration certificates, rather than a potentially degenerate zero-error bootstrap interval, control conditional risk. Pointwise exact-binomial test upper bounds and raw counts are retained in DECISION.json.","",
        "| Seed | Calibration domain | Chosen threshold | Accepted calibration samples | Unsafe | Simultaneous upper bound |","|---|---|---:|---:|---:|---:|"]
    for i,c in enumerate(e3["certificates"]):
        for domain,cert in c.items():
            r=cert["chosen"]
            text.append(f"| {p['model_seeds'][i]} | {domain} | " + (f"{r['threshold']:.3f} | {r['accepted']} | {r['unsafe_accepted']} | {r['upper']:.4f} |" if r else "none | 0 | 0 | not certified |"))
    text += ["","## Reproducibility and next decision","",
        f"All three source freezes, 20 new checkpoints, five old checkpoint identities, dataset/certificate hashes and temporal load records are preserved under `{root}`. "
        f"Independent verification checked {verification['verified_manifest_files']} manifest-listed files and 80 CPU replay images across 20 new models. "
        "It reconstructed scalar choices, mean exposures, regrets, false-safe/acceptance counts and certificate p-values. GPU predictions remain authoritative. Numerical reference precision is an empirical raster-convergence check, not an exact continuous integration theorem.","",
        "The screening thresholds are research decisions, not robot safety standards. Static certification says nothing by itself about episode risk, task completion, dynamics, occlusion or physical damage. The planned round ends here: no additional test-set tuning, grid/checkpoint search, relation/VLM/RL expansion or claimed novel calibration. If pursued as a paper, the defensible scope is measurement and decision-object alignment with simple postprocessing controls; independent-task or physical-outcome evidence is still required for broader claims.",""]
    (out/"PAPER_ADDENDUM.md").write_text("\n".join(text))
    write_json(out/"MANIFEST.json",manifest(out))
    print(json.dumps(decisions,indent=2),flush=True)

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--root",type=Path,required=True)
    main(parser.parse_args().root)
