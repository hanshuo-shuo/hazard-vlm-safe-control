"""Prospective results and compact publication tables; no new inference."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from decompose import now, write_json, sha, manifest

def main(root):
    out=root/"final_publication";out.mkdir(exist_ok=False)
    discovery=json.loads((root/"report/SUMMARY.json").read_text())["splits"]
    confirmation=json.loads((root/"confirmation/SUMMARY.json").read_text())["splits"]["iid_confirmation"]
    h=json.loads((root/"confirmation/HYPOTHESES.json").read_text())
    v=json.loads((root/"confirmation_verification/VERIFICATION.json").read_text())
    sets=[discovery["iid_test"],discovery["target_test"],confirmation]
    labels=["IID\ndiscovery","Appearance\ndiscovery","IID\nconfirmation"]
    plt.rcParams.update({"font.size":10,"axes.spines.top":False,"axes.spines.right":False,"svg.fonttype":"none"})
    fig,axs=plt.subplots(1,3,figsize=(12.7,5.0))
    x=np.arange(3)
    for i,level in enumerate(["pool512","target64"]):
        vals=[s["reselection"][level]["lost_safe_opportunity"] for s in sets]
        y=np.array([100*z["mean"] for z in vals]);ci=np.array([z["ci95"] for z in vals]).T*100
        axs[0].bar(x+(i-.5)*.32,y,width=.30,yerr=np.stack([y-ci[0],ci[1]-y]),capsize=3,
                   color=["#3f87b5","#de8434"][i],label=["Pooled512 oracle","Target64 oracle"][i])
    axs[0].set_title("A. Oracle loses safe opportunities");axs[0].set_ylabel("Fraction of all scenes (%)")
    axs[0].set_xticks(x,labels);axs[0].legend(frameon=False,fontsize=9);axs[0].set_ylim(0,5.4)
    for i,(key,color,label) in enumerate([("pool","#3f87b5","Pooling label change"),("numeric","#777777","Property precision check")]):
        vals=[s["fixed_restored_selection"]["transitions_vs_reference"]["pool512"]["different"] if key=="pool" else
              s["fixed_restored_selection"]["fixed_footprint_numeric_label_change"] for s in sets]
        y=np.array([100*z["mean"] for z in vals]);ci=np.array([z["ci95"] for z in vals]).T*100
        axs[1].bar(x+(i-.5)*.32,y,width=.30,yerr=np.stack([y-ci[0],ci[1]-y]),capsize=3,color=color,label=label)
    axs[1].set_title("B. Fixed model action: changed labels");axs[1].set_ylabel("Model–scene decisions (%)")
    axs[1].set_xticks(x,labels);axs[1].legend(frameon=False,fontsize=9);axs[1].set_ylim(0,5.4)
    for i,(key,color,label) in enumerate([("boundary_small_margin","#984f9a","Small margin ≤ .005"),("boundary_larger_margin","#b9c6d2","Larger margin")]):
        y=[100*s["fixed_restored_selection"]["strata"][key]["flip_rate"]["mean"] for s in sets]
        axs[2].bar(x+(i-.5)*.32,y,width=.30,color=color,label=label)
    axs[2].set_title("C. Boundary overlap and margin");axs[2].set_ylabel("Within-group label changes (%)")
    axs[2].set_xticks(x,labels);axs[2].legend(frameon=False,fontsize=9);axs[2].set_ylim(0,55)
    fig.suptitle("A fixed pooling interface changes usable decisions on a new IID sample",fontsize=15,y=.98)
    fig.text(.5,.035,"2,000 scenes per bank. A: each oracle selects anew; B–C: each restored model's selection is held fixed.\nA–B: descriptive 95% intervals; hypothesis decisions use separately registered one-sided bounds. C: five models share scenes.",ha="center",fontsize=9)
    fig.subplots_adjust(left=.065,right=.99,top=.86,bottom=.23,wspace=.34)
    fig.savefig(out/"oracle_confirmation.png",dpi=180);fig.savefig(out/"oracle_confirmation.svg");plt.close(fig)
    lines=["# Oracle aggregation: discovery and one prospective confirmation","",
           "All figures/statistics computed on Quest. No new training or certification.","",
           "## Each level selects and accepts at .02","",
           "| Bank | Level | Acceptance | Unsafe / accepted | Safe opportunity efficiency | Lost safe opportunity / all scenes |",
           "|---|---|---:|---:|---:|---:|"]
    for name,s in zip(["IID discovery","Appearance discovery","IID confirmation"],sets):
        for level,z in s["reselection"].items():
            lines.append("| "+" | ".join([name,level]+[f"{100*z[k]['mean']:.3f}%" if z[k]['mean'] is not None else "Undefined"
                         for k in ["acceptance","conditional_unsafe","eta","lost_safe_opportunity"]])+" |")
    lines += ["","Zero observed unsafe outcomes are not population risk certificates.","",
              "## Fresh confirmation: registered predictions","",
              f"H1: {h['H1']['events']}/2000 stable-reference lost opportunities; exact one-sided lower {h['H1']['lower']:.8f} > .01.",
              f"H2: {h['H2']['conservative_changes']}/{h['H2']['all_changes']} observed selected-action flips conservative. Original bootstrap lower={h['H2']['lower']:.1f} is degenerate; do not interpret it as population certainty.",
              f"Additional stricter scene check: {v['exact_scene_direction_check']['all_changes_conservative_scenes']}/{v['exact_scene_direction_check']['changed_scenes']} changed scenes have only conservative changes; exact lower={v['exact_scene_direction_check']['exact_lower']:.8f} > .90.",
              f"H3: small-minus-larger margin flip-rate difference={h['H3']['estimate']:.8f}; crossed bootstrap lower={h['H3']['lower']:.8f} > .10.",
              "All three claim-level one-sided alpha values are .05/3. H3 is approximate with five model seeds; the exact direction check conditions on the fixed model bank.",
              "The extra direction check was recorded after job submission but before any fresh outcome was inspected and before HYPOTHESES.json existed; it tightens the claim, with no changed data or original output.",
              "","## Signed costs at fixed restored-model selections","",
              "| Bank | Model | Source raster | Pooling | Total | Total absolute error |","|---|---:|---:|---:|---:|---:|"]
    for name,s in zip(["IID discovery","Appearance discovery","IID confirmation"],sets):
        z=s["fixed_restored_selection"]["signed_cost_terms"]
        lines.append("| "+" | ".join([name]+[f"{z[k]['signed']['mean']:+.9f}" for k in ["model","source","pooling","total"]]+[f"{z['total']['absolute']['mean']:.9f}"])+" |")
    lines += ["","Signed terms telescope; absolute values are not attribution shares. The C cost is recomputed nonlinearly at every exposure level.","",
              "Complete exposure-channel summaries, label-pattern counts, each-model counts, precision checks and covariance-envelope strata are in the JSON reports."]
    (out/"TABLES.md").write_text("\n".join(lines)+"\n")
    write_json(out/"PROVENANCE.json",{"at_utc":now(),"source_sha256":sha(__file__),
        "input_sha256":{str(f.relative_to(root)):sha(f) for f in [root/"report/SUMMARY.json",root/"confirmation/SUMMARY.json",root/"confirmation/HYPOTHESES.json",root/"confirmation_verification/VERIFICATION.json"]}})
    write_json(out/"MANIFEST.json",manifest(out))

if __name__=="__main__":
    ap=argparse.ArgumentParser();ap.add_argument("--root",type=Path,required=True)
    main(ap.parse_args().root)
