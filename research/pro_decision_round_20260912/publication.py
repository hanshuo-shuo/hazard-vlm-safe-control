"""Presentation-only refinement and explicit check of an already registered baseline."""
import argparse
from pathlib import Path
from common import *
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def main(root):
    verify_freeze(root,"E3")
    verify_manifest(root/"final_report")
    p=protocol()
    d=json.loads((root/"final_report/DECISION.json").read_text())
    s=json.loads((root/"e3_report/SUMMARY.json").read_text())
    out=root/"publication"
    out.mkdir(exist_ok=False)
    checks={}
    for domain,scenario in [("iid","iid__source"),("target","target__target_adapted")]:
        rules=[next(v for v in c[domain]["candidate_rules"] if v["threshold"]==.02) for c in s["certificates"]]
        eta=s["scenarios"][scenario]["arms"]["raw_restored"]["eta"]
        checks[domain]={"fixed_threshold":.02,"rules":rules,"all_five_certified":all(v["certified"] for v in rules),
                        "eta":eta,"baseline_screen_pass":all(v["certified"] for v in rules) and eta["ci95"][0]>=.5}
    write_json(out/"FIXED_BASELINE_CHECK.json",{"at_utc":now(),"predeclared_control":True,
        "new_threshold_selected":False,"new_predictions_or_tests":False,"parent_certificate_summary_sha256":sha(root/"e3_report/SUMMARY.json"),
        "checks":checks})
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":10,"axes.spines.top":False,"axes.spines.right":False})
    fig,axes=plt.subplots(1,2,figsize=(10.5,4))
    for j,(metric,title) in enumerate([("conditional_unsafe","Unsafe among accepted"),("eta","Safe opportunity efficiency")]):
        for i,(post,label) in enumerate([("inherited_minus2","Inherited -2"),("restored_zero","Bias restored")]):
            vals=[d["e1"][f"{split}__reference__{post}"][metric] for split in ["random_iid","random_appearance","shape_appearance"]]
            axes[j].plot(range(3),vals,"o-",color=["#c2644b","#237f96"][i],label=label)
        axes[j].set_xticks(range(3),["IID","Appearance","Shape + appearance"])
        axes[j].set_title(title)
        axes[j].set_ylim((-.001,.06) if j==0 else (-.015,1.05))
        axes[j].axhline(.05 if j==0 else .5,color="gray",ls=":",lw=1)
        axes[j].grid(axis="y",alpha=.2)
    axes[0].legend(frameon=False,fontsize=8)
    fig.suptitle("E1: same weights and direct-512 truth; consumed diagnostic scenes")
    fig.tight_layout()
    for ext in ["png","svg"]:
        fig.savefig(out/f"e1_postprocessing.{ext}",dpi=190,bbox_inches="tight")
    plt.close(fig)
    fig,axes=plt.subplots(2,3,figsize=(13.5,7))
    for col,scenario in enumerate(["iid__source","target__source","target__target_adapted"]):
        for row,(metric,label) in enumerate([("conditional_unsafe","Unsafe / accepted"),("eta","Safe opportunity efficiency")]):
            ax=axes[row,col]
            for x,arm in enumerate(p["e3"]["arms"]):
                v=s["scenarios"][scenario]["arms"][arm][metric]
                if v["mean"] is None:
                    ax.text(x,.02,"N/A",ha="center",fontsize=8)
                else:
                    ax.plot(x,v["mean"],"o",color=["#c2644b","#ca9a37","#888888","#237f96","#486b3b"][x])
                    if v["ci95"] is not None:
                        ax.vlines(x,*v["ci95"],color="black",lw=1)
            ax.set_xticks(range(5),["Raw -2","Fixed 0","Pair bound","Selective","Oracle"],rotation=30,ha="right",fontsize=8)
            ax.set_title(scenario.replace("__"," / ").replace("_"," "),fontsize=10)
            ax.set_ylabel(label);ax.set_ylim((-.003,.06) if row==0 else (-.015,1.05))
            ax.axhline(.05 if row==0 else .5,color="gray",ls=":",lw=1);ax.grid(axis="y",alpha=.2)
    fig.suptitle("E3: comparable risk axes; fixed restored-bias baseline also certifies")
    fig.tight_layout()
    for ext in ["png","svg"]:
        fig.savefig(out/f"e3_acceptance.{ext}",dpi=190,bbox_inches="tight")
    plt.close(fig)
    original=(root/"final_report/PAPER_ADDENDUM.md").read_text()
    paragraph="""## The predeclared fixed baseline already suffices

The restored-bias rule at the original fixed acceptance threshold 0.02 is already
one of the 12 registered candidates. All five source certificates and all five
explicit target-adaptation certificates mark this fixed rule valid under the
same 120-hypothesis correction. Its IID test unsafe rate is 1.07%, with eta
96.93% (crossed 95% interval 95.93%-97.82%). The selected-rule method has eta
98.82% and unsafe rate 2.51%; its extra acceptance uses more of the allowed
risk budget. Threshold search is therefore not necessary to pass this round's
screen. This observation reads existing predeclared-control certificates; it
does not introduce a new threshold, certification family or test evaluation.
It strengthens the decision to report a simple repair and existing-method
application instead of claiming a new safety algorithm.

"""
    original=original.replace("](e2_factorial.png)","](../final_report/e2_factorial.png)")
    original=original.replace("## Reproducibility and next decision",paragraph+"## Reproducibility and next decision")
    (out/"PAPER_ADDENDUM.md").write_text(original)
    write_json(out/"PRESENTATION_PROVENANCE.json",{"at_utc":now(),"script_sha256":sha(__file__),
        "parent_report_manifest_sha256":sha(root/"final_report/MANIFEST.json"),
        "changes":"Make the 5% risk line visible on comparable axes, add eta headroom, and expose the already-certified fixed baseline. No numerical experiments changed."})
    write_json(out/"MANIFEST.json",manifest(out))
    print({k:v["baseline_screen_pass"] for k,v in checks.items()},flush=True)

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--root",type=Path,required=True)
    main(parser.parse_args().root)
