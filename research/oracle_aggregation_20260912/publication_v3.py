"""Fixed first-scene illustration and discovery figure; run on Quest."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from decompose import HERE, load_npz, write_json, now, sha, manifest
from common import build_candidate, infer_logits, probabilities, exposure, formula
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def main(root):
    p=json.loads((HERE/"protocol.json").read_text());prior=Path(p["prior_root"])
    d=load_npz(prior/"e3_data/iid_test.npz")
    a=load_npz(root/"discovery/iid_test.npz")
    record=json.loads((prior/"e3_data/iid_test.json").read_text())[0]
    model=build_candidate(20261117)
    ckpt=prior/"e2_runs/independent_64/seed_0/model.pt"
    model.load_state_dict(torch.load(ckpt,map_location="cpu",weights_only=True))
    torch.set_num_threads(2)
    pred=probabilities(infer_logits(model,d["rgb"][:1],torch.device("cpu")))
    costs=formula(exposure(pred,d["footprints_common"][:1]))
    difference=float(abs(costs-a["restored_cost"][0,:1]).max())
    assert difference<.001
    out=root/"publication_v3";out.mkdir(exist_ok=False)
    plt.rcParams.update({"font.size":9,"axes.spines.top":False,"axes.spines.right":False,"svg.fonttype":"none"})
    fig,axes=plt.subplots(2,3,figsize=(12.4,7.3))
    colors=["#552e88","#d47a00","#25904e","#d73555"]
    for v in range(2):
        ax=axes[v,0];ax.imshow(d["rgb"][0,v],extent=(-1,1,1,-1))
        for q,pts in enumerate(record["points"]):
            pts=np.asarray(pts);ax.plot(pts[:,0],pts[:,1],color=colors[q],lw=1.6,label=f"a{q+1}")
        ax.set_title(f"Version {v+1}: RGB and four candidate paths");ax.set_xticks([]);ax.set_yticks([])
        ax.legend(loc="upper right",fontsize=7,ncol=2,framealpha=.85)
        ax=axes[v,1]
        # Render the exact four pooled masks in grey behind the target fields.
        target=d["target64"][0,v]
        rgb=np.ones((16,16,3))*.96
        rgb[...,0]-=.55*target[0]+.08*target[1]
        rgb[...,1]-=.20*target[0]+.65*target[1]
        rgb[...,2]-=.06*target[0]+.42*target[1]
        ax.imshow(rgb,extent=(-1,1,1,-1),interpolation="nearest")
        for q in range(4):
            ax.contour(d["footprints_common"][0,q],levels=[.5],colors=[colors[q]],linewidths=.7,extent=(-1,1,1,-1),origin="upper",alpha=.5)
        for c,color in enumerate(["#0065a8","#bb4920"]):
            ax.contour(pred[0,v,c],levels=[.5],colors=[color],linestyles="dashed",linewidths=1.5,extent=(-1,1,1,-1),origin="upper")
        ax.set_title("16×16 targets, footprints; dashed predictions")
        ax.set_xticks([]);ax.set_yticks([])
        ax=axes[v,2]
        x=np.arange(4)
        for c,color in enumerate(["#0065a8","#bb4920"]):
            ax.bar(x+(c-.5)*.34,a["ref_e"][0,v,:,c],width=.31,color=color,alpha=.32,label=["Water true512","Fragile true512"][c])
            ax.plot(x+(c-.5)*.34,a["restored_e"][0,0,v,:,c],"x",color=color,ms=7,label=["Water predicted","Fragile predicted"][c])
        ax.set_xticks(x,[f"a{k+1}" for k in range(4)]);ax.set_ylabel("Mean exposure")
        ax.set_title("Direct reference and stored model exposure")
    handles,labels=axes[0,2].get_legend_handles_labels()
    fig.legend(handles,labels,loc="lower center",ncol=4,bbox_to_anchor=(.5,.09),frameon=False)
    fig.suptitle("From a visual attribute target to action exposure",fontsize=16,y=.98)
    fig.text(.5,.035,"Fixed first IID scene; model seed 0. Swapping properties preserves shapes and paths.\nFootprints are swept areas; all costs use the same 512-derived footprints. No scene selection by outcome.",ha="center",fontsize=9)
    fig.subplots_adjust(left=.06,right=.98,top=.91,bottom=.20,wspace=.3,hspace=.30)
    fig.savefig(out/"task_scene.png",dpi=180);fig.savefig(out/"task_scene.svg");plt.close(fig)
    s=json.loads((root/"report/SUMMARY.json").read_text())
    fig,axes=plt.subplots(1,3,figsize=(12.4,4.7))
    domains=["iid_test","target_test"]
    for di,domain in enumerate(domains):
        v=s["splits"][domain]["fixed_restored_selection"]
        xx=np.arange(3)+(di-.5)*.22
        vals=[v["signed_cost_terms"][k]["signed"] for k in ["model","source","pooling"]]
        mean=np.array([z["mean"] for z in vals]);ci=np.array([z["ci95"] for z in vals]).T
        axes[0].errorbar(xx,mean,yerr=np.stack([mean-ci[0],ci[1]-mean]),fmt="o",capsize=3,label=["IID","New appearance"][di])
        for offset,level in enumerate(["pool512","target64"]):
            z=v["transitions_vs_reference"][level]["different"]
            axes[1].bar(di+(offset-.5)*.32,100*z["mean"],width=.30,color=["#3f87b5","#de8434"][offset],label=["Pooled512 oracle","Target64 oracle"][offset] if di==0 else None,alpha=.8)
        sr=s["splits"][domain]["reselection"]
        y=[100*sr[k]["lost_safe_opportunity"]["mean"] for k in ["pool512","target64","restored"]]
        axes[2].plot(range(3),y,"o-",label=["IID","New appearance"][di])
    axes[0].axhline(0,color="grey",lw=.8);axes[0].set_xticks(range(3),["Model","Source raster","Pooling"])
    axes[0].set_title("A. Signed terms on fixed model action");axes[0].set_ylabel("Mean cost difference");axes[0].legend(frameon=False)
    axes[1].set_xticks(range(2),["IID","New appearance"]);axes[1].set_ylabel("Danger-label changes (%)")
    axes[1].set_title("B. Oracle changes on fixed model action");axes[1].legend(frameon=False);axes[1].set_ylim(0,5.3)
    axes[2].set_xticks(range(3),["Pooled oracle","Target oracle","Model"],rotation=15)
    axes[2].set_ylim(0,4.5);axes[2].set_ylabel("Scenes with lost safe opportunity (%)");axes[2].set_title("C. Each level selects and accepts");axes[2].legend(frameon=False)
    fig.suptitle("Oracle decomposition on consumed data",fontsize=16,y=.98)
    fig.text(.5,.035,"2,000 scenes/domain; five fixed models share those scenes. Restored outputs and fixed threshold .02.\nA: crossed model/scene 95% discovery intervals. B: label difference is not necessarily unsafe acceptance. C: direct512 defines safety.",ha="center",fontsize=9)
    fig.subplots_adjust(left=.065,right=.99,top=.85,bottom=.23,wspace=.32)
    fig.savefig(out/"oracle_discovery.png",dpi=180);fig.savefig(out/"oracle_discovery.svg");plt.close(fig)
    write_json(out/"PROVENANCE.json",{"at_utc":now(),"scene":record["id"],"selection":"first IID scene before inspecting results",
        "checkpoint_sha256":sha(ckpt),"cpu_replay_cost_max_abs":difference,
        "exposure_markers":"stored E3 GPU predictions","contours":"CPU replay probability=.5",
        "script_sha256":sha(__file__)})
    write_json(out/"MANIFEST.json",manifest(out))

if __name__=="__main__":
    ap=argparse.ArgumentParser();ap.add_argument("--root",type=Path,required=True)
    main(ap.parse_args().root)
