"""Render the completed qualification audit on Quest; no new scientific rule."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True, type=Path)
    root = p.parse_args().root
    audit = json.loads((root/"geometry_audit/GEOMETRY_AUDIT.json").read_text())
    verification = json.loads((root/"artifacts/QUALIFICATION_VERIFICATION.json").read_text())
    assert verification["status"] == "PASS"
    out = root/"publication"
    out.mkdir(exist_ok=False)
    unknown = np.array([[c["unknown_fraction"] for c in r["candidates"]] for r in audit["records"]])
    plt.rcParams.update({"font.size":11,"axes.spines.top":False,"axes.spines.right":False})
    fig, ax = plt.subplots(1,2,figsize=(12,8),gridspec_kw={"width_ratios":[1,1.3]})
    im = ax[0].imshow(unknown*100,aspect="auto",vmin=0,vmax=100,cmap="magma")
    ax[0].set_xticks(range(4),["-12°","-4°","4°","12°"])
    ax[0].set_yticks(range(24),[f"{i:02d}" for i in range(24)])
    ax[0].set(xlabel="Fixed corridor heading",ylabel="Preselected pilot index",title="Unknown area in the current construction")
    fig.colorbar(im,ax=ax[0],label="Unknown area (%)",fraction=.045,pad=.04)
    counts = [0,verification["optimistically_all_four_evaluable_scenes"],24*.8]
    ax[1].barh([2,1,0],counts,color=["#c65e4a","#829da6","#dedede"],height=.5)
    ax[1].set_yticks([2,1,0],["Registered screen","Optimistic support ceiling*","Required 80% of pilot"])
    for y,value in zip([2,1,0],counts):
        ax[1].text(value+.4,y,f"{value:g} / 24",va="center")
    ax[1].set(xlim=(0,25),xlabel="Scenes with all four corridors evaluable",title="Qualification does not pass")
    ax[1].text(0,-.85,"*Drops depth and occlusion checks.\nA diagnostic ceiling, not an acceptance rule.\n\nUnknown is never counted as safe.\nThis evaluates the current geometry interface,\nnot the suitability of RELLIS-3D in general.",fontsize=10,va="top")
    ax[1].set_ylim(-2.0,2.65)
    fig.suptitle("RELLIS-3D: fixed 24-scene geometry qualification before model training",fontsize=14)
    fig.tight_layout()
    fig.savefig(out/"qualification_overview.png",dpi=180)
    fig.savefig(out/"qualification_overview.svg")
    plt.close(fig)
    fig, axes = plt.subplots(8,3,figsize=(15,26))
    for i,axis in enumerate(axes.flat):
        axis.imshow(mpimg.imread(root/f"geometry_audit/pilot_{i:02d}.png"))
        axis.set_title(f"Pilot {i:02d}: " + " / ".join(f"{x:.0%}" for x in unknown[i]) + " unknown",fontsize=10,color="black")
        axis.axis("off")
    fig.suptitle("All fixed pilot scenes, in registered order | green: supported; red: unknown\nRELLIS-3D: Jiang, Osteen, Wigness & Saripalli | CC BY-NC-SA 3.0",fontsize=14)
    fig.tight_layout(rect=(0,0,1,.97))
    fig.savefig(out/"pilot_contact_sheet.png",dpi=140)
    plt.close(fig)
    (out/"ATTRIBUTION.md").write_text("# Attribution\n\nThe pilot contact sheet contains derived RELLIS-3D images by Peng Jiang, Philip Osteen, Maggie Wigness and Srikanth Saripalli. Data and derived image content are CC BY-NC-SA 3.0: https://creativecommons.org/licenses/by-nc-sa/3.0/ . Source: https://github.com/unmannedlab/RELLIS-3D . Modifications are metric sample overlays, resizing and captions. The images are a fixed training-list pilot, not an external test set.\n")
    manifest = {p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in out.iterdir() if p.is_file()}
    manifest["../research/rellis_external_20260912/render_qualification.py"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    (out/"MANIFEST.json").write_text(json.dumps(manifest,indent=2)+"\n")


if __name__ == "__main__":
    main()
