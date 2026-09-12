#!/usr/bin/env python3
"""Generate the editable manuscript, evidence figures, and reviewed PDF draft."""
from pathlib import Path
import json
import os
import re
import sys
from html import escape
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
AUDIT = ROOT / "research/composition_audit_20260912"
DATA = ROOT / "results/c3_composition_audit_20260912"
FIG = HERE / "figures"
OUT = ROOT / "output/pdf/compositional_visual_risk_working_paper.pdf"
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "tmp/pdfs/matplotlib"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from reportlab.lib import colors
from reportlab.lib.enums import TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, PageBreak, KeepTogether
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont


def read(path):
    return json.loads(path.read_text())


def fmt(v, n=4):
    return f"{v:.{n}f}"


def pct(v):
    return f"{100*v:.2f}%"


def interval(v, n=5):
    return f"[{v[0]:.{n}f}, {v[1]:.{n}f}]"


def table(headers, rows):
    return "\n".join(["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
                     + ["| " + " | ".join(str(x) for x in row) + " |" for row in rows])


def figure(path, caption):
    return f"![{caption}]({path})\n\n{caption}"


def savefig(fig, name):
    fig.savefig(FIG / f"{name}.png", dpi=220, bbox_inches="tight", facecolor="white")
    fig.savefig(FIG / f"{name}.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def figures(s):
    FIG.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.spines.top": False,
                         "axes.spines.right": False, "pdf.fonttype": 42})
    fig, ax = plt.subplots(figsize=(7.1, 1.6))
    ax.set(xlim=(0, 10), ylim=(0, 2)); ax.axis("off")
    for x, w, label, color in [(0, 2.0, "RGB image\nSpatial field Rθ", "#e5edf6"),
                               (2.6, 2.7, "Candidate footprints F\nPooled exposure ê", "#e3f0ed"),
                               (6.0, 3.9, "Capability κ and rule q\nFormula g  |  learned head h", "#f5ebdb")]:
        ax.add_patch(FancyBboxPatch((x, .45), w, 1.1, boxstyle="round,pad=0.05", facecolor=color, edgecolor="#65758b"))
        ax.text(x+w/2, 1, label, ha="center", va="center", fontsize=9)
    for a, b in [(2.05, 2.55), (5.35, 5.95)]:
        ax.annotate("", xy=(b, 1), xytext=(a, 1), arrowprops={"arrowstyle": "->", "color": "#3b526b"})
    savefig(fig, "composition_interface")

    b = np.load(DATA / "runs/seed_0/PREDICTIONS.npz")
    r = np.load(DATA / "repair_runs/seed_0/PREDICTIONS.npz")
    fig, axes = plt.subplots(2, 4, figsize=(7.1, 3.4))
    def render_field(value):
        water, fragile = value
        return np.clip(np.ones((*water.shape, 3)) * .94 * (1 - np.maximum(water, fragile)[..., None])
                       + water[..., None] * np.array([.12, .43, .78])
                       + fragile[..., None] * np.array([.85, .42, .12]), 0, 1)
    for row, variant in enumerate(("original", "swapped")):
        key = f"fresh_appearance__{variant}"
        image = plt.imread(DATA / "runs/seed_0" / f"{key}__example_0.png")
        values = [image, render_field(b[f"{key}__target"][0]),
                  render_field(b[f"{key}__fields"][0]), render_field(r[f"{key}__fields"][0])]
        for col, value in enumerate(values):
            axes[row, col].imshow(value, interpolation="nearest")
            axes[row, col].set_xticks([]); axes[row, col].set_yticks([])
            for sp in axes[row, col].spines.values(): sp.set_visible(False)
        axes[row, 0].set_ylabel(variant.capitalize(), fontsize=9)
    for ax, label in zip(axes[0], ["RGB input", "Target field", "Original training", "Balanced training"]): ax.set_title(label, fontsize=9)
    fig.text(.5, .015, "Blue: water property     Orange: fragile-surface property     First probe scene; seed 20260912", ha="center", fontsize=8)
    fig.tight_layout(rect=(0, .04, 1, 1), pad=.5)
    savefig(fig, "matched_property_swap")

    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.8))
    splits = ["fresh_iid", "fresh_appearance", "fresh_joint"]
    labels = ["IID", "Appearance\nshift", "Joint\nfamilies"]
    for method, offset, color, label in [("baseline", -.17, "#bd6249", "Original training"),
                                        ("balanced_repair", .17, "#217e83", "Balanced training")]:
        values = [s[method]["splits"][sp]["paired"]["predicted_formula"]["mean"] * 100 for sp in splits]
        yerr = np.asarray([[v - s[method]["splits"][sp]["paired"]["predicted_formula"]["ci95"][0] * 100,
                           s[method]["splits"][sp]["paired"]["predicted_formula"]["ci95"][1] * 100 - v]
                          for sp, v in zip(splits, values)]).T
        axes[0].bar(np.arange(3)+offset, values, width=.32, color=color, label=label, yerr=yerr, capsize=3)
        for i, sp in enumerate(splits):
            seeds = s[method]["splits"][sp]["paired"]["predicted_formula"]["per_seed"]
            axes[0].scatter(i+offset+np.array([-.055, 0, .055]), np.asarray(seeds)*100, color="#202d3c", s=9, zorder=3)
        regrets = [s[method]["splits"][sp]["balanced"]["predicted_formula"]["all"]["regret"]["mean"] for sp in splits]
        axes[1].bar(np.arange(3)+offset, regrets, width=.32, color=color)
    for ax in axes:
        ax.set_xticks(np.arange(3), labels, fontsize=8)
        ax.grid(axis="y", alpha=.2); ax.set_axisbelow(True)
    axes[0].set_ylim(0, 103); axes[0].set_ylabel("Both decisions correct (%)")
    axes[1].set_ylabel("Balanced semantic regret")
    axes[0].legend(loc="upper center", bbox_to_anchor=(1.05, 1.25), ncol=2, frameon=False, fontsize=8)
    fig.tight_layout(pad=1)
    savefig(fig, "balanced_repair")


def compose(s):
    base, repair = s["baseline"]["splits"], s["balanced_repair"]["splits"]
    b, r = base["fresh_appearance"], repair["fresh_appearance"]
    primary = b["comparisons"]["factorized_no_cf_minus_predicted_formula"]
    rprimary = r["comparisons"]["factorized_no_cf_minus_predicted_formula"]
    labels = {"fresh_iid": "IID", "fresh_appearance": "Appearance", "fresh_joint": "Joint families"}
    arm_names = {"cards_only": "Cards only", "role_metadata_diagnostic": "Role metadata†", "mean_field_no_rgb": "Mean field, no RGB",
                 "dense_cost": "Dense cost", "factorized_no_cf": "Factorized, no CF", "factorized_full": "Factorized + CF",
                 "predicted_formula": "Predicted field + formula", "oracle_formula": "Oracle field + formula", "oracle_learned": "Oracle field + learned"}
    context = {
        "base_iou_original": fmt(b["original"]["field"]["field_miou"]),
        "base_iou_swapped": fmt(b["swapped"]["field"]["field_miou"]),
        "repair_pair": pct(r["paired"]["predicted_formula"]["mean"]),
        "abstract_formula": "On the balanced original/intervened appearance probe, both learned relation arms match the analytic arm in regret for all three model seeds.",
        "architecture_figure": figure(FIG / "composition_interface.png", "Figure 1. Equal-perception composition: the learned and analytic alternatives receive identical predicted exposures and cards."),
        "intervention_figure": figure(FIG / "matched_property_swap.png", "Figure 2. A fixed, non-selected example (first appearance probe, first model seed). The target property identities change with the RGB while candidate footprints are held fixed in evaluation. Original training often preserves the wrong property identity at each position; balanced training follows the swap."),
        "repair_figure": figure(FIG / "balanced_repair.png", "Figure 3. Data repair under the predicted-field analytic arm. Left: paired correctness with conditional scene-bootstrap intervals; dots show individual training seeds. Right: regret averaged equally over original and swapped scenes. The repair is adaptive and the probes are already consumed."),
    }
    context["protocol_table"] = table(["Component", "Fixed choice"], [
        ["Training / validation", "800 / 100 procedural images; original reference scene seeds"],
        ["New probes", "3 × 200 scene pairs; 1,200 distinct RGB images"],
        ["Model seeds", "20260912, 20260925, 20260953"],
        ["Training supervision", "Simulator property fields; no VLM labels"],
        ["Visual input / target", "64×64 RGB / 2×16×16 property field"],
        ["Primary comparison", "Learned no-CF minus analytic regret on balanced appearance pairs"],
        ["Repair", "50% property swaps in train and validation; same maximum 24 epochs"],
        ["Status", "Fixed diagnostic, then adaptive repair; neither a formal holdout"]])
    context["audit_results_text"] = (
        f"The three captured-recipe models achieve appearance-probe mIoU {fmt(b['original']['field']['field_miou'])} on original layouts, "
        f"but only {fmt(b['swapped']['field']['field_miou'])} after property exchange. With analytic composition, ordinary regret is "
        f"{fmt(b['original']['arms']['predicted_formula']['all']['regret'],6)} and intervened regret is "
        f"{fmt(b['swapped']['arms']['predicted_formula']['all']['regret'],6)}. Paired correctness on the 400 eligible changed-card pairs is "
        f"{pct(b['paired']['predicted_formula']['mean'])} for each of the three seeds. Table 1 shows the same pattern across all three appearance conditions.")
    context["audit_table"] = table(["Probe", "mIoU original", "mIoU swapped", "Regret original", "Regret swapped", "Pair correct"], [
        [labels[sp], fmt(v["original"]["field"]["field_miou"]), fmt(v["swapped"]["field"]["field_miou"]),
         fmt(v["original"]["arms"]["predicted_formula"]["all"]["regret"],5), fmt(v["swapped"]["arms"]["predicted_formula"]["all"]["regret"],5),
         pct(v["paired"]["predicted_formula"]["mean"])] for sp, v in base.items()]) + "\n\nTable 1. Three-seed means. Decision columns use predicted fields with the public formula; all four cards are included in regret."
    context["composition_results_text"] = (
        f"On the equally weighted original/swapped appearance probe, the predeclared learned-minus-analytic regret contrast is "
        f"{fmt(primary['mean'],7)}, with conditional 95% scene interval {interval(primary['ci95'],7)}. "
        + ("The interval is positive, favoring the analytic arm. " if primary['ci95'][0] > 0 else "The interval is negative, favoring the learned arm. " if primary['ci95'][1] < 0 else "The interval includes zero. ")
        + "The per-seed differences are " + ", ".join(fmt(v,7) for v in primary['per_seed']) + ". "
        + f"The no-CF-minus-CF regret contrast is {fmt(b['comparisons']['factorized_no_cf_minus_factorized_full']['mean'],7)} "
        + f"with interval {interval(b['comparisons']['factorized_no_cf_minus_factorized_full']['ci95'],7)}. The current data do not establish that the additional CF-trained relation head is necessary.")
    context["composition_table"] = table(["Arm", "Balanced regret", "False-safe", "Pair correct"], [
        [arm_names[arm], fmt(b["balanced"][arm]["all"]["regret"]["mean"],6),
         pct(b["balanced"][arm]["all"]["false_safe"]["mean"]), pct(b["paired"][arm]["mean"])] for arm in arm_names]) + "\n\nTable 2. Original training, appearance probe; original/swapped scenes weighted equally. False-safe is scene-macro. †Privileged role-metadata diagnostic, not a neural-input leakage claim."
    delta = s["repair_comparisons"]["fresh_appearance"]
    context["repair_results_text"] = (
        f"Balancing property locations raises appearance-probe paired correctness from {pct(b['paired']['predicted_formula']['mean'])} to "
        f"{pct(r['paired']['predicted_formula']['mean'])}; individual repaired seeds score "
        + ", ".join(pct(v) for v in r['paired']['predicted_formula']['per_seed']) + ". "
        + f"The conditional 95% scene interval is {interval([v*100 for v in r['paired']['predicted_formula']['ci95']],2)} percentage points. "
        + f"Balanced regret falls from {fmt(b['balanced']['predicted_formula']['all']['regret']['mean'],6)} to "
        + f"{fmt(r['balanced']['predicted_formula']['all']['regret']['mean'],6)}; the paired reduction is "
        + f"{fmt(delta['baseline_minus_repair_regret'],6)} with interval {interval(delta['ci95'],6)}.")
    context["repair_table"] = table(["Probe", "mIoU original", "mIoU swapped", "Balanced regret", "False-safe", "Pair correct"], [
        [labels[sp], fmt(v["original"]["field"]["field_miou"]), fmt(v["swapped"]["field"]["field_miou"]),
         fmt(v["balanced"]["predicted_formula"]["all"]["regret"]["mean"],5),
         pct(v["balanced"]["predicted_formula"]["all"]["false_safe"]["mean"]), pct(v["paired"]["predicted_formula"]["mean"])] for sp,v in repair.items()]) + "\n\nTable 3. Adaptive balanced-data repair; same three seeds and probe pairs. Regret and false-safe average original and swapped scenes."
    context["repair_composition_text"] = (
        f"The repair does not improve every metric: original-layout appearance mIoU falls from "
        f"{fmt(b['original']['field']['field_miou'])} to {fmt(r['original']['field']['field_miou'])}, "
        f"and balanced scene-macro false-safe increases from {pct(b['balanced']['predicted_formula']['all']['false_safe']['mean'])} "
        f"to {pct(r['balanced']['predicted_formula']['all']['false_safe']['mean'])}. Thus the paired-decision repair cannot be interpreted as a uniform safety improvement. "
        f"After repair, the appearance-probe learned-minus-analytic regret contrast is {fmt(rprimary['mean'],7)} "
        f"with interval {interval(rprimary['ci95'],7)}. This comparison is secondary and adaptive. "
        f"For the joint-family probe restricted to card D, repaired analytic regret is "
        f"{fmt(repair['fresh_joint']['balanced']['predicted_formula']['D']['regret']['mean'],6)}; "
        "the D-only estimate is saved separately from the all-card paired diagnostic, whose changed optima concern cards A and D.")
    historic_root = ROOT / "results/c3_quest_teacher_20260911/old_quest_runs"
    rows = []
    for name, label, status in [("baseline-confirm-v2", "Historical baseline", "Reference"),
                                 ("ar03-calibrated-polarity-confirm", "Calibrated polarity", "DISCARD"),
                                 ("ar05-runtime-margin-confirm", "Runtime margin", "KEEP"),
                                 ("ar10-cosine-schedule-confirm", "Cosine schedule", "DISCARD")]:
        values = read(historic_root / name / "RESULTS.json")["aggregate"]["splits"]["appearance_ood"]["factorized_no_cf"]
        rows.append([label, fmt(values["field_miou"]), fmt(values["regret"],6), pct(values["false_safe"]), status])
    context["history_table"] = table(["Historical run", "mIoU", "Regret", "False-safe", "Old rule"], rows) + "\n\nTable 4. Consumed historical development distribution, three historical seeds. The old rule includes invariance and runtime guards as well as regret. Not pooled with Tables 1–3."
    context["conclusion_text"] = (
        "The spatial-field interface makes capability- and rule-conditioned risk inspectable, but ordinary appearance-shift scores do not establish correct property grounding. "
        f"Our matched intervention changes a near-zero paired success rate to {pct(r['paired']['predicted_formula']['mean'])} after a controlled location-balancing repair. "
        "The public composition equation supplies a necessary zero-parameter control and a transparent error decomposition. "
        "The strongest supported result is currently the measured failure and repair of the visual component. General semantic-safe control, the necessity of a CF loss, and successful VLM spatial distillation remain unestablished.")
    context["artifact_table"] = table(["Artifact", "Identifier"], [
        ["Original audit", "COMPOSITION-AUDIT-20260912; Slurm array 6161267"],
        ["Adaptive repair", "COMPOSITION-REPAIR-20260912; Slurm array 6161459"],
        ["Original protocol SHA-256", __import__('hashlib').sha256((AUDIT/'protocol.json').read_bytes()).hexdigest()],
        ["Repair protocol SHA-256", __import__('hashlib').sha256((AUDIT/'balanced_repair/protocol.json').read_bytes()).hexdigest()],
        ["Reference recipe", "838f49158b7c227fcd1ef27758c78e123b39d0c018b1613bb9d7415c9bff4844"],
        ["Checkpoints", "6 exported visual models; full hashes in SUMMARY.json"],
        ["Evidence class", "PILOT_ONLY / diagnostic; no formal validation promotion"]])
    template = (HERE / "manuscript.template.md").read_text()
    for key, value in context.items(): template = template.replace("{{"+key+"}}", value)
    assert "{{" not in template, re.findall(r"{{.*?}}", template)
    (HERE / "manuscript.md").write_text(template)
    (HERE / "NUMERIC_CLAIMS.json").write_text(json.dumps(context, indent=2, ensure_ascii=False) + "\n")
    return template


def make_pdf(markdown):
    font_root = Path(os.environ.get("PAPER_FONT_DIR", str(Path.home()/".cache/codex-runtimes/codex-primary-runtime/dependencies/native/libreoffice-headless/libreoffice/LibreOfficeDev.app/Contents/Resources/fonts/truetype")))
    for name, filename in [("Paper", "DejaVuSerif.ttf"), ("Paper-Bold", "DejaVuSerif-Bold.ttf"),
                            ("PaperSans", "DejaVuSans.ttf"), ("PaperSans-Bold", "DejaVuSans-Bold.ttf")]:
        pdfmetrics.registerFont(TTFont(name, str(font_root / filename)))
    pdfmetrics.registerFontFamily("Paper", normal="Paper", bold="Paper-Bold", italic="Paper", boldItalic="Paper-Bold")
    styles = {
        "body": ParagraphStyle("body", fontName="Paper", fontSize=9.25, leading=12.9, spaceAfter=7, alignment=TA_JUSTIFY),
        "title": ParagraphStyle("title", fontName="PaperSans-Bold", fontSize=23, leading=27, spaceAfter=13, textColor=colors.HexColor("#19334d")),
        "h2": ParagraphStyle("h2", fontName="PaperSans-Bold", fontSize=13, leading=17, spaceBefore=8, spaceAfter=8, keepWithNext=True, textColor=colors.HexColor("#19334d")),
        "h3": ParagraphStyle("h3", fontName="PaperSans-Bold", fontSize=10.3, leading=14, spaceBefore=5, spaceAfter=5, keepWithNext=True),
        "caption": ParagraphStyle("caption", fontName="PaperSans", fontSize=8, leading=10.5, spaceAfter=9, textColor=colors.HexColor("#425369")),
        "cell": ParagraphStyle("cell", fontName="PaperSans", fontSize=7.5, leading=10),
        "headcell": ParagraphStyle("headcell", fontName="PaperSans-Bold", fontSize=7.5, leading=10, textColor=colors.white),
    }
    width, height = A4
    usable = width - 104
    def paragraph(value, style="body"):
        text = escape(value)
        # Stable URLs in the reference list are clickable in the generated PDF.
        text = re.sub(r"https://[^\s<]+", lambda m: f'<link href="{m.group(0)}" color="#215b85">{m.group(0)}</link>', text)
        return Paragraph(text, styles[style])
    story = []
    blocks = markdown.split("\n\n")
    for block in blocks:
        block = block.strip()
        if not block: continue
        if block == "<!-- PAGE -->": continue
        if block.startswith("!["):
            match = re.match(r"!\[(.*?)\]\((.*?)\)", block)
            path = match.group(2)
            from PIL import Image as PILImage
            with PILImage.open(path) as im: iw, ih = im.size
            img = Image(path, width=usable, height=usable*ih/iw)
            img.keepWithNext = True
            gap = Spacer(1, 5); gap.keepWithNext = True
            story.extend([img, gap]); continue
        if block.startswith("| "):
            lines = block.splitlines()
            rows = [[cell.strip() for cell in line.strip("|").split("|")] for line in lines if not re.match(r"\|\s*---", line)]
            n = len(rows[0])
            widths = ([usable*.29, usable*.71] if n == 2 else
                      [usable*.4] + [usable*.6/(n-1)]*(n-1) if n == 4 else
                      [usable*.28]+[usable*.72/(n-1)]*(n-1))
            data = [[paragraph(cell, "headcell" if i == 0 else "cell") for cell in row] for i, row in enumerate(rows)]
            t = Table(data, colWidths=widths, repeatRows=1, hAlign="LEFT")
            t.setStyle(TableStyle([("BACKGROUND", (0,0),(-1,0),colors.HexColor("#294860")),
                                  ("ROWBACKGROUNDS", (0,1),(-1,-1),[colors.HexColor("#f0f4f7"),colors.white]),
                                  ("VALIGN",(0,0),(-1,-1),"TOP"), ("TOPPADDING",(0,0),(-1,-1),5),
                                  ("BOTTOMPADDING",(0,0),(-1,-1),5), ("LEFTPADDING",(0,0),(-1,-1),6),
                                  ("LINEBELOW",(0,-1),(-1,-1),.4,colors.HexColor("#ced7df"))]))
            t.keepWithNext = True
            gap = Spacer(1,8); gap.keepWithNext = True
            story.extend([t, gap]); continue
        if block.startswith("### "): story.append(paragraph(block[4:], "h3"))
        elif block.startswith("## "):
            story.append(paragraph(block[3:], "h2"))
        elif block.startswith("# "): story.append(paragraph(block[2:], "title"))
        elif block.startswith(("Figure ", "Table ")): story.append(paragraph(block, "caption"))
        else: story.append(paragraph(block.replace("\n", " ")))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    def page(canvas, doc):
        canvas.saveState()
        canvas.setFont("PaperSans", 7.5)
        canvas.setFillColor(colors.HexColor("#68788b"))
        canvas.drawString(52, height-31, "COMPOSITIONAL VISUAL RISK  /  RESEARCH WORKING PAPER")
        canvas.setStrokeColor(colors.HexColor("#ccd6df")); canvas.line(52, height-38, width-52, height-38)
        canvas.drawString(52, 27, "12 September 2026 · Development diagnostics · Not a closed-loop safety claim")
        canvas.drawRightString(width-52, 27, str(doc.page))
        canvas.restoreState()
    doc = SimpleDocTemplate(str(OUT), pagesize=A4, leftMargin=52, rightMargin=52, topMargin=54, bottomMargin=47,
                            title="Compositional Visual Risk Under Property Interventions", author="C³-Safe research working draft")
    doc.build(story, onFirstPage=page, onLaterPages=page)
    print(json.dumps({"manuscript": str(HERE/'manuscript.md'), "pdf": str(OUT), "figures": str(FIG)}))


if __name__ == "__main__":
    summary = read(AUDIT / "SUMMARY.json")
    figures(summary)
    make_pdf(compose(summary))
