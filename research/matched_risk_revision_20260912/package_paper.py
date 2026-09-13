"""Quest document delivery: exact visual provenance and an isolated ZIP build."""
import argparse
import json
import re
import shutil
import subprocess
import zipfile
from datetime import datetime, timezone
from pathlib import Path
import pymupdf
from analyze import sha, write, HERE

REVIEWED = '78748a2518113aa270f3cd586b359e1a11cb346f5235b7d058e7d54773e4f44f'
PRIOR_REVIEWED = '5efcafbe09538c0e3d0daf97cf205647a5af84626be680ed4ae467acd53dfecb'
FILES = ['main.tex','evidence.tex','comparison.tex','discussion.tex','appendix.tex',
         'case_table.tex','matched_table.tex','references.bib','iclr2027_conference.sty',
         'iclr2027_conference.bst','fancyhdr.sty','natbib.sty','math_commands.tex',
         'build.sh','README.md','figures/task_scene.png','figures/oracle_confirmation.png',
         'figures/same_information_tradeoff.png','figures/matched_risk.png']


def main(root):
    source=root/'paper_build_6207364/source';build=source/'build'
    prior=root/'paper_build_6206817/source/build'
    out=root/'delivery';out.mkdir(exist_ok=False)
    assert sha(build/'main.pdf')==REVIEWED and sha(prior/'main.pdf')==PRIOR_REVIEWED
    qa=json.loads((build/'PDF_QA.json').read_text())
    assert qa['main_text_pages']==9 and qa['pages']==14
    for key in ('personal_identifier_hits','text_outside_page','overfull_boxes','undefined_warnings'):
        assert not qa[key]
    final_doc,previous_doc=pymupdf.open(build/'main.pdf'),pymupdf.open(prior/'main.pdf')
    unchanged=[]
    for i,(a,b) in enumerate(zip(final_doc,previous_doc),1):
        equal=a.get_pixmap(dpi=110).samples==b.get_pixmap(dpi=110).samples
        if i not in (8,12):assert equal, i
        if equal:unchanged.append(i)
    # Prior 14 pages were visually inspected; changed final pages 8 and 12 were
    # inspected again. The assertion above proves remaining final pages identical.
    visual=dict(previous_pdf_sha256=PRIOR_REVIEWED,previous_pages_inspected=list(range(1,15)),
        final_pdf_sha256=REVIEWED,final_changed_pages_inspected=[8,12],
        final_pages_identical_to_reviewed_predecessor=unchanged,
        reviewer='Codex visual inspection; not completed human author review')
    counts={}
    for stage in ('analysis','verification_publication'):
        entries=json.loads((root/stage/'MANIFEST.json').read_text())
        for name,h in entries.items():assert sha(root/stage/name)==h
        counts[stage]=len(entries)
    freeze=json.loads((root/'ANALYSIS_FREEZE.json').read_text())
    for name,h in freeze['files'].items():assert sha(HERE/name)==h
    summary=json.loads((root/'analysis/SUMMARY.json').read_text())
    for name,h in summary['inputs'].items():assert sha(name)==h
    assert summary['continuation_gate']['decision']=='STOP_NO_NEW_SCENES'
    official=Path('/projects/p33100/siosio/hazard_same_information_reconstruction_20260912/paper/iclr2027/official_style.zip')
    template=json.loads((source/'TEMPLATE_SOURCE.json').read_text());assert sha(official)==template['sha256']
    with zipfile.ZipFile(official) as z:
        for name in ('iclr2027_conference.sty','iclr2027_conference.bst'):
            matches=[x for x in z.namelist() if Path(x).name==name and '__MACOSX' not in x]
            assert len(matches)==1 and z.read(matches[0])==(source/name).read_bytes()
    hashes={};bundle=out/'anonymous_manuscript_source.zip'
    pattern=re.compile(rb'hanshuo|shv7753|/Users/|/projects/|p33100|hazard-vlm-safe-control',re.I)
    with zipfile.ZipFile(bundle,'w',compression=zipfile.ZIP_DEFLATED) as z:
        for name in FILES:
            assert not pattern.search((source/name).read_bytes()),name
            hashes[name]=sha(source/name);z.write(source/name,arcname='manuscript_source/'+name)
    trial=out/'isolated_build'
    with zipfile.ZipFile(bundle) as z:z.extractall(trial)
    with (out/'portable_build.log').open('w') as f:
        subprocess.run(['bash','build.sh'],cwd=trial/'manuscript_source',stdout=f,stderr=subprocess.STDOUT,check=True)
    rebuilt=pymupdf.open(trial/'manuscript_source/build/main.pdf');assert len(rebuilt)==14
    for a,b in zip(final_doc,rebuilt):
        assert a.get_text()==b.get_text()
        assert a.get_pixmap(dpi=110).samples==b.get_pixmap(dpi=110).samples
    log=(trial/'manuscript_source/build/main.log').read_text()
    assert 'FIRST-STATEMENTS-PAGE=10' in log and 'Overfull' not in log and 'undefined' not in log
    shutil.copy2(build/'main.pdf',out/'perfect_coarse_predictions_iclr2027.pdf')
    for name in ('PDF_QA.json','main.log','main.txt'):shutil.copy2(build/name,out/name)
    write(out/'DELIVERY.json',dict(at_utc=datetime.now(timezone.utc).isoformat(),status='PASS',
        main_text_pages=9,total_pages=14,visual_review=visual,stage_manifest_entries=counts,
        frozen_analysis_and_input_hashes_unchanged=True,official_style_unchanged=True,
        anonymous_source_scope='Manuscript only; full anonymous experimental supplement and human review pending',
        portable_build_all_14_pages_text_and_raster_identical=True,manuscript_files=hashes,
        delivery_hashes={name:sha(out/name) for name in ('perfect_coarse_predictions_iclr2027.pdf','anonymous_manuscript_source.zip')},
        no_new_scenes=True,no_new_certificates=True,not_submitted=True))
    print((out/'DELIVERY.json').read_text())


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);main(ap.parse_args().root)
