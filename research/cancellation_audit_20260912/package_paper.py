"""Final document and provenance audit for the independently tested interaction."""
import json
import re
import shutil
import subprocess
import zipfile
from datetime import datetime,timezone
from pathlib import Path
import pymupdf
from audit import sha,write,HERE

ROOT=Path('/projects/p33100/siosio/hazard_cancellation_audit_20260912')
REVIEWED='7b79d8eaa6d38f5724212429e60bed8637af992a0f193d1d99f1030cf1d5e165'
FILES=['main.tex','evidence.tex','comparison.tex','discussion.tex','appendix.tex',
       'case_table.tex','matched_table.tex','independent_cells.tex','intervention_results.tex',
       'intervention_abstract.tex','references.bib','iclr2027_conference.sty','iclr2027_conference.bst',
       'fancyhdr.sty','natbib.sty','math_commands.tex','build.sh','README.md',
       'figures/task_scene.png','figures/oracle_confirmation.png','figures/same_information_tradeoff.png',
       'figures/matched_risk.png']


def main():
    out=ROOT/'delivery';out.mkdir(exist_ok=False)
    source=ROOT/'paper_build_6208943/source';build=source/'build'
    assert sha(build/'main.pdf')==REVIEWED
    qa=json.loads((build/'PDF_QA.json').read_text());assert qa['main_text_pages']==9 and qa['pages']==16
    for key in ('personal_identifier_hits','text_outside_page','overfull_boxes','undefined_warnings'):assert not qa[key]
    stages={}
    for stage in ('audit','verification_publication','independent','independent_verification'):
        entries=json.loads((ROOT/stage/'MANIFEST.json').read_text())
        for name,h in entries.items():assert sha(ROOT/stage/name)==h
        stages[stage]=len(entries)
    for name,h in json.loads((ROOT/'AUDIT_FREEZE.json').read_text())['sources'].items():assert sha(HERE/name)==h
    fact=json.loads((ROOT/'FACTORIAL_FREEZE.json').read_text())
    assert sha(HERE/'verify_publish.py')==fact['source_sha256']
    assert sha(HERE/'factorial_addendum.json')==fact['addendum_sha256']
    ind=ROOT/'independent';freeze=json.loads((ind/'FREEZE.json').read_text())
    assert sha(HERE/'independent.py')==freeze['script_sha256']
    assert sha(HERE/'independent_protocol.json')==freeze['protocol_sha256']
    for group in ('imported_project_files','checkpoint_hashes'):
        for name,h in freeze[group].items():assert sha(name)==h
    for name,h in json.loads((ROOT/'audit/SUMMARY.json').read_text())['input_hashes'].items():assert sha(name)==h
    identity=json.loads((ind/'IDENTITY.json').read_text());assert identity['exact_overlap']==0
    baseline=json.loads((ind/'BASELINE_FREEZE.json').read_text());primary=json.loads((ind/'PRIMARY.json').read_text())
    assert freeze['at_utc']<identity['at_utc']<baseline['at_utc']<primary['at_utc']
    assert baseline['plic_computed'] is False
    assert sha(ind/'BASELINE_FLAGS.npz')==baseline['sha256']
    assert primary['decision']=='SUPPORTED_FIXED_INPUT_OPERATOR_INTERACTION'
    official=Path('/projects/p33100/siosio/hazard_same_information_reconstruction_20260912/paper/iclr2027/official_style.zip')
    template=json.loads((source/'TEMPLATE_SOURCE.json').read_text());assert sha(official)==template['sha256']
    with zipfile.ZipFile(official) as z:
        for name in ('iclr2027_conference.sty','iclr2027_conference.bst'):
            matches=[x for x in z.namelist() if Path(x).name==name and '__MACOSX' not in x]
            assert len(matches)==1 and z.read(matches[0])==(source/name).read_bytes()
    pattern=re.compile(rb'hanshuo|shv7753|/Users/|/projects/|p33100|hazard-vlm-safe-control',re.I)
    bundle=out/'anonymous_manuscript_source.zip';hashes={}
    with zipfile.ZipFile(bundle,'w',compression=zipfile.ZIP_DEFLATED) as z:
        for name in FILES:
            assert not pattern.search((source/name).read_bytes()),name
            hashes[name]=sha(source/name);z.write(source/name,arcname='manuscript_source/'+name)
    trial=out/'isolated_build'
    with zipfile.ZipFile(bundle) as z:z.extractall(trial)
    with (out/'portable_build.log').open('w') as f:
        subprocess.run(['bash','build.sh'],cwd=trial/'manuscript_source',stdout=f,stderr=subprocess.STDOUT,check=True)
    original=pymupdf.open(build/'main.pdf');rebuilt=pymupdf.open(trial/'manuscript_source/build/main.pdf')
    assert len(original)==len(rebuilt)==16
    for a,b in zip(original,rebuilt):
        assert a.get_text()==b.get_text()
        assert a.get_pixmap(dpi=110).samples==b.get_pixmap(dpi=110).samples
    log=(trial/'manuscript_source/build/main.log').read_text()
    assert 'FIRST-STATEMENTS-PAGE=10' in log and 'Overfull' not in log and 'undefined' not in log
    shutil.copy2(build/'main.pdf',out/'perfect_coarse_predictions_iclr2027.pdf')
    for name in ('PDF_QA.json','main.txt','main.log'):shutil.copy2(build/name,out/name)
    write(out/'DELIVERY.json',dict(status='PASS',at_utc=datetime.now(timezone.utc).isoformat(),
        main_text_pages=9,total_pages=16,visual_review=dict(pdf_sha256=REVIEWED,pages_inspected=list(range(1,17)),
            reviewer='Codex page-by-page visual inspection, not completed human author review',result='No clipping, overlap or unreadable figure/table content observed'),
        stage_manifest_entries=stages,source_inputs_checkpoints_and_style_unchanged=True,
        freeze_identity_baseline_and_result_order_verified=True,exact_geometry_overlap=0,
        manuscript_files=hashes,portable_build_all_pages_text_and_raster_identical=True,
        delivery_hashes={name:sha(out/name) for name in ('perfect_coarse_predictions_iclr2027.pdf','anonymous_manuscript_source.zip')},
        scope='One independently supported computational interaction. Earlier score-superiority stop unchanged; oracle flag not a deployment predictor. Source ZIP contains manuscript only. No submission or advisor message.'))
    print((out/'DELIVERY.json').read_text())


if __name__=='__main__':main()
