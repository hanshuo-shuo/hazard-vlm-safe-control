"""Render and inspect the compiled manuscript on Quest using a pinned renderer."""
import argparse
from pathlib import Path
import json
import re
import hashlib
import pymupdf

def main(build):
    pdf=build/'main.pdf';doc=pymupdf.open(pdf)
    (build/'pages').mkdir(exist_ok=True)
    text=[];outside=[];lengths=[]
    for i,page in enumerate(doc):
        value=page.get_text();text.append(value);lengths.append(len(value))
        page.get_pixmap(dpi=110).save(build/'pages'/f'page-{i+1:02d}.png')
        for word in page.get_text('words'):
            x0,y0,x1,y1=word[:4]
            if x0<0 or y0<0 or x1>page.rect.width or y1>page.rect.height:outside.append([i+1,word])
    combined='\n\f\n'.join(text);(build/'main.txt').write_text(combined)
    hits=re.findall(r'hanshuo|shv7753|/Users/|/projects/|p33100|hazard-vlm-safe-control',combined+json.dumps(doc.metadata),re.I)
    log=(build/'main.log').read_text(errors='replace')
    start=int(re.search(r'FIRST-STATEMENTS-PAGE=(\d+)',log).group(1))
    report=dict(pdf_sha256=hashlib.sha256(pdf.read_bytes()).hexdigest(),pages=len(doc),main_text_pages=start-1,
                first_statements_page=start,page_characters=lengths,metadata=doc.metadata,
                personal_identifier_hits=hits,text_outside_page=outside,
                overfull_boxes=re.findall(r'Overfull[^\n]*',log),undefined_warnings=re.findall(r'[^\n]*undefined[^\n]*',log),
                renderer=pymupdf.VersionBind,visual_review='Pending manual inspection of all rendered PNGs')
    (build/'PDF_QA.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))
    assert not hits and not outside and not report['undefined_warnings']
    assert report['main_text_pages']<=9

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--build',type=Path,required=True);main(ap.parse_args().build)
