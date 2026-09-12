"""A conditional illustration from the complete preserved event archive."""
import argparse,json
from pathlib import Path

def main(root):
    events=[json.loads(line) for line in (root/'report/EVENTS.jsonl').read_text().splitlines()]
    match=[e for e in events if e['phase']=='appended' and e['track']=='learned' and e['arm']=='plic' and e['new_unsafe_accept']
           and e['uniform_action']==e['arm_action'] and e['baseline_terms_opposed'] and e['fixed_measurement_absolute_error_improves'] and e['fixed_total_absolute_error_worsens']]
    e=sorted(match,key=lambda e:(e['scene_index'],e['model_index']))[0]
    rows=[('Reference cost',e['actual_before'],e['actual_after']),
          ('Target estimate',e['fixed_target_before'],e['fixed_target_after']),
          ('Learned estimate',e['fixed_uniform_estimate'],e['fixed_arm_estimate']),
          ('Target-measurement cost error',e['fixed_measurement_error_before'],e['fixed_measurement_error_after']),
          ('Model-relative-to-target cost error',e['fixed_model_error_before'],e['fixed_model_error_after']),
          ('Total cost error',e['fixed_uniform_estimate']-e['actual_before'],e['fixed_arm_estimate']-e['actual_before'])]
    lines=[r'\begin{table}[t]',r'\centering\small',
        r'\caption{First event satisfying the stated cancellation pattern, in scene/model order. The same reference-unsafe action is rejected before reconstruction and accepted after it ($\tau=.02$). This is a conditional illustration; all 43 newly unsafe learned events remain in the archive.}',
        r'\label{tab:cancellation}',r'\begin{tabular}{lrr}',r'\toprule',r'Quantity on the same action & Uniform & PLIC\\',r'\midrule']
    for label,before,after in rows:lines.append(f'{label} & {before:+.6f} & {after:+.6f}'+r'\\')
    lines += [r'Decision & Reject & Accept (unsafe)\\',r'\bottomrule',r'\end{tabular}',r'\end{table}']
    (root/'paper/iclr2027/case_table.tex').write_text('\n'.join(lines)+'\n')
    (root/'paper/iclr2027/CASE_SOURCE.json').write_text(json.dumps(e,indent=2)+'\n')

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);main(ap.parse_args().root)
