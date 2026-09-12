"""Show recovery and new danger together; preserve the full development shift bank."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from run import write_json,now,sha,manifest

def main(root):
    s=json.loads((root/'report/SUMMARY.json').read_text());dev=json.loads((root/'OFFSET_DEVELOPMENT.json').read_text())
    out=root/'publication';out.mkdir(exist_ok=False)
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none'})
    fig,axes=plt.subplots(1,2,figsize=(10.4,4.5))
    for ax,track,title in zip(axes,['target','learned'],['A. Perfect target64 input','B. Five restored model inputs']):
        arms=s['phases']['appended'][track]['arms'];x=np.arange(3)
        for off,key,color,label in [(-.18,'recovered_safe','#298b62','Recovered safe opportunity'),(.18,'new_unsafe_accepts','#c84743','New unsafe acceptance')]:
            z=[arms[arm][key] for arm in ['uniform','shift','plic']]
            y=np.asarray([v['mean']*100 for v in z]);ci=np.asarray([v['ci95'] for v in z]).T*100
            ax.bar(x+off,y,.32,color=color,label=label,yerr=np.stack([y-ci[0],ci[1]-y]),capsize=3)
            for xx,yy in zip(x+off,y):ax.text(xx,yy+.08,f'{yy:.2f}',ha='center',fontsize=9)
        ax.set_xticks(x,['Uniform','Global shift','PLIC']);ax.set_ylim(0,3.5)
        ax.set_title(title);ax.set_ylabel('Change relative to uniform (percentage points)')
    handles,labels=axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='lower center',ncol=2,bbox_to_anchor=(.5,.095),frameon=False)
    fig.suptitle('Same coarse information: opportunity recovery comes with new unsafe accepts',fontsize=13,y=.98)
    fig.text(.5,.03,'Appended analysis on 2,000 consumed scenes. No newly lost safe accepts were observed in either track.\nFive model rows share scenes; intervals are descriptive paired/crossed 95% bootstrap intervals. No inherited certificate.',ha='center',fontsize=8)
    fig.subplots_adjust(left=.075,right=.99,top=.86,bottom=.27,wspace=.32)
    fig.savefig(out/'same_information_tradeoff.png',dpi=200);fig.savefig(out/'same_information_tradeoff.svg');plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(10.4,4.7))
    for ax,track,title in zip(axes,['target','learned'],['Perfect target64 development','Five-model development']):
        arms=s['phases']['development'][track]['arms'];bank=dev[track]['candidates'];n=2000 if track=='target' else 10000
        den=arms['uniform']['eta']['denominator']
        x=np.asarray([100*b['unsafe_accepted']/n for b in bank]);y=np.asarray([100*b['safe_accepted']/den for b in bank])
        ax.plot(x,y,'o-',color='#64798e',ms=4,label='Entire fixed offset bank')
        selected=dev[track]['selected'];sx=100*selected['unsafe_accepted']/n;sy=100*selected['safe_accepted']/den
        ax.scatter([sx],[sy],marker='s',s=70,color='#d48225',label='Frozen offset')
        pr=arms['plic'];ax.scatter([100*pr['conditional_unsafe']['numerator']/n],[100*pr['eta']['mean']],marker='*',s=150,color='#703fa0',label='Fixed PLIC')
        ax.set_title(title);ax.set_xlabel('Unsafe accepts / all decisions (%)');ax.set_ylabel('Safe opportunity efficiency (%)')
        ax.set_ylim(80,101);ax.legend(fontsize=8,frameon=False,loc='lower right')
    fig.suptitle('Development tradeoff: larger global shifts can recover more at higher risk',fontsize=13,y=.98)
    fig.text(.5,.03,'Development only; displayed offset bank was fixed before fitting. Lower-efficiency points may fall below the axis.\nThe chosen offset forbids an increase in observed development unsafe counts; PLIC has no such fitted constraint. Not a matched-risk test.',ha='center',fontsize=8)
    fig.subplots_adjust(left=.075,right=.99,top=.86,bottom=.23,wspace=.32)
    fig.savefig(out/'development_shift_tradeoff.png',dpi=200);fig.savefig(out/'development_shift_tradeoff.svg');plt.close(fig)
    rows=['# Same-information appended baseline results','','All numbers refer to the already consumed 2,000-scene confirmation bank; not new confirmation.','',
          '| Track | Arm | Accepted | Unsafe accepted | Safe accepted | Recovered | Newly lost | New unsafe | Eta |',
          '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for track,t in s['phases']['appended'].items():
        for arm,a in t['arms'].items():
            rows.append('| '+' | '.join([track,arm,str(int(a['acceptance']['numerator'])),str(int(a['conditional_unsafe']['numerator'])),str(int(a['eta']['numerator'])),str(int(a['recovered_safe']['numerator'])),str(int(a['newly_lost_safe']['numerator'])),str(int(a['new_unsafe_accepts']['numerator'])),f"{100*a['eta']['mean']:.4f}%"] )+' |')
    rows+=['','Target counts are over 2,000 scenes. Learned counts sum five models over those same scenes, not 10,000 independent trials.',
           'Global shift: target b=.001, learned b=0. It was selected with no increase in development unsafe count. PLIC introduces additional danger; recovery alone is not overall superiority.',
           'All new and resolved events and their exact signed decompositions are retained in EVENTS.jsonl.']
    (out/'TABLES.md').write_text('\n'.join(rows)+'\n')
    write_json(out/'PROVENANCE.json',dict(at_utc=now(),source_sha256=sha(__file__),input_sha256={str(f.relative_to(root)):sha(f) for f in [root/'report/SUMMARY.json',root/'OFFSET_DEVELOPMENT.json']}))
    write_json(out/'MANIFEST.json',manifest(out))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);main(ap.parse_args().root)
