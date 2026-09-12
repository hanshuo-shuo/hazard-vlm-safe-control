"""Paper-scale figures (5.5 inches) so labels remain readable in the PDF."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from run import load_npz,write_json,now,sha,manifest

def save(fig,out,name):
    fig.savefig(out/(name+'.png'),dpi=300)
    fig.savefig(out/(name+'.svg'))
    plt.close(fig)

def main(root):
    out=root/'paper_figures';out.mkdir(exist_ok=False)
    p=json.loads((root/'research/same_information_reconstruction_20260912/protocol.json').read_text())
    prior=Path(p['prior_models_root']);old=Path(p['prior_oracle_root'])
    a=load_npz(root/'development/data.npz');d=load_npz(prior/'e3_data/iid_test.npz')
    record=json.loads((prior/'e3_data/iid_test.json').read_text())[0]
    plt.rcParams.update({'font.size':7.5,'axes.titlesize':8,'axes.labelsize':7.5,'xtick.labelsize':7,'ytick.labelsize':7,
                        'legend.fontsize':6.5,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none'})
    fig,axs=plt.subplots(2,3,figsize=(5.5,3.15));colors=['#592b89','#d68a13','#32894d','#ca3358']
    for v in range(2):
        ax=axs[v,0];ax.imshow(d['rgb'][0,v],extent=(-1,1,1,-1))
        for i,pts in enumerate(record['points']):
            pts=np.asarray(pts);ax.plot(pts[:,0],pts[:,1],lw=.8,color=colors[i],label=str(i+1))
        ax.set_xticks([]);ax.set_yticks([]);ax.set_ylabel(f'Version {v+1}',fontsize=7)
        if v==0:ax.legend(loc='upper right',ncol=2,handlelength=.6,columnspacing=.6,borderpad=.25,framealpha=.85)
        ax=axs[v,1];target=a['coarse'][0,0,v];pred=a['coarse'][1,0,v]
        rgb=np.ones((16,16,3))*.96;rgb[...,0]-=.55*target[0]+.08*target[1];rgb[...,1]-=.20*target[0]+.65*target[1];rgb[...,2]-=.06*target[0]+.42*target[1]
        ax.imshow(rgb,extent=(-1,1,1,-1),interpolation='nearest')
        for i in range(4):ax.contour(d['footprints_common'][0,i],levels=[.5],colors=[colors[i]],linewidths=.45,extent=(-1,1,1,-1),origin='upper',alpha=.45)
        for c,color in enumerate(['#005e9e','#b2471a']):ax.contour(pred[c],levels=[.5],colors=[color],linewidths=.9,linestyles='dashed',extent=(-1,1,1,-1),origin='upper')
        ax.set_xticks([]);ax.set_yticks([])
        ax=axs[v,2];x=np.arange(4)
        for c,color in enumerate(['#005e9e','#b2471a']):
            ax.bar(x+(c-.5)*.36,a['truth'][0,v,:,[0,3][c]],width=.32,color=color,alpha=.3,label=['Water reference','Fragile reference'][c])
            ax.plot(x+(c-.5)*.36,a['uniform_e'][1,0,v,:,c],'x',color=color,ms=4,label=['Water predicted','Fragile predicted'][c])
        ax.set_xticks(x,['1','2','3','4']);ax.set_ylim(0,.155);ax.set_yticks([0,.05,.10,.15]);ax.set_ylabel('Exposure')
    for ax,title in zip(axs[0],['RGB + four paths','Coarse targets + contours','Reference and prediction']):ax.set_title(title)
    handles,labels=axs[0,2].get_legend_handles_labels()
    fig.legend(handles,labels,loc='lower center',ncol=2,bbox_to_anchor=(.5,.01),frameon=False,columnspacing=2)
    fig.subplots_adjust(left=.055,right=.995,top=.91,bottom=.19,wspace=.28,hspace=.22)
    save(fig,out,'task_scene')

    disc=json.loads((old/'report/SUMMARY.json').read_text())['splits'];conf=json.loads((old/'confirmation/SUMMARY.json').read_text())['splits']['iid_confirmation']
    banks=[disc['iid_test'],disc['target_test'],conf];labels=['IID-D','App-D','IID-C'];x=np.arange(3)
    fig,axs=plt.subplots(1,2,figsize=(5.5,2.45))
    z=[s['fixed_restored_selection'] for s in banks]
    axs[0].bar(x-.16,[100*v['transitions_vs_reference']['pool512']['false_danger']['mean'] for v in z],width=.28,color='#3f84ae',label='False danger')
    axs[0].bar(x+.16,[100*v['fixed_footprint_numeric_label_change']['mean'] for v in z],width=.28,color='#aeb4bb',label='Numeric check')
    axs[0].plot(x,[100*v['transitions_vs_reference']['pool512']['false_safe']['mean'] for v in z],'x',ms=5,color='#c74646',label='False safe',clip_on=False)
    axs[0].set_title('A. Fixed-action label changes');axs[0].set_ylabel('Model-scene decisions (%)');axs[0].set_ylim(0,5.2);axs[0].set_xticks(x,labels);axs[0].legend(frameon=False,loc='upper right')
    for off,key,color,label in [(-.16,'boundary_small_margin','#91459a','Small margin'),(.16,'boundary_larger_margin','#c3ccd4','Larger margin')]:
        axs[1].bar(x+off,[100*v['strata'][key]['flip_rate']['mean'] for v in z],width=.28,color=color,label=label)
    axs[1].set_title('B. Conditional on boundary overlap');axs[1].set_ylabel('Within-group label changes (%)');axs[1].set_ylim(0,60);axs[1].set_xticks(x,labels);axs[1].legend(frameon=False,loc='upper right')
    fig.subplots_adjust(left=.08,right=.99,top=.85,bottom=.16,wspace=.35)
    save(fig,out,'oracle_confirmation')

    s=json.loads((root/'report/SUMMARY.json').read_text())
    fig,axs=plt.subplots(1,2,figsize=(5.5,2.55))
    for ax,track,title in zip(axs,['target','learned'],['A. Perfect coarse target','B. Five learned fields']):
        arms=s['phases']['appended'][track]['arms'];x=np.arange(3)
        for off,key,color,label in [(-.16,'recovered_safe','#298b62','Recovered safe'),(.16,'new_unsafe_accepts','#c84743','New unsafe')]:
            vs=[arms[arm][key] for arm in ['uniform','shift','plic']];y=np.asarray([v['mean']*100 for v in vs]);ci=np.asarray([v['ci95'] for v in vs]).T*100
            ax.bar(x+off,y,width=.28,color=color,label=label,yerr=np.stack([y-ci[0],ci[1]-y]),capsize=2,error_kw={'lw':.7})
        ax.set_xticks(x,['Uniform','Shift','PLIC']);ax.set_ylim(0,3.3);ax.set_title(title);ax.set_ylabel('Change / all decisions (pp)')
    handles,labels=axs[0].get_legend_handles_labels();fig.legend(handles,labels,loc='lower center',ncol=2,bbox_to_anchor=(.5,.00),frameon=False)
    fig.subplots_adjust(left=.08,right=.99,top=.86,bottom=.25,wspace=.35)
    save(fig,out,'same_information_tradeoff')
    write_json(out/'PROVENANCE.json',dict(at_utc=now(),source_sha256=sha(__file__),physical_width_inches=5.5,
        note='Plotting only; same frozen data. Task bars and contours use stored CPU replay fields from this appended analysis.'))
    write_json(out/'MANIFEST.json',manifest(out))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);main(ap.parse_args().root)
