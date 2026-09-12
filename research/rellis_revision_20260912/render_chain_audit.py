"""Quest-only standalone figure for development diagnostics, not qualification."""
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path('/projects/p33100/siosio/hazard_rellis_revision_20260912')
sys.path.insert(0, '/projects/p33100/siosio/hazard_independent_confirmation_20260912/report_deps')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    first_path = ROOT/'development_chain/DEVELOPMENT_CHAIN.json'
    motion_path = ROOT/'motion_audit/MOTION_AUDIT.json'
    first, motion = json.loads(first_path.read_text()), json.loads(motion_path.read_text())
    rr = motion['development']
    if len(rr) != 24 or any('centre_motion_correction_displacement_m' not in r for r in rr):
        raise ValueError('Do not silently omit missing development frames')
    dst = ROOT/'publication'
    dst.mkdir(exist_ok=False)
    plt.rcParams.update({'font.size':10, 'axes.titlesize':12, 'axes.labelsize':10,
                         'font.family':'DejaVu Sans', 'svg.fonttype':'none',
                         'axes.spines.top':False, 'axes.spines.right':False})
    fig, axs = plt.subplots(2,2,figsize=(12.4,9.6))
    fig.subplots_adjust(left=.08,right=.98,bottom=.20,top=.84,wspace=.30,hspace=.47)
    x = np.arange(1,25)
    blue, orange, gray = '#1d4ed8', '#c45c13', '#667085'
    lo, hi = np.array([r['point_time_minus_rgb_ms_range'] for r in rr]).T
    axs[0,0].vlines(x,lo,hi,color=blue,lw=3)
    axs[0,0].axhline(0,color=gray,lw=1,ls='--',label='RGB filename timestamp')
    axs[0,0].set(title='A  Point acquisition time\nwithin the central scan',ylabel='Point time minus RGB time (ms)')
    axs[0,0].legend(frameon=False,fontsize=9,loc='upper left')
    q = np.array([[r['centre_motion_correction_displacement_m'][k] for k in ['p50','p95']] for r in rr])
    axs[0,1].plot(x,q[:,0],'-o',ms=3,color=gray,label='Median over points')
    axs[0,1].plot(x,q[:,1],'-o',ms=3,color=blue,label='95th percentile over points')
    axs[0,1].set(title='B  Displacement from the fixed\ndeskew construction',ylabel='Displacement relative to raw coordinates (m)')
    axs[0,1].legend(frameon=False,fontsize=9)
    old = np.array([r['unfitted_scan_residual_under_supplied_poses']['p50_m'] for r in first['development']])
    new = np.array([r['deskewed_unfitted_scan_residual']['p50_m'] for r in rr])
    axs[1,0].plot(x,old,'-o',ms=3,color=gray,label='Rigid scan transform only')
    axs[1,0].plot(x,new,'-o',ms=3,color=blue,label='With per-point deskew')
    axs[1,0].set(title='C  End-scan residual\nwith no additional alignment',ylabel='Median point distance (m)')
    axs[1,0].legend(frameon=False,fontsize=9)
    mean = [r['ouster_gravity_check']['mean_direction_deviation_from_map_up_degrees'] for r in rr]
    p95 = [r['ouster_gravity_check']['instantaneous_deviation_p95_degrees'] for r in rr]
    axs[1,1].plot(x,mean,'-o',ms=3,color=orange,label='Mean acceleration direction')
    axs[1,1].plot(x,p95,'--',color=gray,label='95th percentile of instantaneous angles')
    axs[1,1].set(title='D  Independent Ouster acceleration\nversus hypothesized map up',ylabel='Observed angular difference (degrees)')
    axs[1,1].legend(frameon=False,fontsize=9)
    for ax in axs.flat:
        ax.set_xticks([1,4,8,12,16,20,24])
        ax.set_xlabel('Fixed development frame (1–24)')
        ax.grid(axis='y',alpha=.2)
    fig.suptitle('RELLIS development geometry: observed checks under a pose-time hypothesis\n'
                 'No fresh qualification frames • No measured camera-registration bound',fontsize=14,y=.97)
    fig.text(.03,.087,'B–D condition on the scan-order pose association. C may reuse scans from upstream SLAM; it is not an independent test.\n'
             'D includes vehicle acceleration. Neither scan residuals nor D supplies a physical-error coverage guarantee.',fontsize=10)
    fig.text(.03,.030,'Source: RELLIS-3D — Peng Jiang, Philip Osteen, Maggie Wigness, Srikanth Saripalli (CC BY-NC-SA 3.0).\n'
             'All computation and rendering on Quest. All 24 consumed development frames retained.',fontsize=9,color=gray)
    for suffix in ['png','svg']:
        fig.savefig(dst/f'development_geometry_audit.{suffix}',dpi=180)
    plt.close(fig)
    (dst/'ATTRIBUTION.md').write_text('Derived from RELLIS-3D by Peng Jiang, Philip Osteen, Maggie Wigness and Srikanth Saripalli.\nDataset and derived visualization: CC BY-NC-SA 3.0.\nhttps://github.com/unmannedlab/RELLIS-3D\n\nAll frames are previously consumed development observations. No qualification or mechanism result is implied.\n')
    manifest={'input_sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [first_path,motion_path]},
              'files_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(dst.iterdir()) if p.is_file()}}
    (dst/'MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(manifest,indent=2))


if __name__=='__main__':
    main()
