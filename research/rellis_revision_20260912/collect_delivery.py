"""Create a compact auditable research delivery on Quest; no new science."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import time

ROOT = Path('/projects/p33100/siosio/hazard_rellis_revision_20260912')
OLD = Path('/projects/p33100/siosio/hazard_rellis_external_20260912')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stamp(ns):
    return f'{ns//10**9}.{ns%10**9:09}'


def main():
    verification = json.loads((ROOT/'artifacts/REVISION_VERIFICATION.json').read_text())
    assert verification['status']=='PASS'
    dst = ROOT/'delivery'
    dst.mkdir(exist_ok=False)
    for source, name in [
        ('artifacts/REVISION_DECISION.json','REVISION_DECISION.json'),
        ('artifacts/REVISION_VERIFICATION.json','REVISION_VERIFICATION.json'),
        ('artifacts/REVISION_VERIFICATION_initial_6191612.json','REVISION_VERIFICATION_initial_6191612.json'),
        ('artifacts/DECODE_DIAGNOSIS.json','DECODE_DIAGNOSIS.json'),
        ('development_chain/DEVELOPMENT_CHAIN.json','DEVELOPMENT_CHAIN.json'),
        ('motion_audit/MOTION_AUDIT.json','MOTION_AUDIT.json'),
        ('development_sensors/SUMMARY.json','SENSOR_ACQUISITION_SUMMARY.json'),
        ('sensor_metadata/SUMMARY.json','SENSOR_INDEX_SUMMARY.json'),
        ('source_audit/SOURCE_AUDIT.json','SOURCE_AUDIT.json'),
        ('artifacts/SLURM_ACCOUNTING.txt','SLURM_ACCOUNTING.txt'),
    ]:
        shutil.copyfile(ROOT/source,dst/name)
    shutil.copytree(ROOT/'publication',dst/'publication')
    (dst/'freezes').mkdir()
    for path in sorted((ROOT/'artifacts').glob('FREEZE_*.sha256')):
        shutil.copyfile(path,dst/'freezes'/path.name)
    metadata=json.loads((ROOT/'sensor_metadata/POSE_SCAN_METADATA.json').read_text())
    for item in metadata['sequences'].values():
        records=item.pop('records')
        item['first_scan']=records[0]
        item['last_scan']=records[-1]
        item['full_records_on_quest']='sensor_metadata/POSE_SCAN_METADATA.json'
    (dst/'POSE_SCAN_METADATA_SUMMARY.json').write_text(json.dumps(metadata,indent=2)+'\n')
    sources={
        'RELLIS': {'commit':'c17a118fcaed1559f03cc32cc3a91dedc557f8b8',
                   'url':'https://github.com/unmannedlab/RELLIS-3D/tree/c17a118fcaed1559f03cc32cc3a91dedc557f8b8'},
        'Cartographer': {'commit':'27c15c92fa43ab221508add8597bdc0517a0cccb',
                   'url':'https://github.com/unmannedlab/cartographer/tree/27c15c92fa43ab221508add8597bdc0517a0cccb'},
        'Ouster_history_snapshot': {'commit':'4a0cc273d1711421f24da2495d90143fd7684883',
                   'url':'https://github.com/ouster-lidar/ouster-sdk/tree/4a0cc273d1711421f24da2495d90143fd7684883',
                   'scope':'Latest upstream commit found before acquisition date, not a proven installed driver build'},
        'file_sha256': {}}
    paths=[
        ROOT/'official_cartographer/src/cartographer_ros/cartographer_ros/cartographer_ros/output_pose.cc',
        ROOT/'official_cartographer/src/cartographer_ros/cartographer_ros/cartographer_ros/msg_conversion.cc',
        ROOT/'official_cartographer/cart_config/launch/offline_warthog_3d_ouster.launch',
        ROOT/'official_cartographer/cart_config/config/output_pose.lua',
        ROOT/'official_ouster_snapshot/ouster_client/include/ouster/os1_util.h',
        ROOT/'official_ouster_snapshot/ouster_ros/src/os1_cloud_node.cpp',
        ROOT/'source_audit/raw_folder.html',
        OLD/'official_rellis/catkin_ws/src/platform_description/urdf/warthog.urdf',
        OLD/'metadata/extrinsics/Rellis_3D/00000/transforms.yaml',
        OLD/'metadata/intrinsics/Rellis-3D/00000/camera_info.txt']
    for p in paths:
        sources['file_sha256'][str(p)]=sha(p)
    (dst/'UPSTREAM_PROVENANCE.json').write_text(json.dumps(sources,indent=2)+'\n')
    chain=json.loads((ROOT/'development_chain/DEVELOPMENT_CHAIN.json').read_text())
    lines=['# Fixed-development time and coordinate correspondence', '',
           'All 24 centres are consumed development data. A pose row remains a scan-order hypothesis,',
           'supported by source code and consistency checks but without a release-specific timestamped export.',
           'All central raw x/y/z/t/ring arrays match the timestamped PLY. No fresh qualification is represented.', '',
           '| Development index | RGB filename time (s) | Raw central scan time (s) | Scan / hypothesized pose row | Raw–PLY fields |',
           '|---:|---:|---:|---:|---|']
    for row in chain['development']:
        central=min(row['scan_table'],key=lambda r:abs(r['header_ns']-row['rgb_timestamp_ns']))
        lines.append(f"| {row['index']+1} | {stamp(row['rgb_timestamp_ns'])} | {stamp(row['selected_raw_timestamp_ns'])} | {central['hypothesized_pose_row']} | exact |")
    lines += ['', '## Coordinate chain and evidence limits', '',
        '`raw point (ouster1/os1_lidar, header+t) -> hypothesized sensor-to-map P(header+t)`',
        '`-> inverse P(RGB time) -> current LiDAR -> stored camera-to-LiDAR extrinsic inverse -> camera`.', '',
        'The motion prototype computes through the current LiDAR frame. The final camera projection',
        'is the declared next interface step, not a measured physical-registration validation in this round.',
        'The pinned extrinsic, intrinsic and URDF paths/hashes are in `UPSTREAM_PROVENANCE.json`.',
        'The complete per-scan identities, raw times and body transforms are in `DEVELOPMENT_CHAIN.json`.', '',
        'Static rotation chain: `base_link -> chassis_link -> top_chassis_link -> imu_link`',
        '`-> os1_frame -> ouster1/os1_sensor -> ouster1/os1_lidar`.',
        'Gravity alignment and its empirical uncertainty remain unqualified. Physical camera alignment,',
        'surface/visibility error and a complete spatial partition are not supplied by this correspondence table.', '']
    (dst/'TIME_POSE_CORRESPONDENCE.md').write_text('\n'.join(lines))
    manifest={'created_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
              'collector_job_id':os.environ.get('SLURM_JOB_ID'),
              'scope':'Compact evidence archive; raw sensors and upstream source copies remain on Quest',
              'files_sha256':{str(p.relative_to(dst)):sha(p) for p in sorted(dst.rglob('*')) if p.is_file()}}
    (dst/'DELIVERY_MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps({'files':len(manifest['files_sha256']),'destination':str(dst)},indent=2))


if __name__=='__main__':
    main()
