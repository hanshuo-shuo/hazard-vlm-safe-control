"""Read-only development time/pose/coordinate diagnostics on Quest.

Every pose association here remains a hypothesis until independent observations
support it. No geometry is made known and no fresh qualification data are opened.
"""
import collections
import hashlib
import json
from pathlib import Path
import sys
import time
import xml.etree.ElementTree as ET

ROOT = Path('/projects/p33100/siosio/hazard_rellis_revision_20260912')
OLD = Path('/projects/p33100/siosio/hazard_rellis_external_20260912')
sys.path.insert(0, str(OLD / 'qualification_deps'))
import numpy as np
from scipy.spatial.transform import Rotation, Slerp
from scipy.spatial import cKDTree


def ply_geometry(path):
    types = {'float': '<f4', 'uint': '<u4', 'ushort': '<u2', 'uchar': 'u1'}
    with path.open('rb') as f:
        header = []
        while True:
            line = f.readline().decode('ascii').strip()
            header.append(line)
            if line == 'end_header':
                break
            if len(header) > 100:
                raise ValueError('Unexpected PLY header')
        n = int(next(s.split()[-1] for s in header if s.startswith('element vertex')))
        fields = [(s.split()[2], types[s.split()[1]]) for s in header if s.startswith('property ')]
        array = np.fromfile(f, dtype=fields, count=n)
    return {k: array[k] for k in ['x', 'y', 'z', 't', 'ring'] if k in array.dtype.names}


def vec(obj, keys='xyz'):
    return np.array([obj[k] for k in keys], dtype=float)


def homogeneous(rotation, translation):
    result = np.eye(4)
    result[:3, :3], result[:3, 3] = rotation, translation
    return result


def tf_matrix(t):
    return homogeneous(Rotation.from_quat(vec(t['rotation'], 'xyzw')).as_matrix(), vec(t['translation']))


def pose_interpolator(messages):
    valid = [r for r in messages if 'pose' in r.get('message', {})]
    if len(valid) < 2:
        return None
    valid.sort(key=lambda r: r['header_ns'])
    times, index = np.unique([r['header_ns'] for r in valid], return_index=True)
    valid = [valid[i] for i in index]
    secs = (times - times[0]) / 1e9
    q = np.array([vec(r['message']['pose']['pose']['orientation'], 'xyzw') for r in valid])
    pos = np.array([vec(r['message']['pose']['pose']['position']) for r in valid])
    if np.any(np.linalg.norm(q, axis=1) < .9):
        return None
    slerp = Slerp(secs, Rotation.from_quat(q))

    def at(ns):
        sec = (ns-int(times[0]))/1e9
        if ns < times[0] or ns > times[-1]:
            return None
        return homogeneous(slerp(sec).as_matrix(), [np.interp(sec, secs, pos[:, j]) for j in range(3)])
    return at


def cloud_xyz(record):
    data = np.load(record['full_path'])
    return np.column_stack([data[k] for k in 'xyz']).astype(float)


def compact_cloud(x):
    keep = np.isfinite(x).all(1) & (np.linalg.norm(x, axis=1) > 2) & (np.linalg.norm(x, axis=1) < 20)
    x = x[keep]
    _, indices = np.unique(np.floor(x/.20).astype(int), axis=0, return_index=True)
    return x[np.sort(indices)]


def rigid_residual(a, b, transform):
    """No extra ICP fit: observed cloud residual under supplied transform.

    Nearest neighbours include dynamics and discretization; not a certified
    registration bound or proof that correspondence is correct. These scans
    may already have participated in the upstream SLAM pose fit.
    """
    moved = a @ transform[:3, :3].T + transform[:3, 3]
    dist, _ = cKDTree(b).query(moved)
    return {'n': len(a), 'p50_m': float(np.median(dist)), 'p95_m': float(np.quantile(dist, .95)),
            'fraction_over_035m': float((dist > .35).mean())}


def main():
    dst = ROOT / 'development_chain'
    dst.mkdir(exist_ok=False)
    acquisition = json.loads((ROOT / 'development_sensors/SUMMARY.json').read_text())
    pilot = json.loads((OLD / 'artifacts/PILOT_ACQUISITION.json').read_text())['records']
    pose_meta = json.loads((ROOT / 'sensor_metadata/POSE_SCAN_METADATA.json').read_text())['sequences']['00000']
    all_scan_times = np.array([r['timestamp_ns'] for r in pose_meta['records']], dtype=np.int64)
    poses = np.repeat(np.eye(4)[None], pose_meta['pose_count'], axis=0)
    poses[:, :3] = np.loadtxt(OLD / 'metadata/poses/Rellis-3D/00000/poses.txt').reshape(-1, 3, 4)
    messages, invalid_counts, source_hashes = [], collections.Counter(), {}
    for path in sorted((ROOT / 'development_sensors').glob('*/SENSORS.json')):
        report = json.loads(path.read_text())
        source_hashes[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
        invalid_counts.update(report['invalid_by_topic'])
        for record in report['messages']:
            record['bag'] = path.parent.name
            if 'file' in record:
                record['full_path'] = str(path.parent / record['file'])
            messages.append(record)
    clouds = [r for r in messages if r['topic'] == '/os1_cloud_node/points']
    for r in clouds:
        i = int(np.argmin(np.abs(all_scan_times - r['header_ns'])))
        r['hypothesized_pose_row'] = i
        r['released_scan_timestamp_ns'] = int(all_scan_times[i])
        r['released_scan_delta_ns'] = int(r['header_ns']-all_scan_times[i])
    out = {'created_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
           'new_qualification_frames_opened': 0, 'model_computation': False,
           'source_sha256': source_hashes, 'invalid_messages_by_topic': dict(invalid_counts),
           'topic_counts': dict(collections.Counter(r['topic'] for r in messages)),
           'frames_by_topic': {}, 'static_transforms': [], 'development': [],
           'pose_mapping_validated': False, 'gravity_validated': False,
           'registration_uncertainty_validated': False, 'spatial_isolation_validated': False}
    for topic in out['topic_counts']:
        rr = [r for r in messages if r['topic'] == topic]
        out['frames_by_topic'][topic] = sorted({r.get('frame_id', r.get('message', {}).get('header', {}).get('frame_id', '')) for r in rr})
    for record in messages:
        if record['topic'] == '/tf_static':
            out['static_transforms'].extend(record['message']['transforms'])
    out['static_transforms'] = list({json.dumps(t, sort_keys=True): t for t in out['static_transforms']}.values())
    dynamic_pairs = collections.Counter()
    for r in messages:
        if r['topic'] == '/tf':
            dynamic_pairs.update((t['header']['frame_id'], t['child_frame_id']) for t in r['message']['transforms'])
    out['dynamic_transform_edges'] = [{'parent': a, 'child': b, 'count': n} for (a, b), n in dynamic_pairs.items()]
    # URDF orientations, keeping their source-qualified frame names explicit.
    urdf = ET.parse(OLD / 'official_rellis/catkin_ws/src/platform_description/urdf/warthog.urdf').getroot()
    out['urdf_joints'] = [{'parent': j.find('parent').attrib['link'], 'child': j.find('child').attrib['link'],
                          'origin': j.find('origin').attrib if j.find('origin') is not None else {}} for j in urdf.findall('joint')]
    for index, record in enumerate(pilot):
        row = {'index': index, 'image': record['image'], 'rgb_timestamp_ns': int(round(record['timestamp']*1e9)),
               'released_ply': record['lidar']['member'], 'hypothesized_pose_row': record['frame']}
        cc = sorted([r for r in clouds if index in r['development_indices']], key=lambda r: r['header_ns'])
        row['available_scans'] = len(cc)
        row['scan_table'] = [{k: r[k] for k in ['header_ns', 'record_ns', 'frame_id', 'released_scan_timestamp_ns',
                                             'released_scan_delta_ns', 'hypothesized_pose_row', 'file_sha256']} for r in cc]
        if not cc:
            row['status'] = 'NO_RAW_CLOUDS'
            out['development'].append(row)
            continue
        if len(cc) > 11:
            raise ValueError('Development window exceeds registered maximum scan count')
        chosen = min(cc, key=lambda r: abs(r['header_ns'] - row['rgb_timestamp_ns']))
        row['selected_raw_timestamp_ns'] = chosen['header_ns']
        ply = ply_geometry(OLD / record['lidar']['path'])
        raw = np.load(chosen['full_path'])
        row['raw_to_ply_equal_fields'] = {k: bool(np.array_equal(ply[k], raw[k])) for k in ply if k in raw.files}
        a = np.column_stack([ply[k] for k in 'xyz']).astype(float)
        b = np.column_stack([raw[k] for k in 'xyz']).astype(float)
        row['raw_ply_xyz_max_abs_m'] = float(np.max(np.abs(a-b))) if a.shape == b.shape else None
        row['point_t_min_max'] = [int(raw['t'].min()), int(raw['t'].max())] if 't' in raw.files else None
        row['raw_record_minus_header_ms'] = (chosen['record_ns']-chosen['header_ns'])/1e6
        row['status'] = 'OBSERVED_RAW_PLY_MATCH' if all(row['raw_to_ply_equal_fields'].values()) else 'RAW_PLY_MISMATCH'
        # No pairwise alignment is fitted here. These scans may have been used
        # in upstream SLAM and are NOT independent pose-validation observations.
        first, last = cc[0], cc[-1]
        first_id, last_id = first['hypothesized_pose_row'], last['hypothesized_pose_row']
        transform = np.linalg.inv(poses[first_id]) @ poses[last_id]
        row['supplied_pose_relative_first_to_last'] = transform.tolist()
        row['supplied_pose_motion_m'] = float(np.linalg.norm(transform[:3, 3]))
        row['supplied_pose_motion_degrees'] = float(np.rad2deg(Rotation.from_matrix(transform[:3, :3]).magnitude()))
        pa, pb = compact_cloud(cloud_xyz(first)), compact_cloud(cloud_xyz(last))
        row['unfitted_scan_residual_under_supplied_poses'] = rigid_residual(pb, pa, transform)
        row['unfitted_identity_transform_diagnostic'] = rigid_residual(pb, pa, np.eye(4))
        row['pose_motion_checks'] = {}
        for topic in ['/odometry/filtered', '/vectornav/Odom']:
            rr = [r for r in messages if r['topic'] == topic and index in r['development_indices']]
            at = pose_interpolator(rr)
            if at is None:
                row['pose_motion_checks'][topic] = {'status': 'NO_VALID_BRACKET'}
                continue
            x, y = at(first['header_ns']), at(last['header_ns'])
            if x is None or y is None:
                row['pose_motion_checks'][topic] = {'status': 'NO_VALID_BRACKET'}
                continue
            relative = np.linalg.inv(x) @ y
            row['pose_motion_checks'][topic] = {
                'status': 'DESCRIPTIVE_DIFFERENT_SENSOR_FRAME',
                'rotation_angle_degrees': float(np.rad2deg(Rotation.from_matrix(relative[:3, :3]).magnitude())),
                'translation_norm_m': float(np.linalg.norm(relative[:3, 3])),
                'relative_matrix': relative.tolist()}
        row['imu_summary'] = {}
        for topic in ['/imu/data', '/os1_cloud_node/imu', '/vectornav/IMU']:
            rr = [r for r in messages if r['topic'] == topic and index in r['development_indices']]
            if not rr:
                row['imu_summary'][topic] = {'valid_messages': 0}
                continue
            near = min(rr, key=lambda r: abs(r['header_ns']-row['rgb_timestamp_ns']))
            q = vec(near['message']['orientation'], 'xyzw')
            row['imu_summary'][topic] = {'valid_messages': len(rr), 'nearest': near,
                'quaternion_norm': float(np.linalg.norm(q)),
                'orientation_marked_available': near['message']['orientation_covariance'][0] != -1,
                'acceleration_norm_range': [float(min(np.linalg.norm(vec(r['message']['linear_acceleration'])) for r in rr)),
                                            float(max(np.linalg.norm(vec(r['message']['linear_acceleration'])) for r in rr))]}
        out['development'].append(row)
    out['interpretation'] = 'Raw-to-release content identity verifies a scan-time link when exact. Released-pose row correspondence, gravity, physical registration and complete spatial separation remain separate prerequisites. Scans can already participate in upstream SLAM, so residuals computed without an additional pairwise fit are NOT independent pose-validation errors; plausible odometry numbers alone also do not establish qualification.'
    (dst / 'DEVELOPMENT_CHAIN.json').write_text(json.dumps(out, indent=2) + '\n')
    print(json.dumps({'frames': len(out['development']), 'statuses': dict(collections.Counter(r['status'] for r in out['development'])),
                      'clouds': len(clouds), 'invalid': dict(invalid_counts),
                      'frames_by_topic': out['frames_by_topic'], 'dynamic_edges': out['dynamic_transform_edges']}, indent=2), flush=True)


if __name__ == '__main__':
    main()
