"""Audit the single fixed motion construction on consumed development sensors.

The delivered scan-order pose association is explicitly a hypothesis. Numerical
deskew under it is not authorization to use the result as corridor ground truth.
Independent Ouster acceleration is checked; the VectorNav IMU is an input to the
linked Cartographer recipe and cannot independently validate that recipe.
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
sys.path[:0] = [str(OLD / 'qualification_deps'), str(ROOT / 'research/rellis_revision_20260912')]
import numpy as np
from scipy.spatial.transform import Rotation, Slerp
from audit_development_chain import vec, homogeneous, tf_matrix, compact_cloud, rigid_residual


def chain(edges, child, parent='base_link'):
    result = np.eye(4)
    visited = set()
    while child != parent:
        if child in visited or child not in edges:
            raise ValueError(f'Unresolved/cyclic coordinate chain at {child}')
        visited.add(child)
        next_parent, transform = edges[child]
        result = transform @ result
        child = next_parent
    return result


def main():
    dst = ROOT / 'motion_audit'
    dst.mkdir(exist_ok=False)
    preliminary = json.loads((ROOT / 'development_chain/DEVELOPMENT_CHAIN.json').read_text())
    all_messages = []
    for path in sorted((ROOT / 'development_sensors').glob('*/SENSORS.json')):
        for r in json.loads(path.read_text())['messages']:
            if 'file' in r:
                r['full_path'] = str(path.parent / r['file'])
            all_messages.append(r)
    edges, discrepancies = {}, []
    for t in preliminary['static_transforms']:
        child, parent, transform = t['child_frame_id'], t['header']['frame_id'], tf_matrix(t['transform'])
        if child in edges:
            previous_parent, previous = edges[child]
            if previous_parent != parent or np.max(np.abs(previous-transform)) > 1e-6:
                discrepancies.append(child)
        edges[child] = parent, transform
    urdf_edges = {}
    for j in preliminary['urdf_joints']:
        origin = j['origin']
        rotation = Rotation.from_euler('xyz', np.fromstring(origin.get('rpy', '0 0 0'), sep=' ')).as_matrix()
        translation = np.fromstring(origin.get('xyz', '0 0 0'), sep=' ')
        urdf_edges[j['child']] = j['parent'], homogeneous(rotation, translation)
    body_from_lidar = chain(edges, 'ouster1/os1_lidar')
    body_from_ouster_imu = chain(edges, 'ouster1/os1_imu')
    lidar_from_imu_rotation = body_from_lidar[:3, :3].T @ body_from_ouster_imu[:3, :3]
    pose_meta = json.loads((ROOT / 'sensor_metadata/POSE_SCAN_METADATA.json').read_text())['sequences']['00000']
    times = np.array([r['timestamp_ns'] for r in pose_meta['records']], dtype=np.int64)
    matrices = np.loadtxt(OLD / 'metadata/poses/Rellis-3D/00000/poses.txt').reshape(-1, 3, 4)
    secs = (times-times[0])/1e9
    slerp = Slerp(secs, Rotation.from_matrix(matrices[:, :, :3]))

    def at(ns):
        s = (np.asarray(ns)-times[0])/1e9
        rotation = slerp(s).as_matrix()
        translation = np.stack([np.interp(s, secs, matrices[:, j, 3]) for j in range(3)], axis=-1)
        return rotation, translation

    def deskew(r, target_ns):
        cloud = np.load(r['full_path'])
        xyz = np.column_stack([cloud[k] for k in 'xyz']).astype(float)
        point_ns = r['header_ns'] + cloud['t'].astype(np.int64)
        unique, inverse = np.unique(point_ns, return_inverse=True)
        rotations, translations = at(unique)
        reference_rotation, reference_translation = at(target_ns)
        xyz_map = np.einsum('nij,nj->ni', rotations[inverse], xyz) + translations[inverse]
        corrected = (xyz_map-reference_translation) @ reference_rotation
        finite_valid = np.isfinite(xyz).all(1) & (np.linalg.norm(xyz, axis=1)>2) & (np.linalg.norm(xyz, axis=1)<20)
        return corrected[finite_valid], xyz[finite_valid], point_ns

    out = {'created_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
           'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
           'coordinate_duplicate_discrepancies': discrepancies,
           'body_from_lidar_raw_tf': body_from_lidar.tolist(),
           'body_from_ouster_imu_raw_tf': body_from_ouster_imu.tolist(),
           'raw_tf_vs_pinned_urdf_max_abs': float(np.max(np.abs(body_from_lidar-chain(urdf_edges, 'ouster1/os1_lidar')))),
           'deskew_assumption': 'Released pose row = timestamp-ordered PLY scan identity; t is nanoseconds since header stamp per historical Ouster code. Release-specific timestamp provenance and physical projection remain unqualified.',
           'gravity_observation': 'Independent Ouster proper acceleration compared with hypothesized Cartographer map +Z. Dynamic acceleration and mounting error are not separately identified.',
           'scan_residual_scope': 'No new pairwise alignment fit, but these scans may have participated in upstream Cartographer. These are descriptive consistency residuals, not independent registration errors.',
           'cartographer_imu_input': '/vectornav/IMU (linked offline launch recipe)',
           'development': []}
    clouds = [r for r in all_messages if r['topic']=='/os1_cloud_node/points']
    for record in preliminary['development']:
        i, target = record['index'], record['rgb_timestamp_ns']
        row = {'index': i, 'image': record['image']}
        cc = sorted([r for r in clouds if i in r['development_indices']], key=lambda r:r['header_ns'])
        if not cc:
            row['status'] = 'NO_SCANS'
            out['development'].append(row)
            continue
        first, last = cc[0], cc[-1]
        aa, _, _ = deskew(first, target)
        bb, _, _ = deskew(last, target)
        row['deskewed_unfitted_scan_residual'] = rigid_residual(compact_cloud(bb), compact_cloud(aa), np.eye(4))
        centre = min(cc, key=lambda r: abs(r['header_ns']-target))
        corrected, native, point_ns = deskew(centre, target)
        shift = np.linalg.norm(corrected-native, axis=1)
        row['centre_motion_correction_displacement_m'] = {'p50': float(np.median(shift)), 'p95': float(np.quantile(shift,.95)), 'max':float(shift.max())}
        row['per_point_time_offset_ms'] = [float((point_ns.min()-centre['header_ns'])/1e6), float((point_ns.max()-centre['header_ns'])/1e6)]
        row['point_time_minus_rgb_ms_range'] = [float((point_ns.min()-target)/1e6),float((point_ns.max()-target)/1e6)]
        imu = [r for r in all_messages if r['topic']=='/os1_cloud_node/imu' and i in r['development_indices']]
        accelerations = np.array([vec(r['message']['linear_acceleration']) for r in imu])
        rotations, _ = at(np.array([r['header_ns'] for r in imu]))
        world_acc = np.einsum('nij,nj->ni', rotations, accelerations @ lidar_from_imu_rotation.T)
        mean_acc = world_acc.mean(0)
        row['ouster_gravity_check'] = {
            'n': len(imu), 'mean_acceleration_in_hypothesized_map': mean_acc.tolist(),
            'mean_direction_deviation_from_map_up_degrees': float(np.rad2deg(np.arccos(np.clip(mean_acc[2]/np.linalg.norm(mean_acc),-1,1)))),
            'instantaneous_deviation_p95_degrees': float(np.quantile(np.rad2deg(np.arccos(np.clip(world_acc[:,2]/np.linalg.norm(world_acc,axis=1),-1,1))),.95)),
            'limitation': 'Specific force includes vehicle acceleration; this is an observed discrepancy, not a gravity/calibration confidence bound'}
        row['status'] = 'HYPOTHESIS_DIAGNOSTIC_ONLY'
        out['development'].append(row)
    out['new_qualification_samples'] = 0
    out['physical_projection_error_measured'] = False
    out['cost_uncertainty_qualified'] = False
    (dst/'MOTION_AUDIT.json').write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps({'raw_tf_urdf_difference':out['raw_tf_vs_pinned_urdf_max_abs'],
        'rows':[{k:r.get(k) for k in ['index','centre_motion_correction_displacement_m','ouster_gravity_check']} for r in out['development']]},indent=2))


if __name__=='__main__':
    main()
