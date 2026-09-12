"""Close the bounded prerequisite audit without inventing a 48-frame result."""
import hashlib
import json
from pathlib import Path
import time
import numpy as np

ROOT = Path('/projects/p33100/siosio/hazard_rellis_revision_20260912')


def bounds(x):
    return {'min': float(np.min(x)), 'median': float(np.median(x)), 'max': float(np.max(x))}


def main():
    first = json.loads((ROOT/'development_chain/DEVELOPMENT_CHAIN.json').read_text())
    motion = json.loads((ROOT/'motion_audit/MOTION_AUDIT.json').read_text())
    acquired = json.loads((ROOT/'development_sensors/SUMMARY.json').read_text())
    rr = motion['development']
    before = np.array([r['unfitted_scan_residual_under_supplied_poses']['p50_m'] for r in first['development']])
    after = np.array([r['deskewed_unfitted_scan_residual']['p50_m'] for r in rr])
    summary = {
        'id': 'RELLIS-REVISION-PREREQUISITE-DECISION-20260912',
        'created_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        'decision': 'STOP_PREREQUISITES_UNCERTAIN',
        'budget_exhausted': False,
        'stop_type': 'Early epistemic stop within the registered upper budget; not a 48-frame qualification failure',
        'scope': 'Time/coordinate provenance and fixed per-point motion-compensation development audit',
        'development_frames': len(first['development']),
        'raw_ply_exact_matches': sum(r['status']=='OBSERVED_RAW_PLY_MATCH' for r in first['development']),
        'raw_scans': first['topic_counts']['/os1_cloud_node/points'],
        'invalid_raw_imu_messages': first['invalid_messages_by_topic']['/imu/data_raw'],
        'sensor_window_download_bytes': sum(r['transferred_bytes'] for r in acquired['bags']),
        'record_minus_scan_header_ms': bounds([r['raw_record_minus_header_ms'] for r in first['development']]),
        'raw_vs_released_scan_timestamp_difference_ms': bounds([s['released_scan_delta_ns']/1e6 for r in first['development'] for s in r['scan_table']]),
        'per_point_time_offset_ms': bounds([v for r in rr for v in r['per_point_time_offset_ms']]),
        'point_time_minus_rgb_ms': bounds([v for r in rr for v in r['point_time_minus_rgb_ms_range']]),
        'motion_correction_p95_per_frame_m': bounds([r['centre_motion_correction_displacement_m']['p95'] for r in rr]),
        'scan_consistency_p50_before_m': bounds(before),
        'scan_consistency_p50_after_m': bounds(after),
        'scan_consistency_p50_improved_frames': int((after<before).sum()),
        'scan_consistency_scope': 'Potentially reuses upstream SLAM fitting observations; no independent pose-error bound',
        'ouster_mean_acceleration_direction_error_degrees': bounds([r['ouster_gravity_check']['mean_direction_deviation_from_map_up_degrees'] for r in rr]),
        'ouster_instantaneous_acceleration_direction_p95_degrees': bounds([r['ouster_gravity_check']['instantaneous_deviation_p95_degrees'] for r in rr]),
        'gravity_scope': 'Observed specific-force versus hypothesized map-up discrepancy; dynamic acceleration and mounting are not identified separately; not an angular uncertainty envelope',
        'raw_tf_vs_urdf_max_abs': motion['raw_tf_vs_pinned_urdf_max_abs'],
        'established': [
            'All 24 consumed central PLY scans match raw x/y/z/t/ring fields exactly',
            'Timestamped raw IMU, odometry and scan records are accessible in fixed old-development windows',
            'Raw static LiDAR/body chain agrees numerically with the pinned URDF',
            'A fixed per-point deskew computation under the stated pose-time hypothesis runs on all 24 frames',
            'Full-area binary/continuous qualification paths are separately registered before fresh samples'
        ],
        'blocking_evidence_gaps': [
            'The scan-order released-pose association has source-code and consistency support, but no release-specific timestamp/pose export or independent full-3D registration check establishes its physical error',
            'The linked Cartographer recipe already uses VectorNav IMU; scan-pair residuals may reuse its SLAM fitting data. Neither can be relabelled independent registration evidence',
            'Independent Ouster acceleration is available, but specific force and mounting/attitude errors are not separated into a justified gravity-error range',
            'No independently validated camera alignment/projection-error range or surface-interpolation/visibility-error range has been established; a narrower support interval would therefore be unqualified',
            'A complete support-aware spatial partition with verified cross-sequence registration has not been established'
        ],
        'not_attempted_after_stop': [
            'Promoting corrected single/multi-scan geometry into an admissible semantic corridor evaluator',
            'Selecting or acquiring the new 48 qualification frames',
            'The 12-frame independent human review on that uninstantiated qualification set',
            'Decision/continuous-cost qualification scoring or external model/mechanism training'
        ],
        'fresh_qualification': {'selection_frozen': False, 'frames_acquired': 0, 'frames_evaluated': 0,
                                'all_four_determined': None, 'mixed_scenes': None,
                                'continuous_cost_p95': None, 'outcome': 'NOT_REACHED'},
        'paper_action': 'Retain controlled-case diagnosis. Report a bounded measurement-interface prerequisite stop; neither support nor refute mechanism transfer. No automatic third geometric repair.',
        'input_sha256': {str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [
            ROOT/'research/rellis_revision_20260912/revision_protocol.json',
            ROOT/'development_chain/DEVELOPMENT_CHAIN.json', ROOT/'motion_audit/MOTION_AUDIT.json',
            ROOT/'development_sensors/SUMMARY.json']}
    }
    (ROOT/'artifacts/REVISION_DECISION.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':
    main()
