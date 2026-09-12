"""Read raw bag indexes and fixed-development provenance on Quest only."""
import concurrent.futures
import hashlib
import json
from pathlib import Path
import re
import sys
import time

ROOT = Path('/projects/p33100/siosio/hazard_rellis_revision_20260912')
OLD = Path('/projects/p33100/siosio/hazard_rellis_external_20260912')
sys.path[:0] = [str(ROOT / 'sensor_deps'), str(OLD / 'research/rellis_external_20260912')]
from remote_zip import RemoteZipReader
from rosbags.rosbag1.reader import Reader, Header, RecordType
import numpy as np


RAW = {
    '00000_00.bag': '1TRJOnqbTQaHUxUNJ9oAt3MsK-T7UlkIX',
    '00000_01.bag': '1dGZyt_0CLz8ploMzqPUHDgOVTg6zNVh_',
    '00000_02.bag': '1RPUobvuIr1GjQkzmaZ6eDLuO28YiP7RG',
    '00000_03.bag': '1JiO3OGlx_O4MbZPFM1F70UYNsk__MRIq',
    '00000_04.bag': '1VKLO-GWOwOkL817UenArOKwMmbQyuhoI',
}


def bag_index(spec):
    name, file_id = spec
    result = {'bag': name, 'official_id': file_id}
    try:
        handle = RemoteZipReader(file_id)
        handle.seek(0)
        if handle.read(13) != b'#ROSBAG V2.0\n':
            raise ValueError('Expected raw ROS bag v2')
        header = Header.read(handle, RecordType.BAGHEADER)
        index_pos = header.get_uint64('index_pos')
        if index_pos == 0:
            raise ValueError('Unindexed source bag')
        # Reuse pinned parser methods with a seekable HTTP stream. Reader.open()
        # is deliberately not called: it scans every chunk index in the bag.
        reader = Reader(ROOT / 'research/rellis_revision_20260912/revision_protocol.json')
        reader.bio = handle
        handle.seek(index_pos)
        connections = [reader.read_connection() for _ in range(header.get_uint32('conn_count'))]
        chunks = [reader.read_chunk_info() for _ in range(header.get_uint32('chunk_count'))]
        result.update(status='INDEX_ONLY', file_size=handle.size, index_pos=index_pos,
                      transferred_bytes=handle.transferred,
                      connections=[{'id': c.id, 'topic': c.topic, 'type': c.msgtype,
                                    'definition': c.msgdef.data, 'digest': c.digest} for c in connections],
                      chunks=[c._asdict() for c in chunks])
    except Exception as exc:
        result.update(status='ERROR', error=repr(exc))
    (ROOT / 'sensor_metadata' / (name + '.json')).write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k not in ['connections', 'chunks']}, default=str), flush=True)
    return result


def main():
    dst = ROOT / 'sensor_metadata'
    dst.mkdir(exist_ok=False)
    source = ROOT / 'source_audit'
    inventory = json.loads((source / 'ply_inventory.json').read_text())
    out = {'created_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), 'sequences': {}}
    for seq in range(5):
        s = f'{seq:05}'
        names = [r['name'] for r in inventory if f'/{s}/' in r['name'] and r['name'].endswith('.ply')]
        records = []
        for name in names:
            match = re.search(r'frame(\d+)-(\d+)_(\d+)\.ply', name)
            records.append({'frame': int(match[1]), 'timestamp_ns': int(match[2]) * 10**9 + int(match[3].ljust(9, '0')), 'member': name})
        records.sort(key=lambda r: r['timestamp_ns'])
        poses_path = OLD / f'metadata/poses/Rellis-3D/{s}/poses.txt'
        poses = np.loadtxt(poses_path).reshape(-1, 3, 4)
        rotations = poses[:, :, :3]
        out['sequences'][s] = {
            'ply_count': len(records), 'pose_count': len(poses),
            'ply_ids_exactly_zero_to_n_minus_one': [r['frame'] for r in records] == list(range(len(records))),
            'pose_sha256': hashlib.sha256(poses_path.read_bytes()).hexdigest(),
            'rotation_orthogonality_max': float(np.max(np.abs(rotations @ rotations.transpose(0, 2, 1) - np.eye(3)))),
            'rotation_determinant_range': [float(np.linalg.det(rotations).min()), float(np.linalg.det(rotations).max())],
            'scan_gap_ms_range': [float(np.diff([r['timestamp_ns'] for r in records]).min()/1e6), float(np.diff([r['timestamp_ns'] for r in records]).max()/1e6)],
            'records': records,
            'identity_status': 'UNVERIFIED_HYPOTHESIS: source describes scan-ordered sensor-to-map export; counts/order alone are insufficient'
        }
    (dst / 'POSE_SCAN_METADATA.json').write_text(json.dumps(out, indent=2) + '\n')
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as pool:
        bags = list(pool.map(bag_index, RAW.items()))
    summary = {'at_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
               'raw_folder_sha256': hashlib.sha256((source / 'raw_folder.html').read_bytes()).hexdigest(),
               'new_image_or_label_contents_opened': 0,
               'bags': [{k: v for k, v in b.items() if k not in ['connections', 'chunks']} | {
                    'topics': [c['topic'] for c in b.get('connections', [])],
                    'chunks': len(b.get('chunks', [])),
                    'start_ns': min((c['start_time'] for c in b.get('chunks', [])), default=None),
                    'end_ns': max((c['end_time'] for c in b.get('chunks', [])), default=None)} for b in bags]}
    (dst / 'SUMMARY.json').write_text(json.dumps(summary, indent=2) + '\n')


if __name__ == '__main__':
    main()
