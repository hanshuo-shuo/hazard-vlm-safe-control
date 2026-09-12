"""Independent artifact/count/byte checks on Quest; not physical qualification."""
import collections
import hashlib
import json
from pathlib import Path
import struct
import sys
import time
import numpy as np

ROOT = Path('/projects/p33100/siosio/hazard_rellis_revision_20260912')
OLD = Path('/projects/p33100/siosio/hazard_rellis_external_20260912')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path):
    def reject(s):
        raise ValueError(f'Nonfinite JSON value: {s}')
    return json.loads(path.read_text(), parse_constant=reject)


def main():
    src = ROOT/'research/rellis_revision_20260912'
    decision = read_json(ROOT/'artifacts/REVISION_DECISION.json')
    protocol = read_json(src/'revision_protocol.json')
    pilot = read_json(OLD/'artifacts/PILOT_ACQUISITION.json')['records']
    first = read_json(ROOT/'development_chain/DEVELOPMENT_CHAIN.json')
    motion = read_json(ROOT/'motion_audit/MOTION_AUDIT.json')
    checks, verified_hashes = {}, {}
    freeze_count = 0
    for freeze in sorted((ROOT/'artifacts').glob('FREEZE_*.sha256')):
        for line in freeze.read_text().splitlines():
            expected, path_text = line.split(maxsplit=1)
            path = Path(path_text.strip())
            if path.name == 'acquire_windows.py':
                for job in ['6190490','6190554']:
                    if job in freeze.name:
                        path = src/'attempts'/f'acquire_windows_{job}.py'
            if path.name == 'render_chain_audit.py' and '6191270' in freeze.name:
                path = src/'attempts/render_chain_audit_6191270.py'
            if path.name == 'verify_revision.py' and '6191612' in freeze.name:
                path = src/'attempts/verify_revision_6191612.py'
            assert sha(path) == expected, (freeze.name, str(path))
            freeze_count += 1
    checks['frozen_source_entries'] = freeze_count
    old_audit = read_json(OLD/'artifacts/METADATA_AUDIT.json')
    assert sha(OLD/'research/rellis_external_20260912/qualification_protocol.json') == old_audit['protocol_sha256']
    assert sha(OLD/'artifacts/PILOT_SELECTION.json') == old_audit['pilot_selection_sha256']
    checks['original_protocol_and_pilot_unchanged'] = True
    assert protocol['decision']['all_four_determined_scenes_min'] == 39
    assert protocol['decision']['mixed_compliant_noncompliant_scenes_min'] == 8
    assert protocol['decision']['mixed_blocks_min'] == 4
    assert protocol['decision']['continuous_cost_interval_width_p95_max'] == .005
    checks['registered_thresholds_retained'] = True
    by_scene = collections.defaultdict(list)
    counts, bad_suffixes, invalid_count = collections.Counter(), collections.Counter(), 0
    for path in sorted((ROOT/'development_sensors').glob('*/SENSORS.json')):
        report = read_json(path)
        counts.update(r['topic'] for r in report['messages'])
        for r in report['messages']:
            if r['topic'] != '/os1_cloud_node/points':
                continue
            cloud = path.parent/r['file']
            assert sha(cloud) == r['file_sha256']
            verified_hashes[str(cloud.relative_to(ROOT))] = r['file_sha256']
            with np.load(cloud) as arrays:
                assert set(arrays.files) <= {'x','y','z','t','ring','intensity','range'}
                assert {'x','y','z','t','ring'} <= set(arrays.files)
                assert all(len(arrays[k]) == r['points'] for k in arrays.files)
            for i in r['development_indices']:
                assert abs(r['header_ns']-int(round(pilot[i]['timestamp']*1e9))) <= 500_000_000
                by_scene[i].append((r,cloud))
        for r in report['invalid_messages']:
            path_raw = path.parent/r['file']
            raw = path_raw.read_bytes()
            assert hashlib.sha256(raw).hexdigest() == r['raw_sha256']
            assert r['usable_as_geometry_evidence'] is False
            assert r['connection']['topic'] == '/imu/data_raw'
            nframe = struct.unpack_from('<I',raw,12)[0]
            expected = 16+nframe+37*8
            bad_suffixes[len(raw)-expected] += 1
            invalid_count += 1
            verified_hashes[str(path_raw.relative_to(ROOT))] = r['raw_sha256']
        assert dict(collections.Counter(r['topic'] for r in report['messages'])) == report['counts']
    assert counts['/os1_cloud_node/points'] == 240
    assert set(by_scene) == set(range(24))
    assert all(len(v) == 10 for v in by_scene.values())
    assert invalid_count == decision['invalid_raw_imu_messages']
    checks['valid_clouds'] = 240
    checks['fixed_development_scenes'] = 24
    checks['scans_per_scene'] = 10
    checks['invalid_imu_messages_retained'] = invalid_count
    checks['invalid_imu_extra_byte_counts'] = dict(bad_suffixes)
    # Independently decode fixed scalar PLY records using struct offsets;
    # no reuse of the array loader or geometry helpers in the production audit.
    formats = {'float':'f','uint':'I','ushort':'H','uchar':'B'}
    scalar_values = 0
    for i, old in enumerate(pilot):
        ns = int(round(old['timestamp']*1e9))
        r,npz_path = min(by_scene[i], key=lambda item: abs(item[0]['header_ns']-ns))
        with (OLD/old['lidar']['path']).open('rb') as stream:
            fields, n, offset = {}, None, 0
            while True:
                line = stream.readline().decode('ascii').strip()
                if line.startswith('element vertex'):
                    n = int(line.split()[-1])
                elif line.startswith('property '):
                    _,typ,name = line.split()
                    fields[name] = (offset, formats[typ])
                    offset += struct.calcsize('<'+formats[typ])
                elif line == 'end_header':
                    break
            start = stream.tell()
            assert n == r['points']
            with np.load(npz_path) as npz:
                for index in [0,1,63,64,1024,32767,65535,n-1]:
                    for field in ['x','y','z','t','ring']:
                        within,fmt = fields[field]
                        stream.seek(start+index*offset+within)
                        value = struct.unpack('<'+fmt,stream.read(struct.calcsize('<'+fmt)))[0]
                        assert value == npz[field][index], (i,index,field)
                        scalar_values += 1
        assert first['development'][i]['status']=='OBSERVED_RAW_PLY_MATCH'
        assert first['development'][i]['raw_ply_xyz_max_abs_m']==0
    checks['independent_scalar_ply_values'] = scalar_values
    # Check reported summary statistics directly from immutable row records.
    values = [r['centre_motion_correction_displacement_m']['p95'] for r in motion['development']]
    assert decision['motion_correction_p95_per_frame_m']['min']==min(values)
    assert decision['motion_correction_p95_per_frame_m']['max']==max(values)
    assert decision['fresh_qualification']['frames_evaluated']==0
    assert decision['fresh_qualification']['all_four_determined'] is None
    assert decision['fresh_qualification']['mixed_scenes'] is None
    assert decision['fresh_qualification']['continuous_cost_p95'] is None
    assert decision['decision']=='STOP_PREREQUISITES_UNCERTAIN'
    assert not (ROOT/'fresh_qualification').exists()
    checks['no_fabricated_fresh_sample_result'] = True
    for relative,expected in decision['input_sha256'].items():
        assert sha(ROOT/relative)==expected
        verified_hashes[relative]=expected
    publication = read_json(ROOT/'publication/MANIFEST.json')
    for relative,expected in publication['input_sha256'].items():
        assert sha(ROOT/relative)==expected
    for name,expected in publication['files_sha256'].items():
        assert sha(ROOT/'publication'/name)==expected
        verified_hashes['publication/'+name]=expected
    out={'status':'PASS', 'at_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
         'scope':'Computational integrity and declared-data accounting only; physical registration/geometry qualification remains unresolved',
         'checks':checks,'verified_file_count':len(verified_hashes),'files_sha256':verified_hashes}
    (ROOT/'artifacts/REVISION_VERIFICATION.json').write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps({k:v for k,v in out.items() if k!='files_sha256'},indent=2))


if __name__=='__main__':
    main()
