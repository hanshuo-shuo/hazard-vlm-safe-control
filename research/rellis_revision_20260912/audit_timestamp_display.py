"""Exact filename timestamps for display; preserve earlier float-based records."""
import hashlib
import json
from pathlib import Path
import re
import time

ROOT=Path('/projects/p33100/siosio/hazard_rellis_revision_20260912')


def main():
    rows=json.loads((ROOT/'development_chain/DEVELOPMENT_CHAIN.json').read_text())['development']
    lines=['# Exact source timestamps for the fixed development correspondence', '',
           'The earlier table displays the binary-float-derived internal RGB time. This table instead',
           'prints the literal millisecond filename time. Original numerical records are preserved.',
           'Pose rows remain a scan-order hypothesis; no physical accuracy or qualification is implied.', '',
           '| Development index | Exact RGB filename time (s) | Raw central scan header time (s) | Hypothesized pose row | Internal RGB conversion difference (ns) |',
           '|---:|---:|---:|---:|---:|']
    differences=[]
    changed_window_members=0
    for row in rows:
        match=re.search(r'frame\d+-(\d+)_(\d+)\.',row['image'])
        exact_ns=int(match[1])*10**9+int(match[2].ljust(9,'0'))
        difference=row['rgb_timestamp_ns']-exact_ns
        differences.append(difference)
        central=min(row['scan_table'],key=lambda r:abs(r['header_ns']-exact_ns))
        assert central['header_ns']==row['selected_raw_timestamp_ns']
        for scan in row['scan_table']:
            changed_window_members+=int((abs(scan['header_ns']-exact_ns)<=500_000_000) !=
                                        (abs(scan['header_ns']-row['rgb_timestamp_ns'])<=500_000_000))
        ns=central['header_ns']
        lines.append(f"| {row['index']+1} | {match[1]}.{match[2]} | {ns//10**9}.{ns%10**9:09} | {central['hypothesized_pose_row']} | {difference} |")
    lines += ['', 'The maximum internal conversion difference is 192 ns (0.000192 ms). All retained',
              'scan memberships and all selected central scans are unchanged under exact filename parsing.',
              'This is numerical representation precision, not camera exposure or clock-synchronization accuracy.',
              'For the coordinate chain and its unresolved physical prerequisites, see the original',
              '`TIME_POSE_CORRESPONDENCE.md` and `DEVELOPMENT_CHAIN.json`.', '']
    target=ROOT/'timestamp_display'
    target.mkdir(exist_ok=False)
    table=target/'TIME_POSE_CORRESPONDENCE_EXACT.md'
    table.write_text('\n'.join(lines))
    out={'at_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
         'max_internal_rgb_roundoff_ns':max(map(abs,differences)),
         'retained_scan_memberships_changed':changed_window_members,
         'central_scan_identity_changes':0,
         'original_numerical_outputs_preserved':True,
         'source_sha256':hashlib.sha256((ROOT/'development_chain/DEVELOPMENT_CHAIN.json').read_bytes()).hexdigest(),
         'display_sha256':hashlib.sha256(table.read_bytes()).hexdigest()}
    assert out['max_internal_rgb_roundoff_ns']==192 and changed_window_members==0
    (target/'TIMESTAMP_REPRESENTATION_AUDIT.json').write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps(out,indent=2))


if __name__=='__main__':
    main()
