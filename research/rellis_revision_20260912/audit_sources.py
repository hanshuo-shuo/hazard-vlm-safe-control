"""Quest-only, metadata-first audit. Never opens fresh RGB or semantic labels."""
import hashlib
import json
from pathlib import Path
import re
import sys
import time
import urllib.request
import zipfile

OLD = Path('/projects/p33100/siosio/hazard_rellis_external_20260912')
ROOT = Path('/projects/p33100/siosio/hazard_rellis_revision_20260912')
sys.path.insert(0, str(OLD / 'research/rellis_external_20260912'))
from remote_zip import RemoteZipReader


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    dst = ROOT / 'source_audit'
    dst.mkdir(exist_ok=False)
    out = {'created_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
           'new_image_or_label_contents_opened': 0, 'sources': {}, 'archives': {}}
    repo = OLD / 'official_rellis'
    for path in ['README.md', 'utils/convert_ply2bin.py', 'utils/lidar2img.ipynb',
                 'utils/example/transforms.yaml', 'utils/example/camera_info.txt',
                 'catkin_ws/src/platform_description/urdf/warthog.urdf']:
        src = repo / path
        target = dst / path.replace('/', '__')
        if src.suffix == '.ipynb':
            notebook = json.loads(src.read_text())
            target = target.with_suffix('.code.txt')
            target.write_text('\n\n'.join(''.join(c['source']) for c in notebook['cells'] if c['cell_type'] == 'code'))
        else:
            target.write_bytes(src.read_bytes())
        out['sources'][path] = {'source_sha256': sha(src), 'extract_sha256': sha(target)}
    for seq in range(5):
        for kind, base, fname in [('extrinsics', 'Rellis_3D', 'transforms.yaml'),
                                  ('intrinsics', 'Rellis-3D', 'camera_info.txt'),
                                  ('poses', 'Rellis-3D', 'calib.txt')]:
            src = OLD / 'metadata' / kind / base / f'{seq:05}' / fname
            target = dst / f'{kind}_{seq:05}_{fname}'
            target.write_bytes(src.read_bytes())
            out['sources'][str(src.relative_to(OLD))] = {'sha256': sha(src)}
    # Public discussions are provenance leads, not ground-truth measurements.
    for name, url in [('issues', 'https://api.github.com/repos/unmannedlab/RELLIS-3D/issues?state=all&per_page=100')]:
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'bounded-rellis-research'})
            with urllib.request.urlopen(req, timeout=45) as response:
                payload = response.read(8_000_001)
            if len(payload) > 8_000_000:
                raise ValueError('API response exceeds source budget')
            (dst / f'{name}.json').write_bytes(payload)
            issues = json.loads(payload)
            summaries = []
            for issue in issues:
                if not re.search(r'pose|calib|sync|time|odom|imu|transform|mismatch',
                                 issue['title'], re.I):
                    continue
                entry = {k: issue.get(k) for k in ['number', 'title', 'body', 'html_url', 'comments']}
                entry['author_association'] = issue.get('author_association')
                if issue.get('comments'):
                    with urllib.request.urlopen(urllib.request.Request(issue['comments_url'], headers={'User-Agent': 'bounded-rellis-research'}), timeout=45) as response:
                        comments = json.load(response)
                    entry['responses'] = [{'login': c['user']['login'], 'author_association': c['author_association'], 'body': c['body'], 'url': c['html_url']} for c in comments]
                summaries.append(entry)
            (dst / 'relevant_discussions.json').write_text(json.dumps(summaries, indent=2) + '\n')
            out['sources'][url] = {'status': 'OK', 'issues': len(issues), 'relevant': len(summaries)}
        except Exception as exc:
            out['sources'][url] = {'status': 'ERROR', 'error': repr(exc)}
    archives = {
        'ply': '1BZWrPOeLhbVItdN0xhzolfsABr6ymsRr',
        'kitti': '1lDSVRf_kZrD0zHHMsKJ0V1GN9QATR4wH',
        'merged_seq00000': '1grcYRvtAijiA0Kzu-AV_9K4k2C1Kc3Tn',
        'synced_seq00000': '1bIb-6fWbaiI9Q8Pq9paANQwXWn7GJDtl',
    }
    for name, file_id in archives.items():
        result = {'official_id': file_id}
        try:
            handle = RemoteZipReader(file_id)
            handle.seek(0)
            result.update(size_bytes=handle.size, magic_hex=handle.read(16).hex())
            handle.seek(0)
            if bytes.fromhex(result['magic_hex']).startswith(b'PK'):
                archive = zipfile.ZipFile(handle)
                entries = [{'name': f.filename, 'size': f.file_size, 'compressed_size': f.compress_size,
                            'compression': f.compress_type, 'header_offset': f.header_offset,
                            'crc': f.CRC} for f in archive.infolist()]
                (dst / f'{name}_inventory.json').write_text(json.dumps(entries, indent=2) + '\n')
                result.update(status='ZIP_DIRECTORY_ONLY', members=len(entries))
            else:
                result['status'] = 'RAW_FILE_HEADER_ONLY'
            result['transferred_bytes'] = handle.transferred
        except Exception as exc:
            result.update(status='ERROR', error=repr(exc))
        out['archives'][name] = result
        (dst / 'SOURCE_AUDIT.json').write_text(json.dumps(out, indent=2) + '\n')
        print(json.dumps({name: result}), flush=True)
    out['files_sha256'] = {p.name: sha(p) for p in sorted(dst.iterdir()) if p.is_file() and p.name != 'SOURCE_AUDIT.json'}
    (dst / 'SOURCE_AUDIT.json').write_text(json.dumps(out, indent=2) + '\n')


if __name__ == '__main__':
    main()
