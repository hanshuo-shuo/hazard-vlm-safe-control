"""Fetch only necessary old-development sensor windows from indexed raw bags.

Reuses rosbags record/message decoders. No images or semantic annotations are
decoded. Raw chunks may contain other topics, which are skipped. All geometry,
message accounting and tests run on Quest compute nodes.
"""
import collections
import concurrent.futures
import dataclasses
import hashlib
import io
import json
from pathlib import Path
import sys
import time

ROOT = Path('/projects/p33100/siosio/hazard_rellis_revision_20260912')
OLD = Path('/projects/p33100/siosio/hazard_rellis_external_20260912')
sys.path[:0] = [str(ROOT / 'sensor_deps'), str(OLD / 'research/rellis_external_20260912')]
from remote_zip import RemoteZipReader
from rosbags.rosbag1.reader import Header, RecordType, read_uint32, decompressors
from rosbags.typesys import Stores, get_typestore, get_types_from_msg
import numpy as np

TOPICS = {'/os1_cloud_node/points', '/os1_cloud_node/imu', '/imu/data', '/imu/data_raw',
          '/odometry/filtered', '/vectornav/IMU', '/vectornav/Odom', '/vectornav/GPS',
          '/tf', '/tf_static'}


def plain(value):
    if dataclasses.is_dataclass(value):
        return {f.name: plain(getattr(value, f.name)) for f in dataclasses.fields(value) if f.name != '__msgtype__'}
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (list, tuple)):
        return [plain(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def stamp(header):
    return int(header.stamp.sec) * 10**9 + int(header.stamp.nanosec)


def process_bag(path):
    meta = json.loads(path.read_text())
    target = ROOT / 'development_sensors' / meta['bag']
    target.mkdir(exist_ok=False)
    centres = json.loads((OLD / 'artifacts/PILOT_SELECTION.json').read_text())['records']
    times = np.array([int(round(r['timestamp'] * 1e9)) for r in centres], dtype=np.int64)
    conn = {c['id']: c for c in meta['connections']}
    store = get_typestore(Stores.EMPTY)
    for c in conn.values():
        if c['topic'] in TOPICS and c['type'] not in store.types:
            store.register(get_types_from_msg(c['definition'], c['type']))
    selected_chunks = []
    for chunk in meta['chunks']:
        intersects = np.any((times + 650_000_000 >= chunk['start_time']) & (times - 650_000_000 <= chunk['end_time']))
        has_static = any(conn[int(k)]['topic'] == '/tf_static' for k in chunk['connection_counts'])
        has_topic = any(conn[int(k)]['topic'] in TOPICS for k in chunk['connection_counts'])
        if (intersects and has_topic) or has_static:
            selected_chunks.append(chunk)
    (target / 'CHUNK_SELECTION.json').write_text(json.dumps(selected_chunks, indent=2) + '\n')
    remote = RemoteZipReader(meta['official_id'])
    out = {'bag': meta['bag'], 'official_id': meta['official_id'], 'selected_chunks': len(selected_chunks),
           'images_decoded': 0, 'labels_decoded': 0, 'messages': [], 'chunks': [], 'counts': {}}
    counts = collections.Counter()
    for ci, chunk in enumerate(selected_chunks):
        remote.seek(chunk['pos'])
        header = Header.read(remote, RecordType.CHUNK)
        size = read_uint32(remote)
        if size > 80_000_000:
            raise ValueError('Raw chunk unexpectedly exceeds 80 MB')
        data = remote.read(size)
        unpacked = decompressors[header.get_string('compression')](data)
        if len(unpacked) != header.get_uint32('size'):
            raise ValueError('Decompressed chunk size mismatch')
        out['chunks'].append({'pos': chunk['pos'], 'compressed_bytes': size,
                              'sha256': hashlib.sha256(data).hexdigest()})
        stream = io.BytesIO(unpacked)
        while stream.tell() < len(unpacked):
            h = Header.read(stream)
            n = read_uint32(stream)
            if h.get_uint8('op') != RecordType.MSGDATA:
                stream.seek(n, 1)
                continue
            c = conn[h.get_uint32('conn')]
            rec_ns = h.get_time('time')
            if c['topic'] not in TOPICS or (c['topic'] != '/tf_static' and np.min(np.abs(times - rec_ns)) > 650_000_000):
                stream.seek(n, 1)
                continue
            raw = stream.read(n)
            try:
                message = store.deserialize_ros1(raw, c['type'])
            except Exception as exc:
                (target / 'DECODE_ERROR.json').write_text(json.dumps({
                    'connection': c, 'record_ns': rec_ns, 'raw_length': len(raw),
                    'error': repr(exc), 'chunk_pos': chunk['pos'],
                    'raw_sha256': hashlib.sha256(raw).hexdigest()}, indent=2) + '\n')
                (target / 'decode_error.msg').write_bytes(raw)
                raise
            msg_ns = stamp(message.header) if hasattr(message, 'header') else rec_ns
            ids = np.flatnonzero(np.abs(times - msg_ns) <= 500_000_000).tolist()
            if not ids and c['topic'] != '/tf_static':
                continue
            record = {'topic': c['topic'], 'type': c['type'], 'record_ns': rec_ns,
                      'header_ns': msg_ns, 'development_indices': ids,
                      'raw_sha256': hashlib.sha256(raw).hexdigest()}
            if c['topic'] == '/os1_cloud_node/points':
                types = {1: 'i1', 2: 'u1', 3: '<i2', 4: '<u2', 5: '<i4', 6: '<u4', 7: '<f4', 8: '<f8'}
                dtype = np.dtype({'names': [f.name for f in message.fields],
                                  'formats': [types[f.datatype] for f in message.fields],
                                  'offsets': [f.offset for f in message.fields],
                                  'itemsize': message.point_step})
                if message.is_bigendian or message.row_step != message.width * message.point_step:
                    raise ValueError('Unexpected cloud endianness or row padding')
                array = np.frombuffer(message.data, dtype=dtype)
                keep = {k: array[k] for k in ['x', 'y', 'z', 't', 'ring', 'intensity', 'range'] if k in array.dtype.names}
                fname = f'cloud_{msg_ns}.npz'
                np.savez_compressed(target / fname, **keep)
                record.update(file=fname, file_sha256=hashlib.sha256((target / fname).read_bytes()).hexdigest(),
                              frame_id=message.header.frame_id, fields=[plain(f) for f in message.fields],
                              points=len(array), height=message.height, width=message.width)
            else:
                record['message'] = plain(message)
            out['messages'].append(record)
            counts[c['topic']] += 1
        if ci % 25 == 0:
            print(json.dumps({'bag': meta['bag'], 'chunks_done': ci+1, 'chunks_total': len(selected_chunks),
                              'transferred_MB': round(remote.transferred/1e6, 2)}), flush=True)
        if remote.transferred > 3_000_000_000:
            raise ValueError('Per-bag 3 GB development transfer cap exceeded')
    out['counts'] = dict(counts)
    out['transferred_bytes'] = remote.transferred
    (target / 'SENSORS.json').write_text(json.dumps(out, indent=2) + '\n')
    print(json.dumps({'bag': meta['bag'], 'counts': out['counts'], 'transferred_bytes': remote.transferred}), flush=True)
    return {k: out[k] for k in ['bag', 'counts', 'transferred_bytes', 'selected_chunks', 'images_decoded', 'labels_decoded']}


def main():
    dst = ROOT / 'development_sensors'
    dst.mkdir(exist_ok=False)
    started = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as pool:
        results = list(pool.map(process_bag, sorted((ROOT / 'sensor_metadata').glob('*.bag.json'))))
    summary = {'at_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
               'wall_seconds': time.time()-started, 'bags': results,
               'window_seconds': 1.0, 'record_time_preselection_padding_seconds': .15,
               'header_timestamp_filter_half_width_seconds': .5,
               'note': 'TF static is calibration metadata; no nondevelopment RGB or semantic label is decoded'}
    (dst / 'SUMMARY.json').write_text(json.dumps(summary, indent=2) + '\n')


if __name__ == '__main__':
    main()
