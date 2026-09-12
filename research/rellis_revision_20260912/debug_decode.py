"""Inspect preserved decoding failure; no new sensor content acquisition."""
import json
from pathlib import Path
import struct
import sys
ROOT = Path('/projects/p33100/siosio/hazard_rellis_revision_20260912')
sys.path.insert(0, str(ROOT / 'sensor_deps'))
from rosbags.typesys import Stores, get_typestore, get_types_from_msg

out = []
for path in sorted((ROOT / 'development_sensors').glob('*/DECODE_ERROR.json')):
    info = json.loads(path.read_text())
    raw = (path.parent / 'decode_error.msg').read_bytes()
    store = get_typestore(Stores.EMPTY)
    typ = info['connection']['type']
    store.register(get_types_from_msg(info['connection']['definition'], typ))
    definition = store.get_msgdef(typ)
    message, pos = definition.deserialize_ros1(raw, 0, definition.cls, store)
    seq, sec, nsec, nframe = struct.unpack_from('<IIII', raw, 0)
    row = {'bag': path.parent.name, 'topic': info['connection']['topic'],
           'raw_bytes': len(raw), 'parsed_bytes': pos, 'trailing_hex': raw[pos:].hex(),
           'header_scalar': [seq, sec, nsec, nframe],
           'frame_scalar': raw[16:16+nframe].decode(errors='replace'),
           'expected_imu_bytes': 16+nframe+37*8, 'parsed_message': str(message)}
    out.append(row)
print(json.dumps(out, indent=2))
(ROOT / 'artifacts/DECODE_DIAGNOSIS.json').write_text(json.dumps(out, indent=2) + '\n')
