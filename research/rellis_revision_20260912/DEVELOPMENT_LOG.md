# Development ledger

All numerical processing and tests are on Quest. Start: 2026-09-12 19:00 UTC;
deadline: 2026-09-14 19:00 UTC; engineering upper limit 16 hours. Time is an upper
bound, not an obligation to spend two days after a decisive failed prerequisite.
One fixed per-point motion prototype is evaluated as prerequisite work. Neither
permitted corridor-evaluator version is promoted to qualified semantic scoring.
The first failed geometry round is untouched.

## Protocol and data permissions

The first freeze registers full-area costs, decision versus continuous-cost
qualification, the budget and three outcomes. No fresh qualification identity
list has been instantiated or acquired. A second freeze is required before
new data acquisition and must contain all 48 centres and complete support IDs.

## Source audit

- Job 6190153: official README, calibration, export example and archive directory
  audit. Full merged and synced bags are DEFLATE members of large ZIPs; the raw
  split bags are separately accessible with validated byte ranges.
- Official repository: `c17a118fcaed1559f03cc32cc3a91dedc557f8b8`.
- The author's response to [issue 7](https://github.com/unmannedlab/RELLIS-3D/issues/7#issuecomment-830183291)
  identifies Cartographer as the pose source. Its linked repository is pinned at
  `27c15c92fa43ab221508add8597bdc0517a0cccb`. The exporter iterates point-cloud
  messages, transforms sensor coordinates to the map and emits a matrix without
  its timestamp. Its PointCloud2 converter sets per-point offsets to zero.
  This explains a possible association; it does not validate the delivered rows.
- Issue 27 was resolved by using Ouster, rather than Velodyne, with the Ouster
  calibration. We do not cite its opening complaint as evidence the release is
  generally miscalibrated. The current example and downloaded sequence-00000
  transform files match on inspection; an unanswered discussion is not contrary
  measurement evidence.
- Job 6190343: five raw bag indexes and all-sequence PLY/pose metadata. No image
  or semantic label content is decoded by the metadata stages.

## Parser failures and handling

- 6190490: strict rosbags decoding fails on `/imu/data_raw`; source preserved in
  `attempts/acquire_windows_6190490.py` and Quest's attempt directory.
- 6190554: using embedded message definitions reproduces the failure; source
  and raw messages preserved. This rules out simply blaming a Noetic schema.
- 6190594: independent scalar header/length inspection shows that three saved
  records have 321 bytes while the declared IMU fields consume 320. Each has a
  different unexplained trailing byte. This is a message-format observation,
  not proof that every physical IMU value is wrong.
- 6190625: identical old-development windows are reread with malformed messages
  saved and marked unusable. No bytes are trimmed to manufacture validity; other
  sensor sources are checked separately. Six distinct decision-contract tests
  pass. Repeated invocations are not additional distinct tests.
- 6190757: queued after successful sensor extraction, for content identity,
  scan/pose hypothesis, sensor-frame inventory and descriptive unfitted scan
  residuals. Such residuals are not automatically physical-error bounds.

Raw message content and full upstream repositories stay on Quest. New code uses
an isolated fixed parser installation (`rosbags 0.10.9`, `lz4 4.4.4`,
`zstandard 0.23.0`, `ruamel.yaml 0.18.10`) and preserves the existing experiment
environment. No model, VLM, risk-mechanism or RL computation is run.

## Completed diagnostics and closure

- 6190625 completes 240 valid Ouster scans, ten per fixed old target. 1,295
  `/imu/data_raw` messages are retained as malformed; no suffix is trimmed.
- 6190757 verifies raw-to-central-PLY x/y/z/t/ring equality in all 24 cases.
- 6191163 runs one fixed per-point deskew construction under the scan-order pose
  hypothesis, compares raw static transforms with the URDF, and checks independent
  Ouster specific force. The linked Cartographer offline recipe uses VectorNav IMU;
  neither VectorNav nor potentially reused SLAM scans supplies independent pose accuracy.
- 6191270 renders a first figure; visual inspection detects clipped titles. Its
  source and output remain in the attempt/Quest archive. 6191680 repairs presentation
  only, using the same immutable numerical reports.
- 6191543 records `STOP_PREREQUISITES_UNCERTAIN` at 2026-09-12 19:44:21 UTC.
  This is an early decision within the upper engineering budget, not budget exhaustion.
  No new 48-frame identities, human-review completion, or qualification outcomes exist.
- 6191612 independently verifies 1,542 file hashes, 240 cloud files, all 1,295
  abnormal message payloads (each one extra byte), and 960 fixed scalar PLY fields.
  Verification is rerun after the presentation artifact changes; distinct test
  counts remain six. Physical qualification is not granted by file verification.
- 6191917 completes the final replay after layout repair: 24 frozen-source entries,
  1,542 file hashes and the same scalar/raw-message checks pass.
- 6191945 assembles 29 compact evidence files plus a delivery manifest. Large
  raw payloads remain on Quest.
- 6192017 adds an exact filename-time display and checks the earlier internal
  float conversion: maximum 192 ns, no changed central scans or retained window
  memberships. The initial table and numerical reports are preserved. This
  precision check is separate from physical clock/exposure synchronization.

No geometric error envelope, ground-support interpolation or current-visibility
evidence is invented to make the next gate pass. The external round is closed;
the manuscript retains a controlled-case diagnosis and records the prerequisite
boundary without asserting that a new qualification experiment failed.
