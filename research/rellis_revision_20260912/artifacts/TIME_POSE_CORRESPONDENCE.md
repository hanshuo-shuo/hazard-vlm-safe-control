# Fixed-development time and coordinate correspondence

All 24 centres are consumed development data. A pose row remains a scan-order hypothesis,
supported by source code and consistency checks but without a release-specific timestamped export.
All central raw x/y/z/t/ring arrays match the timestamped PLY. No fresh qualification is represented.

| Development index | RGB filename time (s) | Raw central scan time (s) | Scan / hypothesized pose row | Raw–PLY fields |
|---:|---:|---:|---:|---|
| 1 | 1581624683.549999872 | 1581624683.569894400 | 308 | exact |
| 2 | 1581624689.548999936 | 1581624689.569915904 | 368 | exact |
| 3 | 1581624695.750000128 | 1581624695.769763328 | 430 | exact |
| 4 | 1581624701.749000192 | 1581624701.769927424 | 490 | exact |
| 5 | 1581624707.750000128 | 1581624707.769554432 | 550 | exact |
| 6 | 1581624713.848999936 | 1581624713.869571328 | 611 | exact |
| 7 | 1581624719.849999872 | 1581624719.869622528 | 671 | exact |
| 8 | 1581624725.849999872 | 1581624725.868786944 | 731 | exact |
| 9 | 1581624732.048999936 | 1581624732.068043264 | 793 | exact |
| 10 | 1581624738.048999936 | 1581624738.068282112 | 853 | exact |
| 11 | 1581624744.048999936 | 1581624744.068434688 | 913 | exact |
| 12 | 1581624750.249999872 | 1581624750.267978496 | 975 | exact |
| 13 | 1581624756.249000192 | 1581624756.267709184 | 1035 | exact |
| 14 | 1581624762.448999936 | 1581624762.467666944 | 1097 | exact |
| 15 | 1581624768.450000128 | 1581624768.467596288 | 1157 | exact |
| 16 | 1581624774.448999936 | 1581624774.467424768 | 1217 | exact |
| 17 | 1581624780.648999936 | 1581624780.667128576 | 1279 | exact |
| 18 | 1581624786.650000128 | 1581624786.666621184 | 1339 | exact |
| 19 | 1581624792.648999936 | 1581624792.666420992 | 1399 | exact |
| 20 | 1581624798.848999936 | 1581624798.866750464 | 1461 | exact |
| 21 | 1581624805.048999936 | 1581624805.066351360 | 1523 | exact |
| 22 | 1581624811.049999872 | 1581624811.065945856 | 1583 | exact |
| 23 | 1581624817.249000192 | 1581624817.265573632 | 1645 | exact |
| 24 | 1581624823.249999872 | 1581624823.266051072 | 1705 | exact |

## Coordinate chain and evidence limits

`raw point (ouster1/os1_lidar, header+t) -> hypothesized sensor-to-map P(header+t)`
`-> inverse P(RGB time) -> current LiDAR -> stored camera-to-LiDAR extrinsic inverse -> camera`.

The motion prototype computes through the current LiDAR frame. The final camera projection
is the declared next interface step, not a measured physical-registration validation in this round.
The pinned extrinsic, intrinsic and URDF paths/hashes are in `UPSTREAM_PROVENANCE.json`.
The complete per-scan identities, raw times and body transforms are in `DEVELOPMENT_CHAIN.json`.

Static rotation chain: `base_link -> chassis_link -> top_chassis_link -> imu_link`
`-> os1_frame -> ouster1/os1_sensor -> ouster1/os1_lidar`.
Gravity alignment and its empirical uncertainty remain unqualified. Physical camera alignment,
surface/visibility error and a complete spatial partition are not supplied by this correspondence table.
