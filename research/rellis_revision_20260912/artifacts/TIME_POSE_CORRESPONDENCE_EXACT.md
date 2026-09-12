# Exact source timestamps for the fixed development correspondence

The earlier table displays the binary-float-derived internal RGB time. This table instead
prints the literal millisecond filename time. Original numerical records are preserved.
Pose rows remain a scan-order hypothesis; no physical accuracy or qualification is implied.

| Development index | Exact RGB filename time (s) | Raw central scan header time (s) | Hypothesized pose row | Internal RGB conversion difference (ns) |
|---:|---:|---:|---:|---:|
| 1 | 1581624683.550 | 1581624683.569894400 | 308 | -128 |
| 2 | 1581624689.549 | 1581624689.569915904 | 368 | -64 |
| 3 | 1581624695.750 | 1581624695.769763328 | 430 | 128 |
| 4 | 1581624701.749 | 1581624701.769927424 | 490 | 192 |
| 5 | 1581624707.750 | 1581624707.769554432 | 550 | 128 |
| 6 | 1581624713.849 | 1581624713.869571328 | 611 | -64 |
| 7 | 1581624719.850 | 1581624719.869622528 | 671 | -128 |
| 8 | 1581624725.850 | 1581624725.868786944 | 731 | -128 |
| 9 | 1581624732.049 | 1581624732.068043264 | 793 | -64 |
| 10 | 1581624738.049 | 1581624738.068282112 | 853 | -64 |
| 11 | 1581624744.049 | 1581624744.068434688 | 913 | -64 |
| 12 | 1581624750.250 | 1581624750.267978496 | 975 | -128 |
| 13 | 1581624756.249 | 1581624756.267709184 | 1035 | 192 |
| 14 | 1581624762.449 | 1581624762.467666944 | 1097 | -64 |
| 15 | 1581624768.450 | 1581624768.467596288 | 1157 | 128 |
| 16 | 1581624774.449 | 1581624774.467424768 | 1217 | -64 |
| 17 | 1581624780.649 | 1581624780.667128576 | 1279 | -64 |
| 18 | 1581624786.650 | 1581624786.666621184 | 1339 | 128 |
| 19 | 1581624792.649 | 1581624792.666420992 | 1399 | -64 |
| 20 | 1581624798.849 | 1581624798.866750464 | 1461 | -64 |
| 21 | 1581624805.049 | 1581624805.066351360 | 1523 | -64 |
| 22 | 1581624811.050 | 1581624811.065945856 | 1583 | -128 |
| 23 | 1581624817.249 | 1581624817.265573632 | 1645 | 192 |
| 24 | 1581624823.250 | 1581624823.266051072 | 1705 | -128 |

The maximum internal conversion difference is 192 ns (0.000192 ms). All retained
scan memberships and all selected central scans are unchanged under exact filename parsing.
This is numerical representation precision, not camera exposure or clock-synchronization accuracy.
For the coordinate chain and its unresolved physical prerequisites, see the original
`TIME_POSE_CORRESPONDENCE.md` and `DEVELOPMENT_CHAIN.json`.
