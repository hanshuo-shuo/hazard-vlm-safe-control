# Oracle aggregation: discovery and one prospective confirmation

All figures/statistics computed on Quest. No new training or certification.

## Each level selects and accepts at .02

| Bank | Level | Acceptance | Unsafe / accepted | Safe opportunity efficiency | Lost safe opportunity / all scenes |
|---|---|---:|---:|---:|---:|
| IID discovery | reference512 | 80.200% | 0.000% | 100.000% | 0.000% |
| IID discovery | pool512 | 76.600% | 0.000% | 95.511% | 3.600% |
| IID discovery | target64 | 77.250% | 0.065% | 96.259% | 3.000% |
| IID discovery | restored | 78.580% | 1.069% | 96.933% | 2.460% |
| IID discovery | minus2 | 81.870% | 2.846% | 99.177% | 0.660% |
| Appearance discovery | reference512 | 81.600% | 0.000% | 100.000% | 0.000% |
| Appearance discovery | pool512 | 77.750% | 0.000% | 95.282% | 3.850% |
| Appearance discovery | target64 | 78.300% | 0.064% | 95.895% | 3.350% |
| Appearance discovery | restored | 80.310% | 1.544% | 96.900% | 2.530% |
| Appearance discovery | minus2 | 84.260% | 3.976% | 99.154% | 0.690% |
| IID confirmation | reference512 | 79.600% | 0.000% | 100.000% | 0.000% |
| IID confirmation | pool512 | 76.450% | 0.000% | 96.043% | 3.150% |
| IID confirmation | target64 | 77.000% | 0.130% | 96.608% | 2.700% |
| IID confirmation | restored | 77.830% | 1.028% | 96.771% | 2.570% |
| IID confirmation | minus2 | 81.110% | 3.119% | 98.719% | 1.020% |

Zero observed unsafe outcomes are not population risk certificates.

## Fresh confirmation: registered predictions

H1: 61/2000 stable-reference lost opportunities; exact one-sided lower 0.02286937 > .01.
H2: 322/322 observed selected-action flips conservative. Original bootstrap lower=1.0 is degenerate; do not interpret it as population certainty.
Additional stricter scene check: 68/68 changed scenes have only conservative changes; exact lower=0.94156589 > .90.
H3: small-minus-larger margin flip-rate difference=0.37308001; crossed bootstrap lower=0.26058609 > .10.
All three claim-level one-sided alpha values are .05/3. H3 is approximate with five model seeds; the exact direction check conditions on the fixed model bank.
The extra direction check was recorded after job submission but before any fresh outcome was inspected and before HYPOTHESES.json existed; it tightens the claim, with no changed data or original output.

## Signed costs at fixed restored-model selections

| Bank | Model | Source raster | Pooling | Total | Total absolute error |
|---|---:|---:|---:|---:|---:|
| IID discovery | -0.000292121 | -0.000653205 | +0.001806849 | +0.000861523 | 0.002911001 |
| Appearance discovery | -0.001453001 | -0.000714196 | +0.002059689 | -0.000107509 | 0.003101676 |
| IID confirmation | -0.000439516 | -0.000678022 | +0.001985720 | +0.000868181 | 0.003255901 |

Signed terms telescope; absolute values are not attribution shares. The C cost is recomputed nonlinearly at every exposure level.

Complete exposure-channel summaries, label-pattern counts, each-model counts, precision checks and covariance-envelope strata are in the JSON reports.
