# RELLIS-3D external-study qualification

This bounded round implements [Pro's second review](PRO_REVIEW.md). The
controlled-case paper has been rewritten; the external study stops at an
unsatisfied geometry qualification gate. No external model was trained,
no external risk certificate was asserted, and no final-test image labels
were opened.

- [Chinese progress and decision](../../docs/RELLIS_EXTERNAL_PROGRESS_2026-09-12.md)
- [Full diagnostic manuscript](../../paper/visual_risk_diagnosis/manuscript.md)
- [Claim-to-evidence map](../../paper/visual_risk_diagnosis/CLAIMS_AND_EVIDENCE.md)
- [Qualification protocol](qualification_protocol.json)
- [Fixed pilot identities](artifacts/PILOT_SELECTION.json)
- [Metadata and split audit](artifacts/METADATA_AUDIT.json)
- [Geometry result](artifacts/geometry_audit/GEOMETRY_AUDIT.json)
- [Independent verification](artifacts/QUALIFICATION_VERIFICATION.json)
- [Readable overview](artifacts/publication/qualification_overview.png)
- [Every fixed pilot scene](artifacts/publication/pilot_contact_sheet.png)

## Decision

The current single-scan surface and visibility construction passes **0/24**
scenes at the registered requirement that all four corridor segments have at
most 5% unknown area. An optimistic ceiling that removes projected-depth and
occlusion checks passes only **3/24**; it is a diagnostic ceiling and is not
eligible for action approval. At least 20/24 scenes are required for the 80%
pilot screen.

The unknown-area range is 45.97%-88.47% across 96 candidates. Unknown support
is not a safe semantic label. The result is specific to this geometric
construction; it is not a general claim that RELLIS-3D is unusable, a measured
calibration-error bound, or evidence that the synthetic mechanism transfers.
The body orientation is checked through the official URDF; full gravity
alignment and physical registration accuracy remain unqualified.

The official split contains some held-out frames 0.2 seconds from training
frames in the same sequence. That observation does not show that every test
frame is contaminated. It does prevent treating the release split as an
automatic spatial independence guarantee. The released checkpoint's complete
training and validation provenance still requires spatial audit.

## Fixed task and what was actually implemented

Before reading pilot labels, 24 images were selected at evenly spaced sorted
indices from sequence 00000's official training list. Four 1.2 m-wide corridor
segments have 6-12 m longitudinal range and headings -12/-4/4/12 degrees. There
are 720 equal-area samples per segment. Ranges are LiDAR-centred; the preceding
segment from the current robot position is outside the task.

The implementation uses nonsemantic local surface fitting, a 0.35 m nearest
surface-support radius (eight nearby samples within 0.75 m), 8-pixel depth bins,
and a 0.35 m foreground-depth margin. These implementation choices are frozen
before the pilot evaluation. Missing depth, unsupported surface, off-image
locations and occlusion remain unknown. Human labels are accessed only after
these geometric operations. Cost intervals use the full requested area:
lower=known-prohibited/total; upper=(known-prohibited+unknown)/total. They do not
renormalize a small observed fraction and call the entire corridor safe.

The protocol's intended gravity-aligned frame is not fully qualified; the
implemented frame is explicitly a LiDAR-centred, URDF-body-oriented diagnostic.
No scientific acceptance result is produced from that incomplete prerequisite.

Six distinct tests cover equal metric progress, unknown labels, unsupported
grass, camera depth sign, plane fitting with obstacles, and occlusion. A separate
scalar implementation verifies 96 candidate counts, 69,120 sample labels and
26 manifest-listed files. The independent projection implementation agrees to
1.82e-12 pixels; agreement of implementations is not physical alignment accuracy.

## Provenance and access

All processing took place under
`/projects/p33100/siosio/hazard_rellis_external_20260912` on Quest.
The official repository is pinned at
`c17a118fcaed1559f03cc32cc3a91dedc557f8b8`. Official source:
[unmannedlab/RELLIS-3D](https://github.com/unmannedlab/RELLIS-3D/tree/c17a118fcaed1559f03cc32cc3a91dedc557f8b8).

```
official_rellis/       untouched upstream repository
downloads/            small official calibration/split/example archives
metadata/             extracted calibration, poses and split lists
pilot/                only the 24 selected RGB, label and timestamped PLY triples
artifacts/            acquisition, selection, source freezes and verification
geometry_audit/       original frozen result and all fixed pilot thumbnails
publication/          readable plots and attribution
logs/                 including failures before pilot evaluation
```

`RemoteZipReader` validates each HTTP Content-Range and ZIP CRC. Archive member
names are inspected to find the fixed sample; non-pilot image/label contents
are not read. Large archives are not copied to the local repository. Small
download examples are retained unopened on Quest and are not confirmation data.

The runtime is `/home/shv7753/envs/hazard-exp01b-r2/bin/python`. Geometry reuses
NumPy and loads scipy 1.13.1, opencv-python-headless 4.10.0.84 and PyYAML 6.0.2
from an isolated `qualification_deps/`, without changing the existing environment.
Plotting uses the previous round's isolated `report_deps/`.

Dataset and derived image content are CC BY-NC-SA 3.0. Credit: Peng Jiang,
Philip Osteen, Maggie Wigness and Srikanth Saripalli. See
[image attribution](artifacts/publication/ATTRIBUTION.md). This attribution
does not change the license of unrelated project code.

## Execution and non-scientific repairs

| Job | Outcome |
|---|---|
| 6186376 | Official metadata and unopened example archives obtained |
| 6186684 | Stopped because `git` was absent on a compute node; before pilot analysis |
| 6186742 | Metadata audit and all 24 RGB/label/PLY triples completed |
| 6187158 | Coordinate guard rejected incorrect native +X forward assumption; before pilot image loading |
| 6187239 | Official URDF conversion, six tests and 24-scene geometry audit completed |
| 6187394 | Independent scalar verification completed |
| 6187474 | Publication plots |

The first failed implementations are preserved under `attempts/`; their logs
and original freeze records are retained. Neither repair changes the selection,
candidates or qualification thresholds. Repeated test runs count as six distinct
tests, not twelve.

The `.sbatch` files document the completed run and point to its actual root.
Use a new root for reproduction. Do not overwrite old scientific artifacts.
Run metadata acquisition, then pilot preparation, geometry tests/audit,
independent verification, and report rendering in that order. The qualification
JSON and source hashes must be frozen before reading a new pilot.

## Stopping boundary

No released model is used to generate a purported external confirmation.
The six-arm model comparison, development-selected postprocessing, grouped
calibration, final mechanism prediction, native/coarse measurement comparison
and independent final test are **not run**, because their qualification gate
is unsatisfied. Absence of these results is not a negative model-performance
result.

A continuation first needs an explicitly revised geometric support construction,
verified timestamp/pose and coordinate mapping, and spatial groups that merge
revisited regions. It must use new fixed qualification evidence. It must not
turn unknown regions into safe ones, relax the current screen after results,
or add VLM/relation/RL modules to bypass the unresolved interface.
