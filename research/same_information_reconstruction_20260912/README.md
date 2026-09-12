# One same-information reconstruction control

Pro requested one final interpretive control after `0c18646`, together with
completion of the nine-page manuscript. This round is finished scientifically:
**same coarse information can recover opportunities under another aggregator,
but the reconstruction also introduces unsafe accepts**. It is not a new repair
algorithm, a third mechanism confirmation, or a new safety certificate.

Read [PRO_REVIEW.md](PRO_REVIEW.md), [protocol.json](protocol.json),
[appended tables](artifacts/publication/TABLES.md), and
[verification](artifacts/verification/VERIFICATION.json).

## Fixed comparison and results

- Development: consumed E3 IID discovery, 2,000 scenes. Appended evaluation:
  the already consumed 2,000-scene oracle confirmation bank. No fresh scene.
- Primary input: perfect actual target64-to16 fractions. Secondary: all five
  existing restored model fields. No training or model selection.
- Uniform interpretation, one global cost shift, and one Parker-Youngs PLIC
  reconstruction all have identical coarse fields and F512 access.
- PLIC uses the fixed 3x3 weighted gradient (center weight 2), exact half-plane
  area inversion and exact fine-square integration. No property geometry or fine
  reference label reaches its input interface. Zero-gradient cells remain uniform.
- Global offset is selected only on discovery from the fixed 13-value bank,
  maximizing safe accepts without increasing development unsafe count. It is
  .001 for perfect-target input and 0 shared across the five learned models.
- Appended target PLIC recovers 44 of 54 lost safe opportunities, newly loses 0,
  and adds 16 unsafe accepts. The shift recovers 6 and adds 3 unsafe accepts.
- Appended learned PLIC recovers 83 model-scene decisions, loses 0, and adds 43
  unsafe accepts across five models sharing 2,000 scenes. Conditional unsafe
  frequency rises from 1.03% to 1.56%; eta rises from 96.77% to 97.81%.
- For 25 of those 43 newly unsafe learned events, the same action has opposing
  baseline error terms, improved target-measurement absolute error, and worse
  total absolute error. All events, including the remaining 18, are preserved.

The selected offset has a development risk constraint that PLIC does not have.
The full development offset bank is recorded in
[OFFSET_DEVELOPMENT.json](artifacts/OFFSET_DEVELOPMENT.json); larger
offsets recover more at higher risk. The selected comparison therefore does not
establish matched-risk superiority or that every global correction must fail.

## Computational checks

Six unchanged tests pass. The initial pre-data test failed on a transverse
roundoff residual of 8.33e-16. Pairing the same stencil's differences before
summation fixed it; the original attempt is retained in `attempts/6196439`.
No development scene was read before the repaired tests passed.

Independent verification replays 72,000 scalar decisions and 65,536 polygon-clipped
fine-square areas. Maximum polygon discrepancy is 3.33e-16; uniform lifting differs
from coarse integration by at most 3.71e-14. Every evaluated cell preserves its
coarse area fraction (maximum fine-grid error 3.33e-16 on the appended bank).
The 25-event cancellation pattern is reconstructed independently.

## Reproduction and storage

Quest root:
`/projects/p33100/siosio/hazard_same_information_reconstruction_20260912`.
Old model and oracle roots are read-only references in `protocol.json`.
Python is `/home/shv7753/envs/hazard-exp01b-r2/bin/python`.

Run in a new empty root; existing stage outputs cannot be overwritten:

1. `run.py --root ROOT --phase development`: tests, old-data CPU replay, PLIC,
   offset fitting and `OPERATOR_FREEZE.json`.
2. `run.py --root ROOT --phase appended`: locked implementation/offsets on the
   consumed bank. The source freeze remains unchanged.
3. `report.py --root ROOT`: full-scene benefits/harms, fixed/reselected analyses
   and complete event records with signed terms.
4. `verify.py --root ROOT`: independent scalar/polygon verification.
5. `publication.py`, `paper_figures.py`: report-scale and paper-scale figures.

Large arrays and model assets stay on Quest. `artifacts/` contains compact
reports, full changed-event records, freezes, logs and figures. Its full-stage
manifests can also name large Quest-only arrays; those files are not claimed to
be in this Git archive.

Paper building uses the existing `texlive/2026` module. The legacy Poppler module
could not load libpng12; a pinned PyMuPDF 1.26.7 renderer was installed only in
this round's `paper_deps`. Original scientific environments are unchanged.
LaTeX syntax, BibTeX path and renderer failures are preserved in build logs.
The official style files remain byte-for-byte unchanged; only draft-status text
is replaced in the manuscript's own preamble.

Final manuscript build `6197893` has 9 main-text pages and 13 total pages, all
visually inspected. Delivery job `6198386` extracts and independently compiles
the source ZIP: all 13 pages have identical text and 110-dpi rendered pixels.
It also verifies 21 stage-manifest entries, 14 old inputs and both code freezes.
See [DELIVERY.json](artifacts/delivery/DELIVERY.json) and
[UTC job accounting](artifacts/JOB_ACCOUNTING_UTC.txt). This does not assert
completed human author review. The deliverable ZIP is manuscript source only;
the complete anonymous experimental supplement remains to be finalized.

The numerical construction is the established first-order Parker-Youngs PLIC
described by Pilliod and Puckett (2004), not their second-order ELVIRA method.
The weighted stencil is also described in the authors' accessible preprint,
section 2.5: [primary preprint](https://escholarship.org/content/qt4nc7s7bp/qt4nc7s7bp_noSplash_326b41398b696b3d80df8e08d3b17258.pdf).
New computation implements the area integration directly, with an independent
polygon-clipping check.

This experiment is closed regardless of which result is most appealing. No
automatic extra model, reconstruction, offset fit on appended labels, calibration,
new confirmation, RELLIS, VLM or RL run follows it.
