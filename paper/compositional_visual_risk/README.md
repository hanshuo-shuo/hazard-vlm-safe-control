# Working-paper artifact

The manuscript reports the actual diagnostic findings and their limitations. It is not a submitted
paper, a formal C³-Safe gate approval, or a claim that a VLM-distilled actor has been trained.

- `manuscript.md`: complete editable English draft.
- `manuscript.template.md`: text source with data insertion points.
- `build_paper.py`: inserts verified numeric findings and builds figures/PDF using Matplotlib and ReportLab.
- `NUMERIC_CLAIMS.json`: text/table values inserted from the measured results.
- `figures/`: PNG previews and vector PDF exports.
- Final PDF: `../../output/pdf/compositional_visual_risk_working_paper.pdf`.
- Chinese explanation: `../../docs/C3_PAPER_PROGRESS_2026-09-12.md`.

To preserve edits across regeneration, edit the template or builder as appropriate; the builder
overwrites `manuscript.md` and NUMERIC_CLAIMS.json. It reads the experiment summary and historical
RESULTS.json files and the raw prediction arrays for the fixed first-scene illustration.

The draft was generated with the bundled Codex Python 3.12 runtime, ReportLab, NumPy, Pillow,
Matplotlib 3.10.8 and DejaVu fonts. Matplotlib dependencies were installed only under the temporary
directory `/private/tmp/hazard-paper-plot-deps`; the research training environment was not modified.

```bash
PYTHONPATH=/private/tmp/hazard-paper-plot-deps \
  /Users/hanshuo/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 \
  paper/compositional_visual_risk/build_paper.py
```

On another machine, install Matplotlib/ReportLab/NumPy/Pillow in an isolated environment and set
`PAPER_FONT_DIR` to a directory containing DejaVuSerif, DejaVuSerif-Bold, DejaVuSans and DejaVuSans-Bold
TTF files. Run the builder from any working directory. The artifact paths are derived from its source.

References were checked against primary paper/proceedings pages on 2026-09-12. Related-work
descriptions are scoped to the mechanisms documented there; the draft does not claim comparative
performance against those systems. This internal draft has no assigned author list or target venue.
