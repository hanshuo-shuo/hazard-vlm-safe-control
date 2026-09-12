#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")"
mkdir -p build
pdflatex -interaction=nonstopmode -halt-on-error -output-directory build main.tex
cp references.bib iclr2027_conference.bst build/
(
  cd build
  bibtex main
)
pdflatex -interaction=nonstopmode -halt-on-error -output-directory build main.tex
pdflatex -interaction=nonstopmode -halt-on-error -output-directory build main.tex
