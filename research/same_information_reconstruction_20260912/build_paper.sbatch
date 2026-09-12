#!/bin/bash
#SBATCH --job-name=coarse-paper-pdf
#SBATCH --account=p33100
#SBATCH --partition=short
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=00:20:00
#SBATCH --output=/projects/p33100/siosio/hazard_same_information_reconstruction_20260912/logs/paper_%j.out
#SBATCH --error=/projects/p33100/siosio/hazard_same_information_reconstruction_20260912/logs/paper_%j.err
set -euo pipefail
TASK_ROOT=/projects/p33100/siosio/hazard_same_information_reconstruction_20260912
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH=/projects/p33100/siosio/hazard_independent_confirmation_20260912/report_deps
export MPLCONFIGDIR="$TASK_ROOT/matplotlib_cache"
/home/shv7753/envs/hazard-exp01b-r2/bin/python "$TASK_ROOT/research/same_information_reconstruction_20260912/case_table.py" --root "$TASK_ROOT"
if [ ! -d "$TASK_ROOT/paper_figures" ]; then
  /home/shv7753/envs/hazard-exp01b-r2/bin/python "$TASK_ROOT/research/same_information_reconstruction_20260912/paper_figures.py" --root "$TASK_ROOT"
fi
module load texlive/2026
PAPER_ROOT="$TASK_ROOT/paper/iclr2027"
BUILD_DIR="$PAPER_ROOT/build_${SLURM_JOB_ID}"
mkdir -p "$PAPER_ROOT/figures" "$BUILD_DIR/pages"
cp "$TASK_ROOT/paper_figures/"*.png "$PAPER_ROOT/figures/"
cd "$PAPER_ROOT"
pdflatex -interaction=nonstopmode -halt-on-error -file-line-error -output-directory "$BUILD_DIR" main.tex > "$BUILD_DIR/first_compile.txt"
cp "$PAPER_ROOT/references.bib" "$PAPER_ROOT/iclr2027_conference.bst" "$BUILD_DIR/"
cd "$BUILD_DIR"
bibtex main > bibtex.txt
cd "$PAPER_ROOT"
pdflatex -interaction=nonstopmode -halt-on-error -file-line-error -output-directory "$BUILD_DIR" main.tex > "$BUILD_DIR/second_compile.txt"
pdflatex -interaction=nonstopmode -halt-on-error -file-line-error -output-directory "$BUILD_DIR" main.tex > "$BUILD_DIR/final_compile.txt"
export PYTHONPATH="$TASK_ROOT/paper_deps"
/home/shv7753/envs/hazard-exp01b-r2/bin/python "$TASK_ROOT/research/same_information_reconstruction_20260912/render_paper.py" --build "$BUILD_DIR"
grep -E 'FIRST-STATEMENTS-PAGE|Overfull|undefined|Warning' "$BUILD_DIR/main.log" || true
