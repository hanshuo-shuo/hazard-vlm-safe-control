#!/bin/bash
#SBATCH --job-name=pro-e3-independent-data
#SBATCH --account=p33100
#SBATCH --partition=short
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=00:40:00
#SBATCH --output=/projects/p33100/siosio/hazard_pro_decision_round_20260912/logs/e3_prepare_%j.out
#SBATCH --error=/projects/p33100/siosio/hazard_pro_decision_round_20260912/logs/e3_prepare_%j.err
set -euo pipefail
TASK_ROOT=/projects/p33100/siosio/hazard_pro_decision_round_20260912
PYTHON=/home/shv7753/envs/hazard-exp01b-r2/bin/python
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
cd "$TASK_ROOT/research/pro_decision_round_20260912"
"$PYTHON" test_round.py
"$PYTHON" test_certification.py
"$PYTHON" -c 'from common import *; freeze(Path(sys.argv[1]), "E3", ["common.py", "geometry.py", "prepare_e2.py", "prepare_e3.py", "run_e3.py", "aggregate_e3.py", "test_certification.py", "e3_prepare.sbatch", "e3_gpu.sbatch", "e3_report.sbatch"])' "$TASK_ROOT"
"$PYTHON" prepare_e3.py --root "$TASK_ROOT"
