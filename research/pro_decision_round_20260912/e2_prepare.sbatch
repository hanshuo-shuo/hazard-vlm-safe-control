#!/bin/bash
#SBATCH --job-name=pro-e2-matched-data
#SBATCH --account=p33100
#SBATCH --partition=short
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=00:25:00
#SBATCH --output=/projects/p33100/siosio/hazard_pro_decision_round_20260912/logs/e2_prepare_%j.out
#SBATCH --error=/projects/p33100/siosio/hazard_pro_decision_round_20260912/logs/e2_prepare_%j.err
set -euo pipefail
TASK_ROOT=/projects/p33100/siosio/hazard_pro_decision_round_20260912
PYTHON=/home/shv7753/envs/hazard-exp01b-r2/bin/python
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
cd "$TASK_ROOT/research/pro_decision_round_20260912"
"$PYTHON" test_round.py
"$PYTHON" -c 'from common import *; freeze(Path(sys.argv[1]), "E2", ["common.py", "geometry.py", "prepare_e2.py", "run_e2.py", "aggregate_e2.py", "e2_prepare.sbatch", "e2_gpu.sbatch", "e2_report.sbatch"])' "$TASK_ROOT"
"$PYTHON" prepare_e2.py --root "$TASK_ROOT"
