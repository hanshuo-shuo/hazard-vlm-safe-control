#!/bin/bash
#SBATCH --job-name=pro-final-verification
#SBATCH --account=p33100
#SBATCH --partition=short
#SBATCH --cpus-per-task=4
#SBATCH --mem=20G
#SBATCH --time=00:30:00
#SBATCH --output=/projects/p33100/siosio/hazard_pro_decision_round_20260912/logs/final_report_%j.out
#SBATCH --error=/projects/p33100/siosio/hazard_pro_decision_round_20260912/logs/final_report_%j.err
set -euo pipefail
TASK_ROOT=/projects/p33100/siosio/hazard_pro_decision_round_20260912
PYTHON=/home/shv7753/envs/hazard-exp01b-r2/bin/python
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1
export MPLCONFIGDIR="$TASK_ROOT/matplotlib_cache"
cd "$TASK_ROOT/research/pro_decision_round_20260912"
"$PYTHON" -c 'from common import *; freeze(Path(sys.argv[1]), "REPORT", ["common.py", "verify_round.py", "report_round.py", "final_report.sbatch"])' "$TASK_ROOT"
"$PYTHON" verify_round.py --root "$TASK_ROOT"
PYTHONPATH=/projects/p33100/siosio/hazard_independent_confirmation_20260912/report_deps "$PYTHON" report_round.py --root "$TASK_ROOT"
