#!/bin/bash
#SBATCH --job-name=interaction-secondary
#SBATCH --account=p33100
#SBATCH --partition=short
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --output=/projects/p33100/siosio/hazard_cancellation_audit_20260912/logs/secondary_%j.out
#SBATCH --error=/projects/p33100/siosio/hazard_cancellation_audit_20260912/logs/secondary_%j.err
set -euo pipefail
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
/home/shv7753/envs/hazard-exp01b-r2/bin/python /projects/p33100/siosio/hazard_cancellation_audit_20260912/research/cancellation_audit_20260912/secondary_factorial.py
