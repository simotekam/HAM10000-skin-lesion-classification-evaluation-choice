#!/bin/bash
#SBATCH --job-name=ham
#SBATCH --partition=gpu
#SBATCH --gpus-per-node=rtx_pro_6000:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --array=0-9
#SBATCH --output=logs/%x_%A_%a.out

# One array task = one run = (split, config, seed), mapped by RUN_TABLE in src/train.py.
#   Tier 1: --array=0-9    (config full, split A and B, seeds 0-4)
#   Tier 2: --array=10-19  (split B, no_fl and no_cw, seeds 0-4)
#   Tier 3: --array=20-24  (split B, plain, seeds 0-4)

# No `set -u`: Lmod's `module` shell function references unset variables internally.
set -eo pipefail

PROJECT_DIR=$HOME/healthcare_ai_project
DATA_DIR=/scratch/$USER/ham10000
RESULTS_DIR=/scratch/$USER/ham_results

module purge
module load Python/3.13.1-GCCcore-14.2.0
source $HOME/venvs/ham_venv/bin/activate

mkdir -p logs "$RESULTS_DIR"

# Copy the preprocessed array to fast node-local disk; it is read every epoch.
cp "$DATA_DIR/ham10000_224.npy" "$DATA_DIR/ham10000_index.csv" "$TMPDIR/"

nvidia-smi
python -u "$PROJECT_DIR/src/train.py" \
  --task-id "$SLURM_ARRAY_TASK_ID" \
  --data-dir "$TMPDIR" \
  --splits-dir "$PROJECT_DIR/splits" \
  --output-dir "$RESULTS_DIR" \
  --num-workers 8
