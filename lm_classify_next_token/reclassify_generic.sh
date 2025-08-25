#!/bin/bash
#SBATCH --job-name=rtg-infer
#SBATCH -p batch
#SBATCH -A marlowe-m000152-pm03
#SBATCH --gpus=1
#SBATCH --cpus-per-task=14
#SBATCH --mem=256G
#SBATCH --time=12:00:00
#SBATCH --array=0-7           # 0..NUM_SHARDS-1
#SBATCH -o logs/inf_%A_%a.out
#SBATCH -e logs/inf_%A_%a.err

set -euo pipefail
module load conda
. "$(conda info --base)/etc/profile.d/conda.sh"
conda activate comet

cd /scratch/m000152/comet/generic-resource-type/lm_classify_next_token

# Ensure .env is present here so python-dotenv can load it
IDX=${SLURM_ARRAY_TASK_ID}
python reclassify_generic.py inference "${IDX}"
