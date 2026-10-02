#!/usr/bin/env bash
#SBATCH --job-name=libero10-merge
#SBATCH --account=grp_hbenamor
#SBATCH --partition=lightwork
#SBATCH --qos=public
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:10:00
#SBATCH --export=NONE
#SBATCH --chdir=/home/xzhan742/Coding/thesis
#SBATCH --output=/scratch/xzhan742/thesis/logs/%x-%j.out
#SBATCH --error=/scratch/xzhan742/thesis/logs/%x-%j.err

set -euo pipefail
module purge
module load mamba/latest
source activate /scratch/xzhan742/thesis/envs/smolvla-libero-py312-v3
python hpc/smolvla_libero/aggregate_eval.py "$1"
