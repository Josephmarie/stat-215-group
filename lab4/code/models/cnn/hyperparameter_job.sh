#!/bin/bash

# EXAMPLE USAGE:
# sbatch hyperparameter_job.sh <wandb_api_key>

#SBATCH --job-name=lab4-cnn-hyperparameter-search
#SBATCH --partition=jsteinhardt
#SBATCH --gres=gpu:A4000:1

#SBATCH --mail-user=bogdankostic@berkeley.edu
#SBATCH --mail-type=ALL

# Set W&B API key
export WANDB_API_KEY=$1

# Run the hyperparameter search script
python hyperparameter_search.py > hyperparameter_search.out
