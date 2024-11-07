#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=26
#SBATCH --nodes=1

#SBATCH --mail-user=aqwang@berkeley.edu
#SBATCH --mail-type=ALL

#we perform hyperparameter tuning when training on the first image
python train1hyper.py > train1hyper.out
