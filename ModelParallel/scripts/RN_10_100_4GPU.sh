#!/bin/bash
#SBATCH -N 1                        # number of nodes
#SBATCH -n 1                        # total number of MPI tasks (one task per node)
#SBATCH -t 0-3:00:00               # time in d-hh:mm:ss
#SBATCH --mem=200G                  # Total memory per node (increase based on your requirements)

#SBATCH --gres=gpu:a100:4                # Request 2 GPUs
#SBATCH -o slurm.out.RN_10_100_4GPU.%j.out # file to save job's STDOUT (%j = JobId)
#SBATCH -e slurm.err.RN_10_100_4GPU.%j.err # file to save job's STDERR (%j = JobId)
#SBATCH --mail-type=ALL             # Send an e-mail when a job starts, stops, or fails
#SBATCH --mail-user=pganesa4@asu.edu

#SBATCH --export=NONE               # Purge the job-submitting shell environment

python main_model_parallel.py --model_name ResNet --dataset_name CIFAR10 --epochs 100
python main_model_parallel.py --model_name ResNet --dataset_name CIFAR10 --epochs 100 --eval True