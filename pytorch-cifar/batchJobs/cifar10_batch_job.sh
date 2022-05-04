#!/bin/bash

#SBATCH --no-requeue
#SBATCH --partition=gpu2
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=8000
#SBATCH --mincpus=1
#SBATCH --time=24:00:00
#SBATCH --job-name=CIFAR10_TestRun
#SBATCH --mail-type=ALL
#SBATCH --mail-user=colin.simon@mailbox.tu-dresden.de
#SBATCH --output=output-%j.out
#SBATCH --error=error-%j.err



cd /scratch/ws/0/cosi765e-python_virtual_environment/BenchAugmentations/pytorch-cifar
module --force purge
module load modenv/hiera  GCC/10.2.0  CUDA/11.1.1  OpenMPI/4.0.5 PyTorch/1.9.0

python main.py --epochs 200
python main.py --epochs 200 --randaugment
