#!/bin/bash
                           
#SBATCH --no-requeue
#SBATCH --partition=alpha
#SBATCH --nodes=1                   
#SBATCH --gres=gpu:1
#SBATCH --mem=8000
#SBATCH --mincpus=1
#SBATCH --time=08:00:00                             
#SBATCH --job-name=baseline
#SBATCH --mail-type=ALL
#SBATCH --mail-user=colin.simon@mailbox.tu-dresden.de
#SBATCH --output=output-base-%j.out
#SBATCH --error=error-base-%j.out

module --force purge                          				
module load modenv/hiera CUDA/11.3.1 GCC/11.2.0 Python/3.9.6

source lib.sh

create_or_reuse_environment

cd /scratch/ws/0/cosi765e-python_virtual_environment/BenchAugmentations/pytorch-cifar

python main_lightning.py --epochs 100 --baseline
