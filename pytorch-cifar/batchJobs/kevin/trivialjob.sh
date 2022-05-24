#!/bin/bash
                           
#SBATCH --no-requeue
#SBATCH --partition=alpha
#SBATCH --nodes=1                   
#SBATCH --gres=gpu:4
#SBATCH --mem=8000
#SBATCH --mincpus=4
#SBATCH --time=08:00:00                             
#SBATCH --job-name=trivialaugment
#SBATCH --mail-type=ALL
#SBATCH --mail-user=colin.simon@mailbox.tu-dresden.de
#SBATCH --output=output-trivialaugment-%j.out
#SBATCH --error=error-trivialaugment-%j.out


module --force purge                          				
module load modenv/hiera CUDA/11.3.1 GCC/11.2.0 Python/3.9.6

source lib.sh

create_or_reuse_environment

which python
which pip

cd /scratch/ws/0/cosi765e-python_virtual_environment/BenchAugmentations/pytorch-cifar

python main_lightning.py --epochs 10 --trivialaugment
