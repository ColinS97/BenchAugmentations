#!/bin/bash
                           
#SBATCH --no-requeue
#SBATCH --partition=alpha
#SBATCH --nodes=1                   
#SBATCH --gres=gpu:1
#SBATCH --mem=8000
#SBATCH --mincpus=1
#SBATCH --time=24:00:00                             
#SBATCH --job-name=rand
#SBATCH --mail-type=ALL
#SBATCH --mail-user=kevin.kirsten@mailbox.tu-dresden.de
#SBATCH --output=output-rand-%j.out

##SBATCH --error=error-%j.err

module --force purge                          				
module load modenv/hiera CUDA/11.3.1 GCC/11.2.0 Python/3.9.6

COMPUTE_WS_NAME=pyjob_$SLURM_JOB_ID
COMPUTE_WS_PATH=$(ws_allocate -F ssd $COMPUTE_WS_NAME 7)
echo WS_Name: $COMPUTE_WS_NAME
echo WS_Path: $COMPUTE_WS_PATH

cd /home/keki996e/pytorch/BenchAugmentations/pytorch-cifar

virtualenv $COMPUTE_WS_PATH/pyenv
source $COMPUTE_WS_PATH/pyenv/bin/activate

cd /home/keki996e/pytorch/BenchAugmentations/pytorch-cifar

pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 -f https://download.pytorch.org/whl/torch_stable.html
pip install imgaug pandas


python main.py --epochs 200 --randaugment

deactivate

ws_release -F ssd $COMPUTE_WS_NAME
