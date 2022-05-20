#!/bin/bash
                           
#SBATCH --no-requeue
#SBATCH --partition=alpha
#SBATCH --nodes=1                   
#SBATCH --gres=gpu:4
#SBATCH --mem=8000
#SBATCH --mincpus=4
#SBATCH --time=08:00:00                             
#SBATCH --job-name=deepaugment
#SBATCH --mail-type=ALL
#SBATCH --mail-user=colin.simon@mailbox.tu-dresden.de
#SBATCH --output=output-base-%j.out

##SBATCH --error=error-%j.err

module --force purge                          				
module load modenv/hiera CUDA/11.3.1 GCC/11.2.0 Python/3.9.6

COMPUTE_WS_NAME=pyjob_$SLURM_JOB_ID
COMPUTE_WS_PATH=$(ws_allocate -F ssd $COMPUTE_WS_NAME 7)
echo WS_Name: $COMPUTE_WS_NAME
echo WS_Path: $COMPUTE_WS_PATH

cd /scratch/ws/0/cosi765e-python_virtual_environment/BenchAugmentations/pytorch-cifar

virtualenv $COMPUTE_WS_PATH/pyenv
source $COMPUTE_WS_PATH/pyenv/bin/activate

cd /scratch/ws/0/cosi765e-python_virtual_environment/BenchAugmentations/pytorch-cifar

which python
which pip

pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 -f https://download.pytorch.org/whl/torch_stable.html
pip install imgaug pandas pytorch-lightning lightning-bolts torchmetrics


python main_lightning.py --epochs 100 --deepaugment

deactivate

ws_release -F ssd $COMPUTE_WS_NAME
