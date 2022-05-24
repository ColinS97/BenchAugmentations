#!/bin/bash

create_new_environment() {
    # Create a new virtual environment
    # $1: ID of the SLURM batch Job
    # creates a new environment and installs the necesary packages
    SLURM_JOB_ID=$1
    COMPUTE_WS_NAME=pyjob_$SLURM_JOB_ID
    COMPUTE_WS_PATH=$(ws_allocate -F ssd $COMPUTE_WS_NAME 7)
    echo WS_Name: $COMPUTE_WS_NAME  
    echo WS_Path: $COMPUTE_WS_PATH
    virtualenv $COMPUTE_WS_PATH/pyenv
    source $COMPUTE_WS_PATH/pyenv/bin/activate
    pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 -f https://download.pytorch.org/whl/torch_stable.html
    pip install imgaug pandas pytorch-lightning lightning-bolts torchmetrics
}

remove_new_environment(){
    SLURM_JOB_ID=$1
    COMPUTE_WS_NAME=pyjob_$SLURM_JOB_ID
    deactivate
    ws_release -F ssd $COMPUTE_WS_NAME
}

create_or_reuse_environment() {
    ENV_PATH=/lustre/ssd/ws/cosi765e-pyjob_py39-cu11-torch-lightning
    if [ -d "$ENV_PATH" ] 
    then
        echo "Directory /path/to/dir exists."
        source $ENV_PATH/pyenv/bin/activate

    else
        echo "Error: Directory /path/to/dir does not exists."
        create_new_environment py39-cu11-torch-lightning
    fi
    
    
}