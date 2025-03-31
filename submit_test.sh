#!/bin/bash 
#SBATCH -J LangYa_test
#SBATCH -p gpu                 
#SBATCH -N 4                
#SBATCH --ntasks-per-node=4     
#SBATCH --cpus-per-task=16        
#SBATCH --output=logs/langya_test_%J.log
#SBATCH --gres=gpu:4                 
#SBATCH --mem=900G                   
#SBATCH --time=0-00:00:00

export HDF5_USE_FILE_LOCKING=FALSE
export NCCL_NET_GDR_LEVEL=PHB

export MASTER_ADDR=$(hostname)

# load conda
module load apps/anaconda3/2021.05

# load CUDA and conda environment
export CUDA_HOME="/public/home/yangnan/env/CUDA/CUDA11.8"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$CUDA_HOME/mylib/lib64:$LD_LIBRARY_PATH"
source activate large-model-cuda118

# run the test script
set -x
srun -u --mpi=pmix_v3 \
    bash -c "
    source /public/home/yangnan/LangYa/code/langya_gitee/export_DDP_vars.sh
    python langya_test.py
    "
