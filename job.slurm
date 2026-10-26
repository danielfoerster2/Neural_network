#!/bin/bash

#SBATCH --job-name=train               # label in squeue; becomes %x in the log name
#SBATCH --account=rsk@v100             # allocation charged; also routes to the V100 nodes, i.e. GPUs, change to "rsk@cpu" in case of CPU
#SBATCH --nodes=1                      # one machine GPU, in case of CPU 2 CPU with 20 cores, ie 40 threads, without hyperthreading
#SBATCH --ntasks=1                     # one process (a single python); in case of CPU set to the number of threads you want; e.g. 40 in case of 1 node, or 80 for two nodes, or 80 for one core and hyperthreading enabled
#SBATCH --gres=gpu:1                  # one GPU; a node has four ; remove this line for a CPU only job
#SBATCH --cpus-per-task=10             # 10 physical cores = 1/4 node = 40 GB RAM; keep 10 per GPU, set to 1 in case of cpu
#SBATCH --hint=nomultithread           # count physical cores, not hardware threads; otherwise "--hint=multithread" for the use of hyperthreading
#SBATCH --time=05:00:00                 # hh:mm:ss; 10 min default; max 20 h (100 h with qos_gpu-t4)
#SBATCH --output=slurm_%x_%j.out       # stdout and stderr together; %j = job id
##SBATCH --qos=qos_gpu-dev             # tests up to 2 h, starts faster
##SBATCH --qos=qos_gpu-t4              # jobs longer than 20 h, up to 100 h
##SBATCH --constraint=v100-32g         # only if 16 GB of GPU memory is not enough

# --- environment ------------------------------------------------------------
cd "${SLURM_SUBMIT_DIR}"                       # where sbatch was run; relative paths below are from here
source /etc/profile.d/modules.sh   2>/dev/null || true
source /etc/profile.d/z_modules.sh 2>/dev/null || true
module purge                                   # ignore whatever the login shell had loaded
module load tensorflow-gpu/py3/2.16.1

# --- bookkeeping in the log ---------------------------------------------------
echo "job ${SLURM_JOB_ID} on $(hostname), started $(date)"
#nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true  # write details of the GPU to log

# --- the work -----------------------------------------------------------------
# Everything read here must already be on Jean Zay: compute nodes have no internet.

python3 train.py
rc=$?

echo "finished $(date) with exit code ${rc}"
