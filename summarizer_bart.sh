#!/bin/bash
#SBATCH --out=%j.out
#SBATCH --err=%j.err
#SBATCH --account=PHD_baroncel
#SBATCH --nodes=1
#SBATCH -p boost_usr_prod
#SBATCH --time 00:30:00
#SBATCH --cpus-per-task=32
#SBATCH --tasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --exclusive
#SBATCH --qos=boost_qos_dbg

###ENVIRONMENT###
module load cuda/12.2
source /leonardo_work/PHD_baroncel/sutra_env/bin/activate

echo "SLURM_JOB_NUM_NODES=$SLURM_JOB_NUM_NODES"
echo "SLURM_NNODES=$SLURM_NNODES"


GPUS_PER_NODE=4
NNODES=$SLURM_NNODES
WORLD_SIZE=$((${NNODES}*${GPUS_PER_NODE}))


#### Set network #####
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)

#### Set experiment ####
MODEL_PATH="/leonardo_scratch/large/userinternal/mbaronce/models/bert-base/"
OUTPUT_PATH="./out"
#CONFIG_PATH="config_FSDP.sh"
EPOCHS=1
MAX_STEPS=-1     #consider only EPOCHS
GPU_BS=1


#### Define Launcher, Script and Training Args ####
export LAUNCHER="accelerate launch \
    --num_processes $WORLD_SIZE \
    --num_machines $SLURM_NNODES \
    --rdzv_backend c10d \
    --main_process_ip "$MASTER_ADDR" \
    --main_process_port 6000 \
    --machine_rank $SLURM_PROCID \
    "

export SCRIPT="summarizer_bart.py"
export PY_ARGS="--model_path=$MODEL_PATH \
                --output_path=$OUTPUT_PATH \
                --num_train_epochs=$EPOCHS \
                --max_steps=$MAX_STEPS \
                --per_device_train_batch_size=$GPU_BS"

export CMD="$LAUNCHER $SCRIPT $PY_ARGS"
echo "$CMD"

#### Launch ####
srun $CMD

