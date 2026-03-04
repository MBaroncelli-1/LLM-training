#!/bin/bash
#SBATCH --job-name=summarizer_fsdp
#SBATCH --output=summarizer_fsdp_%j.out
#SBATCH --error=summarizer_fsdp_%j.err
#SBATCH --account=<insert account name>
#SBATCH --nodes=1  #change it depending on the size of the model
#SBATCH --ntasks=1 #change it depending on the size of the model
#SBATCH --cpus-per-task=16  #change it depending on the size of the model
#SBATCH --gres=gpu:2  #change it depending on the size of the model
#SBATCH --time=00:30:00  #change it depending on the chosen partition
#SBATCH --partition=boost_usr_prod   # Change with the partition you want to use
#SBATCH --exclusive
#SBATCH --qos=boost_qos_dbg # Replace with normal or the queue of your choice

###ENVIRONMENT###
module load cuda/12.2
source SUTRA/bin/activate

echo "SLURM_JOB_NUM_NODES=$SLURM_JOB_NUM_NODES"
echo "SLURM_NNODES=$SLURM_NNODES"
WANDB_MODE=offline

GPUS_PER_NODE=4
NNODES=$SLURM_NNODES
WORLD_SIZE=$((${NNODES}*${GPUS_PER_NODE}))


#### Set network #####
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)

#### Set experiment ####

#Models were downloaded to /leonardo_scratch/large/userinternal/$USER/public/models/hub
# by the configure_env.sh script. If you changed the location, update the following two variables.


MODEL_PATH="/leonardo_scratch/large/userinternal/$USER/public/models/hub/Mistral-7B-Instruct-v0.3"
OUTPUT_PATH="./out"
CONFIG_PATH="config_FSDP.sh"


#change these based on your preferred configuration
EPOCHS=6  
MAX_STEPS=-1
GPU_BS=2  



#### Define Launcher, Script and Training Args ####
#define your own configuration both here and in the SFDP config file

export LAUNCHER="accelerate launch \
    --config_file $CONFIG_PATH \
    --num_processes $WORLD_SIZE \
    --num_machines $SLURM_NNODES \
    --rdzv_backend c10d \
    --main_process_ip "$MASTER_ADDR" \
    --main_process_port 6000 \
    --machine_rank $SLURM_PROCID \
    "

export SCRIPT="summarizer_lora.py"
export PY_ARGS="--model_path=$MODEL_PATH \
                --output_path=$OUTPUT_PATH \
                --num_train_epochs=$EPOCHS \
                --max_steps=$MAX_STEPS \
                --per_device_train_batch_size=$GPU_BS"

export CMD="$LAUNCHER $SCRIPT $PY_ARGS"
echo "$CMD"

#### Launch ####
srun $CMD