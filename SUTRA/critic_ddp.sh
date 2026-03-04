#!/bin/bash
#SBATCH --job-name=critic_ddp
#SBATCH --output=critic_ddp%j.out
#SBATCH --errorcritic_ddp_%j.err
#SBATCH --account=<insert account name>
#SBATCH --nodes=1  #change it depending on the size of the model
#SBATCH --ntasks=1 #change it depending on the size of the model
#SBATCH --cpus-per-task=16  #change it depending on the size of the model
#SBATCH --gres=gpu:2  #change it depending on the size of the model
#SBATCH --time=00:30:00  #change it depending on the chosen partition
#SBATCH --partition=boost_usr_prod
#SBATCH --exclusive
#SBATCH --qos=boost_qos_dbg

### ENVIRONMENT ###
module load cuda/12.2
source SUTRA/bin/activate

export WANDB_MODE=offline

echo "Node: $HOSTNAME"
echo "GPUs allocated: $CUDA_VISIBLE_DEVICES"

# ==========================
# EXPERIMENT CONFIG
# ==========================

#Models were downloaded to /leonardo_scratch/large/userinternal/$USER/public/models/hub
# by the configure_env.sh script. If you changed the location, update the following two variables.

MODEL_PATH="/leonardo_scratch/large/userinternal/$USER/public/models/hub/Mistral-7B-Instruct-v0.3"
OUTPUT_PATH="./out"

# ==========================
# LAUNCH TRAINING (DDP automatico)
# ==========================

accelerate launch \
    --multi_gpu \
    --num_processes=2 \
    critic_txt.py \
    --model_path $MODEL_PATH \
    --output_path $OUTPUT_PATH \
    --num_train_epochs $EPOCHS \
    --max_steps $MAX_STEPS \
    --per_device_train_batch_size $GPU_BS