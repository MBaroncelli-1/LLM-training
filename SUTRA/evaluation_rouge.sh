#!/bin/bash
#SBATCH --job-name=evaluation
#SBATCH --output=evaluation_%j.out
#SBATCH --error=evaluation_%j.err
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


python evaluation_summarizer.py
python baseline.py
