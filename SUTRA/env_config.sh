#Run this from login node

# Config paths
HF_MODEL_DIR="/leonardo_scratch/large/userinternal/$USER/public/models/hub"

# Create python3 env
module load python/3.11.6--gcc--8.5.0
python3 -m venv SUTRA
module purge

# Install the required libraries
source SUTRA/bin/activate
pip3 install -r requirements.txt

# Download HF models
mkdir -p $HF_MODEL_DIR
export HF_HOME=$HF_MODEL_DIR
export HF_HUB_CACHE=$HF_MODEL_DIR
export HF_TOKEN=your_huggingface_token

# Mistral7B
huggingface-cli download mistralai/Mistral-7B-Instruct-v0.3
