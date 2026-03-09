## IANUS — Integrative MSCA Achievements and Networks for Unified Strategies



---

## 1. Overview

This repository contains the fine-tuning pipeline developed by CINECA, which aims to maximise the strategic impact of Marie Skłodowska-Curie
Actions (MSCA) projects across the European Research Area (ERA).

The pipeline produces two fine-tuned language models, both based on
[Mistral 7B](https://huggingface.co/mistralai/Mistral-7B-v0.3) (or any compatible
open-weight model):

| Model | Input | Output |
|-------|-------|--------|
| **Summariser** | Full MSCA project proposal | Concise structured abstract |
| **Critic** | Individual proposal section (Excellence / Impact / Implementation) | Section-level qualitative evaluation aligned with EC scoring rubrics |

Both models are trained exclusively on real, non-synthetic documents — actual MSCA
proposals and their official evaluation reports — ensuring authentic language patterns and
genuine assessment criteria.

---

## 2. Project Structure

```
SUTRA/
├── env_config.sh                  # Environment setup and model download
├── requirements.txt               # Python dependencies
├── config_FSDP.sh                 #.sh to configure sharding strategy
├── summariser.py                  # full fine-tuning script (Summariser)
├── summarizer_lora.py             # LoRA fine-tuning script (Summariser)
├── summarizer_ddp.sh              #.sh launch script for ddp (Summarizer)
├── summarizer_fsdp.sh             #.sh lauch script for sfdp (Summarizer)
│
├── evaluation_summarizer.py       # file to run rouge evaluation on summarizer results
├── evaluation_rouge.sh            # .sh launch file for summarizer evaluation
├── baseline.py                    # Script to produce a baseline for the summarizer evaluation
|
├── critic_txt.py                  # Critic LoRa fine-tune
├── critic_ddp.sh                  #.sh lauch script for sfdp (Critic)
├── critic_fsdp.sh                 #.sh lauch script for sfdp (Critic)
|
└── out/                           # Output folder
```

---

## 3. Models

The default base model is **Mistral 7B Instruct v0.1**:

```
mistralai/Mistral-7B-v0.3
```
It gets automatically dowloaded when running env_config.sh

To use a different model, update `env_config.sh` before running
setup. Any Hugging Face model compatible with the `transformers` `AutoModelForCausalLM`
API can be substituted. See [Section 9](#9-hardware-requirements) for how to estimate GPU
requirements for other model sizes.

---

## 4. Dataset Format

Each training example is a dictionary object with a `messages` field following the standard
chat template structure:

```json
{
  "messages": [
    {
      "role": "system",
      "content": "<system prompt defining the model role>"
    },
    {
      "role": "user",
      "content": "<task label>:\n\n<full proposal text or section text>"
    },
    {
      "role": "assistant",
      "content": "<target abstract or evaluation text>"
    }
  ]
}
```

**Summariser examples** pair a full proposal (user turn) with its official abstract (assistant
turn). **Critic examples** pair a single proposal section — Excellence, Impact, or
Implementation — with the corresponding evaluator commentary.

All examples are shuffled at the dataset level before train/validation/test splits are applied,
to prevent position-based overfitting.

The chat template is applied automatically via `tokenizer.apply_chat_template()` using
the template defined in the model's `tokenizer_config.json`.

---

## 5. Installation

### Local (Linux)

```bash
#From inside the repository on the login node: 
# Run the setup script: creates a virtualenv and installs all dependencies
# Also downloads the base model from Hugging Face Hub
bash env_config.sh
```

The script will:
1. Create a Python virtual environment in `./venv`
2. Install all packages from `requirements.txt`
3. Download `mistralai/Mistral-7B-v0.3` (or another model when speified) to the
   local Hugging Face cache

A Hugging Face account and access token may be required for gated models. Set your token
before running:

```bash
export HF_TOKEN=your_huggingface_token
```



## 6. Configuration

### LoRA hyperparameters

Configured at the top of `summarizer_lora.py` and `critic_txt.py`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `LORA_RANK` | `16` | Rank of the low-rank adapter matrices |
| `LORA_ALPHA` | `4` | LoRA scaling factor |
| `LORA_DROPOUT` | `0.15` | Dropout rate on adapter layers |
| `TARGET_MODULES` | `q_proj, k_proj, v_proj, o_proj` | Attention layers to adapt |

### Training hyperparameters

Check both in the .sh and the .py file to pick and change hyperparameters. You can add parameters based on the SFTTraner and SFTConfig classes. 

### FSDP configuration (`config_FSDP.yaml`)

Review and update before running FSDP training:

```bash
compute_environment: LOCAL_MACHINE
debug: true
distributed_type: FSDP #MULTI_GPU per DDP
downcast_bf16: 'no'
enable_cpu_affinity: false
fsdp_config:
  fsdp_auto_wrap_policy: TRANSFORMER_BASED_WRAP
  fsdp_transformer_layer_cls_to_wrap: MistralDecoderLayer
  fsdp_backward_prefetch: BACKWARD_PRE
  fsdp_cpu_ram_efficient_loading: true
  fsdp_forward_prefetch: true
  fsdp_offload_params: true
  fsdp_sharding_strategy: FULL_SHARD
  fsdp_state_dict_type: SHARDED_STATE_DICT
  fsdp_sync_module_states: true
  fsdp_use_orig_params: true
machine_rank: 0
main_training_function: main
mixed_precision: bf16
num_machines: 1          # Set to your number of nodes
num_processes: 4         # Set to your number of GPUs
rdzv_backend: c10d
same_network: true
tpu_env: []
tpu_use_cluster: false
tpu_use_sudo: false
use_cpu: false                          
```

---

## 7. Usage

### 7.1 Creation of the environment
```bash
bash env_config.sh
```

### 7.2 Full Fine-Tuning (Summariser)

Updates all model parameters. Requires multi-GPU setup with FSDP.

```bash
sbatch summarizer_fsdp.sh 
```
At line 72, modify with the script you want to use. 

### 7.2 LoRA Fine-Tuning (Summariser)

Recommended approach. Trains only the low-rank adapter layers.

```bash
sbatch summarizer_ddp.sh 
```

### 7.3 LoRA Fine-Tuning (Critic)

```bash
sbatch critic_ddp.shh
```


### 7.4 Evaluation (Summariser)

Runs inference on the held-out test set with both the fine-tuned model and the unmodified
baseline, then computes ROUGE-1, ROUGE-2, and ROUGE-L scores.

```bash
sbatch evaluation_rouge.sh
```

A summary comparison table is printed to stdout and saved to `out/evaluation_results.csv`.

---
