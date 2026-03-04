import torch
import time
import argparse 
import matplotlib.pyplot as plt
import torch.distributed as dist
import yaml
from datasets import load_from_disk, Dataset
from pathlib import Path
import re


from trl import SFTTrainer, SFTConfig
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from accelerate import Accelerator
accelerator = Accelerator()


if accelerator.is_main_process:
    import wandb
    wandb.init(
        project="mistral-summarization", 
        name=f"run-{time.strftime('%Y%m%d-%H%M%S')}",
        mode="offline"  
    )

####------------------CRITIC CON SFTTRAINER-----------###
#Uso modello mistralai/Mistral-7B-Instruct-v0.3.
#Obiettivo: fine-tune con sfttrainer con conversational language modeling ("messages")


###-------------------TOKENIZER---------------------------###
tokenizer = AutoTokenizer.from_pretrained("/leonardo_scratch/large/userinternal/$USER/public/models/hub/Mistral-7B-Instruct-v0.3")
tokenizer.pad_token = tokenizer.eos_token

###-----------------------ARGUMENTS------------------###
def parse_args():
    parser = argparse.ArgumentParser(description="Arguments via CLI")
    parser.add_argument("--model_path", type=str)
    parser.add_argument("--output_path", type=str)
    parser.add_argument("--num_train_epochs", type=int, default=3)
    parser.add_argument("--max_steps",type=int,default=-1)
    parser.add_argument("--per_device_train_batch_size", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    return parser.parse_args()

# Parse prguments 
args = parse_args()


base_path = Path("/leonardo_work/AIFAC_F01_011")




def normalize_name(name):
    return name.lower().replace("-", "").replace("_", "").replace(" ", "").strip()


def extract_proposal_sections(text):

    flags = re.IGNORECASE | re.MULTILINE

    pattern_excellence = (
        r"^\s*1\s*\.?\s*excellence\s*"
        r"([\s\S]*?)"
        r"(?=^\s*2\s*\.?\s*impact\s*)"
    )

    pattern_impact = (
        r"^\s*2\s*\.?\s*impact\s*"
        r"([\s\S]*?)"
        r"(?=^\s*3\s*\.?\s*quality\s+and\s+efficiency\s+of\s+the\s+implementation\s*)"
    )

    pattern_method = (
        r"^\s*3\s*\.?\s*quality\s+and\s+efficiency\s+of\s+the\s+implementation\s*"
        r"([\s\S]*)"
    )

    sections = {}

    match_excellence = re.search(pattern_excellence, text, flags)
    match_impact = re.search(pattern_impact, text, flags)
    match_method = re.search(pattern_method, text, flags)

    sections["excellence"] = match_excellence.group(1).strip() if match_excellence else None
    sections["impact"] = match_impact.group(1).strip() if match_impact else None
    sections["method"] = match_method.group(1).strip() if match_method else None

    return sections

def extract_esr_sections(text):

    flags = re.IGNORECASE | re.MULTILINE

    pattern_excellence = (
        r"^\s*criterion\s*1\s*[-:]\s*excellence\s*"
        r"([\s\S]*?)"
        r"(?=^\s*criterion\s*2\s*[-:]\s*impact\s*)"
    )

    

    pattern_impact = (
        r"^\s*criterion\s*2\s*[-:]\s*impact\s*"
        r"([\s\S]*?)"
        r"(?=^\s*criterion\s*3\s*[-:]\s*quality\s+and\s+efficiency\s+of\s+the\s+implementation\s*)"
    )

    pattern_method = (
        r"^\s*criterion\s*3\s*[-:]\s*quality\s+and\s+efficiency\s+of\s+the\s+implementation\s*"
        r"([\s\S]*?)"
        r"(?=^\s*scope\s+of\s+the\s+application\s*)"
    )

    sections = {}

    match_excellence = re.search(pattern_excellence, text, flags)
    match_impact = re.search(pattern_impact, text, flags)
    match_method = re.search(pattern_method, text, flags)

    sections["excellence"] = match_excellence.group(1).strip() if match_excellence else None
    sections["impact"] = match_impact.group(1).strip() if match_impact else None
    sections["method"] = match_method.group(1).strip() if match_method else None

    return sections



proposal_files = list(base_path.rglob("*Proposal.txt"))
review_files = list(base_path.rglob("*ESR.txt"))

print("\n================ FILE SANITY CHECK ================")
print("Proposal files found:", len(proposal_files))
print("ESR files found:", len(review_files))
print("===================================================\n")

# ==========================================================
# BUILD DICTIONARIES
# ==========================================================

proposal_dict = {}
review_dict = {}

for file_path in proposal_files:
    project_name = normalize_name(file_path.stem.replace("_Proposal", ""))
    with open(file_path, "r", encoding="utf-8") as f:
        proposal_dict[project_name] = extract_proposal_sections(f.read())

for file_path in review_files:
    project_name = normalize_name(file_path.stem.replace("_ESR", ""))
    with open(file_path, "r", encoding="utf-8") as f:
        review_dict[project_name] = extract_esr_sections(f.read())

# ==========================================================
# MATCH PROJECTS
# ==========================================================

common_projects = set(proposal_dict.keys()) & set(review_dict.keys())

print("\n================ PROJECT MATCH CHECK ================")
print("Matched projects:", len(common_projects))
print("Proposals without ESR:", len(set(proposal_dict) - set(review_dict)))
print("ESR without Proposal:", len(set(review_dict) - set(proposal_dict)))
print("=====================================================\n")

# ==========================================================
# SECTION SANITY CHECK
# ==========================================================

missing_counter = {
    "proposal_excellence": 0,
    "proposal_impact": 0,
    "proposal_method": 0,
    "esr_excellence": 0,
    "esr_impact": 0,
    "esr_method": 0,
}

for project in common_projects:

    p = proposal_dict[project]
    r = review_dict[project]

    if not p["excellence"]: missing_counter["proposal_excellence"] += 1
    if not p["impact"]: missing_counter["proposal_impact"] += 1
    if not p["method"]: missing_counter["proposal_method"] += 1

    if not r["excellence"]: missing_counter["esr_excellence"] += 1
    if not r["impact"]: missing_counter["esr_impact"] += 1
    if not r["method"]: missing_counter["esr_method"] += 1

print("\n================ MISSING SECTIONS ================")
for k, v in missing_counter.items():
    print(k, ":", v)
print("==================================================\n")

# ==========================================================
# SAMPLE DEBUG
# ==========================================================

print("\n================ SAMPLE EXTRACTION ================")

for project in list(common_projects)[:2]:
    print(f"\nPROJECT: {project}")
    for section in ["excellence", "impact", "method"]:
        prop = proposal_dict[project].get(section)
        rev = review_dict[project].get(section)

        print(f"\n--- {section.upper()} ---")
        print("Proposal length:", len(prop) if prop else 0)
        print("ESR length:", len(rev) if rev else 0)

        if prop:
            print("Proposal snippet:", prop[:200])
        if rev:
            print("ESR snippet:", rev[:200])

print("\n===================================================\n")

# ==========================================================
# DATASET CREATION
# ==========================================================

def format_dataset(proposal_dict, review_dict, tokenizer, max_seq_length=4096, buffer=50):

    formatted = []

    for project in common_projects:

        for section in ["excellence", "impact", "method"]:

            proposal_text = proposal_dict[project].get(section)
            review_text = review_dict[project].get(section)

            if not proposal_text or not review_text:
                continue

            proposal_tokens = tokenizer(
                proposal_text,
                add_special_tokens=False
            )["input_ids"]

            review_tokens = tokenizer(
                review_text,
                add_special_tokens=False
            )["input_ids"]

            available = max_seq_length - len(review_tokens) - buffer
            if available <= 0:
                continue

            proposal_tokens = proposal_tokens[:available]
            proposal_truncated = tokenizer.decode(
                proposal_tokens,
                skip_special_tokens=True
            )

            messages = [
                {
                    "role": "user",
                    "content": f"Summarize the {section} section of the proposal:\n\n{proposal_truncated}"
                },
                {
                    "role": "assistant",
                    "content": review_text
                }
            ]

            formatted.append({"messages": messages})

    return formatted


dataset = format_dataset(
    proposal_dict,
    review_dict,
    tokenizer,
)

print("\n================ DATASET CHECK ================")
print("Total training examples:", len(dataset))
print("Maximum possible:", len(common_projects) * 3)
print("================================================\n")

# ==========================================================
# HF DATASET
# ==========================================================

hf_dataset = Dataset.from_list(dataset)
hf_dataset = hf_dataset.shuffle(seed=42)

split = hf_dataset.train_test_split(test_size=0.1, seed=42)

train_dataset = split["train"]
test_dataset = split["test"]

print("Train size:", len(train_dataset))
print("Test size:", len(test_dataset))

test_dataset.save_to_disk("./test_dataset")
print("Test dataset saved to ./test_dataset")


###-----------------------QUANTIZATION CONFIG-------------###

quantization_config=BitsAndBytesConfig(load_in_4bit=True, 
                                       bnb_4bit_compute_dtype=torch.bfloat16, 
                                       bnb_4bit_quant_type="nf4", 
                                       bnb_4bit_use_double_quant=True, 
                                       bnb_4bit_quant_storage=torch.bfloat16
                                       )




###-----------------------MODEL CONFIGURATION----------------###

model = AutoModelForCausalLM.from_pretrained(
    args.model_path, 
    dtype=torch.bfloat16, 
    quantization_config=quantization_config,
    attn_implementation="eager"             
    )


###--------------------LORA CONGIF-----------------------###

# Define LoRA config and model
lora_config=LoraConfig(
    lora_alpha=4,
    lora_dropout=0.15,
    r=16,
    bias="none",
    target_modules= ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",],
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
model.config.use_cache = False



training_args = SFTConfig(
    output_dir=args.output_path,
    dataset_text_field= "messages",


    num_train_epochs=args.num_train_epochs,
    max_steps=args.max_steps,
    lr_scheduler_type="cosine",
    #warmup_ratio=0.03,
    remove_unused_columns=False,


    per_device_train_batch_size=args.per_device_train_batch_size,  
    gradient_accumulation_steps=4,    
    gradient_checkpointing=False,    # for error 
    gradient_checkpointing_kwargs={'use_reentrant': False},     # for error
    bf16=True,


    optim="adamw_torch",
    learning_rate=args.learning_rate,   
    ddp_find_unused_parameters=False,
    prediction_loss_only=True,



    logging_steps=1,
    save_strategy="epoch",
    eval_strategy="epoch",
    eval_steps=1,     


    report_to="wandb"

)

trainer = SFTTrainer(
    model=model, 
    args=training_args,
    train_dataset=train_dataset, 
    eval_dataset=test_dataset,    
    processing_class=tokenizer,
)



if accelerator.is_main_process: print(f"[INFO] Starting fine-tuning.")
s = time.time()
trainer.train()
accelerator.wait_for_everyone()
e = time.time()
if accelerator.is_main_process: print(f"[INFO] Fine-tuning done.")
if accelerator.is_main_process: print(f"[INFO] Elapsed fine-tuining time {e-s} sec.")


model.save_pretrained(args.output_path)
if accelerator.is_main_process: print(f"[INFO] Final fine-tuned model saved to {args.output_path}")
accelerator.end_training()

# Access the log history
log_history = trainer.state.log_history
train_losses = [log["loss"] for log in log_history if "loss" in log]
epoch_train = [log["epoch"] for log in log_history if "loss" in log]
eval_losses = [log["eval_loss"] for log in log_history if "eval_loss" in log]
epoch_eval = [log["epoch"] for log in log_history if "eval_loss" in log]


# Plot the training loss
plt.plot(epoch_train, train_losses, label="Training Loss")
plt.plot(epoch_eval, eval_losses, label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training and Validation Loss per Epoch")
plt.legend()
plt.grid(True)
plt.savefig("Loss.png")

if accelerator.is_main_process: print(f"[INFO] Training and validation plot saved.")
