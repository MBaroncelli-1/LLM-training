import torch
import time
import argparse 
import matplotlib.pyplot as plt
import torch.distributed as dist
import yaml
from datasets import load_from_disk


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

####------------------SUMMARIZER CON SFTTRAINER-----------###
#Uso modello mistralai/Mistral-7B-Instruct-v0.3.
#Obiettivo: fine-tune con sfttrainer con conversational language modeling ("messages")


###-------------------TOKENIZER---------------------------###
tokenizer = AutoTokenizer.from_pretrained("/leonardo_scratch/large/userinternal/mbaronce/models/Mistral-7B-Instruct-v0.3")
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




###-------------------DATA PREPROCCESSING-----------------###
from pathlib import Path
import re
from datasets import Dataset



base_path = Path("/leonardo_work/AIFAC_F01_011")


def normalize_name(name):
    """
    normalizza il nome del progetto per evitare mismatch
    (maiuscole/minuscole, trattini, underscore, spazi).
    """
    return name.lower().replace("-", "").replace("_", "").replace(" ", "").strip()


def extract_abstract_txt(text):
    """
    estrae l'abstract dal file ESR.
    """
    pattern = r"Abstract\s*([\s\S]*?)(?=Evaluation Summary Report)"
    match = re.search(pattern, text, re.IGNORECASE)

    if match:
        return match.group(1).strip()

    return None


# ==============================
# BUILD PROPOSAL DICTIONARY
# ==============================

proposal_dict = {}

proposal_files = list(base_path.rglob("*Proposal.txt"))

for file_path in proposal_files:
    project_name = file_path.stem.replace("_Proposal", "")
    project_name = normalize_name(project_name)

    with open(file_path, "r", encoding="utf-8") as f:
        proposal_dict[project_name] = f.read()

print("Total proposals found:", len(proposal_dict))


# ==============================
# BUILD ABSTRACT DICTIONARY
# ==============================

review_dict = {}

review_files = list(base_path.rglob("*ESR.txt"))

for file_path in review_files:

    project_name = file_path.stem.replace("_ESR", "")
    project_name = normalize_name(project_name)

    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()
        abstract = extract_abstract_txt(text)

        if abstract is not None:
            review_dict[project_name] = abstract

print("Total abstracts found:", len(review_dict))


# ==============================
# MATCH PROPOSALS & ABSTRACTS
# ==============================

common_projects = set(proposal_dict.keys()) & set(review_dict.keys())
only_proposals = set(proposal_dict.keys()) - set(review_dict.keys())
only_abstracts = set(review_dict.keys()) - set(proposal_dict.keys())

print("\n==============================")
print("Matched projects:", len(common_projects))
print("Proposals without abstract:", len(only_proposals))
print("Abstracts without proposal:", len(only_abstracts))
print("==============================\n")

print("---- MATCHED PROJECTS ----")
for name in sorted(common_projects):
    print(name)

print("\n---- PROPOSALS WITHOUT ABSTRACT ----")
for name in sorted(only_proposals):
    print(name)

print("\n---- ABSTRACTS WITHOUT PROPOSAL ----")
for name in sorted(only_abstracts):
    print(name)


# ==============================
# BUILD ORDERED LISTS
# ==============================

proposals = []
abstracts = []

for project in sorted(common_projects):
    proposals.append(proposal_dict[project])
    abstracts.append(review_dict[project])

print("Final dataset size:", len(proposals))


# ==============================
# SANITY CHECK 
# ==============================

print("\nSanity check examples:\n")
for i in range(min(3, len(proposals))):
    print("----")
    print("Proposal snippet:\n", proposals[i][:300])
    print("\nAbstract snippet:\n", abstracts[i][:300])
    print("\n")


# ==============================
# FORMAT FOR TRAINING
# ==============================

def format_dataset(proposals, abstracts, tokenizer, max_seq_length=4096, buffer=50):
    formatted = []

    for proposal, abstract in zip(proposals, abstracts):

        proposal_tokens = tokenizer(
            proposal,
            add_special_tokens=False
        )["input_ids"]

        abstract_tokens = tokenizer(
            abstract,
            add_special_tokens=False
        )["input_ids"]

        available_for_proposal = max_seq_length - len(abstract_tokens) - buffer

        if available_for_proposal <= 0:
            continue  

        proposal_tokens = proposal_tokens[:available_for_proposal]

        proposal_truncated = tokenizer.decode(
            proposal_tokens,
            skip_special_tokens=True
        )

        messages = [{"role": "system", "content": f"You are an expert scientific writer specialising in European research funding proposals.
        Given a full project proposal, your task is to write a concise, accurate abstract that captures the research objectives, methodology, expected outcomes, and relevance to the funding programme.
        The abstract should be written in formal academic English, be self-contained, and not exceed 250 words." },
            {
                "role": "user",
                "content": f"Summarize the following proposal:\n\n{proposal_truncated}"
            },
            {
                "role": "assistant",
                "content": abstract
            }
        ]

        formatted.append({"messages": messages})

    return formatted


dataset = format_dataset(proposals, abstracts, tokenizer)

print("Formatted dataset size:", len(dataset))


# ==============================
# CREATE HF DATASET
# ==============================

hf_dataset = Dataset.from_list(dataset)
hf_dataset = hf_dataset.shuffle(seed=42)

split = hf_dataset.train_test_split(test_size=0.1, seed=42)

train_dataset = split["train"]
test_dataset = split["test"]

print("Train size:", len(train_dataset))
print("Test size:", len(test_dataset))


# ==============================
# SAVE TEST DATASET
# ==============================

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
    #quantization_config=quantization_config,
    attn_implementation="eager"             
    )



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


