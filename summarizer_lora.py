import torch
import time
import argparse 
import matplotlib.pyplot as plt
import torch.distributed as dist

from trl import SFTTrainer, SFTConfig
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig



####------------------SUMMARIZER CON SFTTRAINER-----------###
#Uso modello mistralai/Mistral-7B-Instruct-v0.3 con mistral-inference.
#Obiettivo: fine-tune con sfttrainer con conversational language modeling ("messages")


###-------------------TOKENIZER---------------------------###
tokenizer = AutoTokenizer.from_pretrained("/leonardo_scratch/large/userinternal/mbaronce/models/Mistral-7B-Instruct-v0.3")
tokenizer.pad_token = tokenizer.eos_token


###-------------------DATA PREPROCCESSING-----------------###
from pathlib import Path
import re
from datasets import Dataset

#extract proposal files and review files
proposals=[]
reviews=[]

base_path = Path("/leonardo_work/AIFAC_F01_011")

proposal_files = list(base_path.rglob("*Proposal.txt"))
review_files=list(base_path.rglob("*ESR.txt"))

for file_path in proposal_files:
    with open(file_path, "r", encoding="utf-8") as f:
        proposals.append(f.read())

for file_path in review_files:
    with open(file_path,"r", encoding="utf-8") as f:
        reviews.append(f.read())




#from the review files, extract the abstract only
def extract_abstract(text):
    pattern = r"Abstract\s*([\s\S]*?)(?=Evaluation Summary Report)"
    match = re.search(pattern, text, re.IGNORECASE)

    if match:
        return match.group(1).strip()

    return None


extracted_abstracts=[]

for file in reviews: 
    extracted_abstracts.append(extract_abstract(file))

#truncate the proposal


formatted = []
def format_dataset(proposals, abstracts, tokenizer, max_seq_length=4096, buffer=50):
    for proposal, abstract in zip(proposals, abstracts):
        proposal_tokens = tokenizer(proposal, add_special_tokens=False)["input_ids"]
        abstract_tokens = tokenizer(abstract, add_special_tokens=False)["input_ids"]

        available_for_proposal = max_seq_length - len(abstract_tokens) - buffer
        proposal_tokens = proposal_tokens[:available_for_proposal]
        proposal = tokenizer.decode(proposal_tokens, skip_special_tokens=True)


        messages=[
            {"role": "user", "content": f"Summarize the following proposal:\n\n{proposal}"},
            {"role": "assistant", "content": abstract}
        ]

        formatted.append({"messages": messages})

    return formatted


dataset=format_dataset(proposals, extracted_abstracts, tokenizer)
hf_dataset=Dataset.from_list(dataset)

#print(dataset)
#print(len(dataset))



###---------------------ACCELERATOR------------------###
from accelerate import Accelerator
accelerator = Accelerator()

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
    attn_implementation="eager"                 # Use "flash_attention_2" when running on Ampere or newer GPU
    )


###--------------------LORA CONGIF-----------------------###

# Define LoRA config and model
lora_config=LoraConfig(
    lora_alpha=16,
    lora_dropout=0.05,
    r=16,
    bias="none",
    target_modules= ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",],
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_config)
if accelerator.is_main_process: model.print_trainable_parameters()
model.config.use_cache = False


# Define training parameters and run training
training_args = SFTConfig(
    output_dir=args.output_path,
    num_train_epochs=args.num_train_epochs,
    max_steps=args.max_steps,
    per_device_train_batch_size=args.per_device_train_batch_size,   # start with 1 or 4, then use nvidia-smi ad adapt to best memory usage. 32 OoM
    gradient_accumulation_steps=1,      # 1 is ok, we don't have memory issues
    gradient_checkpointing=True,    # for error 
    gradient_checkpointing_kwargs={'use_reentrant': False},     # for error
    optim="adamw_torch",
    learning_rate=args.learning_rate,   # usually the learning rate is lower for fine-tuning (between 1e-5 a 5e-5), but higher with LoRA (e-4)
    lr_scheduler_type="cosine",
    ddp_find_unused_parameters=False,
    warmup_ratio=0.03,
    logging_steps=10,
    save_strategy="no",
    eval_strategy="no",
    eval_steps=10,     # evaluate every 10% steps
    bf16=True,
    remove_unused_columns=False,
    prediction_loss_only=True,
    dataset_text_field= "messages"
)

trainer = SFTTrainer(
    model=model, 
    args=training_args,
    train_dataset=hf_dataset,  
    processing_class=tokenizer
    )


if accelerator.is_main_process: print(f"[INFO] Starting fine-tuning.")
s = time.time()
trainer.train()
accelerator.wait_for_everyone()
e = time.time()
if accelerator.is_main_process: print(f"[INFO] Fine-tuning done.")
if accelerator.is_main_process: print(f"[INFO] Elapsed fine-tuining time {e-s} sec.")








