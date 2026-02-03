import time
import json
import torch
import argparse 
import matplotlib.pyplot as plt
import torch.distributed as dist
from datasets import load_dataset, DatasetDict
from accelerate import Accelerator
from trl import SFTTrainer, SFTConfig
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, BitsAndBytesConfig


# Define accelerator
accelerator = Accelerator()
device = accelerator.device
if accelerator.is_main_process: print(f"[INFO] accelerator: {accelerator}")

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


# tokenizer
tokenizer = AutoTokenizer.from_pretrained("/leonardo_scratch/large/userinternal/mbaronce/models/Mistral-7B-Instruct-v0.3")
tokenizer.pad_token = tokenizer.eos_token


#dataset loading: 
if accelerator.is_main_process: print(f"[INFO] Processing data...")
dataset=load_dataset("parquet", data_files={"train": "document/train-*.parquet", "validation": "document/validation-*.parquet", "test": "document/test*.parquet"})

raw_datasets = DatasetDict(dataset)
print(raw_datasets)
print(raw_datasets["train"].features)



def format_chattemplate(example):
    abstract=example["abstract"]
    article=example["article"]
    messages=  [{"role": "system", "content": "you are a symmarization tool"},
        {"role": "user", "content": f"Summarize this article: {article}"},
        {"role": "assistant", "content": abstract}]
    
    
    return {"messages": messages}




train_dataset=raw_datasets["train"].map(format_chattemplate, batched=False, remove_columns=raw_datasets["train"].column_names)
test_dataset=raw_datasets["test"].map(format_chattemplate, batched=False, remove_columns=raw_datasets["test"].column_names)
validation_dataset=raw_datasets["validation"].map(format_chattemplate, batched=False, remove_columns=raw_datasets["validation"].column_names)

#print(train_dataset["messages"][:2])

if accelerator.is_main_process:print(f"train set: {train_dataset}")
if accelerator.is_main_process: print(f"test set: {test_dataset}")
if accelerator.is_main_process: print(f"validation set: {validation_dataset}")


# quantization configuration
quantization_config=BitsAndBytesConfig(load_in_4bit=True, 
                                       bnb_4bit_compute_dtype=torch.bfloat16, 
                                       bnb_4bit_quant_type="nf4", 
                                       bnb_4bit_use_double_quant=True, 
                                       bnb_4bit_quant_storage=torch.bfloat16
                                       )

if accelerator.is_main_process: print(f"[INFO] Loading LLM model.")
model = AutoModelForCausalLM.from_pretrained(
    args.model_path, 
    dtype=torch.bfloat16, 
    quantization_config=quantization_config,
    attn_implementation="eager"                 # Use "flash_attention_2" when running on Ampere or newer GPU
    )




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
    eval_strategy="steps",
    eval_steps=10,     # evaluate every 10% steps
    bf16=True,
    remove_unused_columns=False,
    prediction_loss_only=True,
    dataset_text_field= "messages"
)

trainer = SFTTrainer(
    model=model, 
    args=training_args,
    train_dataset=train_dataset, 
    eval_dataset=validation_dataset,  
    processing_class=tokenizer
    )







