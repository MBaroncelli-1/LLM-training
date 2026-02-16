import torch
import time
import argparse 
import matplotlib.pyplot as plt
import torch.distributed as dist

from transformers import  AutoTokenizer,  AutoModelForSeq2SeqLM, Seq2SeqTrainer, Seq2SeqTrainingArguments, DataCollatorForSeq2Seq

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



####------------------SUMMARIZER CON BART-----------###
#l'input deve essere input_idc, attention_mask e labels. Rouge si calcola tra labels e output del decoder

###-------------------TOKENIZER---------------------------###

tokenizer=AutoTokenizer.from_pretrained(args.model_path)
model=AutoModelForSeq2SeqLM.from_pretrained(args.model_path)
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, label_pad_token_id=-100)
tokenizer.pad_token = tokenizer.eos_token

#print(model)


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

proposals=proposals*30
extracted_abstracts=extracted_abstracts*30

data_pairs = [{"document": p, "summary": a}for p, a in zip(proposals, extracted_abstracts)]

dataset=Dataset.from_list(data_pairs)

def preprocess(batch):
    model_inputs=tokenizer(batch["document"], max_length=1024, truncation=True)
    labels=tokenizer(batch["summary"], max_length=512, truncation=True)

    #labels["input_ids"]=[[(token if token != tokenizer.pad_token_id else -100) for token in seq] for seq in labels["input_ids"]]
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

tokenized_dataset = dataset.map(
    preprocess,
    batched=True,
    remove_columns=dataset.column_names,
)




training_args = Seq2SeqTrainingArguments(
    output_dir=args.output_path,
    per_device_train_batch_size=args.per_device_train_batch_size,
    per_device_eval_batch_size=1,
    num_train_epochs=1,
    logging_steps=1,
    predict_with_generate=True,
    bf16=False, #ho troppi pochi dati 
    save_strategy="no",
)


trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    #eval_dataset=tokenized_dataset["test"],
    processing_class=tokenizer,
    data_collator=data_collator
)

if torch.distributed.is_initialized():
    print("WORLD SIZE:", torch.distributed.get_world_size())
print("CUDA devices:", torch.cuda.device_count())

trainer.train()



