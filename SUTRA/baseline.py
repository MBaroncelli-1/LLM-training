import torch
import numpy as np
import pandas as pd

from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from rouge_score import rouge_scorer


# ==========================
# CONFIG
# ==========================

base_model = "/leonardo_scratch/large/userinternal/mbaronce/models/Mistral-7B-Instruct-v0.3"
lora_path = "./out"
dataset_path = "./test_dataset"
output_csv = "evaluation__baseline.csv"


# ==========================
# LOAD MODEL
# ==========================

print("[INFO] Loading base model...")

tokenizer = AutoTokenizer.from_pretrained(base_model)

model = AutoModelForCausalLM.from_pretrained(
    base_model,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)


model.eval()


# ==========================
# LOAD TEST DATA
# ==========================

print("[INFO] Loading test dataset...")
test_dataset = Dataset.load_from_disk(dataset_path)


# ==========================
# GENERATION
# ==========================

generated_summaries = []
references = []
proposals_eval = []

print("[INFO] Starting generation...")

with torch.no_grad():
    for example in test_dataset:

        proposal_text = example["messages"][0]["content"]
        reference_summary = example["messages"][1]["content"]


        messages = [
            {"role": "user", "content": proposal_text}
        ]

        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=4096
        ).to(model.device)

        input_length = inputs["input_ids"].shape[1]

        output_ids = model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=False
        )

        generated_tokens = output_ids[0][input_length:]

        generated_text = tokenizer.decode(
            generated_tokens,
            skip_special_tokens=True
        ).strip()

        generated_summaries.append(generated_text)
        references.append(reference_summary)
        proposals_eval.append(proposal_text)

print("[INFO] Generation completed.")


# ==========================
# ROUGE
# ==========================

print("[INFO] Computing ROUGE...")

scorer = rouge_scorer.RougeScorer(
    ['rouge1', 'rouge2', 'rougeL'],
    use_stemmer=True
)

rouge1_scores = []
rouge2_scores = []
rougeL_scores = []

for gen, ref in zip(generated_summaries, references):
    scores = scorer.score(ref, gen)
    rouge1_scores.append(scores["rouge1"].fmeasure)
    rouge2_scores.append(scores["rouge2"].fmeasure)
    rougeL_scores.append(scores["rougeL"].fmeasure)

mean_r1 = np.mean(rouge1_scores)
mean_r2 = np.mean(rouge2_scores)
mean_rL = np.mean(rougeL_scores)

print("ROUGE-1:", mean_r1)
print("ROUGE-2:", mean_r2)
print("ROUGE-L:", mean_rL)


# ==========================
# SAVE RESULTS
# ==========================

df = pd.DataFrame({
    "proposal": proposals_eval,
    "reference": references,
    "generated": generated_summaries,
    "rouge1": rouge1_scores,
    "rouge2": rouge2_scores,
    "rougeL": rougeL_scores
})

df.to_csv(output_csv, index=False)

print(f"[INFO] Results saved to {output_csv}")

