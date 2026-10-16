"""
02 - PEFT + LoRA for text classification (DistilBERT on IMDB sentiment)
=======================================================================
Use case: you have a labelled NLP dataset (reviews, support tickets, tweets)
and want a classifier, but don't want to train/store a full model copy per task.

What PEFT gives you here:
    - get_peft_model() injects LoRA into the attention layers for you.
    - Only ~1% of params train -> less GPU memory, faster epochs.
    - The saved adapter is ~3 MB instead of ~260 MB for the full model.
      You can keep one base model + many small adapters (one per client/task).

Runs on CPU (slow-ish) or GPU.
    python 02_peft_sentiment_classification.py                 # 2000 train rows
    python 02_peft_sentiment_classification.py --train_size 500  # quick test

Tested on CPU with --train_size 500: ~78% test accuracy after 32 steps, but the
probabilities stay close to 0.5 (under-trained). Use the 2000-row default, or
2-3 epochs, to get confident predictions.
"""
import argparse
import os

import numpy as np
from datasets import load_dataset
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          DataCollatorWithPadding, Trainer, TrainingArguments)

parser = argparse.ArgumentParser()
parser.add_argument("--train_size", type=int, default=2000)
parser.add_argument("--test_size", type=int, default=500)
parser.add_argument("--epochs", type=int, default=1)
args = parser.parse_args()

MODEL = "distilbert-base-uncased"
OUT = "outputs/distilbert-imdb-lora"

# ---- 1. data: small shuffled slices of IMDB ----
ds = load_dataset("stanfordnlp/imdb")
train = ds["train"].shuffle(seed=42).select(range(args.train_size))
test = ds["test"].shuffle(seed=42).select(range(args.test_size))

tok = AutoTokenizer.from_pretrained(MODEL)
def tokenize(batch):
    return tok(batch["text"], truncation=True, max_length=256)
train = train.map(tokenize, batched=True, remove_columns=["text"])
test = test.map(tokenize, batched=True, remove_columns=["text"])

# ---- 2. base model + LoRA config ----
model = AutoModelForSequenceClassification.from_pretrained(MODEL, num_labels=2)

lora_cfg = LoraConfig(
    task_type=TaskType.SEQ_CLS,
    r=8,                         # rank of the update matrices
    lora_alpha=16,               # scaling; alpha/r = 2 is a common default
    lora_dropout=0.1,
    target_modules=["q_lin", "v_lin"],   # DistilBERT's query/value projections
    # The classification head is brand new (random), so it must be trained fully:
    modules_to_save=["pre_classifier", "classifier"],
)
model = get_peft_model(model, lora_cfg)
model.print_trainable_parameters()

# ---- 3. train with the standard HF Trainer (PEFT models drop straight in) ----
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    return {"accuracy": (np.argmax(logits, axis=-1) == labels).mean()}

training_args = TrainingArguments(
    output_dir=OUT,
    learning_rate=5e-4,          # LoRA tolerates a higher LR than full fine-tuning
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    num_train_epochs=args.epochs,
    eval_strategy="epoch",
    save_strategy="no",
    logging_steps=20,
    report_to="none",
)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train,
    eval_dataset=test,
    data_collator=DataCollatorWithPadding(tok),
    compute_metrics=compute_metrics,
)
trainer.train()
print("Eval:", trainer.evaluate())

# ---- 4. save ONLY the adapter, check its size ----
adapter_dir = f"{OUT}/adapter"
model.save_pretrained(adapter_dir)
size_mb = sum(os.path.getsize(os.path.join(adapter_dir, f))
              for f in os.listdir(adapter_dir)) / 1e6
print(f"Adapter saved to {adapter_dir}  ({size_mb:.1f} MB)")

# ---- 5. reload: fresh base model + adapter, then predict ----
base = AutoModelForSequenceClassification.from_pretrained(MODEL, num_labels=2)
loaded = PeftModel.from_pretrained(base, adapter_dir).eval()

samples = [
    "An absolute masterpiece. The plot kept me guessing until the very end.",
    "Two hours of my life I will never get back. Wooden acting, no story.",
    "It was fine. Nothing special, but I didn't hate it.",
]
import torch
with torch.no_grad():
    enc = tok(samples, return_tensors="pt", padding=True, truncation=True)
    probs = loaded(**enc).logits.softmax(-1)
for s, p in zip(samples, probs):
    print(f"P(positive)={p[1]:.2f} | {s}")
