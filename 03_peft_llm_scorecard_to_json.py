"""
03 - LoRA fine-tuning a small LLM: cricket result text -> JSON
==============================================================
Use case: teach an LLM a narrow, repeatable skill - here, structured extraction.
Prompting a small model to "return JSON" is unreliable; a LoRA adapter trained on
a few hundred examples makes it consistent. Same recipe works for:
    resume -> skills JSON, invoice -> fields, support ticket -> {category, urgency}.

This script deliberately uses a hand-written training loop (no Trainer) so every
step is visible: loss masking, which params update, merging.

Model: SmolLM2-135M-Instruct (tiny, trains on CPU in a few minutes).
Swap MODEL for "Qwen/Qwen2.5-0.5B-Instruct" or larger when you have a GPU.

    python 03_peft_llm_scorecard_to_json.py
"""
import json
import random

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "HuggingFaceTB/SmolLM2-135M-Instruct"
OUT = "outputs/smollm-scorecard-lora"
device = "cuda" if torch.cuda.is_available() else "cpu"
random.seed(0)
torch.manual_seed(0)

# ---- 1. synthetic dataset (no download needed) ----
TEAMS = ["India", "Australia", "England", "Pakistan", "South Africa",
         "New Zealand", "Sri Lanka", "Bangladesh", "Afghanistan", "West Indies"]

def make_example():
    a, b = random.sample(TEAMS, 2)
    first = random.randint(180, 360)
    if random.random() < 0.5:   # team batting first wins by runs
        second = first - random.randint(1, 120)
        wk2 = 10
        overs2 = f"{random.randint(35, 49)}.{random.randint(0, 5)}"
        text = (f"{a} {first}/{random.randint(3, 9)} (50 ov) beat "
                f"{b} {second} ({overs2} ov) by {first - second} runs")
        label = {"winner": a, "loser": b, "margin": f"{first - second} runs"}
    else:                        # chasing team wins by wickets
        wk = random.randint(1, 9)
        text = (f"{b} {first + random.randint(1, 6)}/{wk} "
                f"({random.randint(38, 49)}.{random.randint(0, 5)} ov) beat "
                f"{a} {first} (50 ov) by {10 - wk} wickets")
        label = {"winner": b, "loser": a, "margin": f"{10 - wk} wickets"}
    return text, json.dumps(label)

data = [make_example() for _ in range(300)]
train, test = data[:280], data[280:]   # 280 train, 20 test

def prompt(text):
    return f"Extract the match result as JSON.\nResult: {text}\nJSON:"

# ---- 2. tokenizer + loss masking (train only on the answer tokens) ----
tok = AutoTokenizer.from_pretrained(MODEL)
tok.pad_token = tok.pad_token or tok.eos_token

def encode(text, answer):
    p_ids = tok(prompt(text))["input_ids"]
    a_ids = tok(" " + answer)["input_ids"] + [tok.eos_token_id]
    input_ids = p_ids + a_ids
    labels = [-100] * len(p_ids) + a_ids   # -100 = ignored by the loss
    return input_ids, labels

def collate(batch):
    enc = [encode(t, a) for t, a in batch]
    L = max(len(x) for x, _ in enc)
    ids = torch.full((len(enc), L), tok.pad_token_id)
    lab = torch.full((len(enc), L), -100)
    mask = torch.zeros((len(enc), L), dtype=torch.long)
    for i, (x, y) in enumerate(enc):
        ids[i, :len(x)] = torch.tensor(x)
        lab[i, :len(y)] = torch.tensor(y)
        mask[i, :len(x)] = 1
    return ids, mask, lab

# ---- 3. model + LoRA ----
model = AutoModelForCausalLM.from_pretrained(MODEL).to(device)

@torch.no_grad()
def generate(m, text):
    enc = tok(prompt(text), return_tensors="pt").to(device)
    out = m.generate(**enc, max_new_tokens=40, do_sample=False,
                     pad_token_id=tok.pad_token_id)
    return tok.decode(out[0, enc["input_ids"].shape[1]:],
                      skip_special_tokens=True).strip()

def accuracy(m, rows):
    ok = 0
    for text, answer in rows:
        try:
            ok += json.loads(generate(m, text)) == json.loads(answer)
        except json.JSONDecodeError:
            pass
    return ok / len(rows)

model.eval()
print("BEFORE fine-tuning:")
print("  ", test[0][0], "\n   ->", generate(model, test[0][0]))
print(f"   exact-match accuracy on test: {accuracy(model, test):.0%}\n")

cfg = LoraConfig(
    task_type="CAUSAL_LM",
    r=16, lora_alpha=32, lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],  # Llama-style names
)
model = get_peft_model(model, cfg)
model.print_trainable_parameters()

# ---- 4. plain PyTorch training loop ----
opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=2e-4)
EPOCHS, BS = 3, 8
model.train()
for epoch in range(EPOCHS):
    random.shuffle(train)
    total = 0.0
    for i in range(0, len(train), BS):
        ids, mask, lab = (t.to(device) for t in collate(train[i:i + BS]))
        loss = model(input_ids=ids, attention_mask=mask, labels=lab).loss
        loss.backward()
        opt.step()
        opt.zero_grad()
        total += loss.item()
    print(f"epoch {epoch + 1}: mean loss {total / (len(train) / BS):.4f}")

# ---- 5. evaluate, save adapter, merge for deployment ----
model.eval()
print("\nAFTER fine-tuning:")
print("  ", test[0][0], "\n   ->", generate(model, test[0][0]))
print(f"   exact-match accuracy on test: {accuracy(model, test):.0%}")

model.save_pretrained(f"{OUT}/adapter")            # tiny: just LoRA weights
merged = model.merge_and_unload()                  # plain model, no PEFT needed
merged.save_pretrained(f"{OUT}/merged")
tok.save_pretrained(f"{OUT}/merged")
print(f"\nSaved adapter -> {OUT}/adapter, merged model -> {OUT}/merged")

# ---- QLoRA variant (GPU only, needs `pip install bitsandbytes`) ----
# Load the base in 4-bit to fine-tune 7B-8B models on a single 16-24 GB GPU:
#
#   from transformers import BitsAndBytesConfig
#   from peft import prepare_model_for_kbit_training
#   bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
#                            bnb_4bit_compute_dtype=torch.bfloat16)
#   model = AutoModelForCausalLM.from_pretrained(BIG_MODEL, quantization_config=bnb,
#                                                device_map="auto")
#   model = prepare_model_for_kbit_training(model)
#   model = get_peft_model(model, cfg)   # everything else stays the same
