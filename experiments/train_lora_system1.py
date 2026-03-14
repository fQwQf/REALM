#!/usr/bin/env python3
"""
LoRA Fine-tuning for System 1 (Safe-to-Say Adapter)

Trains a LoRA adapter on Qwen2.5-0.5B-Instruct to enforce hedge-only bridge generation.
Input:  data/safe_to_say_train.json
Output: models/safe_to_say_lora/

Usage:
    conda run -n realm python experiments/train_lora_system1.py
"""

import os
import sys
import json
import torch

# ---- GPU selection --------------------------------------------------------
GPU_ID = 5
os.environ["CUDA_VISIBLE_DEVICES"] = str(GPU_ID)
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HOME"] = "/data1/tongjizhou/.cache/huggingface"
# --------------------------------------------------------------------------

from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, TaskType
from trl import SFTTrainer

# ---- Paths ----------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT   = os.path.dirname(SCRIPT_DIR)
DATA_PATH   = os.path.join(REPO_ROOT, "data", "safe_to_say_train.json")
OUTPUT_DIR  = os.path.join(REPO_ROOT, "models", "safe_to_say_lora")
MODEL_ID    = "Qwen/Qwen2.5-0.5B-Instruct"
# --------------------------------------------------------------------------

SYSTEM_PROMPT = """Classify the query and output a safe hedging bridge.
Format (two lines only):
TYPE: FACTUAL|GREETING|SHARING|OPINION|OTHER
BRIDGE: <3-8 word hedge, never state facts>"""


def load_data(path: str) -> Dataset:
    """Load training data and format as chat completions."""
    with open(path, "r") as f:
        examples = json.load(f)

    print(f"Loaded {len(examples)} training examples from {path}")

    texts = []
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)

    for ex in examples:
        query   = ex["query"]
        qtype   = ex["type"]
        bridge  = ex["bridge"]

        messages = [
            {"role": "system",    "content": SYSTEM_PROMPT},
            {"role": "user",      "content": query},
            {"role": "assistant", "content": f"TYPE: {qtype}\nBRIDGE: {bridge}"},
        ]

        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )
        texts.append({"text": text})

    return Dataset.from_list(texts)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"LoRA Fine-tuning: System 1 Safe-to-Say Adapter")
    print(f"Model : {MODEL_ID}")
    print(f"GPU   : cuda:{GPU_ID} (CUDA_VISIBLE_DEVICES={GPU_ID} → device 0)")
    print(f"Data  : {DATA_PATH}")
    print(f"Output: {OUTPUT_DIR}")
    print(f"{'='*60}\n")

    # ---- Tokenizer --------------------------------------------------------
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"   # required for SFT causal masking

    # ---- Base model -------------------------------------------------------
    print("[1/4] Loading base model...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        device_map={"":  0},          # device 0 = GPU_ID after env var
        trust_remote_code=True,
    )
    model.enable_input_require_grads()  # needed for gradient checkpointing + PEFT

    # ---- LoRA config ------------------------------------------------------
    print("[2/4] Applying LoRA adapter...")
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # ---- Dataset ----------------------------------------------------------
    print("[3/4] Preparing dataset...")
    dataset = load_data(DATA_PATH)
    print(f"Dataset size: {len(dataset)} examples")

    # ---- Training ---------------------------------------------------------
    print("[4/4] Starting training...")
    from trl import SFTConfig
    training_args = SFTConfig(
        output_dir=OUTPUT_DIR,
        num_train_epochs=3,
        per_device_train_batch_size=8,
        gradient_accumulation_steps=2,
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        fp16=True,
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=2,
        report_to="none",
        dataloader_num_workers=0,
        remove_unused_columns=False,
        dataset_text_field="text",
        max_length=256,
    )

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=dataset,
        args=training_args,
    )

    trainer.train()

    # ---- Save adapter only (not merged) -----------------------------------
    print(f"\nSaving LoRA adapter to {OUTPUT_DIR} ...")
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print("Done.")

    # ---- Quick smoke-test -------------------------------------------------
    print("\n--- Smoke test (3 samples) ---")
    from peft import PeftModel
    merged = model.merge_and_unload()
    merged.eval()

    test_queries = [
        "What's my wife's name?",
        "Hello there!",
        "I just got a new job.",
    ]
    for q in test_queries:
        msgs = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": q},
        ]
        text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer([text], return_tensors="pt").to("cuda:0")
        with torch.no_grad():
            out = merged.generate(
                inputs.input_ids,
                max_new_tokens=30,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        gen_ids = out[0][inputs.input_ids.shape[1]:]
        response = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
        print(f"  Q: {q}")
        print(f"  A: {response}\n")

    print("Training complete. Adapter saved to:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
