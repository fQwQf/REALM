#!/usr/bin/env python3
import os
import sys
from pathlib import Path

# Auto-detect repository root
REPO_ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(REPO_ROOT))

# Environment variables with fallbacks
HF_HOME = os.environ.get('HF_HOME', os.path.expanduser('~/.cache/huggingface'))
os.environ['HF_HOME'] = HF_HOME
os.environ['HF_ENDPOINT'] = os.environ.get('HF_ENDPOINT', 'https://hf-mirror.com')

# Model directory (for 14B experiments)
MODEL_DIR = os.environ.get('MODEL_DIR', str(REPO_ROOT / 'models'))

import os, torch
os.environ['CUDA_VISIBLE_DEVICES'] = '5'

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

MODEL_ID = 'Qwen/Qwen2.5-0.5B-Instruct'
LORA_PATH = str(REPO_ROOT / 'models/safe_to_say_lora')

tokenizer = AutoTokenizer.from_pretrained(LORA_PATH, trust_remote_code=True)
base = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float16, device_map={'': 0}, trust_remote_code=True)
model = PeftModel.from_pretrained(base, LORA_PATH)
model.eval()
print("Model loaded with LoRA adapter.")

SYS = """Classify the query and output a safe hedging bridge.
Format (two lines only):
TYPE: FACTUAL|GREETING|SHARING|OPINION|OTHER
BRIDGE: <3-8 word hedge, never state facts>"""

queries = [
    "What is my wife's name?",
    "Hello there!",
    "I just got a new job.",
    "How many grandchildren do I have?",
    "What company do I work for?",
    "Where did I go to college?",
    "I love gardening.",
    "Do you think AI is dangerous?",
    "What's my dog called?",
    "I moved to Seattle last year.",
]

for q in queries:
    msgs = [
        {'role': 'system', 'content': SYS},
        {'role': 'user', 'content': q},
    ]
    text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer([text], return_tensors='pt').to('cuda:0')
    attn_mask = (inputs.input_ids != tokenizer.eos_token_id).long()
    with torch.no_grad():
        out = model.generate(
            inputs.input_ids,
            attention_mask=attn_mask,
            max_new_tokens=15,
            do_sample=False,
        )
    resp = tokenizer.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
    print(f'Q: {q}')
    print(f'A: {resp}')
    print()
