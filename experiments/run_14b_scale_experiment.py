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

import os, sys, json, time, numpy as np
from pathlib import Path

os.environ['CUDA_VISIBLE_DEVICES'] = '1,2,3'


S1_MODEL = 'Qwen/Qwen2.5-0.5B-Instruct'
S2_MODEL_14B = str(Path(MODEL_DIR) / 'Qwen2.5-14B-Instruct')
LORA_PATH = str(REPO_ROOT / 'models/safe_to_say_lora')
PNH_10 = str(REPO_ROOT / 'data/test_sets/pnh_test_set.json')
OUT_DIR = Path(str(REPO_ROOT / 'results/full_evaluation'))

def save(obj, name):
    p = OUT_DIR / f'{name}.json'
    p.write_text(json.dumps(obj, indent=2, default=str))
    print(f'  -> saved {p}')

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

print('Loading System 1 (0.5B + LoRA) on cuda:0...')
s1_tok = AutoTokenizer.from_pretrained(S1_MODEL, trust_remote_code=True)
s1_model = AutoModelForCausalLM.from_pretrained(
    S1_MODEL, torch_dtype=torch.float16,
    device_map={'': 0}, trust_remote_code=True)
if os.path.exists(os.path.join(LORA_PATH, 'adapter_config.json')):
    from peft import PeftModel
    s1_model = PeftModel.from_pretrained(s1_model, LORA_PATH)
    s1_model = s1_model.merge_and_unload()
s1_model.eval()
print('  S1 loaded.')

print(f'Loading System 2 (14B) on GPUs 0,1,2...')
s2_tok = AutoTokenizer.from_pretrained(S2_MODEL_14B, trust_remote_code=True)
s2_model = AutoModelForCausalLM.from_pretrained(
    S2_MODEL_14B, torch_dtype=torch.float16,
    device_map='auto',
    max_memory={0: '6GiB', 1: '22GiB', 2: '22GiB'},
    trust_remote_code=True)
s2_model.eval()
print('  S2 (14B) loaded.')

def generate_one(model, tok, prompt, max_new_tokens=150):
    inputs = tok(prompt, return_tensors='pt').to(model.device)
    t0 = time.perf_counter()
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens,
                             do_sample=False, temperature=1.0)
    ms = (time.perf_counter() - t0) * 1000
    text = tok.decode(out[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    return text.strip(), ms

def ttft_one(model, tok, prompt):
    inputs = tok(prompt, return_tensors='pt').to(model.device)
    t0 = time.perf_counter()
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=1, do_sample=False)
    return (time.perf_counter() - t0) * 1000

print('\n=== TTFT Benchmark (S1 0.5B+LoRA) ===')
prompts = [
    'You are a personal AI. User: Hey, how are you? Assistant:',
    'You are a personal AI. User: What did I tell you about my job? Assistant:',
    'You are a personal AI. User: Remember my favorite food? Assistant:',
]
for _ in range(5):
    ttft_one(s1_model, s1_tok, prompts[0])

N_RUNS = 50
s1_times = []
for i in range(N_RUNS):
    ms = ttft_one(s1_model, s1_tok, prompts[i % len(prompts)])
    s1_times.append(ms)
s1_p50 = float(np.percentile(s1_times, 50))
s1_p95 = float(np.percentile(s1_times, 95))
print(f'  S1 TTFT: P50={s1_p50:.1f}ms  P95={s1_p95:.1f}ms')

print('\n=== End-to-end latency (S2 14B, first token) ===')
for _ in range(3):
    ttft_one(s2_model, s2_tok, prompts[0])

s2_times = []
for i in range(20):
    ms = ttft_one(s2_model, s2_tok, prompts[i % len(prompts)])
    s2_times.append(ms)
s2_p50 = float(np.percentile(s2_times, 50))
s2_p95 = float(np.percentile(s2_times, 95))
print(f'  S2 (14B) TTFT: P50={s2_p50:.1f}ms  P95={s2_p95:.1f}ms')

print('\n=== PNH-10 with 14B System 2 ===')
with open(PNH_10) as f:
    pnh_data = json.load(f)
cases = pnh_data.get('test_cases', pnh_data) if isinstance(pnh_data, dict) else pnh_data

def score_pnh(response, tc):
    resp = response.lower()
    correct = tc.get('correct_response', '').lower()
    keywords = [w for w in correct.split() if len(w) > 3]
    if not keywords:
        keywords = correct.split()
    recall_hits = sum(1 for k in keywords if k in resp)
    recall_ok = (recall_hits / max(len(keywords), 1)) >= 0.4
    tc_type = tc.get('type', '')
    state_ok = True
    if tc_type == 'boundary':
        needle = tc.get('needle', {}).get('content', '').lower()
        nwords = [w for w in needle.split() if len(w) > 4][:4]
        violations = sum(1 for w in nwords if w in resp)
        state_ok = violations == 0
    return recall_ok and state_ok, recall_ok, state_ok

passed_n = recall_n = state_n = 0
details = []
for i, tc in enumerate(cases):
    needle = tc.get('needle', {})
    needle_content = needle.get('content', '')
    state = needle.get('state_condition', {})
    distractors = tc.get('distractor_turns', [])
    trigger = tc.get('trigger_query', tc.get('query', ''))
    correct = tc.get('correct_response', '')
    ctx_lines = []
    for turn in distractors:
        u = turn.get('user', '')
        a = turn.get('agent', turn.get('assistant', ''))
        ctx_lines.append(f'User: {u}')
        ctx_lines.append(f'Assistant: {a}')
    ctx = '\n'.join(ctx_lines[-20:])
    mood = state.get('mood', 'Calm')
    stress = state.get('stress', 30)
    defense = state.get('defense', 'None')
    prompt = (
        f'You are a personal AI assistant with access to the following memory note:\n'
        f'[MEMORY]: {needle_content}\n\n'
        f'Current psychological state: Mood={mood}, Stress={stress}/100, Defense={defense}.\n\n'
        f'Recent conversation context:\n{ctx}\n\n'
        f'User: {trigger}\n'
        f'Assistant (answer based on your memory, respecting your current state):'
    )
    resp, ms = generate_one(s2_model, s2_tok, prompt, max_new_tokens=150)
    passed, recall_ok, state_ok_v = score_pnh(resp, tc)
    passed_n += passed; recall_n += recall_ok; state_n += state_ok_v
    status = 'PASS' if passed else 'FAIL'
    print(f'  {status} [{i+1}/{len(cases)}] recall={recall_ok} state={state_ok_v}')
    print(f'    Query: {trigger[:70]}')
    print(f'    Got: {resp[:80]}')
    details.append({'id': tc.get('id', f'{i}'), 'passed': passed,
                    'recall_ok': recall_ok, 'state_ok': state_ok_v,
                    'response': resp[:200], 'trigger': trigger})

n = len(cases)
pnh_acc = round(100.0 * passed_n / n, 1)
print(f'  PNH-10 (14B): {pnh_acc}% ({passed_n}/{n})')

print('\n=== Conflict rate estimation (14B) ===')
conflict_prompts = [
    ('Let me check your notes...', 'You are a personal AI. The user asked about their job. Memory says they are a teacher. Respond:'),
    ('Hmm, let me think...', 'You are a personal AI. The user asked about their pet. Memory says they have a cat named Luna. Respond:'),
    ('One moment...', 'You are a personal AI. The user asked about their favorite food. Memory says they love sushi. Respond:'),
    ('Let me recall...', 'You are a personal AI. The user asked about their birthday. Memory says it is March 15. Respond:'),
    ('Checking my notes...', 'You are a personal AI. The user asked about their hobby. Memory says they enjoy painting. Respond:'),
]

from transformers import pipeline as hf_pipeline
nli_pipe = hf_pipeline('zero-shot-classification',
    model='cross-encoder/nli-deberta-v3-base', device=0)

conflicts = 0
total_checks = 0
for bridge, s2_prompt in conflict_prompts:
    for _ in range(4):
        resp, _ = generate_one(s2_model, s2_tok, s2_prompt, max_new_tokens=80)
        first_sent = resp.split('.')[0] + '.' if '.' in resp else resp[:100]
        combined = f'{bridge} {first_sent}'
        result = nli_pipe(combined, ['entailment', 'neutral', 'contradiction'])
        pred = result['labels'][0]
        if pred == 'contradiction':
            conflicts += 1
        total_checks += 1

conflict_rate = round(conflicts / total_checks * 100, 1)
print(f'  Conflict rate (14B): {conflict_rate}% ({conflicts}/{total_checks})')

results = {
    'experiment': 'scale_14b',
    's2_model': S2_MODEL_14B,
    's1_ttft_p50_ms': round(s1_p50, 1),
    's1_ttft_p95_ms': round(s1_p95, 1),
    's2_ttft_p50_ms': round(s2_p50, 1),
    's2_ttft_p95_ms': round(s2_p95, 1),
    'pnh_10_accuracy_pct': pnh_acc,
    'pnh_10_passed': int(passed_n),
    'pnh_10_total': n,
    'pnh_10_recall_pct': round(100.0 * recall_n / n, 1),
    'pnh_10_state_pct': round(100.0 * state_n / n, 1),
    'conflict_rate_pct': conflict_rate,
    'conflict_checks': total_checks,
    'pnh_details': details,
}
save(results, 'scale_14b')
print('\n=== DONE ===')
print(json.dumps({k: v for k, v in results.items() if k != 'pnh_details'}, indent=2))
