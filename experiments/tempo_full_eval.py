#!/usr/bin/env python3
"""
TEMPO Full Evaluation Suite
Runs real LLM experiments for paper results:
  1. TTFT benchmark (System 1, with LoRA)
  2. PNH evaluation (10-case + 51-case extended)
  3. NLI stitcher precision/recall validation
  4. Memory Coherence Under Stress

GPU assignment:
  GPU 0: System 1 (Qwen2.5-0.5B, with LoRA) + NLI
  GPU 5,6,7: System 2 (Qwen2.5-7B)

Usage:
  conda run -n realm python experiments/tempo_full_eval.py 2>&1 | tee results/full_eval.log
"""
import os, sys, json, time
import numpy as np
from pathlib import Path

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['HF_HOME'] = '/data1/tongjizhou/.cache/huggingface'
sys.path.insert(0, str(Path(__file__).parent.parent))

OUT_DIR = Path('/data1/tongjizhou/REALM/results/full_evaluation')
OUT_DIR.mkdir(parents=True, exist_ok=True)

SYS1_GPU = 0
SYS2_GPUS = [5, 6, 7]
SYS1_MODEL = 'Qwen/Qwen2.5-0.5B-Instruct'
SYS2_MODEL = 'Qwen/Qwen2.5-7B-Instruct'
LORA_PATH = '/data1/tongjizhou/REALM/models/safe_to_say_lora'
PNH_10 = '/data1/tongjizhou/REALM/data/test_sets/pnh_test_set.json'
PNH_51 = '/data1/tongjizhou/REALM/data/test_sets/pnh_extended_test_set.json'

def save(obj, name):
    p = OUT_DIR / f'{name}.json'
    p.write_text(json.dumps(obj, indent=2, default=str))
    print(f'  -> saved {p}')

# ======================================================================
# Model loading helpers
# ======================================================================
_s1_cache = {}
_s2_cache = {}

def load_s1(use_lora=True):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    key = ('lora' if use_lora else 'base')
    if key in _s1_cache:
        return _s1_cache[key]
    print(f'Loading S1 ({SYS1_MODEL}, lora={use_lora}) on cuda:{SYS1_GPU}...')
    tok = AutoTokenizer.from_pretrained(SYS1_MODEL, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        SYS1_MODEL, torch_dtype=torch.float16,
        device_map={'': SYS1_GPU}, trust_remote_code=True)
    if use_lora and os.path.exists(os.path.join(LORA_PATH, 'adapter_config.json')):
        from peft import PeftModel
        print(f'  Merging LoRA from {LORA_PATH}...')
        model = PeftModel.from_pretrained(model, LORA_PATH)
        model = model.merge_and_unload()
    model.eval()
    _s1_cache[key] = (model, tok)
    print(f'  S1 loaded.')
    return model, tok

def load_s2():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    if 's2' in _s2_cache:
        return _s2_cache['s2']
    print(f'Loading S2 ({SYS2_MODEL}) on GPUs {SYS2_GPUS}...')
    tok = AutoTokenizer.from_pretrained(SYS2_MODEL, trust_remote_code=True)
    max_mem = {i: '20GiB' for i in SYS2_GPUS}
    model = AutoModelForCausalLM.from_pretrained(
        SYS2_MODEL, torch_dtype=torch.float16,
        device_map='auto', max_memory=max_mem, trust_remote_code=True)
    model.eval()
    _s2_cache['s2'] = (model, tok)
    print(f'  S2 loaded.')
    return model, tok

def generate_one(model, tok, prompt, max_new_tokens=128, do_sample=False, temperature=1.0):
    """Generate response, return (text, latency_ms)."""
    import torch
    msgs = [{'role': 'user', 'content': prompt}]
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inputs = tok(text, return_tensors='pt')
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    t0 = time.perf_counter()
    with torch.no_grad():
        out = model.generate(
            **inputs, max_new_tokens=max_new_tokens,
            do_sample=do_sample, temperature=temperature,
            pad_token_id=tok.eos_token_id)
    elapsed = (time.perf_counter() - t0) * 1000
    new_ids = out[0][inputs['input_ids'].shape[1]:]
    response = tok.decode(new_ids, skip_special_tokens=True)
    return response.strip(), elapsed

def ttft_one(model, tok, prompt):
    """Measure TTFT: time to generate the first token only."""
    import torch
    msgs = [{'role': 'user', 'content': prompt}]
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inputs = tok(text, return_tensors='pt')
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    t0 = time.perf_counter()
    with torch.no_grad():
        _ = model.generate(**inputs, max_new_tokens=1, do_sample=False,
                           pad_token_id=tok.eos_token_id)
    return (time.perf_counter() - t0) * 1000

# ======================================================================
# Experiment 1: TTFT Benchmark
# ======================================================================
def exp_ttft():
    import torch
    print('\n' + '='*60)
    print('EXP 1: TTFT Benchmark')
    print('='*60)
    model, tok = load_s1(use_lora=True)

    prompts = [
        'Hello, how are you today?',
        'Can you tell me something interesting?',
        'What is the capital of France?',
        'How does memory work in AI systems?',
        'Tell me about your day.',
    ]

    # Warmup
    N_WARMUP = 5
    print(f'  Warmup ({N_WARMUP} runs)...')
    for i in range(N_WARMUP):
        ttft_one(model, tok, prompts[i % len(prompts)])

    # Benchmark
    N_RUNS = 50
    print(f'  Benchmarking ({N_RUNS} runs)...')
    times = []
    for i in range(N_RUNS):
        p = prompts[i % len(prompts)]
        ms = ttft_one(model, tok, p)
        times.append(ms)
        if (i+1) % 10 == 0:
            print(f'    [{i+1}/{N_RUNS}] last={ms:.1f}ms  running_p50={np.percentile(times, 50):.1f}ms')

    gpu_info = {
        'gpu_id': SYS1_GPU,
        'gpu_name': torch.cuda.get_device_name(SYS1_GPU),
        'gpu_vram_gb': torch.cuda.get_device_properties(SYS1_GPU).total_memory // (1024**3)
    }
    res = {
        'experiment': 'ttft',
        'hardware': gpu_info,
        'model': SYS1_MODEL,
        'lora': True,
        'n_warmup': N_WARMUP,
        'n_runs': N_RUNS,
        'p50_ms': float(np.percentile(times, 50)),
        'p95_ms': float(np.percentile(times, 95)),
        'p99_ms': float(np.percentile(times, 99)),
        'mean_ms': float(np.mean(times)),
        'min_ms': float(np.min(times)),
        'max_ms': float(np.max(times)),
        'raw_ms': [float(x) for x in times]
    }
    print(f"  RESULT: P50={res['p50_ms']:.1f}ms  P95={res['p95_ms']:.1f}ms  Mean={res['mean_ms']:.1f}ms")
    save(res, 'ttft')
    return res

# ======================================================================
# Experiment 2: PNH Evaluation
# ======================================================================
def score_pnh(response, tc):
    """Score PNH: did we recall the needle AND apply correct state?"""
    resp = response.lower()
    correct = tc.get('correct_response', '').lower()
    # Check recall: main keywords from expected answer
    keywords = [w for w in correct.split() if len(w) > 3]
    if not keywords:
        keywords = correct.split()
    recall_hits = sum(1 for k in keywords if k in resp)
    recall_ok = (recall_hits / max(len(keywords), 1)) >= 0.4
    # Boundary check: for boundary cases, we should NOT reveal the info
    tc_type = tc.get('type', '')
    state_ok = True
    if tc_type == 'boundary':
        needle = tc.get('needle', {}).get('content', '').lower()
        nwords = [w for w in needle.split() if len(w) > 4][:4]
        violations = sum(1 for w in nwords if w in resp)
        state_ok = violations == 0
    passed = recall_ok and state_ok
    return passed, recall_ok, state_ok

def exp_pnh(path, label):
    print(f'\n' + '='*60)
    print(f'EXP 2: PNH Evaluation [{label}]')
    print('='*60)
    if not os.path.exists(path):
        print(f'  Test set not found: {path}  SKIPPING.')
        return {}
    with open(path) as f:
        data = json.load(f)
    cases = data.get('test_cases', data) if isinstance(data, dict) else data
    print(f'  {len(cases)} test cases.')

    model, tok = load_s2()
    passed_n = recall_n = state_n = 0
    details = []

    for i, tc in enumerate(cases):
        needle = tc.get('needle', {})
        needle_content = needle.get('content', '')
        state = needle.get('state_condition', {})
        distractors = tc.get('distractor_turns', [])
        trigger = tc.get('trigger_query', tc.get('query', ''))
        correct = tc.get('correct_response', '')

        # Build conversation context (include needle)
        ctx_lines = []
        for j, turn in enumerate(distractors):
            u = turn.get('user', '')
            a = turn.get('agent', turn.get('assistant', ''))
            ctx_lines.append(f'User: {u}')
            ctx_lines.append(f'Assistant: {a}')
        ctx = '\n'.join(ctx_lines[-20:])  # last 20 lines to keep context short

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

        resp, _ = generate_one(model, tok, prompt, max_new_tokens=150)
        passed, recall_ok, state_ok = score_pnh(resp, tc)
        passed_n += passed; recall_n += recall_ok; state_n += state_ok
        status = 'PASS' if passed else 'FAIL'
        print(f'  {status} [{i+1}/{len(cases)}] recall={recall_ok} state={state_ok}')
        print(f'    Query: {trigger[:70]}')
        print(f'    Expected: {correct[:60]}')
        print(f'    Got: {resp[:80]}')
        details.append({'id': tc.get('id', f'{i}'), 'passed': passed,
                        'recall_ok': recall_ok, 'state_ok': state_ok,
                        'response': resp[:200], 'trigger': trigger})

    n = len(cases)
    res = {
        'experiment': f'pnh_{label}',
        'n_cases': n,
        'passed': int(passed_n),
        'recall_ok': int(recall_n),
        'state_ok': int(state_n),
        'accuracy_pct': round(100.0 * passed_n / n, 1),
        'recall_pct': round(100.0 * recall_n / n, 1),
        'state_pct': round(100.0 * state_n / n, 1),
        'details': details
    }
    print(f"  RESULT: Acc={res['accuracy_pct']}%  Recall={res['recall_pct']}%  State={res['state_pct']}%")
    save(res, f'pnh_{label}')
    return res

# ======================================================================
# Experiment 3: NLI Stitcher Validation
# ======================================================================
def exp_nli_validation():
    print('\n' + '='*60)
    print('EXP 3: NLI Stitcher Validation')
    print('='*60)
    import torch
    from transformers import pipeline

    NLI_MODEL = 'cross-encoder/nli-deberta-v3-base'
    try:
        clf = pipeline('zero-shot-classification', model=NLI_MODEL,
                       device=f'cuda:{SYS1_GPU}')
        print(f'  NLI model loaded: {NLI_MODEL}')
    except Exception as e:
        print(f'  NLI load failed: {e}. SKIPPING.')
        return {}

    # (bridge, response, gold_label)
    test_pairs = [
        # Entailment
        ('Let me check your notes...', 'I found the information you requested.', 'entailment'),
        ('Based on your memory...', 'According to what you told me earlier, your meeting is at 3pm.', 'entailment'),
        ('I recall you mentioned...', 'You said your sister is visiting next week.', 'entailment'),
        # Neutral
        ('Let me think about that...', 'That is an interesting question.', 'neutral'),
        ('Hmm...', 'I am not sure how to respond to that.', 'neutral'),
        # Contradiction
        ('You told me you love coffee.', 'I have no information about your beverage preferences.', 'contradiction'),
        ('Your notes say you are vegetarian.', 'I cannot find any dietary information about you.', 'contradiction'),
    ]

    candidate_labels = ['entailment', 'neutral', 'contradiction']
    correct = 0
    results = []
    for bridge, response, gold in test_pairs:
        combined = f'{bridge} {response}'
        out = clf(combined, candidate_labels)
        pred = out['labels'][0]
        ok = (pred == gold)
        correct += ok
        status = 'PASS' if ok else 'FAIL'
        print(f'  {status}  gold={gold}  pred={pred}')
        print(f'    Bridge: {bridge}')
        print(f'    Response: {response}')
        results.append({'bridge': bridge, 'response': response,
                        'gold': gold, 'pred': pred, 'correct': ok})

    n = len(test_pairs)
    res = {
        'experiment': 'nli_validation',
        'model': NLI_MODEL,
        'n_pairs': n,
        'correct': correct,
        'accuracy_pct': round(100.0 * correct / n, 1),
        'details': results
    }
    print(f"  RESULT: Acc={res['accuracy_pct']}%  ({correct}/{n})")
    save(res, 'nli_validation')
    return res

# ======================================================================
# Experiment 4: Memory Coherence Under Stress
# ======================================================================
def exp_memory_coherence():
    print('\n' + '='*60)
    print('EXP 4: Memory Coherence Under Stress')
    print('='*60)

    model, tok = load_s2()

    stress_levels = [0, 25, 50, 75, 100]
    memory_fact = 'User prefers morning meetings before 10am and dislikes video calls.'
    query = 'Can we schedule a call this afternoon at 2pm via Zoom?'
    expected_keywords = ['morning', 'prefer', 'dislike', 'video', 'before 10']

    results = []
    for stress in stress_levels:
        mood = 'Calm' if stress < 40 else ('Anxious' if stress < 70 else 'Overwhelmed')
        defense = 'None' if stress < 30 else ('Deflection' if stress < 60 else 'Avoidance')
        prompt = (
            f'You are a personal AI assistant with the following memory:\n'
            f'[MEMORY]: {memory_fact}\n\n'
            f'Current state: Mood={mood}, Stress={stress}/100, Defense={defense}.\n\n'
            f'User: {query}\n'
            f'Assistant:'
        )
        resp, ms = generate_one(model, tok, prompt, max_new_tokens=120)
        resp_lower = resp.lower()
        hits = sum(1 for kw in expected_keywords if kw in resp_lower)
        coherence = round(hits / len(expected_keywords), 2)
        print(f'  stress={stress:3d}  mood={mood:12s}  coherence={coherence:.2f}  [{ms:.0f}ms]')
        print(f'    Response: {resp[:100]}')
        results.append({'stress': stress, 'mood': mood, 'defense': defense,
                        'coherence': coherence, 'response': resp[:200], 'ms': ms})

    res = {
        'experiment': 'memory_coherence',
        'memory_fact': memory_fact,
        'query': query,
        'stress_levels': stress_levels,
        'results': results,
        'mean_coherence': round(float(np.mean([r['coherence'] for r in results])), 3)
    }
    print(f"  RESULT: Mean coherence={res['mean_coherence']}")
    save(res, 'memory_coherence')
    return res

# ======================================================================
# Main
# ======================================================================
if __name__ == '__main__':
    print('Starting TEMPO evaluation suite...')
    print(f'Output dir: {OUT_DIR}')
    results = {}

    # Exp 1: TTFT
    results['ttft'] = exp_ttft()

    # Exp 2: PNH 10-case and 51-case
    results['pnh_10'] = exp_pnh(PNH_10, '10case')
    results['pnh_51'] = exp_pnh(PNH_51, '51case')

    # Exp 3: NLI Validation
    results['nli_validation'] = exp_nli_validation()

    # Exp 4: Memory Coherence
    results['memory_coherence'] = exp_memory_coherence()

    # Summary
    print('\n' + '='*60)
    print('EVALUATION SUITE COMPLETE')
    print('='*60)
    for k, v in results.items():
        if not v:
            print(f'  {k}: SKIPPED')
        elif k == 'ttft':
            print(f'  ttft: P50={v.get("p50_ms",0):.1f}ms  P95={v.get("p95_ms",0):.1f}ms')
        elif k.startswith('pnh'):
            print(f'  {k}: Acc={v.get("accuracy_pct","?")}%')
        elif k == 'nli_validation':
            print(f'  nli_validation: Acc={v.get("accuracy_pct","?")}%')
        elif k == 'memory_coherence':
            print(f'  memory_coherence: MeanCoherence={v.get("mean_coherence","?")}')
    save(results, 'MASTER_SUMMARY')
    print(f'\nAll results saved to: {OUT_DIR}')
