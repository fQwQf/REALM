#!/usr/bin/env python3
"""
Full Evaluation Suite for TEMPO paper
Runs: TTFT benchmark, PNH evaluation, ablation study, NLI stitcher validation
All with real LLMs using available GPU resources.

GPU assignment:
  GPU 0: System 1 (Reflex) - Qwen2.5-0.5B-Instruct
  GPU 5,6,7: System 2 (Reflection) - Qwen2.5-7B-Instruct
"""

import os
import sys
import time
import json
import torch
import numpy as np
from pathlib import Path

# Set HF mirror for China
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['HF_HOME'] = '/data1/tongjizhou/.cache/huggingface'

sys.path.insert(0, str(Path(__file__).parent.parent))

RESULTS_DIR = Path('/data1/tongjizhou/REALM/results/full_evaluation')
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

SYS1_GPU = 0
SYS2_GPUS = [5, 6, 7]
SYS1_MODEL = 'Qwen/Qwen2.5-0.5B-Instruct'
SYS2_MODEL = 'Qwen/Qwen2.5-7B-Instruct'
LORA_PATH = '/data1/tongjizhou/REALM/models/safe_to_say_lora'
PNH_TEST_SET = '/data1/tongjizhou/REALM/data/test_sets/pnh_test_set.json'
PNH_EXTENDED = '/data1/tongjizhou/REALM/data/test_sets/pnh_extended_test_set.json'

print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU count: {torch.cuda.device_count()}")
for i in range(torch.cuda.device_count()):
    free, total = torch.cuda.mem_get_info(i)
    print(f"  GPU {i}: {torch.cuda.get_device_name(i)} | Free: {free//1024//1024}MB / {total//1024//1024}MB")


def load_system1(use_lora=True):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    print(f"\nLoading System 1 ({SYS1_MODEL}) on GPU {SYS1_GPU}...")
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(SYS1_MODEL, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        SYS1_MODEL,
        torch_dtype=torch.float16,
        device_map={"":SYS1_GPU},
        trust_remote_code=True
    )
    if use_lora and os.path.exists(os.path.join(LORA_PATH, 'adapter_config.json')):
        print(f"  Loading LoRA from {LORA_PATH}...")
        model = PeftModel.from_pretrained(model, LORA_PATH)
        model = model.merge_and_unload()
        print("  LoRA merged.")
    model.eval()
    print(f"  System 1 loaded in {time.time()-t0:.1f}s")
    return model, tok


def load_system2():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    print(f"\nLoading System 2 ({SYS2_MODEL}) on GPUs {SYS2_GPUS}...")
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(SYS2_MODEL, trust_remote_code=True)
    # Assign layers across multiple GPUs
    max_memory = {i: '22GiB' for i in SYS2_GPUS}
    model = AutoModelForCausalLM.from_pretrained(
        SYS2_MODEL,
        torch_dtype=torch.float16,
        device_map='auto',
        max_memory=max_memory,
        trust_remote_code=True
    )
    model.eval()
    print(f"  System 2 loaded in {time.time()-t0:.1f}s")
    return model, tok


def generate(model, tokenizer, prompt, max_new_tokens=128, temperature=0.7, do_sample=True,
             time_to_first=False):
    messages = [{'role': 'user', 'content': prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors='pt').to(f'cuda:{SYS1_GPU}')
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    t0 = time.perf_counter()
    with torch.no_grad():
        if time_to_first:
            # Generate just 1 token to measure TTFT
            out = model.generate(**inputs, max_new_tokens=1, do_sample=False)
        else:
            out = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature if do_sample else 1.0,
                do_sample=do_sample,
                pad_token_id=tokenizer.eos_token_id
            )
    elapsed_ms = (time.perf_counter() - t0) * 1000
    # Decode only the new tokens
    new_tokens = out[0][inputs['input_ids'].shape[1]:]
    response = tokenizer.decode(new_tokens, skip_special_tokens=True)
    return response, elapsed_ms


# =========================================================================
# EXPERIMENT 1: TTFT Benchmark
# =========================================================================
def run_ttft_benchmark(s1_model, s1_tok, n_warmup=5, n_runs=30):
    print("\n" + "="*60)
    print("EXPERIMENT 1: TTFT Benchmark")
    print("="*60)

    prompts = [
        "Hello, how are you today?",
        "What's your name?",
        "Tell me a joke.",
        "What is the capital of France?",
        "How does photosynthesis work?",
    ]

    # Warmup
    print(f"  Warmup ({n_warmup} runs)...")
    for i in range(n_warmup):
        p = prompts[i % len(prompts)]
        _, _ = generate(s1_model, s1_tok, p, max_new_tokens=1, time_to_first=True)

    # Benchmark
    print(f"  Benchmarking ({n_runs} runs)...")
    ttft_values = []
    for i in range(n_runs):
        p = prompts[i % len(prompts)]
        _, ms = generate(s1_model, s1_tok, p, max_new_tokens=1, time_to_first=True)
        ttft_values.append(ms)
        print(f"    Run {i+1:02d}: {ms:.1f}ms")

    result = {
        'hardware': {
            'sys1_gpu': SYS1_GPU,
            'sys1_model': SYS1_MODEL,
            'gpu_name': torch.cuda.get_device_name(SYS1_GPU),
            'gpu_vram_gb': torch.cuda.get_device_properties(SYS1_GPU).total_memory // (1024**3)
        },
        'n_runs': n_runs,
        'n_warmup': n_warmup,
        'p50_ms': float(np.percentile(ttft_values, 50)),
        'p95_ms': float(np.percentile(ttft_values, 95)),
        'mean_ms': float(np.mean(ttft_values)),
        'min_ms': float(np.min(ttft_values)),
        'max_ms': float(np.max(ttft_values)),
        'raw_ms': ttft_values
    }

    out = RESULTS_DIR / 'ttft.json'
    out.write_text(json.dumps(result, indent=2))
    print(f"\n  Results: P50={result['p50_ms']:.1f}ms  P95={result['p95_ms']:.1f}ms  Mean={result['mean_ms']:.1f}ms")
    print(f"  Saved to {out}")
    return result


# =========================================================================
# EXPERIMENT 2: PNH Evaluation
# =========================================================================
def score_pnh_response(response, test_case):
    """Score whether response correctly recalls needle AND applies state."""
    response_lower = response.lower()
    correct_response = test_case.get('correct_response', '').lower()
    needle_content = test_case.get('needle', {}).get('content', '').lower()

    # Check recall: key terms from correct response appear in output
    key_terms = [w for w in correct_response.split() if len(w) > 3]
    recall_hits = sum(1 for t in key_terms if t in response_lower)
    recall_ok = (recall_hits / max(len(key_terms), 1)) >= 0.5

    # Check state alignment (rough heuristic: not contradicting boundary)
    state_ok = True
    # For boundary cases, check we're NOT giving the info
    if test_case.get('type') == 'boundary':
        needle_words = [w for w in needle_content.split() if len(w) > 4]
        violations = [w for w in needle_words[:3] if w in response_lower]
        state_ok = len(violations) == 0

    passed = recall_ok and state_ok
    return passed, recall_ok, state_ok


def run_pnh_evaluation(s2_model, s2_tok, test_set_path=None, name='pnh_10'):
    print(f"\n" + "="*60)
    print(f"EXPERIMENT 2: PNH Evaluation ({name})")
    print("="*60)

    if not test_set_path or not os.path.exists(test_set_path):
        print("  Test set not found, skipping.")
        return {}

    with open(test_set_path) as f:
        data = json.load(f)
    test_cases = data.get('test_cases', data) if isinstance(data, dict) else data
    print(f"  Loaded {len(test_cases)} test cases.")

    passed_total = 0
    recall_total = 0
    state_total = 0
    details = []

    for i, tc in enumerate(test_cases):
        # Build conversation context
        context_turns = tc.get('distractor_turns', [])
        needle_info = tc.get('needle', {})
        needle_content = needle_info.get('content', '')
        trigger_query = tc.get('trigger_query', tc.get('query', ''))

        # Build context as text block
        context_lines = []
        for j, turn in enumerate(context_turns):
            context_lines.append(f"User: {turn.get('user', '')}")
            context_lines.append(f"Assistant: {turn.get('agent', turn.get('assistant', ''))}")
        context_text = '\n'.join(context_lines)

        # Inject needle explicitly (simulating storage in memory)
        memory_note = f"[Memory note: {needle_content}]"
        state = needle_info.get('state_condition', {})
        mood = state.get('mood', 'Calm')
        stress = state.get('stress', 30)

        prompt = (
            f"You are a personal AI assistant with memory. {memory_note}\n"
            f"Current state: Mood={mood}, Stress={stress}.\n\n"
            f"Recent conversation:\n{context_text}\n\n"
            f"User: {trigger_query}\n"
            f"Assistant:"
        )

        response, latency_ms = generate(s2_model, s2_tok, prompt, max_new_tokens=150, do_sample=False)
        passed, recall_ok, state_ok = score_pnh_response(response, tc)

        passed_total += int(passed)
        recall_total += int(recall_ok)
        state_total += int(state_ok)

        status = 'PASS' if passed else 'FAIL'
        print(f"  {status} [{i+1}/{len(test_cases)}] recall={recall_ok} state={state_ok} | {trigger_query[:50]}")
        print(f"         Response: {response[:80]}")

        details.append({
            'test_id': tc.get('id', f'pnh_{i:03d}'),
            'name': tc.get('name', ''),
            'passed': passed,
            'recall_ok': recall_ok,
            'state_ok': state_ok,
            'response': response[:200],
            'trigger': trigger_query
        })

    n = len(test_cases)
    result = {
        'name': name,
        'n_cases': n,
        'passed': passed_total,
        'recall_success': recall_total,
        'state_aligned': state_total,
        'accuracy_pct': 100.0 * passed_total / n,
        'recall_pct': 100.0 * recall_total / n,
        'state_alignment_pct': 100.0 * state_total / n,
        'details': details
    }

    out = RESULTS_DIR / f'{name}.json'
    out.write_text(json.dumps(result, indent=2))
    print(f"\n  Accuracy: {result['accuracy_pct']:.1f}%  Recall: {result['recall_pct']:.1f}%  StateAlign: {result['state_alignment_pct']:.1f}%")
    print(f"  Saved to {out}")
    return result


# =========================================================================
# EXPERIMENT 3: NLI Stitcher Validation
# =========================================================================
def run_nli_stitcher_validation():
    print("\n" + "="*60)
    print("EXPERIMENT 3: NLI Stitcher Validation")
    print("="*60)

    try:
        from transformers import pipeline
        import torch
        nli = pipeline(
            'zero-shot-classification',
            model='cross-encoder/nli-deberta-v3-base',
            device=f'cuda:{SYS1_GPU}'
        )
        print("  NLI model loaded.")
    except Exception as e:
        print(f"  NLI model load failed: {e}")
        return {}

    # Test cases: (bridge, response, human_label)
    # Human labels: 'entailment', 'contradiction', 'neutral'
    test_pairs = [
        # Entailment pairs
        ("Let me think about that for a moment.", "I need a moment to gather my thoughts.", "entailment"),
        ("I'll look into that for you.", "I'm checking on that information now.", "entailment"),
        ("That's an interesting question.", "What a thought-provoking query.", "entailment"),
        ("I remember you mentioned something about that.", "Yes, you told me about this topic earlier.", "entailment"),
        ("Let me check my records.", "I'm reviewing what I know about this.", "entailment"),
        ("I understand your concern.", "I hear that you're worried about this.", "entailment"),
        ("One moment please.", "Just a second while I look that up.", "entailment"),
        ("I'll be happy to help with that.", "I'm glad to assist you with this.", "entailment"),
        # Contradiction pairs
        ("Yes, I promised I would do that.", "I have no record of making that promise.", "contradiction"),
        ("Absolutely, I remember exactly what you said.", "I don't have any memory of that conversation.", "contradiction"),
        ("I will definitely handle that for you.", "I cannot take on that responsibility.", "contradiction"),
        ("I agree with you completely.", "Actually, I disagree with that point.", "contradiction"),
        ("I always support your decisions.", "In this case, I think you're making a mistake.", "contradiction"),
        ("I have already completed that task.", "I haven't started working on that yet.", "contradiction"),
        ("I can do that right away.", "That is completely outside my capabilities.", "contradiction"),
        ("I recall you said you were happy.", "You never mentioned anything about your mood.", "contradiction"),
        # Neutral pairs
        ("Let me think about that.", "The weather is nice today.", "neutral"),
        ("I'll check on that for you.", "There are many types of pasta.", "neutral"),
        ("I understand your concern.", "The capital of France is Paris.", "neutral"),
        ("I remember you mentioned that.", "Mathematics is a fascinating subject.", "neutral"),
    ]

    correct = 0
    total = len(test_pairs)
    details = []

    for bridge, response, human_label in test_pairs:
        premise = bridge
        hypothesis = response
        result = nli(hypothesis, candidate_labels=['entailment', 'contradiction', 'neutral'],
                     hypothesis_template='{}', multi_label=False)
        pred_label = result['labels'][0]
        pred_score = result['scores'][0]
        match = (pred_label == human_label)
        correct += int(match)
        status = 'PASS' if match else 'FAIL'
        print(f"  {status} pred={pred_label}({pred_score:.2f}) gold={human_label} | {bridge[:40]}")
        details.append({
            'bridge': bridge,
            'response': response,
            'human_label': human_label,
            'pred_label': pred_label,
            'pred_score': pred_score,
            'correct': match
        })

    accuracy = 100.0 * correct / total
    result_summary = {
        'name': 'nli_stitcher_validation',
        'n_cases': total,
        'correct': correct,
        'accuracy_pct': accuracy,
        'details': details
    }

    out = RESULTS_DIR / 'nli_stitcher_validation.json'
    out.write_text(json.dumps(result_summary, indent=2))
    print(f"\n  NLI Accuracy: {accuracy:.1f}% ({correct}/{total})")
    print(f"  Saved to {out}")
    return result_summary


# =========================================================================
# EXPERIMENT 4: Latency Profiling
# =========================================================================
def run_latency_profiling(s2_model, s2_tok):
    print("\n" + "="*60)
    print("EXPERIMENT 4: Latency Profiling")
    print("="*60)

    prompts = [
        ("short", "What is 2+2?"),
        ("medium", "Explain the concept of machine learning in simple terms."),
        ("long", "Write a detailed essay about the history of artificial intelligence, covering the major milestones from the 1950s to the present day."),
    ]
    n_runs = 5
    results = {}

    for label, prompt in prompts:
        latencies = []
        for _ in range(n_runs):
            _, lat = generate(s2_model, s2_tok, prompt, max_new_tokens=128, do_sample=False)
            latencies.append(lat)
        avg = sum(latencies) / len(latencies)
        mn = min(latencies)
        mx = max(latencies)
        print(f"  [{label}] avg={avg:.1f}ms  min={mn:.1f}ms  max={mx:.1f}ms")
        results[label] = {'avg_ms': avg, 'min_ms': mn, 'max_ms': mx, 'runs': latencies}

    out = RESULTS_DIR / 'latency_profiling.json'
    out.write_text(json.dumps(results, indent=2))
    print(f"  Saved to {out}")
    return results


# =========================================================================
# MAIN
# =========================================================================
def main():
    parser = argparse.ArgumentParser(description='Run SYS-2 experiments')
    parser.add_argument('--pnh-test-set', type=str, default=None,
                        help='Path to PNH test set JSON')
    parser.add_argument('--skip-nli', action='store_true')
    parser.add_argument('--skip-latency', action='store_true')
    parser.add_argument('--skip-pnh', action='store_true')
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading SYS-2 model...")
    s2_model, s2_tok = load_model(SYS2_MODEL, SYS1_GPU)
    print("SYS-2 model ready.")

    all_results = {}

    # Experiment 1: Recall Benchmark
    r1 = run_recall_benchmark(s2_model, s2_tok)
    all_results['recall_benchmark'] = r1

    # Experiment 2: PNH Evaluation
    if not args.skip_pnh:
        r2 = run_pnh_evaluation(s2_model, s2_tok, test_set_path=args.pnh_test_set)
        all_results['pnh_evaluation'] = r2

    # Experiment 3: NLI Stitcher Validation
    if not args.skip_nli:
        r3 = run_nli_stitcher_validation()
        all_results['nli_stitcher'] = r3

    # Experiment 4: Latency Profiling
    if not args.skip_latency:
        r4 = run_latency_profiling(s2_model, s2_tok)
        all_results['latency_profiling'] = r4

    # Save combined results
    combined_out = RESULTS_DIR / 'all_results.json'
    combined_out.write_text(json.dumps(all_results, indent=2))
    print(f"\nAll results saved to {combined_out}")


if __name__ == '__main__':
    main()
