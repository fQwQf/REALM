#!/usr/bin/env python3
"""
LoCoMo Benchmark Evaluation for TEMPO

LoCoMo = Long Conversation Modeling benchmark
Paper: Maharana et al., 2024 (EMNLP)
HuggingFace: maharana/locomo

Evaluates:
  - Single-hop QA (factual recall over long conversations)
  - Multi-hop QA (reasoning across sessions)
  - Summarization quality

Usage:
  conda run -n realm python experiments/locomo_eval.py 2>&1 | tee results/locomo_eval.log
"""
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

import os, sys, json, time, re
import numpy as np
from pathlib import Path


OUT_DIR = Path(str(REPO_ROOT / 'results/locomo_eval'))
OUT_DIR.mkdir(parents=True, exist_ok=True)

SYS2_GPUS = [5, 6, 7]
SYS2_MODEL = 'Qwen/Qwen2.5-7B-Instruct'

_model_cache = {}

def load_s2():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    if 's2' in _model_cache:
        return _model_cache['s2']
    print(f'Loading {SYS2_MODEL} on GPUs {SYS2_GPUS}...')
    tok = AutoTokenizer.from_pretrained(SYS2_MODEL, trust_remote_code=True)
    max_mem = {i: '20GiB' for i in SYS2_GPUS}
    model = AutoModelForCausalLM.from_pretrained(
        SYS2_MODEL, torch_dtype=torch.float16,
        device_map='auto', max_memory=max_mem, trust_remote_code=True)
    model.eval()
    _model_cache['s2'] = (model, tok)
    return model, tok


def generate(model, tok, prompt, max_new_tokens=128):
    import torch
    msgs = [{'role': 'user', 'content': prompt}]
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inputs = tok(text, return_tensors='pt', truncation=True, max_length=4096)
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    t0 = time.perf_counter()
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens,
                             do_sample=False, pad_token_id=tok.eos_token_id)
    elapsed = (time.perf_counter() - t0) * 1000
    new_ids = out[0][inputs['input_ids'].shape[1]:]
    return tok.decode(new_ids, skip_special_tokens=True).strip(), elapsed


def f1_score(prediction, ground_truth):
    """Token-level F1 score."""
    pred_tokens = prediction.lower().split()
    gt_tokens = ground_truth.lower().split()
    if not pred_tokens or not gt_tokens:
        return 0.0
    common = set(pred_tokens) & set(gt_tokens)
    if not common:
        return 0.0
    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(gt_tokens)
    return 2 * precision * recall / (precision + recall)


def exact_match(prediction, ground_truth):
    return float(prediction.lower().strip() == ground_truth.lower().strip())


def load_locomo():
    from datasets import load_dataset
    print('Loading LoCoMo dataset...')
    ds = load_dataset('maharana/locomo', split='test')
    return ds


def build_qa_prompt(conversation_text, question):
    return (
        'The following is a long conversation between two people.\n\n'
        f'{conversation_text}\n\n'
        f'Based on the conversation above, answer the following question concisely.\n'
        f'Question: {question}\n'
        'Answer:'
    )


def build_summary_prompt(conversation_text):
    return (
        'The following is a long conversation between two people.\n\n'
        f'{conversation_text}\n\n'
        'Write a concise summary of the conversation above in 3-5 sentences.'
    )


def truncate_conversation(conv_text, max_chars=12000):
    if len(conv_text) <= max_chars:
        return conv_text
    return conv_text[:max_chars] + '\n[... conversation truncated ...]'


def evaluate_qa(ds, model, tok, qa_type='single_hop', max_samples=100):
    print(f'\n=== Evaluating {qa_type} QA (max {max_samples} samples) ===')
    results = []
    key_q = f'{qa_type}_questions' if qa_type != 'single_hop' else 'qa_pairs'
    skipped = 0
    count = 0
    for item in ds:
        if count >= max_samples:
            break
        conv_sessions = item.get('conversation', [])
        conv_text = '\n'.join(
            f"{turn.get('speaker','?')}: {turn.get('utterance','')}"
            for session in conv_sessions
            for turn in (session if isinstance(session, list) else [session])
        )
        conv_text = truncate_conversation(conv_text)
        qa_list = item.get(key_q, [])
        if not qa_list:
            skipped += 1
            continue
        for qa in qa_list:
            if count >= max_samples:
                break
            question = qa.get('question', '').strip()
            answer = qa.get('answer', '').strip()
            if not question or not answer:
                continue
            prompt = build_qa_prompt(conv_text, question)
            pred, latency_ms = generate(model, tok, prompt, max_new_tokens=64)
            em = exact_match(pred, answer)
            f1 = f1_score(pred, answer)
            results.append({
                'question': question,
                'ground_truth': answer,
                'prediction': pred,
                'exact_match': em,
                'f1': f1,
                'latency_ms': latency_ms,
            })
            count += 1
            if count % 10 == 0:
                avg_f1 = np.mean([r['f1'] for r in results])
                print(f'  [{count}/{max_samples}] avg F1={avg_f1:.4f}')
    if skipped:
        print(f'  Skipped {skipped} items (missing {key_q})')
    avg_em = np.mean([r['exact_match'] for r in results]) if results else 0.0
    avg_f1 = np.mean([r['f1'] for r in results]) if results else 0.0
    avg_lat = np.mean([r['latency_ms'] for r in results]) if results else 0.0
    print(f'  Results: EM={avg_em:.4f}  F1={avg_f1:.4f}  Latency={avg_lat:.1f}ms  N={len(results)}')
    return results, {'exact_match': avg_em, 'f1': avg_f1, 'latency_ms': avg_lat, 'n': len(results)}


def evaluate_summarization(ds, model, tok, max_samples=50):
    from rouge_score import rouge_scorer as rs_module
    scorer = rs_module.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    print(f'\n=== Evaluating Summarization (max {max_samples} samples) ===')
    results = []
    count = 0
    for item in ds:
        if count >= max_samples:
            break
        ref_summary = item.get('summary', '').strip()
        if not ref_summary:
            continue
        conv_sessions = item.get('conversation', [])
        conv_text = '\n'.join(
            f"{turn.get('speaker','?')}: {turn.get('utterance','')}"
            for session in conv_sessions
            for turn in (session if isinstance(session, list) else [session])
        )
        conv_text = truncate_conversation(conv_text)
        prompt = build_summary_prompt(conv_text)
        pred, latency_ms = generate(model, tok, prompt, max_new_tokens=200)
        scores = scorer.score(ref_summary, pred)
        results.append({
            'reference': ref_summary,
            'prediction': pred,
            'rouge1': scores['rouge1'].fmeasure,
            'rouge2': scores['rouge2'].fmeasure,
            'rougeL': scores['rougeL'].fmeasure,
            'latency_ms': latency_ms,
        })
        count += 1
        if count % 10 == 0:
            avg_r1 = np.mean([r['rouge1'] for r in results])
            print(f'  [{count}/{max_samples}] avg ROUGE-1={avg_r1:.4f}')
    avg_r1  = np.mean([r['rouge1']  for r in results]) if results else 0.0
    avg_r2  = np.mean([r['rouge2']  for r in results]) if results else 0.0
    avg_rL  = np.mean([r['rougeL']  for r in results]) if results else 0.0
    avg_lat = np.mean([r['latency_ms'] for r in results]) if results else 0.0
    print(f'  Results: R1={avg_r1:.4f}  R2={avg_r2:.4f}  RL={avg_rL:.4f}  Latency={avg_lat:.1f}ms  N={len(results)}')
    return results, {'rouge1': avg_r1, 'rouge2': avg_r2, 'rougeL': avg_rL, 'latency_ms': avg_lat, 'n': len(results)}


def main():
    print('=== LoCoMo Benchmark Evaluation ===')
    print(f'Output dir: {OUT_DIR}')

    model, tok = load_s2()
    ds = load_locomo()
    print(f'Dataset size: {len(ds)} examples')

    all_results = {}
    summary_metrics = {}

    # Single-hop QA
    sh_results, sh_metrics = evaluate_qa(ds, model, tok, qa_type='single_hop', max_samples=200)
    all_results['single_hop_qa'] = sh_results
    summary_metrics['single_hop_qa'] = sh_metrics

    # Multi-hop QA
    mh_results, mh_metrics = evaluate_qa(ds, model, tok, qa_type='multi_hop', max_samples=200)
    all_results['multi_hop_qa'] = mh_results
    summary_metrics['multi_hop_qa'] = mh_metrics

    # Summarization
    sum_results, sum_metrics = evaluate_summarization(ds, model, tok, max_samples=100)
    all_results['summarization'] = sum_results
    summary_metrics['summarization'] = sum_metrics

    # Save results
    out_file = OUT_DIR / 'locomo_results.json'
    with open(out_file, 'w') as f:
        json.dump({'metrics': summary_metrics, 'details': all_results}, f, indent=2)
    print(f'\nResults saved to {out_file}')

    print('\n=== FINAL SUMMARY ===')
    for task, metrics in summary_metrics.items():
        print(f'{task}: {json.dumps(metrics, indent=2)}')


if __name__ == '__main__':
    main()
