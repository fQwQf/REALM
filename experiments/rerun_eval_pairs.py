#!/usr/bin/env python3
"""
Re-run the 20 human eval pairs with the fixed TEMPO system.
Produces results/human_eval/rerun_tempo_outputs.json with full responses.

Usage:
    conda activate realm
    python experiments/rerun_eval_pairs.py
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

import os, sys, json, time


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.real_realm import RealREALM

# Load eval pairs and answer key
with open('results/human_eval/eval_pairs.json') as f:
    pairs = json.load(f)

with open('results/human_eval/answer_key.json') as f:
    key = {item['pair_id']: item for item in json.load(f)}

BACKGROUND = pairs[0]['background']  # Same for all pairs

# ── Persona facts to seed memory before each query ──────────────────────────
PERSONA_FACTS = [
    "Hi, I'm Alex. I'm a software engineer in San Francisco.",
    "I graduated from Stanford in 2020.",
    "My favorite food is sushi.",
    "I love hiking on weekends.",
    "I have a cat named Whiskers.",
    "My wife's name is Sarah.",
    "We have three children.",
    "I have a grandson named Tommy.",
    "I'm allergic to peanuts.",
    "I switched from coffee to tea.",
    "I went hiking at Yosemite last weekend.",
    "My favorite color is teal.",
    "I also have two dogs named Max and Bella.",
    "My signature dish is duck confit.",
    "I used to work at Boeing for 30 years.",
    "I play volleyball.",
]

def run_tempo(sys1_gpu=5, sys2_gpu=6):
    """Run TEMPO (dual-stream) on all 20 eval queries"""
    print("="*70)
    print("Initializing TEMPO (fixed) on GPUs", sys1_gpu, sys2_gpu)
    print("="*70)

    realm = RealREALM(
        use_real_llm=True,
        sys1_gpu=sys1_gpu,
        sys2_gpus=[sys2_gpu],
        embedding_device=f"cuda:{sys1_gpu}",
        config={
            'dual_stream': True,
            'homeostasis': True,
            'motivated_retrieval': True,
            'use_query_type': True,
            'entropy_threshold': 0.2,
        }
    )

    # Seed memory with persona facts
    print("\nSeeding persona memory...")
    for fact in PERSONA_FACTS:
        _, _ = realm.step(fact)
    print(f"Memory seeded with {len(PERSONA_FACTS)} facts.")

    outputs = []
    for p in pairs:
        query = p['query']
        print(f"\n[Pair {p['pair_id']:2d} | {p['type']:20s}] Q: {query}")
        t0 = time.time()
        response, meta = realm.step(query)
        elapsed = (time.time() - t0) * 1000
        print(f"  TEMPO: {response[:120]}")
        print(f"  TTFT: {meta.get('ttft_ms',0):.0f}ms | S2: {meta.get('system2_latency_ms',0):.0f}ms")
        outputs.append({
            'pair_id': p['pair_id'],
            'type': p['type'],
            'query': query,
            'tempo_response': response,
            'ttft_ms': meta.get('ttft_ms', 0),
        })

    return outputs


def run_vanilla(sys2_gpu=6):
    """Run Vanilla RAG (no dual-stream) on all 20 eval queries"""
    print("\n" + "="*70)
    print("Initializing Vanilla RAG on GPU", sys2_gpu)
    print("="*70)

    realm = RealREALM(
        use_real_llm=True,
        sys1_gpu=5,
        sys2_gpus=[sys2_gpu],
        embedding_device="cuda:5",
        config={
            'dual_stream': False,
            'homeostasis': True,
            'motivated_retrieval': True,
        }
    )

    # Seed memory with persona facts
    print("\nSeeding persona memory...")
    for fact in PERSONA_FACTS:
        _, _ = realm.step(fact)
    print(f"Memory seeded with {len(PERSONA_FACTS)} facts.")

    outputs = []
    for p in pairs:
        query = p['query']
        print(f"\n[Pair {p['pair_id']:2d} | {p['type']:20s}] Q: {query}")
        t0 = time.time()
        response, meta = realm.step(query)
        elapsed = (time.time() - t0) * 1000
        print(f"  Vanilla: {response[:120]}")
        outputs.append({
            'pair_id': p['pair_id'],
            'type': p['type'],
            'query': query,
            'vanilla_response': response,
        })

    return outputs


if __name__ == '__main__':
    os.makedirs('results/human_eval', exist_ok=True)

    print("\n" + "#"*70)
    print("# PHASE 1: TEMPO (fixed)")
    print("#"*70)
    tempo_outputs = run_tempo(sys1_gpu=5, sys2_gpu=6)

    print("\n" + "#"*70)
    print("# PHASE 2: Vanilla RAG")
    print("#"*70)
    vanilla_outputs = run_vanilla(sys2_gpu=7)

    # Merge results
    merged = []
    vanilla_map = {o['pair_id']: o['vanilla_response'] for o in vanilla_outputs}
    for t in tempo_outputs:
        pid = t['pair_id']
        k = key[pid]
        merged.append({
            'pair_id': pid,
            'type': t['type'],
            'query': t['query'],
            'tempo_response': t['tempo_response'],
            'vanilla_response': vanilla_map.get(pid, ''),
            'ttft_ms': t.get('ttft_ms', 0),
            'left_is': k['left_is'],
            'right_is': k['right_is'],
            # Reconstruct the blinded pair as it would appear to raters
            'response_left': t['tempo_response'] if k['left_is'] == 'tempo' else vanilla_map.get(pid, ''),
            'response_right': vanilla_map.get(pid, '') if k['right_is'] == 'vanilla' else t['tempo_response'],
        })

    out_path = 'results/human_eval/rerun_tempo_outputs.json'
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)
    print(f"\n✓ Results saved to {out_path}")

    # Print summary table
    print("\n" + "="*70)
    print("FIXED TEMPO vs VANILLA — Side-by-Side Sample")
    print("="*70)
    for item in merged:
        print(f"\nPair {item['pair_id']:2d} [{item['type']}]")
        print(f"  Q:       {item['query']}")
        print(f"  TEMPO:   {item['tempo_response'][:120]}")
        print(f"  Vanilla: {item['vanilla_response'][:120]}")
