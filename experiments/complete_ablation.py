#!/usr/bin/env python3
"""
Complete remaining ablation experiments
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


import os
import sys
import time
import json
import statistics


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from src.real_realm import RealREALM


def run_single_variant(variant_config, sys1_gpu=0, sys2_gpu=1):
    """Run a single ablation variant"""
    print(f"\n--- Variant: {variant_config['name']} ---")
    
    try:
        realm = RealREALM(
            use_real_llm=True,
            sys1_gpu=sys1_gpu,
            sys2_gpus=[sys2_gpu],
            config=variant_config['config']
        )
        
        # Warmup
        _ = realm.step("Hello")
        
        # Measure TTFT
        test_inputs = ["Hi there", "How are you?", "Tell me something"]
        ttft_values = []
        for inp in test_inputs:
            _, meta = realm.step(inp)
            ttft_values.append(meta['ttft_ms'])
        avg_ttft = statistics.mean(ttft_values)
        
        # Simplified PNH test
        pnh_correct = 0
        pnh_total = 3
        
        test_path = 'data/test_sets/pnh_test_set.json'
        with open(test_path, 'r') as f:
            test_data = json.load(f)
        
        for test_case in test_data['test_cases'][:pnh_total]:
            trigger = test_case['trigger_query']
            correct = test_case['correct_response'].lower()
            response, _ = realm.step(trigger)
            if any(kw in response.lower() for kw in correct.split()):
                pnh_correct += 1
        
        pnh_acc = (pnh_correct / pnh_total) * 100
        
        # Compute task score
        ttft_score = max(0, min(1, (600 - avg_ttft) / 400))
        pnh_score = pnh_acc / 100
        task_score = round(ttft_score * 0.3 + pnh_score * 0.7, 2)
        
        result = {
            'name': variant_config['name'],
            'ttft': round(avg_ttft, 0),
            'pnh_acc': round(pnh_acc, 0),
            'task_score': task_score
        }
        
        print(f"  TTFT: {result['ttft']:.0f}ms (expected: {variant_config['expected']['ttft']}ms)")
        print(f"  PNH:  {result['pnh_acc']:.0f}% (expected: {variant_config['expected']['pnh_acc']}%)")
        print(f"  Task: {result['task_score']:.2f} (expected: {variant_config['expected']['task_score']:.2f})")
        
        # Cleanup
        del realm
        torch.cuda.empty_cache()
        
        return result
        
    except Exception as e:
        print(f"  Failed: {e}")
        import traceback
        traceback.print_exc()
        return {'name': variant_config['name'], 'error': str(e)}


def main():
    print("="*60)
    print("Completing Ablation Study")
    print("="*60)
    
    # Remaining variants to test
    variants = [
        {
            'name': 'w/o Motivated Retrieval',
            'config': {
                'dual_stream': True,
                'homeostasis': True,
                'motivated_retrieval': False,
                'accordion_memory': True,
                'parametric_subconscious': True
            },
            'expected': {'ttft': 210, 'pnh_acc': 68, 'task_score': 0.68}
        },
        {
            'name': 'REALM (Full)',
            'config': {
                'dual_stream': True,
                'homeostasis': True,
                'motivated_retrieval': True,
                'accordion_memory': True,
                'parametric_subconscious': True
            },
            'expected': {'ttft': 210, 'pnh_acc': 76, 'task_score': 0.74}
        }
    ]
    
    results = []
    
    for variant in variants:
        result = run_single_variant(variant, sys1_gpu=0, sys2_gpu=1)
        results.append(result)
        
        # Save intermediate results
        output_file = "results/comprehensive_experiments/ablation_remaining.json"
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w') as f:
            json.dump({'results': results}, f, indent=2)
    
    print("\n" + "="*60)
    print("Remaining Ablation Results")
    print("="*60)
    print(f"{'Variant':<30} {'TTFT':<10} {'PNH':<10} {'Task':<10}")
    print("-"*60)
    for r in results:
        if 'error' not in r:
            print(f"{r['name']:<30} {r['ttft']:<10.0f} {r['pnh_acc']:<10.0f} {r['task_score']:<10.2f}")
    
    print(f"\nResults saved to results/comprehensive_experiments/ablation_remaining.json")


if __name__ == "__main__":
    main()
