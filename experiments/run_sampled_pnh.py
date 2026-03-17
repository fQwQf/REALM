#!/usr/bin/env python3
"""
Sampled Extended PNH Evaluation
================================
Tests a representative sample (20 cases) from the extended test set
for efficiency while maintaining statistical validity.
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
import random
from datetime import datetime
from typing import Dict, List


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch


def run_sampled_pnh_evaluation(sys1_gpu: int, sys2_gpu: int, sample_size: int = 20) -> Dict:
    """Run PNH evaluation with a sampled subset for efficiency."""
    print("\n" + "="*60)
    print(f"Sampled PNH Evaluation ({sample_size} test cases)")
    print("="*60)
    
    from src.real_realm import RealREALM
    
    # Load extended test set
    test_path = 'data/test_sets/pnh_extended_test_set.json'
    with open(test_path, 'r') as f:
        test_data = json.load(f)
    
    all_cases = test_data['test_cases']
    
    # Sample to get representation from each type
    by_type = {}
    for case in all_cases:
        case_type = case['type']
        if case_type not in by_type:
            by_type[case_type] = []
        by_type[case_type].append(case)
    
    # Sample proportionally from each type
    sampled_cases = []
    cases_per_type = max(1, sample_size // len(by_type))
    
    for case_type, cases in by_type.items():
        n_sample = min(cases_per_type, len(cases))
        sampled_cases.extend(random.sample(cases, n_sample))
    
    # Shuffle the sampled cases
    random.shuffle(sampled_cases)
    sampled_cases = sampled_cases[:sample_size]
    
    print(f"Sampled {len(sampled_cases)} test cases from {len(all_cases)} total")
    
    # Initialize TEMPO once and run all tests
    print(f"\nInitializing TEMPO: System 1 on GPU {sys1_gpu}, System 2 on GPU {sys2_gpu}")
    realm = RealREALM(
        use_real_llm=True,
        sys1_gpu=sys1_gpu,
        sys2_gpus=[sys2_gpu]
    )
    
    # Warmup
    print("Warming up...")
    _ = realm.step("Hello, how are you?")
    
    # Results tracking
    results = {
        "total": len(sampled_cases),
        "correct": 0,
        "by_type": {},
        "details": []
    }
    
    # Run each test case
    for i, test_case in enumerate(sampled_cases):
        case_id = test_case['id']
        case_type = test_case['type']
        trigger_query = test_case['trigger_query']
        correct_response = test_case['correct_response']
        
        # Initialize type tracking
        if case_type not in results["by_type"]:
            results["by_type"][case_type] = {"total": 0, "correct": 0}
        
        results["by_type"][case_type]["total"] += 1
        
        # Build conversation context with some distractor turns
        distractors = test_case['distractor_turns'][:3]  # Use fewer distractors for speed
        for turn in distractors:
            _ = realm.step(turn['user'])
        
        # Query the trigger
        response, meta = realm.step(trigger_query)
        
        # Check if correct
        if isinstance(correct_response, str):
            success = correct_response.lower() in response.lower()
        else:
            success = str(correct_response).lower() in response.lower()
        
        if success:
            results["correct"] += 1
            results["by_type"][case_type]["correct"] += 1
        
        results["details"].append({
            "id": case_id,
            "type": case_type,
            "query": trigger_query,
            "expected": correct_response,
            "success": success
        })
        
        if (i + 1) % 5 == 0:
            print(f"  Progress: {i+1}/{len(sampled_cases)}, correct so far: {results['correct']}")
    
    # Calculate accuracy
    results["accuracy"] = (results["correct"] / results["total"]) * 100
    
    # Calculate by-type accuracy
    for case_type in results["by_type"]:
        type_data = results["by_type"][case_type]
        if type_data["total"] > 0:
            type_data["accuracy"] = (type_data["correct"] / type_data["total"]) * 100
    
    # Cleanup
    del realm
    torch.cuda.empty_cache()
    
    return results


def main():
    random.seed(42)  # For reproducibility
    
    print("="*70)
    print("TEMPO Sampled PNH Evaluation")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)
    
    # Run evaluation
    results = run_sampled_pnh_evaluation(sys1_gpu=0, sys2_gpu=1, sample_size=20)
    
    # Print summary
    print("\n" + "="*70)
    print("SAMPLED PNH EVALUATION RESULTS")
    print("="*70)
    
    print(f"\nOverall Accuracy: {results['accuracy']:.1f}% ({results['correct']}/{results['total']})")
    
    print(f"\nBy Type:")
    print(f"{'Type':<20} {'Accuracy':<12} {'Correct/Total':<15}")
    print("-" * 47)
    for case_type, data in sorted(results["by_type"].items()):
        print(f"{case_type:<20} {data.get('accuracy', 0):.1f}%{'':<6} {data['correct']}/{data['total']}")
    
    # Save results
    output_dir = "results/extended_pnh"
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"sampled_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\nResults saved to: {output_file}")
    print(f"\nCompleted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Comparison
    print(f"\n" + "="*70)
    print("COMPARISON")
    print("="*70)
    print(f"Original test (N=10): 90.0% accuracy")
    print(f"Extended test (N={results['total']}): {results['accuracy']:.1f}% accuracy")
    
    return results


if __name__ == "__main__":
    main()
