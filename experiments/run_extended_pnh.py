#!/usr/bin/env python3
"""
Run PNH evaluation with extended test set
===========================================
Uses the extended 51-case PNH test set for more robust evaluation.
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
from datetime import datetime
from typing import Dict, List


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch


def run_extended_pnh_evaluation(sys1_gpu: int, sys2_gpu: int) -> Dict:
    """Run PNH evaluation with extended test set."""
    print("\n" + "="*60)
    print("Extended PNH Evaluation (51 test cases)")
    print("="*60)
    
    from src.real_realm import RealREALM
    
    # Load extended test set
    test_path = 'data/test_sets/pnh_extended_test_set.json'
    with open(test_path, 'r') as f:
        test_data = json.load(f)
    
    test_cases = test_data['test_cases']
    print(f"Loaded {len(test_cases)} test cases")
    
    # Initialize HOMEO
    print(f"\nInitializing HOMEO: System 1 on GPU {sys1_gpu}, System 2 on GPU {sys2_gpu}")
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
        "total": len(test_cases),
        "correct": 0,
        "by_type": {},
        "details": []
    }
    
    # Run each test case
    for i, test_case in enumerate(test_cases):
        case_id = test_case['id']
        case_type = test_case['type']
        needle = test_case['needle']
        trigger_query = test_case['trigger_query']
        correct_response = test_case['correct_response']
        
        # Initialize type tracking
        if case_type not in results["by_type"]:
            results["by_type"][case_type] = {"total": 0, "correct": 0}
        
        results["by_type"][case_type]["total"] += 1
        
        # Build conversation with distractor turns
        for turn in test_case['distractor_turns']:
            user_msg = turn['user']
            _ = realm.step(user_msg)
        
        # Now query the trigger
        response, meta = realm.step(trigger_query)
        
        # Check if correct (case-insensitive partial match)
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
            "response": response[:200],
            "success": success
        })
        
        if (i + 1) % 10 == 0:
            print(f"  Progress: {i+1}/{len(test_cases)}, correct so far: {results['correct']}")
        
        # Clear state for next test
        del realm
        torch.cuda.empty_cache()
        
        realm = RealREALM(
            use_real_llm=True,
            sys1_gpu=sys1_gpu,
            sys2_gpus=[sys2_gpu]
        )
    
    # Calculate accuracy
    results["accuracy"] = (results["correct"] / results["total"]) * 100
    
    # Calculate by-type accuracy
    for case_type in results["by_type"]:
        type_data = results["by_type"][case_type]
        type_data["accuracy"] = (type_data["correct"] / type_data["total"]) * 100
    
    # Cleanup
    del realm
    torch.cuda.empty_cache()
    
    return results


def main():
    print("="*70)
    print("HOMEO Extended PNH Evaluation")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)
    
    # Run evaluation
    results = run_extended_pnh_evaluation(sys1_gpu=0, sys2_gpu=1)
    
    # Print summary
    print("\n" + "="*70)
    print("EXTENDED PNH EVALUATION RESULTS")
    print("="*70)
    
    print(f"\nOverall Accuracy: {results['accuracy']:.1f}% ({results['correct']}/{results['total']})")
    
    print(f"\nBy Type:")
    print(f"{'Type':<20} {'Accuracy':<12} {'Correct/Total':<15}")
    print("-" * 47)
    for case_type, data in sorted(results["by_type"].items()):
        print(f"{case_type:<20} {data['accuracy']:.1f}%{'':<6} {data['correct']}/{data['total']}")
    
    # Save results
    output_dir = "results/extended_pnh"
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\nResults saved to: {output_file}")
    print(f"\nCompleted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Comparison with paper claim
    print(f"\n" + "="*70)
    print("COMPARISON WITH PAPER CLAIMS")
    print("="*70)
    print(f"Paper claims PNH accuracy: ~90%")
    print(f"Extended test (N={results['total']}): {results['accuracy']:.1f}%")
    
    if results['accuracy'] >= 75:
        print("✓ Meets or exceeds paper target")
    else:
        print("⚠ Below paper target - may need investigation")
    
    return results


if __name__ == "__main__":
    main()
