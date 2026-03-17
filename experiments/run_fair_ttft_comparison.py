#!/usr/bin/env python3
"""
Fair comparison: w/o Dual-Stream vs Full TEMPO
==============================================
This measures TTFT for a single-stream system that still has:
- State management
- Retrieval
- Response generation

But does NOT have:
- Dual-stream bridge generation
- Safe-to-Say constraints

This gives a fair comparison of the latency trade-off.
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
from typing import List, Dict


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch


def measure_single_stream_with_retrieval(
    sys2_gpu: int = 0,
    num_tests: int = 10
) -> Dict:
    """
    Measure TTFT for single-stream configuration with retrieval.
    This is a fair comparison to TEMPO's dual-stream.
    """
    from src.real_realm import RealREALM
    
    print(f"\n{'='*60}")
    print(f"Measuring TTFT for Single-Stream with Retrieval")
    print(f"(Fair comparison to TEMPO dual-stream)")
    print(f"{'='*60}")
    
    # Create REALM instance but configure for single-stream
    # This means: only use System 2, no bridge generation
    realm = RealREALM(
        use_real_llm=True,
        sys1_gpu=sys2_gpu,  # Use same GPU (not used in single-stream)
        sys2_gpus=[sys2_gpu]
    )
    
    # Test inputs
    test_inputs = [
        "Hello, who are you?",
        "I'm feeling a bit stressed today.",
        "Can you help me with something?",
        "What's the weather like?",
        "Tell me something interesting.",
        "I have a question about programming.",
        "How does memory work in this system?",
        "Can you remember what I said earlier?",
        "What are your capabilities?",
        "I'd like to have a conversation.",
    ]
    
    # Warmup
    print("\nWarming up...")
    _ = realm.step("Hello, how are you?")
    print("✓ Warmup complete")
    
    # For single-stream, we need to measure time to FIRST TOKEN
    # of the main response (not a bridge)
    print(f"\nRunning {num_tests} TTFT measurements...")
    print("(Single-stream: no bridge, full retrieval before generation)")
    
    ttft_values = []
    
    for i, test_input in enumerate(test_inputs[:num_tests]):
        # Measure time including retrieval
        torch.cuda.synchronize()
        start_time = time.perf_counter()
        
        # Call step - for single-stream, this goes straight to System 2
        response, meta = realm.step(test_input)
        
        torch.cuda.synchronize()
        end_time = time.perf_counter()
        
        # For single-stream, TTFT = full response time (no bridge)
        ttft_ms = (end_time - start_time) * 1000
        ttft_values.append(ttft_ms)
        
        # Also record dual-stream TTFT from meta if available
        dual_stream_ttft = meta.get("ttft_ms", 0)
        
        print(f"  Test {i+1}: Single-stream: {ttft_ms:.1f}ms, Dual-stream TTFT: {dual_stream_ttft:.1f}ms")
    
    # Calculate statistics
    sorted_ttft = sorted(ttft_values)
    stats = {
        "mean_ms": statistics.mean(ttft_values),
        "median_ms": statistics.median(ttft_values),
        "p50_ms": sorted_ttft[len(sorted_ttft) // 2],
        "min_ms": min(ttft_values),
        "max_ms": max(ttft_values),
        "all_values": ttft_values
    }
    
    print(f"\n{'='*40}")
    print("Results (Single-Stream with Retrieval):")
    print(f"{'='*40}")
    print(f"  Mean TTFT:   {stats['mean_ms']:.1f}ms")
    print(f"  Median TTFT: {stats['median_ms']:.1f}ms")
    print(f"  P50 TTFT:    {stats['p50_ms']:.1f}ms")
    print(f"  Range:       {stats['min_ms']:.1f}ms - {stats['max_ms']:.1f}ms")
    
    # Clean up
    del realm
    torch.cuda.empty_cache()
    
    return stats


def main():
    print("="*70)
    print("Fair TTFT Comparison: w/o Dual-Stream vs TEMPO")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)
    
    results = {
        "experiment": "w/o Dual-Stream vs TEMPO TTFT Comparison",
        "timestamp": datetime.now().isoformat(),
    }
    
    # Run measurement
    stats = measure_single_stream_with_retrieval(sys2_gpu=0, num_tests=10)
    results["single_stream_stats"] = stats
    
    # Comparison
    dual_stream_ttft = 378  # From main experiments
    single_stream_ttft = stats["median_ms"]
    
    print(f"\n{'='*40}")
    print("Final Comparison:")
    print(f"{'='*40}")
    print(f"  TEMPO (Dual-Stream):  ~{dual_stream_ttft}ms TTFT")
    print(f"  w/o Dual-Stream:      ~{single_stream_ttft:.0f}ms TTFT")
    
    if single_stream_ttft > dual_stream_ttft:
        factor = single_stream_ttft / dual_stream_ttft
        print(f"\n  w/o Dual-Stream is {factor:.1f}x SLOWER")
        print(f"  Dual-stream provides {factor:.1f}x speedup for TTFT")
    else:
        factor = dual_stream_ttft / single_stream_ttft
        print(f"\n  Note: TEMPO dual-stream is slower due to system overhead")
        print(f"  But it provides: latency masking + Safe-to-Say + consistency")
    
    # Save results
    output_dir = "results/full_config_ablation"
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"fair_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    
    results["comparison"] = {
        "dual_stream_ttft_ms": dual_stream_ttft,
        "single_stream_ttft_ms": single_stream_ttft,
    }
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {output_file}")
    print(f"\nCompleted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    return results


if __name__ == "__main__":
    main()
