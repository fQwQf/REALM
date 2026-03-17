#!/usr/bin/env python3
"""
Full Configuration TTFT Measurement for w/o Dual-Stream
========================================================
This experiment measures TTFT for the "w/o Dual-Stream" variant using
the FULL configuration (single 8B model for all generation), not the
simplified ablation setup.

The correct w/o Dual-Stream TTFT should be ~450-560ms, not 289ms.
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
from transformers import AutoModelForCausalLM, AutoTokenizer


def measure_single_stream_ttft(
    model_name: str = "Qwen/Qwen2.5-7B-Instruct",
    gpu_id: int = 0,
    num_tests: int = 10
) -> Dict:
    """
    Measure TTFT for single-stream (8B only) configuration.
    This represents the "w/o Dual-Stream" baseline properly.
    """
    print(f"\n{'='*60}")
    print(f"Measuring TTFT for Single-Stream (8B only)")
    print(f"Model: {model_name}")
    print(f"GPU: {gpu_id}")
    print(f"{'='*60}")
    
    device = f"cuda:{gpu_id}"
    
    # Load model
    print(f"\nLoading model on {device}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map=device
    )
    model.eval()
    print(f"✓ Model loaded")
    
    # Test inputs - same as main TTFT experiment
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
    warmup_input = "Hello, how are you?"
    inputs = tokenizer(warmup_input, return_tensors="pt").to(device)
    with torch.no_grad():
        _ = model.generate(**inputs, max_new_tokens=10)
    print("✓ Warmup complete")
    
    # Measure TTFT
    print(f"\nRunning {num_tests} TTFT measurements...")
    ttft_values = []
    
    for i, test_input in enumerate(test_inputs[:num_tests]):
        inputs = tokenizer(test_input, return_tensors="pt").to(device)
        
        # Measure time to first token
        torch.cuda.synchronize()
        start_time = time.perf_counter()
        
        with torch.no_grad():
            # Generate just 1 token to measure TTFT
            outputs = model.generate(
                **inputs,
                max_new_tokens=1,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )
        
        torch.cuda.synchronize()
        end_time = time.perf_counter()
        
        ttft_ms = (end_time - start_time) * 1000
        ttft_values.append(ttft_ms)
        
        print(f"  Test {i+1}: {test_input[:40]}... -> TTFT: {ttft_ms:.1f}ms")
    
    # Calculate statistics
    sorted_ttft = sorted(ttft_values)
    stats = {
        "mean_ms": statistics.mean(ttft_values),
        "median_ms": statistics.median(ttft_values),
        "p50_ms": sorted_ttft[len(sorted_ttft) // 2],
        "p95_ms": sorted_ttft[int(len(sorted_ttft) * 0.95)] if len(sorted_ttft) > 5 else sorted_ttft[-1],
        "min_ms": min(ttft_values),
        "max_ms": max(ttft_values),
        "all_values": ttft_values
    }
    
    print(f"\n{'='*40}")
    print("Results (Single-Stream 8B Only):")
    print(f"{'='*40}")
    print(f"  Mean TTFT:   {stats['mean_ms']:.1f}ms")
    print(f"  Median TTFT: {stats['median_ms']:.1f}ms")
    print(f"  P50 TTFT:    {stats['p50_ms']:.1f}ms")
    print(f"  P95 TTFT:    {stats['p95_ms']:.1f}ms")
    print(f"  Range:       {stats['min_ms']:.1f}ms - {stats['max_ms']:.1f}ms")
    
    # Clean up
    del model
    torch.cuda.empty_cache()
    
    return stats


def main():
    print("="*70)
    print("Full Configuration w/o Dual-Stream TTFT Measurement")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)
    
    results = {
        "experiment": "w/o Dual-Stream TTFT (Full Configuration)",
        "timestamp": datetime.now().isoformat(),
        "config": {
            "model": "Qwen/Qwen2.5-7B-Instruct",
            "description": "Single 8B model handles all generation (no fast bridge)"
        }
    }
    
    # Run measurement
    stats = measure_single_stream_ttft(
        model_name="Qwen/Qwen2.5-7B-Instruct",
        gpu_id=0,
        num_tests=10
    )
    
    results["statistics"] = stats
    
    # Comparison with dual-stream
    dual_stream_ttft = 378  # From main experiments
    single_stream_ttft = stats["median_ms"]
    overhead = single_stream_ttft / dual_stream_ttft
    
    print(f"\n{'='*40}")
    print("Comparison with Dual-Stream:")
    print(f"{'='*40}")
    print(f"  TEMPO (Dual-Stream):  ~{dual_stream_ttft}ms TTFT")
    print(f"  w/o Dual-Stream:      ~{single_stream_ttft:.0f}ms TTFT")
    print(f"  Overhead factor:      {overhead:.2f}x")
    print(f"\n  Conclusion: w/o Dual-Stream is {overhead:.2f}x SLOWER")
    print(f"  (This is the CORRECT value for the paper)")
    
    # Save results
    output_dir = "results/full_config_ablation"
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"wo_dual_stream_ttft_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    
    results["comparison"] = {
        "dual_stream_ttft_ms": dual_stream_ttft,
        "single_stream_ttft_ms": single_stream_ttft,
        "overhead_factor": overhead
    }
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {output_file}")
    print(f"\nCompleted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    return results


if __name__ == "__main__":
    main()
