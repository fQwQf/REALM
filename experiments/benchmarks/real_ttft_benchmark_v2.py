#!/usr/bin/env python3
"""
TTFT Benchmark with Real LLM (Fixed Version)
Measures Time To First Token using real model inference
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

# Set HF mirror

# Add project root (parent of experiments/)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import torch
import numpy as np

# Import REALM using direct path
from src.real_realm import RealREALM

class RealTTFTBenchmark:
    """TTFT benchmark using real LLM inference"""
    
    def __init__(self, sys1_gpu=2, sys2_gpus=[4, 5, 6, 7]):
        self.sys1_gpu = sys1_gpu
        self.sys2_gpus = sys2_gpus
        self.realm = None
        self.results = {
            'ttft_values': [],
            'system2_latencies': []
        }
        
    def initialize(self):
        """Initialize REALM with real LLM"""
        print("[Initializing Real REALM...]")
        try:
            self.realm = RealREALM(
                use_real_llm=True,
                sys1_gpu=self.sys1_gpu,
                sys2_gpus=self.sys2_gpus
            )
            print("✓ Initialization complete\n")
            return True
        except Exception as e:
            print(f"✗ Initialization failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def run_ttft_tests(self, test_inputs, num_runs=3):
        """Run TTFT measurement tests (reduced to 3 for stability)"""
        print("="*60)
        print("TTFT Measurement with Real LLM")
        print(f"Running {num_runs} tests...\n")
        print("="*60)
        
        for i, user_input in enumerate(test_inputs[:num_runs]):
            print(f"\nTest {i+1}/{num_runs}: '{user_input}'")
            
            try:
                # Measure full step
                start = time.perf_counter()
                response, metadata = self.realm.step(user_input)
                end = time.perf_counter()
                
                ttft = metadata['ttft_ms']
                sys2_latency = metadata['system2_latency_ms']
                e2e_latency = (end - start) * 1000
                
                self.results['ttft_values'].append(ttft)
                self.results['system2_latencies'].append(sys2_latency)
                
                print(f"  TTFT: {ttft:.2f}ms")
                print(f"  System 2: {sys2_latency:.2f}ms")
                print(f"  E2E: {e2e_latency:.2f}ms")
                print(f"  Bridge: '{metadata['bridge'][:40]}'")
                print(f"  Response: '{response[:80]}...'")
                
            except Exception as e:
                print(f"  ✗ Test failed: {e}")
                import traceback
                traceback.print_exc()
        
    def compute_statistics(self):
        """Compute statistics for all metrics"""
        stats = {}
        
        for key, values in self.results.items():
            if values:
                sorted_values = sorted(values)
                n = len(values)
                
                stats[key] = {
                    'mean': statistics.mean(values),
                    'median': statistics.median(values),
                    'min': min(values),
                    'max': max(values),
                    'stdev': statistics.stdev(values) if n > 1 else 0,
                    'p50': sorted_values[int(n * 0.50)],
                    'p95': sorted_values[int(n * 0.95)] if n > 1 else values[0]
                }
        
        return stats
    
    def evaluate(self, stats):
        """Evaluate against paper targets"""
        print("\n" + "="*60)
        print("Evaluation Against Paper Targets")
        print("="*60)
        
        # Paper targets
        paper_ttft_p50 = 210
        paper_ttft_target = 300
        
        if 'ttft_values' in stats:
            measured_p50 = stats['ttft_values']['p50']
            measured_mean = stats['ttft_values']['mean']
            
            print(f"\nTTFT (Time To First Token):")
            print(f"  Measured P50: {measured_p50:.2f}ms")
            print(f"  Measured Mean: {measured_mean:.2f}ms")
            print(f"  Paper P50: ~{paper_ttft_p50}ms")
            print(f"  Target Threshold: <{paper_ttft_target}ms")
            
            if measured_p50 < paper_ttft_target:
                print(f"  ✓ PASS: P50 below {paper_ttft_target}ms threshold")
            else:
                print(f"  ✗ FAIL: P50 exceeds threshold")
        
        if 'system2_latencies' in stats:
            s2_mean = stats['system2_latencies']['mean']
            print(f"\nSystem 2 Latency:")
            print(f"  Mean: {s2_mean:.2f}ms")
    
    def save_results(self, filepath='results/real_ttft_results_v2.json'):
        """Save results to file"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        stats = self.compute_statistics()
        
        output = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'results': self.results,
            'statistics': stats,
            'config': {
                'sys1_gpu': self.sys1_gpu,
                'sys2_gpus': self.sys2_gpus
            },
            'paper_targets': {
                'ttft_p50_ms': 210,
                'ttft_target_ms': 300
            }
        }
        
        with open(filepath, 'w') as f:
            json.dump(output, f, indent=2)
        
        print(f"\n✓ Results saved to {filepath}")
    
    def run(self):
        """Main benchmark run"""
        print("="*70)
        print("REALM Real LLM TTFT Benchmark (Fixed Version)")
        print("="*70)
        print(f"Start time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        print()
        
        # Initialize
        if not self.initialize():
            print("\n✗ Cannot proceed without successful initialization")
            return
        
        # Test inputs (simplified set for stability)
        test_inputs = [
            "Hello, who are you?",
            "I'm feeling a bit stressed today.",
            "Can you help me?",
            "Thanks for the help."
        ]
        
        # Run tests
        self.run_ttft_tests(test_inputs, num_runs=3)
        
        # Compute statistics
        print("\n" + "="*60)
        print("Statistics Summary")
        print("="*60)
        
        stats = self.compute_statistics()
        for metric, values in stats.items():
            print(f"\n{metric}:")
            print(f"  Mean:   {values['mean']:.2f}ms")
            print(f"  Median: {values['median']:.2f}ms")
            print(f"  Min:    {values['min']:.2f}ms")
            print(f"  Max:    {values['max']:.2f}ms")
            print(f"  P95:    {values['p95']:.2f}ms")
        
        # Evaluate
        self.evaluate(stats)
        
        # Save
        self.save_results()
        
        print("\n" + "="*70)
        print("Real LLM TTFT Benchmark Complete")
        print("="*70)

if __name__ == "__main__":
    benchmark = RealTTFTBenchmark(sys1_gpu=2, sys2_gpus=[4, 5, 6, 7])
    benchmark.run()
