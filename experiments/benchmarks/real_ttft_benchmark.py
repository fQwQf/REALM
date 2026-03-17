#!/usr/bin/env python3
"""
TTFT Benchmark with Real LLM
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

# Add project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import torch
from src.real_realm import RealREALM

class RealTTFTBenchmark:
    """TTFT benchmark using real LLM inference"""
    
    def __init__(self, sys1_gpu=2, sys2_gpus=[4, 5, 6, 7]):
        self.sys1_gpu = sys1_gpu
        self.sys2_gpus = sys2_gpus
        self.realm = None
        self.results = {
            'ttft_values': [],
            'system2_latencies': [],
            'end_to_end_latencies': []
        }
        
    def initialize(self):
        """Initialize REALM with real LLM"""
        print("[Initializing Real REALM...]")
        self.realm = RealREALM(
            use_real_llm=True,
            sys1_gpu=self.sys1_gpu,
            sys2_gpus=self.sys2_gpus
        )
        print("✓ Initialization complete\n")
    
    def run_ttft_tests(self, test_inputs, num_runs=10):
        """Run TTFT measurement tests"""
        print("="*60)
        print("TTFT Measurement with Real LLM")
        print("="*60)
        
        for i, user_input in enumerate(test_inputs[:num_runs]):
            print(f"\nTest {i+1}/{num_runs}: '{user_input}'")
            
            # Measure full step
            start = time.perf_counter()
            response, metadata = self.realm.step(user_input)
            end = time.perf_counter()
            
            ttft = metadata['ttft_ms']
            sys2_latency = metadata['system2_latency_ms']
            e2e_latency = (end - start) * 1000
            
            self.results['ttft_values'].append(ttft)
            self.results['system2_latencies'].append(sys2_latency)
            self.results['end_to_end_latencies'].append(e2e_latency)
            
            print(f"  TTFT: {ttft:.2f}ms")
            print(f"  System 2: {sys2_latency:.2f}ms")
            print(f"  End-to-End: {e2e_latency:.2f}ms")
            print(f"  Bridge: '{metadata['bridge'][:50]}'")
            print(f"  Response: '{response[:80]}...'")
        
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
                    'p95': sorted_values[int(n * 0.95)] if n > 1 else values[0],
                    'p99': sorted_values[int(n * 0.99)] if n > 1 else values[0]
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
            
            if abs(measured_p50 - paper_ttft_p50) < 50:
                print(f"  ✓ MATCH: Close to paper value (~{paper_ttft_p50}ms)")
            elif measured_p50 < paper_ttft_p50:
                print(f"  ✓ BETTER: Faster than paper report")
            else:
                print(f"  ⚠ SLOWER: Higher than paper report")
        
        if 'end_to_end_latencies' in stats:
            e2e_mean = stats['end_to_end_latencies']['mean']
            print(f"\nEnd-to-End Latency:")
            print(f"  Mean: {e2e_mean:.2f}ms")
            print(f"  (Paper reports 1550ms for full system)")
    
    def save_results(self, filepath='results/real_ttft_results.json'):
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
            }
        }
        
        with open(filepath, 'w') as f:
            json.dump(output, f, indent=2)
        
        print(f"\n✓ Results saved to {filepath}")
    
    def run(self):
        """Main benchmark run"""
        print("="*70)
        print("REALM Real LLM TTFT Benchmark")
        print("="*70)
        print(f"Start time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        print()
        
        # Initialize
        try:
            self.initialize()
        except Exception as e:
            print(f"✗ Failed to initialize: {e}")
            import traceback
            traceback.print_exc()
            return
        
        # Test inputs
        test_inputs = [
            "Hello, who are you?",
            "I'm feeling a bit stressed today.",
            "Can you help me organize my schedule?",
            "Wait, did you promise to keep my data private?",
            "Thanks for the help.",
            "What's the weather like?",
            "Tell me a joke.",
            "How do I cook pasta?",
            "What's your favorite color?",
            "Can you explain quantum physics?"
        ]
        
        # Run tests
        self.run_ttft_tests(test_inputs, num_runs=10)
        
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
