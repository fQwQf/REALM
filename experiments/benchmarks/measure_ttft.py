#!/usr/bin/env python3
"""
TTFT (Time To First Token) Measurement Script
Measures the latency of System 1 (Reflex) and System 2 (Reflection) separately.
"""

import sys
import os
import time
import json
import statistics

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.realm import REALM
from src.state import OUStateController

class TTFTBenchmark:
    def __init__(self):
        self.results = {
            'system1_latencies': [],
            'system2_latencies': [],
            'end_to_end_latencies': [],
            'bridge_quality_scores': []
        }
        
    def measure_system1_only(self, realm_agent, test_inputs, num_runs=10):
        """Measure System 1 (Reflex) TTFT - should be < 300ms target"""
        print("\n=== Measuring System 1 (Reflex) TTFT ===")
        
        latencies = []
        for i, user_input in enumerate(test_inputs[:num_runs]):
            start_time = time.perf_counter()
            
            # Only run System 1 bridge generation
            event_embedding = realm_agent.state_controller.get_impulse(
                realm_agent.state_controller.get_state()
            )
            current_state = realm_agent.state_controller.step(event_embedding)
            bridge = realm_agent.system1_bridge(user_input, current_state)
            
            end_time = time.perf_counter()
            latency_ms = (end_time - start_time) * 1000
            latencies.append(latency_ms)
            
            print(f"  Run {i+1}: {latency_ms:.2f}ms - Bridge: '{bridge}'")
        
        self.results['system1_latencies'] = latencies
        return latencies
    
    def measure_system2_only(self, realm_agent, test_inputs, num_runs=5):
        """Measure System 2 (Reflection) latency"""
        print("\n=== Measuring System 2 (Reflection) Latency ===")
        
        latencies = []
        for i, user_input in enumerate(test_inputs[:num_runs]):
            start_time = time.perf_counter()
            
            # Simulate System 2 processing
            event_embedding = realm_agent.state_controller.get_impulse(
                realm_agent.state_controller.get_state()
            )
            current_state = realm_agent.state_controller.step(event_embedding)
            context = realm_agent.memory.retrieve(user_input)
            response = realm_agent.system2_response(user_input, current_state, context)
            
            end_time = time.perf_counter()
            latency_ms = (end_time - start_time) * 1000
            latencies.append(latency_ms)
            
            print(f"  Run {i+1}: {latency_ms:.2f}ms - Response: '{response[:50]}...'")
        
        self.results['system2_latencies'] = latencies
        return latencies
    
    def measure_end_to_end(self, realm_agent, test_inputs, num_runs=10):
        """Measure full REALM end-to-end latency"""
        print("\n=== Measuring End-to-End Latency ===")
        
        latencies = []
        first_token_times = []
        
        for i, user_input in enumerate(test_inputs[:num_runs]):
            start_time = time.perf_counter()
            
            # Full REALM step
            response = realm_agent.step(user_input)
            
            end_time = time.perf_counter()
            latency_ms = (end_time - start_time) * 1000
            latencies.append(latency_ms)
            
            print(f"  Run {i+1}: {latency_ms:.2f}ms - Response: '{response[:60]}...'")
        
        self.results['end_to_end_latencies'] = latencies
        return latencies
    
    def compute_statistics(self):
        """Compute statistics for all measurements"""
        stats = {}
        
        for key, values in self.results.items():
            if values:
                stats[key] = {
                    'mean': statistics.mean(values),
                    'median': statistics.median(values),
                    'min': min(values),
                    'max': max(values),
                    'stdev': statistics.stdev(values) if len(values) > 1 else 0,
                    'p50': statistics.median(values),
                    'p95': sorted(values)[int(len(values) * 0.95)] if len(values) > 1 else values[0],
                    'p99': sorted(values)[int(len(values) * 0.99)] if len(values) > 1 else values[0]
                }
        
        return stats
    
    def evaluate_against_targets(self, stats):
        """Evaluate results against paper targets"""
        print("\n=== Evaluation Against Paper Targets ===")
        
        targets = {
            'system1_ttft': {'target': 300, 'direction': 'less'},
            'pnh_accuracy': {'target': 76, 'direction': 'greater'}
        }
        
        # Check System 1 TTFT
        if 'system1_latencies' in stats:
            s1_mean = stats['system1_latencies']['mean']
            s1_p50 = stats['system1_latencies']['p50']
            
            print(f"\nSystem 1 TTFT:")
            print(f"  Mean: {s1_mean:.2f}ms (Target: <300ms)")
            print(f"  P50:  {s1_p50:.2f}ms (Paper reports ~210ms)")
            
            if s1_mean < 300:
                print(f"  ✓ PASS: Mean TTFT below 300ms threshold")
            else:
                print(f"  ✗ FAIL: Mean TTFT exceeds 300ms threshold")
            
            if s1_p50 <= 210:
                print(f"  ✓ PASS: P50 matches or beats paper (~210ms)")
            elif s1_p50 < 250:
                print(f"  ⚠ CLOSE: P50 close to paper value")
            else:
                print(f"  ✗ FAIL: P50 significantly higher than paper")
        
        # Check End-to-End
        if 'end_to_end_latencies' in stats:
            e2e_mean = stats['end_to_end_latencies']['mean']
            print(f"\nEnd-to-End Latency:")
            print(f"  Mean: {e2e_mean:.2f}ms")
            print(f"  (Paper reports 1550ms for full system)")
        
        return True
    
    def save_results(self, filepath):
        """Save results to JSON file"""
        stats = self.compute_statistics()
        
        output = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'results': self.results,
            'statistics': stats
        }
        
        with open(filepath, 'w') as f:
            json.dump(output, f, indent=2)
        
        print(f"\n✓ Results saved to {filepath}")
        return output

def run_ttft_benchmark():
    """Main benchmark function"""
    print("="*60)
    print("REALM TTFT (Time To First Token) Benchmark")
    print("="*60)
    
    # Initialize REALM
    print("\nInitializing REALM agent...")
    agent = REALM()
    
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
    
    # Create benchmark
    benchmark = TTFTBenchmark()
    
    # Run measurements
    print("\n" + "="*60)
    print("Starting Measurements")
    print("="*60)
    
    # System 1 only (simulated - in real implementation uses actual LLM)
    s1_latencies = benchmark.measure_system1_only(agent, test_inputs, num_runs=10)
    
    # System 2 only (simulated)
    s2_latencies = benchmark.measure_system2_only(agent, test_inputs, num_runs=5)
    
    # End-to-end
    e2e_latencies = benchmark.measure_end_to_end(agent, test_inputs, num_runs=10)
    
    # Compute statistics
    print("\n" + "="*60)
    print("Statistics")
    print("="*60)
    
    stats = benchmark.compute_statistics()
    for metric, values in stats.items():
        print(f"\n{metric}:")
        print(f"  Mean:   {values['mean']:.2f}ms")
        print(f"  Median: {values['median']:.2f}ms")
        print(f"  Min:    {values['min']:.2f}ms")
        print(f"  Max:    {values['max']:.2f}ms")
        print(f"  Stdev:  {values['stdev']:.2f}ms")
        print(f"  P95:    {values['p95']:.2f}ms")
    
    # Evaluate
    benchmark.evaluate_against_targets(stats)
    
    # Save results
    os.makedirs('results', exist_ok=True)
    results_file = 'results/ttft_benchmark_results.json'
    benchmark.save_results(results_file)
    
    print("\n" + "="*60)
    print("TTFT Benchmark Complete")
    print("="*60)
    
    return benchmark

if __name__ == "__main__":
    run_ttft_benchmark()
