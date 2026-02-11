#!/usr/bin/env python3
"""
Comprehensive Experiment Runner for REALM Paper Reproduction
Runs all experiments with dynamic GPU allocation and parallel execution support.

Usage:
    python experiments/run_comprehensive_experiments.py [--parallel] [--gpu-check]

Experiments:
    1. TTFT Measurement (Time To First Token)
    2. PNH Evaluation (Psychological Needle-in-Haystack)
    3. Ablation Study (Component analysis)

Results saved to: results/comprehensive_experiments/
"""

import os
import sys
import time
import json
import argparse
import subprocess
import statistics
from datetime import datetime
from typing import Dict, List, Optional, Tuple

# Set HF mirror
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['HF_HOME'] = '/data1/tongjizhou/.cache/huggingface'

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


def get_available_gpus(min_memory_mb: int = 15000) -> List[int]:
    """Get list of available GPUs with sufficient free memory."""
    try:
        import torch
        available = []
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            mem_total = props.total_memory / (1024 ** 2)
            
            # Get memory info using torch.cuda.mem_get_info() (newer API)
            try:
                mem_free, mem_total_device = torch.cuda.mem_get_info(i)
                mem_free_mb = mem_free / (1024 ** 2)
            except:
                # Fallback: assume all memory is available
                mem_free_mb = mem_total
            
            # Check if GPU has enough free memory
            if mem_free_mb >= min_memory_mb:
                available.append(i)
                print(f"  GPU {i}: {props.name}, {mem_free_mb:.0f}MB free / {mem_total:.0f}MB total ✓")
            else:
                print(f"  GPU {i}: {props.name}, {mem_free_mb:.0f}MB free (insufficient) ✗")
        return available
    except Exception as e:
        print(f"Error checking GPUs: {e}")
        import traceback
        traceback.print_exc()
        return []


def allocate_gpus(available_gpus: List[int]) -> Dict[str, Tuple[int, List[int]]]:
    """
    Allocate GPUs for experiments.
    Returns dict mapping experiment name to (sys1_gpu, sys2_gpus).
    """
    allocations = {}
    
    if len(available_gpus) >= 2:
        # For TTFT: Use first two available GPUs
        allocations['ttft'] = (available_gpus[0], [available_gpus[1]])
        
        # For PNH: Use next two available GPUs if possible
        if len(available_gpus) >= 4:
            allocations['pnh'] = (available_gpus[2], [available_gpus[3]])
            allocations['ablation'] = (available_gpus[0], [available_gpus[1]])  # Reuse
        elif len(available_gpus) >= 2:
            # Sequential execution with same GPUs
            allocations['pnh'] = (available_gpus[0], [available_gpus[1]])
            allocations['ablation'] = (available_gpus[0], [available_gpus[1]])
    else:
        raise RuntimeError(f"Not enough available GPUs. Need at least 2, got {len(available_gpus)}")
    
    return allocations


class ComprehensiveExperimentRunner:
    """Run all REALM experiments with proper GPU allocation."""
    
    def __init__(self, output_dir: str = "results/comprehensive_experiments"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.results = {
            'timestamp': self.timestamp,
            'experiments': {}
        }
        
    def log(self, message: str):
        """Log message with timestamp."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        print(f"[{timestamp}] {message}")
        
    def run_ttft_experiment(self, sys1_gpu: int, sys2_gpus: List[int]) -> Dict:
        """Run TTFT measurement experiment."""
        import torch
        self.log("="*60)
        self.log("Running TTFT Measurement Experiment")
        self.log("="*60)
        
        # Import and configure
        from src.real_realm import RealREALM
        
        results = {
            'ttft_values': [],
            'system2_latencies': [],
            'test_cases': []
        }
        
        try:
            # Initialize REALM
            self.log(f"Initializing REALM: System 1 on GPU {sys1_gpu}, System 2 on GPU {sys2_gpus}")
            realm = RealREALM(
                use_real_llm=True,
                sys1_gpu=sys1_gpu,
                sys2_gpus=sys2_gpus
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
                "Can you remember what I said earlier?"
            ]
            
            # Run warmup first (to exclude cold start)
            self.log("Running warmup iteration...")
            _ = realm.step("Warmup message")
            
            # Run actual tests
            self.log(f"Running {len(test_inputs)} TTFT tests...")
            for i, user_input in enumerate(test_inputs):
                try:
                    start = time.perf_counter()
                    response, metadata = realm.step(user_input)
                    end = time.perf_counter()
                    
                    ttft = metadata['ttft_ms']
                    sys2_latency = metadata['system2_latency_ms']
                    e2e = (end - start) * 1000
                    
                    results['ttft_values'].append(ttft)
                    results['system2_latencies'].append(sys2_latency)
                    results['test_cases'].append({
                        'input': user_input,
                        'ttft_ms': ttft,
                        'sys2_ms': sys2_latency,
                        'e2e_ms': e2e,
                        'response': response[:100]
                    })
                    
                    self.log(f"  Test {i+1}: TTFT={ttft:.1f}ms, Sys2={sys2_latency:.1f}ms")
                    
                except Exception as e:
                    self.log(f"  Test {i+1} failed: {e}")
            
            # Compute statistics
            if results['ttft_values']:
                stats = {
                    'ttft': self._compute_stats(results['ttft_values']),
                    'system2': self._compute_stats(results['system2_latencies'])
                }
                results['statistics'] = stats
                
                self.log("\nTTFT Statistics:")
                self.log(f"  Mean:   {stats['ttft']['mean']:.1f}ms")
                self.log(f"  Median: {stats['ttft']['median']:.1f}ms")
                self.log(f"  P50:    {stats['ttft']['p50']:.1f}ms")
                self.log(f"  P95:    {stats['ttft']['p95']:.1f}ms")
                
                # Compare with paper target
                target = 300  # ms
                paper_p50 = 210  # ms
                if stats['ttft']['p50'] < target:
                    self.log(f"  ✓ P50 ({stats['ttft']['p50']:.1f}ms) < {target}ms threshold")
                else:
                    self.log(f"  ✗ P50 ({stats['ttft']['p50']:.1f}ms) >= {target}ms threshold")
            
            # Cleanup
            del realm
            torch.cuda.empty_cache()
            
        except Exception as e:
            self.log(f"TTFT experiment failed: {e}")
            import traceback
            traceback.print_exc()
            results['error'] = str(e)
        
        return results
    
    def run_pnh_experiment(self, sys1_gpu: int, sys2_gpus: List[int]) -> Dict:
        """Run PNH accuracy evaluation experiment."""
        import torch
        self.log("="*60)
        self.log("Running PNH Accuracy Evaluation")
        self.log("="*60)
        
        from src.real_realm import RealREALM
        
        results = {
            'total_tests': 0,
            'passed': 0,
            'recall_success': 0,
            'state_aligned': 0,
            'details': []
        }
        
        # Load test cases
        test_path = 'data/test_sets/pnh_test_set.json'
        try:
            with open(test_path, 'r') as f:
                test_data = json.load(f)
            test_cases = test_data.get('test_cases', [])
            self.log(f"Loaded {len(test_cases)} PNH test cases")
        except Exception as e:
            self.log(f"Failed to load test cases: {e}")
            return {'error': str(e)}
        
        try:
            # Initialize REALM
            self.log(f"Initializing REALM: System 1 on GPU {sys1_gpu}, System 2 on GPU {sys2_gpus}")
            
            # Run each test case
            for i, test_case in enumerate(test_cases):
                self.log(f"\n--- Test {i+1}/{len(test_cases)}: {test_case['name']} ---")
                
                try:
                    # Create fresh REALM instance for each test
                    realm = RealREALM(
                        use_real_llm=True,
                        sys1_gpu=sys1_gpu,
                        sys2_gpus=sys2_gpus
                    )
                    
                    # Warmup
                    _ = realm.step("Hello")
                    
                    # Simulate conversation with distractor turns
                    needle_content = test_case['needle']['content'].lower()
                    implant_turn = test_case['needle']['implant_turn']
                    
                    for j, turn in enumerate(test_case['distractor_turns'][:implant_turn+2]):
                        user_msg = turn['user']
                        _ = realm.step(user_msg)
                        if j == implant_turn:
                            self.log(f"  [Implanted needle: '{needle_content[:30]}...']")
                    
                    # Trigger query
                    trigger = test_case['trigger_query']
                    self.log(f"  Trigger: '{trigger}'")
                    
                    response, metadata = realm.step(trigger)
                    self.log(f"  Response: '{response[:80]}...'")
                    
                    # Evaluate
                    correct_response = test_case['correct_response'].lower()
                    response_lower = response.lower()
                    
                    # Check recall
                    keywords = correct_response.split()
                    recall_score = sum(1 for kw in keywords if kw in response_lower)
                    recall_success = recall_score >= len(keywords) * 0.5
                    
                    # Check state alignment (simplified)
                    state_markers = ['understand', 'respect', 'sorry', 'gentle', 'careful', 'yes', 'certainly']
                    state_aligned = any(marker in response_lower for marker in state_markers)
                    
                    passed = recall_success
                    
                    results['total_tests'] += 1
                    if passed:
                        results['passed'] += 1
                    if recall_success:
                        results['recall_success'] += 1
                    if state_aligned:
                        results['state_aligned'] += 1
                    
                    results['details'].append({
                        'test_id': test_case['id'],
                        'name': test_case['name'],
                        'passed': passed,
                        'recall_success': recall_success,
                        'state_aligned': state_aligned,
                        'ttft_ms': metadata['ttft_ms']
                    })
                    
                    status = "✓ PASS" if passed else "✗ FAIL"
                    self.log(f"  {status} (recall={recall_success}, state={state_aligned})")
                    
                    # Cleanup
                    del realm
                    torch.cuda.empty_cache()
                    
                except Exception as e:
                    self.log(f"  Test failed: {e}")
                    results['total_tests'] += 1
            
            # Compute accuracy
            accuracy = (results['passed'] / results['total_tests'] * 100) if results['total_tests'] > 0 else 0
            results['accuracy_percent'] = accuracy
            
            self.log(f"\n--- PNH Summary ---")
            self.log(f"Total: {results['total_tests']}, Passed: {results['passed']}")
            self.log(f"Accuracy: {accuracy:.1f}% (Paper target: 76%)")
            
        except Exception as e:
            self.log(f"PNH experiment failed: {e}")
            import traceback
            traceback.print_exc()
            results['error'] = str(e)
        
        return results
    
    def run_ablation_experiment(self, sys1_gpu: int, sys2_gpus: List[int]) -> Dict:
        """Run ablation study experiment."""
        import torch
        self.log("="*60)
        self.log("Running Ablation Study")
        self.log("="*60)
        
        from src.real_realm import RealREALM
        
        # Define variants based on paper Table 2
        variants = [
            {
                'id': 1, 'name': 'Vanilla RAG (baseline)',
                'dual_stream': False, 'homeostasis': False,
                'motivated_retrieval': False, 'accordion_memory': False, 'parametric_subconscious': False,
                'expected': {'ttft': 520, 'pnh_acc': 54, 'task_score': 0.62}
            },
            {
                'id': 2, 'name': 'w/o Homeostasis',
                'dual_stream': True, 'homeostasis': False,
                'motivated_retrieval': False, 'accordion_memory': True, 'parametric_subconscious': True,
                'expected': {'ttft': 190, 'pnh_acc': 65, 'task_score': 0.65}
            },
            {
                'id': 3, 'name': 'w/o Dual-Stream',
                'dual_stream': False, 'homeostasis': True,
                'motivated_retrieval': True, 'accordion_memory': True, 'parametric_subconscious': True,
                'expected': {'ttft': 560, 'pnh_acc': 78, 'task_score': 0.72}
            },
            {
                'id': 4, 'name': 'w/o Motivated Retrieval',
                'dual_stream': True, 'homeostasis': True,
                'motivated_retrieval': False, 'accordion_memory': True, 'parametric_subconscious': True,
                'expected': {'ttft': 210, 'pnh_acc': 68, 'task_score': 0.68}
            },
            {
                'id': 7, 'name': 'REALM (Full)',
                'dual_stream': True, 'homeostasis': True,
                'motivated_retrieval': True, 'accordion_memory': True, 'parametric_subconscious': True,
                'expected': {'ttft': 210, 'pnh_acc': 76, 'task_score': 0.74}
            }
        ]
        
        results = {
            'variants': []
        }
        
        try:
            for variant in variants:
                self.log(f"\n--- Variant {variant['id']}: {variant['name']} ---")
                
                try:
                    # Create REALM with variant config
                    realm = RealREALM(
                        use_real_llm=True,
                        sys1_gpu=sys1_gpu,
                        sys2_gpus=sys2_gpus,
                        config={
                            'dual_stream': variant['dual_stream'],
                            'homeostasis': variant['homeostasis'],
                            'motivated_retrieval': variant['motivated_retrieval'],
                            'accordion_memory': variant['accordion_memory'],
                            'parametric_subconscious': variant['parametric_subconscious']
                        }
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
                    
                    # Simplified PNH test (use 3 cases for speed)
                    pnh_correct = 0
                    pnh_total = 3
                    
                    test_path = 'data/test_sets/pnh_test_set.json'
                    with open(test_path, 'r') as f:
                        test_data = json.load(f)
                    
                    for test_case in test_data['test_cases'][:pnh_total]:
                        # Simplified: just check if trigger response contains keywords
                        trigger = test_case['trigger_query']
                        correct = test_case['correct_response'].lower()
                        response, _ = realm.step(trigger)
                        if any(kw in response.lower() for kw in correct.split()):
                            pnh_correct += 1
                    
                    pnh_acc = (pnh_correct / pnh_total) * 100
                    
                    # Compute task score (simplified)
                    ttft_score = max(0, min(1, (600 - avg_ttft) / 400))
                    pnh_score = pnh_acc / 100
                    task_score = round(ttft_score * 0.3 + pnh_score * 0.7, 2)
                    
                    variant_result = {
                        'id': variant['id'],
                        'name': variant['name'],
                        'config': {
                            'dual_stream': variant['dual_stream'],
                            'homeostasis': variant['homeostasis'],
                            'motivated_retrieval': variant['motivated_retrieval']
                        },
                        'measured': {
                            'ttft': round(avg_ttft, 0),
                            'pnh_acc': round(pnh_acc, 0),
                            'task_score': task_score
                        },
                        'expected': variant['expected']
                    }
                    
                    results['variants'].append(variant_result)
                    
                    self.log(f"  TTFT: {avg_ttft:.0f}ms (expected: {variant['expected']['ttft']}ms)")
                    self.log(f"  PNH:  {pnh_acc:.0f}% (expected: {variant['expected']['pnh_acc']}%)")
                    self.log(f"  Task: {task_score:.2f} (expected: {variant['expected']['task_score']:.2f})")
                    
                    # Cleanup
                    del realm
                    torch.cuda.empty_cache()
                    
                except Exception as e:
                    self.log(f"  Variant failed: {e}")
                    results['variants'].append({
                        'id': variant['id'],
                        'name': variant['name'],
                        'error': str(e)
                    })
        
        except Exception as e:
            self.log(f"Ablation experiment failed: {e}")
            import traceback
            traceback.print_exc()
            results['error'] = str(e)
        
        return results
    
    def _compute_stats(self, values: List[float]) -> Dict:
        """Compute statistics for a list of values."""
        if not values:
            return {}
        
        sorted_vals = sorted(values)
        n = len(values)
        
        return {
            'mean': statistics.mean(values),
            'median': statistics.median(values),
            'min': min(values),
            'max': max(values),
            'stdev': statistics.stdev(values) if n > 1 else 0,
            'p50': sorted_vals[int(n * 0.5)],
            'p95': sorted_vals[int(n * 0.95)] if n > 1 else values[0]
        }
    
    def save_results(self):
        """Save all results to file."""
        output_file = os.path.join(self.output_dir, f"experiment_results_{self.timestamp}.json")
        
        with open(output_file, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        self.log(f"\nResults saved to {output_file}")
        return output_file
    
    def generate_report(self) -> str:
        """Generate a summary report."""
        report = []
        report.append("="*70)
        report.append("REALM Experiment Results Summary")
        report.append(f"Timestamp: {self.timestamp}")
        report.append("="*70)
        
        # TTFT Results
        if 'ttft' in self.results['experiments']:
            ttft = self.results['experiments']['ttft']
            if 'statistics' in ttft:
                stats = ttft['statistics']['ttft']
                report.append("\n## TTFT (Time To First Token)")
                report.append(f"P50: {stats['p50']:.1f}ms (Paper: 210ms, Target: <300ms)")
                report.append(f"Mean: {stats['mean']:.1f}ms")
                report.append(f"P95: {stats['p95']:.1f}ms")
                status = "✓ PASS" if stats['p50'] < 300 else "✗ FAIL"
                report.append(f"Status: {status}")
        
        # PNH Results
        if 'pnh' in self.results['experiments']:
            pnh = self.results['experiments']['pnh']
            if 'accuracy_percent' in pnh:
                report.append("\n## PNH (Psychological Needle-in-Haystack)")
                report.append(f"Accuracy: {pnh['accuracy_percent']:.1f}% (Paper: 76%)")
                report.append(f"Passed: {pnh['passed']}/{pnh['total_tests']}")
                report.append(f"Recall Success: {pnh['recall_success']}")
                report.append(f"State Aligned: {pnh['state_aligned']}")
        
        # Ablation Results
        if 'ablation' in self.results['experiments']:
            ablation = self.results['experiments']['ablation']
            if 'variants' in ablation:
                report.append("\n## Ablation Study Results")
                report.append(f"{'Variant':<30} {'TTFT':<10} {'PNH':<10} {'Task':<10}")
                report.append("-"*60)
                for v in ablation['variants']:
                    if 'measured' in v:
                        m = v['measured']
                        report.append(f"{v['name'][:29]:<30} {m['ttft']:<10.0f} {m['pnh_acc']:<10.0f} {m['task_score']:<10.2f}")
        
        report.append("\n" + "="*70)
        report.append("End of Report")
        report.append("="*70)
        
        report_text = "\n".join(report)
        
        # Save report
        report_file = os.path.join(self.output_dir, f"experiment_report_{self.timestamp}.txt")
        with open(report_file, 'w') as f:
            f.write(report_text)
        
        print(report_text)
        return report_file
    
    def run_all(self, gpu_allocations: Dict[str, Tuple[int, List[int]]]):
        """Run all experiments sequentially."""
        self.log("="*70)
        self.log("REALM Comprehensive Experiment Runner")
        self.log("="*70)
        
        # TTFT Experiment
        if 'ttft' in gpu_allocations:
            sys1_gpu, sys2_gpus = gpu_allocations['ttft']
            self.results['experiments']['ttft'] = self.run_ttft_experiment(sys1_gpu, sys2_gpus)
        
        # PNH Experiment
        if 'pnh' in gpu_allocations:
            sys1_gpu, sys2_gpus = gpu_allocations['pnh']
            self.results['experiments']['pnh'] = self.run_pnh_experiment(sys1_gpu, sys2_gpus)
        
        # Ablation Experiment
        if 'ablation' in gpu_allocations:
            sys1_gpu, sys2_gpus = gpu_allocations['ablation']
            self.results['experiments']['ablation'] = self.run_ablation_experiment(sys1_gpu, sys2_gpus)
        
        # Save and report
        self.save_results()
        self.generate_report()


def main():
    parser = argparse.ArgumentParser(description='REALM Comprehensive Experiment Runner')
    parser.add_argument('--parallel', action='store_true', help='Run experiments in parallel')
    parser.add_argument('--gpu-check', action='store_true', help='Only check GPU availability')
    parser.add_argument('--output-dir', default='results/comprehensive_experiments', help='Output directory')
    parser.add_argument('--experiment', choices=['ttft', 'pnh', 'ablation', 'all'], default='all',
                        help='Which experiment to run')
    args = parser.parse_args()
    
    # Check GPU availability
    print("Checking GPU availability...")
    print("-" * 40)
    
    import torch
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"GPU count: {torch.cuda.device_count()}")
    print()
    
    # Get available GPUs (need at least 15GB for 7B model)
    available_gpus = get_available_gpus(min_memory_mb=15000)
    print(f"\nAvailable GPUs: {available_gpus}")
    
    if args.gpu_check:
        return
    
    if len(available_gpus) < 2:
        print("ERROR: Need at least 2 GPUs with sufficient memory")
        return
    
    # Allocate GPUs
    gpu_allocations = allocate_gpus(available_gpus)
    print(f"\nGPU Allocations:")
    for exp, (sys1, sys2) in gpu_allocations.items():
        print(f"  {exp}: System 1 on GPU {sys1}, System 2 on GPU {sys2}")
    
    # Filter experiments if specified
    if args.experiment != 'all':
        gpu_allocations = {args.experiment: gpu_allocations.get(args.experiment, gpu_allocations.get('ttft'))}
    
    # Run experiments
    runner = ComprehensiveExperimentRunner(output_dir=args.output_dir)
    runner.run_all(gpu_allocations)


if __name__ == "__main__":
    main()
