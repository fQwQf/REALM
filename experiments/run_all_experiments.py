#!/usr/bin/env python3
"""
Main Experiment Runner for REALM Paper Reproduction

This script runs all experiments described in the paper:
1. Basic simulation
2. TTFT measurement
3. PNH accuracy evaluation
4. Ablation study
5. Comparison with baselines

Usage:
    python experiments/run_all_experiments.py
    
Results are saved to results/ directory.
"""

import sys
import os
import time
import json
import subprocess

# Add project root to path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

def print_section(title):
    """Print a formatted section header"""
    print("\n" + "="*70)
    print(f"  {title}")
    print("="*70 + "\n")

def run_command(cmd, description):
    """Run a command and handle errors"""
    print(f"Running: {description}")
    print(f"Command: {' '.join(cmd)}\n")
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout per experiment
        )
        
        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
        
        if result.returncode == 0:
            print(f"✓ {description} completed successfully\n")
            return True
        else:
            print(f"✗ {description} failed with return code {result.returncode}\n")
            return False
    except subprocess.TimeoutExpired:
        print(f"✗ {description} timed out after 5 minutes\n")
        return False
    except Exception as e:
        print(f"✗ {description} failed with error: {e}\n")
        return False

def check_environment():
    """Check if environment is properly set up"""
    print_section("Environment Check")
    
    checks = {
        'realm_env': False,
        'dependencies': False,
        'gpu_available': False,
        'test_data': False
    }
    
    # Check conda environment
    conda_env = os.environ.get('CONDA_DEFAULT_ENV', '')
    if conda_env == 'realm':
        print("✓ Conda environment 'realm' is active")
        checks['realm_env'] = True
    else:
        print("⚠ Conda environment 'realm' not detected")
        print(f"  Current environment: {conda_env or 'None'}")
    
    # Check if we can import dependencies
    try:
        import numpy
        print("✓ NumPy available")
        try:
            import torch
            print(f"✓ PyTorch available (version: {torch.__version__})")
            checks['dependencies'] = True
        except ImportError:
            print("✗ PyTorch not available (will use mock mode)")
    except ImportError:
        print("✗ NumPy not available")
    
    # Check GPU availability
    try:
        import torch
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            print(f"✓ GPUs available: {gpu_count}")
            for i in range(gpu_count):
                name = torch.cuda.get_device_name(i)
                print(f"    GPU {i}: {name}")
            checks['gpu_available'] = True
        else:
            print("⚠ No GPUs available (will run in CPU mode)")
    except:
        print("⚠ Cannot check GPU availability")
    
    # Check test data
    if os.path.exists('data/test_sets/pnh_test_set.json'):
        print("✓ PNH test data available")
        checks['test_data'] = True
    else:
        print("✗ PNH test data not found")
    
    return checks

def run_experiment_1_simulation():
    """Run basic simulation experiment"""
    print_section("Experiment 1: Basic Simulation")
    
    cmd = [sys.executable, 'experiments/run_simulation.py']
    return run_command(cmd, "Basic REALM simulation")

def run_experiment_2_ttft():
    """Run TTFT measurement"""
    print_section("Experiment 2: TTFT Measurement")
    
    cmd = [sys.executable, 'experiments/benchmarks/measure_ttft.py']
    return run_command(cmd, "TTFT benchmark")

def run_experiment_3_pnh():
    """Run PNH evaluation"""
    print_section("Experiment 3: PNH Accuracy Evaluation")
    
    cmd = [sys.executable, 'experiments/benchmarks/evaluate_pnh.py']
    return run_command(cmd, "PNH evaluation")

def run_experiment_4_ablation():
    """Run ablation study"""
    print_section("Experiment 4: Ablation Study")
    
    cmd = [sys.executable, 'experiments/benchmarks/run_ablation_study.py']
    return run_command(cmd, "Ablation study")

def compile_results():
    """Compile all results into a summary"""
    print_section("Results Compilation")
    
    results_dir = 'results'
    if not os.path.exists(results_dir):
        print("✗ Results directory not found")
        return None
    
    summary = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'experiments': {}
    }
    
    # Load individual results
    result_files = {
        'ttft': 'results/ttft_benchmark_results.json',
        'pnh': 'results/pnh_evaluation_results.json',
        'ablation': 'results/ablation_study_results.json'
    }
    
    for exp_name, filepath in result_files.items():
        if os.path.exists(filepath):
            try:
                with open(filepath, 'r') as f:
                    data = json.load(f)
                summary['experiments'][exp_name] = data
                print(f"✓ Loaded {exp_name} results from {filepath}")
            except Exception as e:
                print(f"✗ Failed to load {exp_name}: {e}")
        else:
            print(f"✗ {exp_name} results not found at {filepath}")
    
    # Save summary
    summary_path = 'results/experiment_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n✓ Summary saved to {summary_path}")
    
    return summary

def print_final_report(summary):
    """Print final report comparing with paper"""
    print_section("Final Report: Comparison with Paper")
    
    if not summary or 'experiments' not in summary:
        print("No results available to report")
        return
    
    experiments = summary['experiments']
    
    # Paper targets from Table 1
    paper_targets = {
        'realm_full': {
            'ttft': 210,
            'pnh_acc': 76,
            'consistency': 4.10,
            'naturalness': 4.05
        },
        'vanilla_rag': {
            'ttft': 520,
            'pnh_acc': 54
        },
        'wo_state': {
            'ttft': 190,
            'pnh_acc': 65
        },
        'wo_dual_stream': {
            'ttft': 560,
            'pnh_acc': 78
        }
    }
    
    print("\n" + "="*70)
    print("Comparison with Paper (Table 1: All-in-One)")
    print("="*70)
    
    # TTFT results
    if 'ttft' in experiments and 'statistics' in experiments['ttft']:
        stats = experiments['ttft']['statistics']
        if 'system1_latencies' in stats:
            measured_ttft = stats['system1_latencies']['mean']
            target_ttft = paper_targets['realm_full']['ttft']
            
            print(f"\nTTFT (Time To First Token):")
            print(f"  Measured: {measured_ttft:.1f}ms")
            print(f"  Paper (REALM Full): {target_ttft}ms")
            print(f"  Target (<300ms): {'✓ PASS' if measured_ttft < 300 else '✗ FAIL'}")
            print(f"  Match paper: {'✓' if abs(measured_ttft - target_ttft) < 50 else '✗'}")
    
    # PNH results
    if 'pnh' in experiments and 'summary' in experiments['pnh']:
        pnh_summary = experiments['pnh']['summary']
        measured_pnh = pnh_summary.get('accuracy_percent', 0)
        target_pnh = paper_targets['realm_full']['pnh_acc']
        
        print(f"\nPNH (Psychological Needle-in-Haystack) Accuracy:")
        print(f"  Measured: {measured_pnh:.1f}%")
        print(f"  Paper (REALM Full): {target_pnh}%")
        print(f"  Match paper: {'✓' if abs(measured_pnh - target_pnh) < 5 else '✗'}")
    
    # Ablation results
    if 'ablation' in experiments and 'variants' in experiments['ablation']:
        variants = experiments['ablation']['variants']
        
        print(f"\nAblation Study Results:")
        print(f"  {'Variant':<35} {'TTFT':<12} {'PNH':<12} {'Task Score':<12}")
        print(f"  {'-'*71}")
        
        for variant in variants:
            name = variant['name'][:34]
            measured = variant['measured']
            print(f"  {name:<35} {measured['ttft']:<12.0f} {measured['pnh_acc']:<12.0f} {measured['task_score']:<12.2f}")
    
    print("\n" + "="*70)
    print("Summary: Experiments completed and results saved to results/")
    print("="*70)

def main():
    """Main entry point"""
    print("="*70)
    print("REALM Paper Reproduction - Complete Experiment Suite")
    print("="*70)
    print(f"Start time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Check environment
    env_checks = check_environment()
    
    # Create results directory
    os.makedirs('results', exist_ok=True)
    
    # Run experiments
    results = {}
    
    # Experiment 1: Basic simulation
    results['simulation'] = run_experiment_1_simulation()
    
    # Experiment 2: TTFT measurement
    results['ttft'] = run_experiment_2_ttft()
    
    # Experiment 3: PNH evaluation
    results['pnh'] = run_experiment_3_pnh()
    
    # Experiment 4: Ablation study
    results['ablation'] = run_experiment_4_ablation()
    
    # Compile results
    summary = compile_results()
    
    # Print final report
    print_final_report(summary)
    
    # Done
    print_section("All Experiments Complete")
    print(f"End time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("\nResults saved in:")
    print("  - results/ttft_benchmark_results.json")
    print("  - results/pnh_evaluation_results.json")
    print("  - results/ablation_study_results.json")
    print("  - results/experiment_summary.json")
    print("\nSee EXPERIMENT_PLAN.md for full reproduction documentation.")
    print("="*70)
    
    return results

if __name__ == "__main__":
    main()
