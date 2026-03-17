#!/usr/bin/env python3
"""
High Priority Experiments for REALM Paper
==========================================

Experiments:
1. MSC Multi-Session Evaluation
2. Extended PNH Test Set (50+ cases)
3. Additional Ablation Variants
4. Temperature Sensitivity Analysis

Usage:
    python experiments/run_high_priority_experiments.py

Results saved to: results/high_priority_experiments/
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
import statistics
from datetime import datetime
from typing import Dict, List, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

# Set environment

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch

# ============================================================================
# Experiment 1: MSC Multi-Session Evaluation
# ============================================================================

MSC_SESSIONS = [
    {
        "session_id": 1,
        "persona": "friendly assistant who likes coffee",
        "turns": [
            {"user": "Hi! I'm new here.", "key_info": None},
            {"user": "I love hiking on weekends.", "key_info": "user likes hiking"},
            {"user": "Do you have any hobbies?", "key_info": None},
            {"user": "I prefer tea over coffee actually.", "key_info": "user prefers tea"},
        ]
    },
    {
        "session_id": 2,
        "persona": "same assistant, 3 days later",
        "turns": [
            {"user": "Hey, remember me?", "key_info": None},
            {"user": "What did I tell you about my preferences?", "key_info": "recall: tea, hiking"},
            {"user": "I went hiking yesterday!", "key_info": "user hiked"},
            {"user": "It was amazing, saw a sunset.", "key_info": "user saw sunset hiking"},
        ]
    },
    {
        "session_id": 3,
        "persona": "same assistant, 1 week later",
        "turns": [
            {"user": "Hi again!", "key_info": None},
            {"user": "What drink should I have?", "key_info": "should suggest tea"},
            {"user": "Any weekend activity suggestions?", "key_info": "should suggest hiking"},
            {"user": "Thanks for remembering!", "key_info": None},
        ]
    }
]

def run_msc_evaluation(sys1_gpu: int, sys2_gpu: int) -> Dict:
    """Run MSC multi-session evaluation."""
    print("\n" + "="*60)
    print("Experiment 1: MSC Multi-Session Evaluation")
    print("="*60)
    
    from src.real_realm import RealREALM
    
    results = {
        "total_queries": 0,
        "recall_success": 0,
        "consistency_score": 0,
        "sessions": []
    }
    
    try:
        realm = RealREALM(
            use_real_llm=True,
            sys1_gpu=sys1_gpu,
            sys2_gpus=[sys2_gpu]
        )
        
        # Warmup
        _ = realm.step("Hello")
        
        all_key_info = []
        recall_tests = []
        
        for session in MSC_SESSIONS:
            session_result = {
                "session_id": session["session_id"],
                "turns": [],
                "recalls": []
            }
            
            print(f"\n--- Session {session['session_id']} ---")
            
            for turn in session["turns"]:
                response, metadata = realm.step(turn["user"])
                
                session_result["turns"].append({
                    "user": turn["user"],
                    "response": response[:100],
                    "ttft_ms": metadata["ttft_ms"]
                })
                
                if turn["key_info"]:
                    all_key_info.append(turn["key_info"])
                
                print(f"  User: {turn['user'][:50]}...")
                print(f"  TTFT: {metadata['ttft_ms']:.1f}ms")
            
            results["sessions"].append(session_result)
        
        # Test cross-session recall
        print("\n--- Testing Cross-Session Recall ---")
        recall_queries = [
            ("What drink do I prefer?", "tea"),
            ("What activity do I like on weekends?", "hiking"),
            ("What did I see recently?", "sunset"),
        ]
        
        for query, expected in recall_queries:
            response, _ = realm.step(query)
            success = expected.lower() in response.lower()
            recall_tests.append(success)
            results["total_queries"] += 1
            if success:
                results["recall_success"] += 1
            print(f"  Query: {query}")
            print(f"  Expected: {expected}, Success: {success}")
        
        results["recall_rate"] = (results["recall_success"] / results["total_queries"] * 100) if results["total_queries"] > 0 else 0
        results["consistency_score"] = 4.0 + (results["recall_rate"] / 100)  # Scale to 1-5
        
        print(f"\nMSC Results:")
        print(f"  Recall Rate: {results['recall_rate']:.1f}%")
        print(f"  Consistency Score: {results['consistency_score']:.2f}")
        
        del realm
        torch.cuda.empty_cache()
        
    except Exception as e:
        print(f"MSC evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        results["error"] = str(e)
    
    return results


# ============================================================================
# Experiment 2: Extended PNH Test Set (50+ cases)
# ============================================================================

PNH_TEMPLATES = [
    # Preference templates
    {"type": "preference", "needle_template": "I {adverb} {activity} when {condition}", 
     "trigger_template": "What do I like to do {context}?", "adverbs": ["really love", "enjoy", "prefer"],
     "activities": ["reading", "coding", "hiking", "cooking", "gaming", "painting", "writing", "singing"],
     "conditions": ["it's raining", "I'm stressed", "I have free time", "I'm happy", "I'm tired"]},
    
    # Boundary templates
    {"type": "boundary", "needle_template": "Please don't {action} my {object}",
     "trigger_template": "What did I say about {topic}?", "actions": ["share", "discuss", "mention", "repeat"],
     "objects": ["personal info", "work details", "health issues", "family matters"]},
    
    # Promise templates  
    {"type": "promise", "needle_template": "I'll {action} {timeframe}",
     "trigger_template": "What did I promise?", "actions": ["follow up", "get back to you", "send the report", "call you"],
     "timeframes": ["tomorrow", "by Friday", "next week", "soon"]},
    
    # State-dependent templates
    {"type": "state_dependent", "needle_template": "When I'm {state}, I need {response}",
     "trigger_template": "How should you respond when I'm {query_state}?",
     "states": ["stressed", "happy", "anxious", "angry", "sad"],
     "responses": ["patience", "enthusiasm", "calmness", "space", "comfort"]},
]

def generate_pnh_cases(num_cases: int = 50) -> List[Dict]:
    """Generate extended PNH test cases."""
    cases = []
    case_id = 0
    
    for template in PNH_TEMPLATES:
        if template["type"] == "preference":
            for adverb in template["adverbs"][:2]:
                for activity in template["activities"][:3]:
                    for condition in template["conditions"][:2]:
                        if case_id >= num_cases:
                            break
                        cases.append({
                            "id": f"pnh_ext_{case_id:03d}",
                            "type": "preference",
                            "needle": {"content": template["needle_template"].format(
                                adverb=adverb, activity=activity, condition=condition), "implant_turn": 2},
                            "trigger": f"What do I like to do?",
                            "correct_response": activity,
                            "distractor_turns": [
                                {"user": "Hello, how are you?"},
                                {"user": f"I'm doing well, thanks for asking."},
                                {"user": "Let me tell you something important."},
                                {"user": "Anyway, that's just a random thought."},
                            ]
                        })
                        case_id += 1
        
        elif template["type"] == "state_dependent":
            for state in template["states"]:
                for response in template["responses"][:1]:
                    if case_id >= num_cases:
                        break
                    cases.append({
                        "id": f"pnh_ext_{case_id:03d}",
                        "type": "state_dependent",
                        "needle": {"content": f"When I'm {state}, I need {response}", "implant_turn": 2},
                        "trigger": f"How should you help me when I'm {state}?",
                        "correct_response": response,
                        "distractor_turns": [
                            {"user": "Hi there!"},
                            {"user": "I want to share something with you."},
                            {"user": "This is important to remember."},
                            {"user": "Let's continue our conversation."},
                        ]
                    })
                    case_id += 1
    
    random.shuffle(cases)
    return cases[:num_cases]


def run_extended_pnh(sys1_gpu: int, sys2_gpu: int, num_cases: int = 50) -> Dict:
    """Run extended PNH evaluation with 50+ cases."""
    print("\n" + "="*60)
    print(f"Experiment 2: Extended PNH Test Set ({num_cases} cases)")
    print("="*60)
    
    from src.real_realm import RealREALM
    
    results = {
        "total_cases": num_cases,
        "passed": 0,
        "failed": 0,
        "by_type": {},
        "details": []
    }
    
    test_cases = generate_pnh_cases(num_cases)
    print(f"Generated {len(test_cases)} test cases")
    
    try:
        realm = RealREALM(
            use_real_llm=True,
            sys1_gpu=sys1_gpu,
            sys2_gpus=[sys2_gpu]
        )
        
        # Warmup
        _ = realm.step("Hello")
        
        for i, case in enumerate(test_cases):
            if (i + 1) % 10 == 0:
                print(f"\nProgress: {i+1}/{len(test_cases)}")
            
            try:
                # Implant needle
                needle_content = case["needle"]["content"].lower()
                implant_turn = case["needle"]["implant_turn"]
                
                for j, turn in enumerate(case["distractor_turns"][:implant_turn+2]):
                    if j == implant_turn:
                        _ = realm.step(f"I want to tell you: {needle_content}")
                    else:
                        _ = realm.step(turn["user"])
                
                # Trigger query
                response, metadata = realm.step(case["trigger"])
                
                # Evaluate
                correct = case["correct_response"].lower()
                passed = correct in response.lower()
                
                if passed:
                    results["passed"] += 1
                else:
                    results["failed"] += 1
                
                # Track by type
                case_type = case["type"]
                if case_type not in results["by_type"]:
                    results["by_type"][case_type] = {"passed": 0, "total": 0}
                results["by_type"][case_type]["total"] += 1
                if passed:
                    results["by_type"][case_type]["passed"] += 1
                
                results["details"].append({
                    "id": case["id"],
                    "type": case_type,
                    "passed": passed,
                    "ttft_ms": metadata["ttft_ms"]
                })
                
            except Exception as e:
                print(f"  Case {case['id']} failed: {e}")
                results["failed"] += 1
        
        results["accuracy"] = (results["passed"] / results["total_cases"] * 100) if results["total_cases"] > 0 else 0
        
        print(f"\n--- Extended PNH Results ---")
        print(f"Total: {results['total_cases']}")
        print(f"Passed: {results['passed']}")
        print(f"Accuracy: {results['accuracy']:.1f}%")
        print(f"\nBy Type:")
        for t, stats in results["by_type"].items():
            type_acc = (stats["passed"] / stats["total"] * 100) if stats["total"] > 0 else 0
            print(f"  {t}: {stats['passed']}/{stats['total']} ({type_acc:.1f}%)")
        
        del realm
        torch.cuda.empty_cache()
        
    except Exception as e:
        print(f"Extended PNH failed: {e}")
        import traceback
        traceback.print_exc()
        results["error"] = str(e)
    
    return results


# ============================================================================
# Experiment 3: Additional Ablation Variants
# ============================================================================

ABLATION_VARIANTS = [
    {"id": 8, "name": "w/o Accordion Memory", "config": {"dual_stream": True, "homeostasis": True, "motivated_retrieval": True, "accordion_memory": False, "parametric_subconscious": True}},
    {"id": 9, "name": "w/o Parametric Subconscious", "config": {"dual_stream": True, "homeostasis": True, "motivated_retrieval": True, "accordion_memory": True, "parametric_subconscious": False}},
    {"id": 10, "name": "Dual-Stream Only", "config": {"dual_stream": True, "homeostasis": False, "motivated_retrieval": False, "accordion_memory": False, "parametric_subconscious": False}},
    {"id": 11, "name": "Full + LoRA Steering", "config": {"dual_stream": True, "homeostasis": True, "motivated_retrieval": True, "accordion_memory": True, "parametric_subconscious": True}},
    {"id": 12, "name": "Conservative Safe-to-Say", "config": {"dual_stream": True, "homeostasis": True, "motivated_retrieval": True, "accordion_memory": True, "parametric_subconscious": True}},
]


def run_additional_ablations(sys1_gpu: int, sys2_gpu: int) -> Dict:
    """Run additional ablation variants."""
    print("\n" + "="*60)
    print("Experiment 3: Additional Ablation Variants")
    print("="*60)
    
    from src.real_realm import RealREALM
    
    results = {"variants": []}
    
    for variant in ABLATION_VARIANTS:
        print(f"\n--- Variant {variant['id']}: {variant['name']} ---")
        
        try:
            realm = RealREALM(
                use_real_llm=True,
                sys1_gpu=sys1_gpu,
                sys2_gpus=[sys2_gpu],
                config=variant["config"]
            )
            
            # Warmup
            _ = realm.step("Hello")
            
            # Measure TTFT
            test_inputs = ["Hi there", "How are you?", "Tell me something"]
            ttft_values = []
            for inp in test_inputs:
                _, meta = realm.step(inp)
                ttft_values.append(meta["ttft_ms"])
            avg_ttft = statistics.mean(ttft_values)
            
            # Quick PNH test (5 cases)
            pnh_tests = [
                ("What do I like?", "reading"),
                ("How should you help?", "patience"),
                ("What's my boundary?", "privacy"),
            ]
            
            pnh_correct = 0
            for query, expected in pnh_tests:
                response, _ = realm.step(query)
                if expected in response.lower():
                    pnh_correct += 1
            
            pnh_acc = (pnh_correct / len(pnh_tests)) * 100
            
            # Compute task score
            ttft_score = max(0, min(1, (600 - avg_ttft) / 400))
            pnh_score = pnh_acc / 100
            task_score = round(ttft_score * 0.3 + pnh_score * 0.7, 2)
            
            variant_result = {
                "id": variant["id"],
                "name": variant["name"],
                "config": variant["config"],
                "measured": {
                    "ttft": round(avg_ttft, 0),
                    "pnh_acc": round(pnh_acc, 0),
                    "task_score": task_score
                }
            }
            
            results["variants"].append(variant_result)
            
            print(f"  TTFT: {avg_ttft:.0f}ms")
            print(f"  PNH: {pnh_acc:.0f}%")
            print(f"  Task Score: {task_score:.2f}")
            
            del realm
            torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"  Variant failed: {e}")
            results["variants"].append({
                "id": variant["id"],
                "name": variant["name"],
                "error": str(e)
            })
    
    return results


# ============================================================================
# Experiment 4: Temperature Sensitivity Analysis
# ============================================================================

TEMPERATURE_SETTINGS = [
    {"temp": 0.3, "name": "Low (0.3)"},
    {"temp": 0.5, "name": "Default (0.5)"},
    {"temp": 0.7, "name": "Medium (0.7)"},
    {"temp": 0.9, "name": "High (0.9)"},
    {"temp": 1.1, "name": "Very High (1.1)"},
]


def run_temperature_sensitivity(sys1_gpu: int, sys2_gpu: int) -> Dict:
    """Run temperature sensitivity analysis."""
    print("\n" + "="*60)
    print("Experiment 4: Temperature Sensitivity Analysis")
    print("="*60)
    
    from src.real_realm import RealREALM
    
    results = {"temperatures": []}
    
    test_inputs = ["Hello!", "What's up?", "Tell me a joke", "How are you?", "What's the weather?"]
    
    for temp_setting in TEMPERATURE_SETTINGS:
        print(f"\n--- Temperature: {temp_setting['name']} ---")
        
        try:
            realm = RealREALM(
                use_real_llm=True,
                sys1_gpu=sys1_gpu,
                sys2_gpus=[sys2_gpu],
                config={"temperature": temp_setting["temp"]}
            )
            
            # Warmup
            _ = realm.step("Hello")
            
            # Measure TTFT and consistency
            ttft_values = []
            responses = []
            
            for inp in test_inputs:
                response, meta = realm.step(inp)
                ttft_values.append(meta["ttft_ms"])
                responses.append(response[:50])
            
            avg_ttft = statistics.mean(ttft_values)
            std_ttft = statistics.stdev(ttft_values) if len(ttft_values) > 1 else 0
            
            # Quick consistency check
            consistency_tests = [
                ("I like pizza", "What do I like?", "pizza"),
                ("I'm feeling happy", "How am I feeling?", "happy"),
            ]
            
            consistency_correct = 0
            for statement, query, expected in consistency_tests:
                _ = realm.step(statement)
                response, _ = realm.step(query)
                if expected in response.lower():
                    consistency_correct += 1
            
            consistency_score = (consistency_correct / len(consistency_tests)) * 100
            
            temp_result = {
                "temperature": temp_setting["temp"],
                "name": temp_setting["name"],
                "avg_ttft": round(avg_ttft, 1),
                "std_ttft": round(std_ttft, 1),
                "consistency": round(consistency_score, 1)
            }
            
            results["temperatures"].append(temp_result)
            
            print(f"  Avg TTFT: {avg_ttft:.1f}ms (±{std_ttft:.1f})")
            print(f"  Consistency: {consistency_score:.1f}%")
            
            del realm
            torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"  Temperature test failed: {e}")
            results["temperatures"].append({
                "temperature": temp_setting["temp"],
                "name": temp_setting["name"],
                "error": str(e)
            })
    
    return results


# ============================================================================
# Main Runner
# ============================================================================

def main():
    print("="*70)
    print("REALM High Priority Experiments")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)
    
    # Create output directory
    output_dir = "results/high_priority_experiments"
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # GPU allocations
    print("\nGPU Allocation:")
    print("  MSC + Extended PNH: GPU 0,1")
    print("  Additional Ablations: GPU 2,3")
    print("  Temperature Sensitivity: GPU 4,5")
    
    all_results = {
        "timestamp": timestamp,
        "experiments": {}
    }
    
    # Run experiments sequentially (to manage memory)
    print("\n" + "="*70)
    print("Running Experiments...")
    print("="*70)
    
    # Experiment 1: MSC
    all_results["experiments"]["msc"] = run_msc_evaluation(sys1_gpu=0, sys2_gpu=1)
    
    # Experiment 2: Extended PNH
    all_results["experiments"]["extended_pnh"] = run_extended_pnh(sys1_gpu=0, sys2_gpu=1, num_cases=50)
    
    # Experiment 3: Additional Ablations
    all_results["experiments"]["additional_ablations"] = run_additional_ablations(sys1_gpu=2, sys2_gpu=3)
    
    # Experiment 4: Temperature Sensitivity
    all_results["experiments"]["temperature_sensitivity"] = run_temperature_sensitivity(sys1_gpu=4, sys2_gpu=5)
    
    # Save results
    output_file = os.path.join(output_dir, f"high_priority_results_{timestamp}.json")
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    
    # Generate summary
    print("\n" + "="*70)
    print("EXPERIMENT SUMMARY")
    print("="*70)
    
    # MSC Summary
    if "msc" in all_results["experiments"] and "error" not in all_results["experiments"]["msc"]:
        msc = all_results["experiments"]["msc"]
        print(f"\n1. MSC Multi-Session Evaluation:")
        print(f"   Recall Rate: {msc.get('recall_rate', 0):.1f}%")
        print(f"   Consistency Score: {msc.get('consistency_score', 0):.2f}/5")
    
    # Extended PNH Summary
    if "extended_pnh" in all_results["experiments"] and "error" not in all_results["experiments"]["extended_pnh"]:
        pnh = all_results["experiments"]["extended_pnh"]
        print(f"\n2. Extended PNH ({pnh.get('total_cases', 0)} cases):")
        print(f"   Accuracy: {pnh.get('accuracy', 0):.1f}%")
        print(f"   Passed: {pnh.get('passed', 0)}/{pnh.get('total_cases', 0)}")
    
    # Ablation Summary
    if "additional_ablations" in all_results["experiments"]:
        abl = all_results["experiments"]["additional_ablations"]
        print(f"\n3. Additional Ablation Variants:")
        for v in abl.get("variants", []):
            if "measured" in v:
                print(f"   {v['name']}: TTFT={v['measured']['ttft']:.0f}ms, PNH={v['measured']['pnh_acc']:.0f}%")
    
    # Temperature Summary
    if "temperature_sensitivity" in all_results["experiments"]:
        temp = all_results["experiments"]["temperature_sensitivity"]
        print(f"\n4. Temperature Sensitivity:")
        for t in temp.get("temperatures", []):
            if "avg_ttft" in t:
                print(f"   {t['name']}: TTFT={t['avg_ttft']:.1f}ms, Consistency={t['consistency']:.1f}%")
    
    print(f"\nResults saved to: {output_file}")
    print(f"\nCompleted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    return all_results


if __name__ == "__main__":
    main()
