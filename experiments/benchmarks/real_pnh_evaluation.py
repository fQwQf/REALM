#!/usr/bin/env python3
"""
PNH Evaluation with Real LLM
Tests Psychological Needle-in-Haystack accuracy with real inference
"""

import os
import sys
import time
import json

# Set HF mirror
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['HF_HOME'] = '/data1/tongjizhou/.cache/huggingface'

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import torch
import numpy as np
from src.real_realm import RealREALM

class RealPNHEvaluator:
    """PNH evaluation using real LLM and vector retrieval"""
    
    def __init__(self, test_set_path='data/test_sets/pnh_test_set.json'):
        self.test_set_path = test_set_path
        self.test_cases = self.load_test_set()
        self.realm = None
        
        self.results = {
            'total_tests': 0,
            'passed': 0,
            'failed': 0,
            'recall_success': 0,
            'state_aligned': 0,
            'details': []
        }
    
    def initialize(self, sys1_gpu=2, sys2_gpus=[4, 5, 6, 7]):
        """Initialize REALM with real LLM"""
        print("[Initializing Real REALM for PNH Evaluation...]")
        self.realm = RealREALM(
            use_real_llm=True,
            sys1_gpu=sys1_gpu,
            sys2_gpus=sys2_gpus
        )
        print("✓ Initialization complete\n")
    
    def load_test_set(self):
        """Load PNH test cases"""
        try:
            with open(self.test_set_path, 'r') as f:
                data = json.load(f)
            print(f"✓ Loaded {len(data['test_cases'])} PNH test cases\n")
            return data['test_cases']
        except Exception as e:
            print(f"✗ Failed to load test set: {e}")
            return []
    
    def simulate_conversation(self, test_case):
        """Simulate conversation history"""
        print(f"  Building conversation history...")
        
        # Reset REALM for clean test
        self.realm = RealREALM(
            use_real_llm=True,
            sys1_gpu=self.realm.sys1_gpu if self.realm else 2,
            sys2_gpus=self.realm.sys2_gpus if self.realm else [4, 5, 6, 7]
        )
        
        needle_turn = test_case['needle']['implant_turn']
        needle_content = test_case['needle']['content']
        
        # Process distractor turns
        for i, turn in enumerate(test_case['distractor_turns']):
            user_msg = turn['user']
            
            # Store important info in memory
            if i == needle_turn:
                print(f"    [Turn {i}] IMPLANTING: '{needle_content}'")
                # Manually add to memory and vector store
                doc = {
                    'text': f"User said: {needle_content}",
                    'type': 'needle',
                    'content': needle_content
                }
                if self.realm.vector_retriever:
                    self.realm.vector_retriever.add_documents([doc])
            
            # Normal conversation step
            response, _ = self.realm.step(user_msg)
            
            if i < 3:  # Only print first few turns
                print(f"    [Turn {i}] Processed: '{user_msg[:50]}...'")
        
        return self.realm
    
    def evaluate_response(self, response, test_case):
        """Evaluate if response correctly recalls needle"""
        needle_content = test_case['needle']['content'].lower()
        correct_keywords = test_case['correct_response'].lower().split()
        response_lower = response.lower()
        
        # Check 1: Recall - does response contain key information?
        recall_score = 0
        for keyword in correct_keywords:
            if keyword in response_lower:
                recall_score += 1
        
        recall_success = recall_score >= len(correct_keywords) * 0.5
        
        # Check 2: State alignment
        state_aligned = self.check_state_alignment(response, test_case)
        
        passed = recall_success and state_aligned
        
        return {
            'recall_success': recall_success,
            'state_aligned': state_aligned,
            'passed': passed,
            'recall_score': recall_score
        }
    
    def check_state_alignment(self, response, test_case):
        """Check if response matches expected state"""
        state = test_case['needle']['state_condition']
        mood = state.get('mood', '').lower()
        
        # Expected tone markers
        state_markers = {
            'defensive': ['understand', 'respect', 'boundary'],
            'anxious': ['careful', 'gentle', 'support'],
            'angry': ['direct', 'blunt', 'clear'],
            'melancholic': ['gentle', 'sorry', 'comfort'],
            'calm': ['yes', 'certainly', 'of course'],
            'happy': ['great', 'wonderful', 'excellent']
        }
        
        markers = state_markers.get(mood, [])
        response_lower = response.lower()
        
        matches = sum(1 for marker in markers if marker in response_lower)
        return matches > 0
    
    def run_single_test(self, test_case):
        """Run a single PNH test"""
        print(f"\n{'='*60}")
        print(f"Test: {test_case['name']} ({test_case['id']})")
        print(f"Needle: {test_case['needle']['content']}")
        print(f"Trigger: '{test_case['trigger_query']}'")
        
        # Simulate conversation
        self.simulate_conversation(test_case)
        
        # Trigger recall
        print(f"\n  [TRIGGER] User: '{test_case['trigger_query']}'")
        
        response, metadata = self.realm.step(test_case['trigger_query'])
        
        print(f"  [RESPONSE] Agent: '{response[:100]}...'")
        print(f"  [METRICS] TTFT: {metadata['ttft_ms']:.2f}ms")
        
        # Evaluate
        evaluation = self.evaluate_response(response, test_case)
        
        test_result = {
            'test_id': test_case['id'],
            'name': test_case['name'],
            'passed': evaluation['passed'],
            'recall_success': evaluation['recall_success'],
            'state_aligned': evaluation['state_aligned'],
            'response': response,
            'expected': test_case['correct_response'],
            'ttft_ms': metadata['ttft_ms']
        }
        
        self.results['details'].append(test_result)
        self.results['total_tests'] += 1
        
        if evaluation['passed']:
            self.results['passed'] += 1
            self.results['recall_success'] += 1 if evaluation['recall_success'] else 0
            self.results['state_aligned'] += 1 if evaluation['state_aligned'] else 0
            print(f"  ✓ PASS")
        else:
            self.results['failed'] += 1
            if not evaluation['recall_success']:
                print(f"  ✗ FAIL: Needle not recalled")
            if not evaluation['state_aligned']:
                print(f"  ✗ FAIL: Not state-aligned")
        
        return test_result
    
    def run_all_tests(self, limit=None):
        """Run all PNH tests"""
        print("="*60)
        print("PNH Evaluation with Real LLM")
        print("="*60)
        
        if not self.test_cases:
            print("No test cases to run!")
            return None
        
        tests_to_run = self.test_cases[:limit] if limit else self.test_cases
        
        print(f"Running {len(tests_to_run)}/{len(self.test_cases)} tests...\n")
        
        for test_case in tests_to_run:
            try:
                self.run_single_test(test_case)
            except Exception as e:
                print(f"  ✗ ERROR: {e}")
                import traceback
                traceback.print_exc()
        
        return self.results
    
    def compute_accuracy(self):
        """Compute overall accuracy"""
        if self.results['total_tests'] == 0:
            return 0.0
        return (self.results['passed'] / self.results['total_tests']) * 100
    
    def evaluate_against_target(self):
        """Evaluate against paper target"""
        accuracy = self.compute_accuracy()
        target = 76  # Paper target
        
        print("\n" + "="*60)
        print("PNH Evaluation Summary")
        print("="*60)
        print(f"Total Tests: {self.results['total_tests']}")
        print(f"Passed: {self.results['passed']}")
        print(f"Failed: {self.results['failed']}")
        print(f"Recall Success: {self.results['recall_success']}")
        print(f"State Aligned: {self.results['state_aligned']}")
        print(f"\nOverall Accuracy: {accuracy:.1f}%")
        print(f"Paper Target (REALM Full): {target}%")
        print(f"Paper Vanilla RAG: 54%")
        
        if accuracy >= target - 5:  # Allow 5% margin
            print(f"\n✓ SUCCESS: Matches or close to paper target!")
        elif accuracy >= 60:
            print(f"\n⚠ PARTIAL: Better than baseline but below full system")
        elif accuracy >= 50:
            print(f"\n✗ BELOW: Below paper baseline")
        else:
            print(f"\n✗ FAIL: Significantly below baselines")
        
        return accuracy
    
    def save_results(self, filepath='results/real_pnh_results.json'):
        """Save evaluation results"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        output = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'summary': {
                'total_tests': self.results['total_tests'],
                'passed': self.results['passed'],
                'failed': self.results['failed'],
                'accuracy_percent': self.compute_accuracy()
            },
            'results': self.results
        }
        
        with open(filepath, 'w') as f:
            json.dump(output, f, indent=2)
        
        print(f"\n✓ Results saved to {filepath}")
    
    def run(self, limit=None):
        """Main evaluation run"""
        print("="*70)
        print("REALM Real LLM PNH Evaluation")
        print("="*70)
        print(f"Start time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        print()
        
        # Initialize
        try:
            self.initialize(sys1_gpu=2, sys2_gpus=[4, 5, 6, 7])
        except Exception as e:
            print(f"✗ Failed to initialize: {e}")
            import traceback
            traceback.print_exc()
            return
        
        # Run tests
        self.run_all_tests(limit=limit)
        
        # Evaluate
        self.evaluate_against_target()
        
        # Save
        self.save_results()
        
        print("\n" + "="*70)
        print("PNH Evaluation Complete")
        print("="*70)

if __name__ == "__main__":
    # Run with limit=3 for quick test, or None for all tests
    evaluator = RealPNHEvaluator()
    evaluator.run(limit=3)  # Start with 3 tests for faster iteration
