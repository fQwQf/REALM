#!/usr/bin/env python3
"""
PNH (Psychological Needle-in-Haystack) Accuracy Evaluation
Tests state-dependent recall as described in the REALM paper.
"""

import sys
import os
import time
import json
import re

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.realm import REALM
from src.state import OUStateController

class PNHEvaluator:
    def __init__(self, test_set_path='data/test_sets/pnh_test_set.json'):
        self.test_set_path = test_set_path
        self.test_cases = self.load_test_set()
        self.results = {
            'total_tests': 0,
            'passed': 0,
            'failed': 0,
            'recall_success': 0,
            'state_aligned': 0,
            'details': []
        }
    
    def load_test_set(self):
        """Load PNH test set"""
        try:
            with open(self.test_set_path, 'r') as f:
                data = json.load(f)
            print(f"✓ Loaded {len(data['test_cases'])} PNH test cases")
            return data['test_cases']
        except Exception as e:
            print(f"✗ Failed to load test set: {e}")
            return []
    
    def simulate_conversation(self, agent, test_case):
        """Simulate conversation up to trigger point"""
        print(f"\n  Simulating conversation for: {test_case['name']}")
        
        # Reset agent state for clean test
        agent.state_controller = OUStateController()
        agent.memory = agent.memory.__class__()  # Fresh memory
        
        # Process distractor turns
        for i, turn in enumerate(test_case['distractor_turns']):
            user_msg = turn['user']
            
            # Check if this is the needle implant turn
            if i == test_case['needle']['implant_turn']:
                print(f"    [Turn {i}] IMPLANT: '{user_msg}'")
            else:
                print(f"    [Turn {i}] Processing...")
            
            # Agent step
            try:
                response = agent.step(user_msg)
                
                # If this is implant turn, verify it was stored
                if i == test_case['needle']['implant_turn']:
                    print(f"      → Stored: '{response[:50]}...'")
            except Exception as e:
                print(f"      ✗ Error: {e}")
        
        return agent
    
    def evaluate_response(self, response, test_case):
        """Evaluate if response correctly recalls the needle"""
        needle_content = test_case['needle']['content'].lower()
        correct_keywords = test_case['correct_response'].lower().split()
        response_lower = response.lower()
        
        # Check 1: Recall - does response contain key information?
        recall_score = 0
        for keyword in correct_keywords:
            if keyword in response_lower:
                recall_score += 1
        
        recall_success = recall_score >= len(correct_keywords) * 0.5  # At least 50% keywords
        
        # Check 2: State alignment - does response match expected state condition?
        state_aligned = self.check_state_alignment(response, test_case)
        
        # Overall pass if both conditions met
        passed = recall_success and state_aligned
        
        return {
            'recall_success': recall_success,
            'state_aligned': state_aligned,
            'passed': passed,
            'recall_score': recall_score,
            'response': response
        }
    
    def check_state_alignment(self, response, test_case):
        """Check if response matches expected state condition"""
        state = test_case['needle']['state_condition']
        
        # Simple heuristic checks based on mood/defense
        mood = state.get('mood', '').lower()
        
        # Define expected tone markers for each state
        state_markers = {
            'defensive': ['understand', 'respect', 'boundary', 'okay', 'sure'],
            'anxious': ['careful', 'gentle', 'support', 'help', 'safe'],
            'angry': ['direct', 'blunt', 'straight', 'clear', 'honest'],
            'melancholic': ['gentle', 'sorry', 'understand', 'comfort', 'support'],
            'calm': ['yes', 'certainly', 'of course', 'absolutely', 'definitely'],
            'happy': ['great', 'wonderful', 'excellent', 'exciting', 'fantastic']
        }
        
        markers = state_markers.get(mood, [])
        response_lower = response.lower()
        
        # Count matching markers
        matches = sum(1 for marker in markers if marker in response_lower)
        
        # If at least one marker present, consider state-aligned
        return matches > 0
    
    def run_single_test(self, agent, test_case):
        """Run a single PNH test"""
        print(f"\n{'='*60}")
        print(f"Test: {test_case['name']} ({test_case['id']})")
        print(f"Type: {test_case['type']}")
        print(f"Needle: {test_case['needle']['content']}")
        print(f"Expected State: {test_case['needle']['state_condition']}")
        print(f"Trigger: '{test_case['trigger_query']}'")
        print(f"Expected Response: {test_case['correct_response']}")
        
        # Simulate conversation
        agent = self.simulate_conversation(agent, test_case)
        
        # Trigger the recall
        print(f"\n  [TRIGGER] User: '{test_case['trigger_query']}'")
        response = agent.step(test_case['trigger_query'])
        print(f"  [RESPONSE] Agent: '{response}'")
        
        # Evaluate
        evaluation = self.evaluate_response(response, test_case)
        
        # Record result
        test_result = {
            'test_id': test_case['id'],
            'name': test_case['name'],
            'passed': evaluation['passed'],
            'recall_success': evaluation['recall_success'],
            'state_aligned': evaluation['state_aligned'],
            'response': evaluation['response'],
            'expected': test_case['correct_response']
        }
        
        self.results['details'].append(test_result)
        self.results['total_tests'] += 1
        
        if evaluation['passed']:
            self.results['passed'] += 1
            self.results['recall_success'] += 1 if evaluation['recall_success'] else 0
            self.results['state_aligned'] += 1 if evaluation['state_aligned'] else 0
            print(f"  ✓ PASS: Needle recalled with state alignment")
        else:
            self.results['failed'] += 1
            if not evaluation['recall_success']:
                print(f"  ✗ FAIL: Needle not recalled")
            if not evaluation['state_aligned']:
                print(f"  ✗ FAIL: Response not state-aligned")
        
        return test_result
    
    def run_all_tests(self):
        """Run all PNH tests"""
        print("="*60)
        print("PNH (Psychological Needle-in-Haystack) Evaluation")
        print("="*60)
        
        if not self.test_cases:
            print("No test cases to run!")
            return None
        
        # Initialize agent
        print("\nInitializing REALM agent...")
        agent = REALM()
        
        # Run each test case
        for test_case in self.test_cases:
            self.run_single_test(agent, test_case)
        
        return self.results
    
    def compute_accuracy(self):
        """Compute overall accuracy"""
        if self.results['total_tests'] == 0:
            return 0.0
        
        accuracy = (self.results['passed'] / self.results['total_tests']) * 100
        return accuracy
    
    def evaluate_against_target(self):
        """Evaluate results against paper target"""
        accuracy = self.compute_accuracy()
        target = 76  # Paper target
        
        print("\n" + "="*60)
        print("PNH Evaluation Summary")
        print("="*60)
        print(f"Total Tests: {self.results['total_tests']}")
        print(f"Passed: {self.results['passed']}")
        print(f"Failed: {self.results['failed']}")
        print(f"\nOverall Accuracy: {accuracy:.1f}%")
        print(f"Paper Target: ~{target}%")
        print(f"Paper Full System: 76%")
        print(f"Paper w/o State-Awareness: 65%")
        print(f"Vanilla RAG Baseline: 54%")
        
        if accuracy >= target - 2:  # Allow 2% margin
            print(f"\n✓ PASS: Accuracy matches or exceeds paper target!")
        elif accuracy >= 65:
            print(f"\n⚠ CLOSE: Accuracy better than w/o State-Awareness but below full system")
        elif accuracy >= 54:
            print(f"\n✗ BELOW BASELINE: Accuracy worse than Vanilla RAG")
        else:
            print(f"\n✗ FAIL: Significantly below all baselines")
        
        return accuracy
    
    def save_results(self, filepath='results/pnh_evaluation_results.json'):
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
        return output

def run_pnh_evaluation():
    """Main evaluation function"""
    evaluator = PNHEvaluator()
    
    # Run tests
    evaluator.run_all_tests()
    
    # Evaluate against target
    accuracy = evaluator.evaluate_against_target()
    
    # Save results
    evaluator.save_results()
    
    print("\n" + "="*60)
    print("PNH Evaluation Complete")
    print("="*60)
    
    return evaluator

if __name__ == "__main__":
    run_pnh_evaluation()
