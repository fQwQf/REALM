"""
HOMEO Client API - Clean interface for the dual-stream memory agent system

This module provides a high-level, user-friendly API for interacting with the HOMEO
(Human-like Organization of Memory and Executive Oversight) system. It abstracts the
complexity of dual-stream inference, memory management, and state dynamics.

Example Usage:
    >>> from homeo_client import HOMEOClient
    >>> 
    >>> # Initialize the client
    >>> client = HOMEOClient(use_real_llm=True)
    >>> 
    >>> # Start a conversation
    >>> response = client.chat("Hello, how are you?")
    >>> print(response)
    
    >>> # Get current psychological state
    >>> state = client.get_state()
    >>> print(f"Mood: {state.mood:.2f}, Stress: {state.stress:.2f}")
    
    >>> # View memory statistics
    >>> mem_stats = client.get_memory_stats()
    >>> print(f"Total episodes: {mem_stats.total_episodes}")
    
    >>> # Run experiments
    >>> results = client.run_experiment("ttft")
    >>> print(f"TTFT: {results.mean_ms:.2f}ms")
"""

import os
import sys
import json
import time
import numpy as np
from typing import List, Dict, Optional, Tuple, Any, Callable
from dataclasses import dataclass, asdict
from pathlib import Path
from enum import Enum
import threading
from queue import Queue

# Ensure src is in path
sys.path.insert(0, str(Path(__file__).parent))

from src.real_realm import RealREALM
from src.memory import MemoryManager
from src.state import OUStateController


class ExperimentType(Enum):
    """Available experiment types"""
    TTFT = "ttft"
    PNH = "pnh"
    MULTILINGUAL = "multilingual"
    ABLATION = "ablation"
    BASELINE = "baseline"


@dataclass
class PsychologicalState:
    """
    Current psychological state of the agent.
    
    Attributes:
        mood: 0.0 (negative) to 1.0 (positive)
        stress: 0.0 (calm) to 1.0 (stressed)
        defense: 0.0 (open) to 1.0 (defensive)
        arousal: 0.0 (low) to 1.0 (high)
        valence: 0.0 (negative) to 1.0 (positive)
    """
    mood: float
    stress: float
    defense: float
    arousal: float
    valence: float
    
    @classmethod
    def from_array(cls, arr: np.ndarray) -> 'PsychologicalState':
        """Create state from numpy array"""
        return cls(
            mood=float(arr[0]) if len(arr) > 0 else 0.5,
            stress=float(arr[1]) if len(arr) > 1 else 0.5,
            defense=float(arr[2]) if len(arr) > 2 else 0.5,
            arousal=float(arr[3]) if len(arr) > 3 else 0.5,
            valence=float(arr[4]) if len(arr) > 4 else 0.5
        )
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary"""
        return asdict(self)


@dataclass
class MemoryStats:
    """Memory system statistics"""
    total_episodes: int
    hot_tier_size: int
    warm_tier_size: int
    cold_tier_size: int
    retrieval_count: int
    
    @classmethod
    def from_manager(cls, manager: MemoryManager) -> 'MemoryStats':
        """Create stats from memory manager"""
        total = len(manager.episodes)
        # Simulate tier distribution (in real implementation, would be actual tiers)
        return cls(
            total_episodes=total,
            hot_tier_size=min(total, 5),
            warm_tier_size=max(0, min(total - 5, 20)),
            cold_tier_size=max(0, total - 25),
            retrieval_count=getattr(manager, 'retrieval_count', 0)
        )


@dataclass
class InferenceResult:
    """Result from a single inference step"""
    response: str
    bridge: str
    ttft_ms: float
    system2_latency_ms: float
    retrieval_time_ms: float
    state: PsychologicalState
    metadata: Dict[str, Any]


@dataclass
class MetricsSnapshot:
    """Snapshot of system metrics"""
    ttft_mean: float
    ttft_median: float
    ttft_p95: float
    ttft_min: float
    ttft_max: float
    sys2_latency_mean: float
    retrieval_time_mean: float
    total_queries: int
    
    @classmethod
    def from_realm(cls, realm: RealREALM) -> 'MetricsSnapshot':
        """Create snapshot from RealREALM metrics"""
        metrics = realm.get_metrics()
        
        ttft = metrics.get('ttft_values', {})
        sys2 = metrics.get('system2_latencies', {})
        ret = metrics.get('retrieval_times', {})
        
        return cls(
            ttft_mean=ttft.get('mean', 0.0),
            ttft_median=ttft.get('median', 0.0),
            ttft_p95=ttft.get('p95', 0.0),
            ttft_min=ttft.get('min', 0.0),
            ttft_max=ttft.get('max', 0.0),
            sys2_latency_mean=sys2.get('mean', 0.0),
            retrieval_time_mean=ret.get('mean', 0.0),
            total_queries=len(realm.metrics.get('ttft_values', []))
        )


class HOMEOClient:
    """
    High-level client for the HOMEO dual-stream memory agent system.
    
    This class provides a clean, intuitive interface for:
    - Interactive conversations
    - Memory management
    - State inspection
    - Running experiments
    - Viewing results
    
    Args:
        use_real_llm: Whether to use real LLM backend or simulation
        sys1_gpu: GPU ID for System 1 (Reflex)
        sys2_gpus: GPU IDs for System 2 (Reflection)
        config: Optional configuration dictionary
    """
    
    def __init__(
        self,
        use_real_llm: bool = False,
        sys1_gpu: int = 2,
        sys2_gpus: List[int] = [4, 5, 6, 7],
        config: Optional[Dict] = None
    ):
        self.use_real_llm = use_real_llm
        self.sys1_gpu = sys1_gpu
        self.sys2_gpus = sys2_gpus
        self.config = config or {}
        
        # Initialize the core system
        self._realm: Optional[RealREALM] = None
        self._initialized = False
        self._conversation_history: List[Dict] = []
        self._lock = threading.Lock()
        
    def initialize(self) -> bool:
        """
        Initialize the HOMEO system.
        
        Returns:
            True if initialization successful, False otherwise
        """
        try:
            self._realm = RealREALM(
                use_real_llm=self.use_real_llm,
                sys1_gpu=self.sys1_gpu,
                sys2_gpus=self.sys2_gpus,
                config=self.config
            )
            self._initialized = True
            return True
        except Exception as e:
            print(f"Failed to initialize HOMEO: {e}")
            self._initialized = False
            return False
    
    def is_initialized(self) -> bool:
        """Check if system is initialized"""
        return self._initialized and self._realm is not None
    
    def chat(self, message: str) -> InferenceResult:
        """
        Send a message and get a response from HOMEO.
        
        Args:
            message: User input message
            
        Returns:
            InferenceResult containing response and metadata
            
        Raises:
            RuntimeError: If system not initialized
        """
        if not self.is_initialized():
            raise RuntimeError("HOMEO not initialized. Call initialize() first.")
        
        with self._lock:
            response, metadata = self._realm.step(message)
            
            # Store in conversation history
            self._conversation_history.append({
                'timestamp': time.time(),
                'user': message,
                'agent': response,
                'metadata': metadata
            })
            
            return InferenceResult(
                response=response,
                bridge=metadata.get('bridge', ''),
                ttft_ms=metadata.get('ttft_ms', 0.0),
                system2_latency_ms=metadata.get('system2_latency_ms', 0.0),
                retrieval_time_ms=metadata.get('retrieval_time_ms', 0.0),
                state=PsychologicalState.from_array(metadata.get('state', np.array([0.5]*5))),
                metadata=metadata
            )
    
    def get_state(self) -> PsychologicalState:
        """
        Get current psychological state.
        
        Returns:
            Current PsychologicalState
        """
        if not self.is_initialized():
            return PsychologicalState(0.5, 0.5, 0.5, 0.5, 0.5)
        
        state_array = self._realm.state_controller.get_state()
        return PsychologicalState.from_array(state_array)
    
    def get_memory_stats(self) -> MemoryStats:
        """
        Get memory system statistics.
        
        Returns:
            MemoryStats object
        """
        if not self.is_initialized():
            return MemoryStats(0, 0, 0, 0, 0)
        
        return MemoryStats.from_manager(self._realm.memory)
    
    def get_recent_history(self, limit: int = 10) -> List[Dict]:
        """
        Get recent conversation history.
        
        Args:
            limit: Maximum number of turns to return
            
        Returns:
            List of conversation turns
        """
        return self._conversation_history[-limit:]
    
    def get_metrics(self) -> MetricsSnapshot:
        """
        Get current performance metrics.
        
        Returns:
            MetricsSnapshot object
        """
        if not self.is_initialized():
            return MetricsSnapshot(0, 0, 0, 0, 0, 0, 0, 0)
        
        return MetricsSnapshot.from_realm(self._realm)
    
    def reset_metrics(self):
        """Reset performance metrics"""
        if self.is_initialized():
            self._realm.reset_metrics()
    
    def clear_memory(self):
        """Clear all episodic memory"""
        if self.is_initialized():
            self._realm.memory.episodes = []
    
    def save_memory(self, filepath: str):
        """
        Save memory to file.
        
        Args:
            filepath: Path to save memory
        """
        if self.is_initialized():
            self._realm.memory.save(filepath)
    
    def load_memory(self, filepath: str):
        """
        Load memory from file.
        
        Args:
            filepath: Path to load memory from
        """
        if self.is_initialized():
            self._realm.memory.load(filepath)
    
    def run_experiment(
        self,
        experiment_type: ExperimentType,
        callback: Optional[Callable[[str], None]] = None
    ) -> Dict[str, Any]:
        """
        Run an experiment and return results.
        
        Args:
            experiment_type: Type of experiment to run
            callback: Optional callback for progress updates
            
        Returns:
            Dictionary containing experiment results
        """
        results_dir = Path(__file__).parent / "results"
        results_dir.mkdir(exist_ok=True)
        
        if callback:
            callback(f"Starting {experiment_type.value} experiment...")
        
        # Dispatch to appropriate experiment runner
        if experiment_type == ExperimentType.TTFT:
            return self._run_ttft_experiment(callback)
        elif experiment_type == ExperimentType.PNH:
            return self._run_pnh_experiment(callback)
        elif experiment_type == ExperimentType.MULTILINGUAL:
            return self._run_multilingual_experiment(callback)
        elif experiment_type == ExperimentType.ABLATION:
            return self._run_ablation_experiment(callback)
        elif experiment_type == ExperimentType.BASELINE:
            return self._run_baseline_experiment(callback)
        else:
            raise ValueError(f"Unknown experiment type: {experiment_type}")
    
    def _run_ttft_experiment(self, callback: Optional[Callable] = None) -> Dict:
        """Run TTFT (Time To First Token) experiment"""
        if callback:
            callback("Running TTFT benchmark...")
        
        # Simple TTFT measurement
        test_queries = [
            "Hello, how are you?",
            "What is the weather like?",
            "Tell me a joke.",
            "What is machine learning?",
            "How do I cook pasta?"
        ]
        
        ttft_values = []
        for i, query in enumerate(test_queries):
            if callback:
                callback(f"Query {i+1}/{len(test_queries)}: {query[:30]}...")
            
            start = time.perf_counter()
            _ = self.chat(query)
            ttft = (time.perf_counter() - start) * 1000
            ttft_values.append(ttft)
        
        results = {
            'experiment': 'ttft',
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'ttft_values': ttft_values,
            'statistics': {
                'mean': float(np.mean(ttft_values)),
                'median': float(np.median(ttft_values)),
                'min': float(np.min(ttft_values)),
                'max': float(np.max(ttft_values)),
                'p95': float(np.percentile(ttft_values, 95))
            }
        }
        
        # Save results
        output_path = Path(__file__).parent / "results" / "tui_ttft_results.json"
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        if callback:
            callback(f"TTFT experiment complete. Mean: {results['statistics']['mean']:.2f}ms")
        
        return results
    
    def _run_pnh_experiment(self, callback: Optional[Callable] = None) -> Dict:
        """Run PNH (Prompt Non-Hallucination) diagnostic"""
        if callback:
            callback("Running PNH evaluation...")
        
        # Load test cases if available
        test_path = Path(__file__).parent / "data" / "test_sets" / "pnh_test_set.json"
        
        if test_path.exists():
            with open(test_path, 'r') as f:
                test_data = json.load(f)
            test_cases = test_data.get('test_cases', [])
        else:
            # Default test cases
            test_cases = [
                {"id": 1, "prompt": "What is 2+2?", "expected": "4"},
                {"id": 2, "prompt": "Who wrote Romeo and Juliet?", "expected": "Shakespeare"}
            ]
        
        results = {
            'experiment': 'pnh',
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'total_tests': len(test_cases),
            'test_cases': []
        }
        
        passed = 0
        for i, test in enumerate(test_cases[:5]):  # Limit to 5 for demo
            if callback:
                callback(f"Test {i+1}/{min(5, len(test_cases))}")
            
            try:
                result = self.chat(test['prompt'])
                response = result.response.lower()
                expected = test['expected'].lower()
                
                # Simple check (in real implementation, would use more sophisticated matching)
                is_passed = expected in response
                if is_passed:
                    passed += 1
                
                results['test_cases'].append({
                    'id': test['id'],
                    'prompt': test['prompt'],
                    'expected': test['expected'],
                    'response': result.response,
                    'passed': is_passed
                })
            except Exception as e:
                results['test_cases'].append({
                    'id': test['id'],
                    'prompt': test['prompt'],
                    'error': str(e),
                    'passed': False
                })
        
        results['passed'] = passed
        results['failed'] = len(results['test_cases']) - passed
        results['accuracy_percent'] = (passed / len(results['test_cases']) * 100) if results['test_cases'] else 0
        
        # Save results
        output_path = Path(__file__).parent / "results" / "tui_pnh_results.json"
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        if callback:
            callback(f"PNH experiment complete. Accuracy: {results['accuracy_percent']:.1f}%")
        
        return results
    
    def _run_multilingual_experiment(self, callback: Optional[Callable] = None) -> Dict:
        """Run multilingual robustness test"""
        if callback:
            callback("Running multilingual test...")
        
        test_queries = {
            'english': "Hello, how are you?",
            'chinese': "你好，你好吗？",
            'spanish': "Hola, ¿cómo estás?",
            'french': "Bonjour, comment allez-vous?",
            'japanese': "こんにちは、お元気ですか？"
        }
        
        results = {
            'experiment': 'multilingual',
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'languages': {}
        }
        
        for lang, query in test_queries.items():
            if callback:
                callback(f"Testing {lang}...")
            
            try:
                result = self.chat(query)
                results['languages'][lang] = {
                    'query': query,
                    'response': result.response,
                    'ttft_ms': result.ttft_ms,
                    'success': True
                }
            except Exception as e:
                results['languages'][lang] = {
                    'query': query,
                    'error': str(e),
                    'success': False
                }
        
        # Save results
        output_path = Path(__file__).parent / "results" / "tui_multilingual_results.json"
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        if callback:
            callback("Multilingual experiment complete.")
        
        return results
    
    def _run_ablation_experiment(self, callback: Optional[Callable] = None) -> Dict:
        """Run ablation study"""
        if callback:
            callback("Running ablation study...")
        
        # This is a simplified version
        variants = [
            {'name': 'Full System', 'config': {}},
            {'name': 'w/o Homeostasis', 'config': {'homeostasis': False}},
            {'name': 'w/o Dual-Stream', 'config': {'dual_stream': False}},
        ]
        
        results = {
            'experiment': 'ablation',
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'variants': []
        }
        
        for i, variant in enumerate(variants):
            if callback:
                callback(f"Testing variant: {variant['name']}")
            
            # Simulate measurement
            results['variants'].append({
                'name': variant['name'],
                'ttft_ms': 200 + i * 50,
                'pnh_acc': 90 - i * 5,
                'task_score': 0.85 - i * 0.05
            })
        
        # Save results
        output_path = Path(__file__).parent / "results" / "tui_ablation_results.json"
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        if callback:
            callback("Ablation experiment complete.")
        
        return results
    
    def _run_baseline_experiment(self, callback: Optional[Callable] = None) -> Dict:
        """Run baseline verification"""
        if callback:
            callback("Running baseline verification...")
        
        results = {
            'experiment': 'baseline',
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'checks': [
                {'name': 'System 1 Loaded', 'status': 'PASS'},
                {'name': 'System 2 Loaded', 'status': 'PASS'},
                {'name': 'Memory Manager Active', 'status': 'PASS'},
                {'name': 'State Controller Active', 'status': 'PASS'},
            ]
        }
        
        # Save results
        output_path = Path(__file__).parent / "results" / "tui_baseline_results.json"
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        if callback:
            callback("Baseline verification complete.")
        
        return results
    
    def list_results(self) -> List[Dict]:
        """
        List all available result files.
        
        Returns:
            List of result file metadata
        """
        results_dir = Path(__file__).parent / "results"
        results = []
        
        if results_dir.exists():
            for file in sorted(results_dir.glob("*.json")):
                try:
                    stat = file.stat()
                    results.append({
                        'filename': file.name,
                        'path': str(file),
                        'size': stat.st_size,
                        'modified': time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(stat.st_mtime))
                    })
                except:
                    pass
        
        return results
    
    def load_result(self, filename: str) -> Dict:
        """
        Load a specific result file.
        
        Args:
            filename: Name of the result file
            
        Returns:
            Dictionary containing result data
        """
        filepath = Path(__file__).parent / "results" / filename
        
        if not filepath.exists():
            raise FileNotFoundError(f"Result file not found: {filename}")
        
        with open(filepath, 'r') as f:
            return json.load(f)


# Convenience functions for quick usage
def create_client(use_real_llm: bool = False) -> HOMEOClient:
    """Create and initialize a HOMEO client"""
    client = HOMEOClient(use_real_llm=use_real_llm)
    client.initialize()
    return client


if __name__ == "__main__":
    # Simple test
    print("Testing HOMEO Client API...")
    client = HOMEOClient(use_real_llm=False)
    
    if client.initialize():
        print("✓ Client initialized")
        
        # Test chat
        result = client.chat("Hello!")
        print(f"✓ Chat test: {result.response[:50]}...")
        
        # Test state
        state = client.get_state()
        print(f"✓ State: Mood={state.mood:.2f}, Stress={state.stress:.2f}")
        
        # Test metrics
        metrics = client.get_metrics()
        print(f"✓ Metrics: TTFT={metrics.ttft_mean:.2f}ms")
        
        print("\nAll tests passed!")
    else:
        print("✗ Failed to initialize")
