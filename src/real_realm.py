"""
REALM Real Implementation
Full implementation with real LLM backend and vector retrieval
"""

import os
import sys
import time
import numpy as np
from typing import List, Dict, Optional, Tuple

# Set Hugging Face mirror
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

# Import components
from .state import OUStateController
from .memory import MemoryManager

class RealREALM:
    """
    REALM with real LLM backend and vector retrieval.
    
    GPU Allocation:
    - GPU 2: System 1 (Reflex), Embeddings
    - GPU 4,5,6,7: System 2 (Reflection)
    """
    
    def __init__(
        self,
        use_real_llm: bool = True,
        sys1_gpu: int = 2,
        sys2_gpus: List[int] = [4, 5, 6, 7],
        embedding_device: str = "cuda:2",
        config: Optional[Dict] = None
    ):
        self.use_real_llm = use_real_llm
        self.sys1_gpu = sys1_gpu
        self.sys2_gpus = sys2_gpus
        self.embedding_device = embedding_device
        
        # Configuration
        self.config = config or {
            'dual_stream': True,
            'homeostasis': True,
            'motivated_retrieval': True,
            'accordion_memory': True,
            'parametric_subconscious': True
        }
        
        # Initialize components
        self.state_controller = OUStateController()
        self.memory = MemoryManager()
        
        # LLM Backend
        self.llm_backend = None
        self.vector_retriever = None
        
        # Timing metrics
        self.metrics = {
            'ttft_values': [],
            'system2_latencies': [],
            'retrieval_times': []
        }
        
        if use_real_llm:
            self._init_real_backend()
        
    def _init_real_backend(self):
        """Initialize real LLM backend and vector retriever"""
        try:
            # Import LLM backend
            from .llm_backend import RealLLMBackend
            
            print("[Initializing Real LLM Backend...]")
            self.llm_backend = RealLLMBackend(
                sys1_gpu=self.sys1_gpu,
                sys2_gpus=self.sys2_gpus
            )
            
            # Load models
            self.llm_backend.load_system1()
            self.llm_backend.load_system2()
            
            print("✓ LLM backend ready\n")
            
        except Exception as e:
            print(f"✗ Failed to initialize LLM backend: {e}")
            self.llm_backend = None
        
        try:
            # Import vector retriever
            from .vector_retrieval_v2 import VectorRetriever
            
            print("[Initializing Vector Retriever...]")
            self.vector_retriever = VectorRetriever(
                device=self.embedding_device
            )
            print("✓ Vector retriever ready\n")
            
        except Exception as e:
            print(f"✗ Failed to initialize vector retriever: {e}")
            self.vector_retriever = None
    
    def step(self, user_input: str) -> Tuple[str, Dict]:
        """
        Execute one turn of REALM with real LLM.
        
        Returns:
            (response, metadata) tuple
        """
        metadata = {
            'ttft_ms': 0,
            'system2_latency_ms': 0,
            'retrieval_time_ms': 0,
            'bridge': '',
            'state': None
        }
        
        # 1. Update State
        event_embedding = np.random.randn(10)
        current_state = self.state_controller.step(event_embedding)
        metadata['state'] = current_state.copy()
        
        # 2. System 1: Bridge Generation with Uncertainty-Based Routing
        start_time = time.perf_counter()
        
        # Entropy threshold for System 2 trigger (configurable, default 1.5)
        tau_H = self.config.get('entropy_threshold', 1.5)
        
        if self.use_real_llm and self.llm_backend:
            try:
                # Generate System 1 bridge with entropy tracking
                result = self.llm_backend.generate_system1(
                    user_input,
                    state_vector=current_state,
                    return_entropy=True
                )
                
                bridge = result['response']
                entropy_info = result['entropy_info']
                avg_entropy = entropy_info['avg_first_3']
                
                # Log entropy for debugging
                print(f"[System 1] Bridge: '{bridge}' | Avg Entropy: {avg_entropy:.3f} (threshold: {tau_H})")
                
            except Exception as e:
                print(f"System 1 error: {e}, using fallback")
                bridge = self._fallback_bridge(user_input, current_state)
                avg_entropy = 0.0  # Low entropy for fallback
                entropy_info = {'avg_first_3': 0.0, 'max': 0.0, 'all_entropies': []}
        else:
            bridge = self._fallback_bridge(user_input, current_state)
            avg_entropy = 0.0
            entropy_info = {'avg_first_3': 0.0, 'max': 0.0, 'all_entropies': []}
        
        ttft_ms = (time.perf_counter() - start_time) * 1000
        metadata['ttft_ms'] = ttft_ms
        metadata['bridge'] = bridge
        metadata['entropy'] = entropy_info
        self.metrics['ttft_values'].append(ttft_ms)
        
        # 3. Uncertainty-Based Routing: Decide whether to trigger System 2
        # If entropy is low (< tau_H), System 1 is confident -> skip System 2
        # If entropy is high (>= tau_H), System 1 is uncertain -> trigger System 2
        system2_triggered = avg_entropy >= tau_H
        
        if system2_triggered:
            print(f"[Routing] High entropy ({avg_entropy:.3f} >= {tau_H}), triggering System 2...")
        else:
            print(f"[Routing] Low entropy ({avg_entropy:.3f} < {tau_H}), using System 1 only")
        
        # 4. State-conditioned Retrieval (only if System 2 will be triggered)
        if system2_triggered:
            retrieval_start = time.perf_counter()
            
            if self.config.get('motivated_retrieval', True) and self.vector_retriever:
                retrieved_docs = self.vector_retriever.search(
                    query=user_input,
                    top_k=3,
                    state_vector=current_state if self.config.get('motivated_retrieval') else None
                )
                context = [doc['text'] for doc in retrieved_docs]
            else:
                context = self.memory.retrieve(user_input)
            
            retrieval_time = (time.perf_counter() - retrieval_start) * 1000
            metadata['retrieval_time_ms'] = retrieval_time
            self.metrics['retrieval_times'].append(retrieval_time)
            
            # 5. System 2: Response Generation
            sys2_start = time.perf_counter()
            
            if self.use_real_llm and self.llm_backend:
                try:
                    response = self.llm_backend.generate_system2(
                        user_input,
                        context=context,
                        state_vector=current_state if self.config.get('homeostasis') else None
                    )
                except Exception as e:
                    print(f"System 2 error: {e}, using fallback")
                    response = self._fallback_response(user_input, context)
            else:
                response = self._fallback_response(user_input, context)
            
            sys2_latency = (time.perf_counter() - sys2_start) * 1000
            metadata['system2_latency_ms'] = sys2_latency
            self.metrics['system2_latencies'].append(sys2_latency)
            
            # 6. Conflict Check & Stitching
            final_output = self._conflict_check(bridge, response)
        else:
            # Single-stream mode: use bridge as final output
            metadata['retrieval_time_ms'] = 0
            metadata['system2_latency_ms'] = 0
            final_output = bridge
        
        # 6. Store to memory
        self.memory.add_episode(user_input, final_output)
        
        # Also store to vector index if available
        if self.vector_retriever:
            doc = {
                'text': f"User: {user_input}\nAgent: {final_output}",
                'user_input': user_input,
                'agent_response': final_output,
                'timestamp': time.time()
            }
            self.vector_retriever.add_documents([doc])
        
        return final_output, metadata
    
    def _fallback_bridge(self, user_input: str, state: np.ndarray) -> str:
        """Fallback bridge generation without LLM"""
        mood = state[0]
        if mood > 0.7:
            return "Got it, let me check..."
        elif mood < 0.3:
            return "Hmm, let me see..."
        else:
            return "One moment..."
    
    def _fallback_response(self, user_input: str, context: list) -> str:
        """Fallback response generation without LLM"""
        context_str = " | ".join(context) if context else "No context"
        return f"[Response to '{user_input}' with context: {context_str[:50]}...]"
    
    def _conflict_check(self, bridge: str, response: str) -> str:
        """Check for conflicts and stitch response"""
        # Improved conflict detector as described in paper
        
        # 1. Extract leading sentiment/stance from System 2
        # Heuristic: check first sentence of response
        s2_sentences = response.split('. ')
        s2_lead = s2_sentences[0] if s2_sentences else response
        
        # 2. Check for contradictions
        conflict_detected = False
        
        # Bridge says "Yes" but S2 says "No/Avoid/Don't"
        if ("Of course" in bridge or "Yes" in bridge) and ("avoid" in s2_lead.lower() or "not" in s2_lead.lower() or "no" in s2_lead.lower()):
            conflict_detected = True
            repair_type = "contradiction_major"
        
        # Bridge says "I don't know/sorry" but S2 provides fact
        elif "sorry" in bridge.lower() and ("You have" in s2_lead or "Your" in s2_lead or "You obtained" in s2_lead):
            conflict_detected = True
            repair_type = "false_refusal"
            
        if conflict_detected:
            # Action: Repair
            repair_clause = " (Correction: let me be more precise) "
            return f"{bridge}{repair_clause}{response}"
        
        # No conflict: simple stitch
        return f"{bridge} {response}"
    
    def get_metrics(self) -> Dict:
        """Get timing metrics"""
        import statistics
        
        metrics = {}
        for key, values in self.metrics.items():
            if values:
                metrics[key] = {
                    'mean': statistics.mean(values),
                    'median': statistics.median(values),
                    'min': min(values),
                    'max': max(values),
                    'p95': sorted(values)[int(len(values) * 0.95)] if len(values) > 1 else values[0]
                }
        
        return metrics
    
    def reset_metrics(self):
        """Reset timing metrics"""
        self.metrics = {
            'ttft_values': [],
            'system2_latencies': [],
            'retrieval_times': []
        }
