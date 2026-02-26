#!/usr/bin/env python3
"""
REALM Real LLM Backend
Implements real LLM inference using transformers with GPU allocation.
Uses Qwen2.5 models for System 1 (0.5B) and System 2 (7B).
"""

import os
import sys
import time
import torch
import re
from typing import List, Dict, Optional, Union

# Set Hugging Face mirror for China
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['HF_HOME'] = '/data1/tongjizhou/.cache/huggingface'

# GPU Allocation Strategy:
# GPU 2: System 1 (Reflex) - Qwen2.5-0.5B-Instruct
# GPU 4,5,6,7: System 2 (Reflection) - Qwen2.5-7B-Instruct with device_map

class RealLLMBackend:
    """Real LLM backend using transformers"""
    
    def __init__(
        self,
        sys1_model_id: str = "Qwen/Qwen2.5-0.5B-Instruct",
        sys2_model_id: str = "Qwen/Qwen2.5-7B-Instruct",
        sys1_gpu: int = 2,
        sys2_gpus: List[int] = [4, 5, 6, 7],
        load_in_8bit: bool = False
    ):
        self.sys1_model_id = sys1_model_id
        self.sys2_model_id = sys2_model_id
        self.sys1_gpu = sys1_gpu
        self.sys2_gpus = sys2_gpus
        self.load_in_8bit = load_in_8bit
        
        self.sys1_model = None
        self.sys1_tokenizer = None
        self.sys2_model = None
        self.sys2_tokenizer = None
        
        print(f"Initializing Real LLM Backend...")
        print(f"System 1: {sys1_model_id} on GPU {sys1_gpu}")
        print(f"System 2: {sys2_model_id} on GPUs {sys2_gpus}")
        
    def load_system1(self):
        """Load System 1 (Reflex) - Fast, small model"""
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        print(f"\n[Loading System 1 on GPU {self.sys1_gpu}...]")
        start_time = time.time()
        
        try:
            self.sys1_tokenizer = AutoTokenizer.from_pretrained(
                self.sys1_model_id,
                trust_remote_code=True
            )
            
            # Use specific GPU for System 1
            device_map = {"": self.sys1_gpu}
            
            self.sys1_model = AutoModelForCausalLM.from_pretrained(
                self.sys1_model_id,
                torch_dtype=torch.float16,
                device_map=device_map,
                trust_remote_code=True
            )
            
            self.sys1_model.eval()
            
            load_time = time.time() - start_time
            print(f"✓ System 1 loaded in {load_time:.2f}s")
            
        except Exception as e:
            print(f"✗ Failed to load System 1: {e}")
            import traceback
            traceback.print_exc()
            
    def load_system2(self):
        """Load System 2 (Reflection) - Larger model with multi-GPU support"""
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        print(f"\n[Loading System 2 on GPUs {self.sys2_gpus}...]")
        start_time = time.time()
        
        try:
            self.sys2_tokenizer = AutoTokenizer.from_pretrained(
                self.sys2_model_id,
                trust_remote_code=True
            )
            
            # Multi-GPU setup for System 2
            if len(self.sys2_gpus) > 1:
                # Use device_map='auto' for automatic distribution
                # or specify exact mapping
                device_map = 'auto'
            else:
                device_map = {"": self.sys2_gpus[0]}
            
            load_kwargs = {
                "torch_dtype": torch.float16,
                "device_map": device_map,
                "trust_remote_code": True
            }
            
            if self.load_in_8bit:
                load_kwargs["load_in_8bit"] = True
            
            self.sys2_model = AutoModelForCausalLM.from_pretrained(
                self.sys2_model_id,
                **load_kwargs
            )
            
            self.sys2_model.eval()
            
            load_time = time.time() - start_time
            print(f"✓ System 2 loaded in {load_time:.2f}s")
            
        except Exception as e:
            print(f"✗ Failed to load System 2: {e}")
            import traceback
            traceback.print_exc()

    def generate_system1(
        self,
        user_input: str,
        state_vector: Optional[List[float]] = None,
        max_new_tokens: int = 30,
        temperature: float = 1.0,
        return_entropy: bool = False,
        return_query_type: bool = False
    ) -> Union[str, Dict]:
        """Generate bridge using System 1
        
        Args:
            user_input: User's input text
            state_vector: Current psychological state
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            return_entropy: If True, returns dict with 'response' and 'entropy_info'
            return_query_type: If True, also returns query type classification
        
        Returns:
            str or dict: Generated response, or dict with metadata
        """
        if self.sys1_model is None or self.sys1_tokenizer is None:
            raise RuntimeError("System 1 not loaded")
        
        # Determine mood from state
        mood = "neutral"
        if state_vector is not None:
            mood_val = state_vector[0]
            if mood_val > 0.7:
                mood = "positive"
            elif mood_val < 0.3:
                mood = "concerned"
        
        # Improved prompt: Guide System 1 to generate meaningful bridges AND classify query
        if return_query_type:
            system_prompt = f"""You are a helpful assistant. The current mood is {mood}.

Your task is to:
1. Classify the user's query into ONE of these types:
   - FACTUAL: Asking for specific information (names, dates, facts, preferences)
   - GREETING: Simple hello/goodbye
   - SHARING: User sharing information about themselves
   - OPINION: Asking for opinions or advice
   - OTHER: Anything else

2. Provide a SHORT bridge response (3-6 words)

CRITICAL: Output in this exact format:
TYPE: <classification>
BRIDGE: <your response>

Examples:
User: "What's my name?"
TYPE: FACTUAL
BRIDGE: Let me recall your name...

User: "Hello!"
TYPE: GREETING
BRIDGE: Hello! How can I help?

User: "I love hiking."
TYPE: SHARING
BRIDGE: Got it, noted. Thanks!"""
        else:
            system_prompt = f"""You are a helpful assistant. The current mood is {mood}.

Your task is to provide an IMMEDIATE response (bridge) that:
1. Acknowledges the user's input
2. Shows you understand what they're asking
3. Buys time for deeper processing if needed

CRITICAL: Do not provide specific facts. Instead:
- If they ask for information: "Let me recall that...", "Checking what you mentioned..."
- If they share something: "Got it, noted.", "Thanks for sharing."
- If it's a greeting: "Hello!", "Hi there!"

Keep responses SHORT (3-6 words) and NATURAL."""
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_input}
        ]
        
        text = self.sys1_tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        inputs = self.sys1_tokenizer([text], return_tensors="pt").to(f"cuda:{self.sys1_gpu}")
        
        with torch.no_grad():
            if return_entropy or return_query_type:
                # Generate with scores for entropy calculation
                outputs = self.sys1_model.generate(
                    inputs.input_ids,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=temperature,
                    pad_token_id=self.sys1_tokenizer.eos_token_id,
                    return_dict_in_generate=True,
                    output_scores=True
                )
                
                # Calculate entropy for each generated token
                scores = outputs.scores
                entropies = []
                
                for score in scores:
                    # Apply temperature scaling to logits BEFORE softmax
                    scaled_logits = score[0] / temperature
                    probs = torch.softmax(scaled_logits, dim=-1)
                    # Calculate entropy: -sum(p * log(p))
                    entropy = -torch.sum(probs * torch.log(probs + 1e-10))
                    entropies.append(entropy.item())
                
                # Average entropy over first 3 tokens (as per paper)
                avg_entropy_first_3 = sum(entropies[:3]) / min(3, len(entropies)) if entropies else 0.0
                max_entropy = max(entropies) if entropies else 0.0
                
                # Decode response
                generated_ids = outputs.sequences[0][inputs.input_ids.shape[1]:]
                response = self.sys1_tokenizer.decode(generated_ids, skip_special_tokens=True)
                response = response.strip()
                
                result = {
                    'response': response,
                    'entropy_info': {
                        'avg_first_3': avg_entropy_first_3,
                        'max': max_entropy,
                        'all_entropies': entropies
                    }
                }
                
                # Parse query type if requested
                if return_query_type:
                    query_type = "OTHER"
                    bridge = response
                    
                    # Parse TYPE and BRIDGE from response
                    type_match = re.search(r'TYPE:\s*(\w+)', response, re.IGNORECASE)
                    bridge_match = re.search(r'BRIDGE:\s*(.+?)(?:\n|$)', response, re.IGNORECASE)
                    
                    if type_match:
                        query_type = type_match.group(1).upper()
                    if bridge_match:
                        bridge = bridge_match.group(1).strip()
                    
                    result['query_type'] = query_type
                    result['bridge'] = bridge
                    # Update response to be just the bridge
                    result['response'] = bridge
                
                return result
            else:
                # Standard generation without entropy
                outputs = self.sys1_model.generate(
                    inputs.input_ids,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=temperature,
                    pad_token_id=self.sys1_tokenizer.eos_token_id
                )
                
                # Decode only the generated part
                generated_ids = outputs[0][inputs.input_ids.shape[1]:]
                response = self.sys1_tokenizer.decode(generated_ids, skip_special_tokens=True)
                
                return response.strip()
    
    def generate_system2(
        self,
        user_input: str,
        context: List[str],
        state_vector: Optional[List[float]] = None,
        max_new_tokens: int = 256,
        temperature: float = 0.7
    ) -> str:
        """Generate response using System 2 with retrieval context"""
        if self.sys2_model is None or self.sys2_tokenizer is None:
            raise RuntimeError("System 2 not loaded")
        
        # Construct state-aware prompt with context
        context_str = "\n".join(context) if context else "No relevant context found."
        
        # Extract state info
        state_desc = ""
        if state_vector is not None:
            mood_val = state_vector[0]
            if mood_val > 0.7:
                state_desc = "The user seems to be in a good mood."
            elif mood_val < 0.3:
                state_desc = "The user seems stressed or concerned."
        
        system_prompt = f"""You are a helpful, consistent assistant.
{state_desc}
Use the provided context to answer the user's question.
If the context contains relevant information, incorporate it naturally.
If the context doesn't have the answer, you may use your general knowledge.
Keep your response concise but informative."""
{state_desc}
Use the provided context to give accurate, relevant responses.
Keep your response concise but informative."""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Context from previous conversation:\n{context_str}\n\nUser: {user_input}"}
        ]
        
        text = self.sys2_tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        inputs = self.sys2_tokenizer([text], return_tensors="pt")
        
        # Move to appropriate device (first GPU in the list)
        device = f"cuda:{self.sys2_gpus[0]}"
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.sys2_model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                pad_token_id=self.sys2_tokenizer.eos_token_id
            )
        
        # Decode only the generated part
        generated_ids = outputs[0][inputs['input_ids'].shape[1]:]
        response = self.sys2_tokenizer.decode(generated_ids, skip_special_tokens=True)
        
        return response.strip()

def test_llm_backend():
    """Test the real LLM backend"""
    print("="*60)
    print("Testing Real LLM Backend")
    print("="*60)
    
    # Initialize
    backend = RealLLMBackend(
        sys1_gpu=2,
        sys2_gpus=[4, 5, 6, 7]
    )
    
    # Load models
    backend.load_system1()
    backend.load_system2()
    
    print("\n" + "="*60)
    print("Generating test responses...")
    print("="*60)
    
    # Test System 1
    test_input = "Hello, can you help me?"
    print(f"\n[System 1 Test]")
    print(f"Input: {test_input}")
    
    start = time.time()
    bridge = backend.generate_system1(test_input)
    end = time.time()
    
    print(f"Bridge: {bridge}")
    print(f"TTFT: {(end-start)*1000:.2f}ms")
    
    # Test System 2
    test_input2 = "What did we discuss earlier?"
    context = ["User mentioned they like jazz music", "User asked about Python programming"]
    
    print(f"\n[System 2 Test]")
    print(f"Input: {test_input2}")
    print(f"Context: {context}")
    
    start = time.time()
    response = backend.generate_system2(test_input2, context)
    end = time.time()
    
    print(f"Response: {response}")
    print(f"Latency: {(end-start)*1000:.2f}ms")
    
    print("\n" + "="*60)
    print("Test complete!")
    print("="*60)

if __name__ == "__main__":
    test_llm_backend()
