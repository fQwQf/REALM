import sys
import os
import torch
import numpy as np
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.state import OUStateController
from src.memory import MemoryManager

# Try importing vLLM
try:
    from vllm import LLM, SamplingParams
    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False
    print("Warning: vLLM not installed. System 2 will not run.")

class REALMServer:
    def __init__(self, 
                 sys1_model_id="Qwen/Qwen2.5-0.5B-Instruct",
                 sys2_model_id="Qwen/Qwen2.5-7B-Instruct",
                 sys2_tp_size=1,
                 gpu_memory_utilization=0.85):
        
        print("Initializing REALM Server...")
        
        # --- System 1: Reflex (GPU 0) ---
        print(f"Loading System 1 ({sys1_model_id}) on GPU 0...")
        self.sys1_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.sys1_tokenizer = AutoTokenizer.from_pretrained(sys1_model_id)
        self.sys1_model = AutoModelForCausalLM.from_pretrained(
            sys1_model_id, 
            torch_dtype=torch.float16,
            device_map={"": 0} if torch.cuda.is_available() else None
        )
        
        # --- System 2: Reflection (GPU 0+ or 1+) ---
        if VLLM_AVAILABLE:
            print(f"Loading System 2 ({sys2_model_id}) with TP={sys2_tp_size}...")
            # vLLM handles device placement automatically.
            # If TP=1, it uses GPU 0 (sharing with Sys1) or GPU 1 depending on visibility.
            # We set gpu_memory_utilization to allow sharing if needed.
            self.sys2_llm = LLM(
                model=sys2_model_id, 
                tensor_parallel_size=sys2_tp_size,
                gpu_memory_utilization=gpu_memory_utilization,
                trust_remote_code=True
            )
            self.sys2_sampling_params = SamplingParams(temperature=0.7, max_tokens=512)
        else:
            self.sys2_llm = None

        # --- Components ---
        self.state_controller = OUStateController()
        self.memory = MemoryManager()
        
    def step(self, user_input: str):
        # 1. Update State
        event_embedding = np.random.randn(10) # Mock embedding
        current_state = self.state_controller.step(event_embedding)
        
        # 2. System 1: Bridge
        bridge = self.generate_bridge(user_input, current_state)
        print(f"[System 1 Bridge]: {bridge}")
        
        # 3. Retrieval
        context = self.memory.retrieve(user_input)
        
        # 4. System 2: Response
        response = self.generate_response(user_input, current_state, context)
        print(f"[System 2 Response]: {response}")
        
        # 5. Stitching
        final_output = f"{bridge} {response}"
        
        # 6. Memory
        self.memory.add_episode(user_input, final_output)
        
        return final_output

    def generate_bridge(self, user_input, state):
        # Simple prompt for System 1
        messages = [
            {"role": "system", "content": "You are a helpful assistant. Keep your response very short and non-committal. Just acknowledge the user."},
            {"role": "user", "content": user_input}
        ]
        text = self.sys1_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.sys1_tokenizer([text], return_tensors="pt").to(self.sys1_device)
        
        with torch.no_grad():
            generated_ids = self.sys1_model.generate(
                inputs.input_ids,
                max_new_tokens=20,
                do_sample=True,
                temperature=0.3
            )
        
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
        ]
        response = self.sys1_tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return response

    def generate_response(self, user_input, state, context):
        if not self.sys2_llm:
            return "[System 2 Disabled]"
            
        context_str = "\n".join(context)
        prompt = f"""Context:
{context_str}

User: {user_input}

Assistant:"""
        
        outputs = self.sys2_llm.generate([prompt], self.sys2_sampling_params)
        return outputs[0].outputs[0].text

def main():
    parser = argparse.ArgumentParser(description="Run REALM Server")
    parser.add_argument("--sys1-model", type=str, default="Qwen/Qwen2.5-0.5B-Instruct", help="Model ID for System 1")
    parser.add_argument("--sys2-model", type=str, default="Qwen/Qwen2.5-7B-Instruct", help="Model ID for System 2")
    parser.add_argument("--tp-size", type=int, default=1, help="Tensor Parallel size for System 2")
    parser.add_argument("--gpu-util", type=float, default=0.85, help="GPU memory utilization for vLLM")
    
    args = parser.parse_args()
    
    server = REALMServer(
        sys1_model_id=args.sys1_model,
        sys2_model_id=args.sys2_model,
        sys2_tp_size=args.tp_size,
        gpu_memory_utilization=args.gpu_util
    )
    
    print("\n--- Server Ready ---\n")
    while True:
        try:
            user_input = input("User: ")
            if user_input.lower() in ["exit", "quit"]:
                break
            response = server.step(user_input)
            print(f"Agent: {response}\n")
        except KeyboardInterrupt:
            break

if __name__ == "__main__":
    main()
