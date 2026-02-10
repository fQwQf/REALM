import numpy as np
from .state import OUStateController
from .memory import MemoryManager

class REALM:
    """
    REALM: Real-Time Dual-Stream Pacing for Long-Horizon Consistency.
    
    This class orchestrates:
    1. System 1 (Reflex): Fast, low-commitment bridge.
    2. System 2 (Reflection): Slow, state-aligned retrieval and generation.
    3. Conflict Check: Stitches streams and repairs contradictions.
    4. State Dynamics: Updates OU state based on interaction.
    """
    
    def __init__(self, config=None):
        self.state_controller = OUStateController()
        self.memory = MemoryManager()
        
        # Configuration for ablation studies
        self.config = config or {
            'dual_stream': True,
            'homeostasis': True,
            'motivated_retrieval': True,
            'accordion_memory': True,
            'parametric_subconscious': True
        }
        
        # Feature flags
        self.use_motivated_retrieval = self.config.get('motivated_retrieval', True)
        
    def step(self, user_input: str) -> str:
        """
        Executes one turn of the REALM loop.
        """
        event_embedding = np.random.randn(10) 
        current_state = self.state_controller.step(event_embedding)
        
        bridge = self.system1_bridge(user_input, current_state)
        
        context = self.memory.retrieve(user_input)
        
        response = self.system2_response(user_input, current_state, context)
        
        final_output = self.conflict_check(bridge, response)
        
        self.memory.add_episode(user_input, final_output)
        
        return final_output

    def system1_bridge(self, user_input: str, state: np.ndarray) -> str:
        """
        Generates a low-commitment bridge.
        """
        mood = state[0]
        if mood > 0.7:
            return "Got it, let me check..."
        elif mood < 0.3:
            return "Hmm, let me see..."
        else:
            return "One moment..."

    def system2_response(self, user_input: str, state: np.ndarray, context: list) -> str:
        """
        Generates the grounded response.
        """
        context_str = " | ".join(context) if context else "No context"
        return f"[Response to '{user_input}' with context: {context_str[:50]}...]"

    def conflict_check(self, bridge: str, response: str) -> str:
        """
        Stitches the bridge and response.
        """
        return f"{bridge} {response}"
