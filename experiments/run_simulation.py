import sys
import os

# Add the project root to the python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.realm import REALM

def run_simulation():
    print("Initializing REALM Agent...")
    agent = REALM()
    
    user_inputs = [
        "Hello, who are you?",
        "I'm feeling a bit stressed today.",
        "Can you help me organize my schedule?",
        "Wait, did you promise to keep my data private?",
        "Thanks for the help."
    ]
    
    print("\n--- Starting Simulation ---\n")
    
    for i, user_input in enumerate(user_inputs):
        print(f"Turn {i+1}:")
        print(f"User: {user_input}")
        
        response = agent.step(user_input)
        
        print(f"Agent: {response}")
        print("-" * 30)
        
    print("\n--- Simulation Complete ---")
    print("Final State Vector:")
    print(agent.state_controller.get_state())

if __name__ == "__main__":
    run_simulation()
