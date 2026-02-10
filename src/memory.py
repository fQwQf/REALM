import json
import time
from typing import List, Dict, Any

class MemoryManager:
    """
    Mocks the hierarchical memory structure of REALM.
    
    In the full paper, this involves:
    - Episodic memory (Hot/Warm/Cold stack)
    - Semantic memory (consolidated facts)
    - Vector retrieval (dense embeddings)
    
    For this simulation (No CUDA), we use a simple list-based episodic buffer
    with keyword-based retrieval.
    """
    
    def __init__(self):
        self.episodes: List[Dict[str, Any]] = []

    def add_episode(self, user_input: str, agent_response: str):
        """
        Adds a new interaction turn to the episodic memory.
        """
        episode = {
            "timestamp": time.time(),
            "user": user_input,
            "agent": agent_response
        }
        self.episodes.append(episode)

    def retrieve(self, query: str, limit: int = 3) -> List[str]:
        """
        Retrieves relevant episodes based on the query.
        
        Mock implementation: Returns episodes containing query keywords,
        or recent episodes if no match found.
        """
        query_lower = query.lower()
        matches = []
        
        # Simple keyword search (reverse order to get recent matches first)
        for ep in reversed(self.episodes):
            content = (ep["user"] + " " + ep["agent"]).lower()
            if any(word in content for word in query_lower.split()):
                matches.append(f"User: {ep['user']} | Agent: {ep['agent']}")
                
        if not matches:
            # Fallback to recent history if no matches
            return self.get_recent_history(limit)
            
        return matches[:limit]

    def get_recent_history(self, limit: int = 5) -> List[str]:
        """
        Returns the most recent N turns as formatted strings.
        """
        recent = self.episodes[-limit:]
        return [f"User: {ep['user']} | Agent: {ep['agent']}" for ep in recent]

    def save(self, filepath: str):
        with open(filepath, 'w') as f:
            json.dump(self.episodes, f, indent=2)

    def load(self, filepath: str):
        try:
            with open(filepath, 'r') as f:
                self.episodes = json.load(f)
        except FileNotFoundError:
            self.episodes = []
