"""
Tests for TEMPO TUI Application

Run with: python -m pytest tests/test_tui.py -v
"""

import sys
import pytest
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from tempo_client import TEMPOClient, ExperimentType, PsychologicalState, MemoryStats, InferenceResult


class TestTEMPOClient:
    """Test TEMPO Client API"""
    
    @pytest.fixture
    def client(self):
        """Create initialized client fixture with test mode"""
        client = TEMPOClient(sys1_gpu=2, sys2_gpus=[4,5,6,7], test_mode=True)
        client.initialize()
        return client
    
    def test_client_initialization(self):
        """Test client can be initialized with test mode"""
        client = TEMPOClient(sys1_gpu=2, sys2_gpus=[4,5,6,7], test_mode=True)
        assert not client.is_initialized()
        
        success = client.initialize()
        assert success
        assert client.is_initialized()
    
    def test_chat(self, client):
        """Test chat functionality"""
        result = client.chat("Hello!")
        
        assert isinstance(result, InferenceResult)
        assert isinstance(result.response, str)
        assert len(result.response) > 0
        assert result.ttft_ms >= 0
        assert result.system2_latency_ms >= 0
    
    def test_get_state(self, client):
        """Test state retrieval"""
        state = client.get_state()
        
        assert isinstance(state, PsychologicalState)
        assert 0.0 <= state.mood <= 1.0
        assert 0.0 <= state.stress <= 1.0
        assert 0.0 <= state.defense <= 1.0
    
    def test_get_memory_stats(self, client):
        """Test memory statistics"""
        stats = client.get_memory_stats()
        
        assert isinstance(stats, MemoryStats)
        assert stats.total_episodes >= 0
        assert stats.hot_tier_size >= 0
    
    def test_get_metrics(self, client):
        """Test metrics retrieval"""
        # First make a query to generate metrics
        client.chat("Test query")
        
        metrics = client.get_metrics()
        assert metrics.total_queries >= 1
        assert metrics.ttft_mean >= 0
    
    def test_conversation_history(self, client):
        """Test conversation history tracking"""
        initial_history = client.get_recent_history()
        
        client.chat("Message 1")
        client.chat("Message 2")
        
        history = client.get_recent_history()
        assert len(history) >= 2
    
    def test_memory_clear(self, client):
        """Test memory clearing"""
        client.chat("Test message")
        initial_count = client.get_memory_stats().total_episodes
        
        client.clear_memory()
        
        assert client.get_memory_stats().total_episodes == 0
    
    def test_reset_metrics(self, client):
        """Test metrics reset"""
        client.chat("Test")
        assert client.get_metrics().total_queries >= 1
        
        client.reset_metrics()
        assert client.get_metrics().total_queries == 0
    
    def test_list_results(self, client):
        """Test listing result files"""
        results = client.list_results()
        assert isinstance(results, list)
    
    def test_experiment_type_enum(self):
        """Test experiment type enum"""
        assert ExperimentType.TTFT.value == "ttft"
        assert ExperimentType.PNH.value == "pnh"
        assert ExperimentType.MULTILINGUAL.value == "multilingual"
        assert ExperimentType.ABLATION.value == "ablation"
        assert ExperimentType.BASELINE.value == "baseline"


class TestPsychologicalState:
    """Test PsychologicalState dataclass"""
    
    def test_from_array(self):
        """Test creating state from array"""
        import numpy as np
        arr = np.array([0.7, 0.3, 0.5, 0.6, 0.4])
        state = PsychologicalState.from_array(arr)
        
        assert state.mood == 0.7
        assert state.stress == 0.3
        assert state.defense == 0.5
        assert state.arousal == 0.6
        assert state.valence == 0.4
    
    def test_to_dict(self):
        """Test converting to dictionary"""
        state = PsychologicalState(0.5, 0.5, 0.5, 0.5, 0.5)
        d = state.to_dict()
        
        assert d['mood'] == 0.5
        assert d['stress'] == 0.5
        assert 'defense' in d


class TestTUIImports:
    """Test TUI can be imported"""
    
    def test_tui_import(self):
        """Test TUI module can be imported"""
        try:
            from tui.tempo_tui import TEMPOApp, DashboardScreen, ChatScreen
            assert True
        except ImportError as e:
            pytest.fail(f"Failed to import TUI: {e}")
    
    def test_tui_app_creation(self):
        """Test TUI app can be created"""
        from tui.tempo_tui import TEMPOApp
        
        # Note: We don't run the app, just create it
        app = TEMPOApp()
        assert app is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
