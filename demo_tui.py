#!/usr/bin/env python3
"""
Quick demo of TEMPO TUI functionality

This script demonstrates the API without launching the full TUI.
Run with: python demo_tui.py
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from tempo_client import TEMPOClient, ExperimentType


def main():
    print("=" * 70)
    print("TEMPO TUI Demo - API Showcase")
    print("=" * 70)
    
    # Use test mode for demo (set to False for real LLM)
    use_test_mode = True
    
    print("\n[1/5] Initializing TEMPO client...")
    if use_test_mode:
        print("  Running in TEST MODE (mock LLM)")
        print("  Set use_test_mode=False to load real LLM models")
    else:
        print("  Loading LLM models (this may take a few minutes)...")
    client = TEMPOClient(test_mode=use_test_mode)
    
    if not client.initialize():
        print("✗ Failed to initialize!")
        print("  Make sure you have:")
        print("  - Sufficient GPU memory")
        print("  - Model files available")
        print("  - Dependencies installed")
        return
    print("✓ Client initialized successfully")
    
    # Chat demo - Progressive display (dual-stream advantage)
    print("\n[2/6] Testing chat with progressive display...")
    print("  This demonstrates the dual-stream advantage:")
    print("  - System 1 (Bridge): Fast TTFT (~100ms)")
    print("  - System 2 (Response): Slower but thorough (~500ms)\n")
    
    test_messages = [
        "Hello!",
        "How does dual-stream inference work?",
        "What is the current psychological state?"
    ]
    
    for msg in test_messages:
        print(f"  User: {msg}")
        
        # Use progressive display to show bridge immediately
        def show_bridge(bridge, ttft):
            print(f"    ⏳ {bridge} (TTFT: {ttft:.0f}ms)")
        
        def show_response(response, metadata):
            print(f"    🤖 {response[:60]}...")
            sys2_ms = metadata.get('system2_latency_ms', 0)
            if sys2_ms > 0:
                print(f"       (System 2: {sys2_ms:.0f}ms)\n")
            else:
                print(f"       (Single-stream mode)\n")
        
        result = client.chat_with_progress(msg, on_bridge=show_bridge, on_complete=show_response)
    
    # Demo dual-stream vs single-stream mode
    print("\n[3/6] Testing dual-stream vs single-stream mode...")
    print("  Configuring client with dual_stream=False (single-stream mode)...")
    
    # Create a new client with dual_stream disabled
    client_single = TEMPOClient(test_mode=use_test_mode, config={'dual_stream': False})
    if client_single.initialize():
        print("  Single-stream client initialized")
        result = client_single.chat("Test query")
        print(f"  Response (System 1 only): {result.response}")
        print(f"  System 2 latency: {result.system2_latency_ms:.0f}ms (0 = skipped)\n")
    
    print("  Configuring client with dual_stream=True (dual-stream mode)...")
    client_dual = TEMPOClient(test_mode=use_test_mode, config={'dual_stream': True})
    if client_dual.initialize():
        print("  Dual-stream client initialized")
        result = client_dual.chat("Test query")
        print(f"  Response (System 1 + System 2): {result.response[:60]}...")
        print(f"  System 2 latency: {result.system2_latency_ms:.0f}ms\n")
    
    # State demo
    print("\n[3/5] Current psychological state:")
    state = client.get_state()
    print(f"  Mood:    {state.mood:.2f}")
    print(f"  Stress:  {state.stress:.2f}")
    print(f"  Defense: {state.defense:.2f}")
    print(f"  Arousal: {state.arousal:.2f}")
    print(f"  Valence: {state.valence:.2f}")
    
    # Memory demo
    print("\n[4/5] Memory statistics:")
    mem_stats = client.get_memory_stats()
    print(f"  Total episodes: {mem_stats.total_episodes}")
    print(f"  Hot tier:  {mem_stats.hot_tier_size}")
    print(f"  Warm tier: {mem_stats.warm_tier_size}")
    print(f"  Cold tier: {mem_stats.cold_tier_size}")
    
    # Metrics demo
    print("\n[5/5] Performance metrics:")
    metrics = client.get_metrics()
    print(f"  TTFT Mean:   {metrics.ttft_mean:.2f} ms")
    print(f"  TTFT Median: {metrics.ttft_median:.2f} ms")
    print(f"  TTFT P95:    {metrics.ttft_p95:.2f} ms")
    print(f"  S2 Latency:  {metrics.sys2_latency_mean:.2f} ms")
    print(f"  Total queries: {metrics.total_queries}")
    
    # Results listing
    print("\n[Bonus] Available result files:")
    results = client.list_results()[:5]  # First 5
    if results:
        for r in results:
            print(f"  - {r['filename']} ({r['modified']})")
    else:
        print("  (No result files yet)")
    
    print("\n" + "=" * 70)
    print("Demo complete! Launch the full TUI with:")
    print("  python tui/tempo_tui.py")
    print("=" * 70)


if __name__ == "__main__":
    main()
