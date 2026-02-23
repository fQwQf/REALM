#!/usr/bin/env python3
"""
Quick demo of HOMEO TUI functionality

This script demonstrates the API without launching the full TUI.
Run with: python demo_tui.py
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from homeo_client import HOMEOClient, ExperimentType


def main():
    print("=" * 70)
    print("HOMEO TUI Demo - API Showcase")
    print("=" * 70)
    
    # Use test mode for demo (set to False for real LLM)
    use_test_mode = True
    
    print("\n[1/5] Initializing HOMEO client...")
    if use_test_mode:
        print("  Running in TEST MODE (mock LLM)")
        print("  Set use_test_mode=False to load real LLM models")
    else:
        print("  Loading LLM models (this may take a few minutes)...")
    client = HOMEOClient(test_mode=use_test_mode)
    
    if not client.initialize():
        print("✗ Failed to initialize!")
        print("  Make sure you have:")
        print("  - Sufficient GPU memory")
        print("  - Model files available")
        print("  - Dependencies installed")
        return
    print("✓ Client initialized successfully")
    
    # Chat demo
    print("\n[2/5] Testing chat functionality...")
    test_messages = [
        "Hello!",
        "How does dual-stream inference work?",
        "What is the current psychological state?"
    ]
    
    for msg in test_messages:
        result = client.chat(msg)
        print(f"  User: {msg}")
        print(f"  HOMEO: {result.response[:60]}...")
        print(f"  (TTFT: {result.ttft_ms:.2f}ms)\n")
    
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
    print("  python tui/homeo_tui.py")
    print("=" * 70)


if __name__ == "__main__":
    main()
