#!/usr/bin/env python3
"""
Visualize Large-Scale MSC Results
=================================

Generate publication-quality plots for the paper.
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Load results
results_dir = Path("results/msc_large_scale")
with open(list(results_dir.glob("combined_results_*.json"))[-1]) as f:
    all_results = json.load(f)

# Extract data
methods = []
recall = []
ttft = []
s2_trigger = []

for result in all_results:
    if "error" in result:
        continue
    config = result["config"]
    name = config["name"]
    methods.append(name.replace(" (", "\n("))
    recall.append(result.get("recall_at_1", 0))
    ttft.append(result.get("avg_ttft", 0))
    s2_trigger.append(result.get("system2_trigger_rate", 0))

# Create figure with subplots
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Color scheme
colors = ['#2ecc71', '#3498db', '#e74c3c', '#f39c12']

# Plot 1: Recall@1
ax1 = axes[0]
bars1 = ax1.bar(range(len(methods)), recall, color=colors, alpha=0.8, edgecolor='black')
ax1.set_ylabel('Recall@1 (%)', fontsize=12, fontweight='bold')
ax1.set_title('Multi-Session Chat\nRecall Performance', fontsize=13, fontweight='bold')
ax1.set_xticks(range(len(methods)))
ax1.set_xticklabels(methods, rotation=0, ha='center', fontsize=9)
ax1.set_ylim(0, 100)
ax1.grid(axis='y', alpha=0.3)

# Add value labels on bars
for i, (bar, val) in enumerate(zip(bars1, recall)):
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height + 1,
            f'{val:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')

# Plot 2: Average TTFT
ax2 = axes[1]
bars2 = ax2.bar(range(len(methods)), ttft, color=colors, alpha=0.8, edgecolor='black')
ax2.set_ylabel('Avg TTFT (ms)', fontsize=12, fontweight='bold')
ax2.set_title('Time-To-First-Token\nLatency', fontsize=13, fontweight='bold')
ax2.set_xticks(range(len(methods)))
ax2.set_xticklabels(methods, rotation=0, ha='center', fontsize=9)
ax2.axhline(y=400, color='red', linestyle='--', linewidth=2, label='400ms threshold')
ax2.grid(axis='y', alpha=0.3)
ax2.legend()

# Add value labels
for i, (bar, val) in enumerate(zip(bars2, ttft)):
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height + 10,
            f'{val:.0f}ms', ha='center', va='bottom', fontsize=10, fontweight='bold')

# Plot 3: System 2 Trigger Rate
ax3 = axes[2]
bars3 = ax3.bar(range(len(methods)), s2_trigger, color=colors, alpha=0.8, edgecolor='black')
ax3.set_ylabel('System 2 Trigger Rate (%)', fontsize=12, fontweight='bold')
ax3.set_title('Compute Efficiency\n(S2 Trigger Rate)', fontsize=13, fontweight='bold')
ax3.set_xticks(range(len(methods)))
ax3.set_xticklabels(methods, rotation=0, ha='center', fontsize=9)
ax3.set_ylim(0, 110)
ax3.grid(axis='y', alpha=0.3)

# Add value labels
for i, (bar, val) in enumerate(zip(bars3, s2_trigger)):
    height = bar.get_height()
    ax3.text(bar.get_x() + bar.get_width()/2., height + 2,
            f'{val:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig('results/msc_large_scale/large_scale_msc_comparison.png', dpi=300, bbox_inches='tight')
print("✓ Saved: results/msc_large_scale/large_scale_msc_comparison.png")

# Create trade-off plot (Recall vs S2 Trigger)
fig2, ax = plt.subplots(figsize=(8, 6))

for i, (method, rec, s2) in enumerate(zip(methods, recall, s2_trigger)):
    ax.scatter(s2, rec, s=300, c=colors[i], alpha=0.7, edgecolors='black', linewidth=2, label=method.replace('\n', ' '))
    ax.annotate(method.replace('\n', ' '), (s2, rec), xytext=(5, 5), textcoords='offset points', fontsize=9)

ax.set_xlabel('System 2 Trigger Rate (%)', fontsize=12, fontweight='bold')
ax.set_ylabel('Recall@1 (%)', fontsize=12, fontweight='bold')
ax.set_title('Speed-Accuracy Trade-off\n(Lower-left is better)', fontsize=13, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.set_xlim(0, 110)
ax.set_ylim(40, 90)

# Add ideal region
ax.axhspan(70, 85, alpha=0.1, color='green', label='Target Recall')
ax.axvspan(50, 70, alpha=0.1, color='blue', label='Target Efficiency')

ax.legend(loc='lower right', fontsize=8)
plt.tight_layout()
plt.savefig('results/msc_large_scale/trade_off_analysis.png', dpi=300, bbox_inches='tight')
print("✓ Saved: results/msc_large_scale/trade_off_analysis.png")

print("\nVisualization complete!")
