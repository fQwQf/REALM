# TEMPO: Timed Engagement with Memory and Persona Orchestration

Official implementation of **TEMPO**, a dual-stream agent architecture for real-time personalized dialogue with psychological state modeling and bounded homeostasis.

> **Paper**: *TEMPO: Timed Engagement with Memory and Persona Orchestration*  
> **Conference**: EMNLP 2025 (under review)  
> **Anonymous Submission**: This repository accompanies the anonymous submission for reproducibility review.

---

## Overview

TEMPO addresses the **latency-coherence dilemma** in personalized dialogue agents: how to provide instant responses while maintaining psychological continuity and factual grounding. The system features:

- **Dual-Stream Architecture**: System 1 (Reflex, 0.5B) provides instant low-commitment bridges while System 2 (Reflection, 7B/14B) performs state-conditioned retrieval and reasoning
- **Safe-to-Say Mechanism**: LoRA-steered generation that suppresses implicit commitments (promises, apologies, stance lock-in) during latency masking
- **NLI-Based Stream Stitching**: Entailment-based conflict detection and repair at the bridge–response boundary
- **Bounded OU Tempostasis**: Ornstein–Uhlenbeck process for mean-reverting psychological state dynamics (Mood, Stress, Defense mechanisms)
- **Accordion Memory**: Constant-cap episodic memory with Hot/Warm/Cold tiers and state-motivated retrieval

**Key Results**:
- **TTFT**: 23ms (System 1) vs 1.6s (7B-only) / 6.1s (14B-only)
- **PNH-10**: 90% accuracy on state-conditioned recall
- **Human Eval**: 76.2% win rate (N=20 raters, 400 decisions), naturalness +0.73 [+0.36, +1.10], p<0.001

---

## Installation

### Prerequisites
- Python 3.10+
- CUDA 11.8+ (for GPU acceleration)
- 24GB+ VRAM (RTX 3090 or equivalent) for 7B models; 3×24GB for 14B

### Setup

```bash
# Clone repository
git clone <anonymous-repo-url>
cd REALM

# Create conda environment
conda create -n realm python=3.10
conda activate realm

# Install dependencies
pip install torch==2.1.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install transformers==4.40.0 peft==0.10.0 accelerate bitsandbytes
pip install sentence-transformers faiss-cpu numpy scipy pandas
pip install textual rich  # For TUI

# Set environment variables (optional, defaults to ~/.cache/huggingface)
export HF_HOME=~/.cache/huggingface
export HF_ENDPOINT=https://hf-mirror.com  # Use mirror if needed
```

### Download Models

```bash
# Download base models (Qwen2.5)
python experiments/download_models.py

# Train Safe-to-Say LoRA (or use provided checkpoint)
python experiments/train_lora_system1.py \
  --base Qwen/Qwen2.5-0.5B-Instruct \
  --data data/safe_to_say_train.json \
  --output models/safe_to_say_lora \
  --seed 42 --epochs 3 --lr 2e-4 --r 8 --alpha 16
```

---

## Quick Start

### Interactive TUI

```bash
# Launch terminal interface
python tui/tempo_tui.py

# Or use the demo client
python demo_tui.py
```

### Run Main Evaluation

```bash
# Full evaluation suite (TTFT, PNH-10/51, NLI stitcher)
conda run -n realm python experiments/tempo_full_eval.py \
  --seed 42 \
  --s1 Qwen/Qwen2.5-0.5B-Instruct \
  --s2 Qwen/Qwen2.5-7B-Instruct \
  --lora models/safe_to_say_lora

# Results saved to: results/full_evaluation/MASTER_SUMMARY.json
```

### 14B Scale Experiment

```bash
# Requires 3×RTX 3090 or equivalent
export MODEL_DIR=/path/to/your/models  # Set if using custom model directory

python experiments/run_14b_scale_experiment.py \
  --s2 ${MODEL_DIR}/Qwen2.5-14B-Instruct \
  --seed 42

# Results: results/full_evaluation/scale_14b.json
```

---

## Repository Structure

```
REALM/
├── src/                      # Core implementation
│   ├── realm.py             # Main TEMPO agent
│   ├── state.py             # OU state controller
│   ├── memory.py            # Accordion memory (Hot/Warm/Cold)
│   └── llm_backend.py       # Dual-stream LLM interface
├── experiments/             # Evaluation scripts
│   ├── tempo_full_eval.py   # Main evaluation suite
│   ├── run_14b_scale_experiment.py  # 14B scaling
│   ├── compute_full_stats.py        # Bootstrap CIs, Cohen's d
│   └── benchmarks/          # PNH, TTFT, MSC benchmarks
├── data/                    # Datasets
│   ├── safe_to_say_train.json       # LoRA training data (1772 examples)
│   └── test_sets/
│       ├── pnh_test_set.json        # PNH-10 curated set
│       └── pnh_extended_test_set.json  # PNH-51 stress test
├── models/                  # Model checkpoints
│   └── safe_to_say_lora/    # Trained LoRA adapters
├── results/                 # Experimental results
│   ├── full_evaluation/     # Main results (MASTER_SUMMARY.json)
│   └── human_eval/          # Human evaluation data
├── tui/                     # Terminal UI
│   └── tempo_tui.py         # Interactive interface
└── scripts/                 # Utility scripts
    └── fix_paths.py         # Path normalization
```

---

## Reproducing Paper Results

All commands assume the `realm` conda environment is activated.

### Main Results (Tables 2+3)

```bash
# TTFT, PNH-10/51, NLI stitcher, system comparison
python experiments/tempo_full_eval.py --seed 42
```

### Human Evaluation Statistics (Appendix G)

```bash
# Compute bootstrap CIs, Cohen's d, per-pair breakdown
python experiments/compute_full_stats.py

# Output: results/full_evaluation/STATS_ANALYSIS.json
```

### 14B Scaling (Appendix I.16)

```bash
# Measured results on 3×RTX 3090
python experiments/run_14b_scale_experiment.py --seed 42
```

### Ablation Studies

```bash
# Full ablation matrix (Accordion Memory, Motivated Retrieval, etc.)
python experiments/run_full_config_ablation.py --seed 42
```

---

## Key Components

### Safe-to-Say LoRA

Low-commitment bridge generation via LoRA steering:

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM

# Load base model + LoRA
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
model = PeftModel.from_pretrained(model, "models/safe_to_say_lora")
model = model.merge_and_unload()
```

**Training data**: 1772 query-bridge pairs covering FACTUAL, OPINION, GREETING, SHARING types. Bridges avoid implicit commitments (no "I promise", "I agree", "I think", etc.).

### PNH (Psychological Needle-in-Haystack)

State-conditioned recall diagnostic:

```python
# Example case
needle = "User prefers to be called 'Alex'"
state = {"mood": "Defensive", "stress": 80, "defense": "Rationalization"}
haystack = [6 distractor turns]
trigger = "What name do I prefer again?"

# Scoring: PASS if (recall=1 AND state_alignment=1)
# recall=1: response contains ≥40% of needle keywords
# state_alignment=1: response respects current Ego state
```

**Datasets**:
- `pnh_test_set.json`: 10 curated cases (90% accuracy)
- `pnh_extended_test_set.json`: 51 stress-test cases (80.4% accuracy)

### NLI Stream Stitching

Conflict detection at bridge–response boundary:

```python
from transformers import pipeline

nli = pipeline('zero-shot-classification', 
               model='cross-encoder/nli-deberta-v3-base')

bridge = "Let me check your notes..."
response = "I found the information you requested."
combined = f"{bridge} {response}"

result = nli(combined, ['entailment', 'neutral', 'contradiction'])
# If contradiction → trigger repair clause
```

**Validation**: 85.7% on 13-pair curated set, 73.2% on 41-pair extended set (95% CI [58.5, 85.4]).

---

## Citation

```bibtex
@inproceedings{anonymous2025tempo,
  title={TEMPO: Timed Engagement with Memory and Persona Orchestration},
  author={Anonymous},
  booktitle={Proceedings of EMNLP 2025},
  year={2025},
  note={Under review}
}
```

---

## License

This code is released under the MIT License for research purposes. See `LICENSE` for details.

---

## Reproducibility Checklist

- ✅ Code: Full implementation provided
- ✅ Data: PNH test sets, Safe-to-Say training data, human eval ratings
- ✅ Models: LoRA checkpoints included; base models downloadable via HuggingFace
- ✅ Environment: `requirements.txt` + conda environment spec
- ✅ Seeds: All experiments use fixed seeds (42, 1337, 2024–2026)
- ✅ Hardware: RTX 3090 (24GB) specifications documented
- ✅ Results: JSON outputs with full statistics (bootstrap CIs, Cohen's d, df)

**Estimated compute**: ~8 GPU-hours (RTX 3090) for full evaluation suite.

---

## Contact

For questions about this anonymous submission, please use the conference review system or contact the program chairs.
