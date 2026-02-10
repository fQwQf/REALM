# REALM Server Deployment Guide (8x RTX 3090)

This guide details how to deploy REALM on a server with 8x RTX 3090 GPUs.

## 1. Environment Setup

Create a dedicated Conda environment:

```bash
conda create -n realm python=3.10 -y
conda activate realm
pip install -r requirements.txt
```

## 2. Model Setup

We use **Qwen2.5-0.5B-Instruct** for System 1 (Reflex) and **Qwen2.5-7B-Instruct** (or 72B) for System 2 (Reflection).

## 3. GPU Allocation Strategy

With 8x 3090s (24GB each), we allocate resources as follows:

- **GPU 0**: System 1 (Reflex) + Orchestrator logic.
  - Model: `Qwen/Qwen2.5-0.5B-Instruct` (~1GB VRAM).
  - Fast, low-latency bridge generation.

- **GPU 1-7**: System 2 (Reflection).
  - Model: `Qwen/Qwen2.5-7B-Instruct` (or `Qwen/Qwen2.5-72B-Instruct` with tensor parallelism).
  - Uses `vLLM` for high-throughput serving.
  - If using 7B, 1 GPU is enough, but you can use TP=2 or TP=4 for lower latency.
  - If using 72B, use TP=4 or TP=8.

## 4. Running the Server

We provide a script `experiments/run_server.py` that loads both models.

### Standard Run (All GPUs)
```bash
python experiments/run_server.py
```

### Dynamic GPU Allocation (Shared Server)
If you only have access to specific GPUs (e.g., 4 GPUs), use `CUDA_VISIBLE_DEVICES` and adjust `--tp-size`.

**Example: Using 4 GPUs (IDs 0,1,2,3)**
```bash
# System 1 loads on the first visible GPU (ID 0)
# System 2 uses vLLM with Tensor Parallelism = 4 (uses all 4 visible GPUs)
# We set gpu-util to 0.85 to allow System 1 to coexist on GPU 0
CUDA_VISIBLE_DEVICES=0,1,2,3 python experiments/run_server.py \
    --tp-size 4 \
    --gpu-util 0.85
```

**Example: Using 2 GPUs (IDs 4,5)**
```bash
CUDA_VISIBLE_DEVICES=4,5 python experiments/run_server.py \
    --tp-size 2 \
    --gpu-util 0.80
```

### Command Line Arguments
- `--sys1-model`: Model ID for System 1 (default: `Qwen/Qwen2.5-0.5B-Instruct`)
- `--sys2-model`: Model ID for System 2 (default: `Qwen/Qwen2.5-7B-Instruct`)
- `--tp-size`: Tensor Parallel size for System 2 (default: 1)
- `--gpu-util`: GPU memory utilization for vLLM (default: 0.85)

## 5. Troubleshooting

- **OOM on GPU 0**: Ensure System 1 is loaded in `float16` or `bfloat16`.
- **vLLM Context Conflicts**: The script uses `multiprocessing` or `AsyncLLMEngine` to manage vLLM alongside the main process.
