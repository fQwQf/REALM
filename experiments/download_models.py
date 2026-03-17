#!/usr/bin/env python3
"""
Download Qwen Models for REALM
Uses HF mirror for China network environment
"""
import os
import sys
from pathlib import Path

# Auto-detect repository root
REPO_ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(REPO_ROOT))

# Environment variables with fallbacks
HF_HOME = os.environ.get('HF_HOME', os.path.expanduser('~/.cache/huggingface'))
os.environ['HF_HOME'] = HF_HOME
os.environ['HF_ENDPOINT'] = os.environ.get('HF_ENDPOINT', 'https://hf-mirror.com')

# Model directory (for 14B experiments)
MODEL_DIR = os.environ.get('MODEL_DIR', str(REPO_ROOT / 'models'))


import os
import sys

# Set Hugging Face mirror

print("="*60)
print("Downloading Qwen Models for REALM")
print("="*60)
print(f"HF_ENDPOINT: {os.environ['HF_ENDPOINT']}")
print(f"HF_HOME: {os.environ['HF_HOME']}")
print()

# System 1: 0.5B model (fast, small)
SYS1_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"

# System 2: 7B model (more capable)
SYS2_MODEL = "Qwen/Qwen2.5-7B-Instruct"

print(f"System 1 (Reflex): {SYS1_MODEL}")
print(f"System 2 (Reflection): {SYS2_MODEL}")
print()

def download_model(model_id: str, description: str):
    """Download a model from Hugging Face"""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    print(f"\n[Downloading {description}: {model_id}]")
    print("="*60)
    
    try:
        # Download tokenizer
        print("1. Downloading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            trust_remote_code=True,
            resume_download=True
        )
        print("   ✓ Tokenizer ready")
        
        # Download model (just download, don't load to GPU yet)
        print("2. Downloading model weights...")
        print("   (This may take several minutes)")
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            trust_remote_code=True,
            resume_download=True,
            torch_dtype="auto"  # Don't load to device, just download
        )
        print("   ✓ Model weights downloaded")
        
        print(f"✓ {description} ready!")
        return True
        
    except Exception as e:
        print(f"✗ Failed to download {model_id}: {e}")
        import traceback
        traceback.print_exc()
        return False

# Download models
print("Starting downloads...")
print("Note: This may take 10-30 minutes depending on network speed")
print()

success1 = download_model(SYS1_MODEL, "System 1 (0.5B)")
success2 = download_model(SYS2_MODEL, "System 2 (7B)")

print("\n" + "="*60)
if success1 and success2:
    print("✓ All models downloaded successfully!")
    print("\nYou can now run experiments with real LLM")
    sys.exit(0)
else:
    print("✗ Some models failed to download")
    if not success1:
        print("  - System 1 (0.5B) failed")
    if not success2:
        print("  - System 2 (7B) failed")
    sys.exit(1)
