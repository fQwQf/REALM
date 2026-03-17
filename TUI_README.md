# HOMEO TUI - Terminal User Interface

A modern, interactive terminal interface for the HOMEO (Human-like Organization of Memory and Executive Oversight) dual-stream memory agent system.

## Overview

The HOMEO TUI provides an intuitive way to interact with the dual-stream memory agent system, featuring:

- **Interactive Chat**: Real-time conversation with the HOMEO agent
- **Dashboard**: System status, performance metrics, and statistics
- **Memory Browser**: View and manage episodic memory (Hot/Warm/Cold tiers)
- **State Visualizer**: Monitor psychological state (Mood, Stress, Defense, etc.)
- **Experiments**: Run TTFT, PNH, multilingual, and ablation experiments
- **Results Viewer**: Browse and display experimental results
- **Configuration**: Adjust system parameters interactively

## Installation

The TUI requires the `textual` library, which is already installed in the realm conda environment.

```bash
# Activate the realm environment
conda activate realm

# Or use the direct Python path (adjust to your conda installation)
python tui/homeo_tui.py
```

## Usage

### Starting the TUI

```bash
# Method 1: Run as module
python -m tui.homeo_tui

# Method 2: Run directly
python tui/homeo_tui.py

# Method 3: Use realm environment explicitly
conda run -n realm python tui/homeo_tui.py
```

### Keyboard Shortcuts

| Key | Action |
|-----|--------|
| `q` | Quit the application |
| `d` | Show Dashboard |
| `c` | Show Chat |
| `m` | Show Memory Browser |
| `s` | Show State Visualizer |
| `e` | Show Experiments |
| `r` | Show Results Viewer |
| `o` | Show Configuration |

### Navigation

- Use mouse clicks or keyboard shortcuts to navigate between tabs
- The status bar at the top shows real-time system information
- Use arrow keys or Tab to navigate within a screen

## Screens

### Dashboard

The dashboard provides an overview of the system:
- System Status: Initialization state, GPU allocation, LLM backend
- Performance Metrics: TTFT, System 2 latency, retrieval times
- Memory Statistics: Episode count, tier distribution
- Current State: Mood, Stress, Defense, Arousal, Valence

### Chat

Interactive conversation interface:
- Type messages in the input field and press Enter or click Send
- View System 1 bridge responses (if enabled)
- See full HOMEO responses with metadata (TTFT, latency)
- Conversation history is maintained and stored in memory

### Memory Browser

Browse and manage episodic memory:
- **Episodes Tab**: View conversation history in a table format
- **Statistics Tab**: See memory tier distribution
- Actions: Refresh, Clear, Save, Load memory

### State Visualizer

Monitor the agent's psychological state:
- Current state values (0.0-1.0 scale)
- State history tracking
- OU Dynamics visualization
- Controls to reset state or add impulses

### Experiments

Run diagnostic and benchmark experiments:

1. **TTFT Benchmark**: Measure Time To First Token performance
2. **PNH Diagnostic**: Test Prompt Non-Hallucination accuracy
3. **Multilingual Test**: Verify cross-language robustness
4. **Ablation Study**: Analyze component contributions
5. **Baseline Verification**: Check system health

Results are saved to the `results/` directory.

### Results Viewer

Browse experimental results:
- List all JSON result files
- View file metadata (size, modification time)
- Display result contents

### Configuration

Adjust system parameters:
- GPU Settings: System 1 and System 2 GPU allocation
- System Settings: Enable/disable features
- State Controller: Theta (mean reversion), Sigma (noise)

## API Usage

The TUI is built on top of a clean Python API (`homeo_client.py`):

```python
from homeo_client import HOMEOClient, ExperimentType

# Initialize client
client = HOMEOClient(use_real_llm=False)
client.initialize()

# Chat
result = client.chat("Hello, how are you?")
print(result.response)

# Get state
state = client.get_state()
print(f"Mood: {state.mood:.2f}")

# Run experiments
results = client.run_experiment(ExperimentType.TTFT)
print(f"Mean TTFT: {results['statistics']['mean']:.2f}ms")

# List results
for file_info in client.list_results():
    print(f"{file_info['filename']}: {file_info['modified']}")
```

## Testing

Run the test suite:

```bash
# Run all tests
python -m pytest tests/test_tui.py -v

# Run specific test class
python -m pytest tests/test_tui.py::TestHOMEOClient -v
```

## Architecture

The TUI is built with [Textual](https://textual.textualize.io/), a modern Python framework for terminal user interfaces:

```
tui/
├── homeo_tui.py          # Main TUI application
└── __init__.py           # Package init

homeo_client.py           # Clean API layer
tests/test_tui.py         # Test suite
```

### Key Components

- **HOMEOClient**: High-level API for the HOMEO system
- **HOMEOApp**: Main TUI application class
- **DashboardScreen**: System overview
- **ChatScreen**: Interactive conversation
- **MemoryScreen**: Memory browser
- **StateScreen**: State visualization
- **ExperimentsScreen**: Experiment runner
- **ResultsScreen**: Results viewer
- **ConfigScreen**: Configuration interface

## Troubleshooting

### Import Errors

If you see import errors, ensure you're using the realm environment:

```bash
which python  # Should show realm environment
```

### TUI Won't Start

Check that textual is installed:

```bash
python -c "import textual; print(textual.__version__)"
```

### Slow Performance

The TUI runs with a simulated backend by default. For real LLM inference:

1. Ensure GPUs are available
2. Set `use_real_llm=True` in configuration
3. Models will be loaded on first use (may take several minutes)

## License

Part of the HOMEO research project.

## See Also

- Main Paper: `paper/main.tex`
- Source Code: `src/`
- Experiments: `experiments/`
- Results: `results/`
