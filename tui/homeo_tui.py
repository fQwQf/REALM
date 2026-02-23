"""
HOMEO TUI - Terminal User Interface for the HOMEO system

A modern, interactive TUI built with Textual for the HOMEO dual-stream
memory agent system.

Usage:
    python -m tui.homeo_tui
    
Or:
    python tui/homeo_tui.py
"""

import json
import asyncio
from typing import Optional
from pathlib import Path

from textual.app import App, ComposeResult
from textual.containers import Container, Horizontal, Vertical, Grid
from textual.widgets import (
    Header, Footer, Static, Button, Input, Label, 
    DataTable, ListView, ListItem, RichLog,
    TabbedContent, TabPane, ProgressBar, Select,
    Switch, Checkbox
)
from textual.reactive import reactive
from textual.worker import Worker, get_current_worker
from textual.binding import Binding

# Import HOMEO client
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from homeo_client import HOMEOClient, ExperimentType, InferenceResult


class StatusBar(Static):
    """Status bar showing system state"""
    
    def compose(self) -> ComposeResult:
        with Horizontal(id="status-bar"):
            yield Label("● System: Initializing", id="status-system", classes="status-item")
            yield Label("● Memory: 0 episodes", id="status-memory", classes="status-item")
            yield Label("● State: --", id="status-state", classes="status-item")
            yield Label("● Queries: 0", id="status-queries", classes="status-item")


class DashboardScreen(Container):
    """Main dashboard with system overview"""
    
    def compose(self) -> ComposeResult:
        with Container(id="dashboard"):
            yield Label("🏠 HOMEO Dashboard", id="dashboard-title", classes="title")
            
            with Grid(id="dashboard-grid"):
                # System Status Card
                with Container(classes="card"):
                    yield Label("System Status", classes="card-title")
                    yield Static(id="system-status-content")
                
                # Metrics Card
                with Container(classes="card"):
                    yield Label("Performance Metrics", classes="card-title")
                    yield Static(id="metrics-content")
                
                # Memory Card
                with Container(classes="card"):
                    yield Label("Memory Statistics", classes="card-title")
                    yield Static(id="memory-content")
                
                # State Card
                with Container(classes="card"):
                    yield Label("Current State", classes="card-title")
                    yield Static(id="state-content")
            
            yield Button("Refresh Dashboard", id="btn-refresh-dashboard", variant="primary")


class ChatScreen(Container):
    """Interactive chat interface"""
    
    def compose(self) -> ComposeResult:
        with Container(id="chat-screen"):
            yield Label("💬 Chat with HOMEO", id="chat-title", classes="title")
            
            with Container(id="chat-history-container"):
                yield RichLog(id="chat-history", highlight=True, markup=True)
            
            with Horizontal(id="chat-input-container"):
                yield Input(placeholder="Type your message here...", id="chat-input")
                yield Button("Send", id="btn-send", variant="primary")
                yield Button("Clear", id="btn-clear-chat", variant="default")


class MemoryScreen(Container):
    """Memory browser interface"""
    
    def compose(self) -> ComposeResult:
        with Container(id="memory-screen"):
            yield Label("🧠 Memory Browser", id="memory-title", classes="title")
            
            with TabbedContent(id="memory-tabs"):
                with TabPane("Episodes", id="tab-episodes"):
                    with Container():
                        yield DataTable(id="episodes-table")
                        with Horizontal(classes="button-row"):
                            yield Button("Refresh", id="btn-refresh-memory", variant="primary")
                            yield Button("Clear Memory", id="btn-clear-memory", variant="error")
                            yield Button("Save Memory", id="btn-save-memory", variant="default")
                            yield Button("Load Memory", id="btn-load-memory", variant="default")
                
                with TabPane("Statistics", id="tab-stats"):
                    yield Static(id="memory-stats-content")


class StateScreen(Container):
    """State visualization interface"""
    
    def compose(self) -> ComposeResult:
        with Container(id="state-screen"):
            yield Label("📊 State Visualizer", id="state-title", classes="title")
            
            with Grid(id="state-grid"):
                # State Gauges
                with Container(classes="card"):
                    yield Label("Psychological State", classes="card-title")
                    yield Static(id="state-gauges")
                
                # State History
                with Container(classes="card"):
                    yield Label("State History", classes="card-title")
                    yield Static(id="state-history")
                
                # OU Dynamics
                with Container(classes="card"):
                    yield Label("OU Dynamics", classes="card-title")
                    yield Static(id="ou-dynamics")
                
                # State Controls
                with Container(classes="card"):
                    yield Label("State Controls", classes="card-title")
                    yield Button("Reset State", id="btn-reset-state", variant="default")
                    yield Button("Add Impulse", id="btn-add-impulse", variant="primary")


class ExperimentsScreen(Container):
    """Experiments runner interface"""
    
    def compose(self) -> ComposeResult:
        with Container(id="experiments-screen"):
            yield Label("🧪 Experiments", id="experiments-title", classes="title")
            
            with Grid(id="experiments-grid"):
                # Available Experiments
                with Container(classes="card"):
                    yield Label("Available Experiments", classes="card-title")
                    
                    with Vertical(classes="experiment-list"):
                        yield Button(
                            "▶ TTFT Benchmark\n   Time To First Token measurement",
                            id="btn-run-ttft",
                            variant="primary"
                        )
                        yield Button(
                            "▶ PNH Diagnostic\n   Prompt Non-Hallucination test",
                            id="btn-run-pnh",
                            variant="primary"
                        )
                        yield Button(
                            "▶ Multilingual Test\n   Cross-language robustness",
                            id="btn-run-multilingual",
                            variant="primary"
                        )
                        yield Button(
                            "▶ Ablation Study\n   Component analysis",
                            id="btn-run-ablation",
                            variant="primary"
                        )
                        yield Button(
                            "▶ Baseline Verification\n   System health check",
                            id="btn-run-baseline",
                            variant="primary"
                        )
                
                # Progress
                with Container(classes="card"):
                    yield Label("Progress", classes="card-title")
                    yield ProgressBar(id="experiment-progress", total=100)
                    yield Static(id="experiment-status", content="Ready")
                    yield RichLog(id="experiment-log", highlight=True)
                
                # Quick Results
                with Container(classes="card"):
                    yield Label("Quick Results", classes="card-title")
                    yield Static(id="experiment-results")


class ResultsScreen(Container):
    """Results viewer interface"""
    
    def compose(self) -> ComposeResult:
        with Container(id="results-screen"):
            yield Label("📁 Results Viewer", id="results-title", classes="title")
            
            with Horizontal(id="results-layout"):
                # File list
                with Container(id="results-file-list"):
                    yield Label("Result Files", classes="section-title")
                    yield ListView(id="results-list")
                    yield Button("Refresh List", id="btn-refresh-results", variant="primary")
                
                # Content view
                with Container(id="results-content"):
                    yield Label("Select a file to view", id="results-placeholder")


class ConfigScreen(Container):
    """Configuration interface"""
    
    def compose(self) -> ComposeResult:
        with Container(id="config-screen"):
            yield Label("⚙️ Configuration", id="config-title", classes="title")
            
            with Grid(id="config-grid"):
                # GPU Settings
                with Container(classes="card"):
                    yield Label("GPU Settings", classes="card-title")
                    yield Label("System 1 GPU:")
                    yield Input(value="2", id="config-sys1-gpu", type="integer")
                    yield Label("System 2 GPUs:")
                    yield Input(value="4,5,6,7", id="config-sys2-gpus")
                
                # System Settings
                with Container(classes="card"):
                    yield Label("System Settings", classes="card-title")
                    yield Horizontal(
                        Label("Use Real LLM:"),
                        Switch(id="config-real-llm", value=False)
                    )
                    yield Horizontal(
                        Label("Dual Stream:"),
                        Switch(id="config-dual-stream", value=True)
                    )
                    yield Horizontal(
                        Label("Homeostasis:"),
                        Switch(id="config-homeostasis", value=True)
                    )
                
                # State Controller Settings
                with Container(classes="card"):
                    yield Label("State Controller", classes="card-title")
                    yield Label("Theta (Mean Reversion):")
                    yield Input(value="0.5", id="config-theta", type="number")
                    yield Label("Sigma (Noise):")
                    yield Input(value="0.05", id="config-sigma", type="number")
                
                # Actions
                with Container(classes="card"):
                    yield Label("Actions", classes="card-title")
                    yield Button("Apply Settings", id="btn-apply-config", variant="primary")
                    yield Button("Reset to Defaults", id="btn-reset-config", variant="default")
                    yield Button("Save Config", id="btn-save-config", variant="default")


class HOMEOApp(App):
    """Main HOMEO TUI Application"""
    
    CSS = """
    Screen {
        align: center middle;
    }
    
    .title {
        text-style: bold;
        text-align: center;
        padding: 1;
        color: $primary;
    }
    
    #status-bar {
        dock: top;
        height: 1;
        background: $surface;
        color: $text;
        padding: 0 1;
    }
    
    .status-item {
        margin-right: 3;
    }
    
    .card {
        border: solid $primary;
        padding: 1;
        margin: 1;
    }
    
    .card-title {
        text-style: bold;
        color: $primary;
        margin-bottom: 1;
    }
    
    #dashboard-grid {
        grid-size: 2;
        grid-gutter: 1;
    }
    
    #state-grid {
        grid-size: 2;
        grid-gutter: 1;
    }
    
    #experiments-grid {
        grid-size: 3;
        grid-gutter: 1;
    }
    
    #config-grid {
        grid-size: 2;
        grid-gutter: 1;
    }
    
    #chat-history-container {
        height: 70%;
        border: solid $primary;
        margin: 1;
    }
    
    #chat-input-container {
        height: auto;
        margin: 1;
    }
    
    #chat-input {
        width: 80%;
    }
    
    #btn-send {
        margin-left: 1;
    }
    
    #btn-clear-chat {
        margin-left: 1;
    }
    
    .experiment-list Button {
        margin: 1;
        height: 3;
        content-align: left middle;
    }
    
    #results-layout {
        height: 100%;
    }
    
    #results-file-list {
        width: 30%;
        border: solid $primary;
        padding: 1;
    }
    
    #results-content {
        width: 70%;
        border: solid $primary;
        padding: 1;
    }
    
    #results-list {
        height: 85%;
    }
    
    .button-row {
        margin-top: 1;
    }
    
    .button-row Button {
        margin-right: 1;
    }
    
    #experiment-log {
        height: 60%;
        border: solid $surface;
        margin-top: 1;
    }
    
    DataTable {
        height: 80%;
    }
    """
    
    BINDINGS = [
        Binding(key="q", action="quit", description="Quit"),
        Binding(key="d", action="show_dashboard", description="Dashboard"),
        Binding(key="c", action="show_chat", description="Chat"),
        Binding(key="m", action="show_memory", description="Memory"),
        Binding(key="s", action="show_state", description="State"),
        Binding(key="e", action="show_experiments", description="Experiments"),
        Binding(key="r", action="show_results", description="Results"),
        Binding(key="o", action="show_config", description="Config"),
    ]
    
    def __init__(self):
        super().__init__()
        self.client: Optional[HOMEOClient] = None
        self._initialized = False
        self._state_history = []
    
    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        yield StatusBar()
        
        with TabbedContent(id="main-tabs"):
            with TabPane("Dashboard", id="tab-dashboard"):
                yield DashboardScreen()
            with TabPane("Chat", id="tab-chat"):
                yield ChatScreen()
            with TabPane("Memory", id="tab-memory"):
                yield MemoryScreen()
            with TabPane("State", id="tab-state"):
                yield StateScreen()
            with TabPane("Experiments", id="tab-experiments"):
                yield ExperimentsScreen()
            with TabPane("Results", id="tab-results"):
                yield ResultsScreen()
            with TabPane("Config", id="tab-config"):
                yield ConfigScreen()
        
        yield Footer()
    
    def on_mount(self):
        """Called when app is mounted"""
        self.title = "HOMEO - Human-like Organization of Memory and Executive Oversight"
        self.sub_title = "Dual-Stream Memory Agent System"
        
        # Initialize client
        self.run_worker(self._initialize_client(), exclusive=True)
    
    async def _initialize_client(self):
        """Initialize HOMEO client in background"""
        self.notify("Initializing HOMEO system...", severity="information")
        
        try:
            self.client = HOMEOClient(use_real_llm=False)
            success = await asyncio.get_event_loop().run_in_executor(
                None, self.client.initialize
            )
            
            if success:
                self._initialized = True
                self.notify("HOMEO system initialized successfully!", severity="success")
                self._update_status_bar()
                self._refresh_dashboard()
            else:
                self.notify("Failed to initialize HOMEO system", severity="error")
        except Exception as e:
            self.notify(f"Initialization error: {e}", severity="error")
    
    def _update_status_bar(self):
        """Update status bar with current state"""
        if not self.client or not self._initialized:
            return
        
        # System status
        status_widget = self.query_one("#status-system", Label)
        status_widget.update("● System: Ready" if self._initialized else "● System: Error")
        
        # Memory stats
        try:
            mem_stats = self.client.get_memory_stats()
            mem_widget = self.query_one("#status-memory", Label)
            mem_widget.update(f"● Memory: {mem_stats.total_episodes} episodes")
        except:
            pass
        
        # State
        try:
            state = self.client.get_state()
            state_widget = self.query_one("#status-state", Label)
            state_widget.update(f"● State: M={state.mood:.2f}/S={state.stress:.2f}")
        except:
            pass
        
        # Queries
        try:
            metrics = self.client.get_metrics()
            queries_widget = self.query_one("#status-queries", Label)
            queries_widget.update(f"● Queries: {metrics.total_queries}")
        except:
            pass
    
    def _refresh_dashboard(self):
        """Refresh dashboard content"""
        if not self.client or not self._initialized:
            return
        
        # System status
        try:
            status_content = self.query_one("#system-status-content", Static)
            status_text = f"""
[b]Initialized:[/b] {'Yes' if self._initialized else 'No'}
[b]Real LLM:[/b] {self.client.use_real_llm}
[b]System 1 GPU:[/b] {self.client.sys1_gpu}
[b]System 2 GPUs:[/b] {', '.join(map(str, self.client.sys2_gpus))}
            """.strip()
            status_content.update(status_text)
        except Exception as e:
            pass
        
        # Metrics
        try:
            metrics_content = self.query_one("#metrics-content", Static)
            metrics = self.client.get_metrics()
            metrics_text = f"""
[b]TTFT Mean:[/b] {metrics.ttft_mean:.2f} ms
[b]TTFT Median:[/b] {metrics.ttft_median:.2f} ms
[b]TTFT P95:[/b] {metrics.ttft_p95:.2f} ms
[b]System 2 Latency:[/b] {metrics.sys2_latency_mean:.2f} ms
[b]Retrieval Time:[/b] {metrics.retrieval_time_mean:.2f} ms
[b]Total Queries:[/b] {metrics.total_queries}
            """.strip()
            metrics_content.update(metrics_text)
        except Exception as e:
            pass
        
        # Memory
        try:
            memory_content = self.query_one("#memory-content", Static)
            mem_stats = self.client.get_memory_stats()
            memory_text = f"""
[b]Total Episodes:[/b] {mem_stats.total_episodes}
[b]Hot Tier:[/b] {mem_stats.hot_tier_size}
[b]Warm Tier:[/b] {mem_stats.warm_tier_size}
[cold]Cold Tier:[/b] {mem_stats.cold_tier_size}
            """.strip()
            memory_content.update(memory_text)
        except Exception as e:
            pass
        
        # State
        try:
            state_content = self.query_one("#state-content", Static)
            state = self.client.get_state()
            state_text = f"""
[b]Mood:[/b] {state.mood:.2f}
[b]Stress:[/b] {state.stress:.2f}
[b]Defense:[/b] {state.defense:.2f}
[b]Arousal:[/b] {state.arousal:.2f}
[b]Valence:[/b] {state.valence:.2f}
            """.strip()
            state_content.update(state_text)
        except Exception as e:
            pass
    
    # Button Handlers
    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses"""
        button_id = event.button.id
        
        if button_id == "btn-refresh-dashboard":
            self._refresh_dashboard()
            self._update_status_bar()
        
        elif button_id == "btn-send":
            self._handle_send_message()
        
        elif button_id == "btn-clear-chat":
            chat_history = self.query_one("#chat-history", RichLog)
            chat_history.clear()
        
        elif button_id == "btn-refresh-memory":
            self._refresh_memory()
        
        elif button_id == "btn-clear-memory":
            self._clear_memory()
        
        elif button_id == "btn-save-memory":
            self._save_memory()
        
        elif button_id == "btn-load-memory":
            self._load_memory()
        
        elif button_id in ["btn-run-ttft", "btn-run-pnh", "btn-run-multilingual", 
                          "btn-run-ablation", "btn-run-baseline"]:
            self._run_experiment(button_id)
        
        elif button_id == "btn-refresh-results":
            self._refresh_results_list()
    
    def _handle_send_message(self):
        """Handle sending a chat message"""
        if not self._initialized:
            self.notify("System not initialized yet!", severity="warning")
            return
        
        input_widget = self.query_one("#chat-input", Input)
        message = input_widget.value.strip()
        
        if not message:
            return
        
        chat_history = self.query_one("#chat-history", RichLog)
        
        # Show user message
        chat_history.write(f"[b]You:[/b] {message}")
        input_widget.value = ""
        
        # Get response
        try:
            result = self.client.chat(message)
            
            # Show bridge if available
            if result.bridge:
                chat_history.write(f"[dim][System 1 Bridge: {result.bridge}][/dim]")
            
            # Show response
            chat_history.write(f"[b green]HOMEO:[/b green] {result.response}")
            
            # Show metadata
            chat_history.write(
                f"[dim]TTFT: {result.ttft_ms:.1f}ms | "
                f"S2 Latency: {result.system2_latency_ms:.1f}ms | "
                f"Retrieval: {result.retrieval_time_ms:.1f}ms[/dim]"
            )
            chat_history.write("")
            
            # Update status
            self._update_status_bar()
            
        except Exception as e:
            chat_history.write(f"[b red]Error:[/b red] {e}")
    
    def _refresh_memory(self):
        """Refresh memory display"""
        if not self._initialized:
            return
        
        try:
            table = self.query_one("#episodes-table", DataTable)
            table.clear()
            
            # Add columns if not present
            if not table.columns:
                table.add_columns("#", "Timestamp", "User", "Agent")
            
            # Add rows
            history = self.client.get_recent_history(50)
            for i, turn in enumerate(reversed(history)):
                from datetime import datetime
                timestamp = datetime.fromtimestamp(turn['timestamp']).strftime('%H:%M:%S')
                user = turn['user'][:40] + "..." if len(turn['user']) > 40 else turn['user']
                agent = turn['agent'][:40] + "..." if len(turn['agent']) > 40 else turn['agent']
                table.add_row(str(i+1), timestamp, user, agent)
            
            # Update stats
            stats = self.client.get_memory_stats()
            stats_content = self.query_one("#memory-stats-content", Static)
            stats_text = f"""
[b]Total Episodes:[/b] {stats.total_episodes}
[b]Hot Tier:[/b] {stats.hot_tier_size}
[b]Warm Tier:[/b] {stats.warm_tier_size}
[b]Cold Tier:[/b] {stats.cold_tier_size}
            """.strip()
            stats_content.update(stats_text)
            
        except Exception as e:
            self.notify(f"Error refreshing memory: {e}", severity="error")
    
    def _clear_memory(self):
        """Clear memory"""
        if not self._initialized:
            return
        
        self.client.clear_memory()
        self.notify("Memory cleared", severity="information")
        self._refresh_memory()
        self._update_status_bar()
    
    def _save_memory(self):
        """Save memory to file"""
        if not self._initialized:
            return
        
        filepath = "memory_backup.json"
        self.client.save_memory(filepath)
        self.notify(f"Memory saved to {filepath}", severity="success")
    
    def _load_memory(self):
        """Load memory from file"""
        if not self._initialized:
            return
        
        filepath = "memory_backup.json"
        try:
            self.client.load_memory(filepath)
            self.notify(f"Memory loaded from {filepath}", severity="success")
            self._refresh_memory()
            self._update_status_bar()
        except Exception as e:
            self.notify(f"Error loading memory: {e}", severity="error")
    
    def _run_experiment(self, button_id: str):
        """Run an experiment"""
        if not self._initialized:
            self.notify("System not initialized!", severity="warning")
            return
        
        # Map button to experiment type
        experiment_map = {
            "btn-run-ttft": ExperimentType.TTFT,
            "btn-run-pnh": ExperimentType.PNH,
            "btn-run-multilingual": ExperimentType.MULTILINGUAL,
            "btn-run-ablation": ExperimentType.ABLATION,
            "btn-run-baseline": ExperimentType.BASELINE,
        }
        
        exp_type = experiment_map.get(button_id)
        if not exp_type:
            return
        
        # Run in background
        self.run_worker(
            self._run_experiment_worker(exp_type),
            exclusive=True
        )
    
    async def _run_experiment_worker(self, exp_type: ExperimentType):
        """Worker for running experiments"""
        try:
            # Get UI elements
            progress = self.query_one("#experiment-progress", ProgressBar)
            status = self.query_one("#experiment-status", Static)
            log = self.query_one("#experiment-log", RichLog)
            results = self.query_one("#experiment-results", Static)
            
            progress.update(progress=0)
            log.clear()
            log.write(f"Starting {exp_type.value} experiment...")
            
            # Progress callback
            def progress_callback(msg: str):
                log.write(msg)
                # Update progress bar (simulate)
                current = progress.progress or 0
                progress.update(progress=min(current + 20, 90))
            
            # Run experiment
            exp_result = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.client.run_experiment(exp_type, callback=progress_callback)
            )
            
            progress.update(progress=100)
            status.update(f"{exp_type.value.upper()} Complete")
            
            # Display results
            if 'statistics' in exp_result:
                stats = exp_result['statistics']
                result_text = f"""
[b]{exp_type.value.upper()} Results[/b]

"""
                for key, value in stats.items():
                    if isinstance(value, float):
                        result_text += f"[b]{key}:[/b] {value:.2f}\n"
                    else:
                        result_text += f"[b]{key}:[/b] {value}\n"
                
                results.update(result_text)
            elif 'accuracy_percent' in exp_result:
                results.update(f"[b]Accuracy:[/b] {exp_result['accuracy_percent']:.1f}%")
            else:
                results.update(f"[b]Status:[/b] Complete\n[b]Results saved to:[/b] results/")
            
            self.notify(f"{exp_type.value} experiment complete!", severity="success")
            
        except Exception as e:
            self.notify(f"Experiment error: {e}", severity="error")
    
    def _refresh_results_list(self):
        """Refresh results file list"""
        try:
            list_view = self.query_one("#results-list", ListView)
            list_view.clear()
            
            results = self.client.list_results() if self.client else []
            
            for result in results:
                label = f"{result['filename']}\n  ({result['modified']})"
                list_view.append(ListItem(Label(label)))
            
        except Exception as e:
            self.notify(f"Error loading results: {e}", severity="error")
    
    # Keyboard shortcuts
    def action_show_dashboard(self):
        self.query_one("#main-tabs", TabbedContent).active = "tab-dashboard"
    
    def action_show_chat(self):
        self.query_one("#main-tabs", TabbedContent).active = "tab-chat"
    
    def action_show_memory(self):
        self.query_one("#main-tabs", TabbedContent).active = "tab-memory"
    
    def action_show_state(self):
        self.query_one("#main-tabs", TabbedContent).active = "tab-state"
    
    def action_show_experiments(self):
        self.query_one("#main-tabs", TabbedContent).active = "tab-experiments"
    
    def action_show_results(self):
        self.query_one("#main-tabs", TabbedContent).active = "tab-results"
        self._refresh_results_list()
    
    def action_show_config(self):
        self.query_one("#main-tabs", TabbedContent).active = "tab-config"


def main():
    """Entry point"""
    app = HOMEOApp()
    app.run()


if __name__ == "__main__":
    main()
