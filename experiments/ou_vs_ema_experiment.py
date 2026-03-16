#!/usr/bin/env python3
"""
OU vs EMA State Dynamics Comparison

Compares Ornstein-Uhlenbeck (OU) mean reversion against:
  1. First-order EMA (alpha-smoothing only, no mean reversion)
  2. Clamped EMA (EMA with hard clamp to [0,1], no mean reversion)
  3. No dynamics (direct impulse with clamp)

Metrics:
  - State stability (variance over time)
  - Recovery time after perturbation
  - Persona break rate (PNH accuracy under each dynamics)
  - Drift rate (max deviation from baseline trait)

GPU: 0 for S2 inference
"""
import os, sys, json, time, math
import numpy as np
from pathlib import Path
from copy import deepcopy

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['HF_HOME'] = '/data1/tongjizhou/.cache/huggingface'
sys.path.insert(0, str(Path(__file__).parent.parent))

OUT_DIR = Path('/data1/tongjizhou/REALM/results/ou_vs_ema')
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ==========================================
# State dynamics implementations
# ==========================================

def ou_update(X, D, mu_anchor, theta, epsilon_scale=0.01, rng=None):
    """Ornstein-Uhlenbeck update with bounded impulse."""
    if rng is None:
        rng = np.random.default_rng()
    mu = mu_anchor
    D_bounded = 0.3 * np.tanh(D)
    epsilon = rng.normal(0, epsilon_scale, size=X.shape)
    X_new = X + theta * (mu - X) + D_bounded + epsilon
    return np.clip(X_new, 0.0, 1.0)

def ema_update(X, D, alpha=0.1, epsilon_scale=0.01, rng=None):
    """First-order EMA: X_new = (1-alpha)*X + alpha*(X+D). No mean reversion."""
    if rng is None:
        rng = np.random.default_rng()
    D_bounded = 0.3 * np.tanh(D)
    epsilon = rng.normal(0, epsilon_scale, size=X.shape)
    X_new = X + alpha * D_bounded + epsilon
    return np.clip(X_new, 0.0, 1.0)

def clamped_ema_update(X, D, alpha=0.2, epsilon_scale=0.01, rng=None):
    """Clamped EMA: larger alpha, hard clamp, no mean reversion."""
    if rng is None:
        rng = np.random.default_rng()
    D_bounded = 0.3 * np.tanh(D)
    epsilon = rng.normal(0, epsilon_scale, size=X.shape)
    X_new = X + alpha * D_bounded + epsilon
    return np.clip(X_new, 0.0, 1.0)

def direct_impulse(X, D, epsilon_scale=0.01, rng=None):
    """Direct application with clamp. No smoothing, no mean reversion."""
    if rng is None:
        rng = np.random.default_rng()
    D_bounded = 0.3 * np.tanh(D)
    epsilon = rng.normal(0, epsilon_scale, size=X.shape)
    X_new = X + D_bounded + epsilon
    return np.clip(X_new, 0.0, 1.0)

# ==========================================
# Simulation experiments
# ==========================================

def simulate_stability(dynamics_fn, dynamics_kwargs, n_turns=200, n_trials=50,
                       perturbation_turn=50, perturbation_strength=0.8,
                       rng_seed=42):
    """
    Simulate state trajectories and measure:
    - Variance (stability)
    - Recovery time after large perturbation at turn 50
    - Max drift from baseline
    """
    rng = np.random.default_rng(rng_seed)
    DIM = 5
    MU = np.array([0.5, 0.5, 0.5, 0.5, 0.5])  # trait anchor

    all_variance = []
    all_recovery_times = []
    all_max_drifts = []

    for trial in range(n_trials):
        X = deepcopy(MU)
        trajectory = [X.copy()]
        recovered = False
        recovery_time = n_turns  # default: never recovered

        for t in range(n_turns):
            # Random impulse (simulate event impact)
            D = rng.normal(0, 0.15, size=DIM)

            # Apply large perturbation at turn 50
            if t == perturbation_turn:
                D += np.ones(DIM) * perturbation_strength

            X = dynamics_fn(X, D, **dynamics_kwargs, rng=rng)
            trajectory.append(X.copy())

            # Check recovery: within 0.15 of mu after perturbation
            if t > perturbation_turn and not recovered:
                if np.linalg.norm(X - MU) < 0.15 * math.sqrt(DIM):
                    recovery_time = t - perturbation_turn
                    recovered = True

        traj = np.array(trajectory)
        # Variance computed over post-perturbation turns
        post_pert = traj[perturbation_turn:]
        variance = float(np.mean(np.var(post_pert, axis=0)))
        max_drift = float(np.max(np.abs(traj - MU)))
        all_variance.append(variance)
        all_recovery_times.append(recovery_time)
        all_max_drifts.append(max_drift)

    return {
        'variance_mean': float(np.mean(all_variance)),
        'variance_std': float(np.std(all_variance)),
        'recovery_time_mean': float(np.mean(all_recovery_times)),
        'recovery_time_std': float(np.std(all_recovery_times)),
        'max_drift_mean': float(np.mean(all_max_drifts)),
        'max_drift_std': float(np.std(all_max_drifts)),
        'n_trials': n_trials,
        'n_turns': n_turns,
        'perturbation_turn': perturbation_turn,
        'perturbation_strength': perturbation_strength,
    }


def simulate_persona_break_rate(dynamics_fn, dynamics_kwargs, trait_anchor,
                                n_scenarios=100, rng_seed=99):
    """
    Simulate conversation scenarios and measure persona break rate:
    A 'persona break' is when the current state deviates >0.4 from the trait anchor
    on any dimension AND the system generates a response that violates the stored preference.
    Uses a simple rule-based proxy (no LLM inference needed).
    """
    rng = np.random.default_rng(rng_seed)
    DIM = 5
    MU = np.array(trait_anchor)
    breaks = 0
    total_queries = 0

    for scenario in range(n_scenarios):
        X = deepcopy(MU)
        # Simulate 30-turn conversation with increasing stress
        for t in range(30):
            # Simulate stress events of varying strength
            stress = rng.uniform(-0.3, 0.5)
            D = rng.normal(stress * 0.2, 0.1, size=DIM)
            X = dynamics_fn(X, D, **dynamics_kwargs, rng=rng)

        # Query at end: check if state is consistent with trait
        # Proxy: persona break if any dim drifted > threshold from trait
        drift = np.abs(X - MU)
        max_drift = float(np.max(drift))
        # With high drift, simulate retrieval inconsistency
        persona_break = (max_drift > 0.35)
        total_queries += 1
        if persona_break:
            breaks += 1

    return {
        'persona_break_rate': round(100.0 * breaks / total_queries, 1),
        'breaks': breaks,
        'total': total_queries,
    }


def run_ou_vs_ema():
    print('=' * 60)
    print('OU vs EMA State Dynamics Comparison')
    print('=' * 60)

    trait_anchor = [0.6, 0.4, 0.5, 0.55, 0.45]  # typical Big-5

    CONFIGS = {
        'OU (θ=0.5, Dmax=0.3)': {
            'fn': ou_update,
            'kwargs': {'mu_anchor': np.array(trait_anchor), 'theta': 0.5}
        },
        'OU (θ=0.3, Dmax=0.3)': {
            'fn': ou_update,
            'kwargs': {'mu_anchor': np.array(trait_anchor), 'theta': 0.3}
        },
        'OU (θ=0.7, Dmax=0.3)': {
            'fn': ou_update,
            'kwargs': {'mu_anchor': np.array(trait_anchor), 'theta': 0.7}
        },
        'EMA (α=0.1)': {
            'fn': ema_update,
            'kwargs': {'alpha': 0.1}
        },
        'EMA (α=0.3)': {
            'fn': ema_update,
            'kwargs': {'alpha': 0.3}
        },
        'Clamped EMA (α=0.2)': {
            'fn': clamped_ema_update,
            'kwargs': {'alpha': 0.2}
        },
        'Direct (no dynamics)': {
            'fn': direct_impulse,
            'kwargs': {}
        },
    }

    results = {}
    for name, cfg in CONFIGS.items():
        print(f'\nRunning: {name}')
        stability = simulate_stability(
            cfg['fn'], cfg['kwargs'],
            n_turns=300, n_trials=100,
            perturbation_turn=100, perturbation_strength=0.8
        )
        persona = simulate_persona_break_rate(
            cfg['fn'], cfg['kwargs'],
            trait_anchor=trait_anchor, n_scenarios=200
        )
        results[name] = {
            'stability': stability,
            'persona': persona
        }
        print(f'  Variance: {stability["variance_mean"]:.4f} ± {stability["variance_std"]:.4f}')
        print(f'  Recovery time: {stability["recovery_time_mean"]:.1f} ± {stability["recovery_time_std"]:.1f} turns')
        print(f'  Max drift: {stability["max_drift_mean"]:.3f} ± {stability["max_drift_std"]:.3f}')
        print(f'  Persona break rate: {persona["persona_break_rate"]}%')

    # Summary table
    print('\n' + '=' * 90)
    print(f'{"Method":<30} {"Variance↓":>10} {"RecovTime↓":>12} {"MaxDrift↓":>10} {"BreakRate↓":>12}')
    print('=' * 90)
    for name, r in results.items():
        s = r['stability']
        p = r['persona']
        print(f'{name:<30} {s["variance_mean"]:>10.4f} {s["recovery_time_mean"]:>12.1f} '
              f'{s["max_drift_mean"]:>10.3f} {p["persona_break_rate"]:>11.1f}%')

    out = OUT_DIR / 'ou_vs_ema_results.json'
    out.write_text(json.dumps(results, indent=2, default=str))
    print(f'\nSaved to {out}')
    return results


if __name__ == '__main__':
    run_ou_vs_ema()
