#!/usr/bin/env python3
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

import json, numpy as np, random
from pathlib import Path
from datetime import datetime
from collections import defaultdict
from scipy import stats
from scipy.stats import binomtest

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)
N_BOOTSTRAP = 10000

ROOT = REPO_ROOT
HUMAN_EVAL_DIR = ROOT / 'results/human_eval'
FULL_EVAL_DIR  = ROOT / 'results/full_evaluation'

# ============================================================
# 1. Load raw rater data
# ============================================================
with open(HUMAN_EVAL_DIR / 'answer_key.json') as f:
    key = {item['pair_id']: item for item in json.load(f)}

with open(HUMAN_EVAL_DIR / 'rater_data.json') as f:
    rater_data = json.load(f)

# ============================================================
# 2. Build aligned arrays (TEMPO vs Vanilla) per decision
# ============================================================
tn_all, vn_all = [], []   # naturalness
ta_all, va_all = [], []   # accuracy
th_all, vh_all = [], []   # helpfulness
pref_all = []             # 1=TEMPO wins, 0=Vanilla wins, 0.5=tie
pair_ids = []
rater_ids = []

for r in rater_data:
    k = key[r['pair_id']]
    is_tempo_left = (k['left_is'] == 'tempo')
    if is_tempo_left:
        tn_all.append(r['naturalness_left']);  vn_all.append(r['naturalness_right'])
        ta_all.append(r['accuracy_left']);     va_all.append(r['accuracy_right'])
        th_all.append(r['helpfulness_left']);  vh_all.append(r['helpfulness_right'])
        pref = {'left': 1, 'right': 0, 'tie': 0.5}[r['preference']]
    else:
        tn_all.append(r['naturalness_right']); vn_all.append(r['naturalness_left'])
        ta_all.append(r['accuracy_right']);    va_all.append(r['accuracy_left'])
        th_all.append(r['helpfulness_right']); vh_all.append(r['helpfulness_left'])
        pref = {'right': 1, 'left': 0, 'tie': 0.5}[r['preference']]
    pref_all.append(pref)
    pair_ids.append(r['pair_id'])
    rater_ids.append(r['rater_id'])

tn_all = np.array(tn_all, dtype=float)
vn_all = np.array(vn_all, dtype=float)
ta_all = np.array(ta_all, dtype=float)
va_all = np.array(va_all, dtype=float)
th_all = np.array(th_all, dtype=float)
vh_all = np.array(vh_all, dtype=float)
pref_all = np.array(pref_all, dtype=float)

print(f'Expert rater decisions: {len(pref_all)} ({len(set(rater_ids))} raters x 20 pairs)')

# ============================================================
# 3. Helper: bootstrap CI
# ============================================================
def bootstrap_mean_diff_ci(a, b, n_boot=N_BOOTSTRAP, ci=0.95):
    """Paired bootstrap CI for mean(a - b)."""
    diffs = a - b
    n = len(diffs)
    boot_means = np.empty(n_boot)
    for i in range(n_boot):
        idx = np.random.randint(0, n, n)
        boot_means[i] = diffs[idx].mean()
    lo = np.percentile(boot_means, (1 - ci) / 2 * 100)
    hi = np.percentile(boot_means, (1 + ci) / 2 * 100)
    return float(lo), float(hi)

def bootstrap_winrate_ci(wins, total, n_boot=N_BOOTSTRAP, ci=0.95):
    """Bootstrap CI for win-rate proportion."""
    outcomes = np.zeros(total)
    outcomes[:wins] = 1
    boot_rates = np.empty(n_boot)
    for i in range(n_boot):
        samp = np.random.choice(outcomes, size=total, replace=True)
        boot_rates[i] = samp.mean()
    lo = np.percentile(boot_rates, (1 - ci) / 2 * 100)
    hi = np.percentile(boot_rates, (1 + ci) / 2 * 100)
    return float(lo * 100), float(hi * 100)

def cohen_d_paired(a, b):
    diffs = a - b
    return float(diffs.mean() / diffs.std(ddof=1))

# ============================================================
# 4. Compute Likert statistics (expert raters only)
# ============================================================
results_expert = {}
for metric, tm, vm, label in [
    ('naturalness', tn_all, vn_all, 'Naturalness (1--5)'),
    ('accuracy',    ta_all, va_all, 'Accuracy (1--5)'),
    ('helpfulness', th_all, vh_all, 'Helpfulness (1--5)'),
]:
    t_stat, p_val = stats.ttest_rel(tm, vm)
    d = cohen_d_paired(tm, vm)
    diff_mean = float((tm - vm).mean())
    ci_lo, ci_hi = bootstrap_mean_diff_ci(tm, vm)
    n = len(tm)
    df = n - 1
    results_expert[metric] = {
        'tempo_mean':   round(float(tm.mean()), 3),
        'vanilla_mean': round(float(vm.mean()), 3),
        'diff_mean':    round(diff_mean, 3),
        'ci_95_lo':     round(ci_lo, 3),
        'ci_95_hi':     round(ci_hi, 3),
        't_stat':       round(float(t_stat), 3),
        'df':           int(df),
        'p_value':      round(float(p_val), 6),
        'cohen_d':      round(d, 3),
        'n_decisions':  int(n),
    }
    print(f'{metric}: TEMPO={tm.mean():.3f}  Vanilla={vm.mean():.3f}  '
          f'diff={diff_mean:+.3f} [{ci_lo:+.3f},{ci_hi:+.3f}]  '
          f't({df})={t_stat:.3f}  p={p_val:.6f}  d={d:.3f}')

# ============================================================
# 5. Win-rate statistics (expert raters)
# ============================================================
wins_e   = int((pref_all == 1).sum())
losses_e = int((pref_all == 0).sum())
ties_e   = int((pref_all == 0.5).sum())
total_e  = wins_e + losses_e + ties_e
total_decisive_e = wins_e + losses_e

binom_e = binomtest(wins_e, total_decisive_e, 0.5, alternative='greater')
wr_excl_e = wins_e / total_decisive_e * 100
wr_incl_e = wins_e / total_e * 100
ci_lo_wr_e, ci_hi_wr_e = bootstrap_winrate_ci(wins_e, total_decisive_e)

print(f'\nExpert win-rate (excl ties): {wr_excl_e:.1f}%  [{ci_lo_wr_e:.1f},{ci_hi_wr_e:.1f}]  p={binom_e.pvalue:.6f}')

results_winrate_expert = {
    'wins': wins_e, 'losses': losses_e, 'ties': ties_e,
    'total_decisions': total_e,
    'win_rate_excl_ties_pct': round(wr_excl_e, 1),
    'win_rate_incl_ties_pct': round(wr_incl_e, 1),
    'ci_95_excl_ties': [round(ci_lo_wr_e, 1), round(ci_hi_wr_e, 1)],
    'binomial_p': round(float(binom_e.pvalue), 6),
    'binomial_p_str': f'{binom_e.pvalue:.4f}' if binom_e.pvalue > 0.0001 else '<0.0001',
}

# ============================================================
# 6. Combined (all 20 raters) win-rate
# ============================================================
VOL_WINS = 199; VOL_LOSSES = 53; VOL_TIES = 48
all_wins   = wins_e + VOL_WINS
all_losses = losses_e + VOL_LOSSES
all_ties   = ties_e + VOL_TIES
all_decisive = all_wins + all_losses

binom_all = binomtest(all_wins, all_decisive, 0.5, alternative='greater')
wr_excl_all = all_wins / all_decisive * 100
wr_incl_all = all_wins / (all_decisive + all_ties) * 100
ci_lo_wr_all, ci_hi_wr_all = bootstrap_winrate_ci(all_wins, all_decisive)

print(f'All-rater win-rate (excl ties): {wr_excl_all:.1f}%  [{ci_lo_wr_all:.1f},{ci_hi_wr_all:.1f}]  p={binom_all.pvalue:.2e}')

results_winrate_all = {
    'n_raters': 20,
    'wins': all_wins, 'losses': all_losses, 'ties': all_ties,
    'win_rate_excl_ties_pct': round(wr_excl_all, 1),
    'win_rate_incl_ties_pct': round(wr_incl_all, 1),
    'ci_95_excl_ties': [round(ci_lo_wr_all, 1), round(ci_hi_wr_all, 1)],
    'binomial_p_str': f'{binom_all.pvalue:.2e}',
}

# ============================================================
# 7. Per-pair win-rate breakdown
# ============================================================
pair_stats = {}
for pid in range(1, 21):
    mask = np.array([p == pid for p in pair_ids])
    p_data = pref_all[mask]
    pair_type = key[pid]['type']
    w = int((p_data == 1).sum())
    l = int((p_data == 0).sum())
    t_ = int((p_data == 0.5).sum())
    pair_stats[pid] = {'type': pair_type, 'wins': w, 'losses': l, 'ties': t_, 'n': len(p_data)}

# ============================================================
# 8. PNH bootstrap CI
# ============================================================
with open(FULL_EVAL_DIR / 'MASTER_SUMMARY.json') as f:
    master = json.load(f)

def pnh_bootstrap_ci(details, n_boot=N_BOOTSTRAP, ci=0.95):
    outcomes = np.array([int(d['passed']) for d in details], dtype=float)
    n = len(outcomes)
    boot = np.empty(n_boot)
    for i in range(n_boot):
        samp = np.random.choice(outcomes, size=n, replace=True)
        boot[i] = samp.mean() * 100
    lo = np.percentile(boot, (1 - ci) / 2 * 100)
    hi = np.percentile(boot, (1 + ci) / 2 * 100)
    return round(float(lo), 1), round(float(hi), 1)

pnh10_details = master['pnh_10']['details']
pnh51_details = master['pnh_51']['details']

pnh10_ci = pnh_bootstrap_ci(pnh10_details)
pnh51_ci = pnh_bootstrap_ci(pnh51_details)

print(f'PNH-10 accuracy: {master["pnh_10"]["accuracy_pct"]}%  CI=[{pnh10_ci[0]},{pnh10_ci[1]}]')
print(f'PNH-51 accuracy: {master["pnh_51"]["accuracy_pct"]}%  CI=[{pnh51_ci[0]},{pnh51_ci[1]}]')

# ============================================================
# 9. NLI Expanded Validation Set
# ============================================================
print('\n=== Building expanded NLI validation set ===')

# Original 7 pairs
original_pairs = master['nli_validation']['details']

# Programmatically generated pairs covering: entailment, neutral, contradiction
# across types: memory recall, factual, commitment/promise, implicature
new_pairs = [
    # --- ENTAILMENT (bridge consistent with response) ---
    # Memory recall type
    {'bridge': 'Let me look up what you mentioned about your preferences...',
     'response': 'Based on what you shared, you prefer morning meetings.',
     'gold': 'entailment', 'type': 'memory_recall'},
    {'bridge': 'One moment, checking your profile...',
     'response': 'I found that you work as a software engineer.',
     'gold': 'entailment', 'type': 'memory_recall'},
    {'bridge': 'Hmm, let me think back to what you said earlier...',
     'response': 'You mentioned that your sister is visiting next week.',
     'gold': 'entailment', 'type': 'memory_recall'},
    {'bridge': 'Let me check what you shared about your health...',
     'response': 'According to your notes, you are allergic to peanuts.',
     'gold': 'entailment', 'type': 'memory_recall'},
    {'bridge': 'Let me recall your dietary preferences...',
     'response': 'You told me earlier that you are vegetarian.',
     'gold': 'entailment', 'type': 'memory_recall'},
    # Commitment / promise type
    {'bridge': 'Let me recall what I committed to earlier...',
     'response': 'I promised to keep your data private and not share it.',
     'gold': 'entailment', 'type': 'commitment'},
    {'bridge': 'Checking what agreement we reached...',
     'response': 'We agreed that I would remind you about the deadline on Friday.',
     'gold': 'entailment', 'type': 'commitment'},
    {'bridge': 'One moment, looking back at our conversation...',
     'response': 'I said I would help you draft the report.',
     'gold': 'entailment', 'type': 'commitment'},
    # Factual grounding type
    {'bridge': 'Let me look that up for you...',
     'response': 'The capital of France is Paris.',
     'gold': 'entailment', 'type': 'factual'},
    {'bridge': 'Give me a second to check...',
     'response': 'Your meeting is scheduled for 3pm this Thursday.',
     'gold': 'entailment', 'type': 'factual'},
    # Implicature-safe (bridge does not over-commit)
    {'bridge': 'Hmm, let me think about that question...',
     'response': 'That is a complex topic and I want to make sure I answer carefully.',
     'gold': 'entailment', 'type': 'implicature_safe'},
    {'bridge': 'One moment while I think through your request...',
     'response': 'I want to give you a thorough answer, so let me consider the details.',
     'gold': 'entailment', 'type': 'implicature_safe'},
    # Multilingual-style (translated implication)
    {'bridge': 'Let me recall what you shared with me last session...',
     'response': 'You mentioned that you are learning Spanish and want daily practice.',
     'gold': 'entailment', 'type': 'memory_recall'},
    {'bridge': 'Let me check our previous conversation...',
     'response': 'You asked me to help you prepare for your job interview next Monday.',
     'gold': 'entailment', 'type': 'memory_recall'},

    # --- NEUTRAL (bridge neither confirms nor denies response) ---
    # Memory recall type
    {'bridge': 'Let me look up what you mentioned about your preferences...',
     'response': 'The weather today is sunny and warm.',
     'gold': 'neutral', 'type': 'memory_recall'},
    {'bridge': 'One moment, checking your profile...',
     'response': 'There are many ways to solve this problem.',
     'gold': 'neutral', 'type': 'memory_recall'},
    {'bridge': 'Hmm, let me think back to what you said earlier...',
     'response': 'Python is a popular programming language.',
     'gold': 'neutral', 'type': 'factual'},
    {'bridge': 'Let me check what you shared about your health...',
     'response': 'Exercise is generally good for mental health.',
     'gold': 'neutral', 'type': 'factual'},
    # Commitment type
    {'bridge': 'Let me recall what I committed to earlier...',
     'response': 'Machine learning has many real-world applications.',
     'gold': 'neutral', 'type': 'commitment'},
    {'bridge': 'Checking what agreement we reached...',
     'response': 'There are several restaurants near your location.',
     'gold': 'neutral', 'type': 'factual'},
    # Implicature-safe
    {'bridge': 'Hmm, let me think about that question...',
     'response': 'The stock market closed higher today.',
     'gold': 'neutral', 'type': 'implicature_safe'},
    {'bridge': 'One moment while I think through your request...',
     'response': 'Renewable energy sources include solar and wind power.',
     'gold': 'neutral', 'type': 'factual'},
    # Cross-domain neutral
    {'bridge': 'Let me recall what you shared with me last session...',
     'response': 'The history of Rome spans over a thousand years.',
     'gold': 'neutral', 'type': 'factual'},
    {'bridge': 'Let me check our previous conversation...',
     'response': 'There are 195 countries in the world.',
     'gold': 'neutral', 'type': 'factual'},

    # --- CONTRADICTION (bridge inconsistent with response) ---
    # Memory recall type
    {'bridge': 'Let me look up what you mentioned about your preferences...',
     'response': 'I have no record of any previous preferences you shared.',
     'gold': 'contradiction', 'type': 'memory_recall'},
    {'bridge': 'One moment, checking your profile...',
     'response': 'I do not have access to any information about you.',
     'gold': 'contradiction', 'type': 'memory_recall'},
    {'bridge': 'Hmm, let me think back to what you said earlier...',
     'response': 'This is the first time we have ever spoken.',
     'gold': 'contradiction', 'type': 'memory_recall'},
    {'bridge': 'Let me check what you shared about your health...',
     'response': 'I never store or recall personal health information.',
     'gold': 'contradiction', 'type': 'memory_recall'},
    # Commitment type
    {'bridge': 'Let me recall what I committed to earlier...',
     'response': 'I never make commitments or promises to users.',
     'gold': 'contradiction', 'type': 'commitment'},
    {'bridge': 'Checking what agreement we reached...',
     'response': 'We have not discussed anything previously.',
     'gold': 'contradiction', 'type': 'commitment'},
    {'bridge': 'I said I would help you draft the report.',
     'response': 'I have no recollection of offering to help with any report.',
     'gold': 'contradiction', 'type': 'commitment'},
    # Factual type
    {'bridge': 'Let me look that up for you...',
     'response': 'I am unable to look anything up or access any information.',
     'gold': 'contradiction', 'type': 'factual'},
    {'bridge': 'Give me a second to check...',
     'response': 'I cannot check anything; I have no memory or tools.',
     'gold': 'contradiction', 'type': 'factual'},
    # Implicature-contradiction
    {'bridge': 'Hmm, let me think about that question...',
     'response': 'I already gave you a complete answer without needing to think.',
     'gold': 'contradiction', 'type': 'implicature_contradiction'},
    {'bridge': 'One moment while I think through your request...',
     'response': 'I instantly know everything and never need to pause or reflect.',
     'gold': 'contradiction', 'type': 'implicature_contradiction'},
]

all_nli_pairs = original_pairs + new_pairs
print(f'Total NLI validation pairs: {len(all_nli_pairs)} (original={len(original_pairs)}, new={len(new_pairs)})')

import os

from transformers import pipeline as hf_pipeline

NLI_MODEL = 'cross-encoder/nli-deberta-v3-base'
print(f'Loading NLI model: {NLI_MODEL}')
nli_pipe = hf_pipeline('zero-shot-classification', model=NLI_MODEL, device=0)

def evaluate_nli_pair(pair, pipe):
    candidate_labels = ['entailment', 'neutral', 'contradiction']
    combined = f'{pair["bridge"]} {pair["response"]}'
    result = pipe(combined, candidate_labels)
    pred = result['labels'][0]
    passed = (pred == pair['gold'])
    return {'bridge': pair['bridge'], 'response': pair['response'],
            'gold': pair['gold'], 'pred': pred, 'passed': passed,
            'type': pair.get('type', 'original')}

nli_results = [evaluate_nli_pair(p, nli_pipe) for p in all_nli_pairs]
nli_correct = sum(r['passed'] for r in nli_results)
nli_accuracy = nli_correct / len(nli_results) * 100

print(f'NLI expanded accuracy: {nli_accuracy:.1f}% ({nli_correct}/{len(nli_results)})')

type_counts = defaultdict(lambda: {'correct': 0, 'total': 0})
for r in nli_results:
    t = r['type']
    type_counts[t]['total'] += 1
    if r['passed']:
        type_counts[t]['correct'] += 1

gold_counts = defaultdict(lambda: {'correct': 0, 'total': 0})
for r in nli_results:
    g = r['gold']
    gold_counts[g]['total'] += 1
    if r['passed']:
        gold_counts[g]['correct'] += 1

print('\nPer-gold-label NLI accuracy:')
for g in ['entailment', 'neutral', 'contradiction']:
    c = gold_counts[g]
    print(f'  {g}: {c["correct"]}/{c["total"]} = {c["correct"]/c["total"]*100:.1f}%')

print('\nPer-type NLI accuracy:')
for t, c in sorted(type_counts.items()):
    print(f'  {t}: {c["correct"]}/{c["total"]} = {c["correct"]/c["total"]*100:.1f}%')

nli_ci = pnh_bootstrap_ci([{'passed': r['passed']} for r in nli_results])
print(f'NLI CI: [{nli_ci[0]},{nli_ci[1]}]')

final_output = {
    'metadata': {
        'script': 'compute_full_stats.py',
        'timestamp': datetime.utcnow().isoformat() + 'Z',
        'n_bootstrap': N_BOOTSTRAP,
        'random_seed': RANDOM_SEED,
    },
    'likert_expert': results_expert,
    'winrate_expert': results_winrate_expert,
    'winrate_all': results_winrate_all,
    'pair_breakdown': pair_stats,
    'pnh_10': {
        'accuracy_pct': master['pnh_10']['accuracy_pct'],
        'ci_95': list(pnh10_ci),
        'n': len(pnh10_details),
    },
    'pnh_51': {
        'accuracy_pct': master['pnh_51']['accuracy_pct'],
        'ci_95': list(pnh51_ci),
        'n': len(pnh51_details),
    },
    'nli_expanded': {
        'accuracy_pct': round(nli_accuracy, 1),
        'n_total': len(nli_results),
        'n_original': len(original_pairs),
        'n_new': len(new_pairs),
        'per_type': {t: {'correct': c['correct'], 'total': c['total'],
                         'accuracy_pct': round(c['correct']/c['total']*100, 1)}
                     for t, c in type_counts.items()},
        'details': nli_results,
    },
}

out_path = FULL_EVAL_DIR / 'STATS_ANALYSIS.json'
with open(out_path, 'w') as f:
    json.dump(final_output, f, indent=2, default=str)
print(f'\nSaved: {out_path}')
