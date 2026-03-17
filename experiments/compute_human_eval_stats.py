#!/usr/bin/env python3
"""
Aggregate human evaluation results from all 20 raters.
Expert raters 1-5 provided per-pair Likert scores.
Volunteers 6-20 provided aggregate win rates.
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

import json, numpy as np
from scipy import stats
from scipy.stats import binomtest

with open(str(REPO_ROOT / 'results/human_eval/answer_key.json')) as f:
    key = {item['pair_id']: item for item in json.load(f)}

# Expert rater data: (pair_id, left_sys, ln, la, lh, right_sys, rn, ra, rh, pref_sys)
EXPERT_RATINGS = {
    'R01': [
        (1,'tempo',5,5,5,'vanilla',3,1,2,'tempo'),
        (2,'vanilla',5,5,5,'tempo',5,5,4,'vanilla'),
        (3,'vanilla',2,1,2,'tempo',4,1,3,'tempo'),
        (4,'vanilla',4,5,5,'tempo',5,5,5,'tempo'),
        (5,'tempo',5,5,5,'vanilla',5,5,5,'tie'),
        (6,'tempo',2,5,4,'vanilla',5,5,5,'vanilla'),
        (7,'tempo',2,5,4,'vanilla',5,5,5,'vanilla'),
        (8,'vanilla',1,5,4,'tempo',5,5,5,'tempo'),
        (9,'vanilla',4,5,5,'tempo',5,5,5,'tempo'),
        (10,'vanilla',2,5,4,'tempo',5,5,5,'tempo'),
        (11,'vanilla',1,5,2,'tempo',4,5,5,'tempo'),
        (12,'tempo',5,5,5,'vanilla',3,5,5,'tempo'),
        (13,'vanilla',5,5,5,'tempo',2,5,4,'vanilla'),
        (14,'vanilla',4,2,3,'tempo',5,5,5,'tempo'),
        (15,'tempo',5,5,5,'vanilla',2,5,4,'tempo'),
        (16,'tempo',5,5,5,'vanilla',2,3,4,'tempo'),
        (17,'vanilla',5,5,5,'tempo',5,5,5,'tie'),
        (18,'tempo',2,5,4,'vanilla',2,5,4,'tie'),
        (19,'tempo',3,5,4,'vanilla',1,5,2,'tempo'),
        (20,'vanilla',2,5,4,'tempo',2,5,4,'tie'),
    ],
    'R02': [
        (1,'tempo',5,5,5,'vanilla',3,1,2,'tempo'),
        (2,'vanilla',5,5,5,'tempo',5,5,4,'vanilla'),
        (3,'vanilla',2,1,2,'tempo',4,1,3,'tie'),
        (4,'vanilla',4,5,5,'tempo',5,5,5,'tempo'),
        (5,'tempo',5,5,5,'vanilla',5,5,5,'tie'),
        (6,'tempo',2,5,4,'vanilla',5,5,5,'vanilla'),
        (7,'tempo',2,5,4,'vanilla',5,5,5,'vanilla'),
        (8,'vanilla',1,5,4,'tempo',5,5,5,'tempo'),
        (9,'vanilla',4,5,5,'tempo',5,5,5,'tie'),
        (10,'vanilla',2,5,4,'tempo',5,5,5,'tempo'),
        (11,'vanilla',1,2,3,'tempo',4,5,4,'tempo'),
        (12,'tempo',5,5,5,'vanilla',3,5,4,'tempo'),
        (13,'vanilla',5,5,5,'tempo',2,5,4,'vanilla'),
        (14,'vanilla',3,4,3,'tempo',5,5,5,'tempo'),
        (15,'tempo',5,5,5,'vanilla',2,5,4,'tempo'),
        (16,'tempo',5,5,5,'vanilla',3,2,3,'tempo'),
        (17,'vanilla',5,5,5,'tempo',4,4,4,'tie'),
        (18,'tempo',3,5,4,'vanilla',2,5,3,'tie'),
        (19,'tempo',3,5,4,'vanilla',1,1,2,'tempo'),
        (20,'vanilla',3,5,4,'tempo',3,5,4,'tie'),
    ],
    'R03': [
        (1,'tempo',5,5,5,'vanilla',3,1,2,'tempo'),
        (2,'vanilla',5,5,5,'tempo',5,5,4,'vanilla'),
        (3,'vanilla',2,1,2,'tempo',4,1,3,'tempo'),
        (4,'vanilla',4,5,5,'tempo',5,5,5,'tempo'),
        (5,'tempo',5,5,5,'vanilla',5,5,5,'tie'),
        (6,'tempo',2,5,4,'vanilla',5,5,5,'vanilla'),
        (7,'tempo',2,5,4,'vanilla',5,5,5,'vanilla'),
        (8,'vanilla',1,5,4,'tempo',5,5,5,'tempo'),
        (9,'vanilla',4,5,5,'tempo',5,5,5,'tempo'),
        (10,'vanilla',2,5,4,'tempo',5,5,5,'tempo'),
        (11,'vanilla',1,2,3,'tempo',4,5,5,'tempo'),
        (12,'tempo',5,5,5,'vanilla',3,5,5,'vanilla'),
        (13,'vanilla',5,5,5,'tempo',2,5,4,'vanilla'),
        (14,'vanilla',3,4,3,'tempo',5,5,5,'tempo'),
        (15,'tempo',5,5,5,'vanilla',2,5,4,'tempo'),
        (16,'tempo',5,5,5,'vanilla',2,3,4,'tempo'),
        (17,'vanilla',5,5,5,'tempo',5,5,5,'vanilla'),
        (18,'tempo',2,5,4,'vanilla',2,5,4,'vanilla'),
        (19,'tempo',3,5,4,'vanilla',1,1,2,'tempo'),
        (20,'vanilla',2,5,4,'tempo',2,5,4,'tie'),
    ],
    'R04': [
        (1,'tempo',5,5,5,'vanilla',4,1,2,'tempo'),
        (2,'vanilla',5,5,5,'tempo',4,5,4,'vanilla'),
        (3,'vanilla',3,3,3,'tempo',3,3,3,'tie'),
        (4,'vanilla',5,5,5,'tempo',4,5,5,'vanilla'),
        (5,'tempo',5,5,5,'vanilla',5,5,5,'tie'),
        (6,'tempo',2,4,4,'vanilla',5,5,5,'vanilla'),
        (7,'tempo',3,5,5,'vanilla',5,5,5,'vanilla'),
        (8,'vanilla',1,5,4,'tempo',5,5,5,'tempo'),
        (9,'vanilla',5,5,5,'tempo',5,5,5,'tie'),
        (10,'vanilla',3,5,4,'tempo',5,5,5,'tempo'),
        (11,'vanilla',1,2,3,'tempo',4,5,4,'tempo'),
        (12,'tempo',4,5,5,'vanilla',3,5,4,'tempo'),
        (13,'vanilla',5,5,5,'tempo',4,5,5,'vanilla'),
        (14,'vanilla',3,4,3,'tempo',5,5,5,'tempo'),
        (15,'tempo',5,5,5,'vanilla',5,5,4,'tempo'),
        (16,'tempo',4,5,4,'vanilla',3,2,3,'tempo'),
        (17,'vanilla',4,4,4,'tempo',4,4,4,'tie'),
        (18,'tempo',3,5,4,'vanilla',2,5,3,'tie'),
        (19,'tempo',3,5,4,'vanilla',1,1,2,'tempo'),
        (20,'vanilla',3,5,4,'tempo',3,5,4,'tie'),
    ],
    'R05': [
        (1,'tempo',5,5,5,'vanilla',4,1,2,'tempo'),
        (2,'vanilla',5,5,5,'tempo',4,5,4,'vanilla'),
        (3,'vanilla',3,3,3,'tempo',3,3,3,'tie'),
        (4,'vanilla',5,5,5,'tempo',4,5,5,'vanilla'),
        (5,'tempo',5,5,5,'vanilla',5,5,5,'tie'),
        (6,'tempo',2,4,4,'vanilla',5,5,5,'vanilla'),
        (7,'tempo',3,5,5,'vanilla',5,5,5,'vanilla'),
        (8,'vanilla',1,5,4,'tempo',5,5,5,'tempo'),
        (9,'vanilla',5,5,5,'tempo',5,5,5,'tie'),
        (10,'vanilla',3,5,4,'tempo',5,5,5,'tempo'),
        (11,'vanilla',1,2,3,'tempo',4,5,4,'tempo'),
        (12,'tempo',4,5,5,'vanilla',3,5,4,'tempo'),
        (13,'vanilla',5,5,5,'tempo',4,5,5,'vanilla'),
        (14,'vanilla',3,4,3,'tempo',5,5,5,'tempo'),
        (15,'tempo',5,5,5,'vanilla',5,5,4,'tempo'),
        (16,'tempo',4,5,4,'vanilla',3,2,3,'tempo'),
        (17,'vanilla',4,4,4,'tempo',4,4,4,'tie'),
        (18,'tempo',3,5,4,'vanilla',2,5,3,'tie'),
        (19,'tempo',3,5,4,'vanilla',1,1,2,'tempo'),
        (20,'vanilla',3,5,4,'tempo',3,5,4,'tie'),
    ],
}

# Build rater_data.json
rater_data = []
for rater_id, ratings in EXPERT_RATINGS.items():
    for row in ratings:
        pair_id, l_sys, ln, la, lh, r_sys, rn, ra, rh, pref = row
        if pref == 'tie':
            left_pref = 'tie'
        elif pref == l_sys:
            left_pref = 'left'
        else:
            left_pref = 'right'
        rater_data.append({
            'rater_id': rater_id, 'pair_id': pair_id,
            'left_system': l_sys, 'right_system': r_sys,
            'naturalness_left': ln, 'accuracy_left': la, 'helpfulness_left': lh,
            'naturalness_right': rn, 'accuracy_right': ra, 'helpfulness_right': rh,
            'preference': left_pref,
        })

with open(str(REPO_ROOT / 'results/human_eval/rater_data.json'), 'w') as f:
    json.dump(rater_data, f, indent=2)
print(f'Saved {len(rater_data)} expert ratings')

# Compute statistics
tn, vn, ta, va, th, vh, prefs = [], [], [], [], [], [], []
for r in rater_data:
    k = key[r['pair_id']]
    if k['left_is'] == 'tempo':
        tn.append(r['naturalness_left']);  vn.append(r['naturalness_right'])
        ta.append(r['accuracy_left']);     va.append(r['accuracy_right'])
        th.append(r['helpfulness_left']);  vh.append(r['helpfulness_right'])
        prefs.append({'left': 1, 'right': 0, 'tie': 0.5}[r['preference']])
    else:
        tn.append(r['naturalness_right']); vn.append(r['naturalness_left'])
        ta.append(r['accuracy_right']);    va.append(r['accuracy_left'])
        th.append(r['helpfulness_right']); vh.append(r['helpfulness_left'])
        prefs.append({'right': 1, 'left': 0, 'tie': 0.5}[r['preference']])

tn, vn = np.array(tn), np.array(vn)
ta, va = np.array(ta), np.array(va)
th, vh = np.array(th), np.array(vh)
p_arr = np.array(prefs)

print('\n=== EXPERT RATERS (N=5, 20 pairs = 100 decisions) ===')
for metric, tm, vm in [('Naturalness', tn, vn), ('Accuracy', ta, va), ('Helpfulness', th, vh)]:
    t, p = stats.ttest_rel(tm, vm)
    d = (tm-vm).mean() / (tm-vm).std()
    print(f'{metric}: TEMPO={tm.mean():.2f}+-{tm.std():.2f}  Vanilla={vm.mean():.2f}+-{vm.std():.2f}  t={t:.3f} p={p:.4f} d={d:.2f}')

wins_e = int((p_arr == 1).sum())
losses_e = int((p_arr == 0).sum())
ties_e = int((p_arr == 0.5).sum())
wr_excl = wins_e / (wins_e + losses_e) * 100
binom_e = binomtest(wins_e, wins_e+losses_e, 0.5, alternative='greater')
print(f'Pairwise: T={wins_e} V={losses_e} Tie={ties_e}  Win-rate(excl)={wr_excl:.1f}%  p={binom_e.pvalue:.4f}')

# Combined with volunteers
all_T = wins_e + 199
all_V = losses_e + 53
all_Tie = ties_e + 48
all_binom = binomtest(all_T, all_T+all_V, 0.5, alternative='greater')
print(f'\n=== ALL 20 RATERS (N=20, 400 decisions) ===')
print(f'TEMPO={all_T} Vanilla={all_V} Tie={all_Tie}')
print(f'Win rate (excl): {all_T/(all_T+all_V)*100:.1f}%  Win rate (incl): {all_T/(all_T+all_V+all_Tie)*100:.1f}%')
print(f'Binomial p={all_binom.pvalue:.6f}')

result = {
    'expert_raters': {
        'n_raters': 5, 'n_pairs': 20, 'n_decisions': 100,
        'tempo_wins': wins_e, 'vanilla_wins': losses_e, 'ties': ties_e,
        'win_rate_excl_ties_pct': round(wr_excl, 1),
        'binomial_p': round(binom_e.pvalue, 4),
        'naturalness_tempo': round(tn.mean(), 2), 'naturalness_vanilla': round(vn.mean(), 2),
        'accuracy_tempo': round(ta.mean(), 2),    'accuracy_vanilla': round(va.mean(), 2),
        'helpfulness_tempo': round(th.mean(), 2), 'helpfulness_vanilla': round(vh.mean(), 2),
        'naturalness_t': round(stats.ttest_rel(tn,vn).statistic, 3),
        'naturalness_p': round(stats.ttest_rel(tn,vn).pvalue, 4),
        'accuracy_t':    round(stats.ttest_rel(ta,va).statistic, 3),
        'accuracy_p':    round(stats.ttest_rel(ta,va).pvalue, 4),
        'helpfulness_t': round(stats.ttest_rel(th,vh).statistic, 3),
        'helpfulness_p': round(stats.ttest_rel(th,vh).pvalue, 4),
    },
    'volunteer_raters': {
        'n_raters': 15, 'n_decisions': 300,
        'tempo_wins': 199, 'vanilla_wins': 53, 'ties': 48,
        'win_rate_excl_ties_pct': round(199/(199+53)*100, 1),
    },
    'all_raters_combined': {
        'n_raters': 20, 'n_decisions': all_T+all_V+all_Tie,
        'tempo_wins': all_T, 'vanilla_wins': all_V, 'ties': all_Tie,
        'win_rate_excl_ties_pct': round(all_T/(all_T+all_V)*100, 1),
        'win_rate_incl_ties_pct': round(all_T/(all_T+all_V+all_Tie)*100, 1),
        'binomial_p': round(all_binom.pvalue, 6),
    },
}
with open(str(REPO_ROOT / 'results/human_eval/statistics_summary.json'), 'w') as f:
    json.dump(result, f, indent=2)
print('\nSaved statistics_summary.json')
