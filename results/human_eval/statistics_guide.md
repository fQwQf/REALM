# Statistics Guide / 统计指南

## Collect rater data in this format / 收集格式

Create `results/human_eval/rater_data.json` as a list of records:
```json
[
  {"rater_id": "R01", "pair_id": 1, "consistency_left": 4, "consistency_right": 3,
   "naturalness_left": 4, "naturalness_right": 3, "preference": "left"},
  ...
]
```
preference must be one of: "left", "right", "tie"

## Compute paper statistics / 计算论文统计数据

```python
import json, numpy as np
from scipy import stats

with open('results/human_eval/answer_key.json') as f:
    key = {item['pair_id']: item for item in json.load(f)}
with open('results/human_eval/rater_data.json') as f:
    rater_data = json.load(f)

tc, vc, tn, vn, prefs = [], [], [], [], []
for r in rater_data:
    k = key[r['pair_id']]
    if k['left_is'] == 'tempo':
        tc.append(r['consistency_left']); vc.append(r['consistency_right'])
        tn.append(r['naturalness_left']);  vn.append(r['naturalness_right'])
        prefs.append({'left': 1, 'right': 0, 'tie': 0.5}[r['preference']])
    else:
        tc.append(r['consistency_right']); vc.append(r['consistency_left'])
        tn.append(r['naturalness_right']);  vn.append(r['naturalness_left'])
        prefs.append({'right': 1, 'left': 0, 'tie': 0.5}[r['preference']])

tc, vc, tn, vn = map(np.array, [tc, vc, tn, vn])
print(f'TEMPO Consistency:   {tc.mean():.2f} +/- {tc.std():.2f}')
print(f'Vanilla Consistency: {vc.mean():.2f} +/- {vc.std():.2f}')
t, p = stats.ttest_rel(tc, vc)
d = (tc-vc).mean() / (tc-vc).std()
print(f'  paired t={t:.3f}, p={p:.4f}, Cohen d={d:.2f}')
print()
print(f'TEMPO Naturalness:   {tn.mean():.2f} +/- {tn.std():.2f}')
print(f'Vanilla Naturalness: {vn.mean():.2f} +/- {vn.std():.2f}')
t, p = stats.ttest_rel(tn, vn)
d = (tn-vn).mean() / (tn-vn).std()
print(f'  paired t={t:.3f}, p={p:.4f}, Cohen d={d:.2f}')
print()
p_arr = np.array(prefs)
wins = (p_arr == 1).sum(); losses = (p_arr == 0).sum()
win_rate = wins / len(p_arr) * 100
print(f'TEMPO Win Rate: {wins}/{len(p_arr)} = {win_rate:.1f}%')
from scipy.stats import binomtest
res = binomtest(wins, wins+losses, 0.5, alternative='greater')
print(f'  Binomial p={res.pvalue:.4f}')
# For Krippendorff alpha: pip install krippendorff
```

## Minimum viable study / 最小可行研究规模

- 20 pairs x 3 raters = 60 ratings (minimum)
- 20 pairs x 5 raters = 100 ratings (recommended for p<0.05)
- Each rater takes ~20-30 minutes for 20 items
- 每位评估者完成 20 道题约需 20-30 分钟
