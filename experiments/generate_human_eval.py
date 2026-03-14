#!/usr/bin/env python3
"""
Generate complete human evaluation questionnaire for TEMPO paper.
Produces:
  1. questionnaire.md   -- printable bilingual questionnaire for raters
  2. eval_pairs.json    -- data file with all pairs (no method labels)
  3. answer_key.json    -- secret mapping of pair_id -> method
  4. statistics_guide.md -- how to compute all paper stats from rater data
"""
import json, random, os
random.seed(42)

# ── Load MSC experiment results ─────────────────────────────────────────
D = 'results/msc_large_scale'
with open(f'{D}/combined_results_20260226_133607.json') as f:
    all_results = json.load(f)

vanilla_data = all_results[0]   # Vanilla RAG
tempo_data   = all_results[1]   # TEMPO + Query-Type Routing

vanilla_recalls = vanilla_data['recall_results']
tempo_recalls   = tempo_data['recall_results']

# ── Build blinded pairs ──────────────────────────────────────────────────
pairs = []
for i, (v, t) in enumerate(zip(vanilla_recalls, tempo_recalls)):
    assert v['query'] == t['query']
    if random.random() < 0.5:
        left  = {'response': v['response'], 'method': 'vanilla'}
        right = {'response': t['response'], 'method': 'tempo'}
    else:
        left  = {'response': t['response'], 'method': 'tempo'}
        right = {'response': v['response'], 'method': 'vanilla'}
    pairs.append({
        'pair_id': i + 1,
        'query':      v['query'],
        'type':       v['type'],
        'difficulty': v['difficulty'],
        'left':  left,
        'right': right,
    })

# Persona background (shared across all pairs from the MSC dataset)
BACKGROUND = (
    "The AI assistant previously had several conversations with the user. "
    "Key facts told to the AI: Name=Alex, Job=Software engineer in San Francisco, "
    "Education=Stanford 2020, Food=Sushi, Hobby=Hiking, Pet=Cat named Whiskers, "
    "Wife=Sarah, Children=3, Grandson=Tommy, Allergy=Peanuts, Sport=Volleyball, "
    "Drink=Tea (switched from coffee), Recent event=Hiked at Yosemite last weekend, "
    "Favorite color=Teal, Pets also include dogs Max and Bella (second persona Maria), "
    "Signature dish=Duck confit, Career=Boeing 30 years."
)

# ── Generate questionnaire Markdown ─────────────────────────────────────
os.makedirs('results/human_eval', exist_ok=True)

lines = []
lines.append("# TEMPO Human Evaluation Questionnaire / TEMPO 人类评估问卷\n")
lines.append("## Instructions / 说明\n")
lines.append(f"You will evaluate **{len(pairs)} pairs** of AI assistant responses.\n")
lines.append("您将评估 **{n} 对** AI 助手回复，共 {n} 道题。".format(n=len(pairs)) + "\n")
lines.append("""
For each item you see:
每道题包含：

- **Background** — facts about the user that were told to the AI in earlier sessions  
  背景 — 用户在早期对话中告诉 AI 的信息
- **User Question** — what the user asked in the current session  
  用户当前提问
- **Response Left / Right** — two AI responses (randomly ordered, method hidden)  
  左/右两侧回复（随机排列，方法名隐藏）

### Rating scale / 评分标准

Rate **each response separately** (Left AND Right, independently):
请对**每个回复单独评分**（左侧和右侧分别评分）：

| Score | Persona Consistency / 人格一致性 | Naturalness / 自然度 |
|---|---|---|
| 1 | Ignores or contradicts prior facts / 完全忽略或矛盾 | Very robotic or incoherent / 非常机械或不连贯 |
| 2 | Mostly ignores context / 大部分忽略 | Mostly unnatural / 大部分不自然 |
| 3 | Partial recall, some errors / 部分记住，有错误 | Acceptable but stiff / 可接受但生硬 |
| 4 | Mostly accurate recall / 大部分准确 | Natural and fluent / 自然流畅 |
| 5 | Perfectly recalls and applies / 完美记住并运用 | Very natural, like a real person / 非常自然 |

**Overall Preference / 总体偏好**: Which response do you prefer overall?  
Options: **Left** / **Right** / **Tie (no preference)**  
你总体更喜欢哪个？选项：**左** / **右** / **平局（无偏好）**

---
""")

for p in pairs:
    lines.append(f"## Item {p['pair_id']} / 第 {p['pair_id']} 题  ")
    lines.append(f"*Type: {p['type']} | Difficulty: {p['difficulty']}*\n")
    lines.append(f"**Background / 背景:**  ")
    lines.append(f"{BACKGROUND}\n")
    lines.append(f"**User Question / 用户问题:** *\"{p['query']}\"*\n")
    lines.append(f"**Response Left / 左侧回复:**")
    lines.append(f"> {p['left']['response']}\n")
    lines.append(f"**Response Right / 右侧回复:**")
    lines.append(f"> {p['right']['response']}\n")
    lines.append("| Criterion / 评分项 | Left Score / 左分 | Right Score / 右分 |")
    lines.append("|---|---|---|")
    lines.append("| Persona Consistency / 人格一致性 (1–5) | ___ | ___ |")
    lines.append("| Naturalness / 自然度 (1–5) | ___ | ___ |")
    lines.append("| Overall Preference / 总体偏好 | ☐ Left/左 | ☐ Right/右 ☐ Tie/平局 |")
    lines.append("\n---\n")

with open('results/human_eval/questionnaire.md', 'w', encoding='utf-8') as f:
    f.write('\n'.join(lines))

# ── Save answer key ──────────────────────────────────────────────────────
key = [{'pair_id': p['pair_id'], 'left_is': p['left']['method'],
         'right_is': p['right']['method'], 'type': p['type']} for p in pairs]
with open('results/human_eval/answer_key.json', 'w') as f:
    json.dump(key, f, indent=2)

# ── Save eval pairs (no method labels) ───────────────────────────────────
blind = [{'pair_id': p['pair_id'], 'type': p['type'], 'difficulty': p['difficulty'],
          'background': BACKGROUND, 'query': p['query'],
          'response_left': p['left']['response'],
          'response_right': p['right']['response']} for p in pairs]
with open('results/human_eval/eval_pairs.json', 'w', encoding='utf-8') as f:
    json.dump(blind, f, ensure_ascii=False, indent=2)

# ── Save statistics guide ─────────────────────────────────────────────────
stats_md = '''# Statistics Guide / 统计指南

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
'''
with open('results/human_eval/statistics_guide.md', 'w') as f:
    f.write(stats_md)

print(f'Generated {len(pairs)} evaluation pairs.')
print(f'Files in results/human_eval/:')
print(f'  questionnaire.md     <- 发给评估者的双语问卷')
print(f'  eval_pairs.json      <- 数据文件（无方法标签）')
print(f'  answer_key.json      <- 答案密钥（保密）')
print(f'  statistics_guide.md  <- 如何计算论文统计数据')
