#!/usr/bin/env python3
"""
LoCoMo + LongMemEval-S Benchmark Evaluation for TEMPO

LoCoMo: 10 conversations, each with QA annotations
LongMemEval-S: 500 questions over long chat histories

Metrics:
  - F1 (token overlap)
  - EM (exact match)

GPU: 5,6,7 for Qwen2.5-7B-Instruct
"""
import os, sys, json, time, re, string
import numpy as np
from pathlib import Path
from collections import Counter

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['HF_HOME'] = '/data1/tongjizhou/.cache/huggingface'
sys.path.insert(0, str(Path(__file__).parent.parent))

OUT_DIR = Path('/data1/tongjizhou/REALM/results/benchmarks')
OUT_DIR.mkdir(parents=True, exist_ok=True)

LOCOMO_PATH = '/data1/tongjizhou/REALM/data/benchmarks/locomo10.json'
LME_PATH = '/data1/tongjizhou/REALM/data/benchmarks/longmemeval_s.json'

SYS2_GPUS = [5, 6, 7]
SYS2_MODEL = 'Qwen/Qwen2.5-7B-Instruct'

_model_cache = {}

def load_s2():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    if 's2' in _model_cache:
        return _model_cache['s2']
    print(f'Loading {SYS2_MODEL} on GPUs {SYS2_GPUS}...')
    tok = AutoTokenizer.from_pretrained(SYS2_MODEL, trust_remote_code=True)
    max_mem = {i: '20GiB' for i in SYS2_GPUS}
    model = AutoModelForCausalLM.from_pretrained(
        SYS2_MODEL, torch_dtype=torch.float16,
        device_map='auto', max_memory=max_mem, trust_remote_code=True)
    model.eval()
    _model_cache['s2'] = (model, tok)
    print('  S2 loaded.')
    return model, tok


def generate(model, tok, prompt, max_new_tokens=80):
    import torch
    msgs = [{'role': 'user', 'content': prompt}]
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inputs = tok(text, return_tensors='pt', truncation=True, max_length=4096)
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    t0 = time.perf_counter()
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens,
                             do_sample=False, pad_token_id=tok.eos_token_id)
    elapsed = (time.perf_counter() - t0) * 1000
    new_ids = out[0][inputs['input_ids'].shape[1]:]
    return tok.decode(new_ids, skip_special_tokens=True).strip(), elapsed


def normalize(s):
    """Lowercase, remove punctuation/articles."""
    s = s.lower()
    s = re.sub(r'\b(a|an|the)\b', ' ', s)
    s = ''.join(c for c in s if c not in string.punctuation)
    return ' '.join(s.split())


def token_f1(pred, gold):
    p_tokens = normalize(pred).split()
    g_tokens = normalize(gold).split()
    if not p_tokens or not g_tokens:
        return 0.0, 0.0, 0.0
    common = Counter(p_tokens) & Counter(g_tokens)
    num_common = sum(common.values())
    if num_common == 0:
        return 0.0, 0.0, 0.0
    prec = num_common / len(p_tokens)
    rec  = num_common / len(g_tokens)
    f1   = 2 * prec * rec / (prec + rec)
    return f1, prec, rec


def exact_match(pred, gold):
    return float(normalize(pred) == normalize(gold))


# =====================================================================
# LoCoMo Evaluation
# =====================================================================
def build_locomo_context(conv_dict, max_turns_per_session=20):
    """Build text context from LoCoMo conversation dict."""
    speaker_a = conv_dict.get('speaker_a', 'A')
    speaker_b = conv_dict.get('speaker_b', 'B')
    lines = []
    # Find all session keys in order
    session_keys = sorted(
        [k for k in conv_dict.keys() if re.match(r'session_\d+$', k)],
        key=lambda k: int(k.split('_')[1])
    )
    for sk in session_keys:
        session = conv_dict[sk]
        if not isinstance(session, list):
            continue
        date_key = f'{sk}_date_time'
        date_str = conv_dict.get(date_key, '')
        if date_str:
            lines.append(f'[{date_str}]')
        for turn in session[:max_turns_per_session]:
            if isinstance(turn, dict):
                speaker_id = turn.get('speaker_id', '')
                text = turn.get('text', '')
                speaker_name = speaker_a if speaker_id == 'A' else speaker_b
                if text:
                    lines.append(f'{speaker_name}: {text}')
    return '\n'.join(lines)


def eval_locomo(model, tok, max_samples=200):
    print('\n' + '='*60)
    print('LoCoMo QA Evaluation')
    print('='*60)
    with open(LOCOMO_PATH) as f:
        data = json.load(f)
    print(f'  {len(data)} conversations loaded.')

    results = []
    count = 0
    for conv_item in data:
        conv_dict = conv_item['conversation']
        qa_list   = conv_item.get('qa', [])
        ctx = build_locomo_context(conv_dict)
        ctx_trunc = ctx[:10000]  # truncate to 10k chars

        for qa in qa_list:
            if count >= max_samples:
                break
            question = qa.get('question', '').strip()
            answer   = str(qa.get('answer', '')).strip()
            if not question or not answer:
                continue

            prompt = (
                'The following is a long conversation between two people.\n\n'
                f'{ctx_trunc}\n\n'
                f'Based on the conversation, answer concisely.\n'
                f'Question: {question}\n'
                'Answer (be brief, 1-2 sentences):'
            )
            pred, lat = generate(model, tok, prompt, max_new_tokens=64)
            f1, prec, rec = token_f1(pred, answer)
            em = exact_match(pred, answer)
            results.append({
                'question': question, 'answer': answer, 'prediction': pred,
                'f1': f1, 'em': em, 'latency_ms': lat,
                'category': qa.get('category', 'unknown')
            })
            count += 1
            if count % 20 == 0:
                avg_f1 = np.mean([r['f1'] for r in results])
                print(f'  [{count}/{max_samples}] avg F1={avg_f1:.3f}  last: Q={question[:50]}')

    avg_f1 = np.mean([r['f1'] for r in results]) if results else 0.0
    avg_em = np.mean([r['em'] for r in results]) if results else 0.0
    avg_lat = np.mean([r['latency_ms'] for r in results]) if results else 0.0
    # Per-category breakdown
    categories = set(r['category'] for r in results)
    cat_f1 = {}
    for cat in categories:
        cat_res = [r for r in results if r['category'] == cat]
        cat_f1[cat] = round(np.mean([r['f1'] for r in cat_res]), 3)

    summary = {
        'benchmark': 'locomo',
        'n_questions': len(results),
        'avg_f1': round(avg_f1, 4),
        'avg_em': round(avg_em, 4),
        'avg_latency_ms': round(avg_lat, 1),
        'per_category_f1': cat_f1,
    }
    print(f'  RESULTS: F1={avg_f1:.3f}  EM={avg_em:.3f}  N={len(results)}')
    print(f'  Per-category F1: {cat_f1}')
    return results, summary


# =====================================================================
# LongMemEval-S Evaluation
# =====================================================================
def build_lme_context(item, max_sessions=10, max_turns=30):
    """Build context from LongMemEval sessions."""
    sessions = item.get('haystack_sessions', [])
    lines = []
    dates = item.get('haystack_dates', [])
    for i, session in enumerate(sessions[:max_sessions]):
        date = dates[i] if i < len(dates) else ''
        if date:
            lines.append(f'[Session {i+1} - {date}]')
        else:
            lines.append(f'[Session {i+1}]')
        for turn in session[:max_turns]:
            role = turn.get('role', '')
            content = turn.get('content', '')
            if role and content:
                speaker = 'User' if role == 'user' else 'Assistant'
                lines.append(f'{speaker}: {content[:300]}')
    return '\n'.join(lines)


def eval_longmemeval(model, tok, max_samples=200):
    print('\n' + '='*60)
    print('LongMemEval-S Evaluation')
    print('='*60)
    with open(LME_PATH) as f:
        data = json.load(f)
    print(f'  {len(data)} questions loaded. Using first {max_samples}.')
    data = data[:max_samples]

    results = []
    for i, item in enumerate(data):
        question = item.get('question', '').strip()
        answer   = str(item.get('answer', '')).strip()
        qtype    = item.get('question_type', 'unknown')

        ctx = build_lme_context(item, max_sessions=8, max_turns=20)
        ctx_trunc = ctx[:10000]

        prompt = (
            'The following is a series of conversation sessions between a user and an assistant.\n\n'
            f'{ctx_trunc}\n\n'
            f'Based on the conversations above, answer concisely.\n'
            f'Question: {question}\n'
            'Answer (be brief):'
        )
        pred, lat = generate(model, tok, prompt, max_new_tokens=64)
        f1, prec, rec = token_f1(pred, answer)
        em = exact_match(pred, answer)
        results.append({
            'qid': item.get('question_id', str(i)),
            'question': question, 'answer': answer, 'prediction': pred,
            'f1': f1, 'em': em, 'latency_ms': lat, 'qtype': qtype
        })
        if (i+1) % 20 == 0:
            avg_f1 = np.mean([r['f1'] for r in results])
            print(f'  [{i+1}/{max_samples}] avg F1={avg_f1:.3f}  Q={question[:50]}')

    avg_f1 = np.mean([r['f1'] for r in results]) if results else 0.0
    avg_em = np.mean([r['em'] for r in results]) if results else 0.0
    avg_lat = np.mean([r['latency_ms'] for r in results]) if results else 0.0
    # Per question-type breakdown
    qtypes = set(r['qtype'] for r in results)
    qtype_f1 = {}
    for qt in qtypes:
        qt_res = [r for r in results if r['qtype'] == qt]
        qtype_f1[qt] = round(np.mean([r['f1'] for r in qt_res]), 3)

    summary = {
        'benchmark': 'longmemeval_s',
        'n_questions': len(results),
        'avg_f1': round(avg_f1, 4),
        'avg_em': round(avg_em, 4),
        'avg_latency_ms': round(avg_lat, 1),
        'per_qtype_f1': qtype_f1,
    }
    print(f'  RESULTS: F1={avg_f1:.3f}  EM={avg_em:.3f}  N={len(results)}')
    print(f'  Per-type F1: {qtype_f1}')
    return results, summary


# =====================================================================
# Main
# =====================================================================
if __name__ == '__main__':
    print('=== Public Benchmark Evaluation ===')
    model, tok = load_s2()
    all_summaries = {}

    # LoCoMo
    locomo_results, locomo_summary = eval_locomo(model, tok, max_samples=200)
    all_summaries['locomo'] = locomo_summary
    with open(OUT_DIR / 'locomo_results.json', 'w') as f:
        json.dump({'summary': locomo_summary, 'details': locomo_results}, f, indent=2)
    print(f'LoCoMo saved.')

    # LongMemEval-S
    lme_results, lme_summary = eval_longmemeval(model, tok, max_samples=200)
    all_summaries['longmemeval_s'] = lme_summary
    with open(OUT_DIR / 'longmemeval_s_results.json', 'w') as f:
        json.dump({'summary': lme_summary, 'details': lme_results}, f, indent=2)
    print(f'LongMemEval-S saved.')

    # Combined summary
    with open(OUT_DIR / 'benchmark_summary.json', 'w') as f:
        json.dump(all_summaries, f, indent=2)

    print('\n=== FINAL SUMMARY ===')
    for bm, s in all_summaries.items():
        print(f'{bm}: F1={s["avg_f1"]:.3f}  EM={s["avg_em"]:.3f}  N={s["n_questions"]}')
    print(f'Results saved to {OUT_DIR}')
