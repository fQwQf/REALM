#!/usr/bin/env python3
import json, os

os.makedirs('results/human_eval', exist_ok=True)

pairs = [
    {'pair_id': 1,  'type': 'name',               'query': "What's my name?",               'background': 'Personal AI assistant eval'},
    {'pair_id': 2,  'type': 'job',                'query': 'What do I do for work?',          'background': 'Personal AI assistant eval'},
    {'pair_id': 3,  'type': 'location',           'query': 'Where do I live?',                'background': 'Personal AI assistant eval'},
    {'pair_id': 4,  'type': 'education',          'query': 'Where did I graduate from?',      'background': 'Personal AI assistant eval'},
    {'pair_id': 5,  'type': 'preference',         'query': "What's my favorite food?",       'background': 'Personal AI assistant eval'},
    {'pair_id': 6,  'type': 'hobby',              'query': 'What do I like to do on weekends?', 'background': 'Personal AI assistant eval'},
    {'pair_id': 7,  'type': 'preference',         'query': "What's my favorite color?",      'background': 'Personal AI assistant eval'},
    {'pair_id': 8,  'type': 'pet',                'query': 'Do I have any pets?',             'background': 'Personal AI assistant eval'},
    {'pair_id': 9,  'type': 'family',             'query': "What's my wife's name?",         'background': 'Personal AI assistant eval'},
    {'pair_id': 10, 'type': 'family',             'query': 'How many children do I have?',    'background': 'Personal AI assistant eval'},
    {'pair_id': 11, 'type': 'family',             'query': "Who's my grandson?",             'background': 'Personal AI assistant eval'},
    {'pair_id': 12, 'type': 'updated_preference', 'query': 'What do I drink now?',            'background': 'Personal AI assistant eval'},
    {'pair_id': 13, 'type': 'recent_event',       'query': 'Where did I hike recently?',      'background': 'Personal AI assistant eval'},
    {'pair_id': 14, 'type': 'new_hobby',          'query': "What's my new hobby?",           'background': 'Personal AI assistant eval'},
    {'pair_id': 15, 'type': 'health',             'query': 'What am I allergic to?',          'background': 'Personal AI assistant eval'},
    {'pair_id': 16, 'type': 'tool',               'query': 'What do I use for work?',         'background': 'Personal AI assistant eval'},
    {'pair_id': 17, 'type': 'specialization',     'query': 'What do I write about?',          'background': 'Personal AI assistant eval'},
    {'pair_id': 18, 'type': 'specialization',     'query': "What's my signature dish?",      'background': 'Personal AI assistant eval'},
    {'pair_id': 19, 'type': 'career',             'query': 'Where did I work for 30 years?',  'background': 'Personal AI assistant eval'},
    {'pair_id': 20, 'type': 'sport',              'query': 'What sport do I play?',           'background': 'Personal AI assistant eval'},
]

# Reconstruct left/right assignment from existing rerun_tempo_outputs.json
with open('results/human_eval/rerun_tempo_outputs.json') as f:
    prev = {d['pair_id']: d for d in json.load(f)}

key = []
for p in pairs:
    pid = p['pair_id']
    left_is = prev[pid]['left_is']
    right_is = prev[pid]['right_is']
    key.append({'pair_id': pid, 'type': p['type'], 'left_is': left_is, 'right_is': right_is})

with open('results/human_eval/eval_pairs.json', 'w') as f:
    json.dump(pairs, f, ensure_ascii=False, indent=2)
with open('results/human_eval/answer_key.json', 'w') as f:
    json.dump(key, f, ensure_ascii=False, indent=2)

print(f'Created eval_pairs.json ({len(pairs)} pairs) and answer_key.json ({len(key)} entries)')
