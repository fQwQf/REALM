#!/usr/bin/env python3
"""
Large-Scale Multilingual Test (20 cases per language)
======================================================
"""

import os
import sys
import time
import json
import random
from datetime import datetime
from typing import Dict, List

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['HF_HOME'] = '/data1/tongjizhou/.cache/huggingface'

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch

# 20 test cases per language
LARGE_TEST_SET = {
    "english": {
        "implants": [
            "I love hiking on weekends.",
            "My favorite color is blue.",
            "I work as a software engineer.",
            "I prefer tea over coffee.",
            "I have a cat named Whiskers.",
            "I was born in March.",
            "My favorite food is pizza.",
            "I enjoy reading mystery novels.",
            "I live in New York.",
            "I play the guitar.",
            "I am allergic to peanuts.",
            "My favorite season is autumn.",
            "I speak three languages.",
            "I graduated from MIT.",
            "My hobby is photography.",
            "I dislike horror movies.",
            "I run every morning.",
            "My favorite sport is tennis.",
            "I have two siblings.",
            "I love classical music.",
        ],
        "recalls": [
            ("What activity do I like on weekends?", "hiking"),
            ("What's my favorite color?", "blue"),
            ("What's my job?", "engineer"),
            ("What do I prefer to drink?", "tea"),
            ("Do I have any pets?", "cat"),
            ("When was I born?", "march"),
            ("What's my favorite food?", "pizza"),
            ("What do I enjoy reading?", "mystery"),
            ("Where do I live?", "york"),
            ("Do I play any instruments?", "guitar"),
            ("What am I allergic to?", "peanut"),
            ("What's my favorite season?", "autumn"),
            ("How many languages do I speak?", "three"),
            ("Where did I graduate from?", "mit"),
            ("What's my hobby?", "photography"),
            ("What don't I like?", "horror"),
            ("What do I do every morning?", "run"),
            ("What's my favorite sport?", "tennis"),
            ("How many siblings do I have?", "two"),
            ("What kind of music do I like?", "classical"),
        ]
    },
    "chinese": {
        "implants": [
            "我周末喜欢徒步旅行。",
            "我最喜欢的颜色是蓝色。",
            "我是一名软件工程师。",
            "我更喜欢喝茶而不是咖啡。",
            "我有一只叫小橘的猫。",
            "我是三月份出生的。",
            "我最喜欢的食物是披萨。",
            "我喜欢读推理小说。",
            "我住在北京。",
            "我会弹吉他。",
            "我对花生过敏。",
            "我最喜欢的季节是秋天。",
            "我会说三种语言。",
            "我毕业于清华大学。",
            "我的爱好是摄影。",
            "我不喜欢恐怖电影。",
            "我每天早上跑步。",
            "我最喜欢的运动是网球。",
            "我有两个兄弟姐妹。",
            "我喜欢古典音乐。",
        ],
        "recalls": [
            ("我周末喜欢什么活动？", "徒步"),
            ("我最喜欢什么颜色？", "蓝"),
            ("我的工作是什么？", "工程师"),
            ("我更喜欢喝什么？", "茶"),
            ("我养了什么宠物？", "猫"),
            ("我是哪个月出生的？", "三"),
            ("我最喜欢吃什么？", "披萨"),
            ("我喜欢看什么书？", "推理"),
            ("我住在哪里？", "北京"),
            ("我会弹什么乐器？", "吉他"),
            ("我对什么过敏？", "花生"),
            ("我最喜欢什么季节？", "秋"),
            ("我会说几种语言？", "三"),
            ("我毕业于哪里？", "清华"),
            ("我的爱好是什么？", "摄影"),
            ("我不喜欢什么？", "恐怖"),
            ("我每天早上做什么？", "跑步"),
            ("我最喜欢什么运动？", "网球"),
            ("我有几个兄弟姐妹？", "两"),
            ("我喜欢什么类型的音乐？", "古典"),
        ]
    },
    "japanese": {
        "implants": [
            "週末はハイキングが好きです。",
            "一番好きな色は青です。",
            "私はソフトウェアエンジニアです。",
            "コーヒーより紅茶が好きです。",
            "ミケという猫を飼っています。",
            "3月生まれです。",
            "一番好きな食べ物はピザです。",
            "ミステリー小説が好きです。",
            "東京に住んでいます。",
            "ギターが弾けます。",
            "ピーナッツアレルギーです。",
            "一番好きな季節は秋です。",
            "3ヶ国語を話せます。",
            "東京大学を卒業しました。",
            "趣味は写真撮影です。",
            "ホラー映画は苦手です。",
            "毎朝ジョギングしています。",
            "一番好きなスポーツはテニスです。",
            "兄弟が二人います。",
            "クラシック音楽が好きです。",
        ],
        "recalls": [
            ("週末は何が好きですか？", "ハイキング"),
            ("好きな色は？", "青"),
            ("仕事は何ですか？", "エンジニア"),
            ("何を飲むのが好きですか？", "紅茶"),
            ("ペットはいますか？", "猫"),
            ("誕生日はいつですか？", "3月"),
            ("好きな食べ物は？", "ピザ"),
            ("どんな本が好きですか？", "ミステリー"),
            ("どこに住んでいますか？", "東京"),
            ("楽器は弾けますか？", "ギター"),
            ("アレルギーはありますか？", "ピーナッツ"),
            ("好きな季節は？", "秋"),
            ("何ヶ国語話せますか？", "3"),
            ("どこの大学を出ましたか？", "大学"),
            ("趣味は何ですか？", "写真"),
            ("苦手なものは？", "ホラー"),
            ("毎朝何をしていますか？", "ジョギング"),
            ("好きなスポーツは？", "テニス"),
            ("兄弟はいますか？", "兄弟"),
            ("好きな音楽は？", "クラシック"),
        ]
    }
}


def run_large_scale_multilingual(sys1_gpu: int, sys2_gpu: int) -> Dict:
    """Run large-scale multilingual test with 20 cases per language."""
    print("\n" + "="*60)
    print("Large-Scale Multilingual Test (20 cases per language)")
    print("="*60)
    
    from src.real_realm import RealREALM
    
    results = {"languages": {}}
    
    for lang in ["english", "chinese", "japanese"]:
        print(f"\n{'='*50}")
        print(f"Testing {lang.upper()} (20 cases)")
        print('='*50)
        
        try:
            realm = RealREALM(
                use_real_llm=True,
                sys1_gpu=sys1_gpu,
                sys2_gpus=[sys2_gpu]
            )
            
            # Warmup
            warmup = {"english": "Hello", "chinese": "你好", "japanese": "こんにちは"}
            _ = realm.step(warmup[lang])
            
            # Implant all 20 pieces of information
            print(f"\nPhase 1: Implanting 20 facts...")
            implants = LARGE_TEST_SET[lang]["implants"]
            for i, fact in enumerate(implants):
                _ = realm.step(f"Please remember: {fact}")
                if (i + 1) % 5 == 0:
                    print(f"  Implanted {i+1}/20")
            
            # Test all 20 recalls
            print(f"\nPhase 2: Testing 20 recalls...")
            recalls = LARGE_TEST_SET[lang]["recalls"]
            correct = 0
            total = len(recalls)
            ttft_values = []
            details = []
            
            for i, (query, expected) in enumerate(recalls):
                response, meta = realm.step(query)
                ttft_values.append(meta["ttft_ms"])
                
                success = expected.lower() in response.lower()
                if success:
                    correct += 1
                
                details.append({
                    "query": query,
                    "expected": expected,
                    "success": success,
                    "ttft_ms": meta["ttft_ms"]
                })
                
                if (i + 1) % 5 == 0:
                    print(f"  Tested {i+1}/20, correct so far: {correct}")
            
            accuracy = (correct / total) * 100
            avg_ttft = sum(ttft_values) / len(ttft_values)
            min_ttft = min(ttft_values)
            max_ttft = max(ttft_values)
            
            results["languages"][lang] = {
                "accuracy": accuracy,
                "correct": correct,
                "total": total,
                "avg_ttft": avg_ttft,
                "min_ttft": min_ttft,
                "max_ttft": max_ttft,
                "details": details
            }
            
            print(f"\n{lang.upper()} Results:")
            print(f"  Accuracy: {accuracy:.1f}% ({correct}/{total})")
            print(f"  Avg TTFT: {avg_ttft:.1f}ms")
            print(f"  TTFT range: {min_ttft:.1f}ms - {max_ttft:.1f}ms")
            
            del realm
            torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"  {lang} test failed: {e}")
            import traceback
            traceback.print_exc()
            results["languages"][lang] = {"error": str(e)}
    
    # Compute relative drops
    if "english" in results["languages"] and "accuracy" in results["languages"]["english"]:
        en_acc = results["languages"]["english"]["accuracy"]
        for lang in ["chinese", "japanese"]:
            if lang in results["languages"] and "accuracy" in results["languages"][lang]:
                rel_drop = ((en_acc - results["languages"][lang]["accuracy"]) / en_acc) * 100
                results["languages"][lang]["relative_drop_vs_english"] = rel_drop
    
    return results


def main():
    print("="*70)
    print("REALM Large-Scale Multilingual Experiment")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)
    
    output_dir = "results/large_scale_multilingual"
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    result = run_large_scale_multilingual(sys1_gpu=0, sys2_gpu=1)
    
    # Save results
    output_file = os.path.join(output_dir, f"results_{timestamp}.json")
    with open(output_file, 'w') as f:
        json.dump(result, f, indent=2, default=str)
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY (20 cases per language)")
    print("="*70)
    
    print(f"\n{'Language':<12} {'Accuracy':<12} {'Correct':<10} {'Avg TTFT':<12} {'Rel. Drop':<12}")
    print("-"*60)
    
    en_acc = result["languages"].get("english", {}).get("accuracy", 0)
    
    for lang in ["english", "chinese", "japanese"]:
        data = result["languages"].get(lang, {})
        if "accuracy" in data:
            rel_drop = data.get("relative_drop_vs_english", 0 if lang == "english" else None)
            rel_str = f"{rel_drop:.1f}%" if rel_drop is not None else "--"
            print(f"{lang:<12} {data['accuracy']:.1f}%{'':<6} {data['correct']}/{data['total']:<5} {data['avg_ttft']:.1f}ms{'':<4} {rel_str:<12}")
    
    print(f"\nResults saved to: {output_file}")
    print(f"\nCompleted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    return result


if __name__ == "__main__":
    main()
