#!/usr/bin/env python3
"""
Harder Multilingual Test with Controlled TTFT Measurement
=========================================================
Addresses reviewer concerns:
1. TTFT inconsistency (378ms vs 233ms)
2. Chinese 100% accuracy (too easy test set)

This experiment:
- Uses harder test cases (multi-hop, implicit, contradiction)
- Measures TTFT with same protocol as main experiments
- Tests all three languages fairly
"""

import os
import sys
import time
import json
import random
from datetime import datetime
from typing import Dict, List, Tuple

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['HF_HOME'] = '/data1/tongjizhou/.cache/huggingface'

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch

# Harder test cases requiring deeper understanding
# These require: multi-hop reasoning, implicit information, temporal reasoning, contradictions
HARDER_TEST_SET = {
    "english": {
        "implants": [
            # Complex facts with relationships
            "I moved to New York in 2018, but I originally come from Boston.",
            "I studied computer science at MIT, then got my MBA from Stanford.",
            "I'm allergic to shellfish, so I always avoid seafood restaurants.",
            "My cat Whiskers is 5 years old, and I got her as a kitten.",
            "I play tennis on weekends, but I injured my knee last month.",
            "I prefer tea over coffee, especially green tea in the morning.",
            "I work as a software engineer at a startup in Manhattan.",
            "My favorite vacation spot is Japan, where I've been three times.",
            "I speak English, Spanish, and a little Japanese.",
            "I was born in March, making me a Pisces.",
            "My younger sister lives in Chicago with her two kids.",
            "I enjoy reading mystery novels, especially by Agatha Christie.",
            "I run 5K every morning before work, but not on rainy days.",
            "My favorite season is autumn because of the colorful leaves.",
            "I'm afraid of heights, so I never go on roller coasters.",
            "I graduated in 2015 and started working immediately after.",
            "My favorite food is sushi, which is ironic given my shellfish allergy.",
            "I've been learning to play the guitar for about two years now.",
            "I dislike horror movies because they give me nightmares.",
            "My dream is to visit every continent before I turn 50.",
        ],
        "recalls": [
            # Multi-hop / implicit reasoning
            ("How long have I lived in New York?", "2018"),  # Requires current year reasoning
            ("What degree did I get from Stanford?", "mba"),  # Requires distinguishing degrees
            ("Can I safely eat shrimp?", "no"),  # Requires allergy inference
            ("How old was Whiskers when I got her?", "kitten"),  # Implicit from "as a kitten"
            ("Why haven't I played tennis recently?", "knee"),  # Causal reasoning
            ("What type of tea do I drink in the morning?", "green"),  # Specific detail
            ("Where is my office located?", "manhattan"),  # Location inference
            ("How many times have I visited my favorite vacation destination?", "three"),  # Quantification
            ("Which language am I least fluent in?", "japanese"),  # Comparative inference
            ("What's my zodiac sign?", "pisces"),  # Requires birth month knowledge
            ("Where does my sibling live?", "chicago"),  # Family relationship
            ("Who is my favorite author?", "christie"),  # Author identification
            ("On what days do I NOT run?", "rainy"),  # Conditional reasoning
            ("Why do I like autumn?", "leaves"),  # Causal reasoning
            ("What type of attraction do I avoid?", "roller"),  # Fear inference
            ("How many years of work experience do I have?", "2015"),  # Requires current year
            ("Is there any sushi I can safely eat?", "no"),  # Cross-fact inference (shellfish allergy)
            ("How long have I been learning guitar?", "two"),  # Duration
            ("Why don't I watch scary movies?", "nightmares"),  # Causal reasoning
            ("What's my travel goal by age 50?", "continent"),  # Future plan
        ]
    },
    "chinese": {
        "implants": [
            # Complex facts requiring understanding
            "我2018年搬到了北京，但我原本是上海人。",
            "我在清华大学学的是计算机，后来在北大读了MBA。",
            "我对花生过敏，所以从来不吃坚果类的食物。",
            "我的猫叫小橘，今年5岁了，是我在它还是小猫时领养的。",
            "我周末喜欢打网球，但上个月膝盖受伤了，现在还在恢复。",
            "我更喜欢喝茶，尤其是早上的绿茶。",
            "我在中关村的一家科技公司做软件工程师。",
            "我最喜欢的旅游目的地是日本，已经去过三次了。",
            "我会说中文、英文和一点日语。",
            "我是三月份出生的，所以是双鱼座。",
            "我的妹妹住在杭州，有两个孩子。",
            "我喜欢读推理小说，特别是东野圭吾的作品。",
            "我每天上班前会跑5公里，但下雨天除外。",
            "我最喜欢秋天，因为树叶会变色。",
            "我有恐高症，所以从来不坐过山车。",
            "我2015年毕业，毕业后就开始工作了。",
            "我最喜欢的食物是披萨，里面没有花生所以很安全。",
            "我学吉他已经两年了。",
            "我不喜欢恐怖片，因为会做噩梦。",
            "我的目标是在50岁之前去过所有大洲。",
        ],
        "recalls": [
            # 更难的推理问题
            ("我在北京住了多久了？", "2018"),  # 需要计算年份
            ("我在北大读的是什么学位？", "mba"),  # 需要区分学位
            ("我可以安全地吃花生酱吗？", "不"),  # 需要过敏推理
            ("我领养小橘时它多大？", "小猫"),  # 隐含推理
            ("为什么我最近没打网球？", "膝盖"),  # 因果推理
            ("我早上喝什么茶？", "绿"),  # 具体细节
            ("我的公司在哪个地区？", "中关村"),  # 位置推理
            ("我去过最喜欢的旅游地几次？", "三"),  # 数量
            ("我最不流利的语言是什么？", "日语"),  # 比较推理
            ("我是什么星座？", "双鱼"),  # 需要出生月份知识
            ("我妹妹住在哪里？", "杭州"),  # 家庭关系
            ("我最喜欢的推理小说作者是谁？", "东野"),  # 作者识别
            ("我哪天不跑步？", "下雨"),  # 条件推理
            ("为什么我喜欢秋天？", "叶"),  # 因果推理
            ("我避免什么类型的游乐设施？", "过山车"),  # 恐惧推理
            ("我工作多少年了？", "2015"),  # 需要计算
            ("披萨对我来说安全吗？为什么？", "安全"),  # 跨事实推理
            ("我学吉他多久了？", "两"),  # 时长
            ("为什么我不看恐怖片？", "噩梦"),  # 因果推理
            ("我50岁之前的目标是什么？", "大洲"),  # 未来计划
        ]
    },
    "japanese": {
        "implants": [
            # Complex facts
            "2018年に東京に引っ越しましたが、出身は大阪です。",
            "東京大学でコンピューターサイエンスを学び、その後京都大学でMBAを取りました。",
            "私はピーナッツアレルギーなので、ナッツ類は食べません。",
            "私の猫のミケは5歳で、子猫の時から飼っています。",
            "週末はテニスが好きですが、先月膝を怪我してリハビリ中です。",
            "コーヒーよりお茶が好きで、特に朝は緑茶を飲みます。",
            "渋谷のIT企業でソフトウェアエンジニアとして働いています。",
            "一番好きな旅行先はフランスで、3回行ったことがあります。",
            "日本語、英語、そして少し中国語が話せます。",
            "3月生まれなので、魚座です。",
            "妹は福岡に住んでいて、2人の子供がいます。",
            "ミステリー小説が好きで、特に東野圭吾の作品がお気に入りです。",
            "毎朝仕事前に5キロ走りますが、雨の日は走りません。",
            "一番好きな季節は秋です、紅葉がきれいだからです。",
            "高所恐怖症なので、ジェットコースターには乗りません。",
            "2015年に卒業して、すぐに働き始めました。",
            "一番好きな食べ物は寿司ですが、アレルギーがあるので注意しています。",
            "ギターを習い始めて2年になります。",
            "ホラー映画は嫌いです、悪夢を見るので。",
            "50歳になるまでにすべての大陸を訪れるのが夢です。",
        ],
        "recalls": [
            # Harder reasoning
            ("東京に住んでどのくらいですか？", "2018"),  # Year calculation
            ("京都大学で何の学位を取りましたか？", "mba"),  # Degree distinction
            ("ピーナッツバターを安全に食べられますか？", "いいえ"),  # Allergy inference
            ("ミケを飼い始めた時、何歳でしたか？", "子猫"),  # Implicit reasoning
            ("なぜ最近テニスをしていないのですか？", "膝"),  # Causal reasoning
            ("朝は何のお茶を飲みますか？", "緑"),  # Specific detail
            ("会社はどこにありますか？", "渋谷"),  # Location inference
            ("好きな旅行先に何回行ったことがありますか？", "3"),  # Quantification
            ("どの言語が一番苦手ですか？", "中国"),  # Comparative inference
            ("星座は何ですか？", "魚"),  # Birth month knowledge
            ("妹はどこに住んでいますか？", "福岡"),  # Family relationship
            ("好きな作家は誰ですか？", "東野"),  # Author identification
            ("どんな日にジョギングしませんか？", "雨"),  # Conditional reasoning
            ("なぜ秋が好きですか？", "紅葉"),  # Causal reasoning
            ("どんなアトラクションを避けますか？", "ジェットコースター"),  # Fear inference
            ("仕事経験は何年ですか？", "2015"),  # Year calculation
            ("寿司は安全に食べられますか？", "注意"),  # Cross-fact inference
            ("ギターをどのくらい習っていますか？", "2"),  # Duration
            ("なぜホラー映画を見ないのですか？", "悪夢"),  # Causal reasoning
            ("50歳までの夢は何ですか？", "大陸"),  # Future plan
        ]
    }
}


def run_harder_multilingual_test(sys1_gpu: int, sys2_gpu: int) -> Dict:
    """Run harder multilingual test with controlled TTFT measurement."""
    print("\n" + "="*60)
    print("Harder Multilingual Test with Controlled TTFT")
    print("="*60)
    
    from src.real_realm import RealREALM
    
    results = {"languages": {}, "protocol": "controlled"}
    
    # Use same warmup and setup as main experiments for fair TTFT comparison
    warmup_queries = {
        "english": ["Hello, how are you?", "What's the weather like?"],
        "chinese": ["你好，最近怎么样？", "今天天气怎么样？"],
        "japanese": ["こんにちは、お元気ですか？", "今日の天気はどうですか？"]
    }
    
    for lang in ["english", "chinese", "japanese"]:
        print(f"\n{'='*50}")
        print(f"Testing {lang.upper()} (20 harder cases)")
        print('='*50)
        
        try:
            realm = RealREALM(
                use_real_llm=True,
                sys1_gpu=sys1_gpu,
                sys2_gpus=[sys2_gpu]
            )
            
            # Warmup (same as main experiments)
            print(f"\nWarming up...")
            for warmup in warmup_queries[lang]:
                _ = realm.step(warmup)
            
            # Implant all 20 complex facts
            print(f"\nPhase 1: Implanting 20 complex facts...")
            implants = HARDER_TEST_SET[lang]["implants"]
            for i, fact in enumerate(implants):
                _ = realm.step(f"Please remember: {fact}")
                if (i + 1) % 5 == 0:
                    print(f"  Implanted {i+1}/20")
            
            # Test all 20 harder recalls with TTFT measurement
            print(f"\nPhase 2: Testing 20 harder recalls...")
            recalls = HARDER_TEST_SET[lang]["recalls"]
            correct = 0
            total = len(recalls)
            ttft_values = []
            details = []
            
            for i, (query, expected) in enumerate(recalls):
                response, meta = realm.step(query)
                ttft_values.append(meta["ttft_ms"])
                
                # Check for expected answer (case-insensitive, partial match)
                success = expected.lower() in response.lower()
                if success:
                    correct += 1
                
                details.append({
                    "query": query,
                    "expected": expected,
                    "response": response[:200],  # Truncate for storage
                    "success": success,
                    "ttft_ms": meta["ttft_ms"]
                })
                
                if (i + 1) % 5 == 0:
                    print(f"  Tested {i+1}/20, correct so far: {correct}")
            
            accuracy = (correct / total) * 100
            # Use P50 (median) for TTFT to match main experiments
            sorted_ttft = sorted(ttft_values)
            p50_ttft = sorted_ttft[len(sorted_ttft) // 2]
            avg_ttft = sum(ttft_values) / len(ttft_values)
            
            results["languages"][lang] = {
                "accuracy": accuracy,
                "correct": correct,
                "total": total,
                "ttft_p50_ms": p50_ttft,
                "ttft_avg_ms": avg_ttft,
                "ttft_min_ms": min(ttft_values),
                "ttft_max_ms": max(ttft_values),
                "details": details
            }
            
            print(f"\n{lang.upper()} Results (Harder Test):")
            print(f"  Accuracy: {accuracy:.1f}% ({correct}/{total})")
            print(f"  TTFT P50: {p50_ttft:.1f}ms")
            print(f"  TTFT Avg: {avg_ttft:.1f}ms")
            
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
    print("REALM Harder Multilingual Experiment")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)
    
    output_dir = "results/harder_multilingual"
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Run with same GPU allocation as main experiments for fair TTFT
    result = run_harder_multilingual_test(sys1_gpu=0, sys2_gpu=1)
    
    # Save results
    output_file = os.path.join(output_dir, f"results_{timestamp}.json")
    with open(output_file, 'w') as f:
        json.dump(result, f, indent=2, default=str)
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY (Harder Multilingual Test)")
    print("="*70)
    
    print(f"\n{'Language':<12} {'Accuracy':<12} {'TTFT P50':<12} {'Rel. Drop':<12}")
    print("-"*50)
    
    en_acc = result["languages"].get("english", {}).get("accuracy", 0)
    
    for lang in ["english", "chinese", "japanese"]:
        data = result["languages"].get(lang, {})
        if "accuracy" in data:
            rel_drop = data.get("relative_drop_vs_english", 0 if lang == "english" else None)
            rel_str = f"{rel_drop:.1f}%" if rel_drop is not None else "--"
            print(f"{lang:<12} {data['accuracy']:.1f}%{'':<6} {data['ttft_p50_ms']:.1f}ms{'':<5} {rel_str:<12}")
    
    print(f"\nResults saved to: {output_file}")
    print(f"\nCompleted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    return result


if __name__ == "__main__":
    main()
