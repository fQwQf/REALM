#!/usr/bin/env python3
"""
Download larger PNH test set from HuggingFace
==============================================
Uses HF mirror for Chinese network environment.

We'll use:
1. MSC (Multi-Session Chat) for multi-turn dialogue with persona info
2. Generate additional PNH test cases programmatically
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


import os
import sys
import json
import random
from datetime import datetime
from typing import List, Dict


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


def download_msc_dataset():
    """Download MSC (Multi-Session Chat) dataset via HF mirror."""
    print("Downloading MSC dataset from HuggingFace...")
    
    try:
        from datasets import load_dataset
        
        # MSC dataset for multi-session dialogue
        dataset = load_dataset("anton-l/msc-dialogue", trust_remote_code=True)
        
        print(f"✓ MSC dataset downloaded")
        print(f"  Splits: {list(dataset.keys())}")
        for split in dataset.keys():
            print(f"  {split}: {len(dataset[split])} samples")
        
        return dataset
    
    except Exception as e:
        print(f"✗ Failed to download MSC: {e}")
        return None


def generate_extended_pnh_test_set(num_cases: int = 50) -> List[Dict]:
    """
    Generate extended PNH test cases programmatically.
    
    Types of test cases:
    - preference: User preferences
    - promise: Commitments made
    - autobiographical: Personal facts
    - boundary: User boundaries
    - preference_change: Changed preferences
    - privacy: Privacy requirements
    - work_preference: Work-related preferences
    - goal: Learning/career goals
    - restriction: Dietary/other restrictions
    - communication: Communication style
    """
    
    # Templates for generating test cases
    templates = {
        "preference": [
            ("favorite color", ["blue", "green", "red", "purple", "orange"]),
            ("favorite food", ["sushi", "pizza", "curry", "tacos", "pasta"]),
            ("favorite season", ["spring", "summer", "autumn", "winter"]),
            ("favorite sport", ["tennis", "basketball", "swimming", "running", "cycling"]),
            ("favorite music genre", ["jazz", "classical", "rock", "electronic", "folk"]),
            ("preferred drink", ["coffee", "tea", "water", "juice", "smoothie"]),
            ("favorite animal", ["dogs", "cats", "birds", "fish", "rabbits"]),
            ("favorite movie genre", ["comedy", "drama", "sci-fi", "thriller", "documentary"]),
        ],
        "promise": [
            ("keep your secrets confidential", "confidentiality"),
            ("not share your personal information", "privacy"),
            ("remind you about important deadlines", "reminders"),
            ("help you with your project", "project help"),
            ("be available during work hours", "availability"),
        ],
        "autobiographical": [
            ("birthday is in March", "March"),
            ("graduated from MIT", "MIT"),
            ("works as a software engineer", "software engineer"),
            ("lives in San Francisco", "San Francisco"),
            ("has two siblings", "two siblings"),
            ("speaks three languages", "three languages"),
            ("grew up in Boston", "Boston"),
            ("studied computer science", "computer science"),
        ],
        "boundary": [
            ("don't want to discuss politics", "politics"),
            ("prefer not to talk about religion", "religion"),
            ("avoid discussing salary", "salary"),
            ("don't share work details", "work details"),
            ("keep family matters private", "family matters"),
        ],
        "restriction": [
            ("allergic to peanuts", "peanuts"),
            ("allergic to shellfish", "shellfish"),
            ("gluten intolerant", "gluten"),
            ("lactose intolerant", "dairy"),
            ("diabetic", "sugar"),
        ],
        "goal": [
            ("learning Python programming", "Python"),
            ("training for a marathon", "marathon"),
            ("writing a book", "book"),
            ("learning Japanese", "Japanese"),
            ("starting a business", "business"),
        ],
        "communication": [
            ("prefers email over calls", "email"),
            ("likes detailed explanations", "detailed"),
            ("wants brief summaries", "brief"),
            ("prefers async communication", "async"),
            ("likes direct feedback", "direct"),
        ],
    }
    
    moods = ["Calm", "Anxious", "Happy", "Defensive", "Melancholic", "Angry", "Excited"]
    defenses = ["None", "Rationalization", "Intellectualization", "Denial", "Projection", "Humor", "Sublimation"]
    
    test_cases = []
    case_id = 1
    
    for case_type, items in templates.items():
        for item_name, expected_values in items:
            if isinstance(expected_values, list):
                expected = random.choice(expected_values)
            else:
                expected = expected_values
            
            # Generate distractor turns
            num_distractors = random.randint(4, 8)
            distractor_turns = []
            
            # Generate random distractor dialogue
            generic_responses = [
                ("Tell me more about that.", "I'd be happy to elaborate."),
                ("How are you feeling today?", "I'm doing well, thank you."),
                ("What's on your mind?", "Just processing our conversation."),
                ("Any recommendations?", "I have several suggestions."),
                ("What do you think?", "That's an interesting perspective."),
                ("Can you help me?", "Of course, what do you need?"),
                ("I'm not sure about this.", "Let's explore your concerns."),
                ("Thanks for your help.", "You're welcome!"),
            ]
            
            for _ in range(num_distractors):
                distractor_turns.extend([random.choice(generic_responses)])
            
            # Generate the needle turn based on type
            if case_type == "preference":
                implant_user = f"My {item_name} is {expected}."
                implant_agent = f"Got it, your {item_name} is {expected}."
                query = f"What's my {item_name}?"
            elif case_type == "promise":
                implant_user = f"Please promise me you'll {item_name}."
                implant_agent = f"I promise to {item_name}."
                query = "What did you promise me?"
            elif case_type == "autobiographical":
                implant_user = f"My {item_name}."
                implant_agent = f"I understand, your {item_name}."
                query = f"What do you know about my {item_name.split()[0]}?"
            elif case_type == "boundary":
                implant_user = f"I don't want to discuss {item_name}."
                implant_agent = f"Understood, I won't bring up {item_name}."
                query = f"Can we talk about {item_name}?"
            elif case_type == "restriction":
                implant_user = f"I'm {item_name if 'allergic' in str(item_name) or 'intolerant' in str(item_name) else item_name}."
                implant_agent = f"I'll remember that {item_name} is a concern for you."
                query = "Any dietary restrictions I have?"
            elif case_type == "goal":
                implant_user = f"I'm currently {item_name}."
                implant_agent = f"That's a great goal, {item_name}."
                query = "What am I currently working on?"
            else:  # communication
                implant_user = f"I prefer {expected} communication."
                implant_agent = f"I'll communicate in a {expected} manner."
                query = "How do I prefer to communicate?"
            
            # Insert needle at random position
            implant_position = random.randint(1, num_distractors)
            distractor_turns.insert(implant_position, (implant_user, implant_agent))
            
            test_case = {
                "id": f"pnh_{case_id:03d}",
                "name": f"{case_type.title()} Recall - {random.choice(moods)} State",
                "type": case_type,
                "needle": {
                    "content": implant_user,
                    "implant_turn": implant_position,
                    "state_condition": {
                        "mood": random.choice(moods),
                        "stress": random.randint(10, 85),
                        "defense": random.choice(defenses)
                    }
                },
                "distractor_turns": [{"user": u, "agent": a} for u, a in distractor_turns],
                "trigger_query": query,
                "correct_response": expected if isinstance(expected, str) else expected,
                "evaluation_criteria": {
                    "recall": f"{case_type}_recalled",
                    "state_alignment": "appropriate_tone"
                }
            }
            
            test_cases.append(test_case)
            case_id += 1
    
    # Shuffle to mix different types
    random.shuffle(test_cases)
    
    # Limit to requested number
    return test_cases[:num_cases]


def create_extended_pnh_dataset():
    """Create and save extended PNH test set."""
    print("\n" + "="*60)
    print("Creating Extended PNH Test Set")
    print("="*60)
    
    # Try to download MSC first
    msc_dataset = download_msc_dataset()
    
    # Generate programmatic test cases
    print("\nGenerating programmatic test cases...")
    generated_cases = generate_extended_pnh_test_set(num_cases=50)
    
    # Load existing test cases
    existing_path = "data/test_sets/pnh_test_set.json"
    try:
        with open(existing_path, 'r') as f:
            existing_data = json.load(f)
        existing_cases = existing_data.get("test_cases", [])
        print(f"✓ Loaded {len(existing_cases)} existing test cases")
    except:
        existing_cases = []
        print("No existing test cases found")
    
    # Combine all test cases
    all_cases = existing_cases + generated_cases
    
    # Create dataset
    dataset = {
        "name": "Extended PNH Test Dataset for HOMEO",
        "description": "Psychological Needle-in-Haystack test cases for evaluating state-dependent recall",
        "version": "2.0",
        "created": datetime.now().isoformat(),
        "total_cases": len(all_cases),
        "source": {
            "original": len(existing_cases),
            "generated": len(generated_cases),
            "msc_available": msc_dataset is not None
        },
        "test_cases": all_cases
    }
    
    # Save
    output_dir = "data/test_sets"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "pnh_extended_test_set.json")
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)
    
    print(f"\n✓ Extended PNH test set saved to: {output_path}")
    print(f"  Total test cases: {len(all_cases)}")
    print(f"  Original: {len(existing_cases)}")
    print(f"  Generated: {len(generated_cases)}")
    
    # Summary by type
    type_counts = {}
    for case in all_cases:
        case_type = case.get("type", "unknown")
        type_counts[case_type] = type_counts.get(case_type, 0) + 1
    
    print(f"\n  By type:")
    for case_type, count in sorted(type_counts.items()):
        print(f"    {case_type}: {count}")
    
    return dataset


if __name__ == "__main__":
    create_extended_pnh_dataset()
