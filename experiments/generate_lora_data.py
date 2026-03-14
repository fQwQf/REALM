#!/usr/bin/env python3
"""
Generate Safe-to-Say LoRA training data for System 1 (Qwen2.5-0.5B).

The model must learn ONE behavior: given any user query, output ONLY:
  TYPE: <classification>
  BRIDGE: <short hedge phrase>

It must NEVER output factual claims, names, numbers, places, or placeholders.

Generates ~500 diverse training examples across all query types.
"""
import json
import random
import itertools

random.seed(42)

# ── Hedge phrase templates (the ONLY acceptable System 1 outputs) ────────
FACTUAL_HEDGES = [
    "Let me recall that...",
    "Let me check what you told me...",
    "Hmm, let me think back...",
    "Let me look that up...",
    "Let me recall your details...",
    "Let me check what you shared...",
    "Hmm, let me recall...",
    "Let me think back carefully...",
    "One moment, let me recall...",
    "Let me check my notes...",
    "Let me think about that...",
    "Hmm, let me look into that...",
    "Let me recall what you mentioned...",
    "One moment, checking...",
    "Let me see what I remember...",
]

GREETING_HEDGES = [
    "Hello! How can I help?",
    "Hi there! How are you?",
    "Good to see you! What's up?",
    "Hey! How can I help today?",
    "Hello! Nice to hear from you.",
    "Hi! What can I do for you?",
    "Hey there! How's it going?",
]

SHARING_HEDGES = [
    "Got it, noted.",
    "Got it, I'll remember that.",
    "Noted, thanks for sharing!",
    "Got it, I'll keep that in mind.",
    "Noted, thanks!",
    "Got it, thanks for telling me.",
    "I'll remember that, thanks!",
    "Noted! That's good to know.",
    "Got it, I've noted that down.",
    "Thanks for sharing that!",
]

OPINION_HEDGES = [
    "Interesting question, let me think...",
    "Let me think about that...",
    "Good question, let me consider...",
    "That's worth thinking about...",
    "Let me consider your situation...",
    "Hmm, let me think on that...",
    "That's a thoughtful question...",
]

OTHER_HEDGES = [
    "Of course! One moment...",
    "Sure, let me help with that...",
    "No problem at all.",
    "You're welcome!",
    "Hmm, let me check...",
    "Sure thing!",
    "Let me see what I can do...",
]

# ── Query templates by type ──────────────────────────────────────────────

# FACTUAL: Questions about personal facts the user previously shared
FACTUAL_TEMPLATES = [
    # Name/identity
    "What's my name?", "What is my name?", "Do you remember my name?",
    "Who am I?", "Can you tell me my name?", "What did I say my name was?",
    # Job/career
    "What do I do for work?", "What's my job?", "Where do I work?",
    "What is my profession?", "What do I do for a living?",
    "What's my job title?", "What company do I work for?",
    "Where did I work before?", "What was my last job?",
    "How long have I been at my job?", "When did I start working there?",
    "What did I do for 30 years?", "Where did I work for 30 years?",
    # Location
    "Where do I live?", "What city do I live in?", "Where am I from?",
    "What's my address?", "What neighborhood do I live in?",
    "Where did I grow up?", "What country am I from?",
    "Where was I born?", "What state do I live in?",
    # Education
    "Where did I graduate from?", "What school did I go to?",
    "What university did I attend?", "What did I study?",
    "What's my major?", "When did I graduate?",
    "What degree do I have?", "Where did I go to college?",
    # Food/drink preferences
    "What's my favorite food?", "What do I like to eat?",
    "What's my favorite drink?", "What do I drink now?",
    "What's my comfort food?", "What cuisine do I prefer?",
    "What's my signature dish?", "What do I usually cook?",
    "Am I vegetarian?", "Do I have food allergies?",
    # Hobbies/activities
    "What do I like to do on weekends?", "What are my hobbies?",
    "What do I do for fun?", "What sport do I play?",
    "What's my new hobby?", "What instrument do I play?",
    "Do I exercise?", "What's my favorite activity?",
    "What do I do in my free time?", "What games do I play?",
    # Family
    "What's my wife's name?", "What's my husband's name?",
    "How many children do I have?", "Who's my grandson?",
    "Do I have siblings?", "What are my kids' names?",
    "Who's my partner?", "When did I get married?",
    "What's my daughter's name?", "What's my son's name?",
    "How old are my kids?", "Do I have grandchildren?",
    "What's my mother's name?", "What's my father's name?",
    "Who is my best friend?", "Do I have a partner?",
    # Pets
    "Do I have any pets?", "What's my pet's name?",
    "What kind of pet do I have?", "What's my dog's name?",
    "What's my cat's name?", "How many pets do I have?",
    # Health
    "What am I allergic to?", "Do I have any allergies?",
    "What medications do I take?", "Do I have health issues?",
    "What's my blood type?", "What medical conditions do I have?",
    # Preferences
    "What's my favorite color?", "What's my favorite movie?",
    "What music do I like?", "What's my favorite book?",
    "What TV shows do I watch?", "What's my favorite season?",
    "What's my favorite holiday?", "What's my favorite restaurant?",
    # Recent events
    "Where did I hike recently?", "What did I do last weekend?",
    "Where did I travel recently?", "What happened last week?",
    "What was my last vacation?", "What did I cook last week?",
    # Meta-recall
    "What did I tell you about my job?", "What did I mention about my family?",
    "Do you remember what I said?", "What did I share about my hobbies?",
    "What was I telling you earlier?", "Remind me what I said about my pet.",
    "What did I say my favorite color was?", "Do you remember my wife's name?",
    "What did I tell you about my childhood?", "What goals did I mention?",
    # Work details
    "What tools do I use for work?", "What software do I use?",
    "What team do I work with?", "What project am I working on?",
    "What do I write about?", "What do I specialize in?",
    # Lifestyle
    "What's my morning routine?", "What time do I wake up?",
    "Do I work from home?", "What's my commute like?",
    "What car do I drive?", "Do I own or rent?",
]

GREETING_TEMPLATES = [
    "Hello!", "Hi!", "Hey!", "Good morning!", "Good afternoon!",
    "Good evening!", "Hi there!", "Hey there!", "What's up?",
    "How are you?", "Long time no see!", "It's me again!",
    "Hey, it's me.", "Hello again!", "Nice to see you!",
    "Greetings!", "Howdy!", "Yo!", "Sup?", "Hi, how are you doing?",
    "Good to be back!", "Hey, remember me?", "I'm back!",
    "Hi, it's been a while!", "Hello, how have you been?",
]

SHARING_TEMPLATES = [
    "I love hiking.", "My favorite food is pizza.", "I just got a new cat.",
    "I moved to Seattle last month.", "I graduated from MIT in 2018.",
    "My wife's name is Emily.", "I switched from coffee to green tea.",
    "I've been working at Google for 5 years.", "My grandson was born last spring.",
    "I'm allergic to shellfish.", "I started learning guitar recently.",
    "I have two dogs named Max and Bella.", "My favorite color is blue.",
    "I play basketball on weekends.", "I just came back from Japan.",
    "I'm a software engineer.", "I live in New York City.",
    "My daughter just started college.", "I've been married for 10 years.",
    "I'm training for a marathon.", "I just finished reading a great book.",
    "I cook Italian food a lot.", "My birthday is in March.",
    "I speak three languages.", "I used to live in London.",
    "I have a cat named Luna.", "My son plays soccer.",
    "I'm working on a new project.", "I just got promoted.",
    "I volunteer at the local shelter.", "I'm learning to paint.",
    "I drive a Tesla.", "My favorite movie is Inception.",
    "I listen to jazz music.", "I'm from Brazil originally.",
    "I have a brother named Tom.", "My mom is a teacher.",
    "I went to Stanford.", "I'm a vegetarian now.",
    "I just adopted a puppy.", "I'm planning a trip to Italy.",
]

OPINION_TEMPLATES = [
    "What do you think about AI?", "Should I take this job?",
    "What's the best way to learn Python?", "Do you think I should move?",
    "What would you recommend for dinner?", "Is it worth learning a new language?",
    "Should I get a dog?", "What's a good book to read?",
    "Do you think remote work is better?", "Should I change careers?",
    "What's the best exercise routine?", "Is it a good idea to invest now?",
    "Should I go back to school?", "What's a good hobby to pick up?",
    "Do you think I should travel more?", "What's the best way to save money?",
    "Should I learn to cook?", "What's a good gift for my wife?",
    "Do you think I need more sleep?", "Should I start meditating?",
]

OTHER_TEMPLATES = [
    "Can you help me?", "Thanks!", "Never mind.", "I need some advice.",
    "What's today's date?", "Tell me a joke.", "What can you do?",
    "How does this work?", "I'm bored.", "What should I do?",
    "Can you explain that?", "I don't understand.", "Say that again?",
    "What do you mean?", "That's interesting.", "I see.",
    "OK, got it.", "Let's move on.", "What else?", "Anything else?",
]

def generate_examples():
    examples = []

    # FACTUAL examples (largest category — this is where the model fails most)
    for query in FACTUAL_TEMPLATES:
        hedge = random.choice(FACTUAL_HEDGES)
        examples.append({"query": query, "type": "FACTUAL", "bridge": hedge})

    # Generate additional FACTUAL variations with different phrasings
    factual_prefixes = [
        "Tell me ", "Can you tell me ", "Do you remember ", "What was ",
        "Remind me ", "I forgot, ", "Quick question: ", "Hey, ",
    ]
    factual_subjects = [
        "my name", "my job", "where I live", "my favorite food",
        "my pet's name", "my wife's name", "how many kids I have",
        "what I'm allergic to", "my favorite color", "where I went to school",
        "my hobby", "what sport I play", "my grandson's name",
        "where I worked", "what I drink", "my signature dish",
    ]
    for prefix, subject in itertools.product(factual_prefixes, factual_subjects):
        query = f"{prefix}{subject}?"
        hedge = random.choice(FACTUAL_HEDGES)
        examples.append({"query": query, "type": "FACTUAL", "bridge": hedge})

    # GREETING examples
    for query in GREETING_TEMPLATES:
        hedge = random.choice(GREETING_HEDGES)
        examples.append({"query": query, "type": "GREETING", "bridge": hedge})

    # SHARING examples
    for query in SHARING_TEMPLATES:
        hedge = random.choice(SHARING_HEDGES)
        examples.append({"query": query, "type": "SHARING", "bridge": hedge})

    # OPINION examples
    for query in OPINION_TEMPLATES:
        hedge = random.choice(OPINION_HEDGES)
        examples.append({"query": query, "type": "OPINION", "bridge": hedge})

    # OTHER examples
    for query in OTHER_TEMPLATES:
        hedge = random.choice(OTHER_HEDGES)
        examples.append({"query": query, "type": "OTHER", "bridge": hedge})

    random.shuffle(examples)
    return examples


if __name__ == "__main__":
    examples = generate_examples()
    out_path = "data/safe_to_say_train.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(examples, f, ensure_ascii=False, indent=2)
    print(f"Generated {len(examples)} training examples → {out_path}")

    # Print distribution
    from collections import Counter
    dist = Counter(e["type"] for e in examples)
    for t, c in dist.most_common():
        print(f"  {t}: {c}")
