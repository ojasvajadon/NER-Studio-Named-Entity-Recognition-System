"""
generate_synthetic.py
=====================
Generates ~5 000 synthetic NER-annotated sentences and saves them to
data/synthetic_ner.jsonl in the same format used by the feedback file
(so train_ner.py can load them with load_feedback_examples()).

Run from the project root:
    python generate_synthetic.py [--n 5000] [--seed 42]

Output: data/synthetic_ner.jsonl
"""
from __future__ import annotations

import argparse
import json
import random
from datetime import datetime, timezone
from pathlib import Path

# ── vocabulary pools ────────────────────────────────────────────────────────

FIRST_NAMES = [
    "Aarav", "Aditya", "Akash", "Amit", "Ananya", "Ankit", "Arjun", "Deepa",
    "Divya", "Gaurav", "Isha", "Karan", "Kavya", "Kunal", "Manish", "Meera",
    "Neha", "Nikhil", "Niket", "Pooja", "Priya", "Rahul", "Raj", "Ravi",
    "Rohit", "Sakshi", "Sanjay", "Shruti", "Sneha", "Srishti", "Sundar",
    "Tanya", "Vikas", "Vikram", "Vinay", "Yash",
    # International
    "Alice", "Bob", "Charlie", "David", "Emma", "Ethan", "Grace", "James",
    "Liam", "Maria", "Noah", "Olivia", "Sophia", "William", "Zoe",
]

LAST_NAMES = [
    "Agarwal", "Bansal", "Bose", "Chaudhary", "Dubey", "Gupta", "Iyer",
    "Jain", "Joshi", "Kapoor", "Khan", "Kumar", "Mehta", "Mishra", "Nair",
    "Patel", "Pillai", "Rao", "Reddy", "Sharma", "Singh", "Sinha", "Tiwari",
    "Trivedi", "Verma", "Yadav",
    # International
    "Brown", "Davis", "Garcia", "Johnson", "Jones", "Lee", "Martin",
    "Martinez", "Miller", "Moore", "Patel", "Smith", "Taylor", "Thomas",
    "White", "Wilson",
]

ORGS = [
    "Google", "Microsoft", "Amazon", "Apple", "Meta", "Netflix", "Tesla",
    "OpenAI", "Infosys", "TCS", "Wipro", "HCL Technologies", "Reliance Industries",
    "HDFC Bank", "ICICI Bank", "State Bank of India", "Tata Consultancy Services",
    "Flipkart", "Zomato", "Paytm", "Byju's", "Ola", "Uber", "Swiggy",
    "IIT Delhi", "IIT Bombay", "IISc Bangalore", "University of California",
    "Harvard University", "MIT", "Stanford University", "Oxford University",
    "ISRO", "DRDO", "NASSCOM", "Securities Exchange Board of India",
    "Reserve Bank of India", "World Bank", "United Nations", "WHO",
]

GPES = [
    "Mumbai", "Delhi", "Bangalore", "Chennai", "Hyderabad", "Kolkata", "Pune",
    "Ahmedabad", "Jaipur", "Surat", "Lucknow", "Kanpur", "Nagpur", "Indore",
    "Bhopal", "Patna", "Vadodara", "Coimbatore", "Visakhapatnam", "Kochi",
    "India", "China", "United States", "United Kingdom", "Germany", "France",
    "Japan", "Australia", "Canada", "Singapore", "Dubai", "London", "New York",
    "San Francisco", "Tokyo", "Paris", "Berlin", "Sydney",
]

DATES = [
    "last Monday", "next Friday", "3rd April 2024", "January 2023",
    "Q3 2025", "in 2022", "on 15 August", "by December", "this year",
    "the 1990s", "two years ago", "early 2026", "March 2025",
]

TIMES = [
    "at 9 AM", "by 5 PM", "at noon", "around midnight", "at 10:30 AM",
    "before 6 PM", "at dawn", "in the afternoon",
]

MONEY = [
    "₹50 crore", "₹2,000", "1 million dollars", "$500 million", "€200 billion",
    "₹10 lakh", "50 million euros", "$1.2 billion", "₹3.5 crore",
]

PERCENTS = [
    "12%", "4.5%", "over 30%", "nearly 80%", "just 2%", "up by 15%",
    "a 7% drop", "more than 50%",
]

PRODUCTS = [
    "iPhone 15", "Galaxy S24", "MacBook Pro", "Surface Pro", "Pixel 8",
    "OnePlus 12", "ChatGPT", "Gemini", "Copilot", "Google Maps",
]

WORKS = [
    "Harry Potter", "The Alchemist", "Inception", "2001: A Space Odyssey",
    "Dune", "Ramayana", "Mahabharata", "RRR", "Pathaan", "3 Idiots",
]

LANGS = [
    "Hindi", "English", "Tamil", "Telugu", "Kannada", "Marathi", "Bengali",
    "Gujarati", "Punjabi", "Malayalam", "Odia", "French", "German", "Japanese",
]

NORP = [
    "Indians", "Americans", "Europeans", "Asians", "Muslims", "Hindus",
    "Christians", "Sikhs", "Buddhists", "Jains", "Democrats", "Republicans",
]

# ── template builders ────────────────────────────────────────────────────────

def _person() -> str:
    return f"{random.choice(FIRST_NAMES)} {random.choice(LAST_NAMES)}"


def _make(text: str, entities: list[tuple[str, str]]) -> dict:
    """Return a feedback-format record."""
    ent_list: list[tuple[int, int, str]] = []
    cursor = 0
    for surface, label in entities:
        idx = text.find(surface, cursor)
        if idx == -1:
            continue
        ent_list.append((idx, idx + len(surface), label))
        cursor = idx + len(surface)
    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "text": text,
        "entities": [[s, e, l] for s, e, l in ent_list],
    }


def _gen_person_sentence(rng: random.Random) -> dict:
    templates = [
        lambda p, o, g: (f"My name is {p} and I work at {o} in {g}.",
                         [(p, "PERSON"), (o, "ORG"), (g, "GPE")]),
        lambda p, o, g: (f"{p} is the CEO of {o}.",
                         [(p, "PERSON"), (o, "ORG")]),
        lambda p, o, g: (f"{p} moved from {g} to pursue a career at {o}.",
                         [(p, "PERSON"), (g, "GPE"), (o, "ORG")]),
        lambda p, o, g: (f"I am {p} and I live in {g}.",
                         [(p, "PERSON"), (g, "GPE")]),
        lambda p, o, g: (f"{p} was born in {g} and studied at {o}.",
                         [(p, "PERSON"), (g, "GPE"), (o, "ORG")]),
        lambda p, o, g: (f"The report by {p} from {o} shows impressive growth.",
                         [(p, "PERSON"), (o, "ORG")]),
        lambda p, o, g: (f"{p} left {o} in {g} after five years.",
                         [(p, "PERSON"), (o, "ORG"), (g, "GPE")]),
        lambda p, o, g: (f"According to {p}, the office in {g} will be expanded.",
                         [(p, "PERSON"), (g, "GPE")]),
    ]
    p = _person()
    o = rng.choice(ORGS)
    g = rng.choice(GPES)
    tmpl = rng.choice(templates)
    text, ents = tmpl(p, o, g)
    return _make(text, ents)


def _gen_org_sentence(rng: random.Random) -> dict:
    pct = rng.choice(PERCENTS)
    templates = [
        lambda o, g, d: (f"{o} announced on {d} that it will expand operations in {g}.",
                         [(o, "ORG"), (d, "DATE"), (g, "GPE")]),
        lambda o, g, d: (f"{o} reported a revenue increase of {pct} in {d}.",
                         [(o, "ORG"), (pct, "PERCENT"), (d, "DATE")]),
        lambda o, g, d: (f"The headquarters of {o} is located in {g}.",
                         [(o, "ORG"), (g, "GPE")]),
        lambda o, g, d: (f"{o} partnered with another firm based in {g} on {d}.",
                         [(o, "ORG"), (g, "GPE"), (d, "DATE")]),
        lambda o, g, d: (f"Investors in {g} welcomed {o}'s strategic announcement.",
                         [(g, "GPE"), (o, "ORG")]),
    ]
    o = rng.choice(ORGS)
    g = rng.choice(GPES)
    d = rng.choice(DATES)
    tmpl = rng.choice(templates)
    text, ents = tmpl(o, g, d)
    return _make(text, ents)


def _gen_date_money_sentence(rng: random.Random) -> dict:
    g = rng.choice(GPES)
    pct = rng.choice(PERCENTS)
    templates = [
        lambda o, m, d: (f"{o} raised {m} in its Series B round on {d}.",
                         [(o, "ORG"), (m, "MONEY"), (d, "DATE")]),
        lambda o, m, d: (f"The deal worth {m} was signed in {d}.",
                         [(m, "MONEY"), (d, "DATE")]),
        lambda o, m, d: (f"{o} plans to invest {m} in {g} by {d}.",
                         [(o, "ORG"), (m, "MONEY"), (d, "DATE"), (g, "GPE")]),
        lambda o, m, d: (f"Revenue grew by {pct} to {m} in {d}.",
                         [(pct, "PERCENT"), (m, "MONEY"), (d, "DATE")]),
    ]
    o = rng.choice(ORGS)
    m = rng.choice(MONEY)
    d = rng.choice(DATES)
    tmpl = rng.choice(templates)
    text, ents = tmpl(o, m, d)
    return _make(text, ents)


def _gen_mixed_sentence(rng: random.Random) -> dict:
    p = _person()
    o = rng.choice(ORGS)
    g = rng.choice(GPES)
    d = rng.choice(DATES)
    t = rng.choice(TIMES)
    m = rng.choice(MONEY)
    pct = rng.choice(PERCENTS)
    lang = rng.choice(LANGS)
    product = rng.choice(PRODUCTS)
    work = rng.choice(WORKS)
    norp = rng.choice(NORP)

    templates = [
        (f"{p} presented the product {product} {t} at {o} in {g}.",
         [(p, "PERSON"), (product, "PRODUCT"), (o, "ORG"), (g, "GPE")]),
        (f"{p} authored {work}, which sold over {pct} more copies in {g}.",
         [(p, "PERSON"), (work, "WORK_OF_ART"), (pct, "PERCENT"), (g, "GPE")]),
        (f"The majority of {norp} in {g} speak {lang} as their first language.",
         [(norp, "NORP"), (g, "GPE"), (lang, "LANGUAGE")]),
        (f"{o} introduced {product} on {d} at a price of {m}.",
         [(o, "ORG"), (product, "PRODUCT"), (d, "DATE"), (m, "MONEY")]),
        (f"{p} from {o} will speak at the conference in {g} on {d} {t}.",
         [(p, "PERSON"), (o, "ORG"), (g, "GPE"), (d, "DATE")]),
    ]
    text, ents = rng.choice(templates)
    return _make(text, ents)


# ── main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic NER training sentences.")
    parser.add_argument("--n", type=int, default=5000, help="Number of sentences to generate")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=Path("data/synthetic_ner.jsonl"))
    args = parser.parse_args()

    rng = random.Random(args.seed)
    generators = [
        (_gen_person_sentence, 0.35),
        (_gen_org_sentence, 0.30),
        (_gen_date_money_sentence, 0.20),
        (_gen_mixed_sentence, 0.15),
    ]

    fns, weights = zip(*generators)
    records: list[dict] = []
    seen_texts: set[str] = set()

    while len(records) < args.n:
        gen = rng.choices(fns, weights=weights, k=1)[0]
        record = gen(rng)
        if record["text"] in seen_texts:
            continue
        if not record["entities"]:
            continue
        seen_texts.add(record["text"])
        records.append(record)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"✓ Wrote {len(records)} synthetic examples → {args.output}")
    label_counts: dict[str, int] = {}
    for r in records:
        for _, _, lbl in r["entities"]:
            label_counts[lbl] = label_counts.get(lbl, 0) + 1
    for lbl, cnt in sorted(label_counts.items()):
        print(f"  {lbl}: {cnt}")


if __name__ == "__main__":
    main()
