"""Data preparation for all four tasks (spec §11, stage 3).

Downloads HF datasets (SQL, math, translation) and generates synthetic JSON data.
Formats everything into uniform JSONL files with train/eval splits.
Uses overlap hashing from common.py to guarantee zero train/eval contamination.

Run:  .venv/bin/python -m data.prepare
"""

from __future__ import annotations

import json
import os
import random
import sys
from pathlib import Path

# Ensure project root is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import load_config, set_all_seeds, overlap_hash  # noqa: E402

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def save_jsonl(records: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"  saved {len(records):,} examples → {path}")


def split_train_eval(
    records: list[dict],
    n_train: int,
    n_eval: int,
    seed: int,
) -> tuple[list[dict], list[dict]]:
    """Shuffle, deduplicate by overlap hash, then split."""
    rng = random.Random(seed)
    rng.shuffle(records)

    seen_hashes: set[str] = set()
    deduped: list[dict] = []
    for rec in records:
        h = overlap_hash(rec["input"] + " ||| " + rec["output"])
        if h not in seen_hashes:
            seen_hashes.add(h)
            deduped.append(rec)

    total_needed = n_train + n_eval
    if len(deduped) < total_needed:
        print(f"  WARNING: only {len(deduped)} unique examples, need {total_needed}")
        # Prioritize eval; train gets the remainder
        n_eval = min(n_eval, len(deduped))
        n_train = min(n_train, len(deduped) - n_eval)

    eval_set = deduped[:n_eval]
    train_set = deduped[n_eval : n_eval + n_train]

    # Final overlap check
    train_hashes = {overlap_hash(r["input"] + " ||| " + r["output"]) for r in train_set}
    eval_hashes = {overlap_hash(r["input"] + " ||| " + r["output"]) for r in eval_set}
    overlap = train_hashes & eval_hashes
    assert len(overlap) == 0, f"Train/eval overlap detected: {len(overlap)} examples"

    return train_set, eval_set


# ---------------------------------------------------------------------------
# SQL: b-mc2/sql-create-context
# ---------------------------------------------------------------------------

def prepare_sql(cfg: dict) -> list[dict]:
    from datasets import load_dataset
    print("\n[SQL] Loading b-mc2/sql-create-context...")
    ds = load_dataset("b-mc2/sql-create-context", split="train")
    instruction = cfg["tasks"]["sql"]["instruction"]

    records = []
    for row in ds:
        inp = f"Schema:\n{row['context']}\n\nQuestion: {row['question']}"
        out = row["answer"]
        records.append({"task": "sql", "instruction": instruction, "input": inp, "output": out})

    print(f"  loaded {len(records):,} raw examples")
    return records


# ---------------------------------------------------------------------------
# Math: openai/gsm8k (NOTE: bare "gsm8k" fails in datasets 5.0)
# ---------------------------------------------------------------------------

def prepare_math(cfg: dict) -> list[dict]:
    from datasets import load_dataset
    print("\n[Math] Loading openai/gsm8k...")
    ds = load_dataset("openai/gsm8k", "main", split="train")
    instruction = cfg["tasks"]["math"]["instruction"]

    records = []
    for row in ds:
        inp = row["question"]
        out = row["answer"]
        records.append({"task": "math", "instruction": instruction, "input": inp, "output": out})

    print(f"  loaded {len(records):,} raw examples")
    return records


# ---------------------------------------------------------------------------
# Translation: ai4bharat/samanantar (ta subset), Tamil→English
# NOTE: src=English, tgt=Tamil in this dataset, so we SWAP for Ta→En
# ---------------------------------------------------------------------------

def prepare_translation(cfg: dict, seed: int) -> list[dict]:
    from datasets import load_dataset
    print("\n[Translation] Loading ai4bharat/samanantar (ta)...")
    # This dataset is huge, so use streaming + take
    ds = load_dataset("ai4bharat/samanantar", "ta", split="train", streaming=True)
    instruction = cfg["tasks"]["translation"]["instruction"]

    # We need 8000 + 300 + margin. Take more with a seeded sample.
    # Streaming doesn't support shuffle, so take a big chunk and shuffle locally.
    max_take = 20000
    records = []
    for i, row in enumerate(ds):
        if i >= max_take:
            break
        inp = row["tgt"]  # Tamil (input for Ta→En)
        out = row["src"]  # English (output)
        if inp and out and inp.strip() and out.strip():
            records.append({"task": "translation", "instruction": instruction,
                            "input": inp.strip(), "output": out.strip()})

    print(f"  loaded {len(records):,} raw examples (from first {max_take})")
    return records


# ---------------------------------------------------------------------------
# JSON: Synthetic deterministic generation
# ---------------------------------------------------------------------------

_SCHEMAS = [
    {
        "name": "product",
        "schema": {
            "type": "object",
            "properties": {
                "product_name": {"type": "string"},
                "price": {"type": "number"},
                "currency": {"type": "string"},
                "in_stock": {"type": "boolean"},
            },
            "required": ["product_name", "price", "currency"],
        },
        "templates": [
            'The {product_name} is currently priced at {price} {currency}. {stock_text}',
            'We have the {product_name} available for {price} {currency}. {stock_text}',
            'Looking for the {product_name}? It costs {price} {currency}. {stock_text}',
        ],
        "gen": lambda rng: {
            "product_name": rng.choice(["Widget Pro", "Turbo Blender", "Smart Lamp", "Eco Bottle",
                                         "Power Bank X", "Ultra Mouse", "Flex Keyboard", "Mini Speaker",
                                         "Cloud Router", "Solar Charger"]),
            "price": round(rng.uniform(5, 500), 2),
            "currency": rng.choice(["USD", "EUR", "GBP", "INR"]),
            "in_stock": rng.choice([True, False]),
        },
    },
    {
        "name": "person",
        "schema": {
            "type": "object",
            "properties": {
                "full_name": {"type": "string"},
                "age": {"type": "integer"},
                "occupation": {"type": "string"},
                "city": {"type": "string"},
            },
            "required": ["full_name", "age", "occupation"],
        },
        "templates": [
            '{full_name}, age {age}, works as a {occupation} in {city}.',
            'Meet {full_name}. At {age} years old, this {occupation} lives in {city}.',
            '{full_name} is a {age}-year-old {occupation} based in {city}.',
        ],
        "gen": lambda rng: {
            "full_name": rng.choice(["Alice Johnson", "Raj Patel", "Maria Garcia", "Chen Wei",
                                      "James Okafor", "Fatima Al-Rashid", "Liam O'Brien", "Yuki Tanaka",
                                      "Sofia Rossi", "Priya Sharma"]),
            "age": rng.randint(22, 65),
            "occupation": rng.choice(["software engineer", "teacher", "doctor", "artist",
                                       "chef", "architect", "journalist", "researcher"]),
            "city": rng.choice(["New York", "London", "Mumbai", "Tokyo", "Berlin",
                                "Sydney", "Toronto", "São Paulo", "Lagos", "Chennai"]),
        },
    },
    {
        "name": "event",
        "schema": {
            "type": "object",
            "properties": {
                "event_name": {"type": "string"},
                "date": {"type": "string"},
                "location": {"type": "string"},
                "attendees": {"type": "integer"},
            },
            "required": ["event_name", "date", "location"],
        },
        "templates": [
            'The {event_name} is scheduled for {date} at {location}. Expected attendance: {attendees} people.',
            'Mark your calendar: {event_name} on {date}, held at {location}. About {attendees} guests expected.',
            '{event_name} will take place at {location} on {date}, with approximately {attendees} participants.',
        ],
        "gen": lambda rng: {
            "event_name": rng.choice(["Tech Summit 2026", "Annual Gala", "Startup Pitch Day",
                                       "AI Workshop", "Design Conference", "Music Festival",
                                       "Science Fair", "Book Launch", "Charity Run", "Film Premiere"]),
            "date": f"2026-{rng.randint(1,12):02d}-{rng.randint(1,28):02d}",
            "location": rng.choice(["Convention Center", "Grand Hotel", "City Park", "University Hall",
                                     "Downtown Arena", "Beach Resort", "Art Gallery", "Community Center"]),
            "attendees": rng.randint(50, 5000),
        },
    },
    {
        "name": "book",
        "schema": {
            "type": "object",
            "properties": {
                "title": {"type": "string"},
                "author": {"type": "string"},
                "genre": {"type": "string"},
                "year": {"type": "integer"},
                "rating": {"type": "number"},
            },
            "required": ["title", "author", "genre"],
        },
        "templates": [
            '"{title}" by {author} is a {genre} novel published in {year}. It has a rating of {rating}/5.',
            '{author}\'s {genre} book "{title}" ({year}) received a {rating}-star rating.',
            'In {year}, {author} released "{title}", a {genre} work rated {rating} out of 5.',
        ],
        "gen": lambda rng: {
            "title": rng.choice(["The Silent Code", "Echoes of Tomorrow", "Red Horizon",
                                  "The Glass Garden", "Midnight Algorithm", "Quantum Dreams",
                                  "Iron Compass", "The Last Thread", "Cloud Atlas", "Neon Jungle"]),
            "author": rng.choice(["A. Thompson", "R. Gupta", "K. Nakamura", "L. Santos",
                                   "M. Dubois", "S. Okoye", "J. Mueller", "P. Johansson"]),
            "genre": rng.choice(["science fiction", "mystery", "literary fiction", "thriller",
                                  "fantasy", "historical fiction", "romance", "horror"]),
            "year": rng.randint(1990, 2026),
            "rating": round(rng.uniform(2.0, 5.0), 1),
        },
    },
    {
        "name": "restaurant",
        "schema": {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "cuisine": {"type": "string"},
                "price_range": {"type": "string"},
                "rating": {"type": "number"},
                "vegetarian_friendly": {"type": "boolean"},
            },
            "required": ["name", "cuisine", "price_range"],
        },
        "templates": [
            '{name} serves {cuisine} cuisine. Price range: {price_range}. Rated {rating}/5. {veg_text}',
            'Check out {name} for {cuisine} food. It\'s in the {price_range} range, rated {rating} stars. {veg_text}',
            'The {cuisine} restaurant {name} ({price_range}) has a {rating}-star rating. {veg_text}',
        ],
        "gen": lambda rng: {
            "name": rng.choice(["Golden Spoon", "Casa Luna", "Sakura Garden", "The Green Plate",
                                 "Fire & Smoke", "Ocean Breeze", "Spice Route", "Le Petit Chef",
                                 "Noodle House", "Copper Kettle"]),
            "cuisine": rng.choice(["Italian", "Japanese", "Indian", "Mexican", "Thai",
                                    "French", "Chinese", "Mediterranean", "Korean", "American"]),
            "price_range": rng.choice(["$", "$$", "$$$", "$$$$"]),
            "rating": round(rng.uniform(3.0, 5.0), 1),
            "vegetarian_friendly": rng.choice([True, False]),
        },
    },
    {
        "name": "weather",
        "schema": {
            "type": "object",
            "properties": {
                "city": {"type": "string"},
                "temperature_c": {"type": "number"},
                "condition": {"type": "string"},
                "humidity_pct": {"type": "integer"},
            },
            "required": ["city", "temperature_c", "condition"],
        },
        "templates": [
            'Weather update for {city}: {temperature_c}°C, {condition}. Humidity at {humidity_pct}%.',
            'In {city}, it\'s currently {temperature_c} degrees Celsius with {condition} conditions. Humidity: {humidity_pct}%.',
            '{city} forecast: {condition}, temperature {temperature_c}°C, {humidity_pct}% humidity.',
        ],
        "gen": lambda rng: {
            "city": rng.choice(["Paris", "Seoul", "Nairobi", "Dubai", "Stockholm",
                                "Rio", "Vancouver", "Bangkok", "Cairo", "Melbourne"]),
            "temperature_c": round(rng.uniform(-10, 45), 1),
            "condition": rng.choice(["sunny", "cloudy", "rainy", "partly cloudy",
                                      "snowy", "foggy", "windy", "clear", "thunderstorm"]),
            "humidity_pct": rng.randint(10, 100),
        },
    },
    {
        "name": "job_posting",
        "schema": {
            "type": "object",
            "properties": {
                "job_title": {"type": "string"},
                "company": {"type": "string"},
                "salary_usd": {"type": "integer"},
                "remote": {"type": "boolean"},
            },
            "required": ["job_title", "company", "salary_usd"],
        },
        "templates": [
            '{company} is hiring a {job_title}. Salary: ${salary_usd}/year. {remote_text}',
            'Job opening: {job_title} at {company}, paying ${salary_usd} annually. {remote_text}',
            '{company} seeks a {job_title} (${salary_usd}/yr). {remote_text}',
        ],
        "gen": lambda rng: {
            "job_title": rng.choice(["Data Scientist", "Frontend Developer", "Product Manager",
                                      "DevOps Engineer", "UX Designer", "ML Engineer",
                                      "Backend Developer", "QA Analyst", "CTO", "Tech Lead"]),
            "company": rng.choice(["Acme Corp", "TechNova", "CloudSync", "DataWave",
                                    "BrightPath", "NeuralEdge", "PixelForge", "QuantumLeap"]),
            "salary_usd": rng.randint(40000, 250000),
            "remote": rng.choice([True, False]),
        },
    },
    {
        "name": "movie",
        "schema": {
            "type": "object",
            "properties": {
                "title": {"type": "string"},
                "director": {"type": "string"},
                "year": {"type": "integer"},
                "genre": {"type": "string"},
                "duration_min": {"type": "integer"},
            },
            "required": ["title", "director", "year"],
        },
        "templates": [
            '"{title}" ({year}), directed by {director}, is a {genre} film running {duration_min} minutes.',
            'Director {director}\'s {year} {genre} movie "{title}" clocks in at {duration_min} minutes.',
            'The {genre} film "{title}" by {director} was released in {year} with a runtime of {duration_min} min.',
        ],
        "gen": lambda rng: {
            "title": rng.choice(["Stellar Drift", "The Vanishing Point", "Iron Bloom",
                                  "Glass Meridian", "Echo Chamber", "Crimson Tide II",
                                  "The Quiet Storm", "Parallel Lines", "Neon Nights", "The Verdict"]),
            "director": rng.choice(["A. Kurosawa", "S. Spielberg", "C. Nolan", "G. del Toro",
                                     "D. Villeneuve", "B. Lee", "W. Anderson", "K. Bigelow"]),
            "year": rng.randint(1990, 2026),
            "genre": rng.choice(["sci-fi", "drama", "action", "comedy", "thriller",
                                  "animation", "documentary", "horror"]),
            "duration_min": rng.randint(80, 200),
        },
    },
    {
        "name": "flight",
        "schema": {
            "type": "object",
            "properties": {
                "airline": {"type": "string"},
                "flight_number": {"type": "string"},
                "departure": {"type": "string"},
                "arrival": {"type": "string"},
                "price_usd": {"type": "number"},
            },
            "required": ["airline", "flight_number", "departure", "arrival"],
        },
        "templates": [
            '{airline} flight {flight_number} from {departure} to {arrival}. Ticket price: ${price_usd}.',
            'Book {airline} {flight_number}: {departure} → {arrival} for ${price_usd}.',
            'Flight {flight_number} ({airline}) departs {departure}, arrives {arrival}. Cost: ${price_usd}.',
        ],
        "gen": lambda rng: {
            "airline": rng.choice(["SkyWing", "AeroNova", "Pacific Air", "EuroJet",
                                    "AtlasAir", "Horizon", "Nimbus Airlines", "SwiftAir"]),
            "flight_number": rng.choice(["AA", "BA", "EJ", "PA", "SW", "NM"]) + str(rng.randint(100, 999)),
            "departure": rng.choice(["JFK", "LAX", "LHR", "CDG", "NRT", "SIN", "DXB", "SFO"]),
            "arrival": rng.choice(["ORD", "FRA", "ICN", "SYD", "BOM", "GRU", "YYZ", "HND"]),
            "price_usd": round(rng.uniform(100, 2500), 2),
        },
    },
    {
        "name": "health_record",
        "schema": {
            "type": "object",
            "properties": {
                "patient_name": {"type": "string"},
                "diagnosis": {"type": "string"},
                "medication": {"type": "string"},
                "follow_up_days": {"type": "integer"},
            },
            "required": ["patient_name", "diagnosis", "medication"],
        },
        "templates": [
            'Patient {patient_name} diagnosed with {diagnosis}. Prescribed {medication}. Follow-up in {follow_up_days} days.',
            '{patient_name} presents with {diagnosis}. Treatment: {medication}. Next visit: {follow_up_days} days.',
            'Diagnosis for {patient_name}: {diagnosis}. Medication: {medication}. Review in {follow_up_days} days.',
        ],
        "gen": lambda rng: {
            "patient_name": rng.choice(["John D.", "Anita R.", "Kenji M.", "Sara L.",
                                         "Omar H.", "Elena V.", "David K.", "Mei C.",
                                         "Tomás F.", "Grace N."]),
            "diagnosis": rng.choice(["mild hypertension", "type 2 diabetes", "seasonal allergies",
                                      "vitamin D deficiency", "acute bronchitis", "migraine",
                                      "iron deficiency anemia", "generalized anxiety"]),
            "medication": rng.choice(["Lisinopril 10mg", "Metformin 500mg", "Cetirizine 10mg",
                                       "Cholecalciferol 2000IU", "Amoxicillin 500mg",
                                       "Sumatriptan 50mg", "Ferrous sulfate 325mg", "Sertraline 50mg"]),
            "follow_up_days": rng.randint(7, 90),
        },
    },
]


def prepare_json(cfg: dict, seed: int) -> list[dict]:
    print("\n[JSON] Generating synthetic extraction data...")
    rng = random.Random(seed)
    instruction = cfg["tasks"]["json"]["instruction"]
    n_schemas = cfg["data"]["sources"]["json"]["n_schemas"]
    schemas = _SCHEMAS[:n_schemas]

    # Target: ~9000 examples (enough for 8000 train + 300 eval + margin)
    per_schema = 9200 // len(schemas)
    records = []

    for schema_def in schemas:
        schema = schema_def["schema"]
        templates = schema_def["templates"]
        gen_fn = schema_def["gen"]

        for _ in range(per_schema):
            gold = gen_fn(rng)
            tmpl = rng.choice(templates)

            # Build text passage from template
            extra_kwargs = {}
            if "in_stock" in gold:
                extra_kwargs["stock_text"] = "It is in stock." if gold["in_stock"] else "Currently out of stock."
            if "vegetarian_friendly" in gold:
                extra_kwargs["veg_text"] = "Vegetarian options available." if gold["vegetarian_friendly"] else "Limited vegetarian options."
            if "remote" in gold:
                extra_kwargs["remote_text"] = "Remote work available." if gold["remote"] else "On-site position."

            try:
                text = tmpl.format(**gold, **extra_kwargs)
            except KeyError:
                text = tmpl.format(**{**gold, **extra_kwargs, "stock_text": "", "veg_text": "", "remote_text": ""})

            schema_json = json.dumps(schema, ensure_ascii=False)
            inp = f"Schema:\n{schema_json}\n\nText: {text}"
            out = json.dumps(gold, ensure_ascii=False)
            records.append({"task": "json", "instruction": instruction, "input": inp, "output": out})

    print(f"  generated {len(records):,} synthetic examples from {len(schemas)} schemas")
    return records


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    cfg = load_config()
    seed = cfg["seed"]
    set_all_seeds(seed)

    n_train = cfg["data"]["max_train_examples"]  # 8000
    n_eval = cfg["data"]["eval_items"]            # 300
    out_dir = Path(cfg["data"]["processed_dir"])

    tasks = {
        "sql": lambda: prepare_sql(cfg),
        "math": lambda: prepare_math(cfg),
        "translation": lambda: prepare_translation(cfg, seed),
        "json": lambda: prepare_json(cfg, seed),
    }

    for task_name, loader_fn in tasks.items():
        records = loader_fn()
        train_set, eval_set = split_train_eval(records, n_train, n_eval, seed)
        save_jsonl(train_set, out_dir / f"{task_name}_train.jsonl")
        save_jsonl(eval_set, out_dir / f"{task_name}_eval.jsonl")

    print("\n✅ Data preparation complete!")
    # Summary
    for task_name in tasks:
        train_path = out_dir / f"{task_name}_train.jsonl"
        eval_path = out_dir / f"{task_name}_eval.jsonl"
        tc = sum(1 for _ in open(train_path))
        ec = sum(1 for _ in open(eval_path))
        print(f"  {task_name}: {tc:,} train / {ec:,} eval")


if __name__ == "__main__":
    main()
