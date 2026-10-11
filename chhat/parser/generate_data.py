"""Synthetic Pakistani address generator with token-level labels.

Generates labeled training data for the Chhat address parser.
Labels (BIO scheme): HOUSE, STREET, BLOCK, COLONY, CITY, O.

Run: python -m chhat.parser.generate_data --n 8000 --out data/addresses.jsonl --seed 42
"""
from __future__ import annotations

import argparse
import json
import random

# ----------------------------------------------------------------------------
# Vocabulary
# ----------------------------------------------------------------------------

LAHORE_COLONIES = [
    "Nishat Colony", "DHA Phase 1", "DHA Phase 2", "DHA Phase 3", "DHA Phase 4",
    "DHA Phase 5", "DHA Phase 6", "DHA Phase 7", "DHA Phase 8", "DHA Phase 9",
    "Model Town", "Johar Town", "Gulberg I", "Gulberg II", "Gulberg III",
    "Gulberg IV", "Gulberg V", "Wapda Town", "Valencia Town", "Bahria Town",
    "Lahore Cantt", "Samanabad", "Iqbal Town", "Faisal Town", "Garden Town",
    "Shadman", "Ichhra", "Township", "Green Town", "Sabzazar", "Harbanspura",
    "Mughalpura", "Baghbanpura", "Shahdara", "Walton", "Cavalry Ground",
    "Askari 11", "Lake City", "Etihad Town", "EME Society", "State Life Society",
    "Punjab Society", "NFC Society", "Wapda City", "Al Kareem Garden",
]

OTHER_CITIES = [
    ("Karachi", ["DHA Phase 6", "Gulshan-e-Iqbal", "North Nazimabad", "Malir Cantt",
                 "Bahria Town", "Scheme 33", "Clifton Block 5", "PECHS"]),
    ("Islamabad", ["F-7", "G-13", "DHA Phase 2", "Bahria Town", "E-11", "I-8",
                   "Blue Area", "Margalla Town"]),
    ("Rawalpindi", ["Satellite Town", "Chaklala Scheme III", "Bahria Town",
                    "DHA Phase 1", "Westridge"]),
    ("Faisalabad", ["D Ground", "Model City", "Wadi-e-Sitara", "DHA",
                    "Gulberg City"]),
    ("Multan", ["Bosan Road", "DHA Multan", "Gulgasht Colony", "Shah Rukn-e-Alam Colony"]),
    ("Peshawar", ["Hayatabad Phase 2", "DHA Peshawar", "University Town"]),
]

STREET_WORDS = ["Street", "St", "Road", "Rd", "Lane", "Boulevard", "Avenue"]
HOUSE_WORDS = ["House", "H", "Plot", "Flat", "Shop"]
BLOCK_WORDS = ["Block", "Sector"]
NO_WORDS = ["No", "No.", "Number"]


def _house_token(rng: random.Random) -> tuple[str, str]:
    """Return (text, label-string) for a house token, e.g. '123-A'."""
    num = rng.randint(1, 999)
    suffix = rng.choice(["", "", "", f"-{rng.choice('ABCDEFGH')}", f"/{rng.randint(1,9)}"])
    return f"{num}{suffix}", "HOUSE"


def _block_token(rng: random.Random) -> str:
    return rng.choice("ABCDEFGHJKLMNPRST")


def _street_token(rng: random.Random) -> str:
    n = rng.randint(1, 60)
    return f"{n}"


def _tokenize(text: str) -> list[str]:
    return text.replace(",", " ,").split()


def _label_tokens(tokens: list[str], spans: list[tuple[int, int, str]]) -> list[str]:
    labels = ["O"] * len(tokens)
    for start, end, tag in spans:
        for i in range(start, end):
            labels[i] = f"{'B' if i == start else 'I'}-{tag}"
    return labels


def generate_one(rng: random.Random) -> tuple[list[str], list[str]]:
    """Generate one (tokens, labels) pair."""
    lahore = rng.random() < 0.62
    if lahore:
        colony, city = rng.choice(LAHORE_COLONIES), "Lahore"
    else:
        city, colonies = rng.choice(OTHER_CITIES)
        colony = rng.choice(colonies)

    style = rng.random()
    parts: list[tuple[str, str | None]] = []  # (text, entity or None)

    if style < 0.45:
        # "House 123, Street 4, Block C, Nishat Colony, Lahore"
        h, _ = _house_token(rng)
        parts += [(rng.choice(HOUSE_WORDS), None), (h, "HOUSE"), (",", None),
                  (rng.choice(STREET_WORDS), None), (_street_token(rng), "STREET"), (",", None)]
        if rng.random() < 0.75:
            parts += [(rng.choice(BLOCK_WORDS), None), (_block_token(rng), "BLOCK"), (",", None)]
        parts += [(colony, "COLONY"), (",", None), (city, "CITY")]
    elif style < 0.70:
        # "123-A, Main Boulevard, DHA Phase 5, Lahore"
        h, _ = _house_token(rng)
        parts += [(h, "HOUSE"), (",", None)]
        if rng.random() < 0.5:
            parts += [(rng.choice(["Main", "Central"]), None)]
        parts += [(rng.choice(STREET_WORDS), "STREET"), (",", None),
                  (colony, "COLONY"), (",", None), (city, "CITY")]
    elif style < 0.85:
        # "House No. 45-B, Street No. 12, Model Town, Lahore"
        h, _ = _house_token(rng)
        hw = rng.choice(HOUSE_WORDS)
        parts += [(hw, None), (rng.choice(NO_WORDS), None), (h, "HOUSE"), (",", None),
                  (rng.choice(STREET_WORDS), None), (rng.choice(NO_WORDS), None),
                  (_street_token(rng), "STREET"), (",", None),
                  (colony, "COLONY"), (",", None), (city, "CITY")]
    elif style < 0.93:
        # "Plot 12, Block H, Johar Town, Lahore" (no street)
        h, _ = _house_token(rng)
        parts += [(rng.choice(["Plot", "House"]), None), (h, "HOUSE"), (",", None),
                  (rng.choice(BLOCK_WORDS), None), (_block_token(rng), "BLOCK"), (",", None),
                  (colony, "COLONY"), (",", None), (city, "CITY")]
    else:
        # "Flat 3, Building 9, Gulberg III, Lahore"
        parts += [(rng.choice(["Flat", "Apartment", "Portion"]), None),
                  (str(rng.randint(1, 40)), "HOUSE"), (",", None),
                  ("Building", None), (str(rng.randint(1, 30)), "BLOCK"), (",", None),
                  (colony, "COLONY"), (",", None), (city, "CITY")]

    # flatten multi-word entities into tokens
    tokens: list[str] = []
    spans: list[tuple[int, int, str]] = []
    for text, entity in parts:
        words = text.split()
        if entity:
            spans.append((len(tokens), len(tokens) + len(words), entity))
        tokens.extend(words)
    return tokens, _label_tokens(tokens, spans)


def _build(tokens_with_labels: list[tuple[str, str | None]]) -> tuple[list[str], list[str]]:
    """Build (tokens, BIO labels) from (word, entity-or-None) pairs.

    Convention: descriptor words (House, Street, Makan, Gali, No., ...) are O;
    only the value tokens carry entity tags (HOUSE/STREET/BLOCK/COLONY/CITY).
    """
    tokens: list[str] = []
    spans: list[tuple[int, int, str]] = []
    for text, entity in tokens_with_labels:
        words = text.split()
        if entity:
            spans.append((len(tokens), len(tokens) + len(words), entity))
        tokens.extend(words)
    return tokens, _label_tokens(tokens, spans)


# Urdu-mixed descriptor words people actually type (labeled O, like "House")
URDU_HOUSE = ["Makan", "Makaan", "Mkn"]
URDU_STREET = ["Gali", "Gali No", "Mohallah"]
MESSY_HOUSE_PRE = ["H#", "H.No.", "H.No", "H #", "House No", "House#", "Plot No",
                   "P#"]
MESSY_STREET_PRE = ["St#", "St. No.", "St No", "Street#", "Gali#"]


def generate_messy(rng: random.Random) -> tuple[list[str], list[str]]:
    """Messy real-world address variants. Same label convention as clean data."""
    lahore = rng.random() < 0.62
    if lahore:
        colony, city = rng.choice(LAHORE_COLONIES), "Lahore"
    else:
        city, colonies = rng.choice(OTHER_CITIES)
        colony = rng.choice(colonies)

    h, _ = _house_token(rng)
    style = rng.random()
    comma = "," if rng.random() < 0.45 else ""

    def join(items: list[tuple[str, str | None]]) -> list[tuple[str, str | None]]:
        out: list[tuple[str, str | None]] = []
        for i, it in enumerate(items):
            out.append(it)
            if comma and i < len(items) - 1:
                out.append((comma, None))
        return out

    if style < 0.20:
        # no commas at all: "House 123 Street 4 Block C Nishat Colony Lahore"
        items = [(rng.choice(HOUSE_WORDS), None), (h, "HOUSE"),
                 (rng.choice(STREET_WORDS), None), (_street_token(rng), "STREET")]
        if rng.random() < 0.6:
            items += [(rng.choice(BLOCK_WORDS), None), (_block_token(rng), "BLOCK")]
        items += [(colony, "COLONY"), (city, "CITY")]
        pairs = items
    elif style < 0.38:
        # hash/prefix forms: "H#123 St#4 Nishat Colony Lahore", "H# 200 St # 11 ..."
        hp = rng.choice(MESSY_HOUSE_PRE).replace(" ", "")
        sp = rng.choice(MESSY_STREET_PRE).replace(" ", "")
        if rng.random() < 0.3:
            # spaced hash: "H# 200", "St # 11"
            pairs = join([("H#", None), (h, "HOUSE"),
                          ("St", None), ("#", None), (_street_token(rng), "STREET"),
                          (colony, "COLONY"), (city, "CITY")])
        else:
            pairs = join([(hp + h, "HOUSE"), (sp + _street_token(rng), "STREET"),
                          (colony, "COLONY"), (city, "CITY")])
    elif style < 0.55:
        # Urdu-mixed: "Makan 123 Gali 4 ...", "Makan No 78 ...", "Gali 10 Makan 55 ..."
        uh = rng.choice(URDU_HOUSE)
        us = rng.choice(URDU_STREET).split()[0]
        r = rng.random()
        if r < 0.35:
            pairs = join([(uh, None), ("No", None), (h, "HOUSE"),
                          (colony, "COLONY"), (city, "CITY")])
        elif r < 0.55:
            # reversed order: street first
            pairs = join([(us, None), (_street_token(rng), "STREET"),
                          (uh, None), (h, "HOUSE"),
                          (colony, "COLONY"), (city, "CITY")])
        else:
            pairs = join([(uh, None), (h, "HOUSE"),
                          (us, None), (_street_token(rng), "STREET"),
                          (colony, "COLONY"), (city, "CITY")])
    elif style < 0.68:
        # abbreviations + periods: "H 123, St. 4, Blk C, Nishat Colony, Lahore"
        pairs = join([("H", None), (h, "HOUSE"),
                      (rng.choice(["St.", "St", "Rd."]), None),
                      (_street_token(rng), "STREET"),
                      (rng.choice(["Blk", "Block", "Sec"]), None),
                      (_block_token(rng), "BLOCK"),
                      (colony, "COLONY"), (city, "CITY")])
    elif style < 0.80:
        # lowercase everything: "house 123 street 4 block c nishat colony lahore"
        # "house no 9 street no 11 shadman lahore"
        if rng.random() < 0.4:
            items = [("house", None), ("no", None), (h.lower(), "HOUSE"),
                     ("street", None), ("no", None), (_street_token(rng), "STREET"),
                     (colony.lower(), "COLONY"), (city.lower(), "CITY")]
        else:
            items = [("house", None), (h.lower(), "HOUSE"),
                     ("street", None), (_street_token(rng), "STREET")]
            if rng.random() < 0.6:
                items += [("block", None), (_block_token(rng).lower(), "BLOCK")]
            items += [(colony.lower(), "COLONY"), (city.lower(), "CITY")]
        pairs = join(items)
    elif style < 0.86:
        # house-only, no street/block: "123/4 Nishat Colony Lahore"
        pairs = join([(h, "HOUSE"), (colony, "COLONY"), (city, "CITY")])
    elif style < 0.93:
        # sector form: "12-B, St 7, Sector G-13, Islamabad"
        pairs = join([(_house_token(rng)[0], "HOUSE"),
                      (rng.choice(["St", "St."]), None),
                      (_street_token(rng), "STREET"),
                      ("Sector", None),
                      (f"{_block_token(rng)}-{rng.randint(1, 17)}", "BLOCK"),
                      (colony, "COLONY"), (city, "CITY")])
    elif style < 0.97:
        # attached prefixes: "H.No.7 St.No.3 Block B Valencia Town Lahore"
        # (whole token is the entity; prefix stripped in post-processing)
        pairs = join([(f"H.No.{h}", "HOUSE"), (f"St.No.{_street_token(rng)}", "STREET"),
                      ("Block", None), (_block_token(rng), "BLOCK"),
                      (colony, "COLONY"), (city, "CITY")])
    else:
        # fraction / plot-no / landmark forms
        r = rng.random()
        if not lahore and r < 0.4:
            pairs = join([(f"{h}/{rng.randint(1,9)}", "HOUSE"),
                          ("Sector", None), (f"{_block_token(rng)}-{rng.randint(1,17)}", "BLOCK"),
                          (colony, "COLONY"), (city, "CITY")])
        elif r < 0.7:
            pairs = join([("Plot", None), ("No", None), (h, "HOUSE"),
                          ("Block", None), (_block_token(rng), "BLOCK"),
                          (colony, "COLONY"), (city, "CITY")])
        else:
            # landmark + trailing country: "Near Main Bazaar House 45 ..."
            pairs = join([("Near", None), ("Main", None), ("Bazaar", None),
                          ("House", None), (h, "HOUSE"),
                          ("Street", None), (_street_token(rng), "STREET"),
                          (colony, "COLONY"), (city, "CITY"), ("Pakistan", None)])

    return _build(pairs)


def generate(n: int, seed: int = 42, messy_frac: float = 0.35) -> list[dict]:
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        if rng.random() < messy_frac:
            tokens, labels = generate_messy(rng)
        else:
            tokens, labels = generate_one(rng)
        out.append({"tokens": tokens, "labels": labels,
                    "text": " ".join(tokens).replace(" ,", ",")})
    return out


# ----------------------------------------------------------------------------
# Multi-city expansion (Wave 8): city-specific address formats so judges can
# type ANY Pakistani address, not just Lahore.
# ----------------------------------------------------------------------------

def _khi_house(rng: random.Random) -> str:
    """Karachi house numbers: R-123, 5/12, 23-A, plain numbers."""
    r = rng.random()
    if r < 0.4:
        return f"R-{rng.randint(1, 999)}"
    if r < 0.65:
        return f"{rng.randint(1, 60)}/{rng.randint(1, 40)}"
    if r < 0.85:
        return f"{rng.randint(1, 999)}-{rng.choice('ABCD')}"
    return str(rng.randint(1, 999))


def _isb_sector(rng: random.Random) -> str:
    """Islamabad sector: G-13/2, F-10/3, E-11/4, I-8/1."""
    return f"{rng.choice(['G', 'F', 'E', 'I'])}-{rng.randint(8, 17)}/{rng.randint(1, 4)}"


def generate_multicity_one(rng: random.Random, city: str) -> tuple[list[str], list[str]]:
    """One labeled address in a city-specific format. Same BIO convention."""
    h = _house_token(rng)[0]
    comma = "," if rng.random() < 0.7 else ""

    def join(items):
        out = []
        for i, it in enumerate(items):
            out.append(it)
            if comma and i < len(items) - 1:
                out.append((comma, None))
        return out

    if city == "Karachi":
        style = rng.random()
        if style < 0.30:
            # "House R-123, Block 13, Gulshan-e-Iqbal, Karachi"
            pairs = join([(rng.choice(HOUSE_WORDS), None), (_khi_house(rng), "HOUSE"),
                          ("Block", None), (str(rng.randint(1, 22)), "BLOCK"),
                          ("Gulshan-e-Iqbal", "COLONY"), ("Karachi", "CITY")])
        elif style < 0.50:
            # "Plot 45-C, Khayaban-e-Ittehad, DHA Phase 6, Karachi"
            pairs = join([("Plot", None), (f"{rng.randint(1,200)}-{rng.choice('ABC')}", "HOUSE"),
                          ("Khayaban-e-Ittehad", "STREET"),
                          (f"DHA Phase {rng.randint(2,8)}", "COLONY"), ("Karachi", "CITY")])
        elif style < 0.65:
            # "House 5/12, North Nazimabad, Karachi"
            pairs = join([(rng.choice(HOUSE_WORDS), None), (_khi_house(rng), "HOUSE"),
                          (rng.choice(["North Nazimabad", "Nazimabad", "Malir Cantt",
                                       "PECHS", "Scheme 33"]), "COLONY"),
                          ("Karachi", "CITY")])
        elif style < 0.80:
            # "House 23, Block 5, Clifton, Karachi"
            pairs = join([(rng.choice(HOUSE_WORDS), None), (h, "HOUSE"),
                          ("Block", None), (str(rng.randint(1, 9)), "BLOCK"),
                          ("Clifton", "COLONY"), ("Karachi", "CITY")])
        elif style < 0.90:
            # "Flat 12, Block 3, Gulshan-e-Iqbal, Karachi"
            pairs = join([("Flat", None), (str(rng.randint(1, 60)), "HOUSE"),
                          ("Block", None), (str(rng.randint(1, 22)), "BLOCK"),
                          ("Gulshan-e-Iqbal", "COLONY"), ("Karachi", "CITY")])
        else:
            # messy: "H#R-123 Block 13 Gulshan-e-Iqbal Karachi", "Makan R-45 ..."
            hp = rng.choice(["H#", "Plot#"])
            pairs = join([(hp + _khi_house(rng), "HOUSE"),
                          ("Block", None), (str(rng.randint(1, 22)), "BLOCK"),
                          ("Gulshan-e-Iqbal", "COLONY"), ("Karachi", "CITY")])
            if rng.random() < 0.4:
                # Urdu descriptor stays O: "Makan R-45, Block 7, ..."
                pairs = join([(rng.choice(URDU_HOUSE), None), (_khi_house(rng), "HOUSE"),
                              ("Block", None), (str(rng.randint(1, 22)), "BLOCK"),
                              ("Gulshan-e-Iqbal", "COLONY"), ("Karachi", "CITY")])
    elif city == "Islamabad":
        style = rng.random()
        if style < 0.35:
            # "House 123, Street 45, Sector G-13/2, Islamabad"
            pairs = join([(rng.choice(HOUSE_WORDS), None), (h, "HOUSE"),
                          ("Street", None), (_street_token(rng), "STREET"),
                          ("Sector", None), (_isb_sector(rng), "BLOCK"),
                          ("Islamabad", "CITY")])
        elif style < 0.60:
            # "House 8, Street 12, F-10/3, Islamabad" (bare sector = colony)
            pairs = join([(rng.choice(HOUSE_WORDS), None), (h, "HOUSE"),
                          ("Street", None), (_street_token(rng), "STREET"),
                          (_isb_sector(rng), "COLONY"), ("Islamabad", "CITY")])
        elif style < 0.75:
            # "Plot 22, E-11/4, Islamabad" (no street)
            pairs = join([("Plot", None), (h, "HOUSE"),
                          (_isb_sector(rng), "COLONY"), ("Islamabad", "CITY")])
        elif style < 0.90:
            # "House 56, Margalla Town, Islamabad"
            pairs = join([(rng.choice(HOUSE_WORDS), None), (h, "HOUSE"),
                          (rng.choice(["Margalla Town", "Blue Area", "DHA Phase 2",
                                       "Bahria Town"]), "COLONY"),
                          ("Islamabad", "CITY")])
        else:
            # messy: "H 123 St 45 G-13/2 Islamabad", "Makan 77 Gali 3 E-11/1 ..."
            pairs = join([("H", None), (h, "HOUSE"),
                          ("St", None), (_street_token(rng), "STREET"),
                          (_isb_sector(rng), "COLONY"), ("Islamabad", "CITY")])
    elif city == "Rawalpindi":
        style = rng.random()
        if style < 0.40:
            # "House 234, Block C, Satellite Town, Rawalpindi"
            pairs = join([(rng.choice(HOUSE_WORDS), None), (h, "HOUSE"),
                          ("Block", None), (_block_token(rng), "BLOCK"),
                          ("Satellite Town", "COLONY"), ("Rawalpindi", "CITY")])
        elif style < 0.65:
            # "House 56, Street 7, Bahria Town, Rawalpindi"
            pairs = join([(rng.choice(HOUSE_WORDS), None), (h, "HOUSE"),
                          ("Street", None), (_street_token(rng), "STREET"),
                          (rng.choice(["Bahria Town", "DHA Phase 1", "Westridge"]), "COLONY"),
                          ("Rawalpindi", "CITY")])
        elif style < 0.85:
            # "Plot 89, Chaklala Scheme III, Rawalpindi"
            pairs = join([("Plot", None), (h, "HOUSE"),
                          ("Chaklala Scheme III", "COLONY"), ("Rawalpindi", "CITY")])
        else:
            # messy: "H#234 Block C Satellite Town Rawalpindi"
            pairs = join([("H#" + h, "HOUSE"), ("Block", None),
                          (_block_token(rng), "BLOCK"),
                          ("Satellite Town", "COLONY"), ("Rawalpindi", "CITY")])
    else:  # Faisalabad
        style = rng.random()
        if style < 0.40:
            # "House 78, Block B, Madina Town, Faisalabad"
            pairs = join([(rng.choice(HOUSE_WORDS), None), (h, "HOUSE"),
                          ("Block", None), (_block_token(rng), "BLOCK"),
                          ("Madina Town", "COLONY"), ("Faisalabad", "CITY")])
        elif style < 0.60:
            # "Plot 12, Block A, DHA, Faisalabad"
            pairs = join([("Plot", None), (h, "HOUSE"),
                          ("Block", None), (_block_token(rng), "BLOCK"),
                          ("DHA", "COLONY"), ("Faisalabad", "CITY")])
        elif style < 0.80:
            # "House 33, Street 5, Model City, Faisalabad"
            pairs = join([(rng.choice(HOUSE_WORDS), None), (h, "HOUSE"),
                          ("Street", None), (_street_token(rng), "STREET"),
                          (rng.choice(["Model City", "Wadi-e-Sitara", "Gulberg City"]), "COLONY"),
                          ("Faisalabad", "CITY")])
        else:
            # messy: "Makan 78 Gali 5 Madina Town Faisalabad"
            pairs = join([(rng.choice(URDU_HOUSE), None), (h, "HOUSE"),
                          ("Gali", None), (_street_token(rng), "STREET"),
                          ("Madina Town", "COLONY"), ("Faisalabad", "CITY")])
    return _build(pairs)


MULTICITY_CITIES = ["Karachi", "Islamabad", "Rawalpindi", "Faisalabad"]


def generate_multicity(n: int, seed: int = 123) -> list[dict]:
    """Generate n city-specific labeled addresses, round-robin across cities."""
    rng = random.Random(seed)
    out = []
    for i in range(n):
        city = MULTICITY_CITIES[i % len(MULTICITY_CITIES)]
        tokens, labels = generate_multicity_one(rng, city)
        out.append({"tokens": tokens, "labels": labels,
                    "text": " ".join(tokens).replace(" ,", ","),
                    "city": city})
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=8000)
    ap.add_argument("--out", default="data/addresses.jsonl")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    rows = generate(args.n, args.seed)
    with open(args.out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote {len(rows)} addresses -> {args.out}")


if __name__ == "__main__":
    main()
